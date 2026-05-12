from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.utils import div_ceil, init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    dynamic_kv_allocation: bool = False

    def _estimate_one(self, cached_len: int, input_len: int, output_len: int) -> int:
        if not self.dynamic_kv_allocation:
            return input_len - cached_len + output_len

        chunk_size = min(self.token_budget, input_len - cached_len)
        first_page = div_ceil(cached_len, self.cache_manager.page_size)
        last_page = div_ceil(cached_len + chunk_size, self.cache_manager.page_size)
        return max(last_page - first_page, 0) * self.cache_manager.page_size

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        estimated_len = self._estimate_one(cached_len, req.input_len, req.output_len)

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        if self.dynamic_kv_allocation:
            first_page = div_ceil(cached_len, self.cache_manager.page_size)
            last_page = div_ceil(cached_len + chunk_size, self.cache_manager.page_size)
            self.reserved_size += max(last_page - first_page, 0) * self.cache_manager.page_size
        else:
            self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            if self.dynamic_kv_allocation:
                estimated_len = self._estimate_one(
                    chunked_req.cached_len, pending_req.input_len, pending_req.output_len
                )
                if estimated_len + self.reserved_size > self.cache_manager.available_size:
                    return None
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    _rollback_batch: Batch | None = field(default=None, init=False)
    _rollback_pending_list: List[PendingReq] | None = field(default=None, init=False)
    _rollback_chunked_reqs: List[Tuple[PendingReq, ChunkedReq | None]] = field(
        default_factory=list, init=False
    )

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def add_preempted_req_front(
        self, uid: int, input_ids: torch.Tensor, sampling_params: SamplingParams
    ) -> None:
        assert input_ids.is_cpu, "Preempted request input_ids must be on CPU"
        pending_req = PendingReq(uid, input_ids.clone(), sampling_params)
        self.pending_list.insert(0, pending_req)

    def add_preempted_reqs_front(
        self, reqs: List[Tuple[int, torch.Tensor, SamplingParams]]
    ) -> None:
        pending_reqs: List[PendingReq] = []
        for uid, input_ids, sampling_params in reqs:
            assert input_ids.is_cpu, "Preempted request input_ids must be on CPU"
            pending_reqs.append(PendingReq(uid, input_ids.clone(), sampling_params))
        self.pending_list = pending_reqs + self.pending_list

    def schedule_next_batch(
        self,
        prefill_budget: int,
        dynamic_kv_allocation: bool = False,
        decode_reserve_pages: int = 0,
    ) -> Batch | None:
        assert self._rollback_batch is None, "Previous prefill batch was not committed"
        if decode_reserve_pages < 0:
            raise ValueError("decode_reserve_pages must be non-negative")
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        reserved_size = 0 if dynamic_kv_allocation else self.decode_manager.inflight_tokens
        reserved_size += decode_reserve_pages * self.cache_manager.page_size
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=reserved_size,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            dynamic_kv_allocation=dynamic_kv_allocation,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        rollback_pending_list = list(self.pending_list)
        rollback_chunked_reqs: List[Tuple[PendingReq, ChunkedReq | None]] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                rollback_chunked_reqs.append((pending_req, pending_req.chunked_req))
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        batch = Batch(reqs=reqs, phase="prefill")
        self._rollback_batch = batch
        self._rollback_pending_list = rollback_pending_list
        self._rollback_chunked_reqs = rollback_chunked_reqs
        return batch

    def rollback_batch(self, batch: Batch) -> None:
        assert batch.is_prefill, "Only prefill batches can be rolled back"
        assert self._rollback_batch is batch, "Can only roll back the latest prefill batch"
        assert self._rollback_pending_list is not None, "Missing prefill rollback state"
        for pending_req, old_chunked_req in self._rollback_chunked_reqs:
            pending_req.chunked_req = old_chunked_req
        for req, (_, old_chunked_req) in zip(batch.reqs, self._rollback_chunked_reqs):
            if old_chunked_req is None:
                self.table_manager.free(req.table_idx)
                self.cache_manager.unlock(req.cache_handle)
        self.pending_list = self._rollback_pending_list
        self._clear_rollback()

    def commit_batch(self, batch: Batch) -> None:
        if not batch.is_prefill:
            return
        assert self._rollback_batch is batch, "Can only commit the latest prefill batch"
        self._clear_rollback()

    def _clear_rollback(self) -> None:
        self._rollback_batch = None
        self._rollback_pending_list = None
        self._rollback_chunked_reqs = []

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
