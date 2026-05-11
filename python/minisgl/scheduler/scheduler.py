from __future__ import annotations

import weakref
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class ScheduleAction(NamedTuple):
    phase: str
    batch_uids: List[int]
    preempted_uids: List[int]
    deferred_preempt_uids: List[int]
    resumed_uids: List[int]
    finished_uids: List[int]
    aborted_uids: List[int]


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.config = config
        self.enable_preemption = config.enable_preemption
        self.enable_overlap_preemption = config.enable_overlap_preemption
        if self.enable_overlap_preemption and not self.enable_preemption:
            raise ValueError("enable_overlap_preemption requires enable_preemption")
        self.dynamic_kv_allocation = config.dynamic_kv_allocation or config.enable_preemption
        self.decode_first = config.decode_first or config.enable_preemption
        self.preemption_victim_policy = config.preemption_victim_policy
        self.preempt_min_free_pages = config.preempt_min_free_pages
        if self.preemption_victim_policy not in ["largest_kv", "fcfs_tail"]:
            raise ValueError(f"Unknown preemption victim policy: {self.preemption_victim_policy}")
        if self.preempt_min_free_pages < 0:
            raise ValueError("preempt_min_free_pages must be non-negative")
        self.num_preemptions = 0
        self.num_preempted_prefix_tokens = 0
        self.num_resumed_preempted_reqs = 0
        self.num_deferred_preemptions = 0
        self.num_preemption_stalls = 0
        self.num_prefill_fit_failures = 0
        self.last_schedule_action: ScheduleAction | None = None
        self.last_preempted_uids: List[int] = []
        self.protected_uids: Set[int] = set()
        self.finished_uids: Set[int] = set()
        self.aborted_uids: Set[int] = set()
        self.released_reqs = weakref.WeakSet()
        self.pending_preempted_uids: Set[int] = set()
        self.deferred_preempt_uids: Set[int] = set()
        self.deferred_abort_uids: Set[int] = set()
        self.deferred_preempt_reqs: Dict[int, Req] = {}
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        self._set_protected_forward_data(last_data)
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        self._set_protected_forward_data(ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        self._set_protected_forward_data(None)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._set_protected_forward_data(None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        elif self.enable_preemption and not self.enable_overlap_preemption:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        processed_uids = {req.uid for req in batch.reqs}
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        preempted: List[Tuple[int, torch.Tensor, SamplingParams]] = []
        finished_uids: List[int] = []
        aborted_uids: List[int] = []
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if req.uid in self.deferred_abort_uids:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources_once(req)
                    self.deferred_abort_uids.discard(req.uid)
                    self.deferred_preempt_uids.discard(req.uid)
                    self.deferred_preempt_reqs.pop(req.uid, None)
                    aborted_uids.append(req.uid)
                    continue
                if req.uid in self.aborted_uids:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources_once(req)
                    self.aborted_uids.discard(req.uid)
                    aborted_uids.append(req.uid)
                    continue
                if isinstance(req, ChunkedReq):
                    continue
                if req.uid in self.finished_uids:
                    self.finished_uids.discard(req.uid)
                    continue
                if req in self.released_reqs:
                    continue
                next_token, finished = self._commit_sampled_token(req, next_tokens_cpu[i])
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if req.uid in self.deferred_preempt_uids:
                    stored_req = self.deferred_preempt_reqs.pop(req.uid)
                    assert stored_req is req, f"Deferred preempt request {req.uid} changed identity"
                    self.deferred_preempt_uids.remove(req.uid)
                    self.decode_manager.remove_req(req)
                    if finished:
                        self.finished_uids.add(req.uid)
                        finished_uids.append(req.uid)
                        self._free_req_resources_once(req)
                    else:
                        preempted.append(self._make_preempted_req(req))
                        self.num_preemptions += 1
                        self.num_preempted_prefix_tokens += req.cached_len
                        freed = self._free_req_resources_once(req)
                        assert freed, f"Deferred victim {req.uid} resources were already released"
                elif finished:
                    self.finished_uids.add(req.uid)
                    finished_uids.append(req.uid)
                    self.decode_manager.remove_req(req)
                    self._free_req_resources_once(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.protected_uids.difference_update(processed_uids)
        preempted_uids = [uid for uid, _, _ in preempted]
        if preempted:
            self._finish_preempted_reqs(preempted)
        self._record_schedule_action(
            phase=f"process_{batch.phase}",
            batch_uids=[req.uid for req in batch.reqs],
            preempted_uids=preempted_uids,
            deferred_preempt_uids=[],
            resumed_uids=[],
            finished_uids=finished_uids,
            aborted_uids=aborted_uids,
        )
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            self.finished_uids.discard(msg.uid)
            self.aborted_uids.discard(msg.uid)
            self.pending_preempted_uids.discard(msg.uid)
            self.deferred_preempt_uids.discard(msg.uid)
            self.deferred_abort_uids.discard(msg.uid)
            self.deferred_preempt_reqs.pop(msg.uid, None)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params = replace(msg.sampling_params, max_tokens=max_output_len)
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            self.aborted_uids.add(msg.uid)
            self.pending_preempted_uids.discard(msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            if msg.uid in self.protected_uids or msg.uid in self.deferred_preempt_uids:
                self.deferred_abort_uids.add(msg.uid)
                self.deferred_preempt_uids.discard(msg.uid)
                self.deferred_preempt_reqs.pop(msg.uid, None)
                self.decode_manager.remove_uid(msg.uid)
                req_to_free = None
            else:
                req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
                self.aborted_uids.discard(msg.uid)
            if req_to_free is not None:
                self._free_req_resources_once(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _free_req_resources_once(self, req: Req) -> bool:
        if req in self.released_reqs:
            return False
        self._free_req_resources(req)
        self.released_reqs.add(req)
        return True

    def _forward_data_uids(self, data: ForwardData | None) -> Set[int]:
        if data is None:
            return set()
        return {req.uid for req in data[0].batch.reqs}

    def _set_protected_forward_data(self, data: ForwardData | None) -> None:
        protected_uids = self._forward_data_uids(data)
        self.finished_uids.intersection_update(protected_uids)
        self.aborted_uids.intersection_update(protected_uids | self.deferred_abort_uids)
        self.protected_uids = protected_uids

    def _record_schedule_action(
        self,
        *,
        phase: str,
        batch_uids: List[int],
        preempted_uids: List[int],
        deferred_preempt_uids: List[int],
        resumed_uids: List[int],
        finished_uids: List[int],
        aborted_uids: List[int],
    ) -> None:
        self.last_schedule_action = ScheduleAction(
            phase=phase,
            batch_uids=batch_uids,
            preempted_uids=preempted_uids,
            deferred_preempt_uids=deferred_preempt_uids,
            resumed_uids=resumed_uids,
            finished_uids=finished_uids,
            aborted_uids=aborted_uids,
        )

    def _can_preempt_now(self, req: Req) -> bool:
        return (
            req.uid not in self.protected_uids
            and req.uid not in self.aborted_uids
            and req.uid not in self.finished_uids
            and req.uid not in self.deferred_abort_uids
            and len(req.input_ids) == req.device_len
        )

    def _cannot_schedule_uids(self) -> Set[int]:
        return self.deferred_preempt_uids | self.deferred_abort_uids | self.aborted_uids

    def _commit_sampled_token(self, req: Req, next_token: torch.Tensor) -> Tuple[int, bool]:
        req.append_host(next_token.unsqueeze(0))
        next_token_int = int(next_token.item())
        finished = not req.can_decode
        if not req.sampling_params.ignore_eos:
            finished |= next_token_int == self.eos_token_id
        return next_token_int, finished

    def _can_allocate_batch(self, batch: Batch) -> bool:
        margin = self.preempt_min_free_pages if self.enable_preemption and batch.is_decode else 0
        needed_pages = self.cache_manager.needed_pages_for_reqs(batch.reqs)
        return needed_pages + margin <= self.cache_manager.allocatable_pages

    def _select_preemption_victim(self, reqs: List[Req]) -> Req:
        if self.preemption_victim_policy == "largest_kv":
            return max(
                reqs,
                key=lambda req: (
                    (req.cached_len + self.cache_manager.page_size - 1)
                    // self.cache_manager.page_size,
                    req.uid,
                ),
            )
        if self.preemption_victim_policy == "fcfs_tail":
            return max(reqs, key=lambda req: req.uid)
        raise ValueError(f"Unknown preemption victim policy: {self.preemption_victim_policy}")

    def _preemption_candidates(self, reqs: List[Req]) -> List[Req]:
        excluded_uids = self.finished_uids | self.aborted_uids | self.deferred_abort_uids
        if self.enable_overlap_preemption:
            excluded_uids = excluded_uids | self.deferred_preempt_uids
        else:
            excluded_uids = excluded_uids | self.protected_uids | self.deferred_preempt_uids
        return [req for req in reqs if req.uid not in excluded_uids]

    def _make_preempted_req(self, req: Req) -> Tuple[int, torch.Tensor, SamplingParams]:
        remaining_tokens = req.remain_len
        assert remaining_tokens > 0, f"Decode victim {req.uid} has no remaining tokens"
        sampling_params = replace(req.sampling_params, max_tokens=remaining_tokens)
        return req.uid, req.input_ids.clone(), sampling_params

    def _preempt_now(self, victim: Req, preempted: List[Tuple[int, torch.Tensor, SamplingParams]]) -> None:
        removed = self.decode_manager.remove_uid(victim.uid)
        assert removed is victim, f"Decode victim {victim.uid} is not running"
        preempted.append(self._make_preempted_req(victim))
        self.num_preemptions += 1
        self.num_preempted_prefix_tokens += victim.cached_len
        freed = self._free_req_resources_once(victim)
        assert freed, f"Decode victim {victim.uid} resources were already released"

    def _defer_preemption(self, victim: Req) -> None:
        removed = self.decode_manager.remove_uid(victim.uid)
        assert removed is victim, f"Deferred decode victim {victim.uid} is not running"
        assert victim.uid not in self.deferred_preempt_uids, (
            f"Decode victim {victim.uid} is already deferred"
        )
        self.deferred_preempt_uids.add(victim.uid)
        self.deferred_preempt_reqs[victim.uid] = victim
        self.num_deferred_preemptions += 1

    def _maybe_preempt_to_fit(self, batch: Batch) -> Batch | None:
        self.last_preempted_uids = []
        if self._can_allocate_batch(batch):
            return batch

        if batch.is_prefill:
            needed_pages = self.cache_manager.needed_pages_for_reqs(batch.reqs)
            self.num_prefill_fit_failures += 1
            self.prefill_manager.rollback_batch(batch)
            logger.warning_rank0(
                "Prefill batch does not fit after admission: "
                "needed_pages=%d, allocatable_pages=%d, phase=%s, batch_uids=%s",
                needed_pages,
                self.cache_manager.allocatable_pages,
                batch.phase,
                [req.uid for req in batch.reqs],
            )
            return None

        if not self.enable_preemption:
            needed_pages = self.cache_manager.needed_pages_for_reqs(batch.reqs)
            raise RuntimeError(
                "Decode batch does not fit and preemption is disabled:"
                f" needed_pages={needed_pages},"
                f" allocatable_pages={self.cache_manager.allocatable_pages}"
            )

        preempted: List[Tuple[int, torch.Tensor, SamplingParams]] = []
        while len(batch.reqs) > 0 and not self._can_allocate_batch(batch):
            candidates = self._preemption_candidates(batch.reqs)
            if len(candidates) == 0:
                self.num_preemption_stalls += 1
                self._finish_preempted_reqs(preempted)
                return None
            victim = self._select_preemption_victim(candidates)
            if self._can_preempt_now(victim):
                self._preempt_now(victim, preempted)
            else:
                assert self.enable_overlap_preemption and victim.uid in self.protected_uids, (
                    f"Decode victim {victim.uid} is not safe to preempt but is not protected"
                )
                self._defer_preemption(victim)
            batch.reqs = [req for req in batch.reqs if req.uid != victim.uid]

        self._finish_preempted_reqs(preempted)

        if len(batch.reqs) == 0:
            return None
        if not self._can_allocate_batch(batch):
            needed_pages = self.cache_manager.needed_pages_for_reqs(batch.reqs)
            raise RuntimeError(
                "Decode batch still does not fit after preemption:"
                f" needed_pages={needed_pages},"
                f" allocatable_pages={self.cache_manager.allocatable_pages},"
                f" margin_pages={self.preempt_min_free_pages}"
            )
        return batch

    def _finish_preempted_reqs(
        self, preempted: List[Tuple[int, torch.Tensor, SamplingParams]]
    ) -> None:
        for uid, input_ids, sampling_params in reversed(preempted):
            assert uid not in self.pending_preempted_uids, f"Request {uid} is already pending"
            self.prefill_manager.add_preempted_req_front(uid, input_ids, sampling_params)
            self.pending_preempted_uids.add(uid)
        self.last_preempted_uids = [uid for uid, _, _ in preempted]
        if self.last_preempted_uids:
            logger.info_rank0(
                "Preempted decode requests: uids=%s, policy=%s, total_preemptions=%d",
                self.last_preempted_uids,
                self.preemption_victim_policy,
                self.num_preemptions,
            )

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        if self.decode_first and self.pending_preempted_uids and self.prefill_manager.runnable:
            batch = self.prefill_manager.schedule_next_batch(
                self.prefill_budget, self.dynamic_kv_allocation
            ) or self._schedule_decode_batch()
        elif self.decode_first:
            batch = (
                self._schedule_decode_batch()
                or self.prefill_manager.schedule_next_batch(
                    self.prefill_budget, self.dynamic_kv_allocation
                )
            )
        else:
            batch = self.prefill_manager.schedule_next_batch(
                self.prefill_budget, self.dynamic_kv_allocation
            ) or self._schedule_decode_batch()
        if batch is not None:
            batch = self._maybe_preempt_to_fit(batch)
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        if batch.is_prefill:
            self.prefill_manager.commit_batch(batch)
        self._record_schedule_action(
            phase=batch.phase,
            batch_uids=[req.uid for req in batch.reqs],
            preempted_uids=self.last_preempted_uids,
            deferred_preempt_uids=sorted(self.deferred_preempt_uids),
            resumed_uids=[],
            finished_uids=[],
            aborted_uids=[],
        )
        return forward_input

    def _schedule_decode_batch(self) -> Batch | None:
        if self.enable_preemption:
            if self.enable_overlap_preemption:
                return self.decode_manager.schedule_next_batch(self._cannot_schedule_uids())
            return self.decode_manager.schedule_next_batch(self.protected_uids)
        return self.decode_manager.schedule_next_batch()

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        resumed_uids = []
        for req in forward_input.batch.reqs:
            if req.uid in self.pending_preempted_uids:
                self.pending_preempted_uids.remove(req.uid)
                self.num_resumed_preempted_reqs += 1
                resumed_uids.append(req.uid)
        if resumed_uids:
            self._record_schedule_action(
                phase=f"forward_{batch.phase}",
                batch_uids=[req.uid for req in batch.reqs],
                preempted_uids=[],
                deferred_preempt_uids=[],
                resumed_uids=resumed_uids,
                finished_uids=[],
                aborted_uids=[],
            )
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
