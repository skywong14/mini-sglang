from __future__ import annotations

from dataclasses import dataclass

import torch
from minisgl.core import Req, SamplingParams
from minisgl.kvcache import BaseCacheHandle
from minisgl.scheduler.decode import DecodeManager


@dataclass(frozen=True)
class DummyHandle(BaseCacheHandle):
    def get_matched_indices(self) -> torch.Tensor:
        return torch.arange(self.cached_len, dtype=torch.int32)


def _make_req(uid: int, input_len: int = 4, output_len: int = 4) -> Req:
    cached_len = input_len - 1
    return Req(
        input_ids=torch.arange(input_len, dtype=torch.int32),
        table_idx=uid,
        cached_len=cached_len,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=DummyHandle(cached_len),
    )


def test_decode_manager_schedules_in_uid_order():
    manager = DecodeManager(page_size=4)
    reqs = [_make_req(3), _make_req(1), _make_req(2)]

    manager.filter_reqs(reqs)
    batch = manager.schedule_next_batch()

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [1, 2, 3]


def test_decode_manager_order_is_not_insertion_dependent():
    reqs = {uid: _make_req(uid) for uid in [1, 2, 3, 4]}
    first = DecodeManager(page_size=4)
    second = DecodeManager(page_size=4)

    first.filter_reqs([reqs[4], reqs[1], reqs[3], reqs[2]])
    second.filter_reqs([reqs[2], reqs[3], reqs[1], reqs[4]])

    first_batch = first.schedule_next_batch()
    second_batch = second.schedule_next_batch()

    assert first_batch is not None
    assert second_batch is not None
    assert [req.uid for req in first_batch.reqs] == [1, 2, 3, 4]
    assert [req.uid for req in second_batch.reqs] == [1, 2, 3, 4]
    assert not isinstance(first.running_reqs, set)


def test_decode_manager_can_exclude_protected_uids():
    manager = DecodeManager(page_size=4)
    manager.filter_reqs([_make_req(3), _make_req(1), _make_req(2)])

    batch = manager.schedule_next_batch(exclude_uids={1, 3})

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [2]


def test_decode_manager_abort_is_uid_keyed():
    manager = DecodeManager(page_size=4)
    reqs = [_make_req(1), _make_req(2), _make_req(3)]
    manager.filter_reqs(reqs)

    removed = manager.abort_req(2)

    assert removed is reqs[1]
    assert sorted(manager.running_reqs) == [1, 3]


def test_decode_manager_filter_prunes_finished_requests():
    manager = DecodeManager(page_size=4)
    running = _make_req(1, input_len=4, output_len=1)
    finished = _make_req(2, input_len=4, output_len=0)
    manager.filter_reqs([running, finished])

    assert sorted(manager.running_reqs) == [1]

    running.device_len = running.max_device_len
    manager.filter_reqs([])

    assert manager.running_reqs == {}
