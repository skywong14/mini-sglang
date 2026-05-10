from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import minisgl.scheduler.scheduler as scheduler_module
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache import BaseCacheHandle
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillAdder, PrefillManager
from minisgl.scheduler.scheduler import ForwardInput, Scheduler
from minisgl.server.args import parse_args


@dataclass(frozen=True)
class DummyHandle(BaseCacheHandle):
    def get_matched_indices(self) -> torch.Tensor:
        return torch.arange(self.cached_len, dtype=torch.int32)


@dataclass
class FakeCacheManager:
    page_size: int = 4
    allocatable_pages: int = 1
    cached_finished: list[tuple[int, bool]] | None = None

    def __post_init__(self):
        self.cached_finished = []

    def needed_pages_for_reqs(self, reqs):
        return len(reqs)

    def cache_req(self, req, *, finished: bool):
        assert self.cached_finished is not None
        self.cached_finished.append((req.uid, finished))

    @contextmanager
    def lazy_free_region(self):
        yield


@dataclass
class FakeTableManager:
    freed_slots: list[int] | None = None

    def __post_init__(self):
        self.freed_slots = []

    def free(self, slot: int):
        assert self.freed_slots is not None
        self.freed_slots.append(slot)


@pytest.fixture(autouse=True)
def disable_rank0_logs(monkeypatch):
    monkeypatch.setattr(scheduler_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler_module.logger, "debug_rank0", lambda *args, **kwargs: None)


def _make_decode_req(uid: int, cached_len: int, output_len: int = 8) -> Req:
    input_ids = torch.arange(cached_len + 1, dtype=torch.int32)
    return Req(
        input_ids=input_ids,
        table_idx=uid,
        cached_len=cached_len,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=DummyHandle(cached_len),
    )


def _make_scheduler(policy: str, allocatable_pages: int = 1, margin: int = 0):
    decode_manager = DecodeManager(page_size=4)
    cache_manager = FakeCacheManager(page_size=4, allocatable_pages=allocatable_pages)
    table_manager = FakeTableManager()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache_manager = cache_manager
    scheduler.decode_manager = decode_manager
    scheduler.table_manager = table_manager
    scheduler.prefill_manager = PrefillManager(cache_manager, table_manager, decode_manager)
    scheduler.enable_preemption = True
    scheduler.enable_overlap_preemption = False
    scheduler.preemption_victim_policy = policy
    scheduler.preempt_min_free_pages = margin
    scheduler.num_preemptions = 0
    scheduler.num_preempted_prefix_tokens = 0
    scheduler.num_resumed_preempted_reqs = 0
    scheduler.num_deferred_preemptions = 0
    scheduler.num_prefill_fit_failures = 0
    scheduler.last_preempted_uids = []
    scheduler.protected_uids = set()
    scheduler.finished_uids = set()
    scheduler.aborted_uids = set()
    scheduler.released_reqs = set()
    scheduler.pending_preempted_uids = set()
    scheduler.deferred_preempt_uids = set()
    scheduler.deferred_abort_uids = set()
    scheduler.deferred_preempt_reqs = {}
    scheduler.sent_replies = []
    scheduler.send_result = lambda reply: scheduler.sent_replies.append(reply)
    scheduler.eos_token_id = 999
    return scheduler


class FakeEvent:
    def synchronize(self):
        pass


def _make_forward_data(batch: Batch, next_tokens: list[int]):
    forward_input = ForwardInput(
        batch=batch,
        sample_args=None,
        input_tuple=(None, None),
        write_tuple=(None, None),
    )
    forward_output = (
        torch.tensor(next_tokens, dtype=torch.int32),
        torch.tensor(next_tokens, dtype=torch.int32),
        FakeEvent(),
    )
    return forward_input, forward_output


def test_dynamic_prefill_estimate_ignores_output_len():
    cache_manager = SimpleNamespace(page_size=4)
    dynamic = PrefillAdder(
        token_budget=1,
        reserved_size=0,
        cache_manager=cache_manager,
        table_manager=None,
        dynamic_kv_allocation=True,
    )
    legacy = PrefillAdder(
        token_budget=1,
        reserved_size=0,
        cache_manager=cache_manager,
        table_manager=None,
        dynamic_kv_allocation=False,
    )

    assert dynamic._estimate_one(cached_len=0, input_len=2, output_len=1000) == 4
    assert legacy._estimate_one(cached_len=0, input_len=2, output_len=1000) == 1002


def test_prefill_no_fit_returns_none_and_counts_failure(monkeypatch):
    scheduler = _make_scheduler("largest_kv", allocatable_pages=1)
    reqs = [
        _make_decode_req(uid=1, cached_len=4),
        _make_decode_req(uid=2, cached_len=4),
    ]
    batch = Batch(reqs=list(reqs), phase="prefill")
    warnings = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "warning_rank0",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    new_batch = scheduler._maybe_preempt_to_fit(batch)

    assert new_batch is None
    assert scheduler.num_prefill_fit_failures == 1
    assert scheduler.num_preemptions == 0
    assert [req.uid for req in batch.reqs] == [1, 2]
    assert len(warnings) == 1
    args, kwargs = warnings[0]
    assert kwargs == {}
    assert args[1:] == (2, 1, "prefill", [1, 2])


def test_preemption_requeues_with_remaining_budget_and_removes_decode_req():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=1)
    reqs = [
        _make_decode_req(uid=1, cached_len=16, output_len=5),
        _make_decode_req(uid=3, cached_len=8, output_len=7),
        _make_decode_req(uid=2, cached_len=12, output_len=6),
    ]
    scheduler.decode_manager.filter_reqs(reqs)
    batch = Batch(reqs=list(reqs), phase="decode")

    new_batch = scheduler._maybe_preempt_to_fit(batch)

    assert new_batch is not None
    assert [req.uid for req in new_batch.reqs] == [3]
    assert sorted(scheduler.decode_manager.running_reqs) == [3]
    assert scheduler.last_preempted_uids == [1, 2]
    assert scheduler.num_preemptions == 2
    assert scheduler.num_preempted_prefix_tokens == reqs[0].cached_len + reqs[2].cached_len
    assert scheduler.table_manager.freed_slots == [1, 2]
    assert scheduler.cache_manager.cached_finished == [(1, True), (2, True)]

    pending = scheduler.prefill_manager.pending_list
    assert [req.uid for req in pending] == [1, 2]
    assert torch.equal(pending[0].input_ids, reqs[0].input_ids)
    assert pending[0].input_ids.data_ptr() != reqs[0].input_ids.data_ptr()
    assert pending[0].sampling_params.max_tokens == reqs[0].remain_len
    assert pending[0].sampling_params is not reqs[0].sampling_params


def test_preemption_fcfs_tail_policy_uses_largest_uid():
    scheduler = _make_scheduler("fcfs_tail", allocatable_pages=2)
    reqs = [
        _make_decode_req(uid=1, cached_len=32),
        _make_decode_req(uid=2, cached_len=4),
        _make_decode_req(uid=3, cached_len=8),
    ]
    scheduler.decode_manager.filter_reqs(reqs)
    batch = Batch(reqs=list(reqs), phase="decode")

    new_batch = scheduler._maybe_preempt_to_fit(batch)

    assert new_batch is not None
    assert [req.uid for req in new_batch.reqs] == [1, 2]
    assert scheduler.last_preempted_uids == [3]


def test_preemption_margin_can_force_additional_victims():
    scheduler = _make_scheduler("fcfs_tail", allocatable_pages=2, margin=1)
    reqs = [_make_decode_req(uid=1, cached_len=4), _make_decode_req(uid=2, cached_len=8)]
    scheduler.decode_manager.filter_reqs(reqs)
    batch = Batch(reqs=list(reqs), phase="decode")

    new_batch = scheduler._maybe_preempt_to_fit(batch)

    assert new_batch is not None
    assert [req.uid for req in new_batch.reqs] == [1]
    assert scheduler.last_preempted_uids == [2]


def test_protected_last_data_request_cannot_be_preempted():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=1)
    protected = _make_decode_req(uid=1, cached_len=32)
    victim = _make_decode_req(uid=2, cached_len=4)
    scheduler.decode_manager.filter_reqs([protected, victim])
    scheduler.protected_uids = {protected.uid}
    batch = Batch(reqs=[protected, victim], phase="decode")

    new_batch = scheduler._maybe_preempt_to_fit(batch)

    assert new_batch is not None
    assert [req.uid for req in new_batch.reqs] == [1]
    assert sorted(scheduler.decode_manager.running_reqs) == [1]
    assert scheduler.last_preempted_uids == [2]
    assert scheduler.table_manager.freed_slots == [2]
    assert [req.uid for req in scheduler.prefill_manager.pending_list] == [2]


def test_overlap_preemption_defers_protected_victim_without_freeing():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=1)
    scheduler.enable_overlap_preemption = True
    protected = _make_decode_req(uid=1, cached_len=32)
    protected.complete_one()
    runnable = _make_decode_req(uid=2, cached_len=4)
    scheduler.decode_manager.filter_reqs([protected, runnable])
    scheduler.protected_uids = {protected.uid}
    batch = Batch(reqs=[protected, runnable], phase="decode")

    new_batch = scheduler._maybe_preempt_to_fit(batch)

    assert new_batch is not None
    assert [req.uid for req in new_batch.reqs] == [2]
    assert sorted(scheduler.decode_manager.running_reqs) == [2]
    assert scheduler.deferred_preempt_uids == {1}
    assert scheduler.deferred_preempt_reqs == {1: protected}
    assert scheduler.num_deferred_preemptions == 1
    assert scheduler.num_preemptions == 0
    assert scheduler.table_manager.freed_slots == []
    assert scheduler.cache_manager.cached_finished == []
    assert scheduler.prefill_manager.pending_list == []

    scheduled = scheduler._schedule_decode_batch()
    assert scheduled is not None
    assert [req.uid for req in scheduled.reqs] == [2]


def test_deferred_preempt_process_last_data_commits_token_then_requeues_once():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=0)
    scheduler.enable_overlap_preemption = True
    req = _make_decode_req(uid=1, cached_len=4, output_len=5)
    original_input_ids = req.input_ids.clone()
    req.complete_one()
    scheduler.decode_manager.filter_reqs([req])
    batch = Batch(reqs=[req], phase="decode")
    data = _make_forward_data(batch, [42])
    scheduler._set_protected_forward_data(data)

    assert scheduler._maybe_preempt_to_fit(Batch(reqs=[req], phase="decode")) is None
    assert scheduler.deferred_preempt_uids == {1}
    assert scheduler.decode_manager.running_reqs == {}
    assert scheduler.table_manager.freed_slots == []

    scheduler._process_last_data(data)
    scheduler._process_last_data(data)

    pending = scheduler.prefill_manager.pending_list
    expected_input_ids = torch.cat([original_input_ids, torch.tensor([42], dtype=torch.int32)])
    assert [req.uid for req in pending] == [1]
    assert torch.equal(pending[0].input_ids, expected_input_ids)
    assert pending[0].input_ids.data_ptr() != req.input_ids.data_ptr()
    assert pending[0].sampling_params.max_tokens == req.remain_len
    assert scheduler.pending_preempted_uids == {1}
    assert scheduler.deferred_preempt_uids == set()
    assert scheduler.deferred_preempt_reqs == {}
    assert scheduler.table_manager.freed_slots == [1]
    assert scheduler.cache_manager.cached_finished == [(1, True)]
    assert scheduler.last_preempted_uids == [1]
    assert scheduler.num_deferred_preemptions == 1
    assert scheduler.num_preemptions == 1
    assert len(scheduler.sent_replies) == 2
    assert [(msg.uid, msg.next_token, msg.finished) for msg in scheduler.sent_replies[0]] == [
        (1, 42, False)
    ]
    assert scheduler.sent_replies[1] == []


def test_deferred_preempt_finished_request_is_not_requeued():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=0)
    scheduler.enable_overlap_preemption = True
    scheduler.eos_token_id = 42
    req = _make_decode_req(uid=1, cached_len=4, output_len=5)
    req.complete_one()
    scheduler.decode_manager.filter_reqs([req])
    batch = Batch(reqs=[req], phase="decode")
    data = _make_forward_data(batch, [42])
    scheduler._set_protected_forward_data(data)

    assert scheduler._maybe_preempt_to_fit(Batch(reqs=[req], phase="decode")) is None
    scheduler._process_last_data(data)

    assert scheduler.finished_uids == {1}
    assert scheduler.prefill_manager.pending_list == []
    assert scheduler.pending_preempted_uids == set()
    assert scheduler.deferred_preempt_uids == set()
    assert scheduler.deferred_preempt_reqs == {}
    assert scheduler.table_manager.freed_slots == [1]
    assert scheduler.cache_manager.cached_finished == [(1, True)]
    assert scheduler.num_deferred_preemptions == 1
    assert scheduler.num_preemptions == 0
    assert [(msg.uid, msg.next_token, msg.finished) for msg in scheduler.sent_replies[0]] == [
        (1, 42, True)
    ]


def test_protected_request_can_be_preempted_after_process_last_data():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=0)
    req = _make_decode_req(uid=1, cached_len=8)
    req.complete_one()
    scheduler.decode_manager.filter_reqs([req])
    batch = Batch(reqs=[req], phase="decode")
    data = _make_forward_data(batch, [10])

    scheduler._set_protected_forward_data(data)
    assert scheduler._maybe_preempt_to_fit(Batch(reqs=[req], phase="decode")) is None
    assert scheduler.prefill_manager.pending_list == []
    assert scheduler.table_manager.freed_slots == []

    scheduler._process_last_data(data)
    assert scheduler.protected_uids == set()
    assert scheduler.num_deferred_preemptions == 1

    assert scheduler._maybe_preempt_to_fit(Batch(reqs=[req], phase="decode")) is None
    assert scheduler.last_preempted_uids == [1]
    assert scheduler.table_manager.freed_slots == [1]
    assert [pending.uid for pending in scheduler.prefill_manager.pending_list] == [1]
    assert torch.equal(scheduler.prefill_manager.pending_list[0].input_ids, req.input_ids)


def test_finished_request_is_released_once_for_stale_overlap_data():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=0)
    req = _make_decode_req(uid=1, cached_len=4, output_len=1)
    req.complete_one()
    scheduler.decode_manager.filter_reqs([req])
    batch = Batch(reqs=[req], phase="decode")
    data = _make_forward_data(batch, [10])

    scheduler._process_last_data(data)
    scheduler._process_last_data(data)

    assert scheduler.finished_uids == {1}
    assert scheduler.table_manager.freed_slots == [1]
    assert scheduler.cache_manager.cached_finished == [(1, True)]
    assert len(scheduler.sent_replies) == 2
    assert len(scheduler.sent_replies[0]) == 1
    assert scheduler.sent_replies[1] == []


def test_abort_protected_request_defers_free_until_last_data_processed():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=0)
    scheduler.enable_overlap_preemption = True
    req = _make_decode_req(uid=1, cached_len=4)
    req.complete_one()
    original_input_ids = req.input_ids.clone()
    scheduler.decode_manager.filter_reqs([req])
    batch = Batch(reqs=[req], phase="decode")
    data = _make_forward_data(batch, [10])
    scheduler._set_protected_forward_data(data)

    scheduler._process_one_msg(AbortBackendMsg(uid=req.uid))

    assert scheduler.aborted_uids == {1}
    assert scheduler.deferred_abort_uids == {1}
    assert scheduler.decode_manager.running_reqs == {}
    assert scheduler.table_manager.freed_slots == []
    assert scheduler.cache_manager.cached_finished == []

    scheduler._process_last_data(data)
    scheduler._process_last_data(data)

    assert scheduler.deferred_abort_uids == set()
    assert scheduler.table_manager.freed_slots == [1]
    assert scheduler.cache_manager.cached_finished == [(1, True)]
    assert scheduler.sent_replies == [[], []]
    assert torch.equal(req.input_ids, original_input_ids)


def test_aborted_request_is_not_selected_as_preemption_victim():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=1)
    aborted = _make_decode_req(uid=1, cached_len=32)
    victim = _make_decode_req(uid=2, cached_len=4)
    scheduler.decode_manager.filter_reqs([aborted, victim])
    scheduler.aborted_uids = {aborted.uid}
    batch = Batch(reqs=[aborted, victim], phase="decode")

    new_batch = scheduler._maybe_preempt_to_fit(batch)

    assert new_batch is not None
    assert [req.uid for req in new_batch.reqs] == [1]
    assert scheduler.last_preempted_uids == [2]


def test_preempted_request_is_requeued_once_when_protected_batch_cannot_fit():
    scheduler = _make_scheduler("fcfs_tail", allocatable_pages=0)
    protected = _make_decode_req(uid=1, cached_len=4)
    victim = _make_decode_req(uid=2, cached_len=8)
    scheduler.decode_manager.filter_reqs([protected, victim])
    scheduler.protected_uids = {protected.uid}
    batch = Batch(reqs=[protected, victim], phase="decode")

    assert scheduler._maybe_preempt_to_fit(batch) is None

    assert scheduler.last_preempted_uids == [2]
    assert [pending.uid for pending in scheduler.prefill_manager.pending_list] == [2]
    assert scheduler.pending_preempted_uids == {2}
    assert scheduler.num_deferred_preemptions == 1
    assert scheduler.table_manager.freed_slots == [2]


def test_resumed_preempted_request_counter_increments_once():
    scheduler = _make_scheduler("largest_kv", allocatable_pages=1)
    req = _make_decode_req(uid=1, cached_len=4)
    batch = Batch(reqs=[req], phase="prefill")

    class FakeEngine:
        def forward_batch(self, batch, sample_args):
            return SimpleNamespace(
                next_tokens_gpu=torch.tensor([7], dtype=torch.int32),
                next_tokens_cpu=torch.tensor([7], dtype=torch.int32),
                copy_done_event=FakeEvent(),
            )

    scheduler.engine = FakeEngine()
    scheduler.token_pool = torch.zeros((2, 16), dtype=torch.int32)
    scheduler.pending_preempted_uids = {req.uid}
    forward_input = ForwardInput(
        batch=batch,
        sample_args=None,
        input_tuple=(torch.tensor([req.table_idx]), torch.tensor([0])),
        write_tuple=(torch.tensor([req.table_idx]), torch.tensor([0])),
    )

    scheduler._forward(forward_input)
    scheduler._forward(forward_input)

    assert scheduler.pending_preempted_uids == set()
    assert scheduler.num_resumed_preempted_reqs == 1
    assert sorted(scheduler.decode_manager.running_reqs) == [1]


def test_decode_scheduling_excludes_protected_uids_when_preemption_is_enabled():
    protected = _make_decode_req(uid=1, cached_len=4)
    runnable = _make_decode_req(uid=2, cached_len=4)

    class EmptyPrefillManager:
        def schedule_next_batch(self, prefill_budget, dynamic_kv_allocation=False):
            return None

    scheduler = _make_scheduler("largest_kv", allocatable_pages=4)
    scheduler.decode_first = True
    scheduler.dynamic_kv_allocation = True
    scheduler.prefill_budget = 123
    scheduler.prefill_manager = EmptyPrefillManager()
    scheduler.decode_manager.filter_reqs([protected, runnable])
    scheduler.protected_uids = {protected.uid}
    scheduler._prepare_batch = lambda batch: batch

    scheduled = scheduler._schedule_next_batch()

    assert scheduled is not None
    assert [req.uid for req in scheduled.reqs] == [2]


def test_overlap_preemption_keeps_protected_normal_request_schedulable():
    protected = _make_decode_req(uid=1, cached_len=4)
    runnable = _make_decode_req(uid=2, cached_len=4)
    scheduler = _make_scheduler("largest_kv", allocatable_pages=4)
    scheduler.enable_overlap_preemption = True
    scheduler.decode_manager.filter_reqs([protected, runnable])
    scheduler.protected_uids = {protected.uid}

    scheduled = scheduler._schedule_decode_batch()

    assert scheduled is not None
    assert [req.uid for req in scheduled.reqs] == [1, 2]


def test_deferred_preempt_request_is_not_scheduled_again():
    deferred = _make_decode_req(uid=1, cached_len=4)
    runnable = _make_decode_req(uid=2, cached_len=4)
    scheduler = _make_scheduler("largest_kv", allocatable_pages=4)
    scheduler.enable_overlap_preemption = True
    scheduler.decode_manager.filter_reqs([deferred, runnable])
    scheduler.deferred_preempt_uids = {deferred.uid}
    scheduler.deferred_preempt_reqs = {deferred.uid: deferred}

    scheduled = scheduler._schedule_decode_batch()

    assert scheduled is not None
    assert [req.uid for req in scheduled.reqs] == [2]


def test_decode_first_resumes_preempted_request_before_more_decode():
    prefill_batch = Batch(reqs=[_make_decode_req(uid=1, cached_len=4)], phase="prefill")
    decode_batch = Batch(reqs=[_make_decode_req(uid=2, cached_len=4)], phase="decode")

    class ResumePrefillManager:
        runnable = True

        def __init__(self):
            self.calls = 0

        def schedule_next_batch(self, prefill_budget, dynamic_kv_allocation=False):
            self.calls += 1
            return prefill_batch

    class RecordingDecodeManager:
        def __init__(self):
            self.calls = 0

        def schedule_next_batch(self, exclude_uids=None):
            self.calls += 1
            return decode_batch

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_preemption = True
    scheduler.decode_first = True
    scheduler.dynamic_kv_allocation = True
    scheduler.prefill_budget = 123
    scheduler.pending_preempted_uids = {1}
    scheduler.protected_uids = set()
    scheduler.prefill_manager = ResumePrefillManager()
    scheduler.decode_manager = RecordingDecodeManager()
    scheduler._maybe_preempt_to_fit = lambda batch: batch
    scheduler._prepare_batch = lambda batch: batch

    scheduled = scheduler._schedule_next_batch()

    assert scheduled is prefill_batch
    assert scheduler.prefill_manager.calls == 1
    assert scheduler.decode_manager.calls == 0


def _make_run_forever_scheduler():
    class StopLoop(Exception):
        pass

    class DummyStream:
        def wait_stream(self, stream):
            pass

    @contextmanager
    def dummy_stream_context():
        yield

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine_stream_ctx = dummy_stream_context()
    scheduler.engine = SimpleNamespace(stream=DummyStream())
    scheduler.stream = object()
    return scheduler, StopLoop


def test_preemption_run_forever_uses_normal_loop_by_default(monkeypatch):
    scheduler, stop_loop = _make_run_forever_scheduler()
    scheduler.enable_preemption = True
    scheduler.enable_overlap_preemption = False
    scheduler.normal_loop = lambda: (_ for _ in ()).throw(stop_loop())
    scheduler.overlap_loop = lambda data: pytest.fail("preemption should not use overlap_loop")
    monkeypatch.setattr(scheduler_module.ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)

    with pytest.raises(stop_loop):
        scheduler.run_forever()


def test_preemption_run_forever_can_use_overlap_loop_when_enabled(monkeypatch):
    scheduler, stop_loop = _make_run_forever_scheduler()
    scheduler.enable_preemption = True
    scheduler.enable_overlap_preemption = True
    scheduler.normal_loop = lambda: pytest.fail("overlap preemption should not use normal_loop")
    scheduler.overlap_loop = lambda data: (_ for _ in ()).throw(stop_loop())
    monkeypatch.setattr(scheduler_module.ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    monkeypatch.setattr(scheduler_module.torch.cuda, "current_stream", lambda: scheduler.stream)

    with pytest.raises(stop_loop):
        scheduler.run_forever()


def test_run_forever_env_disable_overlap_uses_normal_loop(monkeypatch):
    scheduler, stop_loop = _make_run_forever_scheduler()
    scheduler.enable_preemption = True
    scheduler.enable_overlap_preemption = True
    scheduler.normal_loop = lambda: (_ for _ in ()).throw(stop_loop())
    scheduler.overlap_loop = lambda data: pytest.fail("disabled overlap should not use overlap_loop")
    monkeypatch.setattr(scheduler_module.ENV.DISABLE_OVERLAP_SCHEDULING, "value", True)

    with pytest.raises(stop_loop):
        scheduler.run_forever()


def test_no_preemption_run_forever_uses_overlap_loop_when_env_allows(monkeypatch):
    scheduler, stop_loop = _make_run_forever_scheduler()
    scheduler.enable_preemption = False
    scheduler.enable_overlap_preemption = False
    scheduler.normal_loop = lambda: pytest.fail("no-preemption overlap mode should not use normal_loop")
    scheduler.overlap_loop = lambda data: (_ for _ in ()).throw(stop_loop())
    monkeypatch.setattr(scheduler_module.ENV.DISABLE_OVERLAP_SCHEDULING, "value", False)
    monkeypatch.setattr(scheduler_module.torch.cuda, "current_stream", lambda: scheduler.stream)

    with pytest.raises(stop_loop):
        scheduler.run_forever()


def test_no_preemption_mode_keeps_prefill_first_scheduling():
    prefill_batch = Batch(reqs=[_make_decode_req(uid=1, cached_len=4)], phase="prefill")
    decode_batch = Batch(reqs=[_make_decode_req(uid=2, cached_len=4)], phase="decode")

    class RecordingPrefillManager:
        def __init__(self):
            self.calls = []

        def schedule_next_batch(self, prefill_budget, dynamic_kv_allocation=False):
            self.calls.append((prefill_budget, dynamic_kv_allocation))
            return prefill_batch

    class RecordingDecodeManager:
        def __init__(self):
            self.calls = 0

        def schedule_next_batch(self):
            self.calls += 1
            return decode_batch

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_preemption = False
    scheduler.decode_first = False
    scheduler.dynamic_kv_allocation = False
    scheduler.prefill_budget = 123
    scheduler.prefill_manager = RecordingPrefillManager()
    scheduler.decode_manager = RecordingDecodeManager()
    scheduler._maybe_preempt_to_fit = lambda batch: batch
    scheduler._prepare_batch = lambda batch: batch

    scheduled = scheduler._schedule_next_batch()

    assert scheduled is prefill_batch
    assert scheduler.prefill_manager.calls == [(123, False)]
    assert scheduler.decode_manager.calls == 0


def test_no_preemption_mode_uses_decode_when_prefill_is_empty():
    decode_batch = Batch(reqs=[_make_decode_req(uid=2, cached_len=4)], phase="decode")

    class EmptyPrefillManager:
        def __init__(self):
            self.calls = []

        def schedule_next_batch(self, prefill_budget, dynamic_kv_allocation=False):
            self.calls.append((prefill_budget, dynamic_kv_allocation))
            return None

    class RecordingDecodeManager:
        def __init__(self):
            self.calls = 0

        def schedule_next_batch(self):
            self.calls += 1
            return decode_batch

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_preemption = False
    scheduler.decode_first = False
    scheduler.dynamic_kv_allocation = False
    scheduler.prefill_budget = 123
    scheduler.prefill_manager = EmptyPrefillManager()
    scheduler.decode_manager = RecordingDecodeManager()
    scheduler._maybe_preempt_to_fit = lambda batch: batch
    scheduler._prepare_batch = lambda batch: batch

    scheduled = scheduler._schedule_next_batch()

    assert scheduled is decode_batch
    assert scheduler.prefill_manager.calls == [(123, False)]
    assert scheduler.decode_manager.calls == 1


def test_server_args_parse_preemption_flags():
    args, _ = parse_args(
        [
            "--model-path",
            "dummy-model",
            "--dtype",
            "float16",
            "--enable-preemption",
            "--enable-overlap-preemption",
            "--dynamic-kv-allocation",
            "--decode-first",
            "--preemption-victim-policy",
            "fcfs_tail",
            "--preempt-min-free-pages",
            "3",
        ]
    )

    assert args.enable_preemption
    assert args.enable_overlap_preemption
    assert args.dynamic_kv_allocation
    assert args.decode_first
    assert args.preemption_victim_policy == "fcfs_tail"
    assert args.preempt_min_free_pages == 3
