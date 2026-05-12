from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    enable_preemption: bool = False
    enable_overlap_preemption: bool = False
    dynamic_kv_allocation: bool = False
    decode_first: bool = False
    preemption_victim_policy: str = "smallest_kv"
    preempt_min_free_pages: int = 1
    preempt_prefill_decode_reserve_pages: int = 0

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    def __post_init__(self) -> None:
        if self.enable_overlap_preemption and not self.enable_preemption:
            raise ValueError("enable_overlap_preemption requires enable_preemption")
        if self.preemption_victim_policy not in ("largest_kv", "smallest_kv", "fcfs_tail"):
            raise ValueError(f"Unknown preemption victim policy: {self.preemption_victim_policy}")
        if self.preempt_min_free_pages < 0:
            raise ValueError("preempt_min_free_pages must be non-negative")
        if self.preempt_prefill_decode_reserve_pages < 0:
            raise ValueError("preempt_prefill_decode_reserve_pages must be non-negative")
        if not self.enable_preemption and self.preempt_prefill_decode_reserve_pages != 0:
            raise ValueError("preempt_prefill_decode_reserve_pages requires enable_preemption")
        if self.enable_preemption and self.num_page_override is not None:
            if self.preempt_min_free_pages >= self.num_page_override:
                raise ValueError("preempt_min_free_pages must be less than num_page_override")
            if self.preempt_prefill_decode_reserve_pages >= self.num_page_override:
                raise ValueError(
                    "preempt_prefill_decode_reserve_pages must be less than num_page_override"
                )

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
