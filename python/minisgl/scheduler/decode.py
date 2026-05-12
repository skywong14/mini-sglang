from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Dict[int, Req] = field(default_factory=dict)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        for uid, req in list(self.running_reqs.items()):
            if not req.can_decode:
                del self.running_reqs[uid]
        for req in reqs:
            if req.can_decode:
                self.running_reqs[req.uid] = req
            else:
                self.running_reqs.pop(req.uid, None)

    def remove_req(self, req: Req) -> None:
        self.running_reqs.pop(req.uid, None)

    def remove_uid(self, uid: int) -> Req | None:
        return self.running_reqs.pop(uid, None)

    def abort_req(self, uid: int) -> Req | None:
        return self.remove_uid(uid)

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs.values()) + tokens_reserved

    def _needed_pages_for_req(self, req: Req) -> int:
        first_page = (req.cached_len + self.page_size - 1) // self.page_size
        last_page = (req.device_len + self.page_size - 1) // self.page_size
        return max(last_page - first_page, 0)

    def _select_forced_preempt_req(self, reqs: list[Req], preemption_victim_policy: str) -> Req:
        if preemption_victim_policy == "largest_kv":
            return max(
                reqs,
                key=lambda req: (
                    (req.cached_len + self.page_size - 1) // self.page_size,
                    req.uid,
                ),
            )
        if preemption_victim_policy == "smallest_kv":
            return min(
                reqs,
                key=lambda req: (
                    (req.cached_len + self.page_size - 1) // self.page_size,
                    req.uid,
                ),
            )
        if preemption_victim_policy == "fcfs_tail":
            return max(reqs, key=lambda req: req.uid)
        raise ValueError(f"Unknown preemption victim policy: {preemption_victim_policy}")

    def schedule_next_batch(
        self,
        exclude_uids: Set[int] | None = None,
        page_budget: int | None = None,
        preemption_victim_policy: str = "smallest_kv",
        forced_preempt_deprioritize_uids: Set[int] | None = None,
    ) -> Batch | None:
        if page_budget is not None and page_budget < 0:
            raise ValueError("page_budget must be non-negative")
        if not self.runnable:
            return None
        if exclude_uids is None:
            exclude_uids = set()
        if forced_preempt_deprioritize_uids is None:
            forced_preempt_deprioritize_uids = set()

        candidates = [
            self.running_reqs[uid] for uid in sorted(self.running_reqs) if uid not in exclude_uids
        ]
        if len(candidates) == 0:
            return None
        if page_budget is None:
            return Batch(reqs=candidates, phase="decode")

        zero_page_reqs = []
        budgeted_reqs = []
        used_pages = 0
        for req in candidates:
            needed_pages = self._needed_pages_for_req(req)
            if needed_pages == 0:
                zero_page_reqs.append(req)
            elif used_pages + needed_pages <= page_budget:
                budgeted_reqs.append(req)
                used_pages += needed_pages

        reqs = zero_page_reqs + budgeted_reqs
        if len(reqs) == 0:
            forced_preempt_candidates = [
                req for req in candidates if req.uid not in forced_preempt_deprioritize_uids
            ]
            if len(forced_preempt_candidates) == 0:
                forced_preempt_candidates = candidates
            reqs = [
                self._select_forced_preempt_req(
                    forced_preempt_candidates, preemption_victim_policy
                )
            ]

        return Batch(reqs=reqs, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
