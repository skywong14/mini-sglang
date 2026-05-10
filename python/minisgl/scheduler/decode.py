from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

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

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        reqs = [self.running_reqs[uid] for uid in sorted(self.running_reqs)]
        return Batch(reqs=reqs, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
