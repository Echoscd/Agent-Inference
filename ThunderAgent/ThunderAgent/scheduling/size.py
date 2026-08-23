"""SizePolicy: legacy token-size ordering, expressed as a single sort_key.

Reproduces the old grouped admission order (REASONING(step>1) > NEW(step==1) >
ACTING, ascending tokens within each group) via a tuple key -- eliminating the
`if use_value` else-branch in the Router.
"""
from __future__ import annotations

from .base import SchedulingPolicy
from ..program import Program, ProgramStatus


class SizePolicy(SchedulingPolicy):
    name = "size"

    def sort_key(self, state: Program):
        # group priority: REASONING(step>1)=0, NEW(step==1)=1, ACTING=2
        if state.step_count == 1:
            group = 1
        elif state.status == ProgramStatus.REASONING:
            group = 0
        else:
            group = 2
        # larger key = higher keep-priority; small group + small tokens should win,
        # so negate both. admit=sort desc -> REASONING then NEW then ACTING, each
        # ascending by tokens.
        return (-group, -state.total_tokens)

    def evict_key(self, state: Program):
        # Legacy size eviction: within a status group (Router already tries ACTING
        # before REASONING) evict the SMALLEST-token program first. argmin(tokens)
        # -> smallest first, matching the old _get_*_sorted(ascending=True)[0].
        return (state.total_tokens,)
