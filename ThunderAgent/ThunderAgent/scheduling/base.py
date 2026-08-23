"""SchedulingPolicy: a scheduling algorithm = one choice for each of the six
decision variables the Router orchestrates. Each policy lives in its own file and
"fills in" these variables; the Router holds a policy instance and never branches
on policy name.

The six decision variables (see the refactor guide):
  1. admission amount  -> peak_pad(state)          (reserve current footprint, or +decode)
  2. admission order   -> sort_key(state)          (who to admit first)
  3. admission timing  -> admits(state)            (greedy fill vs leave headroom / price gate)
  4. eviction timing   -> proactive_evictions(...) + on_epoch(...)   (only-on-overflow vs proactive)
  5. eviction order    -> sort_key(state)          (who to evict first; SAME key as #2)
  6. eviction count    -> orchestrated by the Router's _pause_until_safe while-loop

Variables 2 and 5 are two sides of one ordering (admit high keep-priority, evict low
keep-priority), so a single sort_key expresses both.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Tuple

from ..program import Program
from ..backend import BackendState


class SchedulingPolicy(ABC):
    """A scheduling algorithm = a choice for each of the six decision variables.

    sort_key convention (spans variables 2 and 5):
        LARGER sort_key(state)  =>  HIGHER keep-priority  =>  admitted earlier, evicted later.
        - admission (_greedy_resume): sort by sort_key DESCENDING, admit best first.
        - eviction  (_pause_until_safe): pick argmin(sort_key), evict lowest first.
    Example: SizePolicy wants "REASONING group + small tokens" to win, so it negates
    both (returns (-group, -tokens)) to flip small/low into a large key.
    """

    name: str = "base"

    # ── Variable 1: admission amount ────────────────────────────────────────────
    def peak_pad(self, state: Program) -> int:
        """Extra KV to reserve beyond the current footprint at admission.
        0 = reserve only the current footprint (vLLM may preempt mid-decode);
        return the decode length = peak reservation (FidelityPolicy Gap 1)."""
        return 0

    # ── Variable 2: admission order ─────────────────────────────────────────────
    @abstractmethod
    def sort_key(self, state: Program) -> Tuple:
        """keep-priority sort key for ADMISSION (sorted DESCENDING, admit best first).
        See the class docstring for the direction convention."""
        ...

    # ── Variable 5: eviction order ──────────────────────────────────────────────
    def evict_key(self, state: Program) -> Tuple:
        """Sort key for EVICTION (Router picks argmin, evicts lowest first).

        Defaults to sort_key, so for value-based policies evict = argmin(value) =
        lowest keep-priority (the natural inverse of admission). Override ONLY when
        eviction order differs from admission order: e.g. SizePolicy admits SMALL
        programs first AND wants to evict SMALL programs first -- those are the same
        direction on tokens, which a single key cannot express, so size gives evict
        a separate key (argmin -> smallest tokens)."""
        return self.sort_key(state)

    # ── Variable 3: admission timing (admission gate) ───────────────────────────
    def admits(self, state: Program) -> bool:
        """Whether this program may be admitted right now.
        True = greedily fill capacity; dual_descent overrides to admit only when
        density > lambda, deliberately leaving KV headroom."""
        return True

    # ── Variable 4: eviction timing (proactive vs passive) ──────────────────────
    def proactive_evictions(
        self, backends: Dict[str, BackendState], programs: Dict[str, Program]
    ) -> Iterable[str]:
        """program_ids to proactively evict this epoch (even when KV has not overflowed).
        Default empty = passive: the Router only evicts when remaining_capacity < 0.
        FidelityPolicy overrides to return below-price ACTING programs (Gap 3)."""
        return ()

    # ── Per-epoch internal state update (e.g. dual_descent's lambda) ────────────
    def on_epoch(
        self, backends: Dict[str, BackendState], waiting: Dict[str, Program]
    ) -> None:
        """Called once per scheduler tick, before resume. Default no-op.
        `waiting` = the paused pool (program_id -> Program)."""
        return

    # ── Separate cache-retention interface (variables 4/5, richer than sort_key) ─
    # The single sort_key convention above ties admission order to eviction order.
    # That is too restrictive for retention policies where "which ready turn to run
    # first" and "which inactive prefix is worth keeping" are DIFFERENT quantities
    # (HazardGradePolicy). Such a policy overrides evict_key (per-victim ranking,
    # used by the Router's argmin fallback loop) and/or select_cache_victims (a
    # set-level 0/1 knapsack). Default returns None so the Router keeps its legacy
    # evict_key argmin loop unchanged for every existing policy.
    def cache_value(self, state: Program, now: float) -> float:
        """Retention value of an inactive (between-tool / paused) KV prefix.
        Higher = more valuable to keep resident. Default 0.0 (unused unless a
        policy overrides eviction). See HazardGradePolicy for the hazard rule."""
        return 0.0

    def select_cache_victims(
        self, backend: BackendState, programs: Dict[str, Program], now: float
    ) -> Optional[Iterable[str]]:
        """program_ids to evict THIS overflow, chosen as a set (e.g. keep the
        knapsack-optimal retained subset, evict the complement). Return None to
        defer to the Router's legacy per-victim argmin(evict_key) loop -- this is
        the default for size/density/dual_descent/fidelity, so their behaviour is
        unchanged. `programs` = the programs currently on `backend`."""
        return None
