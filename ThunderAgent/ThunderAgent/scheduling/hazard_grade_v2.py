"""HazardGradeV2Policy: online port of the 2026-08-23 `hazard_grade_knapsack`.

`hazard_grade.py` was ported in July from an earlier revision of the offline
simulator. The simulator was revised on 2026-08-23 and its
`HazardGradeKnapsackPolicy` / `HazardBase` now differ from that port in four
ways that change decisions, not just constants. This policy carries those four
changes across so the deployed algorithm matches the one the simulator evaluates.

Diff against `hazard_grade` (sim reference in parentheses):

  1. EVICTION ORDER is value DENSITY, not raw value.
     sim `HazardBase.emergency_order` ranks by `cache_value / cache_blocks`;
     the July port's `evict_key` returned raw `cache_value`. Raw value evicts a
     large low-value prefix and a small low-value prefix in the same order even
     though shedding the large one frees far more KV. Density is what the
     simulator measures.

  2. ADMISSION GRADE denominator counts the RESERVATION, not just the prompt.
     sim divides by `admit_blocks` = blocks(prefix + prompt + decode quantile);
     the port divided by blocks(total_tokens), omitting the decode reservation
     it then goes on to reserve in `peak_pad`. Since the reservation is a fixed
     4096 tokens it dominates short programs, so omitting it systematically
     over-ranked them.

  3. WARM PREFIX is a BINARY flag, not a continuous fraction.
     sim: `uncached = prompt + (0 if (cache_warm and prefix > 0) else prefix)`.
     The port used `total_tokens * (1 - cached/prompt)`, which mixes a hit ratio
     measured on the last request into a token count for this one.

  4. COLD-RECOMPUTE COST IS SUPERLINEAR IN CONTEXT.
     sim `cold_prefill_seconds(t) = t / aggregate_prefill_tps(1, t)`, and that
     tps carries a context penalty, so the cost is
       t * (1 + prefill_context_penalty * t / context_ref) / prefill_single_tps
     i.e. quadratic in t. The July port used a linear `alpha * total_tokens`.
     Linear cost matters here: divided by blocks (also linear in t) it cancels
     exactly, so change 1's value density degenerates to a constant and the
     eviction order becomes arbitrary. With the penalty term a large prefix is
     correctly more expensive to lose per block than a small one.

  5. BATCH TARGET + anti-deadlock fallback.
     sim `_plan_with_target` admits at most `max_batch - len(active)` programs
     per plan, and if nothing fits and nothing is active it force-admits the
     single smallest ready program. The port had `admits() -> True` with no cap
     and no fallback, delegating everything to the Router's capacity loop.

Not carried across, deliberately: the simulator's `age_bonus` aging term. Its
`HazardGradeKnapsackPolicy.__init__` sets `age_bonus=0.0` and `safety_fraction=0.0`,
so both are inert in the policy being ported; `Program` has no waiting-age field
to implement them against anyway.

Distributions come from `tool_hazard.py` and are class-level only. They are NOT
fitted to real ThunderAgent traces, so treat any effect size as a hypothesis.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, Optional, Tuple

from .hazard_grade import HazardGradePolicy, exact_cache_knapsack
from .tool_hazard import tool_model
from ..program import Program, ProgramStatus, ProgramState
from ..backend import BackendState


class HazardGradeV2Policy(HazardGradePolicy):
    """The 2026-08-23 simulator policy, online. Inherits peak_pad, cache_value and
    the knapsack from `hazard_grade`; overrides the four decisions listed above."""

    name = "hazard_grade_v2"

    def __init__(self, *, max_batch: int = 64,
                 prefill_context_penalty: float = 0.35,
                 context_ref: float = 32768.0, **kwargs):
        super().__init__(**kwargs)
        # sim ServiceModel defaults; the cold-recompute cost curve, not a fit
        self.prefill_context_penalty = float(prefill_context_penalty)
        self.context_ref = float(context_ref)
        # sim: target = view.service.max_batch. Cap on how many programs may be
        # admitted per scheduler tick; the Router's capacity loop still applies.
        self.max_batch = int(max_batch)
        self._slots_left: Optional[int] = None   # set once per epoch by on_epoch
        self._forced_admit = False               # anti-deadlock fallback armed?

    # ── Change 3: binary warm flag ─────────────────────────────────────────────
    def _uncached_tokens(self, s: Program) -> float:
        """sim: prompt + (0 if warm else prefix).

        Online we see `total_tokens` (the whole footprint) and `last_cached_tokens`
        of the previous request. A program whose last request was mostly a cache
        hit is treated as warm, and then only the incremental prompt is uncached.
        """
        warm = (s.last_prompt_tokens > 0
                and s.last_cached_tokens >= 0.5 * s.last_prompt_tokens)
        if warm:
            # incremental prompt only: what the last turn added beyond its cache hit
            return float(max(0, s.last_prompt_tokens - s.last_cached_tokens))
        return float(s.total_tokens)

    # ── Change 2: reservation counts toward the admission footprint ────────────
    def _admit_blocks(self, s: Program) -> int:
        return self._blocks(s.total_tokens + self.decode_reserve, self.block_size_hint)

    def _priority(self, s: Program) -> float:
        uncached = self._uncached_tokens(s)
        current_work = self.alpha * uncached + self.decode_mean
        future_work = self.alpha * self.prompt_mean + self.decode_mean
        remaining = self.prior.remaining_mean(max(1, s.step_count))
        remaining_work = current_work + max(0.0, remaining - 1.0) * future_work
        return 1.0 / max(1e-12, current_work * remaining_work * self._admit_blocks(s))

    # ── Change 4: superlinear cold-recompute cost ──────────────────────────────
    def cache_value(self, state: Program, now: float) -> float:
        """v1's value with the cold cost made superlinear, as in the simulator.

        v1: cold_cost = alpha * tokens.  Here the context penalty is applied, so
        cold_cost = alpha * t * (1 + penalty * t / context_ref). Without it,
        cache_value / blocks is constant and change 1 has no effect.
        """
        t = float(state.total_tokens)
        penalty = 1.0 + self.prefill_context_penalty * t / max(1e-9, self.context_ref)
        if state.status == ProgramStatus.ACTING and state.acting_since is not None:
            age = max(0.0, now - state.acting_since)
            prob = tool_model(getattr(state, "tool_class", "") or self.default_tool_class) \
                .residual_return_prob(age, self.horizon_s)
        else:
            prob = 1.0
        stage = 1.0 + self.completion_bonus * self.prior.terminal_probability(max(1, state.step_count))
        return self.alpha * t * penalty * prob * stage

    # ── Change 1: evict by value DENSITY ───────────────────────────────────────
    def evict_key(self, state: Program) -> Tuple:
        blocks = self._blocks(state.total_tokens, self.block_size_hint)
        return (self.cache_value(state, time.time()) / max(1, blocks),)

    # ── Change 4: per-epoch batch target + anti-deadlock fallback ──────────────
    def on_epoch(
        self, backends: Dict[str, BackendState], waiting: Dict[str, Program]
    ) -> None:
        """sim: slots = max(0, target - len(view.active)), target = max_batch.

        Counted once per tick over all backends; `admits()` then spends the slots.
        """
        active = sum(b.active_program_count for b in backends.values())
        self._slots_left = max(0, self.max_batch - active)
        # sim: "if not admit and not view.active and ranked" -> force one in, so a
        # program larger than the whole budget cannot deadlock an idle backend.
        self._forced_admit = (active == 0 and bool(waiting))

    def admits(self, state: Program) -> bool:
        if self._slots_left is None:      # no epoch ran yet -> behave like v1
            return True
        if self._slots_left > 0:
            self._slots_left -= 1
            return True
        if self._forced_admit:            # idle backend: let exactly one through
            self._forced_admit = False
            return True
        return False

    # ── Retention: knapsack over the headroom, not "shed just enough" ──────────
    def select_cache_victims(
        self, backend: BackendState, programs: Dict[str, Program], now: float
    ) -> Optional[Iterable[str]]:
        """sim: keep = knapsack(caches not admitted, cap - used).

        v1 sheds the minimum that clears the overflow. The simulator instead packs
        the retained set into the headroom that remains after admission, which can
        evict more than the overflow when what is left is low-value, and less when
        a large prefix is worth keeping.
        """
        cc = backend.cache_config
        if not cc or not getattr(cc, "block_size", 0):
            return None
        bs = cc.block_size
        overflow = -backend.remaining_capacity()
        if overflow <= 0:
            return []
        pausable = [
            (pid, s) for pid, s in programs.items()
            if s.status == ProgramStatus.ACTING and s.state == ProgramState.ACTIVE
        ]
        if not pausable:
            return None
        items = [(pid, self._blocks(s.total_tokens, bs), self.cache_value(s, now))
                 for pid, s in pausable]
        total_blocks = sum(w for _, w, _ in items)
        overflow_blocks = int(math.ceil(overflow / bs))
        retain_cap = max(0, total_blocks - overflow_blocks)
        retained = exact_cache_knapsack(items, retain_cap)
        return [pid for pid, _, _ in items if pid not in retained]
