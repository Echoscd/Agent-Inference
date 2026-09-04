"""Sim0823Policy: standalone online implementation of the 2026-08-23 simulator
policy `hazard_grade_knapsack`.

This is written directly against the simulator (`sim/code/realistic_agentic_sim.py`,
`HazardBase` + `HazardGradeKnapsackPolicy`), NOT as a modification of
`hazard_grade.py`. That file is a July port of an earlier simulator revision;
sharing code with it would mean a future edit there silently changes this
algorithm, and the two differ in enough decisions that the inheritance would be
misleading rather than economical. Everything this policy needs is defined here.

The algorithm, in the simulator's own terms:

  ADMISSION AMOUNT (peak_pad)
    A fixed decode quantile, never the realized decode length. The policy is
    non-clairvoyant by construction: `known_decode` is deliberately ignored.

  ADMISSION ORDER (sort_key)
    priority = 1 / (current_work * remaining_work * admit_blocks)
      current_work   = alpha * uncached + decode_mean
      future_work    = alpha * prompt_mean + decode_mean
      remaining_work = current_work + max(0, remaining_rounds - 1) * future_work
      uncached       = prompt if the prefix is warm, else the whole footprint
      admit_blocks   = blocks(footprint + decode reservation)
    Short current work AND short posterior remaining work first, discounted by
    the KV the admission will actually cost. No realized output length and no
    remaining tool time enter.

  ADMISSION GATE (admits)
    Greedy up to a per-tick batch target of `max_batch - active`, with one
    exception: if the backend is idle and something is waiting, exactly one
    program is force-admitted so an oversized program cannot deadlock.

  RETENTION VALUE (cache_value)
    cold_recompute_cost * P(tool returns within horizon | age) * stage_bonus
      cold_recompute_cost = alpha * t * (1 + ctx_penalty * t / ctx_ref)
        superlinear in context, as in the simulator's prefill curve. This matters:
        against a linear cost the eviction density below cancels to a constant.
      stage_bonus = 1 + completion_bonus * P(this is the last round)

  EVICTION ORDER (evict_key)
    cache_value / blocks -- value DENSITY. Shedding a large low-value prefix
    frees more KV than a small one of equal value, so density is what ranks.

  EVICTION SET (select_cache_victims)
    Exact 0/1 knapsack in KV-block units over the pausable prefixes: keep the
    maximum-value subset that fits in the headroom, evict the complement.

Deliberately not implemented: the simulator's `age_bonus` and `safety_fraction`.
`HazardGradeKnapsackPolicy.__init__` sets both to 0, so they are inert in the
policy being ported, and `Program` carries no waiting-age field to drive them.

The round prior and tool hazard model come from `tool_hazard.py` and are
class-level distributions, not fits to ThunderAgent traces. Treat any effect
size from this policy as a hypothesis until they are fitted.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .base import SchedulingPolicy
from .tool_hazard import ROUND_PRIORS, RoundPrior, tool_model
from ..program import Program, ProgramStatus, ProgramState
from ..backend import BackendState


def block_knapsack(items: Sequence[Tuple[str, int, float]], capacity: int) -> Set[str]:
    """Exact 0/1 knapsack in KV-block units: the max-value subset with total
    blocks <= capacity. Falls back to greedy value-density when the DP table
    would exceed ~5M cells, matching the simulator's bound.

    Pure Python: the online path runs on a handful of programs per tick, so the
    numpy vectorisation the simulator uses buys nothing here.
    """
    valid = [(pid, int(w), float(v)) for pid, w, v in items
             if w > 0 and w <= capacity and v > 0.0]
    if capacity <= 0 or not valid:
        return set()
    if sum(w for _, w, _ in valid) <= capacity:
        return {pid for pid, _, _ in valid}

    n = len(valid)
    if n * (capacity + 1) > 5_000_000:
        chosen: Set[str] = set()
        used = 0
        for pid, w, v in sorted(valid, key=lambda z: (z[2] / z[1], z[2]), reverse=True):
            if used + w <= capacity:
                chosen.add(pid)
                used += w
        return chosen

    NEG = float("-inf")
    dp = [NEG] * (capacity + 1)
    dp[0] = 0.0
    take = [[False] * (capacity + 1) for _ in range(n)]
    for i, (_pid, w, value) in enumerate(valid):
        for c in range(capacity, w - 1, -1):      # descending: each item used once
            cand = dp[c - w] + value
            if cand > dp[c] + 1e-14:
                dp[c] = cand
                take[i][c] = True
    c = max(range(capacity + 1), key=lambda k: dp[k])
    selected: Set[str] = set()
    for i in range(n - 1, -1, -1):
        pid, w, _ = valid[i]
        if c >= w and take[i][c]:
            selected.add(pid)
            c -= w
    return selected


class Sim0823Policy(SchedulingPolicy):
    """The 2026-08-23 simulator policy, online."""

    name = "sim0823"

    def __init__(
        self,
        *,
        alpha: float = 0.03,
        decode_mean: float = 1650.0,
        prompt_mean: float = 1800.0,
        decode_reserve: int = 4096,
        horizon_s: float = 10.0,
        completion_bonus: float = 1.5,
        prior_name: str = "swebench9",
        default_tool_class: str = "pooled",
        block_size_hint: int = 16,
        max_batch: int = 64,
        warm_hit_threshold: float = 0.5,
        ctx_penalty: float = 0.35,
        ctx_ref: float = 32768.0,
    ):
        self.alpha = float(alpha)
        self.decode_mean = float(decode_mean)
        self.prompt_mean = float(prompt_mean)
        self.decode_reserve = int(decode_reserve)
        self.horizon_s = float(horizon_s)
        self.completion_bonus = float(completion_bonus)
        self.prior: RoundPrior = ROUND_PRIORS.get(prior_name, ROUND_PRIORS["swebench9"])
        self.default_tool_class = default_tool_class
        self.block_size_hint = int(block_size_hint)
        self.max_batch = int(max_batch)
        # a prefix counts as warm when this fraction of the last prompt was a hit
        self.warm_hit_threshold = float(warm_hit_threshold)
        # simulator ServiceModel prefill curve: cost is superlinear in context
        self.ctx_penalty = float(ctx_penalty)
        self.ctx_ref = float(ctx_ref)
        # per-tick admission budget, refreshed by on_epoch
        self._slots_left: Optional[int] = None
        self._force_one = False

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _blocks(tokens: float, block_size: int) -> int:
        return max(1, int(math.ceil(max(0.0, tokens) / max(1, block_size))))

    def _uncached_tokens(self, s: Program) -> float:
        """sim: `prompt + (0 if warm else prefix)`.

        Online, a program is warm when most of its last request was a prefix-cache
        hit; then only the incremental prompt is uncached work. Otherwise the whole
        footprint has to be recomputed.
        """
        if s.last_prompt_tokens > 0 and \
           s.last_cached_tokens >= self.warm_hit_threshold * s.last_prompt_tokens:
            return float(max(0, s.last_prompt_tokens - s.last_cached_tokens))
        return float(s.total_tokens)

    def _admit_blocks(self, s: Program) -> int:
        """KV the admission actually costs: footprint plus the reservation."""
        return self._blocks(s.total_tokens + self.decode_reserve, self.block_size_hint)

    def _cold_cost(self, tokens: float) -> float:
        """Cost of recomputing a dropped prefix. Superlinear in context, mirroring
        `cold_prefill_seconds(t) = t / aggregate_prefill_tps(1, t)`."""
        return self.alpha * tokens * (1.0 + self.ctx_penalty * tokens / max(1e-9, self.ctx_ref))

    # ── Variable 1: admission amount ───────────────────────────────────────────
    def peak_pad(self, state: Program) -> int:
        """Fixed decode quantile. `state.known_decode` is ignored on purpose."""
        return self.decode_reserve

    # ── Variable 2: admission order ────────────────────────────────────────────
    def _priority(self, s: Program) -> float:
        uncached = self._uncached_tokens(s)
        current_work = self.alpha * uncached + self.decode_mean
        future_work = self.alpha * self.prompt_mean + self.decode_mean
        remaining = self.prior.remaining_mean(max(1, s.step_count))
        remaining_work = current_work + max(0.0, remaining - 1.0) * future_work
        return 1.0 / max(1e-12, current_work * remaining_work * self._admit_blocks(s))

    def sort_key(self, state: Program) -> Tuple:
        return (self._priority(state),)

    # ── Variable 3: admission gate ─────────────────────────────────────────────
    def on_epoch(
        self, backends: Dict[str, BackendState], waiting: Dict[str, Program]
    ) -> None:
        active = sum(b.active_program_count for b in backends.values())
        self._slots_left = max(0, self.max_batch - active)
        self._force_one = (active == 0 and bool(waiting))

    def admits(self, state: Program) -> bool:
        if self._slots_left is None:       # before the first tick: no budget yet
            return True
        if self._slots_left > 0:
            self._slots_left -= 1
            return True
        if self._force_one:                # idle backend must not deadlock
            self._force_one = False
            return True
        return False

    # ── Variables 4/5: retention ───────────────────────────────────────────────
    def cache_value(self, state: Program, now: float) -> float:
        t = float(state.total_tokens)
        if state.status == ProgramStatus.ACTING and state.acting_since is not None:
            age = max(0.0, now - state.acting_since)
            prob = tool_model(getattr(state, "tool_class", "") or self.default_tool_class) \
                .residual_return_prob(age, self.horizon_s)
        else:
            prob = 1.0
        stage = 1.0 + self.completion_bonus * self.prior.terminal_probability(max(1, state.step_count))
        return self._cold_cost(t) * prob * stage

    def evict_key(self, state: Program) -> Tuple:
        """Router evicts argmin. Rank by value DENSITY, not raw value."""
        blocks = self._blocks(state.total_tokens, self.block_size_hint)
        return (self.cache_value(state, time.time()) / max(1, blocks),)

    def select_cache_victims(
        self, backend: BackendState, programs: Dict[str, Program], now: float
    ) -> Optional[Iterable[str]]:
        cc = backend.cache_config
        if not cc or not getattr(cc, "block_size", 0):
            return None                      # no block info: defer to the argmin loop
        bs = cc.block_size
        overflow = -backend.remaining_capacity()
        if overflow <= 0:
            return []
        # Only ACTING programs are safe to shed: they are between turns with no
        # generation in flight. REASONING is left to the Router's mark path.
        pausable = [(pid, s) for pid, s in programs.items()
                    if s.status == ProgramStatus.ACTING and s.state == ProgramState.ACTIVE]
        if not pausable:
            return None
        items: List[Tuple[str, int, float]] = [
            (pid, self._blocks(s.total_tokens, bs), self.cache_value(s, now))
            for pid, s in pausable
        ]
        total_blocks = sum(w for _, w, _ in items)
        overflow_blocks = int(math.ceil(overflow / bs))
        retain_cap = max(0, total_blocks - overflow_blocks)
        retained = block_knapsack(items, retain_cap)
        return [pid for pid, _, _ in items if pid not in retained]
