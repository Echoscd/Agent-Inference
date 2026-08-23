"""HazardGradePolicy: prediction-light admission grade + hazard cache knapsack.

Online port of `hazard_grade_knapsack`, the recommended deployment candidate from
the revised offline simulator (`agentic_serving_modified/`). See its
IMPLEMENTATION_GUIDE.md. Three ideas, none of which need a realized next decode
length or a realized remaining tool time:

  1. ADMISSION grade (variable 2, sort_key): rank ready turns by
       run_score = 1 / (current_work * remaining_work * admit_blocks)
     -- short current work AND short posterior remaining work first. This is the
     student's strong ordering with the unstable global dual price REMOVED
     (admits() is always True; the revised held-out runs found lambda == 0 in all
     45 scenarios, so the price reduced to static ordering anyway).

  2. RESERVATION (variable 1, peak_pad): reserve a fixed output QUANTILE, not the
     realized/oracle per-request decode length. Deployment must stay non-clairvoyant.

  3. RETENTION (variables 4/5): value an inactive (between-tool) prefix by the
     probability its tool returns within a short horizon x its cold-recompute cost
       cache_value = cold_cost * return_prob * (1 + completion_bonus * terminal_prob)
     and keep the max-value subset that fits, via an exact 0/1 KV-block knapsack
     (select_cache_victims); evict the complement. evict_key falls back to the
     same per-victim value for the Router's argmin loop.

Distributions come from `tool_hazard.py` (round posterior + lognormal tool model)
and are class-level only -- replace them with fits from real ThunderAgent traces
before reading effect sizes as a GPU forecast.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .base import SchedulingPolicy
from .tool_hazard import ROUND_PRIORS, RoundPrior, tool_model
from ..program import Program, ProgramStatus, ProgramState
from ..backend import BackendState


def exact_cache_knapsack(items: Sequence[Tuple[str, int, float]], capacity: int) -> Set[str]:
    """Exact 0/1 knapsack in KV-block units (max retained value <= capacity),
    with a bounded greedy value-density fallback for very large tables. Pure-Python
    port of the reference simulator's exact_cache_knapsack."""
    valid = [(pid, int(w), float(v)) for pid, w, v in items if w > 0 and w <= capacity and v > 0.0]
    if capacity <= 0 or not valid:
        return set()
    if sum(w for _, w, _ in valid) <= capacity:
        return {pid for pid, _, _ in valid}
    n = len(valid)
    # Bounded-cost guard (mirrors the reference 5M-cell cap): greedy by value/block.
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
        # iterate capacities descending so each item is used at most once
        for c in range(capacity, w - 1, -1):
            cand = dp[c - w] + value
            if cand > dp[c] + 1e-14:
                dp[c] = cand
                take[i][c] = True
    # best achievable capacity
    best_c = max(range(capacity + 1), key=lambda c: dp[c])
    selected: Set[str] = set()
    c = best_c
    for i in range(n - 1, -1, -1):
        if take[i][c]:
            pid, w, _ = valid[i]
            selected.add(pid)
            c -= w
    return selected


class HazardGradePolicy(SchedulingPolicy):
    name = "hazard_grade"

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
    ):
        self.alpha = alpha
        self.decode_mean = float(decode_mean)
        self.prompt_mean = float(prompt_mean)
        self.decode_reserve = int(decode_reserve)
        self.horizon_s = float(horizon_s)
        self.completion_bonus = float(completion_bonus)
        self.prior: RoundPrior = ROUND_PRIORS.get(prior_name, ROUND_PRIORS["swebench9"])
        self.default_tool_class = default_tool_class
        self.block_size_hint = int(block_size_hint)

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _blocks(tokens: int, block_size: int) -> int:
        return max(1, int(math.ceil(max(0, tokens) / max(1, block_size))))

    def _warm_fraction(self, s: Program) -> float:
        if s.last_prompt_tokens > 0:
            return min(1.0, max(0.0, s.last_cached_tokens / s.last_prompt_tokens))
        return 0.0

    # ── Variable 1: reservation = fixed output quantile (NOT known_decode) ───────
    def peak_pad(self, state: Program) -> int:
        """Distributional Q~95 decode reservation. Deliberately independent of any
        realized/oracle per-request decode length so the policy is non-clairvoyant."""
        return self.decode_reserve

    # ── Variable 3: no global price gate -> greedily admit by grade ──────────────
    def admits(self, state: Program) -> bool:
        return True

    # ── Variable 2: admission grade (run_score); LARGER = admit first ────────────
    def _priority(self, s: Program) -> float:
        warm = self._warm_fraction(s)
        uncached = float(s.total_tokens) * (1.0 - warm)
        current_work = self.alpha * uncached + self.decode_mean
        future_work = self.alpha * self.prompt_mean + self.decode_mean
        remaining = self.prior.remaining_mean(max(1, s.step_count))
        remaining_work = current_work + max(0.0, remaining - 1.0) * future_work
        blocks = self._blocks(s.total_tokens, self.block_size_hint)
        return 1.0 / max(1e-12, current_work * remaining_work * blocks)

    def sort_key(self, state: Program) -> Tuple:
        return (self._priority(state),)

    # ── Variables 4/5: retention value (higher = keep). Used by evict_key argmin ─
    def cache_value(self, state: Program, now: float) -> float:
        # return probability: an ACTING program is mid-tool -> hazard of returning
        # soon; anything else is treated as ready (prob 1).
        if state.status == ProgramStatus.ACTING and state.acting_since is not None:
            age = max(0.0, now - state.acting_since)
            prob = tool_model(getattr(state, "tool_class", "") or self.default_tool_class) \
                .residual_return_prob(age, self.horizon_s)
        else:
            prob = 1.0
        cold_cost = self.alpha * float(state.total_tokens)  # recompute cost if evicted
        stage = 1.0 + self.completion_bonus * self.prior.terminal_probability(max(1, state.step_count))
        return cold_cost * prob * stage

    def evict_key(self, state: Program) -> Tuple:
        # argmin(evict_key) -> evict the LOWEST-value prefix first (Router fallback
        # loop + residual cleanup after select_cache_victims).
        return (self.cache_value(state, time.time()),)

    # ── Variables 4/5: set-level 0/1 knapsack victim selection ───────────────────
    def select_cache_victims(
        self, backend: BackendState, programs: Dict[str, Program], now: float
    ) -> Optional[Iterable[str]]:
        cc = backend.cache_config
        if not cc or not getattr(cc, "block_size", 0):
            return None  # no block info -> defer to legacy argmin loop
        bs = cc.block_size
        overflow = -backend.remaining_capacity()
        if overflow <= 0:
            return []  # already within capacity
        # Candidates safe to pause: ACTING programs (between turns, no in-flight
        # generation). REASONING (on GPU) is left to the Router's mark path.
        acting = [
            (pid, s) for pid, s in programs.items()
            if s.status == ProgramStatus.ACTING and s.state == ProgramState.ACTIVE
        ]
        if not acting:
            return None  # nothing to shed here; legacy loop handles REASONING marks
        items = [(pid, self._blocks(s.total_tokens, bs), self.cache_value(s, now)) for pid, s in acting]
        total_blocks = sum(w for _, w, _ in items)
        overflow_blocks = int(math.ceil(overflow / bs))
        # We must shed >= overflow_blocks; retain the max-value subset that fits in
        # what remains, evict the complement.
        retain_cap = max(0, total_blocks - overflow_blocks)
        retained = exact_cache_knapsack(items, retain_cap)
        victims = [pid for pid, _, _ in items if pid not in retained]
        return victims
