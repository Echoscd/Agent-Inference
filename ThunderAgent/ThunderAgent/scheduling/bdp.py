"""BDPPolicy: online port of `bdp` from the calibrated simulator
(branch sim/minimal-calibrated-bdp, commit f4e1722).

Bayesian SERPT plus one KV dual price. For a ready program i:

    mu_i      posterior expected remaining work (turns still to run, in work units)
    q_i       expected KV footprint over the rest of the program
    v_i       = 1 / mu_i                       -- shortest-expected-remaining first
    lambda    the marginal value density v/q that exactly fills currently free KV
    margin_i  = v_i - lambda * q_i

Programs with a positive margin are admitted in descending margin order, subject
to KV feasibility and a batch cap. **There is no fitted policy parameter**: the
prior, the decode mixture and alpha come from the calibrated workload model, and
lambda is recomputed every tick by water-filling.

Two things this policy deliberately does NOT do, both because paired ablations in
the simulator showed they did not help:

  - router-level cache retention during tool phases (reuse comes from the
    engine's own prefix cache instead), so there is no knapsack and no
    `select_cache_victims` here;
  - any block normalisation or heuristic tie-break.

Non-clairvoyance: `Program.known_decode` (the A-arm tape's look-ahead) is never
read. The posterior uses `Program.observed_decode`, which is the program's own
finished turns.

Calibration defaults are the simulator's `policy_*` values, which are also the
flags the 26-29 experiments used: prompt_mean 1600, decode_mean 300, alpha 0.03,
decode_reserve 1024, prior coder16. The decode mixture (small 162/cv 1.19, large
3619/cv 0.80 at p=0.068, program cv 0.50) is the calibrated workload model.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .base import SchedulingPolicy
from .tool_hazard import ROUND_PRIORS, RoundPrior
from ..program import Program
from ..backend import BackendState


def _gamma_logpdf(x: float, shape: float, scale: float) -> float:
    if x <= 0.0 or shape <= 0.0 or scale <= 0.0:
        return -1e300
    return ((shape - 1.0) * math.log(x) - x / scale
            - math.lgamma(shape) - shape * math.log(scale))


class BDPPolicy(SchedulingPolicy):
    name = "bdp"

    def __init__(
        self,
        *,
        alpha: float = 0.03,
        decode_mean: float = 300.0,
        prompt_mean: float = 1600.0,
        decode_reserve: int = 1024,
        prior_name: str = "coder16",
        context_limit: int = 32768,   # = max-model-len 40960 - max_tokens 8192,
                                      # the point at which the server 400s the next
                                      # prompt. The simulator approximated it as 35000.
        max_batch: int = 80,
        # calibrated decode mixture, used only by the posterior shrinkage
        decode_small_mean: float = 162.0,
        decode_small_cv: float = 1.19,
        decode_large_prob: float = 0.068,
        decode_large_mean: float = 3619.0,
        decode_large_cv: float = 0.80,
        decode_program_cv: float = 0.50,
        block_size_hint: int = 16,
    ):
        self.alpha = float(alpha)
        self.decode_mean = float(decode_mean)
        self.prompt_mean = float(prompt_mean)
        self.decode_reserve = int(decode_reserve)
        self.prior: RoundPrior = ROUND_PRIORS.get(prior_name, ROUND_PRIORS["swebench9"])
        self.context_limit = float(context_limit)
        self.max_batch = int(max_batch)
        self.decode_small_mean = float(decode_small_mean)
        self.decode_small_cv = float(decode_small_cv)
        self.decode_large_prob = float(decode_large_prob)
        self.decode_large_mean = float(decode_large_mean)
        self.decode_large_cv = float(decode_large_cv)
        self.decode_program_cv = float(decode_program_cv)
        self.block_size_hint = int(block_size_hint)
        # per-epoch state
        self.lam = 0.0
        self.epoch = 0
        self.price_sum = 0.0
        self.price_max = 0.0
        self.binding_updates = 0
        self._slots_left: Optional[int] = None
        self._force_one = False

    # ── posterior over this program's decode scale ─────────────────────────────
    def _posterior_decode(self, s: Program) -> float:
        """decode_mean scaled by a Bayesian shrinkage over the program's own
        finished turns.

        A rare long turn is evidence for the large component of the calibrated
        mixture, not for the whole program being permanently heavy -- so each
        observation is first assigned a responsibility between the two
        components, then normalised by that component's mean.
        """
        obs = list(getattr(s, "observed_decode", ()) or ())
        if not obs or self.decode_program_cv <= 0.0:
            return self.decode_mean
        precision = 1.0 / (self.decode_program_cv ** 2)
        evidence = 0.0
        scale = 1.0
        small_shape = 1.0 / max(1e-12, self.decode_small_cv ** 2)
        large_cv = self.decode_large_cv or self.decode_small_cv
        large_shape = 1.0 / max(1e-12, large_cv ** 2)
        for seen, decode in enumerate(obs, start=1):
            normalized = decode / max(1e-12, self.decode_small_mean)
            if self.decode_large_prob > 0.0:
                log_small = (math.log(max(1e-12, 1.0 - self.decode_large_prob))
                             + _gamma_logpdf(decode, small_shape,
                                             scale * self.decode_small_mean / small_shape))
                log_large = (math.log(max(1e-12, self.decode_large_prob))
                             + _gamma_logpdf(decode, large_shape,
                                             scale * self.decode_large_mean / large_shape))
                d = max(-700.0, min(700.0, log_small - log_large))
                resp_large = 1.0 / (1.0 + math.exp(d))
                normalized = ((1.0 - resp_large) * decode / max(1e-12, self.decode_small_mean)
                              + resp_large * decode / max(1e-12, self.decode_large_mean))
            evidence += normalized
            scale = (precision + evidence) / (precision + seen)
        return self.decode_mean * scale

    # ── work estimate: remaining work and expected footprint ───────────────────
    def _estimate(self, s: Program) -> Tuple[float, float]:
        """-> (remaining_work, expected_footprint_tokens)."""
        warm = (s.last_prompt_tokens > 0
                and s.last_cached_tokens >= 0.5 * s.last_prompt_tokens)
        uncached = (float(max(0, s.last_prompt_tokens - s.last_cached_tokens))
                    if warm else float(s.total_tokens))
        dec = self._posterior_decode(s)
        current = self.alpha * uncached + dec
        future = self.alpha * self.prompt_mean + dec
        future_turns = max(0.0, self.prior.remaining_mean(max(1, s.step_count)) - 1.0)
        if self.context_limit > 0:
            # the harness refuses the next prompt once context passes its guard,
            # so condition the remaining-turn posterior on that observable budget
            context_after = float(s.total_tokens) + dec
            room = (self.context_limit - context_after + dec) / max(1e-12, self.prompt_mean + dec)
            future_turns = min(future_turns, max(0.0, room))
        remaining_work = current + future_turns * future
        footprint = (float(s.total_tokens) + dec
                     + 0.5 * future_turns * (self.prompt_mean + dec))
        return remaining_work, max(1.0, footprint)

    def _value(self, s: Program) -> float:
        remaining, _ = self._estimate(s)
        return 1.0 / max(1e-12, remaining)

    def _margin(self, s: Program) -> float:
        remaining, footprint = self._estimate(s)
        return 1.0 / max(1e-12, remaining) - self.lam * footprint

    # ── Variable 1: reserve a decode quantile, never the realized length ───────
    def peak_pad(self, state: Program) -> int:
        return self.decode_reserve

    # ── Variable 2/5: order by margin ─────────────────────────────────────────
    def sort_key(self, state: Program) -> Tuple:
        return (self._margin(state),)

    # ── Variable 3: price gate + batch cap ────────────────────────────────────
    def admits(self, state: Program) -> bool:
        if self._margin(state) <= 0.0:
            return bool(self._force_one) and self._take_forced()
        if self._slots_left is None:
            return True
        if self._slots_left > 0:
            self._slots_left -= 1
            return True
        return bool(self._force_one) and self._take_forced()

    def _take_forced(self) -> bool:
        self._force_one = False
        return True

    # ── per-epoch: water-fill lambda over the free KV ─────────────────────────
    def on_epoch(
        self, backends: Dict[str, BackendState], waiting: Dict[str, Program]
    ) -> None:
        free = sum(max(0, b.remaining_capacity()) for b in backends.values()
                   if b.cache_config)
        active = sum(b.active_program_count for b in backends.values())
        self._slots_left = max(0, self.max_batch - active)
        self._force_one = (active == 0 and bool(waiting))

        self.epoch += 1
        if not waiting:
            self.lam = 0.0
            self.price_sum += self.lam
            return
        densities: List[Tuple[float, float]] = []
        for s in waiting.values():
            remaining, footprint = self._estimate(s)
            densities.append((1.0 / max(1e-12, remaining) / footprint, footprint))
        densities.sort(reverse=True)
        self.lam = 0.0
        if sum(f for _d, f in densities) > free:
            used = 0.0
            for density, footprint in densities:
                self.lam = density
                used += footprint
                if used >= free:
                    break
            self.binding_updates += 1
        self.price_sum += self.lam
        self.price_max = max(self.price_max, self.lam)

    def diagnostics(self) -> Dict[str, float]:
        return {
            "bdp_lambda_last": self.lam,
            "bdp_lambda_mean": self.price_sum / max(1, self.epoch),
            "bdp_lambda_max": self.price_max,
            "bdp_updates": float(self.epoch),
            "bdp_binding_fraction": self.binding_updates / max(1, self.epoch),
        }
