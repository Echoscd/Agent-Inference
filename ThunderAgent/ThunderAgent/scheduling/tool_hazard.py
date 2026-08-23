"""Prediction-light distributions used by HazardGradePolicy.

Ported (pure-Python, no numpy/scipy) from the revised offline simulator
`agentic_serving_modified/code/realistic_agentic_sim.py`:

  * RoundPrior  -- discrete posterior over a program's TOTAL number of LLM rounds
                   (DiscretePrior in the reference). Gives, at the current step,
                   the terminal probability and the expected remaining rounds.
  * ToolModel   -- lognormal class-level tool-duration model. Gives the residual
                   return probability P(tool returns within H | not yet returned).

Both consume only class-level statistics and the OBSERVABLE elapsed age/step --
never a realized future decode length or a realized remaining tool time -- so the
policy stays non-clairvoyant (see IMPLEMENTATION_GUIDE.md, "Non-clairvoyance").
"""
from __future__ import annotations

import math
from typing import Dict, List


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via erf (avoids a scipy dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class RoundPrior:
    """Discrete prior over the total round count K in {1..k_max}.

    Mirrors the reference DiscretePrior: a discretized Gamma weight
    w(k) proportional to k^(shape-1) * exp(-k/scale), normalized.
    """

    def __init__(self, name: str, pmf: List[float]):
        self.name = name
        self.pmf = pmf
        self.k_max = len(pmf)
        # survival[j] = P(K >= j) for j in 1..k_max ; 0 beyond.
        surv = [0.0] * (self.k_max + 2)
        for j in range(1, self.k_max + 2):
            surv[j] = sum(pmf[j - 1:]) if j <= self.k_max else 0.0
        self._survival = surv

    @staticmethod
    def discrete_gamma(name: str, shape: float, scale: float, k_max: int) -> "RoundPrior":
        ks = list(range(1, k_max + 1))
        logw = [(shape - 1.0) * math.log(k) - k / scale for k in ks]
        m = max(logw)
        w = [math.exp(x - m) for x in logw]
        s = sum(w)
        w = [x / s for x in w]
        return RoundPrior(name, w)

    @property
    def mean(self) -> float:
        return sum((i + 1) * p for i, p in enumerate(self.pmf))

    def tail(self, j: int) -> float:
        """P(K >= j)."""
        if j <= 0:
            return 1.0
        if j > self.k_max:
            return 0.0
        return self._survival[j]

    def terminal_probability(self, stage: int) -> float:
        """P(K == stage | K >= stage): chance the CURRENT round is the last one."""
        stage = max(1, int(stage))
        den = self.tail(stage)
        if den <= 0.0 or stage > self.k_max:
            return 1.0
        return self.pmf[stage - 1] / den

    def remaining_mean(self, stage: int) -> float:
        """E[rounds still to run | K >= stage], counting the current round as 1."""
        stage = max(1, int(stage))
        den = self.tail(stage)
        if den <= 0.0:
            return 1.0
        total = 0.0
        for k in range(stage, self.k_max + 1):
            total += (k - stage + 1.0) * (self.pmf[k - 1] / den)
        return total


# Named priors (same parameters as the reference simulator). "swebench9" is the
# 8-9 turn average observed in the real SWE-bench runs and is the online default.
ROUND_PRIORS: Dict[str, RoundPrior] = {
    "short": RoundPrior.discrete_gamma("short", 2.0, 2.0, 30),
    "swebench9": RoundPrior.discrete_gamma("swebench9", 2.5, 3.5, 60),
    "quick10": RoundPrior.discrete_gamma("quick10", 2.5, 4.0, 60),
    "long": RoundPrior.discrete_gamma("long", 3.0, 6.2, 80),
    # ~16-round average measured from the real Qwen3-Coder-30B SWE-bench run
    # (exp 23: mean 14.3, median 16, capped at max_turns 20). For non-CoT coder
    # agents that take many short turns.
    "coder16": RoundPrior.discrete_gamma("coder16", 5.0, 3.2, 40),
}


class ToolModel:
    """Lognormal class-level tool-duration model."""

    def __init__(self, name: str, mean_s: float, cv: float):
        self.name = name
        self.mean_s = mean_s
        self.cv = cv
        self.sigma = math.sqrt(math.log1p(cv * cv))
        self.mu = math.log(mean_s) - 0.5 * self.sigma * self.sigma

    def cdf(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        z = (math.log(t) - self.mu) / max(self.sigma, 1e-12)
        return _normal_cdf(z)

    def residual_return_prob(self, elapsed: float, horizon: float) -> float:
        """P(T <= elapsed + horizon | T > elapsed), clipped to [0, 1].

        Rises toward 1 as the tool ages past its typical duration -- an old tool
        is about to return, so its warm prefix is worth keeping."""
        f0 = self.cdf(max(0.0, elapsed))
        f1 = self.cdf(max(0.0, elapsed + horizon))
        p = (f1 - f0) / max(1e-12, 1.0 - f0)
        return min(1.0, max(0.0, p))


# Class-level tool models. "pooled" is the sparse-class fallback (mixture-mean of
# the swebench_proxy mix); replace all of these with an empirical per-class
# survival table fit from real ThunderAgent tool timestamps before trusting the
# effect sizes (see IMPLEMENTATION_GUIDE.md step 4).
TOOL_MODELS: Dict[str, ToolModel] = {
    "fast": ToolModel("fast", mean_s=1.8, cv=0.55),
    "medium": ToolModel("medium", mean_s=9.0, cv=0.90),
    "long": ToolModel("long", mean_s=48.0, cv=1.25),
    "pooled": ToolModel("pooled", mean_s=13.5, cv=1.0),
}


def tool_model(tool_class: str) -> ToolModel:
    """Look up a class model, falling back to the pooled distribution."""
    return TOOL_MODELS.get(tool_class or "pooled", TOOL_MODELS["pooled"])
