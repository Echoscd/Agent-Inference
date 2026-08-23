#!/usr/bin/env python3
"""A prediction-light simulator for agentic LLM serving under KV pressure.

Compared with the student's original simulator, this version adds the system
features that most strongly affect real GPU behavior:

1. stochastic tool intervals between LLM turns;
2. no policy access to realized next decode length or remaining tool time;
3. continuously changing, concave batch-dependent decode throughput;
4. prefill/decode interference;
5. block-granular KV accounting and a distributional, rather than oracle,
   decode reserve;
6. a periodic outer scheduler, matching a practical router control loop.

Policies implemented:
  thunder                  Thunder-style phase/size policy with hysteresis.
  student_bdp              The student's Bayes-Dual-Price heuristic.
  hazard_knapsack          New hazard cache rule + Thunder-style admission.
  hazard_grade_knapsack    Recommended hybrid: BDP-grade admission + hazard cache knapsack.
  hazard_batch_knapsack    Experimental service-knee batch cap + hazard cache knapsack.

The default GPU service curves are intentionally labelled "uncalibrated".  They
have plausible saturation shapes but should be replaced by measured curves from
the target model/GPU before deployment claims are made.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import gamma as gamma_dist


# ---------------------------------------------------------------------------
# Workload distributions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscretePrior:
    name: str
    pmf: np.ndarray                 # support 1,...,k_max
    survival: np.ndarray            # survival[j] = P(K >= j)

    @property
    def k_max(self) -> int:
        return len(self.pmf)

    @property
    def mean(self) -> float:
        return float(np.dot(np.arange(1, self.k_max + 1), self.pmf))

    @staticmethod
    def discrete_gamma(name: str, shape: float, scale: float, k_max: int) -> "DiscretePrior":
        k = np.arange(1, k_max + 1, dtype=float)
        logw = (shape - 1.0) * np.log(k) - k / scale
        w = np.exp(logw - logw.max())
        w /= w.sum()
        surv = np.zeros(k_max + 2, dtype=float)
        for j in range(1, k_max + 2):
            surv[j] = float(w[j - 1 :].sum()) if j <= k_max else 0.0
        return DiscretePrior(name, w, surv)

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(np.arange(1, self.k_max + 1), p=self.pmf))

    def tail(self, j: int) -> float:
        if j <= 0:
            return 1.0
        if j > self.k_max:
            return 0.0
        return float(self.survival[j])

    def terminal_probability(self, stage: int) -> float:
        den = self.tail(stage)
        if den <= 0.0 or stage > self.k_max:
            return 1.0
        return float(self.pmf[stage - 1] / den)

    def remaining_mean(self, stage: int) -> float:
        den = self.tail(stage)
        if den <= 0.0:
            return 1.0
        k = np.arange(stage, self.k_max + 1, dtype=float)
        probs = self.pmf[stage - 1 :] / den
        return float(np.dot(k - stage + 1.0, probs))


PRIORS: Dict[str, DiscretePrior] = {
    "short": DiscretePrior.discrete_gamma("short", 2.0, 2.0, 30),
    # Proxy for the 8--9 turn average reported by the real SWE-bench runs.
    "swebench9": DiscretePrior.discrete_gamma("swebench9", 2.5, 3.5, 60),
    "quick10": DiscretePrior.discrete_gamma("quick10", 2.5, 4.0, 60),
    "long": DiscretePrior.discrete_gamma("long", 3.0, 6.2, 80),
}


@dataclass(frozen=True)
class ToolModel:
    name: str
    mean_s: float
    cv: float

    @property
    def sigma(self) -> float:
        return math.sqrt(math.log1p(self.cv * self.cv))

    @property
    def mu(self) -> float:
        return math.log(self.mean_s) - 0.5 * self.sigma * self.sigma

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.lognormal(self.mu, self.sigma))

    def cdf(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        z = (math.log(t) - self.mu) / max(self.sigma, 1e-12)
        return float(ndtr(z))

    def residual_return_prob(self, elapsed: float, horizon: float) -> float:
        """P(T <= elapsed+horizon | T > elapsed)."""
        f0 = self.cdf(max(0.0, elapsed))
        f1 = self.cdf(max(0.0, elapsed + horizon))
        return float(np.clip((f1 - f0) / max(1e-12, 1.0 - f0), 0.0, 1.0))


TOOL_MODELS: Dict[str, ToolModel] = {
    "fast": ToolModel("fast", mean_s=1.8, cv=0.55),
    "medium": ToolModel("medium", mean_s=9.0, cv=0.90),
    "long": ToolModel("long", mean_s=48.0, cv=1.25),
}

TOOL_MIXES: Dict[str, Dict[str, float]] = {
    "swe_regular": {"fast": 0.68, "medium": 0.27, "long": 0.05},
    "swe_mixed": {"fast": 0.50, "medium": 0.35, "long": 0.15},
    "heavy_tail": {"fast": 0.30, "medium": 0.35, "long": 0.35},
    # The presentation does not contain per-tool timestamps.  This mix is an
    # explicit proxy, not a trace fit, and should be replaced once those events
    # are exported from ThunderAgent.
    "swebench_proxy": {"fast": 0.45, "medium": 0.35, "long": 0.20},
}


@dataclass(frozen=True)
class RequestSpec:
    prompt_tokens: int
    decode_tokens_actual: int       # hidden from every policy
    tool_type_after: Optional[str]
    tool_duration_actual: float     # hidden from every policy


@dataclass(frozen=True)
class ProgramSpec:
    requests: Tuple[RequestSpec, ...]
    prior: DiscretePrior


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    n_programs: int = 64
    prompt_mean: float = 120.0
    prompt_cv: float = 0.55
    # Agent traces have a large repository/task prompt followed by smaller tool
    # deltas.  None preserves the original identical-per-round distribution.
    initial_prompt_mean: Optional[float] = None
    initial_prompt_cv: Optional[float] = None
    decode_mean: float = 150.0
    decode_cv: float = 0.80
    prior_name: str = "quick10"
    tool_mix: str = "swe_mixed"
    capacity_tokens: int = 56_000
    block_size: int = 16
    reserve_quantile: float = 0.95
    seed: int = 0


def _sample_gamma_int(rng: np.random.Generator, mean: float, cv: float) -> int:
    shape = 1.0 / max(cv * cv, 1e-12)
    scale = mean / shape
    return max(1, int(round(rng.gamma(shape, scale))))


def generate_workload(cfg: WorkloadConfig) -> Tuple[List[ProgramSpec], Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    prior = PRIORS[cfg.prior_name]
    mix = TOOL_MIXES[cfg.tool_mix]
    tool_names = list(mix)
    probs = np.array([mix[x] for x in tool_names], dtype=float)
    probs /= probs.sum()

    programs: List[ProgramSpec] = []
    total_requests = 0
    total_prompt = 0
    total_decode = 0
    total_initial_prompt = 0
    total_tool = 0.0
    max_context = 0
    for _ in range(cfg.n_programs):
        k = prior.sample(rng)
        total_requests += k
        reqs: List[RequestSpec] = []
        context = 0
        for j in range(k):
            if j == 0 and cfg.initial_prompt_mean is not None:
                pmean = cfg.initial_prompt_mean
                pcv = cfg.initial_prompt_cv if cfg.initial_prompt_cv is not None else cfg.prompt_cv
            else:
                pmean, pcv = cfg.prompt_mean, cfg.prompt_cv
            p = _sample_gamma_int(rng, pmean, pcv)
            d = _sample_gamma_int(rng, cfg.decode_mean, cfg.decode_cv)
            total_prompt += p
            total_decode += d
            if j == 0:
                total_initial_prompt += p
            if j + 1 < k:
                tname = str(rng.choice(tool_names, p=probs))
                tdur = TOOL_MODELS[tname].sample(rng)
                total_tool += tdur
            else:
                tname, tdur = None, 0.0
            reqs.append(RequestSpec(p, d, tname, tdur))
            context += p + d
        max_context = max(max_context, context)
        programs.append(ProgramSpec(tuple(reqs), prior))
    return programs, {
        "total_requests": float(total_requests),
        "mean_rounds": total_requests / cfg.n_programs,
        "mean_initial_prompt": total_initial_prompt / cfg.n_programs,
        "mean_prompt_per_turn": total_prompt / max(1, total_requests),
        "mean_decode_per_turn": total_decode / max(1, total_requests),
        "mean_total_prompt_per_program": total_prompt / cfg.n_programs,
        "mean_total_decode_per_program": total_decode / cfg.n_programs,
        "mean_tool_s_per_program": total_tool / cfg.n_programs,
        "max_final_context": float(max_context),
    }


# ---------------------------------------------------------------------------
# GPU service model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceModel:
    # Uncalibrated but plausible concave shapes.
    prefill_single_tps: float = 7_000.0
    prefill_sat: float = 4.0
    decode_single_tps: float = 45.0
    decode_sat: float = 16.0
    context_ref: float = 4_096.0
    prefill_context_penalty: float = 0.12
    decode_context_penalty: float = 0.30
    mixed_prefill_share: float = 0.30
    max_batch: int = 64

    def aggregate_prefill_tps(self, n: int, mean_uncached: float) -> float:
        if n <= 0:
            return 0.0
        gain = n / (1.0 + (n - 1.0) / self.prefill_sat)
        penalty = 1.0 + self.prefill_context_penalty * max(0.0, mean_uncached) / self.context_ref
        return self.prefill_single_tps * gain / penalty

    def aggregate_decode_tps(self, n: int, mean_context: float) -> float:
        if n <= 0:
            return 0.0
        n_eff = min(n, self.max_batch)
        gain = n_eff / (1.0 + (n_eff - 1.0) / self.decode_sat)
        penalty = 1.0 + self.decode_context_penalty * max(0.0, mean_context) / self.context_ref
        return self.decode_single_tps * gain / penalty

    def per_seq_decode_tps(self, n: int, mean_context: float) -> float:
        return self.aggregate_decode_tps(n, mean_context) / max(1, n)

    def batch_knee(self, marginal_ratio: float, mean_context: float) -> int:
        one = self.aggregate_decode_tps(1, mean_context)
        threshold = marginal_ratio * one
        for k in range(1, self.max_batch):
            delta = self.aggregate_decode_tps(k + 1, mean_context) - self.aggregate_decode_tps(k, mean_context)
            if delta <= threshold:
                return k
        return self.max_batch

    def cold_prefill_seconds(self, tokens: float) -> float:
        return max(0.0, tokens) / max(1e-9, self.aggregate_prefill_tps(1, tokens))


SERVICE_PROFILE_KWARGS: Dict[str, Dict[str, float]] = {
    "synthetic": {},
    # Aggregate proxy derived from the Qwen3-32B/vLLM summary: roughly
    # 250--300 decode tok/s around batch 10--15 at long contexts, slower cold
    # prefill than the original synthetic model, and stronger mixed interference.
    # It is intentionally named proxy because raw per-call GPU timings were not
    # available in the presentation.
    "qwen3_32b_vllm_proxy": {
        "prefill_single_tps": 2_500.0,
        "prefill_sat": 3.0,
        "decode_single_tps": 55.0,
        "decode_sat": 12.0,
        "context_ref": 32_768.0,
        "prefill_context_penalty": 0.35,
        "decode_context_penalty": 0.25,
        "mixed_prefill_share": 0.20,
        "max_batch": 64,
    },
}


def make_service_model(profile: str) -> ServiceModel:
    if profile not in SERVICE_PROFILE_KWARGS:
        raise ValueError(f"unknown service profile {profile!r}; known={sorted(SERVICE_PROFILE_KWARGS)}")
    return ServiceModel(**SERVICE_PROFILE_KWARGS[profile])


# ---------------------------------------------------------------------------
# Runtime state and policy-safe views
# ---------------------------------------------------------------------------


READY, PREFILL, DECODE, TOOL, DONE = "READY", "PREFILL", "DECODE", "TOOL", "DONE"


@dataclass
class RuntimeProgram:
    pid: int
    spec: ProgramSpec
    idx: int = 0
    phase: str = READY
    prefix: int = 0
    prompt: int = 0
    cache_warm: bool = False
    prefill_remaining: float = 0.0
    generated: float = 0.0
    active_kv: float = 0.0
    reserve_blocks: int = 0
    tool_elapsed: float = 0.0
    tool_remaining: float = 0.0
    ready_since: float = 0.0
    admitted_at: float = 0.0
    completion_time: float = math.nan

    @property
    def stage(self) -> int:
        return self.idx + 1

    @property
    def request(self) -> RequestSpec:
        return self.spec.requests[self.idx]

    @property
    def initial(self) -> bool:
        return self.idx == 0 and self.prefix == 0


@dataclass(frozen=True)
class ProgramView:
    pid: int
    phase: str
    stage: int
    prefix: int
    prompt: int
    cache_warm: bool
    tool_type: Optional[str]
    tool_elapsed: float
    waiting_age: float
    terminal_prob: float
    remaining_rounds: float
    initial: bool
    admit_blocks: int
    cache_blocks: int
    reserve_blocks: int
    current_context: float


@dataclass(frozen=True)
class SystemView:
    now: float
    block_size: int
    capacity_blocks: int
    decode_mean: float
    prompt_mean: float
    decode_reserve: int
    alpha_work: float
    control_interval: float
    service: ServiceModel
    programs: Tuple[ProgramView, ...]

    @property
    def ready(self) -> List[ProgramView]:
        return [p for p in self.programs if p.phase == READY]

    @property
    def active(self) -> List[ProgramView]:
        return [p for p in self.programs if p.phase in (PREFILL, DECODE)]

    @property
    def caches(self) -> List[ProgramView]:
        return [p for p in self.programs if p.cache_warm and p.cache_blocks > 0 and p.phase in (TOOL, READY)]

    @property
    def active_reserved(self) -> int:
        return int(sum(p.reserve_blocks for p in self.active))

    def return_prob(self, p: ProgramView, horizon: float) -> float:
        if p.phase == READY:
            return 1.0
        if p.phase == TOOL and p.tool_type:
            return TOOL_MODELS[p.tool_type].residual_return_prob(p.tool_elapsed, horizon)
        return 0.0


@dataclass
class Plan:
    admit: List[int] = field(default_factory=list)
    keep_cache: Set[int] = field(default_factory=set)


class Policy:
    name = "base"

    def reset(self, cfg: WorkloadConfig, service: ServiceModel) -> None:
        pass

    def plan(self, view: SystemView) -> Plan:
        raise NotImplementedError

    def emergency_order(self, view: SystemView) -> List[int]:
        return [p.pid for p in sorted(view.caches, key=lambda x: (x.cache_blocks, x.pid))]

    def diagnostics(self) -> Dict[str, float]:
        return {}


# ---------------------------------------------------------------------------
# Knapsack utility
# ---------------------------------------------------------------------------


def exact_cache_knapsack(items: Sequence[Tuple[int, int, float]], capacity: int) -> Set[int]:
    """Exact 0/1 knapsack in block units, with a bounded greedy fallback."""
    valid = [(pid, int(w), float(v)) for pid, w, v in items if w > 0 and w <= capacity and v > 0.0]
    if capacity <= 0 or not valid:
        return set()
    if sum(w for _, w, _ in valid) <= capacity:
        return {pid for pid, _, _ in valid}
    n = len(valid)
    if n * (capacity + 1) > 5_000_000:
        chosen: Set[int] = set()
        used = 0
        for pid, w, v in sorted(valid, key=lambda z: (z[2] / z[1], z[2]), reverse=True):
            if used + w <= capacity:
                chosen.add(pid)
                used += w
        return chosen

    neg = -1e300
    dp = np.full(capacity + 1, neg, dtype=np.float64)
    dp[0] = 0.0
    take = np.zeros((n, capacity + 1), dtype=np.bool_)
    for i, (_pid, w, value) in enumerate(valid):
        old = dp
        new = old.copy()
        cand = old[: capacity + 1 - w] + value
        better = cand > new[w:] + 1e-14
        if np.any(better):
            idx = np.flatnonzero(better) + w
            new[idx] = cand[better]
            take[i, idx] = True
        dp = new
    c = int(np.argmax(dp))
    selected: Set[int] = set()
    for i in range(n - 1, -1, -1):
        pid, w, _ = valid[i]
        if c >= w and take[i, c]:
            selected.add(pid)
            c -= w
    return selected


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class ThunderPolicy(Policy):
    """Single-worker approximation of Thunder's phase-aware scheduler."""

    name = "thunder"

    def __init__(self, high: float = 0.95, target: float = 0.80, hysteresis: float = 0.10,
                 decay_tau: float = 30.0, force_resume_s: float = 1_800.0):
        self.high = high
        self.target = target
        self.hysteresis = hysteresis
        self.decay_tau = decay_tau
        self.force_resume_s = force_resume_s

    def _evict_key(self, p: ProgramView) -> Tuple[float, float, int]:
        group = 0.0 if p.phase == TOOL else 1.0
        eff = p.cache_blocks * math.exp(-p.tool_elapsed / max(1e-9, self.decay_tau)) if p.phase == TOOL else p.cache_blocks
        return group, eff, p.pid

    def plan(self, view: SystemView) -> Plan:
        cache = {p.pid: p for p in view.caches}
        keep = set(cache)
        used = view.active_reserved + sum(p.cache_blocks for p in view.caches)
        admit: List[int] = []

        ready = sorted(
            view.ready,
            key=lambda p: (
                0 if p.waiting_age >= self.force_resume_s else 1,
                0 if not p.initial else 1,
                p.prefix + p.prompt,
                -p.waiting_age,
                p.pid,
            ),
        )

        # A practical work-conserving interpretation of Thunder's hysteresis:
        # when no reasoning request is running, reclaim acting/paused caches down
        # to the target before resuming.  Otherwise a cache-only state inside the
        # hysteresis band can leave the GPU idle indefinitely.
        if not view.active and ready and used > self.target * view.capacity_blocks:
            for victim in sorted((cache[x] for x in keep), key=self._evict_key):
                keep.remove(victim.pid)
                used -= victim.cache_blocks
                if used <= self.target * view.capacity_blocks + 1e-9:
                    break

        # Resume below the lower band, or whenever the GPU is empty after the
        # work-conserving release above.  Fill greedily up to the high watermark.
        can_resume = used <= (self.high - self.hysteresis) * view.capacity_blocks or (not view.active and ready)
        if can_resume:
            for p in ready:
                old = cache[p.pid].cache_blocks if p.pid in keep else 0
                inc = p.admit_blocks - old
                if used + inc <= self.high * view.capacity_blocks + 1e-9:
                    admit.append(p.pid)
                    keep.discard(p.pid)
                    used += inc

        # Under pressure, pause/evict ACTING programs first and stop at the lower
        # target, preserving non-preemptive reasoning requests.
        if used > self.high * view.capacity_blocks + 1e-9:
            for p in sorted((cache[x] for x in keep), key=self._evict_key):
                keep.remove(p.pid)
                used -= p.cache_blocks
                if used <= self.target * view.capacity_blocks + 1e-9:
                    break
        return Plan(admit, keep)

    def emergency_order(self, view: SystemView) -> List[int]:
        return [p.pid for p in sorted(view.caches, key=self._evict_key)]


class ThunderGreedyPolicy(Policy):
    """Work-conserving proxy for the real Size/Thunder scheduler in the slides.

    Unlike ``ThunderPolicy``, this policy has no high/low admission gate.  It
    greedily fills active reservations first on every epoch, then retains caches
    in the residual capacity.  The real SizePolicy evicts the smallest inactive
    footprint first, so the residual packing keeps larger prefixes first.
    """

    name = "thunder_greedy"

    def plan(self, view: SystemView) -> Plan:
        ready = sorted(
            view.ready,
            key=lambda p: (
                # A continuation returning from a tool is a reasoning/resume
                # candidate in the router state machine and precedes NEW work.
                0 if not p.initial else -1,
                -(p.prefix + p.prompt),
                p.waiting_age,
                -p.pid,
            ),
            reverse=True,
        )
        used = view.active_reserved
        admit: List[int] = []
        slots = max(0, view.service.max_batch - len(view.active))
        for p in ready:
            if slots <= 0:
                break
            if used + p.admit_blocks <= view.capacity_blocks:
                admit.append(p.pid)
                used += p.admit_blocks
                slots -= 1

        admitted = set(admit)
        keep: Set[int] = set()
        # Evict-smallest-first is equivalent to preferentially retaining larger
        # inactive prefixes when active reservations have already been placed.
        for p in sorted(view.caches, key=lambda x: (x.cache_blocks, x.pid), reverse=True):
            if p.pid in admitted:
                continue
            if used + p.cache_blocks <= view.capacity_blocks:
                keep.add(p.pid)
                used += p.cache_blocks
        return Plan(admit, keep)

    def emergency_order(self, view: SystemView) -> List[int]:
        return [p.pid for p in sorted(view.caches, key=lambda x: (x.cache_blocks, x.pid))]


class DualPriceProxyPolicy(Policy):
    """Approximation of the real density-minus-global-price policy in the slides.

    This is distinct from ``StudentBDPPolicy``: it uses the real implementation's
    one-turn density ``1 / (service_time * footprint)`` and a scalar admission
    gate.  Q95 decode reservation remains a simulator-wide safety mechanism; the
    real base dual policy used no peak pad, which requires active preemption to
    model faithfully and is therefore left as an explicit calibration gap.
    """

    name = "dual_price_proxy"

    def __init__(self, alpha: float = 0.03, decode_hat: float = 1_000.0, eta0: float = 0.10):
        self.alpha = alpha
        self.decode_hat = decode_hat
        self.eta0 = eta0
        self.lam = 0.0
        self.epoch = 0
        self.scale: Optional[float] = None

    def reset(self, cfg: WorkloadConfig, service: ServiceModel) -> None:
        self.lam, self.epoch, self.scale = 0.0, 0, None

    def density(self, p: ProgramView) -> float:
        footprint = max(1.0, float(p.prefix + p.prompt))
        tau = self.alpha * footprint + self.decode_hat
        return 1.0 / max(1e-12, tau * footprint)

    def plan(self, view: SystemView) -> Plan:
        density = {p.pid: self.density(p) for p in view.ready}
        free_tokens = max(0.0, (view.capacity_blocks - view.active_reserved) * view.block_size)
        if view.ready and free_tokens > 0.0:
            positive = [density[p.pid] for p in view.ready if density[p.pid] - self.lam > 0.0]
            desired = sum(
                (p.prefix + p.prompt)
                for p in view.ready
                if density[p.pid] - self.lam > 0.0
            )
            if self.scale is None:
                base = positive if positive else list(density.values())
                self.scale = max(float(np.median(base)), 1e-12)
            previous = self.lam
            step = self.eta0 * self.scale / math.sqrt(self.epoch + 1.0)
            proposed = max(0.0, previous + step * (desired / max(1.0, free_tokens) - 1.0))
            # Apply the documented +20% guard after the zero-price cold start.
            self.lam = min(proposed, 1.20 * previous) if previous > 0.0 else proposed
            self.epoch += 1

        ranked = sorted(
            (p for p in view.ready if density[p.pid] - self.lam > 0.0),
            key=lambda p: (density[p.pid] - self.lam, p.waiting_age, -p.pid),
            reverse=True,
        )
        used = view.active_reserved
        admit: List[int] = []
        for p in ranked:
            if used + p.admit_blocks <= view.capacity_blocks:
                admit.append(p.pid)
                used += p.admit_blocks

        admitted = set(admit)
        keep: Set[int] = set()
        caches = sorted(
            (p for p in view.caches if p.pid not in admitted),
            key=lambda p: (self.density(p), p.cache_blocks, -p.pid),
            reverse=True,
        )
        for p in caches:
            if used + p.cache_blocks <= view.capacity_blocks:
                keep.add(p.pid)
                used += p.cache_blocks
        return Plan(admit, keep)

    def diagnostics(self) -> Dict[str, float]:
        return {"dual_proxy_lambda_last": self.lam, "dual_proxy_updates": float(self.epoch)}


class StudentBDPPolicy(Policy):
    """Student Bayes-Dual-Price, using distributions rather than oracle d/tool time."""

    name = "student_bdp"

    def __init__(self, eta0: float = 0.10):
        self.eta0 = eta0
        self.lam = 0.0
        self.epoch = 0
        self.scale: Optional[float] = None

    def reset(self, cfg: WorkloadConfig, service: ServiceModel) -> None:
        self.lam, self.epoch, self.scale = 0.0, 0, None

    def _run(self, p: ProgramView, v: SystemView) -> Tuple[float, float, float, float]:
        warm = p.cache_warm and p.prefix > 0
        uncached = p.prompt + (0 if warm else p.prefix)
        tau = v.alpha_work * uncached + v.decode_mean
        future = v.alpha_work * v.prompt_mean + v.decode_mean
        mu = tau + max(0.0, p.remaining_rounds - 1.0) * future
        qbar = p.prefix + p.prompt + v.decode_mean + 0.5 * max(0.0, p.remaining_rounds - 1.0) * (v.prompt_mean + v.decode_mean)
        value = 1.0 / max(1e-12, tau * mu)
        return tau, mu, qbar, value

    def _hold_value(self, p: ProgramView, v: SystemView) -> float:
        prompt = p.prompt if p.phase == READY else v.prompt_mean
        tau = v.alpha_work * prompt + v.decode_mean
        future = v.alpha_work * v.prompt_mean + v.decode_mean
        mu = tau + max(0.0, p.remaining_rounds - 1.0) * future
        return v.alpha_work * p.prefix / max(1e-12, mu)

    def plan(self, view: SystemView) -> Plan:
        stats = {p.pid: self._run(p, view) for p in view.ready}
        free_tokens = max(0.0, (view.capacity_blocks - view.active_reserved) * view.block_size)
        ratios, desired = [], 0.0
        for p in view.ready:
            _tau, _mu, qbar, value = stats[p.pid]
            ratios.append(value / max(1.0, qbar))
            if value - self.lam * qbar > 0:
                desired += qbar
        if view.ready and free_tokens > 0:
            if self.scale is None:
                self.scale = max(float(np.median(ratios)), 1e-12)
            step = self.eta0 * self.scale / math.sqrt(self.epoch + 1.0)
            self.lam = max(0.0, self.lam + step * (desired / max(1.0, free_tokens) - 1.0))
            self.epoch += 1

        ranked = []
        for p in view.ready:
            tau, _mu, qbar, value = stats[p.pid]
            margin = value - self.lam * qbar
            if margin > 0:
                ranked.append((margin / max(1, p.admit_blocks), margin, -tau, p))
        ranked.sort(key=lambda z: z[:-1], reverse=True)
        used = view.active_reserved
        admit: List[int] = []
        for *_x, p in ranked:
            if used + p.admit_blocks <= view.capacity_blocks:
                admit.append(p.pid)
                used += p.admit_blocks
        if not admit and not view.active and view.ready:
            feasible = [p for p in view.ready if p.admit_blocks <= view.capacity_blocks]
            if feasible:
                p = min(feasible, key=lambda x: (x.admit_blocks, -x.waiting_age, x.pid))
                admit.append(p.pid)
                used += p.admit_blocks

        admitted = set(admit)
        holds = []
        for p in view.caches:
            if p.pid in admitted:
                continue
            margin = self._hold_value(p, view) - self.lam * p.cache_blocks * view.block_size
            if margin > 0:
                holds.append((margin / max(1, p.cache_blocks), margin, p))
        holds.sort(key=lambda z: z[:-1], reverse=True)
        keep: Set[int] = set()
        for *_x, p in holds:
            if used + p.cache_blocks <= view.capacity_blocks:
                keep.add(p.pid)
                used += p.cache_blocks
        return Plan(admit, keep)

    def emergency_order(self, view: SystemView) -> List[int]:
        x = []
        for p in view.caches:
            margin = self._hold_value(p, view) - self.lam * p.cache_blocks * view.block_size
            x.append((margin / max(1, p.cache_blocks), p.pid))
        return [pid for _, pid in sorted(x)]

    def diagnostics(self) -> Dict[str, float]:
        return {"bdp_lambda_last": self.lam, "bdp_updates": float(self.epoch)}


class HazardBase(Policy):
    def __init__(self, horizon_s: float = 10.0, completion_bonus: float = 1.5,
                 age_bonus: float = 0.35, age_scale_s: float = 120.0,
                 safety_fraction: float = 0.02):
        self.horizon_s = horizon_s
        self.completion_bonus = completion_bonus
        self.age_bonus = age_bonus
        self.age_scale_s = age_scale_s
        self.safety_fraction = safety_fraction
        self.knapsack_calls = 0
        self.knapsack_time = 0.0

    def reset(self, cfg: WorkloadConfig, service: ServiceModel) -> None:
        self.knapsack_calls = 0
        self.knapsack_time = 0.0

    def cache_value(self, p: ProgramView, v: SystemView) -> float:
        prob = v.return_prob(p, self.horizon_s)
        cold = v.service.cold_prefill_seconds(p.prefix)
        stage = 1.0 + self.completion_bonus * p.terminal_prob
        age = 1.0 + self.age_bonus * min(2.0, p.waiting_age / max(1e-9, self.age_scale_s))
        return cold * prob * stage * age

    def select_caches(self, candidates: Sequence[ProgramView], cap: int, v: SystemView) -> Set[int]:
        t0 = time.perf_counter()
        ans = exact_cache_knapsack([(p.pid, p.cache_blocks, self.cache_value(p, v)) for p in candidates], cap)
        self.knapsack_time += time.perf_counter() - t0
        self.knapsack_calls += 1
        return ans

    def emergency_order(self, view: SystemView) -> List[int]:
        scored = [(self.cache_value(p, view) / max(1, p.cache_blocks), p.pid) for p in view.caches]
        return [pid for _, pid in sorted(scored)]

    def diagnostics(self) -> Dict[str, float]:
        return {
            "knapsack_calls": float(self.knapsack_calls),
            "mean_knapsack_ms": 1000.0 * self.knapsack_time / max(1, self.knapsack_calls),
        }


class HazardKnapsackPolicy(HazardBase):
    name = "hazard_knapsack"

    def plan(self, view: SystemView) -> Plan:
        safety = int(math.ceil(self.safety_fraction * view.capacity_blocks))
        cap = max(0, view.capacity_blocks - safety)
        ready = sorted(view.ready, key=lambda p: (0 if not p.initial else 1, p.prefix + p.prompt, -p.waiting_age, p.pid))
        used = view.active_reserved
        admit: List[int] = []
        slots = max(0, view.service.max_batch - len(view.active))
        for p in ready:
            if slots <= 0:
                break
            if used + p.admit_blocks <= cap:
                admit.append(p.pid)
                used += p.admit_blocks
                slots -= 1
        admitted = set(admit)
        cache_cap = max(0, cap - used)
        keep = self.select_caches([p for p in view.caches if p.pid not in admitted], cache_cap, view)
        return Plan(admit, keep)


class HazardGradeKnapsackPolicy(HazardBase):
    """Implementation-first hybrid selected on the validation scenarios.

    It keeps the student's useful posterior current-work x remaining-work
    admission ordering, removes the unstable global dual gate, and replaces the
    hold rule by residual-tool-hazard values plus an exact cache knapsack.  It
    needs only stable distributions, not point predictions.
    """

    name = "hazard_grade_knapsack"

    def __init__(self, **kwargs):
        # The q95 distributional reservation already provides headroom.  Extra
        # fixed headroom and an aging distortion did not help on validation.
        kwargs.setdefault("age_bonus", 0.0)
        kwargs.setdefault("safety_fraction", 0.0)
        super().__init__(**kwargs)

    def priority(self, p: ProgramView, v: SystemView, target: int, mean_ctx: float) -> float:
        # Strong part of BDP: short current work and short posterior remaining
        # work.  No realized output length or remaining tool time enters.
        warm = p.cache_warm and p.prefix > 0
        uncached = p.prompt + (0 if warm else p.prefix)
        current_work = v.alpha_work * uncached + v.decode_mean
        future_work = v.alpha_work * v.prompt_mean + v.decode_mean
        remaining_work = current_work + max(0.0, p.remaining_rounds - 1.0) * future_work
        aging = 1.0 + self.age_bonus * min(2.0, p.waiting_age / max(1e-9, self.age_scale_s))
        return aging / max(1e-12, current_work * remaining_work * max(1, p.admit_blocks))

    def _plan_with_target(self, view: SystemView, target: int) -> Plan:
        contexts = [p.current_context for p in view.active] + [p.prefix + p.prompt for p in view.ready]
        mean_ctx = float(np.mean(contexts)) if contexts else 2_000.0
        ranked = sorted(
            view.ready,
            key=lambda p: (self.priority(p, view, target, mean_ctx), p.waiting_age, -p.admit_blocks, -p.pid),
            reverse=True,
        )
        safety = int(math.ceil(self.safety_fraction * view.capacity_blocks))
        cap = max(0, view.capacity_blocks - safety)
        used = view.active_reserved
        slots = max(0, target - len(view.active))
        admit: List[int] = []
        for p in ranked:
            if slots <= 0:
                break
            if used + p.admit_blocks <= cap:
                admit.append(p.pid)
                used += p.admit_blocks
                slots -= 1
        if not admit and not view.active and ranked:
            for p in ranked:
                if p.admit_blocks <= cap:
                    admit = [p.pid]
                    used += p.admit_blocks
                    break
        admitted = set(admit)
        keep = self.select_caches([p for p in view.caches if p.pid not in admitted], max(0, cap - used), view)
        return Plan(admit, keep)

    def plan(self, view: SystemView) -> Plan:
        return self._plan_with_target(view, view.service.max_batch)


class HazardBatchKnapsackPolicy(HazardGradeKnapsackPolicy):
    """Experimental full variant with a service-curve batch-knee cap.

    This is retained as an ablation.  Validation indicated that the cap does not
    improve mean completion time before the service curve is calibrated from the
    real GPU, so it is not the default deployment candidate.
    """

    name = "hazard_batch_knapsack"

    def __init__(self, marginal_ratio: float = 0.12, **kwargs):
        kwargs.setdefault("age_bonus", 0.35)
        kwargs.setdefault("safety_fraction", 0.02)
        super().__init__(**kwargs)
        self.marginal_ratio = marginal_ratio
        self.last_target = 0

    def reset(self, cfg: WorkloadConfig, service: ServiceModel) -> None:
        super().reset(cfg, service)
        self.last_target = 0

    def plan(self, view: SystemView) -> Plan:
        contexts = [p.current_context for p in view.active] + [p.prefix + p.prompt for p in view.ready]
        mean_ctx = float(np.mean(contexts)) if contexts else 2_000.0
        target = view.service.batch_knee(self.marginal_ratio, mean_ctx)
        self.last_target = target
        return self._plan_with_target(view, target)

    def diagnostics(self) -> Dict[str, float]:
        x = super().diagnostics()
        x["last_batch_target"] = float(self.last_target)
        return x


POLICIES = {
    "thunder": ThunderPolicy,
    "thunder_greedy": ThunderGreedyPolicy,
    "dual_price_proxy": DualPriceProxyPolicy,
    "student_bdp": StudentBDPPolicy,
    "hazard_knapsack": HazardKnapsackPolicy,
    "hazard_grade_knapsack": HazardGradeKnapsackPolicy,
    "hazard_batch_knapsack": HazardBatchKnapsackPolicy,
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    control_interval: float = 5.0
    dt: float = 0.5
    max_time: float = 20_000.0
    alpha_work: float = 0.25


class Simulator:
    def __init__(self, cfg: WorkloadConfig, specs: Sequence[ProgramSpec], policy: Policy,
                 service: Optional[ServiceModel] = None, sim_cfg: Optional[SimConfig] = None):
        self.cfg = cfg
        self.service = service or ServiceModel()
        self.sim_cfg = sim_cfg or SimConfig()
        self.policy = policy
        self.policy.reset(cfg, self.service)
        self.block_size = cfg.block_size
        self.capacity_blocks = cfg.capacity_tokens // cfg.block_size
        shape = 1.0 / (cfg.decode_cv * cfg.decode_cv)
        scale = cfg.decode_mean / shape
        self.decode_reserve = int(math.ceil(gamma_dist.ppf(cfg.reserve_quantile, a=shape, scale=scale)))
        self.p: List[RuntimeProgram] = []
        for pid, spec in enumerate(specs):
            self.p.append(RuntimeProgram(pid=pid, spec=spec, prompt=spec.requests[0].prompt_tokens))
        self.now = 0.0
        self.next_tick = 0.0

        # metrics
        self.program_ct: List[float] = []
        self.request_ct: List[float] = []
        self.queue_times: List[float] = []
        self.warm_admits = self.cold_admits = 0
        self.hit_prefix = self.requested_prefix = 0.0
        self.prefill_tokens = self.recompute_tokens = self.decode_tokens = 0.0
        self.evictions = self.emergency_evictions = 0
        self.evicted_tokens = 0.0
        self.reserve_expansions = self.stall_s = self.memory_violations = 0
        self.physical_area = self.reserved_area = self.cache_area = 0.0
        self.active_area = self.decode_area = self.prefill_area = self.busy_s = 0.0
        self.decision_calls = 0
        self.decision_time = 0.0
        self.decision_samples: List[float] = []

    def blocks(self, tokens: float) -> int:
        return 0 if tokens <= 1e-9 else int(math.ceil(tokens / self.block_size))

    def admit_blocks(self, p: RuntimeProgram) -> int:
        return self.blocks(p.prefix + p.prompt + self.decode_reserve)

    def cache_blocks(self, p: RuntimeProgram) -> int:
        return self.blocks(p.prefix) if p.cache_warm and p.phase in (TOOL, READY) else 0

    def active(self) -> List[RuntimeProgram]:
        return [p for p in self.p if p.phase in (PREFILL, DECODE)]

    def ready(self) -> List[RuntimeProgram]:
        return [p for p in self.p if p.phase == READY]

    def planning_used(self) -> int:
        return sum(p.reserve_blocks for p in self.active()) + sum(self.cache_blocks(p) for p in self.p)

    def physical_used(self) -> int:
        return sum(self.blocks(p.active_kv) for p in self.active()) + sum(self.cache_blocks(p) for p in self.p)

    def view(self) -> SystemView:
        views: List[ProgramView] = []
        for p in self.p:
            if p.phase == DONE:
                continue
            tool_type = p.spec.requests[p.idx - 1].tool_type_after if p.phase == TOOL and p.idx > 0 else None
            views.append(ProgramView(
                pid=p.pid,
                phase=p.phase,
                stage=p.stage,
                prefix=p.prefix,
                prompt=p.prompt if p.phase in (READY, PREFILL, DECODE) else 0,
                cache_warm=p.cache_warm,
                tool_type=tool_type,
                tool_elapsed=p.tool_elapsed,
                waiting_age=max(0.0, self.now - p.ready_since) if p.phase == READY else 0.0,
                terminal_prob=p.spec.prior.terminal_probability(p.stage),
                remaining_rounds=p.spec.prior.remaining_mean(p.stage),
                initial=p.initial,
                admit_blocks=self.admit_blocks(p) if p.phase == READY else 0,
                cache_blocks=self.cache_blocks(p),
                reserve_blocks=p.reserve_blocks,
                current_context=p.active_kv if p.phase in (PREFILL, DECODE) else float(p.prefix),
            ))
        return SystemView(
            now=self.now,
            block_size=self.block_size,
            capacity_blocks=self.capacity_blocks,
            decode_mean=self.cfg.decode_mean,
            prompt_mean=self.cfg.prompt_mean,
            decode_reserve=self.decode_reserve,
            alpha_work=self.sim_cfg.alpha_work,
            control_interval=self.sim_cfg.control_interval,
            service=self.service,
            programs=tuple(views),
        )

    def evict(self, p: RuntimeProgram, emergency: bool = False) -> None:
        if not p.cache_warm or p.phase not in (TOOL, READY):
            return
        p.cache_warm = False
        self.evictions += 1
        self.evicted_tokens += p.prefix
        if emergency:
            self.emergency_evictions += 1

    def free_blocks(self, need: int, emergency: bool) -> int:
        if need <= 0:
            return 0
        order = self.policy.emergency_order(self.view())
        by = {p.pid: p for p in self.p}
        freed = 0
        for pid in order:
            p = by[pid]
            w = self.cache_blocks(p)
            if w > 0:
                self.evict(p, emergency)
                freed += w
                if freed >= need:
                    break
        return freed

    def admit(self, p: RuntimeProgram) -> bool:
        reserve = self.admit_blocks(p)
        old = self.cache_blocks(p)
        if self.planning_used() - old + reserve > self.capacity_blocks:
            return False
        warm = p.cache_warm and p.prefix > 0
        self.requested_prefix += p.prefix
        if warm:
            self.warm_admits += 1
            self.hit_prefix += p.prefix
            p.active_kv = float(p.prefix)
        else:
            self.cold_admits += 1
            self.recompute_tokens += p.prefix
            p.active_kv = 0.0
        uncached = p.prompt + (0 if warm else p.prefix)
        p.prefill_remaining = float(uncached)
        p.generated = 0.0
        p.reserve_blocks = reserve
        p.cache_warm = False
        p.phase = PREFILL if uncached > 1e-9 else DECODE
        p.admitted_at = self.now
        return True

    def apply_plan(self, plan: Plan) -> None:
        by = {p.pid: p for p in self.p}
        admit_order = list(dict.fromkeys(plan.admit))
        admitted = set(admit_order)
        keep = set(plan.keep_cache) - admitted
        for p in self.p:
            if self.cache_blocks(p) > 0 and p.pid not in keep and p.pid not in admitted:
                self.evict(p)
        for pid in admit_order:
            p = by.get(pid)
            if p and p.phase == READY:
                self.admit(p)
        if self.planning_used() > self.capacity_blocks:
            self.free_blocks(self.planning_used() - self.capacity_blocks, emergency=False)
        self.check_memory()

    def run_scheduler(self) -> None:
        t0 = time.perf_counter()
        plan = self.policy.plan(self.view())
        dt = time.perf_counter() - t0
        self.decision_calls += 1
        self.decision_time += dt
        self.decision_samples.append(dt)
        self.apply_plan(plan)

    def check_memory(self) -> None:
        if self.planning_used() > self.capacity_blocks or self.physical_used() > self.capacity_blocks:
            self.memory_violations += 1

    def ensure_reserve(self, p: RuntimeProgram, target: int) -> bool:
        if target <= p.reserve_blocks:
            return True
        extra = target - p.reserve_blocks
        free = self.capacity_blocks - self.planning_used()
        if free < extra:
            self.free_blocks(extra - free, emergency=True)
            free = self.capacity_blocks - self.planning_used()
        if free >= extra:
            p.reserve_blocks = target
            self.reserve_expansions += 1
            return True
        return False

    def allowed_progress(self, p: RuntimeProgram, proposed: float) -> float:
        target = self.blocks(p.active_kv + proposed)
        if target <= p.reserve_blocks or self.ensure_reserve(p, target):
            return proposed
        boundary = p.reserve_blocks * self.block_size
        return max(0.0, min(proposed, boundary - p.active_kv - 1e-9))

    def integrals(self, dt: float) -> None:
        active = self.active()
        n_p = sum(p.phase == PREFILL for p in active)
        n_d = sum(p.phase == DECODE for p in active)
        self.physical_area += self.physical_used() * dt
        self.reserved_area += self.planning_used() * dt
        self.cache_area += sum(self.cache_blocks(p) for p in self.p) * dt
        self.active_area += len(active) * dt
        self.prefill_area += n_p * dt
        self.decode_area += n_d * dt
        if active:
            self.busy_s += dt

    def advance_tools(self, dt: float) -> None:
        for p in self.p:
            if p.phase != TOOL:
                continue
            p.tool_elapsed += dt
            p.tool_remaining -= dt
            if p.tool_remaining <= 1e-9:
                p.tool_remaining = 0.0
                p.phase = READY
                p.prompt = p.request.prompt_tokens
                p.ready_since = self.now + dt

    def complete_turn(self, p: RuntimeProgram, at: float) -> None:
        req = p.request
        self.request_ct.append(at)
        self.queue_times.append(max(0.0, p.admitted_at - p.ready_since))
        p.prefix += p.prompt + req.decode_tokens_actual
        p.idx += 1
        p.reserve_blocks = 0
        p.active_kv = float(p.prefix)
        p.prefill_remaining = p.generated = 0.0
        if p.idx >= len(p.spec.requests):
            p.phase = DONE
            p.cache_warm = False
            p.active_kv = 0.0
            p.completion_time = at
            self.program_ct.append(at)
        else:
            p.phase = TOOL
            p.cache_warm = True
            p.prompt = 0
            p.tool_elapsed = 0.0
            p.tool_remaining = req.tool_duration_actual

    def advance_gpu(self, dt: float) -> None:
        prefills = [p for p in self.p if p.phase == PREFILL]
        decodes = [p for p in self.p if p.phase == DECODE]
        if not prefills and not decodes:
            return
        if prefills and decodes:
            pf_share, dec_share = self.service.mixed_prefill_share, 1.0 - self.service.mixed_prefill_share
        elif prefills:
            pf_share, dec_share = 1.0, 0.0
        else:
            pf_share, dec_share = 0.0, 1.0

        if prefills:
            rate = pf_share * self.service.aggregate_prefill_tps(len(prefills), float(np.mean([p.prefill_remaining for p in prefills]))) / len(prefills)
            for p in list(prefills):
                proposed = min(p.prefill_remaining, rate * dt)
                allowed = self.allowed_progress(p, proposed)
                p.prefill_remaining -= allowed
                p.active_kv += allowed
                self.prefill_tokens += allowed
                if p.prefill_remaining <= 1e-7:
                    p.prefill_remaining = 0.0
                    p.active_kv = float(p.prefix + p.prompt)
                    p.phase = DECODE

        decodes = [p for p in self.p if p.phase == DECODE]
        if decodes and dec_share > 0:
            mean_ctx = float(np.mean([p.active_kv for p in decodes]))
            rate = dec_share * self.service.aggregate_decode_tps(len(decodes), mean_ctx) / len(decodes)
            finished: List[RuntimeProgram] = []
            for p in list(decodes):
                remaining = max(0.0, p.request.decode_tokens_actual - p.generated)
                proposed = min(remaining, rate * dt)
                allowed = self.allowed_progress(p, proposed)
                if allowed + 1e-10 < proposed:
                    self.stall_s += dt
                p.generated += allowed
                p.active_kv += allowed
                self.decode_tokens += allowed
                if p.generated + 1e-7 >= p.request.decode_tokens_actual:
                    p.generated = float(p.request.decode_tokens_actual)
                    p.active_kv = float(p.prefix + p.prompt + p.request.decode_tokens_actual)
                    finished.append(p)
            for p in finished:
                self.complete_turn(p, self.now + dt)

    def run(self) -> Dict[str, float]:
        loops = 0
        while len(self.program_ct) < len(self.p):
            loops += 1
            if loops > 2_000_000 or self.now > self.sim_cfg.max_time:
                counts = {x: sum(p.phase == x for p in self.p) for x in (READY, PREFILL, DECODE, TOOL, DONE)}
                raise RuntimeError(f"simulation stalled at t={self.now:.2f}; phases={counts}; used={self.planning_used()}/{self.capacity_blocks}")
            ready = bool(self.ready())
            active = bool(self.active())
            if self.now + 1e-9 >= self.next_tick or (ready and not active):
                self.run_scheduler()
                self.next_tick = self.now + self.sim_cfg.control_interval

            dt = self.sim_cfg.dt
            if self.next_tick > self.now + 1e-9:
                dt = min(dt, self.next_tick - self.now)
            remaining_tools = [p.tool_remaining for p in self.p if p.phase == TOOL and p.tool_remaining > 1e-9]
            if remaining_tools:
                dt = min(dt, min(remaining_tools))
            dt = max(1e-6, dt)
            self.integrals(dt)
            self.advance_tools(dt)
            self.advance_gpu(dt)
            self.now += dt
            self.check_memory()

        C = np.array(self.program_ct)
        R = np.array(self.request_ct)
        Q = np.array(self.queue_times)
        T = float(C.max())
        out: Dict[str, float] = {
            "policy": self.policy.name,
            "mean_program_ct": float(C.mean()),
            "p50_program_ct": float(np.percentile(C, 50)),
            "p90_program_ct": float(np.percentile(C, 90)),
            "p95_program_ct": float(np.percentile(C, 95)),
            "makespan": T,
            "programs_per_hour": len(C) / T * 3600.0,
            "steps_per_min": len(R) / T * 60.0,
            "completed_programs": float(len(C)),
            "completed_requests": float(len(R)),
            "mean_request_ct": float(R.mean()),
            "mean_queue_s": float(Q.mean()),
            "p95_queue_s": float(np.percentile(Q, 95)),
            "cache_hit_prefix_ratio": self.hit_prefix / max(1.0, self.requested_prefix),
            "warm_admits": float(self.warm_admits),
            "cold_admits": float(self.cold_admits),
            "prefill_tokens": self.prefill_tokens,
            "cold_recompute_tokens": self.recompute_tokens,
            "decode_tokens": self.decode_tokens,
            "cache_evictions": float(self.evictions),
            "evicted_prefix_tokens": self.evicted_tokens,
            "emergency_evictions": float(self.emergency_evictions),
            "reserve_expansions": float(self.reserve_expansions),
            "decode_stall_s": self.stall_s,
            "memory_violations": float(self.memory_violations),
            "physical_kv_util": self.physical_area / max(1e-9, self.capacity_blocks * T),
            "reserved_kv_util": self.reserved_area / max(1e-9, self.capacity_blocks * T),
            "idle_cache_fraction": self.cache_area / max(1e-9, self.capacity_blocks * T),
            "mean_active": self.active_area / T,
            "mean_decode_batch": self.decode_area / T,
            "mean_prefill_batch": self.prefill_area / T,
            "gpu_busy_fraction": self.busy_s / T,
            "decision_calls": float(self.decision_calls),
            "mean_decision_ms": 1000.0 * self.decision_time / max(1, self.decision_calls),
            "p95_decision_ms": 1000.0 * float(np.percentile(self.decision_samples, 95)),
            "decode_reserve": float(self.decode_reserve),
        }
        out.update(self.policy.diagnostics())
        return out


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


DEFAULT_POLICIES = ["thunder", "student_bdp", "hazard_knapsack", "hazard_grade_knapsack", "hazard_batch_knapsack"]


def matrix(preset: str, seed0: int) -> List[WorkloadConfig]:
    rows: List[WorkloadConfig] = []
    if preset == "smoke":
        return [WorkloadConfig("smoke", n_programs=24, capacity_tokens=20_000, seed=seed0)]
    if preset == "real_proxy_smoke":
        return [WorkloadConfig(
            "swebench_real_proxy_smoke",
            n_programs=80,
            initial_prompt_mean=14_000.0,
            initial_prompt_cv=0.35,
            prompt_mean=1_800.0,
            prompt_cv=0.75,
            decode_mean=1_650.0,
            decode_cv=0.80,
            prior_name="swebench9",
            tool_mix="swebench_proxy",
            capacity_tokens=480_000,
            seed=seed0,
        )]
    if preset == "real_proxy":
        for cap in (360_000, 480_000, 600_000):
            for r in range(3):
                rows.append(WorkloadConfig(
                    f"swebench_real_proxy_cap{cap}_r{r}",
                    n_programs=80,
                    initial_prompt_mean=14_000.0,
                    initial_prompt_cv=0.35,
                    prompt_mean=1_800.0,
                    prompt_cv=0.75,
                    decode_mean=1_650.0,
                    decode_cv=0.80,
                    prior_name="swebench9",
                    tool_mix="swebench_proxy",
                    capacity_tokens=cap,
                    seed=seed0 + cap + r,
                ))
        return rows
    if preset == "validation":
        for mix in ("swe_mixed", "heavy_tail"):
            for cap in (40_000, 56_000):
                for r in range(2):
                    rows.append(WorkloadConfig(f"{mix}_cap{cap}_v{r}", n_programs=48, tool_mix=mix,
                                               capacity_tokens=cap, seed=seed0 + cap + r + 10_000 * (mix == "heavy_tail")))
        return rows
    if preset == "test":
        for mix_i, mix in enumerate(("swe_regular", "swe_mixed", "heavy_tail")):
            for cap in (40_000, 56_000, 72_000):
                for r in range(5):
                    rows.append(WorkloadConfig(f"{mix}_cap{cap}_r{r}", n_programs=64, tool_mix=mix,
                                               capacity_tokens=cap, seed=seed0 + cap + r + 100_000 * mix_i))
        return rows
    raise ValueError(preset)


def bootstrap_ci(x: np.ndarray, reps: int = 5000, seed: int = 7) -> Tuple[float, float]:
    if len(x) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(raw: pd.DataFrame, out_dir: str, thunder_policy: str = "thunder") -> None:
    os.makedirs(out_dir, exist_ok=True)
    raw.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
    keys = ["scenario", "seed", "tool_mix", "capacity_tokens"]
    base_cols = ["mean_program_ct", "p90_program_ct", "makespan", "steps_per_min", "cold_recompute_tokens"]
    base = raw[raw.policy == thunder_policy][keys + base_cols].rename(columns={x: x + "_thunder" for x in base_cols})
    paired = raw.merge(base, on=keys, how="left")
    for x in ("mean_program_ct", "p90_program_ct", "makespan", "cold_recompute_tokens"):
        paired[x + "_impr_vs_thunder_pct"] = 100.0 * (paired[x + "_thunder"] - paired[x]) / paired[x + "_thunder"]
    paired["steps_per_min_impr_vs_thunder_pct"] = 100.0 * (paired.steps_per_min / paired.steps_per_min_thunder - 1.0)
    paired.to_csv(os.path.join(out_dir, "paired_vs_thunder.csv"), index=False)

    # A second paired table is more informative for algorithm development: the
    # student's BDP is the immediate predecessor, while Thunder is the systems
    # baseline.  Use NaN when a recomputation denominator is zero.
    bdp = raw[raw.policy == "student_bdp"][keys + base_cols + ["cache_hit_prefix_ratio", "mean_decision_ms"]].rename(
        columns={x: x + "_bdp" for x in base_cols + ["cache_hit_prefix_ratio", "mean_decision_ms"]}
    )
    paired_bdp = raw.merge(bdp, on=keys, how="left")
    for x in ("mean_program_ct", "p90_program_ct", "makespan"):
        paired_bdp[x + "_impr_vs_bdp_pct"] = 100.0 * (paired_bdp[x + "_bdp"] - paired_bdp[x]) / paired_bdp[x + "_bdp"]
    paired_bdp["steps_per_min_impr_vs_bdp_pct"] = 100.0 * (paired_bdp.steps_per_min / paired_bdp.steps_per_min_bdp - 1.0)
    den = paired_bdp.cold_recompute_tokens_bdp
    paired_bdp["cold_recompute_tokens_impr_vs_bdp_pct"] = np.where(
        den > 0, 100.0 * (den - paired_bdp.cold_recompute_tokens) / den, np.nan
    )
    paired_bdp["cache_hit_delta_vs_bdp_pp"] = 100.0 * (
        paired_bdp.cache_hit_prefix_ratio - paired_bdp.cache_hit_prefix_ratio_bdp
    )
    paired_bdp.to_csv(os.path.join(out_dir, "paired_vs_bdp.csv"), index=False)

    bdp_summary = paired_bdp.groupby("policy", as_index=False).agg(
        mean_ct_impr_vs_bdp=("mean_program_ct_impr_vs_bdp_pct", "mean"),
        p90_impr_vs_bdp=("p90_program_ct_impr_vs_bdp_pct", "mean"),
        makespan_impr_vs_bdp=("makespan_impr_vs_bdp_pct", "mean"),
        steps_impr_vs_bdp=("steps_per_min_impr_vs_bdp_pct", "mean"),
        recompute_impr_vs_bdp=("cold_recompute_tokens_impr_vs_bdp_pct", "mean"),
        cache_hit_delta_vs_bdp_pp=("cache_hit_delta_vs_bdp_pp", "mean"),
    )
    ci_bdp = []
    for policy in sorted(paired_bdp.policy.unique()):
        vals = paired_bdp.loc[paired_bdp.policy == policy, "mean_program_ct_impr_vs_bdp_pct"].dropna().to_numpy()
        lo, hi = bootstrap_ci(vals)
        ci_bdp.append({"policy": policy, "mean_ct_vs_bdp_ci_low": lo, "mean_ct_vs_bdp_ci_high": hi})
    bdp_summary = bdp_summary.merge(pd.DataFrame(ci_bdp), on="policy").sort_values("mean_ct_impr_vs_bdp", ascending=False)
    bdp_summary.to_csv(os.path.join(out_dir, "summary_vs_bdp.csv"), index=False)

    summary = paired.groupby("policy", as_index=False).agg(
        mean_program_ct=("mean_program_ct", "mean"),
        p90_program_ct=("p90_program_ct", "mean"),
        makespan=("makespan", "mean"),
        steps_per_min=("steps_per_min", "mean"),
        mean_ct_impr_vs_thunder=("mean_program_ct_impr_vs_thunder_pct", "mean"),
        p90_impr_vs_thunder=("p90_program_ct_impr_vs_thunder_pct", "mean"),
        makespan_impr_vs_thunder=("makespan_impr_vs_thunder_pct", "mean"),
        steps_impr_vs_thunder=("steps_per_min_impr_vs_thunder_pct", "mean"),
        cache_hit=("cache_hit_prefix_ratio", "mean"),
        cold_recompute=("cold_recompute_tokens", "mean"),
        evictions=("cache_evictions", "mean"),
        mean_active=("mean_active", "mean"),
        mean_decode_batch=("mean_decode_batch", "mean"),
        physical_kv_util=("physical_kv_util", "mean"),
        reserved_kv_util=("reserved_kv_util", "mean"),
        mean_decision_ms=("mean_decision_ms", "mean"),
        p95_decision_ms=("p95_decision_ms", "mean"),
        memory_violations=("memory_violations", "sum"),
    )
    ci = []
    for policy in sorted(paired.policy.unique()):
        vals = paired.loc[paired.policy == policy, "mean_program_ct_impr_vs_thunder_pct"].to_numpy()
        lo, hi = bootstrap_ci(vals)
        ci.append({"policy": policy, "mean_ct_ci_low": lo, "mean_ct_ci_high": hi})
    summary = summary.merge(pd.DataFrame(ci), on="policy").sort_values("mean_program_ct")
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    setting = paired.groupby(["tool_mix", "capacity_tokens", "policy"], as_index=False).agg(
        mean_program_ct=("mean_program_ct", "mean"),
        mean_ct_impr=("mean_program_ct_impr_vs_thunder_pct", "mean"),
        p90_impr=("p90_program_ct_impr_vs_thunder_pct", "mean"),
        steps_impr=("steps_per_min_impr_vs_thunder_pct", "mean"),
        cache_hit=("cache_hit_prefix_ratio", "mean"),
        cold_recompute=("cold_recompute_tokens", "mean"),
        mean_decode_batch=("mean_decode_batch", "mean"),
        mean_decision_ms=("mean_decision_ms", "mean"),
    )
    setting.to_csv(os.path.join(out_dir, "summary_by_setting.csv"), index=False)

    try:
        import matplotlib.pyplot as plt
        fig = os.path.join(out_dir, "figures")
        os.makedirs(fig, exist_ok=True)
        for mix in setting.tool_mix.unique():
            part = setting[setting.tool_mix == mix]
            plt.figure(figsize=(8, 4.8))
            for policy in DEFAULT_POLICIES:
                q = part[part.policy == policy].sort_values("capacity_tokens")
                if len(q):
                    plt.plot(q.capacity_tokens, q.mean_ct_impr, marker="o", label=policy)
            plt.axhline(0, linewidth=1)
            plt.xlabel("KV capacity (tokens)")
            plt.ylabel("Mean program CT improvement vs Thunder (%)")
            plt.title(f"Revised simulator: {mix}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(fig, f"mean_ct_{mix}.png"), dpi=170)
            plt.close()
    except Exception:
        pass


def run_experiment(preset: str, out_dir: str, policy_names: Sequence[str], seed0: int,
                   control_interval: float, dt: float, overrides: Mapping[str, Mapping[str, float]],
                   service_profile: str = "synthetic", thunder_policy: str = "thunder") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    unknown = [name for name in policy_names if name not in POLICIES]
    if unknown:
        raise ValueError(f"unknown policies: {unknown}; known={sorted(POLICIES)}")
    if thunder_policy not in policy_names:
        raise ValueError(f"thunder baseline {thunder_policy!r} must be included in --policies")
    configs = matrix(preset, seed0)
    service = make_service_model(service_profile)
    sim_cfg = SimConfig(control_interval=control_interval, dt=dt)
    rows: List[Dict] = []
    for si, cfg in enumerate(configs):
        specs, meta = generate_workload(cfg)
        for name in policy_names:
            policy = POLICIES[name](**dict(overrides.get(name, {})))
            sim = Simulator(cfg, specs, policy, service=service, sim_cfg=sim_cfg)
            t0 = time.perf_counter()
            m = sim.run()
            runtime = time.perf_counter() - t0
            row = dict(m)
            row.update(
                scenario=cfg.name,
                seed=cfg.seed,
                tool_mix=cfg.tool_mix,
                capacity_tokens=cfg.capacity_tokens,
                reserve_quantile=cfg.reserve_quantile,
                n_programs=cfg.n_programs,
                control_interval=control_interval,
                runtime_s=runtime,
                total_requests=meta["total_requests"],
                mean_rounds=meta["mean_rounds"],
                mean_initial_prompt=meta["mean_initial_prompt"],
                mean_prompt_per_turn=meta["mean_prompt_per_turn"],
                mean_decode_per_turn=meta["mean_decode_per_turn"],
                mean_total_prompt_per_program=meta["mean_total_prompt_per_program"],
                mean_total_decode_per_program=meta["mean_total_decode_per_program"],
                mean_tool_s_per_program=meta["mean_tool_s_per_program"],
            )
            rows.append(row)
            # Long experiment matrices are restart-sensitive on shared systems;
            # persist every completed policy/scenario pair before continuing.
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "raw_checkpoint.csv"), index=False)
            print(f"[{si+1:02d}/{len(configs):02d}] {cfg.name:30s} {name:26s} "
                  f"mean={m['mean_program_ct']:.1f} p90={m['p90_program_ct']:.1f} "
                  f"steps/min={m['steps_per_min']:.2f} hit={m['cache_hit_prefix_ratio']:.3f} "
                  f"decision={m['mean_decision_ms']:.2f}ms", flush=True)
    raw = pd.DataFrame(rows)
    summarize(raw, out_dir, thunder_policy=thunder_policy)
    with open(os.path.join(out_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "preset": preset,
            "seed0": seed0,
            "policies": list(policy_names),
            "thunder_policy": thunder_policy,
            "service_profile": service_profile,
            "control_interval": control_interval,
            "dt": dt,
            "service_model": asdict(service),
            "sim_config": asdict(sim_cfg),
            "overrides": overrides,
        }, f, indent=2)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "validation", "test", "real_proxy_smoke", "real_proxy"], default="smoke")
    ap.add_argument("--out", default="results/smoke")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--seed0", type=int, default=20260710)
    ap.add_argument("--control-interval", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--service-profile", choices=sorted(SERVICE_PROFILE_KWARGS), default="synthetic")
    ap.add_argument("--thunder-policy", choices=["thunder", "thunder_greedy"], default="thunder")
    ap.add_argument("--overrides", default="", help="JSON policy kwargs")
    a = ap.parse_args()
    policies = [x.strip() for x in a.policies.split(",") if x.strip()]
    overrides = json.loads(a.overrides) if a.overrides else {}
    run_experiment(a.preset, a.out, policies, a.seed0, a.control_interval, a.dt, overrides,
                   service_profile=a.service_profile, thunder_policy=a.thunder_policy)


if __name__ == "__main__":
    main()
