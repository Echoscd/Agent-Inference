#!/usr/bin/env python3
"""Minimal calibrated simulator for Qwen3-Coder/vLLM agentic serving.

The core contains one workload model, one service profile, a strict FCFS
baseline, the paper ThunderAgent action policy, and the BDP mechanism. New algorithms use ``Policy`` or
``BDPLikePolicy`` and register in ``POLICIES``; all engine mechanisms stay shared.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
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

    @staticmethod
    def from_counts(name: str, counts: Mapping[int, int]) -> "DiscretePrior":
        k_max = max(counts)
        w = np.array([float(counts.get(k, 0)) for k in range(1, k_max + 1)], dtype=float)
        w /= w.sum()
        surv = np.zeros(k_max + 2, dtype=float)
        for j in range(1, k_max + 1):
            surv[j] = float(w[j - 1 :].sum())
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
    # Pooled 26/27/28 public per-agent traces (six arms, n=480). The mass at 20
    # is censored by max_turns rather than treated as a natural terminal mode.
    "coder16_empirical": DiscretePrior.from_counts("coder16_empirical", {
        3: 10, 4: 5, 5: 13, 6: 16, 7: 17, 8: 25, 9: 15,
        10: 16, 11: 20, 12: 19, 13: 21, 14: 23, 15: 18,
        16: 20, 17: 23, 18: 18, 19: 17, 20: 184,
    }),
    # Natural termination proxy before the 40,960-token harness guard is
    # applied: 70% run to max_turns, 30% submit/exit uniformly at turns 3--19.
    "coder20_natural": DiscretePrior.from_counts("coder20_natural", {
        **{k: 30 for k in range(3, 20)}, 20: 1_190,
    }),
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
    # The public summaries report roughly 11--18 tool seconds per program.
    "coder_fast": ToolModel("coder_fast", mean_s=0.30, cv=1.00),
    "coder_medium": ToolModel("coder_medium", mean_s=1.20, cv=1.00),
    "coder_long": ToolModel("coder_long", mean_s=5.00, cv=1.20),
}

TOOL_MIXES: Dict[str, Dict[str, float]] = {
    "coder_tools": {"coder_fast": 0.60, "coder_medium": 0.32, "coder_long": 0.08},
}


def _gamma_logpdf_scalar(value: float, shape: float, scale: float) -> float:
    """Allocation-free scalar Gamma log-density for the scheduling hot path."""
    value = max(1e-12, value)
    scale = max(1e-12, scale)
    return (
        (shape - 1.0) * math.log(value)
        - value / scale
        - math.lgamma(shape)
        - shape * math.log(scale)
    )


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
    prompt_large_prob: float = 0.0
    prompt_large_mean: Optional[float] = None
    prompt_large_cv: Optional[float] = None
    # Agent traces have a large repository/task prompt followed by smaller tool
    # deltas.  None preserves the original identical-per-round distribution.
    initial_prompt_mean: Optional[float] = None
    initial_prompt_cv: Optional[float] = None
    decode_mean: float = 150.0
    decode_cv: float = 0.80
    decode_large_prob: float = 0.0
    decode_large_mean: Optional[float] = None
    decode_large_cv: Optional[float] = None
    # A single program-level scale correlates decode lengths across turns. Zero
    # preserves IID turns; the Coder traces require this to reproduce their
    # task-total CV rather than only the pooled per-call marginal.
    decode_program_cv: float = 0.0
    prior_name: str = "short"
    policy_prior_name: Optional[str] = None
    policy_prompt_mean: Optional[float] = None
    policy_decode_mean: Optional[float] = None
    policy_alpha_work: Optional[float] = None
    tool_mix: str = "coder_tools"
    capacity_tokens: int = 56_000
    block_size: int = 16
    reserve_quantile: float = 0.95
    decode_reserve_tokens: Optional[int] = None
    context_limit_tokens: Optional[int] = None
    seed: int = 0


def _sample_gamma_int(rng: np.random.Generator, mean: float, cv: float) -> int:
    shape = 1.0 / max(cv * cv, 1e-12)
    scale = mean / shape
    return max(1, int(round(rng.gamma(shape, scale))))


def generate_workload(cfg: WorkloadConfig) -> Tuple[List[ProgramSpec], Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    # Keep the program-level latent independent from the established per-turn
    # stream so enabling correlation does not silently change prompt/tool draws.
    program_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    prior = PRIORS[cfg.prior_name]
    policy_prior = PRIORS[cfg.policy_prior_name] if cfg.policy_prior_name else prior
    mix = TOOL_MIXES[cfg.tool_mix]
    tool_names = list(mix)
    probs = np.array([mix[x] for x in tool_names], dtype=float)
    probs /= probs.sum()

    programs: List[ProgramSpec] = []
    total_requests = 0
    total_prompt = 0
    total_submitted_prompt = 0
    total_decode = 0
    total_initial_prompt = 0
    total_tool = 0.0
    max_context = 0
    program_decode_totals: List[float] = []
    if cfg.decode_program_cv > 0.0:
        sigma = math.sqrt(math.log1p(cfg.decode_program_cv * cfg.decode_program_cv))
        decode_scales = program_rng.lognormal(
            -0.5 * sigma * sigma, sigma, size=cfg.n_programs
        )
    else:
        decode_scales = np.ones(cfg.n_programs, dtype=float)
    for program_i in range(cfg.n_programs):
        k = prior.sample(rng)
        reqs: List[RequestSpec] = []
        context = 0
        decode_scale = float(decode_scales[program_i])
        for j in range(k):
            if j == 0 and cfg.initial_prompt_mean is not None:
                pmean = cfg.initial_prompt_mean
                pcv = cfg.initial_prompt_cv if cfg.initial_prompt_cv is not None else cfg.prompt_cv
            else:
                pmean, pcv = cfg.prompt_mean, cfg.prompt_cv
                if cfg.prompt_large_mean is not None and rng.random() < cfg.prompt_large_prob:
                    pmean = cfg.prompt_large_mean
                    pcv = cfg.prompt_large_cv if cfg.prompt_large_cv is not None else cfg.prompt_cv
            p = _sample_gamma_int(rng, pmean, pcv)
            if j > 0 and cfg.context_limit_tokens is not None and context + p > cfg.context_limit_tokens:
                break
            dmean, dcv = cfg.decode_mean, cfg.decode_cv
            if cfg.decode_large_mean is not None and rng.random() < cfg.decode_large_prob:
                dmean = cfg.decode_large_mean
                dcv = cfg.decode_large_cv if cfg.decode_large_cv is not None else cfg.decode_cv
            d = _sample_gamma_int(rng, dmean * decode_scale, dcv)
            # The server's prompt_tokens counter is the submitted input length
            # on every call, so it repeatedly counts the accumulated context.
            total_submitted_prompt += context + p
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
        # A context-limit failure occurs on the next model call, after the last
        # successful turn. Do not invent a subsequent tool interval.
        if reqs and reqs[-1].tool_type_after is not None:
            total_tool -= reqs[-1].tool_duration_actual
            last = reqs[-1]
            reqs[-1] = RequestSpec(last.prompt_tokens, last.decode_tokens_actual, None, 0.0)
        total_requests += len(reqs)
        max_context = max(max_context, context)
        program_decode_totals.append(float(sum(x.decode_tokens_actual for x in reqs)))
        programs.append(ProgramSpec(tuple(reqs), policy_prior))
    decode_totals = np.asarray(program_decode_totals, dtype=float)
    return programs, {
        "total_requests": float(total_requests),
        "mean_rounds": total_requests / cfg.n_programs,
        "mean_initial_prompt": total_initial_prompt / cfg.n_programs,
        "mean_prompt_per_turn": total_prompt / max(1, total_requests),
        "mean_decode_per_turn": total_decode / max(1, total_requests),
        "mean_total_prompt_per_program": total_prompt / cfg.n_programs,
        "mean_submitted_prompt_per_program": total_submitted_prompt / cfg.n_programs,
        "mean_total_decode_per_program": total_decode / cfg.n_programs,
        "cv_total_decode_per_program": float(decode_totals.std() / max(1e-9, decode_totals.mean())),
        "mean_tool_s_per_program": total_tool / cfg.n_programs,
        "max_final_context": float(max_context),
    }


# ---------------------------------------------------------------------------
# GPU service model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceModel:
    # Concave aggregate service shapes; the shipped profile is calibrated below.
    prefill_single_tps: float = 7_000.0
    prefill_sat: float = 4.0
    decode_single_tps: float = 45.0
    decode_sat: float = 16.0
    context_ref: float = 4_096.0
    prefill_context_penalty: float = 0.12
    decode_context_penalty: float = 0.30
    mixed_prefill_share: float = 0.30
    # When set, these replace the fallback zero-sum share. Prefill and decode have
    # different token/s units, so measured profiles use independent interference
    # multipliers instead of pretending they share one token bandwidth.
    mixed_prefill_multiplier: Optional[float] = None
    mixed_decode_multiplier: Optional[float] = None
    max_batch: int = 64
    prefill_chunk_tokens: Optional[int] = None

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

    def cold_prefill_seconds(self, tokens: float) -> float:
        return max(0.0, tokens) / max(1e-9, self.aggregate_prefill_tps(1, tokens))


SERVICE_PROFILE_KWARGS: Dict[str, Dict[str, float]] = {
    # One measured profile for all public repetitions.  The prefill curve is
    # anchored by 0.12 ms/token near 5--10k and 0.16 ms/token at long context;
    # the decode curve is jointly checked against the 420--495 whole-run tok/s
    # operating point after workload ramp and drain are simulated.
    "qwen3_coder_30b_vllm_012": {
        "prefill_single_tps": 8_333.0,
        # The public cold engine queues drain in about 4.6 s. Together with the
        # independent single-request slopes, that identifies the batch-prefill
        # saturation without adding an engine-admission-rate parameter.
        "prefill_sat": 8.0,
        "decode_single_tps": 85.0,
        "decode_sat": 11.0,
        "context_ref": 40_960.0,
        "prefill_context_penalty": 0.33,
        "decode_context_penalty": 0.07,
        "mixed_prefill_share": 0.15,
        "mixed_prefill_multiplier": 1.0,
        "mixed_decode_multiplier": 0.85,
        "max_batch": 80,
        # vLLM 0.12 OpenAI server default on >=70 GiB non-A100 GPUs.
        "prefill_chunk_tokens": 8_192,
    },
}


def make_service_model(profile: str) -> ServiceModel:
    if profile not in SERVICE_PROFILE_KWARGS:
        raise ValueError(f"unknown service profile {profile!r}; known={sorted(SERVICE_PROFILE_KWARGS)}")
    return ServiceModel(**SERVICE_PROFILE_KWARGS[profile])


# ---------------------------------------------------------------------------
# Runtime state and policy-safe views
# ---------------------------------------------------------------------------


# READY is the router's paused pool. ENGINE_WAITING is a request already sent to
# vLLM but not currently holding scheduled KV blocks. Keeping these states
# separate is essential: the real router explicitly subtracts vLLM's physical
# occupancy from its REASONING-token estimate, and vLLM reports waiting/running
# independently.
READY, ENGINE_WAITING, PREFILL, DECODE, TOOL, DONE = (
    "READY", "ENGINE_WAITING", "PREFILL", "DECODE", "TOOL", "DONE"
)


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
    engine_queued_at: float = 0.0
    cache_accounted: bool = False
    completion_time: float = math.nan
    preempted: bool = False
    preemptions: int = 0
    marked_for_pause: bool = False

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
    observed_decode_lengths: Tuple[int, ...]
    terminal_prob: float
    remaining_rounds: float
    initial: bool
    admit_blocks: int
    cache_blocks: int
    reserve_blocks: int
    current_context: float
    preempted: bool
    marked_for_pause: bool


@dataclass(frozen=True)
class WorkEstimate:
    """Distribution-only quantities available to BDP-like policies."""

    uncached_prompt: float
    current_work: float
    remaining_work: float
    expected_footprint_tokens: float


@dataclass(frozen=True)
class SystemView:
    now: float
    block_size: int
    capacity_blocks: int
    decode_mean: float
    decode_program_cv: float
    decode_small_mean: float
    decode_small_cv: float
    decode_large_prob: float
    decode_large_mean: Optional[float]
    decode_large_cv: Optional[float]
    prompt_mean: float
    context_limit: Optional[int]
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
    def engine_waiting(self) -> List[ProgramView]:
        return [p for p in self.programs if p.phase == ENGINE_WAITING]

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

    def estimate_work(self, p: ProgramView) -> WorkEstimate:
        """Build a BDP-style estimate without realized future information."""
        warm = p.cache_warm and p.prefix > 0
        uncached = float(p.prompt + (0 if warm else p.prefix))
        completed = len(p.observed_decode_lengths)
        posterior_decode = self.decode_mean
        if completed > 0 and self.decode_program_cv > 0.0:
            # Separate the calibrated per-turn small/large mixture from the
            # persistent program scale. A rare large turn is evidence for its
            # mixture component, not automatically a 10x-heavy whole program.
            # Component responsibilities and prior precision are fully derived
            # from the existing workload model; there is no policy coefficient.
            precision = 1.0 / (self.decode_program_cv * self.decode_program_cv)
            evidence = 0.0
            scale = 1.0
            for turns_seen, decode in enumerate(p.observed_decode_lengths, start=1):
                normalized = decode / max(1e-12, self.decode_small_mean)
                if self.decode_large_prob > 0.0 and self.decode_large_mean is not None:
                    small_shape = 1.0 / max(1e-12, self.decode_small_cv ** 2)
                    large_cv = (
                        self.decode_large_cv
                        if self.decode_large_cv is not None else self.decode_small_cv
                    )
                    large_shape = 1.0 / max(1e-12, large_cv ** 2)
                    log_small = (
                        math.log(max(1e-12, 1.0 - self.decode_large_prob))
                        + _gamma_logpdf_scalar(
                            decode,
                            small_shape,
                            scale * self.decode_small_mean / small_shape,
                        )
                    )
                    log_large = (
                        math.log(max(1e-12, self.decode_large_prob))
                        + _gamma_logpdf_scalar(
                            decode,
                            large_shape,
                            scale * self.decode_large_mean / large_shape,
                        )
                    )
                    large_responsibility = 1.0 / (
                        1.0 + math.exp(float(np.clip(log_small - log_large, -700.0, 700.0)))
                    )
                    normalized = (
                        (1.0 - large_responsibility)
                        * decode / max(1e-12, self.decode_small_mean)
                        + large_responsibility
                        * decode / max(1e-12, self.decode_large_mean)
                    )
                evidence += normalized
                # Sequential likelihood classification uses the posterior mean
                # obtained from all evidence observed so far.
                scale = (precision + evidence) / (precision + turns_seen)
            posterior_decode *= scale
        current = self.alpha_work * uncached + posterior_decode
        future = self.alpha_work * self.prompt_mean + posterior_decode
        future_turns = max(0.0, p.remaining_rounds - 1.0)
        if self.context_limit is not None:
            # The harness rejects the next prompt once accumulated context
            # exceeds its guard. Condition the K posterior on this observable
            # budget instead of treating every surviving program alike.
            future_context = self.prompt_mean + posterior_decode
            context_after_current = p.prefix + p.prompt + posterior_decode
            context_limited_turns = max(
                0.0,
                (self.context_limit - context_after_current + posterior_decode)
                / max(1e-12, future_context),
            )
            future_turns = min(future_turns, context_limited_turns)
        return WorkEstimate(
            uncached_prompt=uncached,
            current_work=current,
            remaining_work=current + future_turns * future,
            expected_footprint_tokens=(
                p.prefix + p.prompt + posterior_decode
                + 0.5 * future_turns * (self.prompt_mean + posterior_decode)
            ),
        )


@dataclass
class Plan:
    admit: List[int] = field(default_factory=list)
    keep_cache: Set[int] = field(default_factory=set)
    mark_after_turn: Set[int] = field(default_factory=set)


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

    def decode_pad(self, calibrated_default: int) -> int:
        """Policy-specific output reservation used by the real router."""
        return calibrated_default

    def initial_decode_pad(self) -> int:
        """Reservation for a brand-new program; simple baselines use 0."""
        return 0

    def retain_during_tool(self) -> bool:
        """Whether an admitted program keeps its router slot while using a tool."""
        return True


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class FCFSPolicy(Policy):
    """Strict router-level first-come, first-served baseline.

    READY requests are ordered only by queue arrival. The head request blocks
    later requests when it cannot fit, and tool returns rejoin the same queue.
    Engine APC remains shared, but FCFS assigns no policy value to cached work.
    """

    name = "fcfs"

    def decode_pad(self, calibrated_default: int) -> int:
        return 0

    def retain_during_tool(self) -> bool:
        return False

    def plan(self, view: SystemView) -> Plan:
        ready = sorted(
            view.ready,
            key=lambda p: (-p.waiting_age, p.pid),
        )
        used = view.active_reserved
        admit: List[int] = []
        slots = max(0, view.service.max_batch - len(view.active))
        for p in ready:
            if slots <= 0:
                break
            if used + p.admit_blocks > view.capacity_blocks:
                break
            admit.append(p.pid)
            used += p.admit_blocks
            slots -= 1
        return Plan(admit, set())


class ThunderPolicy(Policy):
    """Single-worker projection of ThunderAgent's paper scheduling policy.

    This implements paper Section 4.3 exactly where the current simulator has
    the required state: Eq. (7) acting-token decay ``2**(-tool_age)``, Eq. (10)
    reasoning-before-new shortest-first restore, Eq. (11) acting-first
    shortest-first pause, a 100-token per-program buffer, and mark-for-pause at
    the next tool boundary.  Cross-worker placement and migration are outside
    the simulator's one-worker scope.
    """

    name = "thunder"
    PAPER_BUFFER_TOKENS = 100
    PAPER_ACTING_DECAY_BASE = 2.0

    def __init__(self):
        # Fixed paper constants, exported as effective configuration rather
        # than exposed as tunable constructor parameters.
        self.buffer_tokens = self.PAPER_BUFFER_TOKENS
        self.acting_decay_base = self.PAPER_ACTING_DECAY_BASE
        self.marked = 0
        self.decay_calls = 0
        self.decayed_cache_fraction_sum = 0.0

    def reset(self, cfg: WorkloadConfig, service: ServiceModel) -> None:
        self.marked = 0
        self.decay_calls = 0
        self.decayed_cache_fraction_sum = 0.0

    def decode_pad(self, calibrated_default: int) -> int:
        return self.buffer_tokens

    def initial_decode_pad(self) -> int:
        return self.buffer_tokens

    def acting_weight(self, p: ProgramView) -> float:
        return (
            self.acting_decay_base ** (-max(0.0, p.tool_elapsed))
            if p.phase == TOOL else 1.0
        )

    def plan(self, view: SystemView) -> Plan:
        # Eq. (10): the phase indicator dominates 1/context, so a paused
        # REASONING continuation precedes NEW; both groups are shortest-first.
        ready = sorted(
            view.ready,
            key=lambda p: (
                0 if not p.initial else 1,
                p.prefix + p.prompt,
                p.pid,
            ),
        )
        caches = list(view.caches)
        full_cache = sum(p.cache_blocks for p in caches)
        decayed_cache = sum(
            p.cache_blocks * self.acting_weight(p)
            for p in caches
        )
        self.decay_calls += 1
        self.decayed_cache_fraction_sum += decayed_cache / max(1.0, full_cache)

        # Paper restore uses decay-adjusted remaining capacity. Each admitted
        # request is still charged its full footprint plus the 100-token buffer.
        effective_used = float(view.active_reserved) + decayed_cache
        slots = max(0, view.service.max_batch - len(view.active))
        admit: List[int] = []
        for p in ready:
            if slots <= 0:
                break
            if effective_used + p.admit_blocks <= view.capacity_blocks:
                admit.append(p.pid)
                effective_used += p.admit_blocks
                slots -= 1

        admitted = set(admit)
        retained = [p for p in caches if p.pid not in admitted]
        # The paper then enforces real (non-decayed) capacity. Eq. (11) pauses
        # ACTING programs shortest-first, so the remaining cache set is the
        # largest prefixes that fit after active/admitted reservations.
        actual_used = (
            view.active_reserved
            + sum(p.admit_blocks for p in ready if p.pid in admitted)
            + sum(p.cache_blocks for p in retained)
        )
        keep = {p.pid for p in retained}
        for p in sorted(
            retained,
            key=lambda p: (0 if p.phase == TOOL else 1, p.cache_blocks, p.pid),
        ):
            if actual_used <= view.capacity_blocks:
                break
            keep.remove(p.pid)
            actual_used -= p.cache_blocks

        # If ACTING eviction is insufficient, the paper marks the smallest
        # REASONING programs and pauses them when they next enter ACTING.
        already_marked = {p.pid for p in view.active if p.marked_for_pause}
        future_release = sum(
            p.reserve_blocks for p in view.active if p.marked_for_pause
        )
        mark: Set[int] = set()
        deficit = max(
            0,
            view.active_reserved - future_release - view.capacity_blocks,
        )
        for p in sorted(
            (p for p in view.active if p.pid not in already_marked),
            key=lambda p: (p.current_context, p.pid),
        ):
            if deficit <= 0:
                break
            mark.add(p.pid)
            deficit -= p.reserve_blocks
        self.marked += len(mark)
        return Plan(admit, keep, mark)

    def emergency_order(self, view: SystemView) -> List[int]:
        # Eq. (11): ACTING first, then shortest context.
        return [
            p.pid for p in sorted(
                view.caches,
                key=lambda p: (0 if p.phase == TOOL else 1, p.cache_blocks, p.pid),
            )
        ]

    def diagnostics(self) -> Dict[str, float]:
        return {
            "thunder_marked_for_pause": float(self.marked),
            "thunder_mean_decayed_cache_fraction": (
                self.decayed_cache_fraction_sum / max(1, self.decay_calls)
            ),
        }


class BDPLikePolicy(Policy):
    """Causal estimate/score interface with shared physical feasibility."""

    def update_state(
        self,
        view: SystemView,
        estimates: Dict[int, WorkEstimate],
    ) -> None:
        """Optional hook for a dual variable or other causal policy state."""

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        """Return the causal work estimate used by this policy."""
        return view.estimate_work(p)

    def admission_score(
        self,
        p: ProgramView,
        estimate: WorkEstimate,
        view: SystemView,
    ) -> float:
        raise NotImplementedError

    def rank_ready(
        self,
        view: SystemView,
        estimates: Dict[int, WorkEstimate],
    ) -> List[ProgramView]:
        scores = {
            p.pid: self.admission_score(p, estimates[p.pid], view)
            for p in view.ready
        }
        return sorted(
            view.ready,
            key=lambda p: (scores[p.pid], -p.pid),
            reverse=True,
        )

    def fallback_ready(
        self,
        feasible: Sequence[ProgramView],
        view: SystemView,
    ) -> ProgramView:
        return feasible[0]

    def plan(self, view: SystemView) -> Plan:
        estimates = {p.pid: self.estimate(p, view) for p in view.ready}
        self.update_state(view, estimates)
        ranked = self.rank_ready(view, estimates)
        capacity = view.capacity_blocks
        used = view.active_reserved
        slots = max(0, view.service.max_batch - len(view.active))
        admit: List[int] = []
        for p in ranked:
            if slots <= 0:
                break
            if self.admission_score(p, estimates[p.pid], view) <= 0.0:
                continue
            if used + p.admit_blocks <= capacity:
                admit.append(p.pid)
                used += p.admit_blocks
                slots -= 1
        if not admit and not view.active:
            feasible = [p for p in ranked if p.admit_blocks <= capacity]
            if feasible:
                fallback = self.fallback_ready(feasible, view)
                admit = [fallback.pid]
                used += fallback.admit_blocks
        return Plan(admit, set())


class BDPPolicy(BDPLikePolicy):
    """Bayesian SERPT policy with a capacity-clearing KV dual price.

    The value of a program is the inverse posterior expected remaining work,
    ``1 / mu``. At every control tick, lambda is the marginal value density that
    clears currently free KV in the fractional relaxation. This is a simple
    Bayesian shortest-expected-remaining-processing-time objective plus one
    causal shadow price. There are no policy parameters.
    """

    name = "bdp"

    def __init__(self):
        self.lam = 0.0
        self.epoch = 0
        self.price_sum = 0.0
        self.price_max = 0.0
        self.binding_updates = 0

    def reset(self, cfg: WorkloadConfig, service: ServiceModel) -> None:
        self.lam = 0.0
        self.epoch = 0
        self.price_sum = 0.0
        self.price_max = 0.0
        self.binding_updates = 0

    def retain_during_tool(self) -> bool:
        # Reuse is provided by the engine APC free list. Router-level retention
        # did not improve reuse or completion time in paired ablations.
        return False

    def update_state(
        self,
        view: SystemView,
        estimates: Dict[int, WorkEstimate],
    ) -> None:
        free_tokens = max(
            0.0,
            (view.capacity_blocks - view.active_reserved) * view.block_size,
        )
        if not estimates:
            self.lam = 0.0
            self.epoch += 1
            self.price_sum += self.lam
            return

        densities: List[Tuple[float, float]] = []
        for estimate in estimates.values():
            value = self.bayesian_value(estimate)
            footprint = max(1.0, estimate.expected_footprint_tokens)
            densities.append((value / footprint, footprint))

        densities.sort(reverse=True)
        self.lam = 0.0
        if sum(footprint for _density, footprint in densities) > free_tokens:
            used = 0.0
            for density, footprint in densities:
                self.lam = density
                used += footprint
                if used >= free_tokens:
                    break
            self.binding_updates += 1
        self.epoch += 1
        self.price_sum += self.lam
        self.price_max = max(self.price_max, self.lam)

    @staticmethod
    def bayesian_value(estimate: WorkEstimate) -> float:
        """SERPT value under the posterior over remaining program turns."""
        return 1.0 / max(1e-12, estimate.remaining_work)

    def admission_score(
        self,
        p: ProgramView,
        estimate: WorkEstimate,
        view: SystemView,
    ) -> float:
        value = self.bayesian_value(estimate)
        return value - self.lam * estimate.expected_footprint_tokens

    def diagnostics(self) -> Dict[str, float]:
        return {
            "bdp_lambda_last": self.lam,
            "bdp_lambda_mean": self.price_sum / max(1, self.epoch),
            "bdp_lambda_max": self.price_max,
            "bdp_updates": float(self.epoch),
            "bdp_binding_fraction": self.binding_updates / max(1, self.epoch),
        }


POLICIES = {
    "fcfs": FCFSPolicy,
    "thunder": ThunderPolicy,
    "bdp": BDPPolicy,
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
                 service: Optional[ServiceModel] = None, sim_cfg: Optional[SimConfig] = None,
                 record_trace: bool = False):
        self.cfg = cfg
        self.service = service or ServiceModel()
        self.sim_cfg = sim_cfg or SimConfig()
        self.policy = policy
        self.record_trace = record_trace
        self.policy.reset(cfg, self.service)
        self.block_size = cfg.block_size
        self.capacity_blocks = cfg.capacity_tokens // cfg.block_size
        shape = 1.0 / (cfg.decode_cv * cfg.decode_cv)
        scale = cfg.decode_mean / shape
        self.decode_reserve = (
            int(cfg.decode_reserve_tokens)
            if cfg.decode_reserve_tokens is not None
            else int(math.ceil(gamma_dist.ppf(cfg.reserve_quantile, a=shape, scale=scale)))
        )
        self.p: List[RuntimeProgram] = []
        for pid, spec in enumerate(specs):
            self.p.append(RuntimeProgram(
                pid=pid,
                spec=spec,
                prompt=spec.requests[0].prompt_tokens,
            ))
        self.now = 0.0
        self.next_tick = 0.0
        self.next_sample = self.sim_cfg.dt
        # vLLM APC hashes remain attached to blocks on its free list. They are
        # reusable, but do not count as occupied KV. New active allocations
        # overwrite the oldest hashes. Values are contiguous reusable prefixes;
        # order is block-LRU and needs no fitted hit-rate parameter.
        self.shadow_cache: "OrderedDict[int, float]" = OrderedDict()
        # These counters are exact derived state, not model parameters.  The
        # hot path used to recompute both values by scanning every program or
        # cached prefix for every generated token.
        self._physical_blocks = 0
        self._shadow_blocks = 0

        self._reset_metrics()

    def _reset_metrics(self) -> None:
        """Initialize aggregate metrics for one all-at-once batch."""
        self.program_ct: List[float] = []
        self.request_ct: List[float] = []
        self.queue_times: List[float] = []
        self.warm_admits = self.cold_admits = 0
        self.hit_prefix = self.requested_prefix = 0.0
        self.prefill_tokens = self.recompute_tokens = self.decode_tokens = 0.0
        self.evictions = self.emergency_evictions = 0
        self.evicted_tokens = 0.0
        self.reserve_expansions = self.stall_s = self.memory_violations = 0
        self.active_preemptions = 0
        self.preempted_context_tokens = 0.0
        self.max_active = 0
        self.max_waiting = 0
        self.max_router_paused = 0
        self.physical_area = self.reserved_area = self.cache_area = 0.0
        self.active_area = self.decode_area = self.prefill_area = self.busy_s = 0.0
        self.engine_waiting_area = self.router_paused_area = 0.0
        self.decision_calls = 0
        self.decision_time = 0.0
        self.decision_samples: List[float] = []
        # The public vLLM traces are sampled every 0.5 s. Keep their core
        # counters plus derived state/rate fields so aggregate means cannot hide
        # a wrong cold-start, completion ramp, or drain mechanism.
        self.timeseries: List[Dict[str, float]] = []
        # Detailed traces are opt-in so ordinary calibration and policy sweeps
        # retain their small memory footprint and fast hot path.
        self.program_samples: List[Dict[str, object]] = []
        self.scheduler_events: List[Dict[str, object]] = []
        self._sample_time = 0.0
        self._sample_decode_tokens = 0.0
        self._sample_prefill_tokens = 0.0

    def blocks(self, tokens: float) -> int:
        return 0 if tokens <= 1e-9 else int(math.ceil(tokens / self.block_size))

    def admit_blocks(self, p: RuntimeProgram) -> int:
        # The router's output pad is only an admission forecast. vLLM
        # materializes KV incrementally and may recompute-preempt on pressure.
        pad = (
            self.policy.initial_decode_pad()
            if p.initial and not p.preempted
            else self.policy.decode_pad(self.decode_reserve)
        )
        return max(1, self.blocks(p.prefix + p.prompt + p.generated + pad))

    def cache_blocks(self, p: RuntimeProgram) -> int:
        return self.blocks(p.prefix) if p.cache_warm and p.phase in (TOOL, READY) else 0

    def active(self) -> List[RuntimeProgram]:
        return [p for p in self.p if p.phase in (PREFILL, DECODE)]

    def ready(self) -> List[RuntimeProgram]:
        return [p for p in self.p if p.phase == READY]

    def engine_waiting(self) -> List[RuntimeProgram]:
        return [p for p in self.p if p.phase == ENGINE_WAITING]

    def planning_used(self) -> int:
        return sum(p.reserve_blocks for p in self.active()) + sum(self.cache_blocks(p) for p in self.p)

    def physical_used(self) -> int:
        # Free-list APC hashes and router-retained tool programs are not physical
        # occupancy. Only currently scheduled requests hold referenced blocks.
        return self._physical_blocks

    def set_active_kv(self, p: RuntimeProgram, tokens: float) -> None:
        """Update one context and its exact block-granular occupancy."""
        old_blocks = self.blocks(p.active_kv)
        p.active_kv = float(tokens)
        self._physical_blocks += self.blocks(p.active_kv) - old_blocks

    def set_shadow_prefix(self, pid: int, tokens: float) -> None:
        """Insert or replace one APC prefix while maintaining cached blocks."""
        old = self.shadow_cache.get(pid, 0.0)
        self._shadow_blocks -= self.blocks(old)
        self.shadow_cache[pid] = float(tokens)
        self._shadow_blocks += self.blocks(tokens)

    def pop_shadow_prefix(self, pid: int) -> float:
        """Remove one APC prefix and return its token length."""
        tokens = self.shadow_cache.pop(pid, 0.0)
        self._shadow_blocks -= self.blocks(tokens)
        return float(tokens)

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
                observed_decode_lengths=tuple(
                    req.decode_tokens_actual for req in p.spec.requests[:p.idx]
                ),
                terminal_prob=p.spec.prior.terminal_probability(p.stage),
                remaining_rounds=p.spec.prior.remaining_mean(p.stage),
                initial=p.initial,
                admit_blocks=self.admit_blocks(p) if p.phase == READY else 0,
                cache_blocks=self.cache_blocks(p),
                reserve_blocks=p.reserve_blocks,
                current_context=p.active_kv if p.phase in (PREFILL, DECODE) else float(p.prefix),
                preempted=p.preempted,
                marked_for_pause=p.marked_for_pause,
            ))
        return SystemView(
            now=self.now,
            block_size=self.block_size,
            capacity_blocks=self.capacity_blocks,
            decode_mean=(self.cfg.policy_decode_mean if self.cfg.policy_decode_mean is not None else self.cfg.decode_mean),
            decode_program_cv=self.cfg.decode_program_cv,
            decode_small_mean=self.cfg.decode_mean,
            decode_small_cv=self.cfg.decode_cv,
            decode_large_prob=self.cfg.decode_large_prob,
            decode_large_mean=self.cfg.decode_large_mean,
            decode_large_cv=self.cfg.decode_large_cv,
            prompt_mean=(self.cfg.policy_prompt_mean if self.cfg.policy_prompt_mean is not None else self.cfg.prompt_mean),
            context_limit=self.cfg.context_limit_tokens,
            decode_reserve=self.decode_reserve,
            alpha_work=(self.cfg.policy_alpha_work if self.cfg.policy_alpha_work is not None else self.sim_cfg.alpha_work),
            control_interval=self.sim_cfg.control_interval,
            service=self.service,
            programs=tuple(views),
        )

    def evict(self, p: RuntimeProgram, emergency: bool = False) -> None:
        if not p.cache_warm or p.phase not in (TOOL, READY):
            return
        p.cache_warm = False
        # Router eviction cannot delete a vLLM APC hash; it only stops treating
        # the tool program as retained for admission planning.
        self.evictions += 1
        self.evicted_tokens += p.prefix
        if emergency:
            self.emergency_evictions += 1

    def reclaim_shadow(self, need: int) -> int:
        """Overwrite LRU free-list APC hashes, allowing one partial prefix."""
        freed = 0
        while need > freed and self.shadow_cache:
            pid, tokens = self.shadow_cache.popitem(last=False)
            blocks = self.blocks(tokens)
            self._shadow_blocks -= blocks
            take = min(blocks, need - freed)
            remain_blocks = blocks - take
            freed += take
            if remain_blocks > 0:
                # Prefix caching can only reuse a contiguous prefix.
                self.set_shadow_prefix(pid, float(remain_blocks * self.block_size))
                self.shadow_cache.move_to_end(pid, last=False)
        return freed

    def trim_shadow_cache(self) -> None:
        """Keep cached hashes within the engine's currently free block count."""
        free = max(0, self.capacity_blocks - self.physical_used())
        if self._shadow_blocks > free:
            self.reclaim_shadow(self._shadow_blocks - free)

    def remember_engine_prefix(self, p: RuntimeProgram, tokens: float) -> None:
        """Release an engine context to the APC free list."""
        if tokens <= 0.0:
            return
        self.set_shadow_prefix(p.pid, float(tokens))
        self.shadow_cache.move_to_end(p.pid)
        self.trim_shadow_cache()

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
        """Resume at the router and submit the request to vLLM's waiting queue."""
        reserve = self.admit_blocks(p)
        old = self.cache_blocks(p)
        if self.planning_used() - old + reserve > self.capacity_blocks:
            return False
        p.reserve_blocks = reserve
        p.cache_warm = False
        p.admitted_at = self.now
        p.engine_queued_at = self.now
        p.phase = ENGINE_WAITING
        return True

    def engine_start(self, p: RuntimeProgram, committed_prefill: Optional[float] = None) -> bool:
        """Move one submitted request from vLLM waiting to running."""
        if p.phase != ENGINE_WAITING:
            return False
        replay_tokens = float(p.prefix + p.prompt + (p.generated if p.preempted else 0.0))
        cached = min(replay_tokens, self.shadow_cache.get(p.pid, 0.0))
        # vLLM 0.12 uses chunked prefill. Cached blocks must be referenced at
        # once; only work scheduled in the current service window needs new KV
        # immediately. Decode remains incremental and may later preempt.
        uncached = max(0.0, replay_tokens - cached)
        chunk = self.service.prefill_chunk_tokens
        if p.preempted:
            # Do not immediately cycle the same recompute victim through one
            # tiny chunk while pressure is unchanged. vLLM leaves it waiting
            # until its replay can be scheduled safely.
            needed = max(1, self.blocks(replay_tokens))
        else:
            committed_uncached = uncached if committed_prefill is None else min(uncached, committed_prefill)
            if chunk is not None:
                committed_uncached = min(committed_uncached, float(chunk))
            needed = max(1, self.blocks(cached + committed_uncached))
        if self.physical_used() + needed > self.capacity_blocks:
            return False
        self.pop_shadow_prefix(p.pid)
        if not p.cache_accounted:
            submitted_prompt = float(p.prefix + p.prompt)
            prefix_hit = min(float(p.prefix), cached)
            self.requested_prefix += submitted_prompt
            self.hit_prefix += prefix_hit
            if prefix_hit > 0.0:
                self.warm_admits += 1
            else:
                self.cold_admits += 1
            self.recompute_tokens += max(0.0, float(p.prefix) - prefix_hit)
            p.cache_accounted = True
        elif p.preempted:
            # Recompute after engine preemption is internal work, not a second
            # client-side prefix-cache query.
            self.recompute_tokens += max(0.0, replay_tokens - cached)
        self.set_active_kv(p, cached)
        p.prefill_remaining = max(0.0, replay_tokens - cached)
        p.phase = PREFILL if p.prefill_remaining > 1e-9 else DECODE
        p.preempted = False
        self.trim_shadow_cache()
        return True

    def schedule_engine(self, window_s: Optional[float] = None) -> None:
        """FCFS starts under the shared continuous-time prefill budget.

        ``max_num_batched_tokens`` is a per-engine-step limit, while this model
        advances in much coarser ``dt`` windows.  We therefore derive the work
        available in one simulator window from the measured prefill curve and
        use ``prefill_chunk_tokens`` only as the largest first chunk for one
        request.  This introduces no fitted queue-rate parameter and preserves
        already-running KV while the cold batch ramps up.
        """
        slots = max(0, self.service.max_batch - len(self.active()))
        if slots <= 0:
            return
        waiting = sorted(self.engine_waiting(), key=lambda x: (x.engine_queued_at, x.pid))
        if not waiting:
            return

        # The observable 61--69 request queue is a cold-start burst: all 80
        # clients submit their first call together.  Once that burst has been
        # consumed, tool returns are sparse and fit in ordinary engine steps;
        # applying a coarse dt-sized budget to them creates an artificial queue
        # feedback loop.  This switch is state-derived, not a fitted threshold.
        cold_start_burst = any(p.initial and not p.cache_accounted for p in waiting)
        if not cold_start_burst:
            for p in waiting:
                if slots <= 0:
                    break
                if self.engine_start(p):
                    slots -= 1
            return

        prefills = [p for p in self.active() if p.phase == PREFILL]
        replay = [
            max(0.0, float(p.prefix + p.prompt + (p.generated if p.preempted else 0.0))
                - min(float(p.prefix + p.prompt + (p.generated if p.preempted else 0.0)),
                      self.shadow_cache.get(p.pid, 0.0)))
            for p in waiting[:slots]
        ]
        demand = [p.prefill_remaining for p in prefills] + replay
        mean_uncached = float(np.mean(demand)) if demand else 0.0
        n_prefill = min(self.service.max_batch, max(1, len(prefills) + min(slots, len(waiting))))
        window_budget = self.service.aggregate_prefill_tps(n_prefill, mean_uncached) * (
            self.sim_cfg.dt if window_s is None else window_s
        )
        if prefills and any(p.phase == DECODE for p in self.active()):
            if self.service.mixed_prefill_multiplier is not None:
                window_budget *= self.service.mixed_prefill_multiplier
            else:
                window_budget *= self.service.mixed_prefill_share
        available = max(0.0, window_budget - sum(min(p.prefill_remaining, window_budget) for p in prefills))
        chunk = float(self.service.prefill_chunk_tokens) if self.service.prefill_chunk_tokens is not None else math.inf

        for p, uncached in zip(waiting, replay):
            if slots <= 0:
                break
            # A cache-only continuation still consumes one scheduled decode
            # token. An uncached request needs some prefill capacity to start.
            cost = 1.0 if uncached <= 1e-9 else min(uncached, chunk, available)
            if uncached > 1e-9 and cost <= 1e-9:
                break
            if self.engine_start(p, committed_prefill=cost):
                slots -= 1
                available = max(0.0, available - cost)

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
        for pid in plan.mark_after_turn:
            p = by.get(pid)
            if p and p.phase in (PREFILL, DECODE):
                p.marked_for_pause = True
        if self.planning_used() > self.capacity_blocks:
            self.free_blocks(self.planning_used() - self.capacity_blocks, emergency=False)
        self.check_memory()

    def refresh_router_occupancy(self) -> None:
        # The router sees occupancy only on its metrics/control tick.
        for p in self.active():
            observed = self.blocks(p.active_kv)
            if observed > p.reserve_blocks:
                p.reserve_blocks = observed
                self.reserve_expansions += 1

    def run_scheduler(self) -> None:
        self.refresh_router_occupancy()
        before = self.view()
        t0 = time.perf_counter()
        plan = self.policy.plan(before)
        dt = time.perf_counter() - t0
        self.decision_calls += 1
        self.decision_time += dt
        self.decision_samples.append(dt)
        if self.record_trace:
            admitted = [self.p[pid] for pid in dict.fromkeys(plan.admit)]
            remaining_decodes = [
                float(sum(req.decode_tokens_actual for req in p.spec.requests[p.idx :]))
                for p in admitted
            ]
            remaining_rounds = [float(len(p.spec.requests) - p.idx) for p in admitted]
            self.scheduler_events.append({
                "t_s": self.now,
                "ready": float(len(before.ready)),
                "active": float(len(before.active)),
                "engine_waiting": float(len(before.engine_waiting)),
                "admit_count": float(len(admitted)),
                "admitted_pids": ",".join(str(p.pid) for p in admitted),
                "admitted_mean_actual_remaining_decode": (
                    float(np.mean(remaining_decodes)) if remaining_decodes else math.nan
                ),
                "admitted_mean_actual_remaining_rounds": (
                    float(np.mean(remaining_rounds)) if remaining_rounds else math.nan
                ),
            })
        self.apply_plan(plan)

    def check_memory(self) -> None:
        if self.physical_used() > self.capacity_blocks:
            self.memory_violations += 1

    def preempt_active(self, p: RuntimeProgram) -> None:
        """Recompute-preempt inside vLLM; the router still considers it active."""
        if p.phase not in (PREFILL, DECODE):
            return
        self.active_preemptions += 1
        self.preempted_context_tokens += p.active_kv
        p.preemptions += 1
        p.preempted = True
        released = p.active_kv
        p.phase = ENGINE_WAITING
        p.cache_warm = False
        p.prefill_remaining = 0.0
        self.set_active_kv(p, 0.0)
        p.reserve_blocks = 0
        p.engine_queued_at = self.now
        self.remember_engine_prefix(p, released)

    def free_incremental_blocks(self, requester: RuntimeProgram, need: int) -> int:
        """Free physical blocks, matching vLLM's recompute-preemption path."""
        if need <= 0:
            return 0
        before = self.physical_used()
        while self.capacity_blocks - self.physical_used() < need:
            victims = [
                p for p in self.active()
                if p.pid != requester.pid
            ]
            if not victims:
                break
            # Stock vLLM uses a last-in style victim order for recompute
            # preemption. PID is a deterministic tie breaker for the cold ramp.
            victim = max(victims, key=lambda x: (x.engine_queued_at, x.pid))
            self.preempt_active(victim)
        return max(0, before - self.physical_used())

    def allowed_progress(self, p: RuntimeProgram, proposed: float) -> float:
        current = self.blocks(p.active_kv)
        target = self.blocks(p.active_kv + proposed)
        extra = max(0, target - current)
        free = self.capacity_blocks - self.physical_used()
        if free < extra:
            self.free_incremental_blocks(p, extra)
            free = self.capacity_blocks - self.physical_used()
        if free >= extra:
            return proposed
        boundary = (current + max(0, free)) * self.block_size
        return max(0.0, min(proposed, boundary - p.active_kv - 1e-9))

    def integrals(self, dt: float) -> None:
        active = self.active()
        n_p = sum(p.phase == PREFILL for p in active)
        n_d = sum(p.phase == DECODE for p in active)
        self.physical_area += self.physical_used() * dt
        self.reserved_area += self.planning_used() * dt
        self.cache_area += self._shadow_blocks * dt
        self.active_area += len(active) * dt
        self.engine_waiting_area += len(self.engine_waiting()) * dt
        self.router_paused_area += len(self.ready()) * dt
        self.prefill_area += n_p * dt
        self.decode_area += n_d * dt
        if active:
            self.busy_s += dt

    def sample(self, at: float) -> None:
        """Record the same coarse engine counters exposed by the real runs."""
        running = len(self.active())
        waiting = len(self.engine_waiting())
        elapsed = max(1e-12, at - self._sample_time)
        prefill_tps = (self.prefill_tokens - self._sample_prefill_tokens) / elapsed
        decode_tps = (self.decode_tokens - self._sample_decode_tokens) / elapsed
        self._sample_time = at
        self._sample_prefill_tokens = self.prefill_tokens
        self._sample_decode_tokens = self.decode_tokens
        self.max_active = max(self.max_active, running)
        self.max_waiting = max(self.max_waiting, waiting)
        self.max_router_paused = max(self.max_router_paused, len(self.ready()))
        self.timeseries.append({
            "t_s": at,
            "kv_perc": 100.0 * self.physical_used() / max(1, self.capacity_blocks),
            "running": float(running),
            "waiting": float(waiting),
            "router_ready": float(len(self.ready())),
            "tool": float(sum(p.phase == TOOL for p in self.p)),
            "prefill": float(sum(p.phase == PREFILL for p in self.p)),
            "decode": float(sum(p.phase == DECODE for p in self.p)),
            "completed_programs": float(len(self.program_ct)),
            "completed_requests": float(len(self.request_ct)),
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "prefix_hit_cum": 100.0 * self.hit_prefix / max(1.0, self.requested_prefix),
            "preemptions": float(self.active_preemptions),
        })
        if self.record_trace:
            self.program_samples.extend({
                "t_s": at,
                "pid": p.pid,
                "phase": p.phase,
                "stage": p.stage,
            } for p in self.p if p.phase != DONE)

    def advance_tools(self, dt: float) -> None:
        for p in self.p:
            if p.phase != TOOL:
                continue
            p.tool_elapsed += dt
            p.tool_remaining -= dt
            if p.tool_remaining <= 1e-9:
                p.tool_remaining = 0.0
                p.prompt = p.request.prompt_tokens
                p.ready_since = self.now + dt
                if p.cache_warm:
                    # A router-active program sends its next request immediately;
                    # only an explicitly paused program returns to READY.
                    p.reserve_blocks = self.admit_blocks(p)
                    p.cache_warm = False
                    p.engine_queued_at = self.now + dt
                    p.phase = ENGINE_WAITING
                else:
                    p.phase = READY

    def complete_turn(self, p: RuntimeProgram, at: float) -> None:
        req = p.request
        marked_for_pause = p.marked_for_pause
        self.request_ct.append(at)
        self.queue_times.append(max(0.0, p.admitted_at - p.ready_since))
        p.prefix += p.prompt + req.decode_tokens_actual
        p.idx += 1
        p.reserve_blocks = 0
        self.set_active_kv(p, 0.0)
        p.prefill_remaining = p.generated = 0.0
        p.preempted = False
        p.cache_accounted = False
        p.marked_for_pause = False
        if p.idx >= len(p.spec.requests):
            p.phase = DONE
            p.cache_warm = False
            self.set_active_kv(p, 0.0)
            p.completion_time = at
            self.program_ct.append(at)
        else:
            p.phase = TOOL
            p.cache_warm = not marked_for_pause and self.policy.retain_during_tool()
            p.prompt = 0
            p.tool_elapsed = 0.0
            p.tool_remaining = req.tool_duration_actual
        self.remember_engine_prefix(p, p.prefix)

    def advance_gpu(self, dt: float) -> None:
        prefills = [p for p in self.p if p.phase == PREFILL]
        decodes = [p for p in self.p if p.phase == DECODE]
        if not prefills and not decodes:
            return
        if prefills and decodes:
            if self.service.mixed_prefill_multiplier is not None:
                pf_share = self.service.mixed_prefill_multiplier
                dec_share = (
                    self.service.mixed_decode_multiplier
                    if self.service.mixed_decode_multiplier is not None else 1.0
                )
            else:
                pf_share, dec_share = self.service.mixed_prefill_share, 1.0 - self.service.mixed_prefill_share
        elif prefills:
            pf_share, dec_share = 1.0, 0.0
        else:
            pf_share, dec_share = 0.0, 1.0

        if prefills:
            rate = pf_share * self.service.aggregate_prefill_tps(len(prefills), float(np.mean([p.prefill_remaining for p in prefills]))) / len(prefills)
            for p in list(prefills):
                if p.phase != PREFILL:
                    continue
                proposed = min(p.prefill_remaining, rate * dt)
                allowed = self.allowed_progress(p, proposed)
                p.prefill_remaining -= allowed
                self.set_active_kv(p, p.active_kv + allowed)
                self.trim_shadow_cache()
                self.prefill_tokens += allowed
                if p.prefill_remaining <= 1e-7:
                    p.prefill_remaining = 0.0
                    self.set_active_kv(p, p.prefix + p.prompt + p.generated)
                    p.phase = DECODE

        decodes = [p for p in self.p if p.phase == DECODE]
        if decodes and dec_share > 0:
            mean_ctx = float(np.mean([p.active_kv for p in decodes]))
            rate = dec_share * self.service.aggregate_decode_tps(len(decodes), mean_ctx) / len(decodes)
            finished: List[RuntimeProgram] = []
            for p in list(decodes):
                if p.phase != DECODE:
                    continue
                remaining = max(0.0, p.request.decode_tokens_actual - p.generated)
                proposed = min(remaining, rate * dt)
                allowed = self.allowed_progress(p, proposed)
                if allowed + 1e-10 < proposed:
                    self.stall_s += dt
                p.generated += allowed
                self.set_active_kv(p, p.active_kv + allowed)
                self.trim_shadow_cache()
                self.decode_tokens += allowed
                if p.generated + 1e-7 >= p.request.decode_tokens_actual:
                    p.generated = float(p.request.decode_tokens_actual)
                    self.set_active_kv(p, p.prefix + p.prompt + p.request.decode_tokens_actual)
                    finished.append(p)
            for p in finished:
                self.complete_turn(p, self.now + dt)

    def run(self) -> Dict[str, float]:
        loops = 0
        while len(self.program_ct) < len(self.p):
            loops += 1
            if loops > 2_000_000 or self.now > self.sim_cfg.max_time:
                counts = {x: sum(p.phase == x for p in self.p) for x in (READY, ENGINE_WAITING, PREFILL, DECODE, TOOL, DONE)}
                raise RuntimeError(f"simulation stalled at t={self.now:.2f}; phases={counts}; used={self.planning_used()}/{self.capacity_blocks}")
            ready = bool(self.ready())
            active = bool(self.active())
            waiting = bool(self.engine_waiting())
            if self.now + 1e-9 >= self.next_tick or (ready and not active and not waiting):
                self.run_scheduler()
                self.next_tick = self.now + self.sim_cfg.control_interval
            dt = self.sim_cfg.dt
            if self.next_tick > self.now + 1e-9:
                dt = min(dt, self.next_tick - self.now)
            remaining_tools = [p.tool_remaining for p in self.p if p.phase == TOOL and p.tool_remaining > 1e-9]
            if remaining_tools:
                dt = min(dt, min(remaining_tools))
            dt = max(1e-6, dt)
            self.schedule_engine(dt)
            self.integrals(dt)
            self.advance_tools(dt)
            self.advance_gpu(dt)
            self.schedule_engine(dt)
            at = self.now + dt
            if at + 1e-9 >= self.next_sample:
                self.sample(at)
                while self.next_sample <= at + 1e-9:
                    self.next_sample += self.sim_cfg.dt
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
            "aggregate_gen_tps": self.decode_tokens / max(1e-9, T),
            "cache_evictions": float(self.evictions),
            "evicted_prefix_tokens": self.evicted_tokens,
            "emergency_evictions": float(self.emergency_evictions),
            "reserve_expansions": float(self.reserve_expansions),
            "active_preemptions": float(self.active_preemptions),
            "preempted_context_tokens": self.preempted_context_tokens,
            "decode_stall_s": self.stall_s,
            "memory_violations": float(self.memory_violations),
            "physical_kv_util": self.physical_area / max(1e-9, self.capacity_blocks * T),
            "reserved_kv_util": self.reserved_area / max(1e-9, self.capacity_blocks * T),
            "idle_cache_fraction": self.cache_area / max(1e-9, self.capacity_blocks * T),
            "mean_active": self.active_area / T,
            "max_active": float(self.max_active),
            "max_waiting": float(self.max_waiting),
            "mean_engine_waiting": self.engine_waiting_area / T,
            "max_router_paused": float(self.max_router_paused),
            "mean_router_paused": self.router_paused_area / T,
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


DEFAULT_POLICIES = ["fcfs", "thunder", "bdp"]


def coder_workload(name: str, seed: int) -> WorkloadConfig:
    """Return the single calibrated workload shape with a new realized seed."""
    return WorkloadConfig(
        name,
        n_programs=80,
        initial_prompt_mean=2_788.0,
        initial_prompt_cv=0.42,
        prompt_mean=184.0,
        prompt_cv=1.70,
        prompt_large_prob=0.255,
        prompt_large_mean=4_930.0,
        prompt_large_cv=0.16,
        decode_mean=162.0,
        decode_cv=1.19,
        decode_large_prob=0.068,
        decode_large_mean=3_619.0,
        decode_large_cv=0.80,
        decode_program_cv=0.50,
        prior_name="coder20_natural",
        policy_prior_name="coder16_empirical",
        policy_prompt_mean=1_600.0,
        policy_decode_mean=300.0,
        policy_alpha_work=0.03,
        tool_mix="coder_tools",
        capacity_tokens=670_000,
        block_size=16,
        decode_reserve_tokens=1_024,
        context_limit_tokens=35_000,
        seed=seed,
    )


def matrix(preset: str, seed0: int) -> List[WorkloadConfig]:
    """Canonical calibration seeds or fresh held-out algorithm seeds."""
    repetitions = {
        "coder_calibration_smoke": 1,
        "coder_calibration": 3,
        "coder_experiment": 20,
    }.get(preset)
    if repetitions is None:
        raise ValueError(
            f"unknown preset {preset!r}; expected coder_calibration_smoke, "
            "coder_calibration, or coder_experiment"
        )
    label = "calibration" if preset.startswith("coder_calibration") else "experiment"
    return [
        coder_workload(f"coder80_{label}_r{r}", seed0 + r)
        for r in range(repetitions)
    ]
