#!/usr/bin/env python3
"""Unified simulator for agentic LLM KV-cache scheduling.

This file consolidates the project variants in the repository:

* Bayesian Program Grade (BPG): commit_many, least_rounds, bayes_grade.
* Bayes-Dual-Price (BDP): current_density, current dual descent, BDP.
* Campaign-LP approximations: full LP update, solve once, selective update,
  infrequent update, online dual descent, short MPC, dual threshold.

The execution model follows the reports in the repository: all programs arrive
at time 0, a program is a sequential chain of LLM requests separated by tool
calls, active requests are non-preemptive, prefill is non-blocking, decode is a
full batch, active requests reserve peak KV, and inactive warm prefixes may be
held or evicted.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


# ---------------------------------------------------------------------------
# Data model and distributions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Request:
    p: int
    d: int


@dataclass(frozen=True)
class DiscreteGammaPrior:
    name: str
    shape: float
    scale: float
    k_max: int
    pmf: np.ndarray
    survival: np.ndarray
    mean: float

    @staticmethod
    def make(name: str, shape: float, scale: float, k_max: int) -> "DiscreteGammaPrior":
        k = np.arange(1, k_max + 1, dtype=float)
        logw = (shape - 1.0) * np.log(k) - k / scale
        w = np.exp(logw - logw.max())
        w /= w.sum()
        survival = np.zeros(k_max + 2, dtype=float)
        for j in range(1, k_max + 2):
            survival[j] = w[j - 1 :].sum() if j <= k_max else 0.0
        mean = float(np.dot(k, w))
        return DiscreteGammaPrior(name, shape, scale, k_max, w, survival, mean)

    def sample(self, rng: np.random.Generator) -> int:
        support = np.arange(1, self.k_max + 1)
        return int(rng.choice(support, p=self.pmf))

    def tail(self, j: int) -> float:
        if j <= 0:
            return 1.0
        if j > self.k_max:
            return 0.0
        return float(self.survival[j])

    def remaining_mean(self, stage_j: int) -> float:
        denom = self.tail(stage_j)
        if denom <= 0.0:
            return 1.0
        k_vals = np.arange(stage_j, self.k_max + 1, dtype=float)
        probs = self.pmf[stage_j - 1 :] / denom
        rem = k_vals - stage_j + 1.0
        return float(np.dot(rem, probs))


@dataclass(frozen=True)
class ProgramInstance:
    requests: List[Request]
    prior: DiscreteGammaPrior


@dataclass
class ProgramState:
    pid: int
    requests: List[Request]
    prior: DiscreteGammaPrior
    idx: int = 0
    A: int = 0
    h: bool = True
    mode: str = "waiting"
    reveal_order: int = 0
    waiting_since: float = 0.0
    committed: bool = False
    commit_order: int = 0
    prefill_remaining: float = 0.0
    decode_remaining: int = 0
    q_reserved: int = 0

    @property
    def j(self) -> int:
        return self.idx + 1

    @property
    def p(self) -> int:
        return self.requests[self.idx].p if self.idx < len(self.requests) else 0

    @property
    def d(self) -> int:
        return self.requests[self.idx].d if self.idx < len(self.requests) else 0

    @property
    def q(self) -> int:
        return self.A + self.p + self.d


@dataclass(frozen=True)
class Scenario:
    name: str = "balanced_mixed_equal"
    n: int = 80
    p_mean: float = 120.0
    d_mean: float = 150.0
    token_cv: float = 0.35
    alpha: float = 0.25
    prior_mix: str = "mixed_equal"
    seed: int = 0
    k_max: int = 100

    @property
    def sbar(self) -> float:
        return self.alpha * self.p_mean + self.d_mean

    @property
    def gbar(self) -> float:
        return self.p_mean + self.d_mean


def build_priors(k_max: int = 100) -> Dict[str, DiscreteGammaPrior]:
    return {
        "short": DiscreteGammaPrior.make("short", 2.0, 2.0, min(k_max, 30)),
        "medium": DiscreteGammaPrior.make("medium", 2.5, 3.8, min(k_max, 50)),
        "quick10": DiscreteGammaPrior.make("quick10", 2.5, 4.0, min(k_max, 60)),
        "long": DiscreteGammaPrior.make("long", 3.0, 6.2, min(k_max, 80)),
        "very_long": DiscreteGammaPrior.make("very_long", 3.2, 9.0, k_max),
    }


def build_profiles() -> Dict[str, Tuple[float, float]]:
    # Decode means are intentionally in the 100-200 range requested by the user.
    return {
        "prefill_heavy": (160.0, 100.0),
        "balanced": (120.0, 150.0),
        "decode_heavy": (80.0, 200.0),
    }


def sample_prior(rng: np.random.Generator, priors: Mapping[str, DiscreteGammaPrior], mix: str) -> DiscreteGammaPrior:
    if mix in priors:
        return priors[mix]
    names = ["short", "medium", "long", "very_long"]
    if mix == "mixed_equal":
        probs = [0.25, 0.25, 0.25, 0.25]
    elif mix == "mixed_long_heavy":
        probs = [0.10, 0.20, 0.35, 0.35]
    elif mix == "mixed_very_long":
        probs = [0.05, 0.10, 0.25, 0.60]
    else:
        raise ValueError(f"unknown prior_mix={mix}")
    return priors[str(rng.choice(names, p=probs))]


def sample_positive_gamma_int(rng: np.random.Generator, mean: float, cv: float) -> int:
    shape = 1.0 / max(1e-12, cv * cv)
    scale = mean / shape
    return max(1, int(round(rng.gamma(shape, scale))))


def generate_instance(scen: Scenario) -> Tuple[List[ProgramInstance], int, int]:
    rng = np.random.default_rng(scen.seed)
    priors = build_priors(scen.k_max)
    programs: List[ProgramInstance] = []
    qmax = 1
    total_requests = 0
    for _ in range(scen.n):
        prior = sample_prior(rng, priors, scen.prior_mix)
        K = prior.sample(rng)
        total_requests += K
        reqs: List[Request] = []
        A = 0
        for _j in range(K):
            p = sample_positive_gamma_int(rng, scen.p_mean, scen.token_cv)
            d = sample_positive_gamma_int(rng, scen.d_mean, scen.token_cv)
            reqs.append(Request(p, d))
            qmax = max(qmax, A + p + d)
            A += p + d
        programs.append(ProgramInstance(reqs, prior))
    return programs, qmax, total_requests


# ---------------------------------------------------------------------------
# Local value functions
# ---------------------------------------------------------------------------

PRIOR_REGISTRY: Dict[str, DiscreteGammaPrior] = {}


def register_prior(prior: DiscreteGammaPrior) -> None:
    PRIOR_REGISTRY[prior.name] = prior


def clear_value_caches() -> None:
    PRIOR_REGISTRY.clear()
    _grade_cached.cache_clear()
    _posterior_arrays.cache_clear()


def service_time(st: ProgramState, alpha: float, warm: Optional[bool] = None) -> float:
    if warm is None:
        warm = st.h
    return alpha * (st.p + (0 if warm else st.A)) + st.d


def expected_remaining_work(st: ProgramState, scen: Scenario, warm: Optional[bool] = None) -> float:
    er = st.prior.remaining_mean(st.j)
    tau = service_time(st, scen.alpha, warm)
    return tau + max(0.0, er - 1.0) * scen.sbar


def projected_footprint(st: ProgramState, scen: Scenario) -> float:
    er = st.prior.remaining_mean(st.j)
    return st.q + 0.5 * max(0.0, er - 1.0) * scen.gbar


@lru_cache(maxsize=500_000)
def _grade_cached(prior_name: str, j: int, s1_x100: int, future_x100: int) -> float:
    prior = PRIOR_REGISTRY[prior_name]
    denom0 = prior.tail(j)
    if denom0 <= 0.0:
        return 0.0
    s1 = s1_x100 / 100.0
    future = future_x100 / 100.0
    service = 0.0
    best = 0.0
    m_max = prior.k_max - j + 1
    for m in range(1, m_max + 1):
        surv = prior.tail(j + m - 1) / denom0
        service += (s1 if m == 1 else future) * surv
        complete_prob = 1.0 - prior.tail(j + m) / denom0
        if service > 0.0:
            best = max(best, complete_prob / service)
    return best


def bayes_grade(st: ProgramState, scen: Scenario, warm: bool) -> float:
    register_prior(st.prior)
    s1 = service_time(st, scen.alpha, warm)
    return _grade_cached(st.prior.name, st.j, int(round(100 * s1)), int(round(100 * scen.sbar)))


def mu_inverse_value(st: ProgramState, scen: Scenario, warm: bool) -> float:
    return 1.0 / max(1e-12, expected_remaining_work(st, scen, warm))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class BasePolicy:
    name = "base"

    def reset(self, scen: Scenario) -> None:
        pass

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, float]:
        return {}


class CommitManyPolicy(BasePolicy):
    name = "commit_many"

    def __init__(self) -> None:
        self.next_commit_order = 0

    def reset(self, scen: Scenario) -> None:
        self.next_commit_order = 0

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0

        committed = [p for p in waiting if p.committed]
        committed.sort(key=lambda p: (p.commit_order, p.reveal_order, p.pid))
        for p in committed:
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q
            elif p.h and p.A > 0 and used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A

        new = [p for p in waiting if not p.committed]
        new.sort(key=lambda p: (p.q, service_time(p, sim.scen.alpha, p.h), p.reveal_order))
        for p in new:
            if used + p.q <= free:
                p.committed = True
                p.commit_order = self.next_commit_order
                self.next_commit_order += 1
                actions[p.pid] = "ADMIT"
                used += p.q

        for p in committed:
            if actions[p.pid] == "EVICT" and p.h and p.A > 0 and used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class LeastRoundsPolicy(BasePolicy):
    name = "least_rounds"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0

        def key(p: ProgramState) -> Tuple[float, float, int, int]:
            return (p.idx, service_time(p, sim.scen.alpha, p.h), p.reveal_order, p.pid)

        for p in sorted(waiting, key=key):
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q

        for p in sorted(waiting, key=key):
            if actions[p.pid] == "EVICT" and p.h and p.A > 0 and used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class RandomPolicy(BasePolicy):
    """Trivial lower-bound baseline: admit waiting programs in a uniformly random
    order (subject to current KV feasibility), then HOLD warm prefixes with any
    residual memory in the same order. The RNG is seeded from the scenario in
    reset() so results are reproducible across runs."""

    name = "random"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def reset(self, scen: Scenario) -> None:
        # Offset the seed so the policy stream does not coincide with the
        # instance-generation RNG (np.random.default_rng(scen.seed)).
        self.rng = np.random.default_rng(scen.seed * 2 + 1)

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0

        order = [waiting[i] for i in self.rng.permutation(len(waiting))]
        for p in order:
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q

        for p in order:
            if actions[p.pid] == "EVICT" and p.h and p.A > 0 and used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class FIFOPolicy(BasePolicy):
    """Trivial baseline: first-come-first-served by arrival (reveal) order. Admit
    the oldest-arriving waiting programs first (subject to current KV
    feasibility), then HOLD warm prefixes with residual memory in the same
    order."""

    name = "fifo"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0

        order = sorted(waiting, key=lambda p: (p.reveal_order, p.pid))
        for p in order:
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q

        for p in order:
            if actions[p.pid] == "EVICT" and p.h and p.A > 0 and used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class SizePolicy(BasePolicy):
    """Token-size baseline, mirroring ThunderAgent's 'size' policy.

    Admit by (continuing-before-new, then ASCENDING footprint q=A+p+d): a program
    on a later round (idx>0, ThunderAgent REASONING) outranks a brand-new one
    (idx==0, NEW); within each group the smaller footprint goes first. Whatever
    does not fit is EVICT; residual memory HOLDs warm prefixes in the same order.
    Pure size ordering -- no value/decode awareness.
    """

    name = "size"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0

        def key(p: ProgramState) -> Tuple[int, int, int, int]:
            group = 0 if p.idx > 0 else 1          # continuing (step>1) before new (step==1)
            return (group, p.q, p.reveal_order, p.pid)   # ascending footprint within group

        for p in sorted(waiting, key=key):
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q

        for p in sorted(waiting, key=key):
            if actions[p.pid] == "EVICT" and p.h and p.A > 0 and used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class BayesianGradePolicy(BasePolicy):
    name = "bayes_grade"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0
        scored = []
        for p in waiting:
            g = bayes_grade(p, sim.scen, p.h)
            scored.append((g / max(1, p.q), g, -service_time(p, sim.scen.alpha, p.h), p))
        scored.sort(key=lambda x: x[:-1], reverse=True)
        for *_prefix, p in scored:
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q

        hold = []
        for p in waiting:
            if actions[p.pid] == "EVICT" and p.h and p.A > 0:
                g = bayes_grade(p, sim.scen, True)
                hold.append((g, -p.A, p))
        hold.sort(key=lambda x: x[:-1], reverse=True)
        for *_prefix, p in hold:
            if used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class BayesianGradeWarmMarginalPolicy(BasePolicy):
    """BPG variant from warm_marginal_full.py.

    For warm prefixes it divides the grade by only the incremental request
    footprint p+d; cold starts still pay the full active footprint.
    """

    name = "bayes_grade_warm_marginal"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0
        scored = []
        for p in waiting:
            g = bayes_grade(p, sim.scen, p.h)
            denom = max(1, p.p + p.d) if p.h else max(1, p.q)
            scored.append((g / denom, g, -service_time(p, sim.scen.alpha, p.h), p))
        scored.sort(key=lambda x: x[:-1], reverse=True)
        for *_prefix, p in scored:
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q
        hold = [(bayes_grade(p, sim.scen, True), p) for p in waiting if actions[p.pid] == "EVICT" and p.h and p.A > 0]
        hold.sort(key=lambda x: x[0], reverse=True)
        for _g, p in hold:
            if used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class CurrentDensityPolicy(BasePolicy):
    name = "current_density"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        used = 0
        scored = []
        for p in waiting:
            tau = service_time(p, sim.scen.alpha, p.h)
            scored.append((1.0 / max(1e-12, tau * max(1, p.q)), -tau, p))
        scored.sort(key=lambda x: x[:-1], reverse=True)
        for *_prefix, p in scored:
            if used + p.q <= free:
                actions[p.pid] = "ADMIT"
                used += p.q
        for *_prefix, p in scored:
            if actions[p.pid] == "EVICT" and p.h and p.A > 0 and used + p.A <= free:
                actions[p.pid] = "HOLD"
                used += p.A
        return actions


class DualDescentCurrentPolicy(BasePolicy):
    """Commitbase dual descent: current-request value plus one KV price."""

    name = "dual_descent_current"

    def __init__(self, eta0: float = 0.10) -> None:
        self.eta0 = eta0
        self.lam = 0.0
        self.epoch = 0
        self.scale: Optional[float] = None

    def reset(self, scen: Scenario) -> None:
        self.lam = 0.0
        self.epoch = 0
        self.scale = None

    def diagnostics(self) -> Dict[str, float]:
        return {"lambda_current_last": self.lam, "dual_updates": self.epoch}

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        if free <= 0 or not waiting:
            return actions

        desired = 0.0
        ratios = []
        for p in waiting:
            tau = service_time(p, sim.scen.alpha, p.h)
            v = 1.0 / max(1e-12, tau)
            ratios.append(v / max(1, p.q))
            if v - self.lam * p.q > 0:
                desired += p.q
        if self.scale is None:
            self.scale = max(float(np.median(ratios)) if ratios else 1e-9, 1e-12)
        step = self.eta0 * self.scale / math.sqrt(self.epoch + 1.0)
        self.lam = max(0.0, self.lam + step * (desired / max(1.0, free) - 1.0))
        self.epoch += 1

        run = []
        for p in waiting:
            tau = service_time(p, sim.scen.alpha, p.h)
            v = 1.0 / max(1e-12, tau)
            margin = v - self.lam * p.q
            if margin > 0:
                run.append((margin / max(1, p.q), margin, -tau, p))
        return pack_simple_run_hold(sim, actions, run, lambda p: (sim.scen.alpha * p.A) / max(1e-12, service_time(p, sim.scen.alpha, True)) - self.lam * p.A)


class BayesDualPricePolicy(BasePolicy):
    """Commitbase Bayes-Dual-Price: 1/(tau*mu) minus projected-KV price."""

    name = "bayes_dual_price"

    def __init__(self, eta0: float = 0.10) -> None:
        self.eta0 = eta0
        self.lam = 0.0
        self.epoch = 0
        self.scale: Optional[float] = None

    def reset(self, scen: Scenario) -> None:
        self.lam = 0.0
        self.epoch = 0
        self.scale = None

    def diagnostics(self) -> Dict[str, float]:
        return {"lambda_projected_last": self.lam, "dual_updates": self.epoch}

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        if free <= 0 or not waiting:
            return actions

        stats = {}
        desired = 0.0
        ratios = []
        for p in waiting:
            tau = service_time(p, sim.scen.alpha, p.h)
            mu = expected_remaining_work(p, sim.scen, p.h)
            v = 1.0 / (max(1e-12, tau) * max(1e-12, mu))
            qbar = projected_footprint(p, sim.scen)
            hold_val = (sim.scen.alpha * p.A) / max(1e-12, mu) if p.h and p.A > 0 else 0.0
            stats[p.pid] = (tau, mu, v, qbar, hold_val)
            ratios.append(v / max(1.0, qbar))
            if v - self.lam * qbar > 0:
                desired += qbar
        if self.scale is None:
            self.scale = max(float(np.median(ratios)) if ratios else 1e-9, 1e-12)
        step = self.eta0 * self.scale / math.sqrt(self.epoch + 1.0)
        self.lam = max(0.0, self.lam + step * (desired / max(1.0, free) - 1.0))
        self.epoch += 1

        run = []
        for p in waiting:
            tau, _mu, v, qbar, _hold = stats[p.pid]
            margin = v - self.lam * qbar
            if margin > 0:
                run.append((margin / max(1, p.q), margin, -tau, p))
        return pack_simple_run_hold(sim, actions, run, lambda p: stats[p.pid][4] - self.lam * p.A)


class MuTwoPricePolicy(BasePolicy):
    """Variant suggested by the repository images.

    Uses v_i=1/mu_i and separates present KV price lambda_t from projected
    context-growth price nu_t:
        Delta_run = v_i - lambda_t q_i - nu_t (qbar_i - q_i).
        Delta_hold = (v_warm - v_cold) - lambda_t A_i.
    """

    name = "mu_two_price"

    def __init__(self, eta0: float = 0.10) -> None:
        self.eta0 = eta0
        self.lam = 0.0
        self.nu = 0.0
        self.epoch = 0
        self.scale: Optional[float] = None

    def reset(self, scen: Scenario) -> None:
        self.lam = 0.0
        self.nu = 0.0
        self.epoch = 0
        self.scale = None

    def diagnostics(self) -> Dict[str, float]:
        return {"lambda_present_last": self.lam, "lambda_future_last": self.nu, "dual_updates": self.epoch}

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        waiting = list(sim.waiting)
        free = sim.free_memory()
        actions = {p.pid: "EVICT" for p in waiting}
        if free <= 0 or not waiting:
            return actions

        stats = {}
        demand_present = 0.0
        demand_growth = 0.0
        ratios = []
        for p in waiting:
            v = mu_inverse_value(p, sim.scen, p.h)
            vw = mu_inverse_value(p, sim.scen, True)
            vc = mu_inverse_value(p, sim.scen, False)
            qbar = projected_footprint(p, sim.scen)
            growth = max(0.0, qbar - p.q)
            margin = v - self.lam * p.q - self.nu * growth
            stats[p.pid] = (v, vw, vc, qbar, growth, margin)
            ratios.append(v / max(1.0, qbar))
            if margin > 0:
                demand_present += p.q
                demand_growth += growth
        if self.scale is None:
            self.scale = max(float(np.median(ratios)) if ratios else 1e-9, 1e-12)
        step = self.eta0 * self.scale / math.sqrt(self.epoch + 1.0)
        self.lam = max(0.0, self.lam + step * (demand_present / max(1.0, free) - 1.0))
        self.nu = max(0.0, self.nu + step * (demand_growth / max(1.0, free) - 0.25))
        self.epoch += 1

        run = []
        for p in waiting:
            v, _vw, _vc, _qbar, growth, _old = stats[p.pid]
            margin = v - self.lam * p.q - self.nu * growth
            if margin > 0:
                run.append((margin / max(1, p.q), margin, -service_time(p, sim.scen.alpha, p.h), p))
        return pack_simple_run_hold(sim, actions, run, lambda p: max(0.0, stats[p.pid][1] - stats[p.pid][2]) - self.lam * p.A)


def pack_simple_run_hold(
    sim: "Simulator",
    actions: Dict[int, str],
    run_list: List[Tuple[float, float, float, ProgramState]],
    hold_margin_fn,
) -> Dict[int, str]:
    free = sim.free_memory()
    used = 0
    run_list.sort(key=lambda x: x[:-1], reverse=True)
    for *_prefix, p in run_list:
        if used + p.q <= free:
            actions[p.pid] = "ADMIT"
            used += p.q

    if sim.waiting and not sim.active and not any(a == "ADMIT" for a in actions.values()):
        feasible = [p for p in sim.waiting if p.q <= free]
        if feasible:
            p = max(feasible, key=lambda x: (1.0 / max(1e-12, service_time(x, sim.scen.alpha, x.h) * max(1, x.q)), -x.q))
            actions[p.pid] = "ADMIT"
            used = p.q

    holds = []
    for p in sim.waiting:
        if actions[p.pid] == "EVICT" and p.h and p.A > 0:
            margin = float(hold_margin_fn(p))
            if margin > 0:
                holds.append((margin / max(1, p.A), margin, p))
    holds.sort(key=lambda x: x[:-1], reverse=True)
    for *_prefix, p in holds:
        if used + p.A <= free:
            actions[p.pid] = "HOLD"
            used += p.A
    return actions


# ---------------------------------------------------------------------------
# Campaign LP surrogate and approximations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CampaignConfig:
    horizon_requests: int = 8
    terminal_beta: float = 0.0
    max_options: int = 6


@dataclass(frozen=True)
class CampaignOption:
    m: int
    completion_prob: float
    expected_service: float
    expected_area: float
    value_rate: float
    qbar: float
    survival_after: float


@dataclass
class ProgramSurrogate:
    pid: int
    q: float
    A: float
    warm: bool
    options: List[CampaignOption]
    service_grade: float
    area_grade: float
    hold_value: float


@dataclass
class LPSolution:
    success: bool
    lambda_future: float = 0.0
    lambda_peak: float = 0.0
    objective: float = 0.0
    run_fraction: Dict[int, float] = field(default_factory=dict)
    hold_fraction: Dict[int, float] = field(default_factory=dict)
    solve_wall_sec: float = 0.0
    status: str = ""


@lru_cache(maxsize=100_000)
def _posterior_arrays(prior_name: str, j: int, m_cap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = PRIOR_REGISTRY[prior_name]
    denom = prior.tail(j)
    if denom <= 0 or m_cap <= 0:
        z = np.zeros(0, dtype=float)
        return z, z, z
    m_cap = min(m_cap, prior.k_max - j + 1)
    surv = prior.survival[j : j + m_cap].astype(float, copy=True) / denom
    survival_after = prior.survival[j + 1 : j + m_cap + 1].astype(float, copy=True) / denom
    complete = 1.0 - survival_after
    return surv, complete, survival_after


def _future_grade(prior: DiscreteGammaPrior, j: int, service: float) -> float:
    if j > prior.k_max or prior.tail(j) <= 0:
        return 0.0
    register_prior(prior)
    sx = int(round(100.0 * service))
    return _grade_cached(prior.name, j, sx, sx)


def build_program_surrogate(st: ProgramState, scen: Scenario, cfg: CampaignConfig) -> ProgramSurrogate:
    register_prior(st.prior)
    m_cap = min(cfg.horizon_requests, st.prior.k_max - st.j + 1)
    surv, complete, survival_after = _posterior_arrays(st.prior.name, st.j, m_cap)
    if len(surv) == 0:
        return ProgramSurrogate(st.pid, float(max(1, st.q)), float(max(0, st.A)), st.h, [], 0.0, 0.0, 0.0)

    service = np.full(len(surv), scen.sbar, dtype=float)
    service[0] = service_time(st, scen.alpha, st.h)
    q_path = st.q + scen.gbar * np.arange(len(surv), dtype=float)
    B = np.cumsum(service * surv)
    Area = np.cumsum(q_path * service * surv)
    reward = complete.copy()
    if cfg.terminal_beta > 0:
        for k in range(len(reward)):
            j_next = st.j + k + 1
            g_next = _future_grade(st.prior, j_next, scen.sbar)
            terminal_budget = max(scen.sbar, cfg.horizon_requests * scen.sbar / 2.0)
            reward[k] += cfg.terminal_beta * survival_after[k] * min(1.0, terminal_budget * g_next)

    value_rate = np.divide(reward, B, out=np.zeros_like(B), where=B > 0)
    service_grade_arr = np.divide(complete, B, out=np.zeros_like(B), where=B > 0)
    area_grade_arr = np.divide(complete, Area, out=np.zeros_like(Area), where=Area > 0)

    candidates = {0, len(surv) - 1, int(np.argmax(service_grade_arr)), int(np.argmax(area_grade_arr))}
    for target in (0.25, 0.50, 0.80, 0.95):
        loc = np.flatnonzero(complete >= target)
        if len(loc):
            candidates.add(int(loc[0]))
    idx = sorted(candidates)
    if len(idx) > cfg.max_options:
        must = sorted({0, len(surv) - 1, int(np.argmax(service_grade_arr)), int(np.argmax(area_grade_arr))})
        rest = [x for x in idx if x not in must]
        slots = max(0, cfg.max_options - len(must))
        if slots and rest:
            picks = np.linspace(0, len(rest) - 1, slots).round().astype(int)
            must += [rest[k] for k in picks]
        idx = sorted(set(must))[: cfg.max_options]

    options = [
        CampaignOption(
            m=k + 1,
            completion_prob=float(complete[k]),
            expected_service=float(B[k]),
            expected_area=float(Area[k]),
            value_rate=float(value_rate[k]),
            qbar=float(Area[k] / max(B[k], 1e-12)),
            survival_after=float(survival_after[k]),
        )
        for k in idx
    ]
    if st.h and st.A > 0:
        hold_value = max(0.0, bayes_grade(st, scen, True) - bayes_grade(st, scen, False))
    else:
        hold_value = 0.0
    return ProgramSurrogate(
        pid=st.pid,
        q=float(max(1, st.q)),
        A=float(max(0, st.A)),
        warm=st.h,
        options=options,
        service_grade=float(service_grade_arr.max(initial=0.0)),
        area_grade=float(area_grade_arr.max(initial=0.0)),
        hold_value=float(hold_value),
    )


def surrogate_cache_key(st: ProgramState, scen: Scenario, cfg: CampaignConfig) -> Tuple:
    return (
        st.pid,
        st.idx,
        st.A,
        bool(st.h),
        st.p,
        st.d,
        st.prior.name,
        scen.p_mean,
        scen.d_mean,
        scen.alpha,
        cfg.horizon_requests,
        cfg.terminal_beta,
        cfg.max_options,
    )


def build_surrogates(sim: "Simulator", cfg: CampaignConfig, cache: Optional[MutableMapping[Tuple, ProgramSurrogate]]) -> List[ProgramSurrogate]:
    out = []
    for st in sim.waiting:
        key = surrogate_cache_key(st, sim.scen, cfg)
        sg = cache.get(key) if cache is not None else None
        if sg is None:
            sg = build_program_surrogate(st, sim.scen, cfg)
            if cache is not None:
                cache[key] = sg
        out.append(sg)
    return out


def solve_campaign_lp(sim: "Simulator", cfg: CampaignConfig, cache: Optional[MutableMapping[Tuple, ProgramSurrogate]]) -> Tuple[LPSolution, List[ProgramSurrogate]]:
    surrogates = build_surrogates(sim, cfg, cache)
    free = float(max(0, sim.free_memory()))
    if free <= 0 or not surrogates:
        return LPSolution(False, status="empty_or_zero_capacity"), surrogates

    records: List[Tuple[int, str, float, float, float]] = []
    for sg in surrogates:
        for op in sg.options:
            records.append((sg.pid, "run", op.value_rate, sg.q, op.qbar))
        if sg.warm and sg.A > 0 and sg.hold_value > 0:
            records.append((sg.pid, "hold", sg.hold_value, sg.A, sg.A))
    if not records:
        return LPSolution(False, status="no_variables"), surrogates

    pids = [sg.pid for sg in surrogates]
    row_for = {pid: 2 + i for i, pid in enumerate(pids)}
    A = lil_matrix((2 + len(pids), len(records)), dtype=float)
    b = np.ones(2 + len(pids), dtype=float)
    b[0] = free
    b[1] = free
    c = np.zeros(len(records), dtype=float)
    for col, (pid, _kind, val, peak, future) in enumerate(records):
        c[col] = -val
        A[0, col] = peak
        A[1, col] = future
        A[row_for[pid], col] = 1.0

    t0 = time.perf_counter()
    try:
        res = linprog(c, A_ub=A.tocsr(), b_ub=b, bounds=(0.0, 1.0), method="highs")
    except Exception as exc:
        return LPSolution(False, solve_wall_sec=time.perf_counter() - t0, status=f"exception:{exc}"), surrogates
    elapsed = time.perf_counter() - t0
    if not res.success or res.x is None:
        return LPSolution(False, solve_wall_sec=elapsed, status=str(res.message)), surrogates

    try:
        lam_peak = max(0.0, -float(res.ineqlin.marginals[0]))
        lam_future = max(0.0, -float(res.ineqlin.marginals[1]))
    except Exception:
        lam_peak = 0.0
        lam_future = 0.0

    run_fraction = {pid: 0.0 for pid in pids}
    hold_fraction = {pid: 0.0 for pid in pids}
    for x, (pid, kind, _val, _peak, _future) in zip(res.x, records):
        if kind == "run":
            run_fraction[pid] += float(x)
        else:
            hold_fraction[pid] += float(x)
    return LPSolution(True, lam_future, lam_peak, float(-res.fun), run_fraction, hold_fraction, elapsed, "optimal"), surrogates


def reduced_values(
    surrogates: Sequence[ProgramSurrogate],
    lambda_future: float,
    lambda_peak: float,
    lp_solution: Optional[LPSolution] = None,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
    run_value: Dict[int, float] = {}
    hold_value: Dict[int, float] = {}
    run_density: Dict[int, float] = {}
    hold_density: Dict[int, float] = {}
    for sg in surrogates:
        if sg.options:
            margins = [op.value_rate - lambda_future * op.qbar - lambda_peak * sg.q for op in sg.options]
            run_value[sg.pid] = max(0.0, float(max(margins)))
            if lp_solution is not None:
                run_value[sg.pid] += 1e-12 * lp_solution.run_fraction.get(sg.pid, 0.0)
            run_density[sg.pid] = max(op.value_rate / max(sg.q, 1e-12) for op in sg.options)
        else:
            run_value[sg.pid] = 0.0
            run_density[sg.pid] = 0.0
        if sg.warm and sg.A > 0:
            hold_value[sg.pid] = max(0.0, sg.hold_value - (lambda_future + lambda_peak) * sg.A)
            if lp_solution is not None:
                hold_value[sg.pid] += 1e-12 * lp_solution.hold_fraction.get(sg.pid, 0.0)
            hold_density[sg.pid] = sg.hold_value / max(sg.A, 1e-12)
        else:
            hold_value[sg.pid] = 0.0
            hold_density[sg.pid] = 0.0
    return run_value, hold_value, run_density, hold_density


def pack_run_then_hold(
    sim: "Simulator",
    run_value: Mapping[int, float],
    hold_value: Mapping[int, float],
    run_density_fallback: Mapping[int, float],
    hold_density_fallback: Mapping[int, float],
) -> Dict[int, str]:
    waiting = list(sim.waiting)
    free = int(sim.free_memory())
    actions = {p.pid: "EVICT" for p in waiting}
    used = 0
    candidates = []
    for p in waiting:
        rv = float(run_value.get(p.pid, 0.0))
        if rv > 0 and p.q <= free:
            tau = service_time(p, sim.scen.alpha, p.h)
            candidates.append((rv / max(1, p.q), rv, float(run_density_fallback.get(p.pid, 0.0)), -tau, p))
    candidates.sort(key=lambda x: x[:-1], reverse=True)
    for *_prefix, p in candidates:
        if used + p.q <= free:
            actions[p.pid] = "ADMIT"
            used += p.q

    if waiting and not sim.active and not any(v == "ADMIT" for v in actions.values()):
        feasible = [p for p in waiting if p.q <= free]
        if feasible:
            p = max(feasible, key=lambda x: (float(run_density_fallback.get(x.pid, 0.0)), -service_time(x, sim.scen.alpha, x.h), -x.q))
            actions[p.pid] = "ADMIT"
            used = p.q

    holds = []
    for p in waiting:
        if actions[p.pid] == "ADMIT" or not p.h or p.A <= 0:
            continue
        hv = float(hold_value.get(p.pid, 0.0))
        if hv > 0:
            holds.append((hv / max(1, p.A), float(hold_density_fallback.get(p.pid, 0.0)), hv, p))
    holds.sort(key=lambda x: x[:-1], reverse=True)
    for *_prefix, p in holds:
        if used + p.A <= free:
            actions[p.pid] = "HOLD"
            used += p.A
    return actions


class AreaKnapsackPolicy(BasePolicy):
    name = "area_knapsack"

    def __init__(self, horizon_requests: int = 16) -> None:
        self.cfg = CampaignConfig(horizon_requests=horizon_requests, max_options=8)
        self.cache: Dict[Tuple, ProgramSurrogate] = {}

    def reset(self, scen: Scenario) -> None:
        self.cache = {}

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        sur = build_surrogates(sim, self.cfg, self.cache)
        rv = {s.pid: s.area_grade * s.q for s in sur}
        rd = {s.pid: s.area_grade for s in sur}
        hv = {s.pid: (s.area_grade * s.q * sim.scen.alpha * s.A if s.warm else 0.0) for s in sur}
        hd = {s.pid: (s.area_grade * s.q * sim.scen.alpha if s.A > 0 else 0.0) for s in sur}
        return pack_run_then_hold(sim, rv, hv, rd, hd)


class LPPolicyBase(BasePolicy):
    long_cfg = CampaignConfig(horizon_requests=8, terminal_beta=0.0, max_options=6)

    def __init__(self) -> None:
        self.lambda_future = 0.0
        self.lambda_peak = 0.0
        self.initialized = False
        self.resolve_count = 0
        self.lp_failures = 0
        self.policy_epoch = 0
        self.lambda_history: List[float] = []
        self.cache: Dict[Tuple, ProgramSurrogate] = {}

    def reset(self, scen: Scenario) -> None:
        self.__init__()

    def diagnostics(self) -> Dict[str, float]:
        return {
            "lp_resolves": self.resolve_count,
            "lp_failures": self.lp_failures,
            "lambda_future_last": self.lambda_future,
            "lambda_peak_last": self.lambda_peak,
            "lambda_mean": float(np.mean(self.lambda_history)) if self.lambda_history else self.lambda_future,
        }

    def _resolve(self, sim: "Simulator", cfg: Optional[CampaignConfig] = None) -> Tuple[LPSolution, List[ProgramSurrogate]]:
        sol, sur = solve_campaign_lp(sim, cfg or self.long_cfg, self.cache)
        self.resolve_count += 1
        if sol.success:
            self.lambda_future = sol.lambda_future
            self.lambda_peak = sol.lambda_peak
            self.initialized = True
        else:
            self.lp_failures += 1
        self.lambda_history.append(self.lambda_future)
        return sol, sur

    def _sur(self, sim: "Simulator", cfg: Optional[CampaignConfig] = None) -> List[ProgramSurrogate]:
        return build_surrogates(sim, cfg or self.long_cfg, self.cache)

    def _actions(self, sim: "Simulator", sur: List[ProgramSurrogate], sol: Optional[LPSolution] = None) -> Dict[int, str]:
        rv, hv, rd, hd = reduced_values(sur, self.lambda_future, self.lambda_peak, sol)
        return pack_run_then_hold(sim, rv, hv, rd, hd)

    @staticmethod
    def summary(sim: "Simulator") -> np.ndarray:
        unfinished = len(sim.states) - len(sim.done)
        waiting = sim.waiting
        if not waiting:
            return np.array([unfinished, 0, 0, 0, 0, 0], dtype=float)
        return np.array(
            [
                unfinished,
                len(waiting),
                sum(p.q for p in waiting) / max(1, sim.M),
                sum(p.A for p in waiting) / max(1, sim.M),
                np.mean([p.j for p in waiting]),
                np.mean([1.0 if p.h else 0.0 for p in waiting]),
            ],
            dtype=float,
        )

    def pressure(self, sim: "Simulator", sur: Sequence[ProgramSurrogate]) -> float:
        demand = sum(max((op.qbar for op in s.options), default=0.0) for s in sur)
        cap = max(1e-12, sim.free_memory())
        return float(demand / cap)


class FullLPUpdatePolicy(LPPolicyBase):
    name = "lp_update_long"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        self.policy_epoch += 1
        sol, sur = self._resolve(sim)
        if not sol.success:
            sur = self._sur(sim)
        return self._actions(sim, sur, sol if sol.success else None)


class SolveOnceLPIndexPolicy(LPPolicyBase):
    name = "lp_solve_once"

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        self.policy_epoch += 1
        if not self.initialized:
            sol, sur = self._resolve(sim)
            if not sol.success:
                sur = self._sur(sim)
            return self._actions(sim, sur, sol if sol.success else None)
        sur = self._sur(sim)
        self.lambda_history.append(self.lambda_future)
        return self._actions(sim, sur)


class SelectiveLPUpdatePolicy(LPPolicyBase):
    name = "lp_selective"

    def __init__(self, drift_threshold: float = 0.40, max_skip: int = 80) -> None:
        super().__init__()
        self.drift_threshold = drift_threshold
        self.max_skip = max_skip
        self.anchor: Optional[np.ndarray] = None
        self.skips = 0
        self.total_skips = 0

    def diagnostics(self) -> Dict[str, float]:
        d = super().diagnostics()
        d["selective_skips"] = self.total_skips
        return d

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        self.policy_epoch += 1
        now = self.summary(sim)
        resolve = not self.initialized
        if self.anchor is not None and not resolve:
            drift = float(np.max(np.abs(now - self.anchor) / np.maximum(1.0, np.abs(self.anchor))))
            resolve = drift > self.drift_threshold or self.skips >= self.max_skip
        if resolve:
            sol, sur = self._resolve(sim)
            self.anchor = now
            self.skips = 0
            if not sol.success:
                sur = self._sur(sim)
            return self._actions(sim, sur, sol if sol.success else None)
        self.skips += 1
        self.total_skips += 1
        sur = self._sur(sim)
        self.lambda_history.append(self.lambda_future)
        return self._actions(sim, sur)


class InfrequentResolvingPolicy(LPPolicyBase):
    name = "lp_infrequent"

    def __init__(self) -> None:
        super().__init__()
        self.fractions = [1.0, 0.75, 0.50, 0.25, 0.125, 0.0625]
        self.next_idx = 0
        self.base_n = 0
        self.anchor_price = 0.0
        self.anchor_pressure = 1.0

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        self.policy_epoch += 1
        if self.base_n == 0:
            self.base_n = len(sim.states)
        frac = (len(sim.states) - len(sim.done)) / max(1, self.base_n)
        resolve = not self.initialized
        if self.next_idx < len(self.fractions) and frac <= self.fractions[self.next_idx] + 1e-12:
            resolve = True
            self.next_idx += 1
        if resolve:
            sol, sur = self._resolve(sim)
            if not sol.success:
                sur = self._sur(sim)
            self.anchor_price = self.lambda_future
            self.anchor_pressure = max(1e-6, self.pressure(sim, sur))
            return self._actions(sim, sur, sol if sol.success else None)
        sur = self._sur(sim)
        pressure = max(1e-6, self.pressure(sim, sur))
        if self.anchor_price > 0:
            self.lambda_future = float(self.anchor_price * np.clip(pressure / self.anchor_pressure, 0.25, 4.0))
        self.lambda_history.append(self.lambda_future)
        return self._actions(sim, sur)


class OnlineLPDualDescentPolicy(LPPolicyBase):
    name = "lp_dual_descent"

    def __init__(self, eta0: float = 0.10) -> None:
        super().__init__()
        self.eta0 = eta0
        self.updates = 0
        self.price_scale = 1e-8

    def diagnostics(self) -> Dict[str, float]:
        d = super().diagnostics()
        d["dual_updates"] = self.updates
        return d

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        self.policy_epoch += 1
        if not self.initialized:
            sol, sur = self._resolve(sim)
            if not sol.success:
                sur = self._sur(sim)
            candidates = [op.value_rate / max(op.qbar, 1e-12) for s in sur for op in s.options if op.value_rate > 0]
            self.price_scale = float(np.median(candidates)) if candidates else 1e-8
        else:
            sur = self._sur(sim)
        desired = 0.0
        for s in sur:
            choices = [(0.0, 0.0)]
            choices += [(op.value_rate - self.lambda_future * op.qbar - self.lambda_peak * s.q, op.qbar) for op in s.options]
            if s.warm and s.A > 0:
                choices.append((s.hold_value - (self.lambda_future + self.lambda_peak) * s.A, s.A))
            val, foot = max(choices, key=lambda x: x[0])
            if val > 0:
                desired += foot
        cap = max(1e-12, sim.free_memory())
        self.updates += 1
        step = self.eta0 * self.price_scale / math.sqrt(self.updates)
        self.lambda_future = max(0.0, self.lambda_future + step * (desired / cap - 1.0))
        self.lambda_history.append(self.lambda_future)
        return self._actions(sim, sur)


class ShortHorizonMPCPolicy(LPPolicyBase):
    name = "mpc_short"

    def __init__(self, horizon_requests: int = 2, terminal_beta: float = 0.65) -> None:
        super().__init__()
        self.cfg = CampaignConfig(horizon_requests, terminal_beta, 4)

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        self.policy_epoch += 1
        sol, sur = self._resolve(sim, self.cfg)
        if not sol.success:
            sur = self._sur(sim, self.cfg)
        return self._actions(sim, sur, sol if sol.success else None)


class DualThresholdPolicy(LPPolicyBase):
    name = "dual_threshold"

    def __init__(self, pressure_exponent: float = 0.5) -> None:
        super().__init__()
        self.exp = pressure_exponent
        self.initial_price = 0.0
        self.initial_pressure = 1.0

    def decide(self, sim: "Simulator") -> Dict[int, str]:
        self.policy_epoch += 1
        if not self.initialized:
            sol, sur = self._resolve(sim)
            if not sol.success:
                sur = self._sur(sim)
            self.initial_price = self.lambda_future
            self.initial_pressure = max(1e-6, self.pressure(sim, sur))
        else:
            sur = self._sur(sim)
            ratio = np.clip(max(1e-6, self.pressure(sim, sur)) / self.initial_pressure, 0.25, 4.0)
            self.lambda_future = float(self.initial_price * ratio**self.exp)
            self.lambda_history.append(self.lambda_future)

        rv: Dict[int, float] = {}
        hv: Dict[int, float] = {}
        rd: Dict[int, float] = {}
        hd: Dict[int, float] = {}
        for s in sur:
            margins = [op.value_rate - self.lambda_future * op.qbar - self.lambda_peak * s.q for op in s.options]
            raw = max((op.value_rate / max(s.q, 1e-12) for op in s.options), default=0.0)
            rv[s.pid] = s.q * raw if margins and max(margins) > 0 else 0.0
            rd[s.pid] = raw
            hm = s.hold_value - (self.lambda_future + self.lambda_peak) * s.A
            hv[s.pid] = s.A * max(0.0, s.hold_value / max(s.A, 1e-12)) if hm > 0 else 0.0
            hd[s.pid] = s.hold_value / max(s.A, 1e-12) if s.A > 0 else 0.0
        return pack_run_then_hold(sim, rv, hv, rd, hd)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    def __init__(self, scen: Scenario, programs: Sequence[ProgramInstance], M: int, policy: BasePolicy):
        self.scen = scen
        self.programs = programs
        self.M = int(M)
        self.policy = policy
        self.policy.reset(scen)
        self.t = 0.0
        self.reveal_counter = 0
        self.states: List[ProgramState] = []
        for pid, inst in enumerate(programs):
            st = ProgramState(
                pid=pid,
                requests=inst.requests,
                prior=inst.prior,
                idx=0,
                A=0,
                h=True,
                mode="waiting",
                reveal_order=self.reveal_counter,
                waiting_since=0.0,
            )
            self.reveal_counter += 1
            self.states.append(st)
        self.waiting: List[ProgramState] = list(self.states)
        self.active: List[ProgramState] = []
        self.done: List[ProgramState] = []
        self.request_completion: List[Tuple[float, float]] = []
        self.program_completion: List[Tuple[float, float]] = []
        self.recompute_service = 0.0
        self.evictions = 0
        self.cold_admits = 0
        self.warm_admits = 0
        self.preempt_events = 0
        self.preempted_request_keys: set[Tuple[int, int]] = set()
        self.memory_delete_events = 0
        self.memory_keep_events = 0
        self.memory_violations = 0
        self.decision_wall_sec = 0.0
        self.decision_cpu_sec = 0.0
        self.decision_calls = 0
        self.decision_samples: List[float] = []
        self.memory_area = 0.0
        self.active_area = 0.0

    def memory_used(self) -> int:
        return sum(p.q_reserved for p in self.active) + sum(p.A for p in self.waiting if p.h)

    def free_memory(self) -> int:
        return self.M - sum(p.q_reserved for p in self.active)

    def check_memory(self) -> None:
        if self.memory_used() > self.M + 1e-9:
            self.memory_violations += 1

    def schedule(self) -> None:
        if not self.waiting:
            return
        w0 = time.perf_counter()
        c0 = time.process_time()
        actions = self.policy.decide(self)
        wall = time.perf_counter() - w0
        cpu = time.process_time() - c0
        self.decision_wall_sec += wall
        self.decision_cpu_sec += cpu
        self.decision_calls += 1
        self.decision_samples.append(wall)

        new_waiting: List[ProgramState] = []
        for p in list(self.waiting):
            act = actions.get(p.pid, "HOLD" if p.h else "EVICT")
            if act == "ADMIT":
                warm_start = p.h
                recompute = 0 if warm_start else p.A
                p.prefill_remaining = self.scen.alpha * (p.p + recompute)
                p.decode_remaining = p.d
                p.q_reserved = p.q
                p.mode = "active"
                if warm_start:
                    self.warm_admits += 1
                else:
                    self.cold_admits += 1
                    self.recompute_service += self.scen.alpha * recompute
                p.h = False
                self.active.append(p)
            elif act == "HOLD":
                if p.h and p.A > 0:
                    self.preempt_events += 1
                    self.memory_keep_events += 1
                    self.preempted_request_keys.add((p.pid, p.idx))
                p.h = True
                p.mode = "waiting"
                new_waiting.append(p)
            elif act == "EVICT":
                if p.h and p.A > 0:
                    self.evictions += 1
                    self.memory_delete_events += 1
                p.h = False
                p.mode = "waiting"
                new_waiting.append(p)
            else:
                raise ValueError(f"unknown action {act}")
        self.waiting = new_waiting
        self.check_memory()

    def complete_finished(self, finished: List[ProgramState]) -> None:
        for p in finished:
            if p not in self.active:
                continue
            self.active.remove(p)
            self.request_completion.append((self.t, self.decision_wall_sec))
            old_p, old_d = p.p, p.d
            p.A += old_p + old_d
            p.idx += 1
            p.q_reserved = 0
            if p.idx >= len(p.requests):
                p.mode = "done"
                p.h = False
                self.done.append(p)
                self.program_completion.append((self.t, self.decision_wall_sec))
            else:
                p.mode = "waiting"
                p.h = True
                p.prefill_remaining = 0.0
                p.decode_remaining = 0
                p.reveal_order = self.reveal_counter
                p.waiting_since = self.t
                self.reveal_counter += 1
                self.waiting.append(p)
        self.check_memory()

    def advance_integrals(self, dt: float) -> None:
        self.memory_area += self.memory_used() * dt
        self.active_area += len(self.active) * dt

    def force_admit_one(self) -> None:
        feasible = sorted(self.waiting, key=lambda p: service_time(p, self.scen.alpha, p.h))
        for p in feasible:
            if p.q <= self.M:
                warm_start = p.h
                recompute = 0 if warm_start else p.A
                p.prefill_remaining = self.scen.alpha * (p.p + recompute)
                p.decode_remaining = p.d
                p.q_reserved = p.q
                p.mode = "active"
                if warm_start:
                    self.warm_admits += 1
                else:
                    self.cold_admits += 1
                    self.recompute_service += self.scen.alpha * recompute
                p.h = False
                self.active.append(p)
                self.waiting.remove(p)
                return
        raise RuntimeError("stalled: no feasible request can fit in memory")

    def run(self) -> Dict[str, float]:
        need_schedule = True
        loops = 0
        while len(self.done) < len(self.states):
            loops += 1
            if loops > 50_000_000:
                raise RuntimeError("simulation loop exceeded guard")
            if need_schedule:
                self.schedule()
                need_schedule = False
                if not self.active and self.waiting:
                    self.force_admit_one()
            if not self.active:
                break

            ready = [p for p in self.active if p.prefill_remaining <= 1e-9]
            if ready:
                dt = 1.0
                self.advance_integrals(dt)
                for p in self.active:
                    if p.prefill_remaining > 1e-9:
                        p.prefill_remaining = max(0.0, p.prefill_remaining - dt)
                finished: List[ProgramState] = []
                for p in ready:
                    p.decode_remaining -= 1
                    if p.decode_remaining <= 0:
                        finished.append(p)
                self.t += dt
                if finished:
                    self.complete_finished(finished)
                    need_schedule = True
            else:
                dt = min(p.prefill_remaining for p in self.active if p.prefill_remaining > 1e-9)
                self.advance_integrals(dt)
                for p in self.active:
                    p.prefill_remaining = max(0.0, p.prefill_remaining - dt)
                self.t += dt

        C = np.array([x[0] for x in self.program_completion], dtype=float)
        C_dec = np.array([x[1] for x in self.program_completion], dtype=float)
        R = np.array([x[0] for x in self.request_completion], dtype=float)
        makespan = float(C.max()) if len(C) else math.nan
        delete_den = self.memory_keep_events + self.memory_delete_events
        out: Dict[str, float] = {
            "policy": self.policy.name,
            "mean_program_ct": float(C.mean()) if len(C) else math.nan,
            "p50_program_ct": float(np.percentile(C, 50)) if len(C) else math.nan,
            "p90_program_ct": float(np.percentile(C, 90)) if len(C) else math.nan,
            "p95_program_ct": float(np.percentile(C, 95)) if len(C) else math.nan,
            "makespan": makespan,
            "program_throughput": len(C) / makespan if makespan > 0 else math.nan,
            "mean_request_ct": float(R.mean()) if len(R) else math.nan,
            "completed_programs": len(C),
            "completed_requests": len(R),
            "warm_admits": self.warm_admits,
            "cold_admits": self.cold_admits,
            "recompute_service": float(self.recompute_service),
            "preempt_events": self.preempt_events,
            "preempted_requests_unique": len(self.preempted_request_keys),
            "preempted_request_ratio": len(self.preempted_request_keys) / max(1, len(R)),
            "memory_keep_events": self.memory_keep_events,
            "memory_delete_events": self.memory_delete_events,
            "evictions": self.evictions,
            "deleted_from_memory_ratio": self.memory_delete_events / delete_den if delete_den else 0.0,
            "memory_violations": self.memory_violations,
            "decision_wall_sec": self.decision_wall_sec,
            "decision_cpu_sec": self.decision_cpu_sec,
            "decision_calls": self.decision_calls,
            "mean_decision_ms": 1000.0 * self.decision_wall_sec / max(1, self.decision_calls),
            "p95_decision_ms": 1000.0 * float(np.percentile(self.decision_samples, 95)) if self.decision_samples else 0.0,
            "memory_utilization": self.memory_area / max(1e-12, self.M * self.t),
            "mean_active": self.active_area / max(1e-12, self.t),
        }
        for step in (0.001, 0.010, 0.050):
            label = f"{int(round(1000 * step))}ms"
            adj = C + C_dec / step
            out[f"mean_program_ct_e2e_{label}"] = float(adj.mean()) if len(adj) else math.nan
            out[f"p90_program_ct_e2e_{label}"] = float(np.percentile(adj, 90)) if len(adj) else math.nan
            out[f"makespan_e2e_{label}"] = float(adj.max()) if len(adj) else math.nan
        out.update(self.policy.diagnostics())
        return out


# ---------------------------------------------------------------------------
# Experiments and reporting
# ---------------------------------------------------------------------------

POLICY_FACTORIES = {
    "commit_many": CommitManyPolicy,
    "least_rounds": LeastRoundsPolicy,
    "size": SizePolicy,
    "random": RandomPolicy,
    "fifo": FIFOPolicy,
    "bayes_grade": BayesianGradePolicy,
    "bayes_grade_warm_marginal": BayesianGradeWarmMarginalPolicy,
    "current_density": CurrentDensityPolicy,
    "dual_descent_current": DualDescentCurrentPolicy,
    "bayes_dual_price": BayesDualPricePolicy,
    "mu_two_price": MuTwoPricePolicy,
    "area_knapsack": AreaKnapsackPolicy,
    "lp_update_long": FullLPUpdatePolicy,
    "lp_solve_once": SolveOnceLPIndexPolicy,
    "lp_selective": SelectiveLPUpdatePolicy,
    "lp_infrequent": InfrequentResolvingPolicy,
    "lp_dual_descent": OnlineLPDualDescentPolicy,
    "mpc_short": ShortHorizonMPCPolicy,
    "dual_threshold": DualThresholdPolicy,
}

CORE_POLICIES = [
    "commit_many",
    "least_rounds",
    "size",
    "random",
    "fifo",
    "current_density",
    "bayes_grade",
    "dual_descent_current",
    "bayes_dual_price",
    "mu_two_price",
    "lp_dual_descent",
    "lp_selective",
]

ALL_POLICIES = list(POLICY_FACTORIES.keys())


def warm_solver() -> None:
    linprog([-1.0, -2.0], A_ub=[[1.0, 1.0]], b_ub=[1.0], bounds=(0.0, 1.0), method="highs")


def parse_policy_set(value: str) -> List[str]:
    if value == "core":
        return CORE_POLICIES
    if value == "all":
        return ALL_POLICIES
    if value == "lp":
        return ["current_density", "bayes_grade", "area_knapsack", "lp_update_long", "lp_solve_once", "lp_selective", "lp_infrequent", "lp_dual_descent", "mpc_short", "dual_threshold"]
    names = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in names if x not in POLICY_FACTORIES]
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    return names


def preset_config(preset: str) -> Dict:
    if preset == "smoke":
        return dict(n=18, reps=1, profiles=["balanced"], mixes=["mixed_long_heavy"], cvs=[0.35], mem_factors=[1], seed0=20260624)
    if preset == "quick":
        return dict(
            n=80,
            n_by_factor={5: 170, 10: 320, 15: 480, 20: 640},
            reps=1,
            profiles=["balanced"],
            mixes=["quick10"],
            cvs=[0.35],
            mem_factors=[5, 10, 15, 20],
            seed0=20260624,
        )
    if preset == "standard":
        return dict(n=80, reps=2, profiles=["prefill_heavy", "balanced", "decode_heavy"], mixes=["mixed_equal", "mixed_long_heavy"], cvs=[0.35, 0.60], mem_factors=[1, 2, 4], seed0=20260624)
    if preset == "stress":
        return dict(n=150, reps=2, profiles=["balanced", "decode_heavy"], mixes=["mixed_long_heavy", "mixed_very_long"], cvs=[0.35], mem_factors=[1, 2, 4], seed0=20260624)
    raise ValueError(preset)


def run_experiment(preset: str, out_dir: str, policy_names: Sequence[str]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    warm_solver()
    cfg = preset_config(preset)
    profiles = build_profiles()
    rows: List[Dict] = []
    for mix in cfg["mixes"]:
        for profile in cfg["profiles"]:
            p_mean, d_mean = profiles[profile]
            for cv in cfg["cvs"]:
                for rep in range(cfg["reps"]):
                    seed = cfg["seed0"] + rep + 10_000 * cfg["profiles"].index(profile) + 100_000 * cfg["mixes"].index(mix) + int(1000 * cv)
                    for f in cfg["mem_factors"]:
                        n = int(cfg.get("n_by_factor", {}).get(f, cfg["n"]))
                        scen = Scenario(
                            name=f"{profile}_{mix}_cv{cv}_f{f}",
                            n=n,
                            p_mean=p_mean,
                            d_mean=d_mean,
                            token_cv=cv,
                            prior_mix=mix,
                            seed=seed,
                        )
                        programs, qmax, total_requests = generate_instance(scen)
                        M = int(math.ceil(f * qmax))
                        for pname in policy_names:
                            clear_value_caches()
                            policy = POLICY_FACTORIES[pname]()
                            sim = Simulator(scen, programs, M, policy)
                            t0 = time.perf_counter()
                            metrics = sim.run()
                            runtime = time.perf_counter() - t0
                            row = dict(metrics)
                            row.update(
                                preset=preset,
                                n=n,
                                rep=rep,
                                seed=seed,
                                profile=profile,
                                prior_mix=mix,
                                token_cv=cv,
                                memory_factor=f,
                                p_mean=p_mean,
                                d_mean=d_mean,
                                M=M,
                                qmax_obs=qmax,
                                total_requests_instance=total_requests,
                                mean_requests_per_program=total_requests / max(1, n),
                                runtime_sec=runtime,
                            )
                            rows.append(row)
                            print(
                                f"{preset} {profile}/{mix} cv={cv} rep={rep} f={f} {pname}: "
                                f"mean={metrics['mean_program_ct']:.2f} p90={metrics['p90_program_ct']:.2f} "
                                f"preempt={metrics['preempt_events']} delete={metrics['deleted_from_memory_ratio']:.3f}",
                                flush=True,
                            )
    raw = pd.DataFrame(rows)
    raw_path = os.path.join(out_dir, "raw_results.csv")
    raw.to_csv(raw_path, index=False)
    write_summaries(raw, out_dir)
    return {
        "raw": raw_path,
        "paired": os.path.join(out_dir, "paired_vs_commit_many.csv"),
        "summary": os.path.join(out_dir, "summary.csv"),
        "ranking": os.path.join(out_dir, "ranking.csv"),
    }


def write_summaries(raw: pd.DataFrame, out_dir: str) -> None:
    key = ["preset", "n", "rep", "seed", "profile", "prior_mix", "token_cv", "memory_factor"]
    metrics = [
        "mean_program_ct",
        "p90_program_ct",
        "p95_program_ct",
        "makespan",
        "program_throughput",
        "mean_program_ct_e2e_10ms",
        "recompute_service",
        "decision_wall_sec",
    ]
    base = raw[raw.policy == "commit_many"][key + metrics].rename(columns={m: f"{m}_base" for m in metrics})
    paired = raw.merge(base, on=key, how="left")
    for m in ["mean_program_ct", "p90_program_ct", "p95_program_ct", "makespan", "mean_program_ct_e2e_10ms", "recompute_service", "decision_wall_sec"]:
        paired[f"{m}_impr_pct"] = 100.0 * (paired[f"{m}_base"] - paired[m]) / paired[f"{m}_base"]
    paired["program_throughput_impr_pct"] = 100.0 * (paired["program_throughput"] / paired["program_throughput_base"] - 1.0)
    paired.to_csv(os.path.join(out_dir, "paired_vs_commit_many.csv"), index=False)

    agg_cols = dict(
        mean_program_ct=("mean_program_ct", "mean"),
        p90_program_ct=("p90_program_ct", "mean"),
        makespan=("makespan", "mean"),
        throughput=("program_throughput", "mean"),
        mean_e2e10=("mean_program_ct_e2e_10ms", "mean"),
        mean_program_impr=("mean_program_ct_impr_pct", "mean"),
        p90_program_impr=("p90_program_ct_impr_pct", "mean"),
        makespan_impr=("makespan_impr_pct", "mean"),
        throughput_impr=("program_throughput_impr_pct", "mean"),
        recompute_service=("recompute_service", "mean"),
        recompute_impr=("recompute_service_impr_pct", "mean"),
        mean_active=("mean_active", "mean"),
        mean_requests_per_program=("mean_requests_per_program", "mean"),
        preempt_events=("preempt_events", "mean"),
        preempted_request_ratio=("preempted_request_ratio", "mean"),
        deleted_from_memory_ratio=("deleted_from_memory_ratio", "mean"),
        evictions=("evictions", "mean"),
        memory_violations=("memory_violations", "sum"),
        decision_wall_sec=("decision_wall_sec", "mean"),
        mean_decision_ms=("mean_decision_ms", "mean"),
        runtime_sec=("runtime_sec", "mean"),
    )
    summary = paired.groupby("policy", as_index=False).agg(**agg_cols)
    summary["composite"] = 0.5 * summary["mean_program_impr"] + 0.25 * summary["p90_program_impr"] + 0.25 * summary["makespan_impr"]
    summary = summary.sort_values(["composite", "mean_program_impr"], ascending=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    by_mem = paired.groupby(["memory_factor", "policy"], as_index=False).agg(
        mean_program_impr=("mean_program_ct_impr_pct", "mean"),
        p90_program_impr=("p90_program_ct_impr_pct", "mean"),
        makespan_impr=("makespan_impr_pct", "mean"),
        throughput_impr=("program_throughput_impr_pct", "mean"),
        mean_active=("mean_active", "mean"),
        mean_requests_per_program=("mean_requests_per_program", "mean"),
        preempt_events=("preempt_events", "mean"),
        deleted_from_memory_ratio=("deleted_from_memory_ratio", "mean"),
    )
    by_mem.to_csv(os.path.join(out_dir, "summary_by_memory.csv"), index=False)

    rank = summary.copy()
    rank["rank_mean_ct"] = rank["mean_program_ct"].rank(method="average")
    rank["rank_e2e10"] = rank["mean_e2e10"].rank(method="average")
    rank = rank.sort_values(["rank_mean_ct", "rank_e2e10"])
    rank.to_csv(os.path.join(out_dir, "ranking.csv"), index=False)
    write_latex_table(summary, os.path.join(out_dir, "summary_table.tex"))
    write_plots(summary, by_mem, out_dir)


def write_latex_table(summary: pd.DataFrame, path: str) -> None:
    keep = summary.head(10).copy()
    cols = ["policy", "mean_active", "mean_program_impr", "p90_program_impr", "makespan_impr", "preempt_events", "deleted_from_memory_ratio"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Policy & Avg active & Mean impr. & P90 impr. & Makespan impr. & Preempt events & Delete ratio \\\\\n\\midrule\n")
        for _, r in keep[cols].iterrows():
            polname = r['policy'].replace('_', '\\_')   # (backslash out of f-string expr: Py<3.12 compat)
            f.write(
                f"{polname} & {r['mean_active']:.2f} & {r['mean_program_impr']:.2f} & {r['p90_program_impr']:.2f} & "
                f"{r['makespan_impr']:.2f} & {r['preempt_events']:.1f} & {r['deleted_from_memory_ratio']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_plots(summary: pd.DataFrame, by_mem: pd.DataFrame, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    plot_policies = [p for p in ["commit_many", "current_density", "bayes_grade", "dual_descent_current", "bayes_dual_price", "lp_dual_descent", "lp_selective", "mu_two_price"] if p in set(summary.policy)]
    plt.figure(figsize=(8.0, 4.8))
    for pname in plot_policies:
        part = by_mem[by_mem.policy == pname].sort_values("memory_factor")
        if len(part):
            plt.plot(part.memory_factor, part.mean_program_impr, marker="o", label=pname)
    plt.axhline(0, linewidth=1)
    plt.xlabel("memory factor M / observed maximum footprint")
    plt.ylabel("mean program CT improvement vs commit_many (%)")
    plt.title("Same-setting policy comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "mean_program_improvement.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8.0, 4.8))
    for pname in plot_policies:
        part = by_mem[by_mem.policy == pname].sort_values("memory_factor")
        if len(part):
            plt.plot(part.memory_factor, part.deleted_from_memory_ratio, marker="o", label=pname)
    plt.xlabel("memory factor M / observed maximum footprint")
    plt.ylabel("deleted-from-memory ratio")
    plt.title("Prefix deletion ratio")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "delete_ratio.png"), dpi=160)
    plt.close()

    top = summary.sort_values("mean_program_ct").head(12)
    plt.figure(figsize=(9.0, 4.8))
    x = np.arange(len(top))
    plt.bar(x, top.mean_decision_ms)
    plt.xticks(x, top.policy, rotation=45, ha="right")
    plt.ylabel("mean decision time (ms)")
    plt.title("Scheduler overhead")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "decision_overhead.png"), dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "quick", "standard", "stress"], default="quick")
    parser.add_argument("--out", default="results/quick")
    parser.add_argument("--policies", default="core", help="'core', 'all', 'lp', or comma-separated policy names")
    args = parser.parse_args()
    policies = parse_policy_set(args.policies)
    paths = run_experiment(args.preset, args.out, policies)
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
