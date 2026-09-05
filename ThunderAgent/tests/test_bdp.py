"""Tests for BDPPolicy: the online port of the calibrated simulator's `bdp`.

BDP is Bayesian SERPT plus one KV dual price, with no fitted policy parameter.
These pin the pieces that make it that, so a later edit cannot quietly turn it
into a heuristic.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ThunderAgent.scheduling import make_policy, POLICY_NAMES
from ThunderAgent.scheduling.bdp import BDPPolicy
from ThunderAgent.scheduling.base import SchedulingPolicy
from ThunderAgent.program import Program, ProgramStatus, ProgramState

KW = dict(alpha=0.03, decode_hat=1000.0, dd_eta0=0.1, hz_decode_mean=300.0,
          hz_prompt_mean=1600.0, hz_decode_reserve=1024, hz_prior="coder16")


def prog(tokens, step=3, observed=(), last_prompt=None, last_cached=0):
    s = Program(program_id=f"p{tokens}_{step}")
    s.total_tokens = tokens
    s.step_count = step
    s.status = ProgramStatus.ACTING
    s.state = ProgramState.ACTIVE
    s.last_prompt_tokens = tokens if last_prompt is None else last_prompt
    s.last_cached_tokens = last_cached
    s.observed_decode = list(observed)
    return s


class _Backend:
    class _CC:
        block_size = 16
        total_tokens_capacity = 735296

    cache_config = _CC()

    def __init__(self, free, active=0):
        self._free = free
        self.active_program_count = active

    def remaining_capacity(self):
        return self._free


@pytest.fixture
def p():
    return make_policy("bdp", **KW)


# ── registration and independence ─────────────────────────────────────────────
def test_registered_and_standalone():
    assert "bdp" in POLICY_NAMES
    assert BDPPolicy.__bases__ == (SchedulingPolicy,)
    assert make_policy("bdp", **KW).name == "bdp"


def test_has_no_fitted_policy_parameter(p):
    """Everything BDP reads is either the calibrated workload model or lambda,
    which it recomputes. There is no tunable weight or bonus."""
    for forbidden in ("completion_bonus", "age_bonus", "safety_fraction",
                      "horizon_s", "marginal_ratio"):
        assert not hasattr(p, forbidden)


def test_is_non_clairvoyant(p):
    """known_decode is the A-arm look-ahead and must never move a decision."""
    a, b = prog(10_000, 5), prog(10_000, 5)
    b.known_decode = 8192
    assert p.sort_key(a) == p.sort_key(b)
    assert p.peak_pad(b) == KW["hz_decode_reserve"]


# ── the posterior over decode scale ───────────────────────────────────────────
def test_no_history_gives_the_prior_mean(p):
    assert p._posterior_decode(prog(10_000, observed=())) == KW["hz_decode_mean"]


def test_short_turns_shrink_the_posterior_down(p):
    assert p._posterior_decode(prog(10_000, observed=(150,) * 6)) < KW["hz_decode_mean"]


def test_a_rare_long_turn_is_not_read_as_a_heavy_program(p):
    """The mixture assigns a 3600-token turn to its large component rather than
    concluding the whole program generates 10x more from now on."""
    quiet = p._posterior_decode(prog(10_000, observed=(150, 150, 150, 150)))
    spike = p._posterior_decode(prog(10_000, observed=(150, 150, 3600, 150)))
    assert spike > quiet
    assert spike < 2.0 * quiet          # nothing like the 24x the raw mean implies


def test_posterior_is_bounded_by_the_mixture(p):
    """Even an all-long history stays near the prior, because those turns are
    explained by the large component."""
    assert p._posterior_decode(prog(10_000, observed=(3600,) * 8)) < 2.0 * KW["hz_decode_mean"]


# ── the work estimate ─────────────────────────────────────────────────────────
def test_later_turns_have_less_remaining_work(p):
    early, _ = p._estimate(prog(8_000, step=2))
    late, _ = p._estimate(prog(8_000, step=15))
    assert late < early


def test_context_guard_zeroes_future_turns(p):
    """Past the guard the server refuses the next prompt, so no future turns
    remain no matter what the round prior says."""
    _, _ = p._estimate(prog(1_000, step=2))
    r_near, f_near = p._estimate(prog(int(p.context_limit) + 5_000, step=2))
    r_far, f_far = p._estimate(prog(int(p.context_limit) + 5_000, step=18))
    assert r_near == pytest.approx(r_far)      # both have zero future turns


def test_footprint_grows_with_expected_future_turns(p):
    _, f_early = p._estimate(prog(8_000, step=2))
    _, f_late = p._estimate(prog(8_000, step=15))
    assert f_early > f_late


# ── the dual price ────────────────────────────────────────────────────────────
def test_lambda_is_zero_when_kv_is_plentiful(p):
    p.on_epoch({"b": _Backend(free=10 ** 9)}, {"a": prog(5_000), "b2": prog(6_000)})
    assert p.lam == 0.0


def test_lambda_is_positive_under_pressure(p):
    waiting = {f"p{i}": prog(30_000, step=3) for i in range(40)}
    p.on_epoch({"b": _Backend(free=50_000)}, waiting)
    assert p.lam > 0.0
    assert p.binding_updates == 1


def test_lambda_rises_as_free_kv_shrinks(p):
    waiting = {f"p{i}": prog(30_000, step=3) for i in range(40)}
    p.on_epoch({"b": _Backend(free=400_000)}, waiting)
    loose = p.lam
    p.on_epoch({"b": _Backend(free=40_000)}, waiting)
    assert p.lam >= loose


def test_price_gate_rejects_negative_margin(p):
    waiting = {f"p{i}": prog(30_000, step=3) for i in range(60)}
    # active > 0, so the idle-backend fallback is not armed and the gate is the
    # only thing deciding
    p.on_epoch({"b": _Backend(free=20_000, active=5)}, waiting)
    assert p.lam > 0.0
    # a program whose value cannot pay for its footprint at this price
    assert p.admits(prog(120_000, step=2)) is False
    assert p._margin(prog(120_000, step=2)) < 0.0


def test_the_idle_fallback_overrides_the_gate(p):
    """With nothing active, one program is admitted even at a negative margin --
    otherwise an oversized program could deadlock an empty backend."""
    waiting = {f"p{i}": prog(30_000, step=3) for i in range(60)}
    p.on_epoch({"b": _Backend(free=20_000, active=0)}, waiting)
    assert p._margin(prog(120_000, step=2)) < 0.0
    assert p.admits(prog(120_000, step=2)) is True     # the forced one
    assert p.admits(prog(120_000, step=2)) is False    # and only one


def test_empty_waiting_pool_clears_the_price(p):
    waiting = {f"p{i}": prog(30_000) for i in range(40)}
    p.on_epoch({"b": _Backend(free=10_000)}, waiting)
    assert p.lam > 0.0
    p.on_epoch({"b": _Backend(free=10_000)}, {})
    assert p.lam == 0.0


# ── admission ─────────────────────────────────────────────────────────────────
def test_batch_cap_limits_admissions_per_tick(p):
    p.max_batch = 5
    p.on_epoch({"b": _Backend(free=10 ** 9, active=3)}, {"a": prog(1_000)})
    assert [p.admits(prog(1_000)) for _ in range(4)] == [True, True, False, False]


def test_idle_backend_force_admits_one(p):
    p.max_batch = 0
    p.on_epoch({"b": _Backend(free=10 ** 9, active=0)}, {"a": prog(1_000)})
    assert p.admits(prog(1_000)) is True
    assert p.admits(prog(1_000)) is False


def test_diagnostics_report_the_price(p):
    p.on_epoch({"b": _Backend(free=10_000)},
               {f"p{i}": prog(30_000) for i in range(40)})
    d = p.diagnostics()
    assert d["bdp_lambda_last"] > 0 and d["bdp_binding_fraction"] == 1.0


# ── what BDP deliberately does not do ─────────────────────────────────────────
def test_no_router_level_cache_retention(p):
    """Reuse comes from the engine's prefix cache; the simulator's ablation
    found router retention did not help, so there is no knapsack here."""
    assert p.select_cache_victims(_Backend(free=-1000), {"a": prog(5_000)}, 0.0) is None
    assert p.cache_value(prog(5_000), 0.0) == 0.0
