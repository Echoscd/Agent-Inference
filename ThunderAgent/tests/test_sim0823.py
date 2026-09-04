"""Tests for Sim0823Policy: the standalone 2026-08-23 simulator policy.

These pin the algorithm's own behaviour, stated from the simulator, without
reference to hazard_grade. A few tests additionally assert that this policy is
independent of hazard_grade, so a future refactor cannot quietly couple them.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ThunderAgent.scheduling import make_policy, POLICY_NAMES
from ThunderAgent.scheduling.sim0823 import Sim0823Policy, block_knapsack
from ThunderAgent.scheduling.base import SchedulingPolicy
from ThunderAgent.program import Program, ProgramStatus, ProgramState

KW = dict(alpha=0.03, decode_hat=1000.0, dd_eta0=0.1, hz_decode_mean=300.0,
          hz_prompt_mean=1600.0, hz_decode_reserve=1024, hz_prior="coder16")


def prog(tokens, step=3, last_prompt=None, last_cached=0, status=ProgramStatus.ACTING):
    s = Program(program_id=f"p{tokens}_{step}")
    s.total_tokens = tokens
    s.step_count = step
    s.status = status
    s.state = ProgramState.ACTIVE
    s.acting_since = None
    s.last_prompt_tokens = tokens if last_prompt is None else last_prompt
    s.last_cached_tokens = last_cached
    return s


@pytest.fixture
def p():
    return make_policy("sim0823", **KW)


class _Backend:
    def __init__(self, n):
        self.active_program_count = n


# ── registration and independence ─────────────────────────────────────────────
def test_registered_and_named():
    assert "sim0823" in POLICY_NAMES
    assert make_policy("sim0823", **KW).name == "sim0823"


def test_is_a_standalone_policy_not_a_hazard_grade_subclass():
    """The algorithm must not inherit from the July port: an edit there must not
    be able to change this policy's decisions."""
    from ThunderAgent.scheduling.hazard_grade import HazardGradePolicy
    assert Sim0823Policy.__bases__ == (SchedulingPolicy,)
    assert not issubclass(Sim0823Policy, HazardGradePolicy)


def test_max_batch_knob_reaches_the_policy():
    assert make_policy("sim0823", hz_max_batch=7, **KW).max_batch == 7


# ── admission amount: non-clairvoyant ─────────────────────────────────────────
def test_reservation_ignores_the_known_decode_length(p):
    s = prog(5_000)
    s.known_decode = 99_999
    assert p.peak_pad(s) == KW["hz_decode_reserve"]


# ── admission order ───────────────────────────────────────────────────────────
def test_admission_footprint_includes_the_reservation(p):
    s = prog(1_000)
    assert p._admit_blocks(s) == p._blocks(1_000 + p.decode_reserve, p.block_size_hint)


def test_shorter_remaining_work_ranks_higher(p):
    """A program near its last round outranks one with many rounds left."""
    early, late = prog(10_000, step=1), prog(10_000, step=14)
    assert p._priority(late) > p._priority(early)


def test_bigger_footprint_ranks_lower_all_else_equal(p):
    assert p._priority(prog(5_000)) > p._priority(prog(50_000))


def test_warm_prefix_is_scored_on_the_incremental_prompt_only(p):
    warm = prog(50_000, last_prompt=50_000, last_cached=45_000)
    cold = prog(50_000, last_prompt=50_000, last_cached=0)
    assert p._uncached_tokens(warm) == 5_000
    assert p._uncached_tokens(cold) == 50_000
    assert p._priority(warm) > p._priority(cold)


def test_warm_is_a_binary_flag_at_the_threshold(p):
    assert p._uncached_tokens(prog(10_000, last_prompt=10_000, last_cached=5_000)) == 5_000
    assert p._uncached_tokens(prog(10_000, last_prompt=10_000, last_cached=4_999)) == 10_000


# ── admission gate ────────────────────────────────────────────────────────────
def test_admits_freely_before_the_first_tick(p):
    assert p.admits(prog(1_000)) is True


def test_batch_target_caps_admissions_per_tick(p):
    p.max_batch = 5
    p.on_epoch({"b": _Backend(3)}, {})
    assert [p.admits(prog(1_000)) for _ in range(4)] == [True, True, False, False]


def test_slots_are_counted_across_backends(p):
    p.max_batch = 10
    p.on_epoch({"a": _Backend(4), "b": _Backend(5)}, {})
    assert [p.admits(prog(1_000)) for _ in range(2)] == [True, False]


def test_idle_backend_force_admits_exactly_one(p):
    p.max_batch = 0
    p.on_epoch({"b": _Backend(0)}, {"w": prog(1_000)})
    assert p.admits(prog(1_000)) is True
    assert p.admits(prog(1_000)) is False


def test_no_force_admit_when_nothing_waits(p):
    p.max_batch = 0
    p.on_epoch({"b": _Backend(0)}, {})
    assert p.admits(prog(1_000)) is False


# ── retention value and eviction order ────────────────────────────────────────
def test_cold_cost_is_superlinear_in_context(p):
    """Doubling the context more than doubles the recompute cost."""
    assert p._cold_cost(60_000) > 2.0 * p._cold_cost(30_000)


def test_eviction_ranks_by_density_and_favours_large_prefixes(p):
    d_small, d_big = p.evict_key(prog(2_000))[0], p.evict_key(prog(60_000))[0]
    assert d_big > d_small                              # denser -> evicted later
    assert p.cache_value(prog(60_000), 0.0) / p.cache_value(prog(2_000), 0.0) > 10
    assert d_big / d_small < 2                          # but it is a density, not the value


def test_density_would_degenerate_without_the_context_penalty(p):
    """Why the superlinear cost is load-bearing: a linear cost cancels exactly
    against the block count and the eviction order becomes a constant."""
    p.ctx_penalty = 0.0
    assert p.evict_key(prog(2_000))[0] == pytest.approx(p.evict_key(prog(60_000))[0])


def test_terminal_programs_are_worth_more_to_keep(p):
    assert p.cache_value(prog(10_000, step=14), 0.0) > p.cache_value(prog(10_000, step=1), 0.0)


# ── the knapsack ──────────────────────────────────────────────────────────────
def test_knapsack_keeps_everything_when_it_fits():
    assert block_knapsack([("a", 2, 1.0), ("b", 3, 1.0)], 10) == {"a", "b"}


def test_knapsack_maximises_value_not_count():
    """One heavy high-value item beats two light low-value ones."""
    assert block_knapsack([("big", 5, 10.0), ("s1", 2, 1.0), ("s2", 3, 1.0)], 5) == {"big"}


def test_knapsack_respects_capacity():
    chosen = block_knapsack([("a", 4, 5.0), ("b", 4, 5.0), ("c", 4, 5.0)], 9)
    assert sum(4 for _ in chosen) <= 9 and len(chosen) == 2


def test_knapsack_edge_cases():
    assert block_knapsack([], 10) == set()
    assert block_knapsack([("a", 2, 1.0)], 0) == set()
    assert block_knapsack([("a", 20, 1.0)], 10) == set()      # item exceeds capacity
    assert block_knapsack([("a", 2, 0.0)], 10) == set()       # zero value is not kept


def test_greedy_fallback_is_used_for_huge_tables():
    """Above ~5M DP cells the exact solve is skipped; the result must still be
    feasible rather than empty or over capacity."""
    items = [(f"p{i}", 3, float(i + 1)) for i in range(40)]
    chosen = block_knapsack(items, 200_000)
    assert chosen and sum(3 for _ in chosen) <= 200_000


# ── victim selection ──────────────────────────────────────────────────────────
class _CacheCfg:
    block_size = 16


class _FullBackend:
    cache_config = _CacheCfg()

    def __init__(self, overflow):
        self._overflow = overflow

    def remaining_capacity(self):
        return -self._overflow


def test_no_victims_when_within_capacity(p):
    assert list(p.select_cache_victims(_FullBackend(-1), {"a": prog(1_000)}, 0.0)) == []


def test_defers_when_the_backend_has_no_block_info(p):
    class NoCfg:
        cache_config = None
    assert p.select_cache_victims(NoCfg(), {"a": prog(1_000)}, 0.0) is None


def test_only_acting_programs_are_shed(p):
    reasoning = prog(50_000, status=ProgramStatus.REASONING)
    assert p.select_cache_victims(_FullBackend(10_000), {"r": reasoning}, 0.0) is None


def test_victims_free_at_least_the_overflow(p):
    programs = {f"p{i}": prog(10_000) for i in range(6)}
    victims = list(p.select_cache_victims(_FullBackend(20_000), programs, 0.0))
    freed_blocks = sum(p._blocks(programs[v].total_tokens, 16) for v in victims)
    assert freed_blocks >= 20_000 / 16
