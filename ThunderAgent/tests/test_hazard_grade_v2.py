"""Tests for HazardGradeV2Policy: the 2026-08-23 simulator port.

Each test pins one of the five decisions that differ from `hazard_grade`, so a
future edit that silently reverts to v1 behaviour fails here.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ThunderAgent.scheduling import make_policy, POLICY_NAMES
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
def v1():
    return make_policy("hazard_grade", **KW)


@pytest.fixture
def v2():
    return make_policy("hazard_grade_v2", **KW)


def test_policy_is_registered():
    assert "hazard_grade_v2" in POLICY_NAMES
    assert make_policy("hazard_grade_v2", **KW).name == "hazard_grade_v2"


def test_max_batch_knob_reaches_the_policy():
    assert make_policy("hazard_grade_v2", hz_max_batch=7, **KW).max_batch == 7


# ── change 1 + 4: eviction is value density, and density is not degenerate ─────
def test_evict_key_is_value_density_and_grows_with_size(v2):
    small, big = prog(2_000), prog(60_000)
    d_small, d_big = v2.evict_key(small)[0], v2.evict_key(big)[0]
    assert d_big > d_small, "a larger prefix must cost more per block to lose"
    # and it is a density, not the raw value: raw value is ~30x larger for big
    assert v2.cache_value(big, 0.0) / v2.cache_value(small, 0.0) > 10
    assert d_big / d_small < 2


def test_v1_evict_key_is_raw_value_not_density(v1):
    """Pins the difference itself: v1 ranks by raw value, so it is ~linear."""
    r = v1.evict_key(prog(60_000))[0] / v1.evict_key(prog(2_000))[0]
    assert r > 20


def test_linear_cost_would_make_density_constant(v2):
    """Why change 4 exists: with no context penalty the density degenerates."""
    v2.prefill_context_penalty = 0.0
    assert v2.evict_key(prog(2_000))[0] == pytest.approx(v2.evict_key(prog(60_000))[0])


# ── change 2: the decode reservation counts toward the admission footprint ─────
def test_admission_grade_counts_the_reservation(v2):
    s = prog(1_000)
    assert v2._admit_blocks(s) == v2._blocks(1_000 + v2.decode_reserve, v2.block_size_hint)
    assert v2._admit_blocks(s) > v2._blocks(1_000, v2.block_size_hint)


def test_reservation_penalises_short_programs_relative_to_v1(v1, v2):
    """Short programs are over-ranked by v1 because it ignores the 1024-token
    reservation it then reserves anyway."""
    short, long_ = prog(1_000), prog(40_000)
    assert v1._priority(short) / v1._priority(long_) > v2._priority(short) / v2._priority(long_)


# ── change 3: warm is a binary flag over the incremental prompt ────────────────
def test_warm_program_is_scored_on_its_incremental_prompt(v2):
    warm = prog(50_000, last_prompt=50_000, last_cached=45_000)
    cold = prog(50_000, last_prompt=50_000, last_cached=0)
    assert v2._uncached_tokens(warm) == 5_000
    assert v2._uncached_tokens(cold) == 50_000
    assert v2._priority(warm) > v2._priority(cold)


def test_warm_flag_is_binary_at_half(v2):
    just_warm = prog(10_000, last_prompt=10_000, last_cached=5_000)
    just_cold = prog(10_000, last_prompt=10_000, last_cached=4_999)
    assert v2._uncached_tokens(just_warm) == 5_000
    assert v2._uncached_tokens(just_cold) == 10_000


# ── change 5: batch target and the anti-deadlock fallback ──────────────────────
class _Backend:
    def __init__(self, n):
        self.active_program_count = n


def test_admits_everything_before_the_first_epoch(v2):
    assert v2.admits(prog(1_000)) is True


def test_batch_target_caps_admissions_per_tick(v2):
    v2.max_batch = 5
    v2.on_epoch({"b": _Backend(3)}, {})
    assert [v2.admits(prog(1_000)) for _ in range(4)] == [True, True, False, False]


def test_full_backend_admits_nothing(v2):
    v2.max_batch = 4
    v2.on_epoch({"b": _Backend(4)}, {})
    assert v2.admits(prog(1_000)) is False


def test_idle_backend_force_admits_exactly_one(v2):
    """A program larger than the budget must not deadlock an empty backend."""
    v2.max_batch = 0
    v2.on_epoch({"b": _Backend(0)}, {"w": prog(1_000)})
    assert v2.admits(prog(1_000)) is True    # the forced one
    assert v2.admits(prog(1_000)) is False   # and only one


def test_no_force_admit_when_nothing_is_waiting(v2):
    v2.max_batch = 0
    v2.on_epoch({"b": _Backend(0)}, {})
    assert v2.admits(prog(1_000)) is False


def test_slots_counted_across_backends(v2):
    v2.max_batch = 10
    v2.on_epoch({"a": _Backend(4), "b": _Backend(5)}, {})
    assert [v2.admits(prog(1_000)) for _ in range(2)] == [True, False]


# ── inherited behaviour that must NOT change ──────────────────────────────────
def test_peak_pad_is_still_the_distributional_reservation(v1, v2):
    s = prog(5_000)
    s.known_decode = 9_999            # must be ignored: policy stays non-clairvoyant
    assert v2.peak_pad(s) == v1.peak_pad(s) == KW["hz_decode_reserve"]
