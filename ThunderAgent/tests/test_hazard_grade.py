"""Tests for HazardGradePolicy (online port of hazard_grade_knapsack).

Run:  cd ThunderAgent && python3 -m pytest tests/test_hazard_grade.py -q
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ThunderAgent.program import Program, ProgramStatus, ProgramState
from ThunderAgent.scheduling import make_policy, POLICY_NAMES
from ThunderAgent.scheduling.base import SchedulingPolicy
from ThunderAgent.scheduling.hazard_grade import HazardGradePolicy, exact_cache_knapsack
from ThunderAgent.scheduling.tool_hazard import ROUND_PRIORS, tool_model


NOW = 1_000_000.0


def mk(pid, tokens, step, *, status=ProgramStatus.ACTING, acting_age=None,
       last_p=0, last_c=0, tool_class="", known_decode=0):
    s = Program(program_id=pid)
    s.total_tokens = tokens
    s.step_count = step
    s.status = status
    s.state = ProgramState.ACTIVE
    s.acting_since = (NOW - acting_age) if acting_age is not None else None
    s.last_prompt_tokens = last_p
    s.last_cached_tokens = last_c
    s.tool_class = tool_class
    s.known_decode = known_decode
    return s


def hz(**kw):
    return HazardGradePolicy(**kw)


# --------------------------------------------------------------------------- #
# factory / registration
# --------------------------------------------------------------------------- #
def test_factory_builds_hazard_grade():
    assert "hazard_grade" in POLICY_NAMES
    p = make_policy("hazard_grade", alpha=0.03, decode_hat=1000, dd_eta0=0.1)
    assert isinstance(p, HazardGradePolicy)
    assert p.name == "hazard_grade"


def test_factory_passes_hazard_kwargs():
    p = make_policy("hazard_grade", alpha=0.03, decode_hat=1000, dd_eta0=0.1,
                    hz_decode_reserve=2048, hz_prior="quick10", hz_horizon_s=5.0)
    assert p.decode_reserve == 2048
    assert p.prior.name == "quick10"
    assert p.horizon_s == 5.0


# --------------------------------------------------------------------------- #
# non-clairvoyance: never uses the realized decode length
# --------------------------------------------------------------------------- #
def test_peak_pad_ignores_known_decode():
    p = hz(decode_reserve=4096)
    a = mk("a", 5000, 3, known_decode=0)
    b = mk("b", 5000, 3, known_decode=50000)
    assert p.peak_pad(a) == 4096
    assert p.peak_pad(b) == 4096  # oracle must not change the reservation


def test_source_does_not_read_known_decode():
    # Precise non-clairvoyance check: no attribute ACCESS to `.known_decode`
    # anywhere in the module (comments/docstrings that merely mention it are fine).
    import ast
    import ThunderAgent.scheduling.hazard_grade as mod
    tree = ast.parse(open(mod.__file__).read())
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "known_decode"]
    assert not reads, "hazard_grade must stay non-clairvoyant (no .known_decode reads)"


# --------------------------------------------------------------------------- #
# admission grade (sort_key): short current+remaining work first
# --------------------------------------------------------------------------- #
def test_admission_prefers_smaller_footprint():
    p = hz()
    small = mk("s", 5000, 3)
    big = mk("b", 40000, 3)
    assert p.sort_key(small) > p.sort_key(big)


def test_admission_prefers_fewer_remaining_rounds():
    p = hz()
    early = mk("e", 8000, 1)    # many rounds left
    late = mk("l", 8000, 12)    # nearly done
    assert p.sort_key(late) > p.sort_key(early)


def test_admission_warm_beats_cold():
    p = hz()
    cold = mk("c", 20000, 4, last_p=20000, last_c=0)       # nothing cached
    warm = mk("w", 20000, 4, last_p=20000, last_c=18000)   # mostly cached
    assert p.sort_key(warm) > p.sort_key(cold)


def test_admits_always_true_no_price_gate():
    p = hz()
    assert p.admits(mk("x", 999999, 20)) is True


# --------------------------------------------------------------------------- #
# hazard cache value
# --------------------------------------------------------------------------- #
def test_cache_value_old_tool_beats_young():
    p = hz()
    old = mk("o", 8000, 5, acting_age=60.0, tool_class="long")
    young = mk("y", 8000, 5, acting_age=1.0, tool_class="long")
    assert p.cache_value(old, NOW) > p.cache_value(young, NOW)


def test_cache_value_terminal_bonus():
    p = hz(completion_bonus=1.5)
    likely_terminal = mk("t", 8000, 20, acting_age=5.0, tool_class="fast")
    early = mk("e", 8000, 1, acting_age=5.0, tool_class="fast")
    assert p.cache_value(likely_terminal, NOW) > p.cache_value(early, NOW)


def test_evict_key_is_cache_value_ascending():
    # argmin(evict_key) must drop the LOWEST-value prefix first.
    # evict_key() reads the wall clock internally (no `now` param in the base
    # interface), so build acting_since off real time.time() here.
    p = hz()
    rnow = time.time()
    low = Program(program_id="low")
    low.total_tokens, low.step_count = 2000, 2
    low.status, low.state = ProgramStatus.ACTING, ProgramState.ACTIVE
    low.acting_since, low.tool_class = rnow - 0.5, "long"   # young long tool, small
    high = Program(program_id="high")
    high.total_tokens, high.step_count = 30000, 18
    high.status, high.state = ProgramStatus.ACTING, ProgramState.ACTIVE
    high.acting_since, high.tool_class = rnow - 2.0, "fast"  # big, about to return, terminal
    assert p.evict_key(low) < p.evict_key(high)


# --------------------------------------------------------------------------- #
# exact 0/1 knapsack
# --------------------------------------------------------------------------- #
def test_knapsack_picks_max_value_subset():
    # capacity 5 blocks; items (id, blocks, value)
    items = [("A", 4, 40.0), ("B", 3, 25.0), ("C", 2, 24.0)]
    chosen = exact_cache_knapsack(items, 5)
    assert chosen == {"B", "C"}      # 49 > 40 (A alone) and fits (5 blocks)


def test_knapsack_keeps_all_when_fits():
    items = [("A", 2, 5.0), ("B", 1, 5.0)]
    assert exact_cache_knapsack(items, 10) == {"A", "B"}


def test_knapsack_empty_on_zero_capacity():
    assert exact_cache_knapsack([("A", 2, 5.0)], 0) == set()


# --------------------------------------------------------------------------- #
# select_cache_victims (set-level eviction) with a fake backend
# --------------------------------------------------------------------------- #
class FakeCacheConfig:
    def __init__(self, block_size, capacity):
        self.block_size = block_size
        self.total_tokens_capacity = capacity


class FakeBackend:
    def __init__(self, block_size, capacity, remaining):
        self.cache_config = FakeCacheConfig(block_size, capacity)
        self._remaining = remaining

    def remaining_capacity(self):
        return self._remaining


def test_select_victims_none_without_cache_config():
    p = hz()
    b = FakeBackend(16, 100000, -5000)
    b.cache_config = None
    assert p.select_cache_victims(b, {}, NOW) is None


def test_select_victims_empty_when_within_capacity():
    p = hz()
    b = FakeBackend(16, 100000, 5000)  # positive remaining
    progs = {"a": mk("a", 8000, 3, acting_age=1.0)}
    assert p.select_cache_victims(b, progs, NOW) == []


def test_select_victims_frees_overflow_and_keeps_valuable():
    p = hz()
    bs = 16
    # over capacity by ~8000 tokens -> must shed >= ceil(8000/16)=500 blocks
    b = FakeBackend(bs, 100000, -8000)
    progs = {
        # valuable: big + about-to-return + likely terminal -> should be RETAINED
        "keep": mk("keep", 16000, 18, acting_age=40.0, tool_class="fast"),
        # low value: young long tool -> should be a VICTIM
        "drop1": mk("drop1", 8000, 2, acting_age=0.5, tool_class="long"),
        "drop2": mk("drop2", 8000, 2, acting_age=0.5, tool_class="long"),
    }
    victims = p.select_cache_victims(b, progs, NOW)
    assert victims is not None
    freed_tokens = sum(progs[v].total_tokens for v in victims)
    assert freed_tokens >= 8000                    # sheds at least the overflow
    assert "keep" not in victims                    # the valuable prefix survives


def test_select_victims_only_targets_acting():
    p = hz()
    b = FakeBackend(16, 100000, -5000)
    progs = {
        "reasoning": mk("reasoning", 30000, 3, status=ProgramStatus.REASONING),
    }
    # no ACTING candidates -> defer to Router legacy loop (None)
    assert p.select_cache_victims(b, progs, NOW) is None


# --------------------------------------------------------------------------- #
# backward compatibility: existing policies unchanged
# --------------------------------------------------------------------------- #
def test_other_policies_defer_eviction():
    b = FakeBackend(16, 100000, -5000)
    progs = {"a": mk("a", 8000, 3, acting_age=1.0)}
    for name in ("size", "density", "dual_descent", "fidelity"):
        pol = make_policy(name, alpha=0.03, decode_hat=1000, dd_eta0=0.1)
        assert pol.select_cache_victims(b, progs, NOW) is None
        assert pol.cache_value(progs["a"], NOW) == 0.0


# --------------------------------------------------------------------------- #
# distributions
# --------------------------------------------------------------------------- #
def test_round_prior_probabilities_valid():
    pr = ROUND_PRIORS["swebench9"]
    assert abs(sum(pr.pmf) - 1.0) < 1e-9
    for stage in (1, 3, 9, 20):
        tp = pr.terminal_probability(stage)
        assert 0.0 <= tp <= 1.0
    # remaining rounds shrink as the program advances
    assert pr.remaining_mean(1) > pr.remaining_mean(9)
    # terminal probability rises as we go deeper
    assert pr.terminal_probability(20) >= pr.terminal_probability(1)


def test_tool_residual_prob_properties():
    tm = tool_model("medium")  # lognormal mean 9s
    # (1) valid probability, and monotone increasing in the horizon
    assert 0.0 <= tm.residual_return_prob(5.0, 5.0) <= tm.residual_return_prob(5.0, 20.0) <= 1.0
    # (2) a tool aged toward its mean is likelier to return within one mean-horizon
    #     than a just-started one (rising-hazard region). NOTE: lognormal hazard is
    #     non-monotone -- far in the tail the return prob falls again -- so this is
    #     deliberately compared near the mean, matching the reference model.
    young = tm.residual_return_prob(0.5, 9.0)
    near_mean = tm.residual_return_prob(9.0, 9.0)
    assert near_mean > young


# --------------------------------------------------------------------------- #
# end-to-end: Router._pause_until_safe with a real BackendState + fake metrics
# --------------------------------------------------------------------------- #
class _FakeCacheConfig:
    block_size = 16
    @property
    def total_tokens_capacity(self):
        return 16 * 2000  # 32000 tokens


class _FakeMetrics:
    healthy = True
    cache_config = _FakeCacheConfig()
    def calculate_shared_tokens(self, reasoning_tokens):
        return 0


def _build_router(policy):
    from ThunderAgent.scheduler.router import MultiBackendRouter
    r = MultiBackendRouter("http://fake:8000", scheduling_enabled=True, policy=policy)
    b = r.backends["http://fake:8000"]
    b.metrics_client = _FakeMetrics()
    b.shared_tokens = 0
    rnow = time.time()
    specs = [
        ("keep", 16000, 18, 40.0, "fast"),   # valuable: big, about-to-return, terminal
        ("d1", 9000, 2, 0.5, "long"),
        ("d2", 9000, 2, 0.5, "long"),
        ("d3", 8000, 2, 0.5, "long"),
        ("d4", 8000, 2, 0.5, "long"),
    ]
    for pid, tok, step, age, tc in specs:
        s = Program(program_id=pid)
        s.total_tokens, s.step_count = tok, step
        s.status, s.state = ProgramStatus.ACTING, ProgramState.ACTIVE
        s.acting_since, s.tool_class = rnow - age, tc
        s.backend_url = "http://fake:8000"
        r.programs[pid] = s
        b.register_program(pid, s)
    return r, b


def test_router_pause_until_safe_hazard_grade():
    import asyncio
    r, b = _build_router("hazard_grade")
    assert b.remaining_capacity() < 0                  # starts oversubscribed
    asyncio.run(r._pause_until_safe(b))
    assert b.remaining_capacity() >= 0                  # recovered within capacity
    assert r.programs["keep"].state == ProgramState.ACTIVE  # valuable prefix retained
    paused = [p for p in r.programs.values() if p.state == ProgramState.PAUSED]
    assert len(paused) >= 1


def test_router_pause_until_safe_legacy_still_works():
    import asyncio
    r, b = _build_router("size")                        # unchanged legacy argmin path
    assert b.remaining_capacity() < 0
    asyncio.run(r._pause_until_safe(b))
    assert b.remaining_capacity() >= 0                  # legacy eviction still recovers


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
