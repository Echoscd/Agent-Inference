from __future__ import annotations

from dataclasses import replace
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

from export_setting import build_setting  # noqa: E402
from run_algorithm_experiment import policy_initial_values, policy_parameters  # noqa: E402
from realistic_agentic_sim import (  # noqa: E402
    BDPLikePolicy,
    DEFAULT_POLICIES,
    DECODE,
    ENGINE_WAITING,
    POLICIES,
    TOOL,
    SimConfig,
    Simulator,
    ThunderPolicy,
    WorkEstimate,
    WorkloadConfig,
    generate_workload,
    make_service_model,
    matrix,
)


def test_policy_view_hides_realized_future() -> None:
    cfg = WorkloadConfig("view", n_programs=4, capacity_tokens=8_000, seed=11)
    specs, _ = generate_workload(cfg)
    p = Simulator(cfg, specs, POLICIES["bdp"]()).view().ready[0]
    assert not hasattr(p, "decode_actual")
    assert not hasattr(p, "tool_remaining")
    assert not hasattr(p, "tool_duration_actual")


def test_bdp_like_interface_runs_on_distributional_estimates() -> None:
    class Candidate(BDPLikePolicy):
        name = "candidate"

        def admission_score(self, p, estimate: WorkEstimate, view) -> float:
            return 1.0 / max(1.0, estimate.current_work * estimate.remaining_work)

    cfg = WorkloadConfig("candidate", n_programs=8, capacity_tokens=10_000, seed=13)
    specs, _ = generate_workload(cfg)
    out = Simulator(cfg, specs, Candidate()).run()
    assert out["completed_programs"] == cfg.n_programs
    assert out["memory_violations"] == 0


def test_bdp_updates_one_causal_dual_price() -> None:
    cfg = WorkloadConfig("bdp", n_programs=8, capacity_tokens=10_000, seed=14)
    specs, _ = generate_workload(cfg)
    policy = POLICIES["bdp"]()
    sim = Simulator(cfg, specs, policy)
    view = sim.view()
    estimates = {p.pid: view.estimate_work(p) for p in view.ready}
    policy.update_state(view, estimates)
    densities = sorted(
        (
            1.0 / max(1e-12, e.remaining_work)
            / max(1.0, e.expected_footprint_tokens),
            max(1.0, e.expected_footprint_tokens),
        )
        for e in estimates.values()
    )[::-1]
    free = (view.capacity_blocks - view.active_reserved) * view.block_size
    expected_lambda = 0.0
    if sum(q for _density, q in densities) > free:
        used = 0.0
        for density, q in densities:
            expected_lambda = density
            used += q
            if used >= free:
                break
    assert policy.lam == pytest.approx(expected_lambda)
    first = view.ready[0]
    estimate = estimates[first.pid]
    assert policy.admission_score(first, estimate, view) == pytest.approx(
        1.0 / estimate.remaining_work
        - expected_lambda * estimate.expected_footprint_tokens
    )

    out = sim.run()
    assert out["bdp_updates"] > 0
    assert out["bdp_lambda_last"] >= 0
    assert out["bdp_lambda_max"] >= out["bdp_lambda_mean"] >= 0
    assert 0 <= out["bdp_binding_fraction"] <= 1


def test_bdp_value_is_inverse_posterior_remaining_work() -> None:
    estimate = WorkEstimate(
        uncached_prompt=10.0,
        current_work=20.0,
        remaining_work=50.0,
        expected_footprint_tokens=100.0,
    )
    assert POLICIES["bdp"].bayesian_value(estimate) == pytest.approx(1.0 / 50.0)


def test_work_estimate_updates_observed_program_decode_scale() -> None:
    cfg = matrix("coder_calibration_smoke", 20260826)[0]
    specs, _ = generate_workload(cfg)
    view = Simulator(cfg, specs, POLICIES["bdp"]()).view()
    view = replace(view, decode_large_prob=0.0, decode_small_mean=view.decode_mean)
    program = replace(
        view.ready[0],
        stage=3,
        prefix=100,
        prompt=50,
        observed_decode_lengths=(600, 600),
        remaining_rounds=3.0,
    )
    estimate = view.estimate_work(program)
    precision = 1.0 / (cfg.decode_program_cv ** 2)
    scale = (precision + 1_200.0 / view.decode_mean) / (precision + 2)
    posterior_decode = view.decode_mean * scale
    expected = (
        view.alpha_work * 150.0 + posterior_decode
        + 2.0 * (view.alpha_work * view.prompt_mean + posterior_decode)
    )
    assert estimate.remaining_work == pytest.approx(expected)


def test_decode_mixture_does_not_turn_one_large_round_into_program_scale() -> None:
    cfg = matrix("coder_calibration_smoke", 20260826)[0]
    specs, _ = generate_workload(cfg)
    view = Simulator(cfg, specs, POLICIES["bdp"]()).view()
    program = replace(
        view.ready[0],
        stage=2,
        observed_decode_lengths=(3_600,),
        remaining_rounds=5.0,
    )
    estimate = view.estimate_work(program)
    inferred_decode = estimate.current_work - view.alpha_work * estimate.uncached_prompt
    naive_scale = (
        1.0 / cfg.decode_program_cv ** 2 + 3_600.0 / view.decode_mean
    ) / (1.0 / cfg.decode_program_cv ** 2 + 1.0)
    assert inferred_decode < 0.5 * view.decode_mean * naive_scale


def test_work_estimate_conditions_turns_on_observable_context_guard() -> None:
    cfg = matrix("coder_calibration_smoke", 20260826)[0]
    specs, _ = generate_workload(cfg)
    view = Simulator(cfg, specs, POLICIES["bdp"]()).view()
    program = replace(
        view.ready[0],
        prefix=34_000,
        prompt=500,
        remaining_rounds=10.0,
    )
    limited = view.estimate_work(program)
    unlimited = replace(view, context_limit=None).estimate_work(program)
    assert limited.remaining_work < unlimited.remaining_work
    posterior_decode = view.decode_mean
    future_context = view.prompt_mean + posterior_decode
    expected_future_turns = (
        cfg.context_limit_tokens - (34_000 + 500 + posterior_decode)
        + posterior_decode
    ) / future_context
    expected = (
        limited.current_work
        + expected_future_turns
        * (view.alpha_work * view.prompt_mean + posterior_decode)
    )
    assert limited.remaining_work == pytest.approx(expected)


def test_bdp_reuses_engine_apc_without_router_cache_retention() -> None:
    cfg = WorkloadConfig("bdp_cache", n_programs=3, capacity_tokens=8_000, seed=19)
    specs, _ = generate_workload(cfg)
    policy = POLICIES["bdp"]()
    view = Simulator(cfg, specs, policy).view()
    assert not policy.retain_during_tool()
    policy.lam = 1.0
    epoch = policy.epoch
    policy.update_state(view, {})
    assert policy.lam == 0.0
    assert policy.epoch == epoch + 1


def test_thunder_matches_paper_restore_decay_and_pause_order() -> None:
    cfg = WorkloadConfig("thunder_formula", n_programs=3, capacity_tokens=8_000, seed=15)
    specs, _ = generate_workload(cfg)
    policy = ThunderPolicy()
    sim = Simulator(cfg, specs, policy)
    initial = sim.p[0]
    assert sim.admit_blocks(initial) == sim.blocks(initial.prompt + 100)

    base = sim.view().ready[0]
    reasoning = replace(
        base, pid=10, initial=False, prefix=400, prompt=100, admit_blocks=10
    )
    new = replace(base, pid=11, initial=True, prefix=0, prompt=50, admit_blocks=10)
    acting_small = replace(
        base, pid=12, phase=TOOL, initial=False, prompt=0,
        cache_warm=True, cache_blocks=5, tool_elapsed=1.0,
    )
    acting_large = replace(
        acting_small, pid=13, cache_blocks=10,
    )
    view = replace(
        sim.view(),
        capacity_blocks=30,
        programs=(new, reasoning, acting_small, acting_large),
    )
    plan = policy.plan(view)
    assert policy.acting_weight(acting_small) == pytest.approx(0.5)
    assert plan.admit == [reasoning.pid, new.pid]
    assert plan.keep_cache == {acting_large.pid}

    active = replace(
        reasoning, pid=14, phase=DECODE, reserve_blocks=12,
        current_context=192.0, admit_blocks=0,
    )
    overflow = replace(view, capacity_blocks=10, programs=(active,))
    assert policy.plan(overflow).mark_after_turn == {active.pid}


@pytest.mark.parametrize("policy_name", DEFAULT_POLICIES)
def test_registered_policies_complete_without_memory_violation(policy_name: str) -> None:
    cfg = WorkloadConfig("smoke", n_programs=12, capacity_tokens=12_000, seed=17)
    specs, _ = generate_workload(cfg)
    out = Simulator(cfg, specs, POLICIES[policy_name]()).run()
    assert out["completed_programs"] == cfg.n_programs
    assert out["memory_violations"] == 0


def test_reproducible_paired_workload() -> None:
    cfg = WorkloadConfig("repeat", n_programs=8, capacity_tokens=10_000, seed=123)
    assert generate_workload(cfg) == generate_workload(cfg)


def test_fcfs_uses_arrival_order_and_head_of_line_blocking() -> None:
    cfg = WorkloadConfig("fcfs", n_programs=3, capacity_tokens=8_000, seed=18)
    specs, _ = generate_workload(cfg)
    policy = POLICIES["fcfs"]()
    sim = Simulator(cfg, specs, policy)
    base = sim.view().ready[0]
    oldest = replace(base, pid=10, waiting_age=5.0, admit_blocks=501)
    later_small = replace(base, pid=11, waiting_age=4.0, admit_blocks=1)
    view = replace(
        sim.view(), capacity_blocks=500, programs=(later_small, oldest)
    )
    assert policy.plan(view).admit == []
    assert not policy.retain_during_tool()


def test_coder_profile_matches_public_workload_scale() -> None:
    cfg = matrix("coder_calibration_smoke", 20260826)[0]
    _specs, meta = generate_workload(cfg)
    assert cfg.capacity_tokens == 670_000
    assert cfg.block_size == 16
    assert cfg.context_limit_tokens == 35_000
    assert 14.0 < meta["mean_rounds"] < 16.0
    assert 5_000 < meta["mean_total_decode_per_program"] < 7_000
    assert 0.75 < meta["cv_total_decode_per_program"] < 1.15
    assert 200_000 < meta["mean_submitted_prompt_per_program"] < 270_000


def test_measured_service_matches_micro_context_slopes() -> None:
    service = make_service_model("qwen3_coder_30b_vllm_012")
    ms_10k = 1_000.0 / service.aggregate_prefill_tps(1, 10_000)
    ms_40k = 1_000.0 / service.aggregate_prefill_tps(1, 40_960)
    decode_growth = service.aggregate_decode_tps(1, 0) / service.aggregate_decode_tps(1, 40_960) - 1
    assert 0.11 < ms_10k < 0.14
    assert 0.15 < ms_40k < 0.17
    assert decode_growth == pytest.approx(0.07)


def test_exported_setting_matches_calibration_runner() -> None:
    setting = build_setting(
        "coder_calibration", "qwen3_coder_30b_vllm_012", 20260826, 3.0, 0.5
    )
    assert setting["simulation_config"]["control_interval"] == 3.0
    assert setting["service_model"]["parameters"]["prefill_chunk_tokens"] == 8_192
    assert set(setting["policies"]) == set(DEFAULT_POLICIES)
    thunder = setting["policies"]["thunder"]["effective_initial_values"]
    assert thunder["buffer_tokens"] == 100
    assert thunder["acting_decay_base"] == 2.0
    assert len(setting["scenarios"]) == 3


def test_incremental_apc_reclaims_lru_without_memory_violation() -> None:
    cfg = matrix("coder_calibration_smoke", 20260826)[0]
    specs, _ = generate_workload(cfg)
    sim = Simulator(cfg, specs, POLICIES["fcfs"](), make_service_model("qwen3_coder_30b_vllm_012"))
    out = sim.run()
    assert out["memory_violations"] == 0
    assert out["cache_hit_prefix_ratio"] > 0
    assert sim.shadow_cache
    assert sim.physical_used() == 0
    assert sim._physical_blocks == sum(sim.blocks(p.active_kv) for p in sim.p)
    assert sim._shadow_blocks == sum(sim.blocks(x) for x in sim.shadow_cache.values())


def test_router_admission_is_distinct_from_engine_start() -> None:
    cfg = matrix("coder_calibration_smoke", 20260826)[0]
    specs, _ = generate_workload(cfg)
    sim = Simulator(cfg, specs, POLICIES["fcfs"](), make_service_model("qwen3_coder_30b_vllm_012"))
    sim.run_scheduler()
    assert any(p.phase == ENGINE_WAITING for p in sim.p)
    assert not sim.active()
    sim.schedule_engine()
    assert sim.active()


def test_cold_start_budget_and_engine_trace() -> None:
    cfg = matrix("coder_calibration_smoke", 20260826)[0]
    specs, _ = generate_workload(cfg)
    sim = Simulator(
        cfg,
        specs,
        POLICIES["fcfs"](),
        make_service_model("qwen3_coder_30b_vllm_012"),
        SimConfig(control_interval=3.0, dt=0.5),
    )
    out = sim.run()
    assert 55 <= out["max_waiting"] <= 72
    assert set(sim.timeseries[0]) == {
        "t_s", "kv_perc", "running", "waiting", "router_ready", "tool",
        "prefill", "decode", "completed_programs", "completed_requests",
        "prefill_tps", "decode_tps", "prefix_hit_cum", "preemptions",
    }
    assert sim.timeseries[0]["waiting"] > 0
    assert any(row["waiting"] == 0 for row in sim.timeseries)


def test_detailed_service_trace_is_opt_in() -> None:
    cfg = WorkloadConfig("trace", n_programs=4, capacity_tokens=8_000, seed=31)
    specs, _ = generate_workload(cfg)
    ordinary = Simulator(cfg, specs, POLICIES["bdp"]())
    ordinary.run()
    assert ordinary.program_samples == []
    assert ordinary.scheduler_events == []

    traced = Simulator(cfg, specs, POLICIES["bdp"](), record_trace=True)
    traced.run()
    assert traced.program_samples
    assert traced.scheduler_events
    assert {"t_s", "pid", "phase", "stage"} == set(traced.program_samples[0])
    assert "admitted_pids" in traced.scheduler_events[0]


def test_algorithm_seed_splits_are_disjoint() -> None:
    splits = [
        {cfg.seed for cfg in matrix("coder_calibration", 20260826)},
        {cfg.seed for cfg in matrix("coder_experiment", 20261301)},
        {cfg.seed for cfg in matrix("coder_experiment", 20262001)},
        {cfg.seed for cfg in matrix("coder_experiment", 20262101)},
        {cfg.seed for cfg in matrix("coder_experiment", 20262201)},
        {cfg.seed for cfg in matrix("coder_experiment", 20262301)},
        {cfg.seed for cfg in matrix("coder_experiment", 20262401)},
    ]
    assert [len(split) for split in splits] == [3, 20, 20, 20, 20, 20, 20]
    assert all(a.isdisjoint(b) for i, a in enumerate(splits) for b in splits[i + 1 :])
    assert policy_parameters(["bdp"])["bdp"] == {}
    bdp = policy_initial_values(["bdp"])["bdp"]
    assert "safety_fraction" not in bdp
    assert "knapsack_calls" not in bdp
    thunder = policy_initial_values(["thunder"])["thunder"]
    assert thunder["buffer_tokens"] == 100
    assert thunder["acting_decay_base"] == 2.0
