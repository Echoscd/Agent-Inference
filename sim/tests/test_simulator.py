from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

from realistic_agentic_sim import (  # noqa: E402
    DEFAULT_POLICIES,
    POLICIES,
    Simulator,
    WorkloadConfig,
    exact_cache_knapsack,
    generate_workload,
    make_service_model,
    matrix,
)
from export_setting import build_setting  # noqa: E402
from trace_compare import TraceSimulator  # noqa: E402


def test_exact_cache_knapsack() -> None:
    # Capacity 5: items 2+3 with value 11 beat the single size-5 item value 10.
    chosen = exact_cache_knapsack([(1, 2, 5.0), (2, 3, 6.0), (3, 5, 10.0)], 5)
    assert chosen == {1, 2}


def test_policy_view_hides_realized_future() -> None:
    cfg = WorkloadConfig("view", n_programs=4, capacity_tokens=8_000, seed=11)
    specs, _ = generate_workload(cfg)
    sim = Simulator(cfg, specs, POLICIES["hazard_grade_knapsack"]())
    view = sim.view()
    assert view.ready
    p = view.ready[0]
    assert not hasattr(p, "decode_actual")
    assert not hasattr(p, "tool_remaining")
    assert not hasattr(p, "tool_duration_actual")


@pytest.mark.parametrize("policy_name", DEFAULT_POLICIES)
def test_smoke_completes_without_memory_violation(policy_name: str) -> None:
    cfg = WorkloadConfig("smoke_test", n_programs=12, capacity_tokens=12_000, seed=17)
    specs, _ = generate_workload(cfg)
    out = Simulator(cfg, specs, POLICIES[policy_name]()).run()
    assert out["completed_programs"] == cfg.n_programs
    assert out["memory_violations"] == 0
    assert out["mean_program_ct"] > 0


def test_reproducible_paired_workload() -> None:
    cfg = WorkloadConfig("repeat", n_programs=8, capacity_tokens=10_000, seed=123)
    specs1, meta1 = generate_workload(cfg)
    specs2, meta2 = generate_workload(cfg)
    assert specs1 == specs2
    assert meta1 == meta2


def test_real_proxy_has_long_initial_context_and_trace_scale_decode() -> None:
    cfg = matrix("real_proxy_smoke", 20260717)[0]
    specs, meta = generate_workload(cfg)
    assert cfg.n_programs == 80
    assert meta["mean_initial_prompt"] > 10_000
    assert 7.0 < meta["mean_rounds"] < 11.0
    assert meta["mean_decode_per_turn"] > 1_000
    assert all(p.requests[0].prompt_tokens > 0 for p in specs)


def test_real_proxy_service_makes_long_cold_prefill_more_expensive() -> None:
    synthetic = make_service_model("synthetic")
    proxy = make_service_model("qwen3_32b_vllm_proxy")
    assert proxy.cold_prefill_seconds(16_000) > synthetic.cold_prefill_seconds(16_000)
    # The proxy is anchored to the 250--300 tok/s range reported around a
    # 10--15-way active batch, allowing context variation around that target.
    rate = proxy.aggregate_decode_tps(14, 32_768)
    assert 200 < rate < 350


def test_thunder_greedy_proxy_completes_without_memory_violation() -> None:
    cfg = WorkloadConfig("greedy", n_programs=12, capacity_tokens=12_000, seed=29)
    specs, _ = generate_workload(cfg)
    out = Simulator(cfg, specs, POLICIES["thunder_greedy"]()).run()
    assert out["completed_programs"] == cfg.n_programs
    assert out["memory_violations"] == 0


def test_dual_price_proxy_completes_without_memory_violation() -> None:
    cfg = WorkloadConfig("dual_proxy", n_programs=12, capacity_tokens=12_000, seed=31)
    specs, _ = generate_workload(cfg)
    out = Simulator(cfg, specs, POLICIES["dual_price_proxy"]()).run()
    assert out["completed_programs"] == cfg.n_programs
    assert out["memory_violations"] == 0
    assert out["dual_proxy_updates"] > 0


def test_trace_simulator_records_initial_and_later_admissions() -> None:
    cfg = WorkloadConfig("trace", n_programs=6, capacity_tokens=8_000, seed=37)
    specs, _ = generate_workload(cfg)
    sim = TraceSimulator(cfg, specs, POLICIES["hazard_grade_knapsack"]())
    out = sim.run()
    assert out["completed_programs"] == cfg.n_programs
    assert sim.snapshots[0]["time"] == 0.0
    assert sim.snapshots[0]["n_admit"] > 0
    assert any(x["event"].startswith("ADMIT_") for x in sim.events)
    assert any(x["new_phase"] == "DONE" for x in sim.transitions)


def test_exported_synthetic_setting_matches_code_matrix() -> None:
    setting = build_setting("test", "synthetic", 20260711, 5.0, 0.5)
    assert setting["setting_kind"] == "simulator_environment_not_results"
    assert len(setting["scenarios"]) == 45
    assert setting["simulation_config"]["control_interval"] == 5.0
    assert setting["service_model"]["parameters"]["prefill_single_tps"] == 7_000.0
    assert "realized future decode length" in setting["information_boundary"]["policy_does_not_observe"]


def test_exported_real_proxy_setting_matches_code_matrix() -> None:
    setting = build_setting(
        "real_proxy", "qwen3_32b_vllm_proxy", 20260717, 5.0, 0.5
    )
    assert len(setting["scenarios"]) == 9
    assert {x["capacity_tokens"] for x in setting["scenarios"]} == {
        360_000,
        480_000,
        600_000,
    }
    assert setting["service_model"]["parameters"]["context_ref"] == 32_768.0
    assert all(x["initial_prompt_mean"] == 14_000.0 for x in setting["scenarios"])
