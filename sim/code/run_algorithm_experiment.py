#!/usr/bin/env python3
"""Paired held-out evaluation against the FCFS and Thunder baselines."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from realistic_agentic_sim import (
    DEFAULT_POLICIES,
    POLICIES,
    SimConfig,
    Simulator,
    generate_workload,
    make_service_model,
    matrix,
)


PRIMARY_METRICS = (
    "mean_program_ct",
    "p90_program_ct",
    "p95_program_ct",
    "makespan",
    "aggregate_gen_tps",
    "physical_kv_util",
    "cache_hit_prefix_ratio",
    "mean_active",
    "mean_router_paused",
    "active_preemptions",
)


def bootstrap_mean_ci(values: np.ndarray, seed: int = 7) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(5_000, len(values)))].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def policy_parameters(policy_names: List[str]) -> Dict[str, Dict[str, object]]:
    """Record constructor defaults so every result directory is reproducible."""
    return {
        policy_name: {
            name: parameter.default
            for name, parameter in inspect.signature(POLICIES[policy_name]).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        for policy_name in policy_names
    }


def policy_initial_values(policy_names: List[str]) -> Dict[str, Dict[str, object]]:
    """Record fixed constants as well as tunable constructor arguments."""
    return {
        policy_name: {
            name: value
            for name, value in vars(POLICIES[policy_name]()).items()
            if isinstance(value, (str, int, float, bool))
        }
        for policy_name in policy_names
    }


def run(
    out_dir: str,
    seed0: int,
    policy_names: List[str],
    preset: str = "coder_experiment",
    control_interval: float = 3.0,
) -> None:
    unknown = sorted(set(policy_names) - set(POLICIES))
    if unknown:
        raise ValueError(f"unknown policies {unknown}; registered={sorted(POLICIES)}")
    if "fcfs" not in policy_names:
        raise ValueError("paired evaluation requires the fcfs baseline")

    os.makedirs(out_dir, exist_ok=True)
    service = make_service_model("qwen3_coder_30b_vllm_012")
    sim_cfg = SimConfig(control_interval=control_interval, dt=0.5)
    rows: List[Dict[str, object]] = []
    configs = matrix(preset, seed0)
    for cfg in configs:
        specs, workload = generate_workload(cfg)
        for policy_name in policy_names:
            result = Simulator(
                cfg,
                specs,
                POLICIES[policy_name](),
                service,
                sim_cfg,
            ).run()
            rows.append({
                "scenario": cfg.name,
                "seed": cfg.seed,
                **workload,
                **result,
            })
            print(
                f"seed={cfg.seed} policy={policy_name} "
                f"mean_ct={result['mean_program_ct']:.1f} "
                f"makespan={result['makespan']:.1f}",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
    baseline = raw[raw.policy == "fcfs"][["seed", *PRIMARY_METRICS]].rename(
        columns={metric: f"{metric}_fcfs" for metric in PRIMARY_METRICS}
    )
    paired = raw.merge(baseline, on="seed", validate="many_to_one")
    for metric in ("mean_program_ct", "p90_program_ct", "p95_program_ct", "makespan"):
        paired[f"{metric}_improvement_pct"] = 100.0 * (
            paired[f"{metric}_fcfs"] - paired[metric]
        ) / paired[f"{metric}_fcfs"]
    paired["aggregate_gen_tps_improvement_pct"] = 100.0 * (
        paired.aggregate_gen_tps / paired.aggregate_gen_tps_fcfs - 1.0
    )
    paired.to_csv(os.path.join(out_dir, "paired_vs_fcfs.csv"), index=False)

    summaries = []
    for policy_name, part in paired.groupby("policy"):
        delta = part.mean_program_ct_improvement_pct.to_numpy()
        lo, hi = bootstrap_mean_ci(delta)
        summaries.append({
            "policy": policy_name,
            **{metric: float(part[metric].mean()) for metric in PRIMARY_METRICS},
            "mean_ct_improvement_pct": float(delta.mean()),
            "mean_ct_improvement_ci_low": lo,
            "mean_ct_improvement_ci_high": hi,
            "p90_ct_improvement_pct": float(part.p90_program_ct_improvement_pct.mean()),
            "p95_ct_improvement_pct": float(part.p95_program_ct_improvement_pct.mean()),
            "win_rate_vs_fcfs": float((delta > 0.0).mean()),
            "makespan_improvement_pct": float(part.makespan_improvement_pct.mean()),
            "gen_tps_improvement_pct": float(part.aggregate_gen_tps_improvement_pct.mean()),
        })
    pd.DataFrame(summaries).sort_values(
        "mean_ct_improvement_pct", ascending=False
    ).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    with open(os.path.join(out_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "preset": preset,
            "seed0": seed0,
            "repetitions": len(configs),
            "policies": policy_names,
            "policy_parameters": policy_parameters(policy_names),
            "policy_initial_values": policy_initial_values(policy_names),
            "baseline": "fcfs",
            "service_profile": "qwen3_coder_30b_vllm_012",
            "sim_config": asdict(sim_cfg),
        }, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/algorithm_experiment")
    parser.add_argument("--seed0", type=int, default=20262401)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--control-interval", type=float, default=3.0)
    parser.add_argument(
        "--preset",
        choices=["coder_calibration_smoke", "coder_experiment"],
        default="coder_experiment",
    )
    args = parser.parse_args()
    run(
        args.out,
        args.seed0,
        [x.strip() for x in args.policies.split(",") if x.strip()],
        preset=args.preset,
        control_interval=args.control_interval,
    )


if __name__ == "__main__":
    main()
