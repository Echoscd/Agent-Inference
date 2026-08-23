#!/usr/bin/env python3
"""One-factor-at-a-time robustness checks for the revised simulator."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from realistic_agentic_sim import (  # noqa: E402
    POLICIES,
    ServiceModel,
    SimConfig,
    Simulator,
    WorkloadConfig,
    generate_workload,
)

POLICY_NAMES = ["thunder", "student_bdp", "hazard_grade_knapsack"]


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root, "results", "sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    settings = []
    for sat in (8.0, 16.0, 32.0):
        settings.append(("decode_batch_sat", sat, ServiceModel(decode_sat=sat), 5.0, 0.95))
    for interval in (1.0, 5.0, 10.0):
        settings.append(("control_interval_s", interval, ServiceModel(), interval, 0.95))


    rows = []
    for factor, value, service, interval, reserve_q in settings:
        for rep in range(3):
            cfg = WorkloadConfig(
                name=f"{factor}_{value}_r{rep}",
                n_programs=64,
                tool_mix="swe_mixed",
                capacity_tokens=40_000,
                reserve_quantile=reserve_q,
                seed=20260720 + rep + int(1000 * value) + 100_000 * settings.index((factor, value, service, interval, reserve_q)),
            )
            specs, meta = generate_workload(cfg)
            for policy_name in POLICY_NAMES:
                t0 = time.perf_counter()
                sim = Simulator(
                    cfg,
                    specs,
                    POLICIES[policy_name](),
                    service=service,
                    sim_cfg=SimConfig(control_interval=interval, dt=0.5),
                )
                m = sim.run()
                rows.append({
                    **m,
                    "factor": factor,
                    "value": value,
                    "rep": rep,
                    "scenario": cfg.name,
                    "seed": cfg.seed,
                    "runtime_s": time.perf_counter() - t0,
                    "total_requests": meta["total_requests"],
                })
                print(f"{factor}={value} rep={rep} {policy_name}: mean={m['mean_program_ct']:.2f}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
    keys = ["factor", "value", "rep", "scenario", "seed"]
    bdp = raw[raw.policy == "student_bdp"][keys + ["mean_program_ct", "p90_program_ct", "steps_per_min"]].rename(
        columns={"mean_program_ct": "mean_bdp", "p90_program_ct": "p90_bdp", "steps_per_min": "steps_bdp"}
    )
    paired = raw.merge(bdp, on=keys, how="left")
    paired["mean_impr_vs_bdp_pct"] = 100.0 * (paired.mean_bdp - paired.mean_program_ct) / paired.mean_bdp
    paired["p90_impr_vs_bdp_pct"] = 100.0 * (paired.p90_bdp - paired.p90_program_ct) / paired.p90_bdp
    paired["steps_impr_vs_bdp_pct"] = 100.0 * (paired.steps_per_min / paired.steps_bdp - 1.0)
    paired.to_csv(os.path.join(out_dir, "paired_vs_bdp.csv"), index=False)
    summary = paired.groupby(["factor", "value", "policy"], as_index=False).agg(
        mean_program_ct=("mean_program_ct", "mean"),
        mean_impr_vs_bdp=("mean_impr_vs_bdp_pct", "mean"),
        p90_impr_vs_bdp=("p90_impr_vs_bdp_pct", "mean"),
        steps_impr_vs_bdp=("steps_impr_vs_bdp_pct", "mean"),
        cache_hit=("cache_hit_prefix_ratio", "mean"),
        decision_ms=("mean_decision_ms", "mean"),
    )
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
