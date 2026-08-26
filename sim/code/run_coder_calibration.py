#!/usr/bin/env python3
"""Validate the single measured Coder/vLLM profile against public experiments.

This intentionally does not optimize parameters.  It runs one shared profile,
reports every mismatch, and keeps the micro-experiment constraints separate
from the three end-to-end repetitions.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import pandas as pd

from realistic_agentic_sim import (
    POLICIES,
    SimConfig,
    Simulator,
    generate_workload,
    make_service_model,
    matrix,
)


WORKLOAD_RANGES: Dict[str, Tuple[float, float]] = {
    "mean_rounds": (14.6, 15.3),
    "mean_total_decode_per_program": (5_499.2, 6_629.6),
    "cv_total_decode_per_program": (0.90, 0.97),
    "mean_submitted_prompt_per_program": (235_136.0, 251_274.0),
    "mean_tool_s_per_program": (10.8, 18.3),
}


def _range_row(kind: str, metric: str, value: float, bounds: Tuple[float, float]) -> Dict[str, object]:
    lo, hi = bounds
    if value < lo:
        relative_error = (value - lo) / max(abs(lo), 1e-12)
    elif value > hi:
        relative_error = (value - hi) / max(abs(hi), 1e-12)
    else:
        relative_error = 0.0
    return {
        "kind": kind,
        "metric": metric,
        "simulated": value,
        "public_min": lo,
        "public_max": hi,
        "within_public_range": lo <= value <= hi,
        "relative_error_to_range": relative_error,
    }


def micro_checks() -> Dict[str, float]:
    service = make_service_model("qwen3_coder_30b_vllm_012")
    ms_10k = 1_000.0 / service.aggregate_prefill_tps(1, 10_000.0)
    ms_40k = 1_000.0 / service.aggregate_prefill_tps(1, 40_960.0)
    decode_latency_growth = service.aggregate_decode_tps(1, 0.0) / service.aggregate_decode_tps(1, 40_960.0) - 1.0

    # One 15-turn trajectory with the measured mean token increments.  Reuse
    # prefills only the delta; drop repeatedly prefills the full context.
    context = 2_788.0
    reuse_s = service.cold_prefill_seconds(context)
    drop_s = reuse_s
    for _ in range(19):
        delta = 1_260.0 + 396.0
        context += delta
        reuse_s += delta / service.aggregate_prefill_tps(1, context)
        drop_s += context / service.aggregate_prefill_tps(1, context)
    return {
        "prefill_ms_per_token_10k": ms_10k,
        "prefill_ms_per_token_40k": ms_40k,
        "decode_latency_growth_40k": decode_latency_growth,
        "prefix_drop_over_reuse": drop_s / reuse_s,
    }


def run(out_dir: str, seed0: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    service = make_service_model("qwen3_coder_30b_vllm_012")
    sim_cfg = SimConfig(control_interval=3.0, dt=0.5)
    raw = []
    checks = []
    workload_rows = []
    trace_rows = []

    for cfg in matrix("coder_calibration", seed0):
        specs, meta = generate_workload(cfg)
        workload_rows.append(meta)
        sim = Simulator(cfg, specs, POLICIES["fcfs"](), service, sim_cfg)
        result = sim.run()
        trace_rows.extend({
            "scenario": cfg.name,
            "seed": cfg.seed,
            "policy": "fcfs",
            **row,
        } for row in sim.timeseries)
        raw.append({"scenario": cfg.name, "seed": cfg.seed, **meta, **result})

    raw_df = pd.DataFrame(raw)
    workload_means = pd.DataFrame(workload_rows).mean(numeric_only=True)
    for metric, bounds in WORKLOAD_RANGES.items():
        checks.append(_range_row("workload", metric, float(workload_means[metric]), bounds))
    micro = micro_checks()
    micro_bounds = {
        "prefill_ms_per_token_10k": (0.11, 0.13),
        "prefill_ms_per_token_40k": (0.15, 0.17),
        "decode_latency_growth_40k": (0.06, 0.08),
        "prefix_drop_over_reuse": (10.0, 13.0),
    }
    for metric, value in micro.items():
        checks.append(_range_row("micro", metric, value, micro_bounds[metric]))

    checks_df = pd.DataFrame(checks)
    raw_df.to_csv(os.path.join(out_dir, "simulated_repetitions.csv"), index=False)
    pd.DataFrame(trace_rows).to_csv(os.path.join(out_dir, "engine_timeseries.csv"), index=False)
    checks_df.to_csv(os.path.join(out_dir, "calibration_checks.csv"), index=False)
    summary = {
        "profile": "qwen3_coder_30b_vllm_012",
        "seed0": seed0,
        "checks": len(checks_df),
        "within_public_range": int(checks_df.within_public_range.sum()),
        "failed_checks": checks_df.loc[~checks_df.within_public_range, ["kind", "metric"]].to_dict("records"),
        "micro": micro,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/coder_calibration")
    parser.add_argument("--seed0", type=int, default=20260826)
    args = parser.parse_args()
    run(args.out, args.seed0)


if __name__ == "__main__":
    main()
