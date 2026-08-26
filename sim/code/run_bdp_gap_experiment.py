#!/usr/bin/env python3
"""Batch-only context ablation and hindsight-gap diagnostics for BDP."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from realistic_agentic_sim import (
    BDPPolicy,
    POLICIES,
    ProgramSpec,
    ProgramView,
    SimConfig,
    Simulator,
    SystemView,
    WorkEstimate,
    generate_workload,
    make_service_model,
    matrix,
)
from run_algorithm_experiment import bootstrap_mean_ci


BASELINES = ("fcfs", "thunder")


class NoContextBDP(BDPPolicy):
    """Diagnostic ablation of observable context-limit conditioning."""

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        return replace(view, context_limit=None).estimate_work(p)


class HybridOracleBDP(BDPPolicy):
    """Diagnostic index with exact turns and selected realized token fields."""

    def __init__(
        self,
        specs: Sequence[ProgramSpec],
        exact_prompt: bool = False,
        exact_decode: bool = False,
        decode_by_pid: Sequence[float] | None = None,
    ):
        self.specs = specs
        self.exact_prompt = exact_prompt
        self.exact_decode = exact_decode
        self.decode_by_pid = decode_by_pid
        super().__init__()

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        requests = self.specs[p.pid].requests[p.stage - 1 :]
        base = view.estimate_work(p)
        posterior_decode = max(
            1e-12, base.current_work - view.alpha_work * base.uncached_prompt
        )
        warm = p.cache_warm and p.prefix > 0
        uncached = float(p.prompt + (0 if warm else p.prefix))
        work = 0.0
        context = float(p.prefix)
        contexts: List[float] = []
        current_work = 0.0
        for index, req in enumerate(requests):
            prompt = (
                p.prompt if index == 0
                else req.prompt_tokens if self.exact_prompt else view.prompt_mean
            )
            if self.decode_by_pid is not None:
                decode = float(self.decode_by_pid[p.pid])
            else:
                decode = (
                    req.decode_tokens_actual if self.exact_decode
                    else posterior_decode
                )
            turn_work = view.alpha_work * (
                uncached if index == 0 else prompt
            ) + decode
            if index == 0:
                current_work = turn_work
            work += turn_work
            context += prompt + decode
            contexts.append(context)
        return WorkEstimate(
            uncached_prompt=uncached,
            current_work=current_work,
            remaining_work=work,
            expected_footprint_tokens=float(np.mean(contexts)),
        )

def true_latent_decode(cfg: WorkloadConfig) -> np.ndarray:
    sigma = math.sqrt(math.log1p(cfg.decode_program_cv ** 2))
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    scales = rng.lognormal(
        -0.5 * sigma * sigma, sigma, size=cfg.n_programs
    )
    marginal = (
        (1.0 - cfg.decode_large_prob) * cfg.decode_mean
        + cfg.decode_large_prob * cfg.decode_large_mean
    )
    return scales * marginal


def run_case(cfg, specs, policy) -> Dict[str, float]:
    return Simulator(
        cfg,
        specs,
        policy,
        make_service_model("qwen3_coder_30b_vllm_012"),
        SimConfig(control_interval=3.0, dt=0.5),
    ).run()


def write_summary(raw: pd.DataFrame, out_dir: str) -> None:
    rows = []
    comparisons = (
        ("bdp_vs_no_context_bdp", "bdp", "no_context_bdp"),
        ("bdp_vs_fcfs", "bdp", "fcfs"),
        ("bdp_vs_thunder", "bdp", "thunder"),
        ("turns_oracle_vs_bdp", "turns_oracle", "bdp"),
        ("latent_oracle_vs_bdp", "latent_oracle", "bdp"),
        ("program_mean_oracle_vs_bdp", "program_mean_oracle", "bdp"),
        ("hindsight_index_vs_bdp", "hindsight_index", "bdp"),
    )
    for metric in ("mean_program_ct", "p90_program_ct", "p95_program_ct"):
        wide = raw.pivot(index="seed", columns="case", values=metric)
        for name, candidate, baseline in comparisons:
            delta = 100.0 * (wide[baseline] - wide[candidate]) / wide[baseline]
            lo, hi = bootstrap_mean_ci(delta.to_numpy())
            rows.append({
                "metric": metric,
                "comparison": name,
                "mean_pct": float(delta.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "positive_seeds": int((delta > 0.0).sum()),
                "repetitions": len(delta),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "paired_summary.csv"), index=False
    )
    raw.groupby("case", as_index=False).mean(numeric_only=True).to_csv(
        os.path.join(out_dir, "case_summary.csv"), index=False
    )


def run(out_dir: str, seed0: int, repetitions: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for cfg in matrix("coder_experiment", seed0)[:repetitions]:
        specs, workload = generate_workload(cfg)
        program_mean = np.asarray([
            np.mean([req.decode_tokens_actual for req in spec.requests])
            for spec in specs
        ])
        cases = [
            ("fcfs", cfg, specs, POLICIES["fcfs"]()),
            ("thunder", cfg, specs, POLICIES["thunder"]()),
            ("no_context_bdp", cfg, specs, NoContextBDP()),
            ("bdp", cfg, specs, BDPPolicy()),
            ("turns_oracle", cfg, specs, HybridOracleBDP(specs)),
            (
                "latent_oracle", cfg, specs,
                HybridOracleBDP(
                    specs, decode_by_pid=true_latent_decode(cfg)
                ),
            ),
            (
                "program_mean_oracle", cfg, specs,
                HybridOracleBDP(specs, decode_by_pid=program_mean),
            ),
            (
                "hindsight_index", cfg, specs,
                HybridOracleBDP(
                    specs, exact_prompt=True, exact_decode=True
                ),
            ),
        ]
        for name, case_cfg, case_specs, policy in cases:
            result = run_case(case_cfg, case_specs, policy)
            rows.append({"seed": cfg.seed, "case": name, **workload, **result})
            print(
                f"seed={cfg.seed} case={name} mean_ct={result['mean_program_ct']:.2f}",
                flush=True,
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
    write_summary(raw, out_dir)
    with open(os.path.join(out_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "setting": "80_programs_all_arrive_at_t0",
            "seed0": seed0,
            "repetitions": repetitions,
            "baselines": list(BASELINES),
            "online_candidates": ["bdp"],
            "diagnostic_only": [
                "no_context_bdp", "turns_oracle", "latent_oracle",
                "program_mean_oracle", "hindsight_index",
            ],
            "oracle_base": "bdp_with_one_nested_information_change",
        }, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/bdp_gap_experiment")
    parser.add_argument("--seed0", type=int, default=20262401)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    run(args.out, args.seed0, args.repetitions)


if __name__ == "__main__":
    main()
