#!/usr/bin/env python3
"""Paired mechanism ablation for the smallest logically complete BDP."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from realistic_agentic_sim import (
    BDPPolicy,
    PRIORS,
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


class FixedDecodeBDP(BDPPolicy):
    """Remove online program-scale learning; retain the calibrated mean."""

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        return replace(view, decode_program_cv=0.0).estimate_work(p)


class NaiveDecodeBDP(BDPPolicy):
    """Treat every completed token count as persistent program-scale evidence."""

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        if not p.observed_decode_lengths or view.decode_program_cv <= 0.0:
            return view.estimate_work(p)
        precision = 1.0 / (view.decode_program_cv * view.decode_program_cv)
        scale = (
            precision
            + sum(p.observed_decode_lengths) / max(1e-12, view.decode_mean)
        ) / (precision + len(p.observed_decode_lengths))
        return replace(
            view,
            decode_program_cv=0.0,
            decode_mean=view.decode_mean * scale,
        ).estimate_work(p)


class NoContextBDP(BDPPolicy):
    """Remove conditioning on the observable 35k context guard."""

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        return replace(view, context_limit=None).estimate_work(p)


class NoSurvivalUpdateBDP(BDPPolicy):
    """Use the unconditional K mean minus elapsed turns, not K >= stage."""

    def __init__(self) -> None:
        super().__init__()
        self.prior_mean = 1.0

    def reset(self, cfg, service) -> None:
        super().reset(cfg, service)
        prior_name = cfg.policy_prior_name or cfg.prior_name
        self.prior_mean = PRIORS[prior_name].mean

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        unconditional_remaining = max(1.0, self.prior_mean - p.stage + 1.0)
        return view.estimate_work(replace(p, remaining_rounds=unconditional_remaining))


class CurrentTurnOnlyBDP(BDPPolicy):
    """Remove the program objective and rank only the current request work."""

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        return view.estimate_work(replace(p, remaining_rounds=1.0))


class NoDualBDP(BDPPolicy):
    """Remove the KV shadow price while retaining Bayesian remaining work."""

    def update_state(self, view, estimates) -> None:
        self.lam = 0.0
        self.epoch += 1
        self.price_sum += self.lam


class AdmissionFootprintBDP(BDPPolicy):
    """Price only immediate admission blocks instead of future KV residency."""

    def estimate(self, p: ProgramView, view: SystemView) -> WorkEstimate:
        estimate = view.estimate_work(p)
        return replace(
            estimate,
            expected_footprint_tokens=float(p.admit_blocks * view.block_size),
        )


class FullBDP(BDPPolicy):
    """Pre-ablation reference with block normalization and tie heuristics."""

    def admission_score(self, p, estimate, view) -> float:
        margin = self.bayesian_value(estimate) - self.lam * estimate.expected_footprint_tokens
        return margin / max(1, p.admit_blocks)

    def rank_ready(self, view, estimates) -> List[ProgramView]:
        def key(p: ProgramView):
            estimate = estimates[p.pid]
            margin = (
                self.bayesian_value(estimate)
                - self.lam * estimate.expected_footprint_tokens
            )
            return (
                margin / max(1, p.admit_blocks),
                margin,
                -estimate.current_work,
                -p.pid,
            )

        return sorted(view.ready, key=key, reverse=True)

    def fallback_ready(self, feasible, view):
        return min(feasible, key=lambda p: (p.admit_blocks, -p.waiting_age, p.pid))


CASES: Tuple[Tuple[str, type[BDPPolicy], str], ...] = (
    ("bdp_full", BDPPolicy, "reference"),
    ("no_decode_update", FixedDecodeBDP, "Bayesian decode-scale update"),
    ("naive_decode_update", NaiveDecodeBDP, "mixture-aware decode evidence"),
    ("no_context_guard", NoContextBDP, "observable context conditioning"),
    ("no_k_survival_update", NoSurvivalUpdateBDP, "K survival posterior"),
    ("current_turn_only", CurrentTurnOnlyBDP, "remaining-program objective"),
    ("no_dual_price", NoDualBDP, "KV dual price"),
    ("admission_footprint", AdmissionFootprintBDP, "future KV footprint"),
    ("legacy_action", FullBDP, "minimal margin replaced by block normalization and tie heuristics"),
)


def summarize(raw: pd.DataFrame, cases=CASES) -> pd.DataFrame:
    wide = raw.pivot(index="seed", columns="case", values="mean_program_ct")
    rows: List[Dict[str, object]] = []
    for case, _policy, removed in cases:
        delta = 100.0 * (wide[case] - wide.bdp_full) / wide.bdp_full
        lo, hi = bootstrap_mean_ci(delta.to_numpy())
        part = raw[raw.case == case]
        rows.append({
            "case": case,
            "mechanism_removed_or_replaced": removed,
            "mean_program_ct": float(part.mean_program_ct.mean()),
            "degradation_vs_full_pct": float(delta.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "worse_seeds": int((delta > 0.0).sum()),
            "repetitions": len(delta),
            "mean_decision_ms": float(part.mean_decision_ms.mean()),
        })
    return pd.DataFrame(rows).sort_values("degradation_vs_full_pct", ascending=False)


def summarize_metrics(raw: pd.DataFrame, cases=CASES) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric in ("mean_program_ct", "p90_program_ct", "p95_program_ct", "makespan"):
        wide = raw.pivot(index="seed", columns="case", values=metric)
        for case, _policy, description in cases:
            degradation = 100.0 * (wide[case] - wide.bdp_full) / wide.bdp_full
            lo, hi = bootstrap_mean_ci(degradation.to_numpy())
            rows.append({
                "metric": metric,
                "case": case,
                "description": description,
                "degradation_vs_bdp_pct": float(degradation.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "worse_seeds": int((degradation > 0.0).sum()),
                "repetitions": len(degradation),
            })
    return pd.DataFrame(rows)


def run(out_dir: str, seed0: int, repetitions: int, case_names: Sequence[str]) -> None:
    cases = tuple(case for case in CASES if case[0] in case_names)
    unknown = sorted(set(case_names) - {case[0] for case in CASES})
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    if not cases or "bdp_full" not in case_names:
        raise ValueError("ablation must include bdp_full")
    os.makedirs(out_dir, exist_ok=True)
    service = make_service_model("qwen3_coder_30b_vllm_012")
    sim_cfg = SimConfig(control_interval=3.0, dt=0.5)
    rows: List[Dict[str, object]] = []
    for cfg in matrix("coder_experiment", seed0)[:repetitions]:
        specs, workload = generate_workload(cfg)
        for case, policy_type, _removed in cases:
            result = Simulator(cfg, specs, policy_type(), service, sim_cfg).run()
            rows.append({"seed": cfg.seed, "case": case, **workload, **result})
            print(
                f"seed={cfg.seed} case={case} mean={result['mean_program_ct']:.2f} "
                f"p90={result['p90_program_ct']:.2f} make={result['makespan']:.2f}",
                flush=True,
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out_dir, "raw_results.csv"), index=False)
    summarize(raw, cases).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    summarize_metrics(raw, cases).to_csv(
        os.path.join(out_dir, "metric_summary.csv"), index=False
    )
    with open(os.path.join(out_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "setting": "80_programs_all_arrive_at_t0",
            "preset": "coder_experiment",
            "seed0": seed0,
            "repetitions": repetitions,
            "service_profile": "qwen3_coder_30b_vllm_012",
            "sim_config": {"control_interval": 3.0, "dt": 0.5},
            "selection_rule": (
                "an action candidate must improve mean completion time and show no "
                "statistically supported P90, P95, or makespan regression"
            ),
            "cases": [
                {"name": name, "removed_or_replaced": removed}
                for name, _policy, removed in cases
            ],
        }, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/bdp_ablation")
    parser.add_argument("--seed0", type=int, default=20261301)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--cases", default=",".join(name for name, _policy, _removed in CASES))
    args = parser.parse_args()
    run(
        args.out,
        args.seed0,
        args.repetitions,
        [name.strip() for name in args.cases.split(",") if name.strip()],
    )


if __name__ == "__main__":
    main()
