#!/usr/bin/env python3
"""Run dual-price policies and plot their lambda trajectories."""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Type

import pandas as pd

import agentic_kv_sim as simlib


def trace_row(policy: str, decision: int, lam: float, sim: simlib.Simulator) -> Dict[str, float]:
    free = sim.free_memory()
    kv_used = sim.M - free
    return {
        "policy": policy,
        "decision": decision,
        "time": sim.t,
        "lambda": lam,
        "free_memory": free,
        "kv_used": kv_used,
        "kv_utilization": kv_used / max(1.0, sim.M),
        "waiting": len(sim.waiting),
    }


class TraceDualDescentCurrent(simlib.DualDescentCurrentPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.trace: List[Dict[str, float]] = []

    def decide(self, sim: simlib.Simulator) -> Dict[int, str]:
        actions = super().decide(sim)
        self.trace.append(trace_row(self.name, len(self.trace), self.lam, sim))
        return actions


class TraceBayesDualPrice(simlib.BayesDualPricePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.trace: List[Dict[str, float]] = []

    def decide(self, sim: simlib.Simulator) -> Dict[int, str]:
        actions = super().decide(sim)
        self.trace.append(trace_row(self.name, len(self.trace), self.lam, sim))
        return actions


TRACE_POLICIES: Dict[str, Type] = {
    "dual_descent_current": TraceDualDescentCurrent,
    "bayes_dual_price": TraceBayesDualPrice,
}


def build_scenario(preset: str, memory_factor: int) -> tuple[simlib.Scenario, int, int]:
    cfg = simlib.preset_config(preset)
    profile = cfg["profiles"][0]
    mix = cfg["mixes"][0]
    cv = cfg["cvs"][0]
    n = int(cfg.get("n_by_factor", {}).get(memory_factor, cfg["n"]))
    p_mean, d_mean = simlib.build_profiles()[profile]
    seed = cfg["seed0"] + int(1000 * cv)
    scen = simlib.Scenario(
        name=f"{profile}_{mix}_cv{cv}_f{memory_factor}",
        n=n,
        p_mean=p_mean,
        d_mean=d_mean,
        token_cv=cv,
        prior_mix=mix,
        seed=seed,
    )
    programs, qmax, _total_requests = simlib.generate_instance(scen)
    return scen, int(math.ceil(memory_factor * qmax)), qmax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "quick", "standard", "stress"], default="quick")
    parser.add_argument("--memory-factor", type=int, default=5)
    parser.add_argument("--out", default="results/dual_descent_lambda")
    parser.add_argument("--policies", default="dual_descent_current,bayes_dual_price")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows: List[Dict[str, float]] = []
    metrics: List[Dict[str, float]] = []
    selected = [p.strip() for p in args.policies.split(",") if p.strip()]
    scen, memory, qmax = build_scenario(args.preset, args.memory_factor)

    for policy_name in selected:
        if policy_name not in TRACE_POLICIES:
            raise ValueError(f"unsupported traced policy: {policy_name}")
        programs, _qmax, total_requests = simlib.generate_instance(scen)
        simlib.clear_value_caches()
        policy = TRACE_POLICIES[policy_name]()
        runner = simlib.Simulator(scen, programs, memory, policy)
        result = runner.run()
        rows.extend(policy.trace)
        result.update(
            preset=args.preset,
            memory_factor=args.memory_factor,
            M=memory,
            qmax_obs=qmax,
            total_requests_instance=total_requests,
        )
        metrics.append(result)

    trace = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(metrics)
    trace_path = os.path.join(args.out, "lambda_trace.csv")
    metrics_path = os.path.join(args.out, "metrics.csv")
    trace.to_csv(trace_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, wrote CSV only: {exc}")
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for policy_name, part in trace.groupby("policy"):
        part = part.sort_values("decision")
        ax.plot(part["decision"], part["lambda"], linewidth=1.8, label=policy_name)
    ax.set_xlabel("decision epoch")
    ax.set_ylabel("lambda")
    ax.set_title(f"Dual price lambda trajectory ({args.preset}, memory factor={args.memory_factor})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(args.out, "lambda_trace.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(selected), 1, figsize=(9.5, 3.0 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, policy_name in zip(axes, selected):
        part = trace[trace.policy == policy_name].sort_values("decision")
        ax.plot(part["decision"], part["lambda"], color="tab:blue", linewidth=1.8, label="lambda")
        ax.set_ylabel("lambda", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(True, alpha=0.25)
        ax.set_title(policy_name)

        kv_ax = ax.twinx()
        kv_ax.plot(part["decision"], part["kv_utilization"], color="tab:orange", linewidth=1.4, alpha=0.85, label="KV utilization")
        kv_ax.set_ylabel("KV utilization", color="tab:orange")
        kv_ax.tick_params(axis="y", labelcolor="tab:orange")
        kv_ax.set_ylim(0.0, 1.05)
    axes[-1].set_xlabel("decision epoch")
    fig.suptitle(f"Dual price and KV utilization ({args.preset}, memory factor={args.memory_factor})", y=0.995)
    fig.tight_layout()
    dual_kv_path = os.path.join(args.out, "lambda_kv_utilization.png")
    fig.savefig(dual_kv_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for policy_name, part in trace.groupby("policy"):
        part = part.sort_values("decision")
        ax.plot(part["decision"], part["kv_used"], linewidth=1.6, label=policy_name)
    ax.set_xlabel("decision epoch")
    ax.set_ylabel("KV used")
    ax.set_title(f"KV usage trajectory ({args.preset}, memory factor={args.memory_factor}, M={memory})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    kv_path = os.path.join(args.out, "kv_used_trace.png")
    fig.savefig(kv_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {trace_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {dual_kv_path}")
    print(f"Wrote {kv_path}")


if __name__ == "__main__":
    main()
