#!/usr/bin/env python3
"""Render one paired workload as PPT-style service and program timelines."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from realistic_agentic_sim import (
    BDPPolicy,
    POLICIES,
    Policy,
    ProgramSpec,
    SimConfig,
    Simulator,
    generate_workload,
    make_service_model,
    matrix,
)
from run_bdp_gap_experiment import HybridOracleBDP


ONLINE_POLICIES = ("fcfs", "thunder", "bdp")
DIAGNOSTICS = ("hindsight_index",)
COLORS = {
    "fcfs": "#6b7280",
    "thunder": "#d97706",
    "bdp": "#2563eb",
    "hindsight_index": "#059669",
}
PHASE_COLORS = {
    "READY": "#d1d5db",
    "ENGINE_WAITING": "#f59e0b",
    "PREFILL": "#60a5fa",
    "DECODE": "#2563eb",
    "TOOL": "#10b981",
}


class BDPActionAudit(Policy):
    """Run causal BDP while comparing each action on the same state to hindsight."""

    name = "bdp"

    def __init__(self, specs: Sequence[ProgramSpec]):
        self.specs = specs
        self.causal = BDPPolicy()
        self.oracle = HybridOracleBDP(specs, exact_prompt=True, exact_decode=True)
        self.rows: List[Dict[str, object]] = []

    def reset(self, cfg, service) -> None:
        self.causal.reset(cfg, service)
        self.oracle.reset(cfg, service)
        self.rows = []

    def retain_during_tool(self) -> bool:
        return False

    def emergency_order(self, view):
        return self.causal.emergency_order(view)

    def plan(self, view):
        causal = self.causal.plan(view)
        oracle = self.oracle.plan(view)
        causal_set = set(causal.admit)
        oracle_set = set(oracle.admit)
        ready = {p.pid: p for p in view.ready}
        estimates = {pid: view.estimate_work(p) for pid, p in ready.items()}

        def mean(pids: set[int], value) -> float:
            values = [value(pid, ready[pid], estimates[pid]) for pid in pids]
            return float(np.mean(values)) if values else np.nan

        union = causal_set | oracle_set
        self.rows.append({
            "t_s": view.now,
            "ready": len(view.ready),
            "causal_admit_count": len(causal_set),
            "oracle_admit_count": len(oracle_set),
            "action_jaccard": len(causal_set & oracle_set) / max(1, len(union)),
            "causal_actual_remaining_decode": mean(
                causal_set,
                lambda pid, p, _e: sum(
                    req.decode_tokens_actual for req in self.specs[pid].requests[p.stage - 1 :]
                ),
            ),
            "oracle_actual_remaining_decode": mean(
                oracle_set,
                lambda pid, p, _e: sum(
                    req.decode_tokens_actual for req in self.specs[pid].requests[p.stage - 1 :]
                ),
            ),
            "causal_observed_decode_per_turn": mean(
                causal_set,
                lambda _pid, p, _e: sum(p.observed_decode_lengths) / max(1, p.stage - 1),
            ),
            "oracle_observed_decode_per_turn": mean(
                oracle_set,
                lambda _pid, p, _e: sum(p.observed_decode_lengths) / max(1, p.stage - 1),
            ),
            "causal_stage": mean(causal_set, lambda _pid, p, _e: p.stage),
            "oracle_stage": mean(oracle_set, lambda _pid, p, _e: p.stage),
            "causal_estimated_remaining_work": mean(
                causal_set, lambda _pid, _p, estimate: estimate.remaining_work,
            ),
            "oracle_estimated_remaining_work": mean(
                oracle_set, lambda _pid, _p, estimate: estimate.remaining_work,
            ),
            "causal_expected_footprint": mean(
                causal_set, lambda _pid, _p, estimate: estimate.expected_footprint_tokens,
            ),
            "oracle_expected_footprint": mean(
                oracle_set, lambda _pid, _p, estimate: estimate.expected_footprint_tokens,
            ),
            "causal_pids": ",".join(map(str, sorted(causal_set))),
            "oracle_pids": ",".join(map(str, sorted(oracle_set))),
        })
        return causal

    def diagnostics(self):
        return self.causal.diagnostics()


def policy_for(name: str, specs: Sequence[ProgramSpec]):
    if name == "bdp":
        return BDPActionAudit(specs)
    if name in POLICIES:
        return POLICIES[name]()
    if name == "hindsight_index":
        return HybridOracleBDP(specs, exact_prompt=True, exact_decode=True)
    raise ValueError(f"unknown trace policy {name!r}")


def phase_segments(samples: pd.DataFrame, dt: float) -> pd.DataFrame:
    """Compress trace-only program samples into contiguous Gantt segments."""
    rows: List[Dict[str, object]] = []
    for (policy, pid), part in samples.groupby(["policy", "pid"], sort=False):
        part = part.sort_values("t_s")
        start = 0.0
        phase = str(part.iloc[0].phase)
        previous = float(part.iloc[0].t_s)
        stage = int(part.iloc[0].stage)
        for row in part.iloc[1:].itertuples(index=False):
            current = float(row.t_s)
            if row.phase != phase or current - previous > 1.5 * dt:
                rows.append({
                    "policy": policy, "pid": int(pid), "phase": phase,
                    "stage": stage, "start_s": start, "end_s": previous + dt,
                })
                start = max(0.0, current - dt)
                phase = str(row.phase)
                stage = int(row.stage)
            previous = current
        rows.append({
            "policy": policy, "pid": int(pid), "phase": phase,
            "stage": stage, "start_s": start, "end_s": previous + dt,
        })
    return pd.DataFrame(rows)


def _save_policy_curve(policy: str, trace: pd.DataFrame, summary: Dict[str, float],
                       out_dir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    t = trace.t_s.to_numpy()

    ax = axes[0, 0]
    ax.fill_between(t, trace.kv_perc, color="#bfdbfe", alpha=0.55)
    ax.plot(t, trace.kv_perc, color="#2563eb", linewidth=1.2, label="KV used (%)")
    ax.set_ylabel("KV cache used (%)")
    ax.set_ylim(0, 105)
    twin = ax.twinx()
    twin.plot(t, trace.running, color="#dc2626", linewidth=1.0, label="running")
    twin.plot(t, trace.waiting, color="#f59e0b", linewidth=0.9, label="engine waiting")
    twin.set_ylabel("programs")
    lines = ax.lines + twin.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)
    ax.set_title("KV cache and engine concurrency")

    ax = axes[0, 1]
    ax.plot(t, trace.router_ready, label="router ready/paused", color="#6b7280")
    ax.plot(t, trace.tool, label="tool", color="#10b981")
    ax.plot(t, trace.prefill, label="prefill", color="#60a5fa")
    ax.plot(t, trace.decode, label="decode", color="#2563eb")
    ax.set_ylabel("programs")
    ax.set_title("Program-state load")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    window = max(1, int(round(10.0 / max(1e-9, np.median(np.diff(t))))))
    ax.plot(t, trace.decode_tps.rolling(window, min_periods=1).mean(),
            label="decode tok/s", color="#2563eb")
    ax.set_ylabel("decode tokens/s")
    prefill_ax = ax.twinx()
    prefill_ax.plot(t, trace.prefill_tps.rolling(window, min_periods=1).mean(),
                    label="prefill tok/s", color="#7c3aed", alpha=0.75)
    prefill_ax.set_ylabel("prefill tokens/s")
    ax.set_title("Delivered service rate")
    lines = ax.lines + prefill_ax.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    ax = axes[1, 1]
    ax.step(t, trace.completed_programs, where="post", color="#059669", linewidth=1.8)
    ax.axvline(summary["p50_program_ct"], color="#6b7280", linestyle="--", linewidth=1,
               label=f"P50 {summary['p50_program_ct']:.0f}s")
    ax.axvline(summary["p90_program_ct"], color="#d97706", linestyle="--", linewidth=1,
               label=f"P90 {summary['p90_program_ct']:.0f}s")
    ax.axvline(summary["p95_program_ct"], color="#dc2626", linestyle="--", linewidth=1,
               label=f"P95 {summary['p95_program_ct']:.0f}s")
    ax.set_ylabel("completed programs")
    ax.set_title("Completion curve and tail")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("wall-clock time (s)")
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        f"{policy} — mean CT {summary['mean_program_ct']:.1f}s, "
        f"makespan {summary['makespan']:.1f}s, preemptions {summary['active_preemptions']:.0f}",
        fontsize=14,
    )
    fig.savefig(os.path.join(out_dir, f"service_curve_{policy}.png"), dpi=170)
    plt.close(fig)


def _save_timeline(policy: str, samples: pd.DataFrame, completions: pd.DataFrame,
                   out_dir: str) -> None:
    order = completions.sort_values("completion_time_s").pid.astype(int).tolist()
    phases = list(PHASE_COLORS)
    code = {phase: index for index, phase in enumerate(phases)}
    times = np.sort(samples.t_s.unique())
    by_time = {float(t): index for index, t in enumerate(times)}
    by_pid = {pid: index for index, pid in enumerate(order)}
    matrix = np.full((len(order), len(times)), np.nan)
    for row in samples.itertuples(index=False):
        matrix[by_pid[int(row.pid)], by_time[float(row.t_s)]] = code[str(row.phase)]
    fig, ax = plt.subplots(figsize=(14, 13), constrained_layout=True)
    cmap = ListedColormap([PHASE_COLORS[phase] for phase in phases])
    cmap.set_bad("white")
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(0.0, float(times[-1]), -0.5, len(order) - 0.5),
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, len(phases) + 0.5), len(phases)),
    )
    ticks = list(range(0, len(order), 5))
    ax.set_yticks(ticks, [str(order[y]) for y in ticks])
    ax.set_ylim(-1, len(order))
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("program id, ordered by completion")
    ax.set_title(f"{policy} — per-program state timeline")
    handles = [plt.Line2D([0], [0], color=color, linewidth=7, label=phase)
               for phase, color in PHASE_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", ncol=len(handles), fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(os.path.join(out_dir, f"program_timeline_{policy}.png"), dpi=150)
    plt.close(fig)


def _save_comparison(timeseries: pd.DataFrame, completions: pd.DataFrame,
                     summary: pd.DataFrame, policies: Iterable[str], out_dir: str) -> None:
    policies = list(policies)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for policy in policies:
        part = timeseries[timeseries.policy == policy]
        color = COLORS.get(policy, None)
        axes[0, 0].step(part.t_s, part.completed_programs, where="post",
                        label=policy, color=color, linewidth=1.5)
        axes[0, 1].plot(part.t_s, part.kv_perc, label=policy, color=color, linewidth=1.0)
        window = max(1, int(round(10.0 / max(1e-9, np.median(np.diff(part.t_s))))))
        axes[1, 0].plot(part.t_s, part.decode_tps.rolling(window, min_periods=1).mean(),
                        label=policy, color=color, linewidth=1.0)
        ct = np.sort(completions[completions.policy == policy].completion_time_s.to_numpy())
        axes[1, 1].plot(np.arange(1, len(ct) + 1) / len(ct) * 100.0, ct,
                        label=policy, color=color, linewidth=1.5)

    titles = (
        "Cumulative completed programs", "Physical KV utilization",
        "Delivered decode throughput", "Completion-time quantiles",
    )
    ylabels = ("programs", "KV used (%)", "decode tokens/s, 10 s mean", "completion time (s)")
    xlabels = ("wall-clock time (s)", "wall-clock time (s)", "wall-clock time (s)", "program percentile")
    for ax, title, ylabel, xlabel in zip(axes.flat, titles, ylabels, xlabels):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    seed = int(summary.seed.iloc[0])
    fig.suptitle(f"Paired service-curve comparison — seed {seed}", fontsize=14)
    fig.savefig(os.path.join(out_dir, "policy_comparison.png"), dpi=170)
    plt.close(fig)


def _save_action_gap(audit: pd.DataFrame, timeseries: pd.DataFrame,
                     completions: pd.DataFrame, out_dir: str) -> None:
    bdp = timeseries[timeseries.policy == "bdp"].reset_index(drop=True)
    oracle = timeseries[timeseries.policy == "hindsight_index"].reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    active = audit[audit.ready > 0].copy()
    window = max(1, min(15, len(active)))
    axes[0, 0].plot(active.t_s, active.action_jaccard.rolling(window, min_periods=1).mean(),
                    color="#7c3aed")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].set_ylabel("admission-set Jaccard")
    axes[0, 0].set_title("BDP action agreement with hindsight on the same state")

    axes[0, 1].plot(active.t_s,
                    active.causal_actual_remaining_decode.rolling(window, min_periods=1).mean(),
                    color=COLORS["bdp"], label="BDP-selected")
    axes[0, 1].plot(active.t_s,
                    active.oracle_actual_remaining_decode.rolling(window, min_periods=1).mean(),
                    color=COLORS["hindsight_index"], label="hindsight-selected")
    axes[0, 1].set_ylabel("actual remaining decode tokens")
    axes[0, 1].set_title("Post-hoc heaviness of each action")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].step(bdp.t_s, bdp.completed_programs, where="post",
                    color=COLORS["bdp"], label="BDP")
    axes[1, 0].step(oracle.t_s, oracle.completed_programs, where="post",
                    color=COLORS["hindsight_index"], label="hindsight index")
    axes[1, 0].set_ylabel("completed programs")
    axes[1, 0].set_title("Different actions become a mid-run completion gap")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].step(bdp.t_s, 80 - bdp.completed_programs, where="post",
                    color=COLORS["bdp"], label="BDP")
    axes[1, 1].step(oracle.t_s, 80 - oracle.completed_programs, where="post",
                    color=COLORS["hindsight_index"], label="hindsight index")
    axes[1, 1].set_yscale("symlog", linthresh=1)
    axes[1, 1].set_ylabel("unfinished programs")
    axes[1, 1].set_title("Tail drain")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("wall-clock time (s)")
        ax.grid(True, alpha=0.2)
    fig.suptitle("BDP versus hindsight: action difference → service-curve difference", fontsize=14)
    fig.savefig(os.path.join(out_dir, "bdp_hindsight_action_gap.png"), dpi=170)
    plt.close(fig)


def run(out_dir: str, seed: int, policies: Sequence[str], control_interval: float = 3.0,
        dt: float = 0.5) -> None:
    unknown = sorted(set(policies) - set(ONLINE_POLICIES) - set(DIAGNOSTICS))
    if unknown:
        raise ValueError(f"unknown policies {unknown}")
    os.makedirs(out_dir, exist_ok=True)
    cfg = matrix("coder_experiment", seed)[0]
    specs, workload = generate_workload(cfg)
    service = make_service_model("qwen3_coder_30b_vllm_012")
    sim_cfg = SimConfig(control_interval=control_interval, dt=dt)
    time_rows: List[pd.DataFrame] = []
    sample_rows: List[pd.DataFrame] = []
    event_rows: List[pd.DataFrame] = []
    completion_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    action_audits: List[pd.DataFrame] = []

    for name in policies:
        policy = policy_for(name, specs)
        sim = Simulator(cfg, specs, policy, service, sim_cfg, record_trace=True)
        result = sim.run()
        summary_rows.append({"seed": seed, **workload, **result, "policy": name})
        time_rows.append(pd.DataFrame(sim.timeseries).assign(policy=name))
        sample_rows.append(pd.DataFrame(sim.program_samples).assign(policy=name))
        event_rows.append(pd.DataFrame(sim.scheduler_events).assign(policy=name))
        if isinstance(policy, BDPActionAudit):
            action_audits.append(pd.DataFrame(policy.rows).assign(policy=name))
        for p in sim.p:
            completion_rows.append({
                "seed": seed,
                "policy": name,
                "pid": p.pid,
                "completion_time_s": p.completion_time,
                "rounds": len(p.spec.requests),
                "total_prompt_tokens": sum(req.prompt_tokens for req in p.spec.requests),
                "total_decode_tokens": sum(req.decode_tokens_actual for req in p.spec.requests),
            })
        print(f"seed={seed} policy={name} mean_ct={result['mean_program_ct']:.2f}", flush=True)

    timeseries = pd.concat(time_rows, ignore_index=True)
    samples = pd.concat(sample_rows, ignore_index=True)
    events = pd.concat(event_rows, ignore_index=True)
    completions = pd.DataFrame(completion_rows)
    summary = pd.DataFrame(summary_rows)
    segments = phase_segments(samples, dt)

    timeseries.to_csv(os.path.join(out_dir, "timeseries.csv"), index=False)
    events.to_csv(os.path.join(out_dir, "scheduler_events.csv"), index=False)
    completions.to_csv(os.path.join(out_dir, "program_completions.csv"), index=False)
    segments.to_csv(os.path.join(out_dir, "program_segments.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    audit = pd.concat(action_audits, ignore_index=True) if action_audits else pd.DataFrame()
    if not audit.empty:
        audit.to_csv(os.path.join(out_dir, "bdp_action_audit.csv"), index=False)
    for name in policies:
        trace = timeseries[timeseries.policy == name].reset_index(drop=True)
        row = summary[summary.policy == name].iloc[0].to_dict()
        _save_policy_curve(name, trace, row, out_dir)
        _save_timeline(
            name,
            samples[samples.policy == name],
            completions[completions.policy == name],
            out_dir,
        )
    _save_comparison(timeseries, completions, summary, policies, out_dir)
    if not audit.empty and "hindsight_index" in policies:
        _save_action_gap(audit, timeseries, completions, out_dir)
    with open(os.path.join(out_dir, "experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "setting": "80_programs_all_arrive_at_t0",
            "seed": seed,
            "policies": list(policies),
            "diagnostic_only": [x for x in policies if x in DIAGNOSTICS],
            "service_profile": "qwen3_coder_30b_vllm_012",
            "control_interval_s": control_interval,
            "sample_interval_s": dt,
        }, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/service_curve")
    parser.add_argument("--seed", type=int, default=20262001)
    parser.add_argument("--policies", default="fcfs,thunder,bdp,hindsight_index")
    parser.add_argument("--control-interval", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()
    run(
        args.out,
        args.seed,
        [x.strip() for x in args.policies.split(",") if x.strip()],
        args.control_interval,
        args.dt,
    )


if __name__ == "__main__":
    main()
