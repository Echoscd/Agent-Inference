#!/usr/bin/env python3
"""Decision-level trace and visualization for paired scheduler simulations.

The tracer deliberately lives outside the simulator's policy view: it records
realized tool returns and decode completions for post-hoc diagnosis, while the
policies still receive only ``SystemView`` and therefore remain non-clairvoyant.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from realistic_agentic_sim import (
    DECODE,
    DONE,
    POLICIES,
    PREFILL,
    READY,
    TOOL,
    Plan,
    SimConfig,
    Simulator,
    WorkloadConfig,
    generate_workload,
    make_service_model,
    matrix,
)


class TraceSimulator(Simulator):
    """Simulator subclass that records decisions without changing semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_policy = self.policy.name
        self.snapshots: List[Dict] = []
        self.candidates: List[Dict] = []
        self.events: List[Dict] = []
        self.transitions: List[Dict] = []
        self._sequence = 0
        for p in self.p:
            self._transition(p.pid, "START", READY, 0.0)

    def _base(self, pid: int, at: float) -> Dict:
        p = self.p[pid]
        tool_type = p.spec.requests[p.idx - 1].tool_type_after if p.phase == TOOL and p.idx > 0 else None
        return {
            "policy": self.trace_policy,
            "time": float(at),
            "pid": pid,
            "stage": p.stage,
            "phase": p.phase,
            "prefix_tokens": p.prefix,
            "prompt_tokens": p.prompt,
            "cache_warm": bool(p.cache_warm),
            "tool_type": tool_type,
            "tool_elapsed": p.tool_elapsed,
            "tool_remaining_actual": p.tool_remaining if p.phase == TOOL else np.nan,
        }

    def _event(self, name: str, pid: int, at: float, **extra) -> None:
        row = self._base(pid, at)
        row.update(event=name, sequence=self._sequence, **extra)
        self._sequence += 1
        self.events.append(row)

    def _transition(self, pid: int, old: str, new: str, at: float) -> None:
        self.transitions.append({
            "policy": self.trace_policy,
            "time": float(at),
            "sequence": self._sequence,
            "pid": pid,
            "old_phase": old,
            "new_phase": new,
        })
        self._sequence += 1

    def run_scheduler(self) -> None:
        view = self.view()
        start = time.perf_counter()
        plan = self.policy.plan(view)
        elapsed = time.perf_counter() - start
        admitted = set(plan.admit)
        kept = set(plan.keep_cache) - admitted
        evicted = [p for p in view.caches if p.pid not in admitted and p.pid not in kept]
        planned = {p.pid: p for p in view.programs}
        runtime = {p.pid: p for p in self.p}
        cache_ids = {p.pid for p in view.caches}
        evicted_ids = {p.pid for p in evicted}
        self.snapshots.append({
            "policy": self.trace_policy,
            "time": self.now,
            "active": len(view.active),
            "ready": len(view.ready),
            "tool": sum(p.phase == TOOL for p in view.programs),
            "warm_caches": len(view.caches),
            "physical_kv_blocks": self.physical_used(),
            "reserved_kv_blocks": self.planning_used(),
            "capacity_blocks": self.capacity_blocks,
            "n_admit": len(plan.admit),
            "n_keep": len(kept),
            "n_evict": len(evicted),
            "admit_ids": ";".join(map(str, plan.admit)),
            "warm_admit_ids": ";".join(
                str(pid) for pid in plan.admit
                if pid in planned and planned[pid].cache_warm and planned[pid].prefix > 0
            ),
            "cold_admit_ids": ";".join(
                str(pid) for pid in plan.admit
                if pid in planned and not (planned[pid].cache_warm and planned[pid].prefix > 0)
            ),
            "keep_ids": ";".join(map(str, sorted(kept))),
            "evict_ids": ";".join(str(p.pid) for p in evicted),
            "decision_ms": 1_000.0 * elapsed,
        })
        for p in view.programs:
            if p.phase != READY and p.pid not in cache_ids:
                continue
            if p.pid in admitted:
                decision = "ADMIT"
            elif p.pid in kept:
                decision = "KEEP"
            elif p.pid in evicted_ids:
                decision = "EVICT"
            elif p.phase == READY:
                decision = "WAIT"
            else:
                continue
            self.candidates.append({
                "policy": self.trace_policy,
                "time": self.now,
                "pid": p.pid,
                "decision": decision,
                "phase": p.phase,
                "stage": p.stage,
                "prefix_tokens": p.prefix,
                "prompt_tokens": p.prompt,
                "admit_blocks": p.admit_blocks,
                "cache_blocks": p.cache_blocks,
                "cache_warm": p.cache_warm,
                "waiting_age": p.waiting_age,
                "tool_type": p.tool_type,
                "tool_elapsed": p.tool_elapsed,
                "return_prob_10s": view.return_prob(p, 10.0),
                "tool_remaining_actual": runtime[p.pid].tool_remaining if p.phase == TOOL else np.nan,
                "terminal_prob": p.terminal_prob,
                "remaining_rounds": p.remaining_rounds,
            })
        self.decision_calls += 1
        self.decision_time += elapsed
        self.decision_samples.append(elapsed)
        self.apply_plan(plan)

    def evict(self, p, emergency: bool = False) -> None:
        if p.cache_warm and p.phase in (TOOL, READY):
            self._event(
                "EMERGENCY_EVICT" if emergency else "EVICT",
                p.pid,
                self.now,
                cache_blocks=self.cache_blocks(p),
                tool_elapsed=p.tool_elapsed,
            )
        super().evict(p, emergency=emergency)

    def admit(self, p) -> bool:
        old_phase = p.phase
        was_warm = bool(p.cache_warm and p.prefix > 0)
        waiting = max(0.0, self.now - p.ready_since)
        reserve = self.admit_blocks(p)
        ok = super().admit(p)
        if ok:
            self._event(
                "ADMIT_WARM" if was_warm else "ADMIT_COLD",
                p.pid,
                self.now,
                waiting_age=waiting,
                reserve_blocks=reserve,
            )
            self._transition(p.pid, old_phase, p.phase, self.now)
        return ok

    def advance_tools(self, dt: float) -> None:
        before = {p.pid: p.phase for p in self.p}
        super().advance_tools(dt)
        at = self.now + dt
        for p in self.p:
            if before[p.pid] == TOOL and p.phase == READY:
                self._event("TOOL_RETURN", p.pid, at)
                self._transition(p.pid, TOOL, READY, at)

    def complete_turn(self, p, at: float) -> None:
        completed_stage = p.stage
        super().complete_turn(p, at)
        self._event("PROGRAM_DONE" if p.phase == DONE else "TOOL_START", p.pid, at,
                    completed_stage=completed_stage)
        self._transition(p.pid, DECODE, p.phase, at)

    def advance_gpu(self, dt: float) -> None:
        before = {p.pid: p.phase for p in self.p}
        super().advance_gpu(dt)
        at = self.now + dt
        for p in self.p:
            if before[p.pid] == PREFILL and p.phase == DECODE:
                self._event("PREFILL_DONE", p.pid, at)
                self._transition(p.pid, PREFILL, DECODE, at)


def _state_matrix(transitions: pd.DataFrame, policy: str, n_programs: int,
                  grid: np.ndarray) -> np.ndarray:
    phase_code = {READY: 0, PREFILL: 1, DECODE: 2, TOOL: 3, DONE: 4}
    out = np.zeros((n_programs, len(grid)), dtype=np.int8)
    part = transitions[transitions.policy == policy]
    for pid in range(n_programs):
        q = part[part.pid == pid].sort_values(["time", "sequence"], kind="stable")
        times = q.time.to_numpy(float)
        phases = q.new_phase.map(phase_code).to_numpy(np.int8)
        idx = np.searchsorted(times, grid, side="right") - 1
        idx = np.maximum(idx, 0)
        out[pid] = phases[idx]
    return out


def plot_overview(snapshots: pd.DataFrame, policies: Sequence[str], out_dir: str) -> None:
    fig, axes = plt.subplots(3, len(policies), figsize=(7 * len(policies), 10), squeeze=False)
    for col, policy in enumerate(policies):
        q = snapshots[snapshots.policy == policy].sort_values("time")
        axes[0, col].plot(q.time, q.active, label="active", linewidth=1.4)
        axes[0, col].plot(q.time, q.ready, label="ready", linewidth=1.1)
        axes[0, col].plot(q.time, q.tool, label="tool", linewidth=1.1)
        axes[0, col].set_title(policy)
        axes[0, col].set_ylabel("programs")
        axes[0, col].legend(fontsize=8)

        physical = q.physical_kv_blocks / q.capacity_blocks
        reserved = q.reserved_kv_blocks / q.capacity_blocks
        axes[1, col].plot(q.time, physical, label="physical KV")
        axes[1, col].plot(q.time, reserved, label="reserved KV", alpha=0.85)
        axes[1, col].axhline(1.0, color="black", linewidth=0.7)
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].set_ylabel("KV utilization")
        axes[1, col].legend(fontsize=8)

        axes[2, col].step(q.time, q.n_admit, where="post", label="admit/tick")
        axes[2, col].step(q.time, q.n_evict, where="post", label="evict/tick")
        axes[2, col].set_xlabel("simulation time (s)")
        axes[2, col].set_ylabel("decisions")
        axes[2, col].legend(fontsize=8)
    fig.suptitle("Scheduler trace overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trace_overview.png"), dpi=180)
    plt.close(fig)


def plot_event_raster(events: pd.DataFrame, policies: Sequence[str], out_dir: str,
                      early_window: float) -> None:
    fig, axes = plt.subplots(len(policies), 1, figsize=(15, 5 * len(policies)), squeeze=False)
    styles = {
        "ADMIT_WARM": ("o", "#2ca02c", 24),
        "ADMIT_COLD": ("o", "#ff7f0e", 24),
        "EVICT": ("x", "#d62728", 28),
        "EMERGENCY_EVICT": ("X", "#8c0000", 32),
        "TOOL_RETURN": (".", "#9467bd", 12),
        "PROGRAM_DONE": ("*", "black", 45),
    }
    for row, policy in enumerate(policies):
        ax = axes[row, 0]
        q = events[(events.policy == policy) & (events.time <= early_window)]
        for event, (marker, color, size) in styles.items():
            z = q[q.event == event]
            if len(z):
                ax.scatter(z.time, z.pid, marker=marker, c=color, s=size, label=event,
                           alpha=0.85, linewidths=0.8)
        ax.set_title(f"{policy}: early decision events")
        ax.set_ylabel("program id")
        ax.set_xlim(0, early_window)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(ncol=3, fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("simulation time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "early_decision_raster.png"), dpi=190)
    plt.close(fig)


def plot_state_heatmap(transitions: pd.DataFrame, policies: Sequence[str], n_programs: int,
                       makespan: float, step: float, out_dir: str) -> None:
    grid = np.arange(0.0, makespan + step, step)
    cmap = ListedColormap(["#bdbdbd", "#ffb347", "#2b83ba", "#8e63b0", "#ffffff"])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)
    fig, axes = plt.subplots(len(policies), 1, figsize=(16, 5.5 * len(policies)), squeeze=False)
    for row, policy in enumerate(policies):
        state = _state_matrix(transitions, policy, n_programs, grid)
        ax = axes[row, 0]
        ax.imshow(state, aspect="auto", interpolation="nearest", origin="lower",
                  extent=[grid[0], grid[-1], -0.5, n_programs - 0.5], cmap=cmap, norm=norm)
        ax.set_title(f"{policy}: per-program state")
        ax.set_ylabel("program id")
    axes[-1, 0].set_xlabel("simulation time (s)")
    handles = [plt.Line2D([0], [0], color=cmap(i / 4.0), linewidth=8, label=name)
               for i, name in enumerate([READY, PREFILL, DECODE, TOOL, DONE])]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "program_state_heatmap.png"), dpi=180)
    plt.close(fig)


def write_summary(snapshots: pd.DataFrame, metrics: pd.DataFrame, policies: Sequence[str],
                  out_dir: str) -> None:
    lines = ["Paired scheduler trace summary", "=" * 30, ""]
    for policy in policies:
        q = snapshots[snapshots.policy == policy].sort_values("time")
        zero = q.iloc[0]
        m = metrics[metrics.policy == policy].iloc[0]
        lines.extend([
            policy,
            "-" * len(policy),
            f"t=0 admitted request IDs: {zero.admit_ids or '(none)'}",
            f"t=0 evicted request IDs: {zero.evict_ids or '(none)'}",
            f"mean CT={m.mean_program_ct:.3f}s, makespan={m.makespan:.3f}s, "
            f"cache hit={m.cache_hit_prefix_ratio:.4f}",
            "first scheduler decisions:",
        ])
        for _, x in q.head(12).iterrows():
            lines.append(
                f"  t={x.time:8.3f} active={int(x.active):2d} ready={int(x.ready):2d} "
                f"admit=[{x.admit_ids}] evict=[{x.evict_ids}]"
            )
        lines.append("")
    with open(os.path.join(out_dir, "trace_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_trace(preset: str, scenario_index: int, seed0: int, policy_names: Sequence[str],
              service_profile: str, control_interval: float, dt: float, out_dir: str,
              early_window: float, state_step: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    configs = matrix(preset, seed0)
    if not 0 <= scenario_index < len(configs):
        raise ValueError(f"scenario-index must be in [0, {len(configs) - 1}]")
    cfg: WorkloadConfig = configs[scenario_index]
    specs, workload_meta = generate_workload(cfg)
    service = make_service_model(service_profile)
    sim_cfg = SimConfig(control_interval=control_interval, dt=dt)

    all_snapshots, all_candidates, all_events, all_transitions, metric_rows = [], [], [], [], []
    for policy_name in policy_names:
        if policy_name not in POLICIES:
            raise ValueError(f"unknown policy {policy_name!r}")
        sim = TraceSimulator(cfg, specs, POLICIES[policy_name](), service=service, sim_cfg=sim_cfg)
        result = sim.run()
        all_snapshots.extend(sim.snapshots)
        all_candidates.extend(sim.candidates)
        all_events.extend(sim.events)
        all_transitions.extend(sim.transitions)
        metric_rows.append(result)

    snapshots = pd.DataFrame(all_snapshots)
    candidates = pd.DataFrame(all_candidates)
    events = pd.DataFrame(all_events)
    transitions = pd.DataFrame(all_transitions)
    metrics = pd.DataFrame(metric_rows)
    workload = pd.DataFrame([
        {
            "pid": pid,
            "rounds": len(spec.requests),
            "total_prompt_tokens": sum(r.prompt_tokens for r in spec.requests),
            "total_decode_tokens": sum(r.decode_tokens_actual for r in spec.requests),
            "total_tool_seconds": sum(r.tool_duration_actual for r in spec.requests),
        }
        for pid, spec in enumerate(specs)
    ])

    snapshots.to_csv(os.path.join(out_dir, "scheduler_snapshots.csv"), index=False)
    candidates.to_csv(os.path.join(out_dir, "scheduler_candidates.csv"), index=False)
    events.to_csv(os.path.join(out_dir, "decision_events.csv"), index=False)
    transitions.to_csv(os.path.join(out_dir, "state_transitions.csv"), index=False)
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    workload.to_csv(os.path.join(out_dir, "workload_programs.csv"), index=False)
    if len(policy_names) == 2:
        left = snapshots[snapshots.policy == policy_names[0]].copy()
        right = snapshots[snapshots.policy == policy_names[1]].copy()
        compare = left.merge(right, on="time", suffixes=(f"_{policy_names[0]}", f"_{policy_names[1]}"))
        different = compare[
            (compare[f"admit_ids_{policy_names[0]}"] != compare[f"admit_ids_{policy_names[1]}"])
            | (compare[f"keep_ids_{policy_names[0]}"] != compare[f"keep_ids_{policy_names[1]}"])
            | (compare[f"evict_ids_{policy_names[0]}"] != compare[f"evict_ids_{policy_names[1]}"])
        ]
        different.to_csv(os.path.join(out_dir, "paired_decision_differences.csv"), index=False)
    with open(os.path.join(out_dir, "trace_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "preset": preset,
            "scenario_index": scenario_index,
            "workload": asdict(cfg),
            "workload_meta": workload_meta,
            "service_profile": service_profile,
            "service": asdict(service),
            "sim_config": asdict(sim_cfg),
            "policies": list(policy_names),
        }, f, indent=2)

    plot_overview(snapshots, policy_names, out_dir)
    plot_event_raster(events, policy_names, out_dir, early_window)
    plot_state_heatmap(transitions, policy_names, cfg.n_programs,
                       float(metrics.makespan.max()), state_step, out_dir)
    write_summary(snapshots, metrics, policy_names, out_dir)
    print(metrics[["policy", "mean_program_ct", "p90_program_ct", "makespan",
                   "cache_hit_prefix_ratio", "mean_active", "mean_decode_batch"]].to_string(index=False))
    print(f"trace written to {os.path.abspath(out_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "validation", "test", "real_proxy_smoke", "real_proxy"],
                        default="real_proxy_smoke")
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--seed0", type=int, default=20260717)
    parser.add_argument("--policies", default="thunder_greedy,hazard_grade_knapsack")
    parser.add_argument("--service-profile", default="qwen3_32b_vllm_proxy")
    parser.add_argument("--control-interval", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--early-window", type=float, default=600.0)
    parser.add_argument("--state-step", type=float, default=5.0)
    parser.add_argument("--out", default="results/trace_real_proxy")
    args = parser.parse_args()
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    run_trace(args.preset, args.scenario_index, args.seed0, policies,
              args.service_profile, args.control_interval, args.dt, args.out,
              args.early_window, args.state_step)


if __name__ == "__main__":
    main()
