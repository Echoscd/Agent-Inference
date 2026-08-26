#!/usr/bin/env python3
"""Export a code-derived, machine-readable simulator setting.

The Markdown setting explains semantics.  This exporter captures the exact
numeric values used by a preset so reviewers can distinguish environment,
workload, policy, and result files.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict
from typing import Any, Dict

from realistic_agentic_sim import (
    POLICIES,
    PRIORS,
    TOOL_MIXES,
    TOOL_MODELS,
    SimConfig,
    SERVICE_PROFILE_KWARGS,
    make_service_model,
    matrix,
)


PRIOR_FAMILIES = {
    "short": {"family": "discretized_gamma_weights", "shape": 2.0, "scale": 2.0},
    "coder16_empirical": {
        "family": "empirical_pmf_from_public_26_27_28_per_agent_traces",
        "sample_size": 480,
        "censor_at": 20,
    },
    "coder20_natural": {
        "family": "max_turns_mixture_before_context_limit_censoring",
        "max_turn_probability": 0.70,
        "early_exit_probability": 0.30,
        "early_exit_support": [3, 19],
    },
}


def _constructor_defaults(policy_name: str) -> Dict[str, Any]:
    signature = inspect.signature(POLICIES[policy_name])
    defaults: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is not inspect.Parameter.empty:
            defaults[name] = parameter.default
    policy = POLICIES[policy_name]()
    effective = {
        name: value
        for name, value in vars(policy).items()
        if isinstance(value, (str, int, float, bool)) and name not in {"lam", "epoch"}
    }
    return {"constructor_defaults": defaults, "effective_initial_values": effective}


def build_setting(preset: str, service_profile: str, seed0: int,
                  control_interval: float, dt: float) -> Dict[str, Any]:
    service = make_service_model(service_profile)
    sim_config = SimConfig(control_interval=control_interval, dt=dt)
    scenarios = matrix(preset, seed0)
    return {
        "schema_version": 1,
        "setting_kind": "simulator_environment_not_results",
        "preset": preset,
        "seed0": seed0,
        "units": {
            "time": "seconds",
            "token_count": "model tokens",
            "kv_capacity": "tokens rounded to physical blocks",
            "throughput": "tokens/second",
        },
        "execution_flow": [
            "generate one ProgramSpec workload per scenario",
            "reuse the identical realized ProgramSpec objects for every paired policy",
            "run outer scheduler at t=0, periodically, and whenever router-paused READY work cannot otherwise progress",
            "submit admitted turns to ENGINE_WAITING; throttle the all-at-once cold burst with the shared prefill/chunk budget, then use FCFS for sparse returns",
            "advance asynchronous tools and shared prefill/decode GPU service by event-bounded dt",
            "sample physical KV, states, token rates, completions, prefix hit, and preemptions at dt cadence",
            "expand output reservations by evicting inactive caches when necessary",
            "stop when every program reaches DONE",
        ],
        "program_state_machine": {
            "states": ["READY", "ENGINE_WAITING", "PREFILL", "DECODE", "TOOL", "DONE"],
            "normal_path": "READY -> ENGINE_WAITING -> PREFILL -> DECODE -> TOOL; retained TOOL returns directly to ENGINE_WAITING, paused TOOL returns to READY",
            "active_preemption": {
                "mode": "last-in-first-preempt to ENGINE_WAITING; dropped KV is recomputed inside the engine",
            },
            "tool_gpu_consumption": 0,
            "all_programs_release_at": 0.0,
        },
        "information_boundary": {
            "policy_observes": [
                "phase, stage, current prefix, revealed prompt",
                "warm/cold flag and block counts",
                "tool class and elapsed tool age",
                "completed per-turn decode lengths for mixture-aware Bayesian scale updates",
                "posterior terminal probability and expected remaining rounds",
                "the fixed context guard used to condition remaining rounds",
                "class-level prompt/decode/tool distributions",
                "service model, capacity, and control interval",
            ],
            "policy_does_not_observe": [
                "realized future decode length",
                "realized remaining tool duration",
                "realized total number of program turns",
            ],
        },
        "random_distributions": {
            "prompt_and_decode": {
                "family": "gamma_then_rounded_to_positive_integer",
                "shape_formula": "1 / CV^2",
                "scale_formula": "mean / shape",
                "program_decode_scale": "optional mean-one lognormal latent shared by a program's turns",
            },
            "turn_count_priors": {
                name: {
                    **PRIOR_FAMILIES[name],
                    "support": [1, prior.k_max],
                    "normalized_mean": prior.mean,
                }
                for name, prior in PRIORS.items()
            },
            "tool_duration": {
                name: {
                    "family": "lognormal",
                    "mean_seconds": tool.mean_s,
                    "cv": tool.cv,
                }
                for name, tool in TOOL_MODELS.items()
            },
            "tool_mixes": TOOL_MIXES,
        },
        "service_model": {
            "profile": service_profile,
            "parameters": asdict(service),
            "aggregate_prefill_formula": (
                "single_tps * n/(1+(n-1)/prefill_sat) / "
                "(1+prefill_context_penalty*mean_uncached/context_ref)"
            ),
            "aggregate_decode_formula": (
                "single_tps * n_eff/(1+(n_eff-1)/decode_sat) / "
                "(1+decode_context_penalty*mean_context/context_ref)"
            ),
            "fallback_mixed_gpu_sharing": {
                "prefill_share": service.mixed_prefill_share,
                "decode_share": 1.0 - service.mixed_prefill_share,
            },
            "measured_mixed_interference": {
                "prefill_multiplier": service.mixed_prefill_multiplier,
                "decode_multiplier": service.mixed_decode_multiplier,
            },
        },
        "kv_model": {
            "block_rounding": "ceil(tokens / block_size)",
            "inactive_cache_blocks": "ceil(prefix_tokens / block_size)",
            "admission_reservation": (
                "ceil((prefix + revealed_prompt + distributional_decode_quantile) / block_size)"
            ),
            "planning_used": "sum(active reserved blocks) + sum(inactive warm-cache blocks)",
            "physical_used": "active materialized KV blocks only",
            "warm_admission_prefill": "revealed incremental prompt only",
            "cold_admission_prefill": "evicted prefix + revealed incremental prompt",
            "router_overflow": "evict inactive retained programs by policy emergency order",
            "allocation": "materialize KV by physical token growth; recompute-preempt active requests on pressure",
            "apc": "completed/preempted blocks become hashes on the engine free list; allocation overwrites block-LRU hashes and may leave a partial prefix",
        },
        "simulation_config": asdict(sim_config),
        "policies": {
            name: _constructor_defaults(name)
            for name in POLICIES
        },
        "scenarios": [asdict(cfg) for cfg in scenarios],
        "paired_evaluation": {
            "paired_keys": ["scenario", "seed", "tool_mix", "capacity_tokens"],
            "baselines": ["fcfs", "thunder"],
            "primary_metric": "mean_program_completion_time",
        },
        "known_model_gaps": [
            "one worker and one KV pool",
            "aggregate fitted service curves rather than vLLM iteration traces",
            "program-prefix LRU rather than the exact global hash-block order",
            "preemption victim order is modeled, but vLLM's internal scheduler implementation is not replayed",
            "one aggregate chunked-prefill budget rather than exact per-iteration token scheduling",
            "no multiworker routing, locality, migration, or network cost",
            "pooled tool classes rather than per-tool semantic replay",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["coder_calibration_smoke", "coder_calibration", "coder_experiment"],
        required=True,
    )
    parser.add_argument(
        "--service-profile",
        choices=sorted(SERVICE_PROFILE_KWARGS),
        required=True,
    )
    parser.add_argument("--seed0", type=int, required=True)
    parser.add_argument("--control-interval", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    setting = build_setting(
        args.preset,
        args.service_profile,
        args.seed0,
        args.control_interval,
        args.dt,
    )
    parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(setting, handle, indent=2)
        handle.write("\n")
    print(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
