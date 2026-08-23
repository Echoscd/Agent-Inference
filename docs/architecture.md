# Architecture

How a run is wired together, and where a scheduling policy plugs in.

## The pipeline

```
run_swebench_eval.py            N concurrent agents, one thread each
  └─ swebench_edit_agent.py     the agent loop: Reasoning + OPEN/RUN/EDIT/SUBMIT
       ├─ X-Session-ID: <program_id>   so the proxy can track a program
       ├─ X-Decode-Len: <n>            arm B only, the known decode from arm A's tape
       └─ ab_tape.py             records every call (prompt, completion, timing)
  └─ swebench_local_harness.py  conda testbeds, patch application, grading
  └─ metrics.py                 samples vLLM /metrics every 0.5 s
         ↓
ThunderAgent proxy :8300  (policy decides who holds KV)
         ↓
vLLM 0.12.0 backend :8000  (prefix caching on, 40960 context)
```

## Entry points

| script | what it does |
|---|---|
| `run_AB_experiment.sh` | one A/B: cold vLLM + arm A, cold vLLM + arm B, then verify, plot, and emit steady metrics |
| `run_swebench_eval.py` | the driver on its own: `--workers --max-turns --agent edit --model <served-name>` |
| `prebuild96.py` | build the conda testbeds up front (do this before any timing run) |
| `serve_dev_vllm.sh` | start just the backend |
| `warmup_metrics.py` | post-warmup / steady-state metrics for finished runs |
| `plots.py` | `pdt \| saturation \| kvlog \| compare \| gantt \| concurrency \| conckv` |
| `diagnose_agent.py` | run one instance, save the full transcript |
| `concurrency_bench.py` | synthetic fixed-prompt stress, no agent involved |
| `cache_ab_experiment.py` | prefix-cache reuse vs drop on one trajectory |

`../scripts/run_replicates.sh` wraps `run_AB_experiment.sh` for repeated runs.

## The A/B design

Arm A is the baseline policy (`size` by default), arm B the policy under test.
Both run the real agent at temperature 0 on the same 80 instances, and vLLM is
restarted before each arm so both start with an empty KV and prefix cache. Arm A
records `tape_A.jsonl`; arm B reads it and sends each call's **known** decode
length as `X-Decode-Len`, which is what lets a value-based policy score a program
by the decode it is about to do rather than by a guess.

Temperature 0 was meant to make the two arms reproduce turn for turn. It does
not, quite: vLLM batch variance makes them diverge, and the harness prints the
drift at the end of every run. Treat the A/B as two samples of the same workload
distribution, not as a paired comparison.

Each arm has a wall-clock limit (7th arg). On hit the arm stops, keeps its
partial results, and the experiment moves on — same limit on both arms, so it
doubles as a fixed time budget per arm.

## Where a policy lives

A scheduling algorithm is one file in `../ThunderAgent/ThunderAgent/scheduling/`
plus one line in `factory.py` and one entry in `__main__.py`'s `--policy`
choices. The router is pure orchestration and holds `self.policy`; there are no
policy branches left in it.

A policy answers six questions, as methods:

| decision | method |
|---|---|
| how much to admit | `peak_pad(state)` |
| admission order | `sort_key(state)` — larger = higher keep-priority |
| eviction order | `evict_key(state)` — defaults to `sort_key` |
| admission gate | `admits(state)` — greedy fill vs leave headroom vs price gate |
| eviction timing | `proactive_evictions(...)`, `on_epoch(...)` |
| eviction count | the router's `_pause_until_safe` loop; the policy only supplies order |

Existing policies: `size` (the baseline: grouped by program phase, largest
first), `density`, `dual_descent`, `fidelity`, `hazard_grade`, `tool_hazard`.
