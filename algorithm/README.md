# algorithm/

The experiment harness: the agent, the concurrent driver, the metric layer and
the figures. The scheduling policies themselves live in `../ThunderAgent/`.

Nothing here hardcodes a machine path — `paths.py` resolves everything from the
repo root, overridable with `AGENT_EXP_ROOT`, `AGENT_EXP_RESULT`,
`AGENT_EXP_WORK_ROOT`, `AGENT_EXP_DATA`, `AGENT_EXP_CONDA`.

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

## Metrics

**`run_metrics.py` is the single source of truth.** One percentile formula, one
tape parser, one warmup rule, one throughput attribution. `run_swebench_eval.py`,
`plots.py`, `concurrency_bench.py` and `warmup_metrics.py` all delegate to it.
`test_run_metrics.py` pins the semantics.

- `RunArtifacts(run_dir, arm)` — `.calls`, `.kv`, `.programs`, `.windows()`
- `Window(name, t0, t1)` — `.gen_tokens()` overlap-weighted, `.metrics()`
- `RunMetrics` — nested dict for json, flat row for csv
- `pctl` / `dist` — the only percentile implementation

Statistics come from the **tape**, not `results.jsonl`: the tape is written as
each call happens, so it covers programs that never returned. `results.jsonl`
only has finishers, which biases toward the easy instances.

Two latency notions, deliberately different: per-program (`ttft + decode +
tool_wait`, tool time included) in `results_*_summary.json`, and per-LLM-call
(`wait + decode`, tool time excluded) in `steady_metrics.*`. `wait` is TTFT and
bundles proxy pause, server queue and prefill — not separable from the client.

## Files

- `paths.py` — repo-relative paths, conda auto-detection
- `run_metrics.py` / `test_run_metrics.py` — metric definitions and their tests
- `swebench_agent.py` — shared dataclasses (`TurnTiming`, `AgentResult`), config,
  and the bash-style ReAct agent (`--agent bash`, not used by the reported runs)
- `swebench_edit_agent.py` — the edit agent used in every reported experiment
- `swebench_local_harness.py` — Docker-free conda harness: build, apply patch,
  evaluate, reset. `--patch gold` validates an instance.
- `metrics.py` — vLLM `/metrics`: in-process `MetricsMonitor` plus a standalone
  sampler (`python3 metrics.py <out.csv> [url] [interval]`)
- `ab_tape.py` — the tape: env-gated recording (`AB_RECORD_TAPE`) and
  known-decode lookup (`AB_KNOWN_DECODE_TAPE`)
- `legacy/` — superseded HumanEval pipeline, kept for reference only

## History

The scheduling code was refactored out of the router into policy classes
(`ThunderAgent/ThunderAgent/scheduling/`), and the metric code was unified into
`run_metrics.py`. Earlier revisions of this file described `_program_density` and
"density branches in `_greedy_resume`" inside `router.py`; those no longer exist —
the router is pure orchestration and delegates to `self.policy`.
