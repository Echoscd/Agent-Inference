# Metrics


All definitions live in **`algorithm/run_metrics.py`** — one percentile formula,
one tape parser, one warmup rule. `run_swebench_eval.py`, `plots.py`,
`concurrency_bench.py` and `warmup_metrics.py` all delegate to it, and
`algorithm/test_run_metrics.py` pins the semantics.

Two latency notions, deliberately distinct:

- **per-program** (`results_*_summary.json`): `ttft + decode + tool_wait` summed
  over an agent's turns — a whole agent's wall time, tool execution included.
- **per-LLM-call** (`steady_metrics.*`): `wait + decode` for one call — tool time
  excluded. `wait` is TTFT and bundles proxy pause, server queue and prefill;
  those three are not separable from the client side.

Windows: a run launches all 80 programs at once, so the head is a cold ramp and
the tail is a long drain. `full` = whole timeline, `warm` = warmup dropped,
`steady` = warmup and drain dropped. Warmup ends when KV utilisation first
reaches 90% of that run's own peak (relative, so a policy with a lower steady KV
level is not penalised). Throughput is overlap-weighted across window edges;
latency percentiles cover calls that *start* inside the window.

Statistics come from the **tape**, not `results.jsonl`: the tape records every
call as it happens, so it includes programs that never returned. `results.jsonl`
only has finishers, which biases toward the easy instances.
