# 33 — Post-warmup steady-state throughput & tail latency

Re-analysis (no new GPU run) of every existing 80-way A/B run in `result/`.
Whole-run numbers in `results_*_summary.json` mix in the cold ramp (all 80
programs launch at once, KV fills from empty) and the long drain tail, so this
recomputes throughput and latency percentiles on windows that exclude them.

Windows, per arm:
- `full`   [0, T]
- `warm`   [t_warm, T]         — warmup dropped
- `steady` [t_warm, t_drain]   — warmup and drain tail dropped

`t_warm` / `t_drain` = first / last time KV utilisation reaches 90% of its run
max, read from `kv_<A|B>.csv`.

Source of truth is `tape_<A|B>.jsonl` (every LLM call, including programs that
never returned), not `results_*.jsonl` (finishers only). The tape has no
absolute timestamps, so a call's absolute start is the per-program cumsum of
(wait_s + decode_s + tool_wait_s). Reconstructed wall clock matches the KV
sampler within ~1% on most arms (74–91% on 19B/26A/27A, where client-side
build/eval time is unaccounted).

- throughput: overlap-weighted — a call's gen tokens are credited to a window
  in proportion to how much of its decode interval falls inside it.
- latency: per LLM call, `wait_s + decode_s` (tool time excluded), over calls
  that START inside the window.

Files: `steady_metrics.csv` / `.json` (all windows, p50/p90/p95/p99 for call
latency, queue wait and decode), `ab_delta.txt` (A→B deltas), `warmup_vs_full.png`.

Excluded as aborted: 19A, 19B, 25B.

All definitions now live in `algorithm/run_metrics.py` (RunArtifacts / Window /
RunMetrics + the single `pctl`/`dist`); `warmup_metrics.py` is a thin CLI over it,
and `run_swebench_eval.py`, `plots.py`, `concurrency_bench.py` share the same
percentile and tape parser. `algorithm/test_run_metrics.py` pins the semantics.

Tapes written from 2026-08-23 carry `t_start_s`, so the timeline is read rather
than reconstructed; each row's `timeline` field says which applied.

Now wired into the harness: `run_AB_experiment.sh` calls `warmup_metrics.py --out
$OUTDIR $OUTDIR` after the plots, so every NEW run drops its own
`steady_metrics.{json,csv}` in its result folder. `run_swebench_eval.py` also
emits **p95** now (in every `_stats` block plus `p95_latency_s`), next to the
p90/p99 it already had.

Regenerate:
    python3 algorithm/warmup_metrics.py result/1[7-9]* result/2[1-7]*
    python3 algorithm/plot_warmup_metrics.py
