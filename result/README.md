# result/

One folder per experiment. What each one is, and which ones are invalid, is
documented in [../docs/experiments.md](../docs/experiments.md).

Per folder: `results_<arm>.jsonl` + `_summary.json` (per-program, whole run),
`steady_metrics.json`/`.csv` (full / post-warmup / steady windows),
`kv_<arm>.csv` (0.5 s server sampling), and figures. Raw `tape_<arm>.jsonl` and
`decision_trace_<arm>.jsonl` are not committed — see
[../docs/reproducing.md](../docs/reproducing.md).

`33_warmup_steady_metrics/` is a cross-run rollup rather than a run, and keeps
its own README describing the analysis.
