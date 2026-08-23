# Agentic serving: scheduling policies for multi-turn LLM agents

Two halves of the same question — **when an inference server is oversubscribed by
many concurrent multi-turn agents, which program should hold KV cache?**

- **`algorithm/` + `ThunderAgent/`** — the real system. 80 SWE-bench agents
  against one vLLM backend, routed through a scheduling proxy, measured end to end.
- **`sim/`** — the simulator the policies were designed in, plus its calibration
  against traces from the real runs.

Everything in `result/` was produced by the code here. Raw per-call tapes are not
committed (60 MB per arm); the summary statistics and figures derived from them are.

---

## Layout

```
algorithm/          experiment harness: the agent, the eval driver, all metrics
  run_AB_experiment.sh    one A/B: cold vLLM + arm A, cold vLLM + arm B
  run_swebench_eval.py    drives N concurrent agents, writes results + summary
  swebench_edit_agent.py  the edit agent (OPEN / RUN / EDIT / SUBMIT loop)
  run_metrics.py          ALL metric definitions (see "Metrics" below)
  warmup_metrics.py       CLI: post-warmup / steady-state metrics for a run
  plots.py                figures from a tape or a kv csv
  paths.py                repo-relative paths; nothing hardcodes a machine path
ThunderAgent/       the scheduling proxy (vendored, MIT, see "Attribution")
  ThunderAgent/scheduling/   one file per policy: size, density, dual_descent,
                             fidelity, hazard_grade, tool_hazard
scripts/            run_replicates.sh (launch experiments), fetch_data.sh
data/               ids80.txt (the 80 instances every run uses); benchmark
                    JSONL is downloaded, not vendored
result/             one folder per experiment, README.md explains each
sim/                the simulator + settings + calibration; sim/legacy_v1 is the
                    earlier generation kept for the lambda-trace figures
```

## Setup

```bash
pip install -r requirements.txt
pip install -e ThunderAgent
scripts/fetch_data.sh          # downloads SWE-bench Verified into data/
```

Hardware: the reported runs used one 140 GB GPU (sm_90). 80-way concurrency at
40960 context needs a large KV pool; smaller cards work with fewer workers.

## Reproduce an A/B

One replicate of the headline comparison (arm A = `size` baseline, arm B =
`hazard_grade`), on the coder model:

```bash
scripts/run_replicates.sh coder 29 1
```

This runs, per arm: a cold vLLM 0.12.0 backend, a ThunderAgent proxy with that
policy, and 80 concurrent edit agents at temperature 0 with a 6000 s cap. Both
arms restart vLLM first, so each begins with an empty KV and prefix cache. Arm B
replays arm A's decode lengths (`X-Decode-Len`) so a policy can score by the real
upcoming decode. Output lands in `result/29_AB_coder_size_vs_hazard/`:

| file | what |
|---|---|
| `results_<arm>.jsonl` + `_summary.json` | per-program results and aggregate stats |
| `steady_metrics.json` / `.csv` | full / post-warmup / steady-state windows |
| `kv_<arm>.csv` | 0.5 s samples of KV %, running, waiting, preemptions |
| `tape_<arm>.jsonl` | every LLM call incl. prompt + completion (gitignored) |
| `pdt_*.png`, `gantt_*.png`, `kv_compare.png`, `concurrency.png` | figures |

Budget ~40 min per replicate on the coder preset.

### What a fresh clone can and cannot recompute

Each run folder ships `results_<arm>.jsonl`, `results_<arm>_summary.json`,
`kv_<arm>.csv`, `steady_metrics.json`/`.csv` and the figures. The cross-run
rollup and its figure regenerate from those, no GPU needed:

```bash
python3 algorithm/plot_warmup_metrics.py     # -> result/33_warmup_steady_metrics/
```

`algorithm/warmup_metrics.py` cannot be re-run against the committed runs,
because it reads `tape_<arm>.jsonl` and those are excluded (60 MB per arm). The
`steady_metrics.*` it produced are committed, so the numbers are auditable; to
recompute them from raw calls you have to run the experiment yourself, which
writes a fresh tape.

## Metrics

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

## Known limitations

- **Trajectory drift.** Both arms run at temperature 0 and should reproduce
  turn for turn, but vLLM batch variance makes them diverge (typically ~15% of
  (program, turn) pairs differ). The A/B is therefore not a perfectly paired
  comparison; `run_AB_experiment.sh` prints the drift at the end of every run.
- **Absolute call timestamps** were added to the tape on 2026-08-23. Runs before
  that reconstruct the timeline by cumsumming per-turn durations, which omits
  client-side build/eval time between turns; each row's `timeline` field says
  which applied.
- **Replicate count.** The size-vs-hazard_grade comparison currently has three
  replicates (26, 27, 28). Throughput and tail-latency deltas are not significant
  at that n; the resolve-rate difference is consistent across all three.
- Per-turn `max_tokens` is 8192 and context is capped at 40960, so long agents
  end with a 400 from the server rather than a graceful stop. Those programs are
  in the stats (status `NO_PATCH` with an error string), by design.

## Attribution

`ThunderAgent/` is a vendored copy of
[HaoKang-Timmy/ThunderAgent](https://github.com/HaoKang-Timmy/ThunderAgent)
(MIT, Copyright (c) 2026 Hao Kang), with the scheduling-policy layer in
`ThunderAgent/ThunderAgent/scheduling/` and the router changes added for this
work. Its `LICENSE.md` is preserved. Upstream git history is not included.

vLLM 0.12.0 is used unmodified, from pip.
