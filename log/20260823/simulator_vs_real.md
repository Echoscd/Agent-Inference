# Simulator vs the real system: where they differ, and what to measure

Written against the calibration plan (E0–E5). This document does three things:

1. checks the plan's real-side numbers against the runs actually in `result/`,
   including the third replicate the plan did not have;
2. lists the concrete differences between the simulator's current settings and
   the real deployment;
3. audits what the plan asks us to measure against what the harness records
   today, so the instrumentation work is scoped before E1 starts.

## 1. The plan's real-side numbers are experiments 26 and 27

Every figure in the plan's "Real Size / Real hazard_grade" table is the mean of
`result/26_AB_coder_size_vs_hazard` and `result/27_...` (447.4 vs 447.5 quoted,
1191.2 vs 1191.0, 393.2 vs 393.3, 14.75, 5.83k, 237.7k, 54.4 vs 54.5, 77.84 vs
77.85, 23). The pairing is confirmed.

**A third replicate has since run (experiment 28, same config as 26), and it
changes one of the plan's premises.**

| | 26 size | 26 hazard | 27 size | 27 hazard | 28 size | 28 hazard |
|---|---|---|---|---|---|---|
| mean task latency (s) | 428.2 | 469.2 | 466.7 | 473.1 | 480.5 | **433.5** |
| makespan (s) | 1258.0 | 1022.7 | 1124.3 | 1116.5 | 1111.4 | 908.1 |
| generation throughput (tok/s) | 366.8 | 430.2 | 419.7 | 435.4 | 477.2 | 495.0 |
| turns per task | 14.6 | 15.3 | 14.9 | 15.1 | 14.7 | 15.1 |
| decode tokens per task | 5.77k | 5.50k | 5.90k | 6.08k | 6.63k | 5.62k |
| prompt tokens per task | 240.3k | 248.7k | 235.1k | 251.3k | 235.4k | 239.5k |
| average KV utilisation | 48.6% | 65.6% | 60.2% | 63.7% | 62.4% | 65.8% |
| prefix-cache hit rate | 78.1% | 77.0% | 77.6% | 80.5% | 70.7% | 79.3% |
| average running | 21.0 | 28.8 | 25.0 | 27.1 | 27.5 | 28.6 |
| vLLM preemptions | 8 | 5 | 7 | 7 | **19** | 6 |
| resolved / 80 | 11 | 14 | 12 | 16 | 11 | 14 |

Three-run means: size 458.5 s latency / 1164.6 s makespan / 421.2 tok/s;
hazard_grade 458.6 s / 1015.8 s / 453.5 tok/s.

### The consequence for E5

The plan states that on real hardware `hazard_grade` **increases** mean task
latency in both repetitions (+9.6%, +1.3%), and treats the simulator's −20.2%
prediction as a qualitative failure. With the third replicate the real delta is
**+9.6%, +1.4%, −9.8%** — the sign flips, and the three-run mean difference is
+0.02%, i.e. nothing.

So of the five policy deltas E5's pass condition requires, four are consistent
across all three replicates and one is not:

| policy delta (hazard vs size) | 26 | 27 | 28 | consistent? |
|---|---|---|---|---|
| makespan shorter | yes | yes | yes | **yes, 3/3** |
| throughput higher | yes | yes | yes | **yes, 3/3** |
| average running higher | yes | yes | yes | **yes, 3/3** |
| cache-hit rate similar | −1.1 pts | +2.9 pts | +8.6 pts | roughly, but 28 is wide |
| mean latency higher | +9.6% | +1.4% | **−9.8%** | **no** |

**Recommendation: drop mean task latency from E5's direction test, or restate it
as "within noise".** Requiring the simulator to reproduce the direction of a
quantity whose own sign is not stable across three real runs would either fail a
correct simulator or, worse, reward one that happens to match the two runs used
to write the plan. The other four deltas are fair tests.

Two caveats on the table itself. Experiment 28's size arm is an outlier on two
counts — 19 preemptions against 5–8 elsewhere, and a 70.7% prefix-cache hit rate
against ~78% — so its higher mean latency is partly a worse-behaved baseline
rather than a better hazard arm. And the arms are not paired: temperature 0 does
not make them run identical work, because vLLM batch variance drifts the
trajectories (typically ~15% of (program, turn) pairs differ). Task-level
bootstrap resampling, as the plan proposes, is the right response, but it cannot
recover pairing that the runs do not have.

## 2. Why the simulator is at a different operating point

The plan's `real_proxy_smoke` trace is not slightly off, it is a different
regime: 9495 s mean latency against 447 s, 18265 s makespan against 1191 s,
69.7 tok/s against 393 tok/s. Reading `sim/settings/real_proxy.json` against the
real deployment, the mismatches that plausibly produce that gap are:

### Workload (the largest single mismatch)

| | simulator setting | real, measured |
|---|---|---|
| turn count | `swebench9` prior, discretized gamma, mean **8.758** | **14.7–14.9** turns per task, censored at 20 |
| decode per turn | independent gamma, mean **1650** | ~390 tok/turn (5.8k per task over ~15 turns) |
| decode per task | ~15.9k tokens | **5.5–6.6k** tokens |
| release | all at t=0 | all at t=0 — **matches**, confirmed by the tape |
| tools | 3-class lognormal, `swebench_proxy` mix 0.45/0.35/0.2 | measured per call in the tape, not yet fitted |

The simulator does ~2x the decode work per task in ~0.64x the turns. Both errors
push in the same direction on decode-bound wall time. This alone can account for
most of the throughput and makespan gap, which is why the plan is right to put
empirical replay (E1) before anything else.

### Service model

`profile: qwen3_32b_vllm_proxy` — the profile is a **Qwen3-32B proxy while the
real runs use Qwen3-Coder-30B-A3B**, a mixture-of-experts model with very
different decode economics. `decode_single_tps: 55` and `decode_sat: 12` are
fitted to the wrong model. E2 must be redone on the actual model; this is not a
parameter refit but a wrong measurement target.

`mixed_prefill_share: 0.2` is a fixed constant. The real engine interleaves
prefill and decode under a token budget with chunked prefill, so the share is
a function of the queue, not a constant. E2's mixed test targets exactly this.

### KV and cache mechanisms

| | simulator | real (vLLM 0.12 V1) |
|---|---|---|
| admission | reserves prefix + prompt + a decode quantile up front | allocates incrementally as tokens are produced |
| active work | `active_non_preemptive: True` | vLLM **does** preempt active requests, by recompute |
| prefix retention | `partial_cache_retention: False`, all-or-nothing | block-level; partial reuse is normal |
| overflow | evict inactive caches, then stall | preempt-by-recompute, then continue |

These four differences interact: a simulator that reserves the whole decode up
front and never preempts will admit fewer programs and stall instead of
thrashing, which is consistent with its low average active count (12.0 against a
real 21–28) and its very low prefix-hit rate (14%/34% against a real 71–80%).

### Control cadence

`control_interval: 5.0` s in the simulator; the real router ticks at **3 s**
(`--scheduler-interval 3`), and vLLM schedules per engine step, far finer. E4's
step 6 covers this.

## 3. Instrumentation audit: what E1/E3 need vs what we record

This is the part that gates the work. Current per-run artifacts are
`tape_<arm>.jsonl` (one row per LLM call), `kv_<arm>.csv` (0.5 s samples),
`decision_trace_<arm>.jsonl` (one row per 3 s router tick), `results_<arm>.jsonl`
(one row per program) and the raw vLLM log.

### Per LLM call

| plan requires | status | source / what is missing |
|---|---|---|
| program release, turn-ready timestamps | **partial** | tape has `t_start_s` (request sent) since 2026-08-23; turn-ready = previous call end + tool time, derivable |
| router-admit timestamp | **partial** | `decision_trace` `admitted[]`, but only at 3 s granularity |
| engine-start, first-token, final-token | **partial** | first-token = `t_start_s + wait_s`; final = `+ decode_s`. Engine-start is **not separable** — `wait_s` is one number bundling router pause + vLLM queue + prefill |
| tool start/end | **have** | `tool_wait_s` plus the reconstructed timeline |
| total input tokens | **have** | `prompt_tokens` (submitted) |
| **cached vs uncached input tokens** | **missing** | not recorded per call; only a run-level cumulative hit rate exists. This is the quantity the plan says must not be conflated |
| cumulative context before/after | derivable | from the per-turn prompt token series |
| warm/cold admission, miss reason | **missing** | not recorded |
| prefill / decode / queue / tool time split | **partial** | decode and tool yes; prefill vs queue **not separable** (see above) |
| finish reason, 20-turn censoring | **have** | `results_*.jsonl` status/error; censoring visible as `turns == 20` |
| preemption time, mode, lost blocks, recomputed tokens | **missing** | only a cumulative count in `kv_*.csv`; per-request detail would have to come from the vLLM log or an engine hook |

### Per engine step / time bin

| plan requires | status |
|---|---|
| running, waiting counts | **have** (0.5 s) |
| prefill vs decode sequence counts | **missing** |
| tokens scheduled for prefill / decode | **missing** |
| `num_batched_tokens`, token budget, chunked-prefill events | **missing** |
| physical KV blocks used | **partial** — `kv_cache_usage_perc` only, not blocks |
| reusable prefix blocks, reserved blocks, total usable blocks | **missing** |
| prefix-cache queries / hits / denominator | **have** — `vllm:prefix_cache_queries` and `_hits`, **token-weighted** (this answers one E0 item outright) |
| router and engine ordering | **partial** — router order in `decision_trace`; engine order not recorded |
| GPU busy time | **missing** |

### Run manifest

**Missing entirely.** No run records the code commit, seeds, model revision,
tokenizer revision, vLLM arguments, or policy parameters. The plan is right that
without it two repetitions cannot be assumed paired. This is cheap to add and
should be done before E1, not after — every run from here on can carry a
`manifest.json`, and the existing runs can be reconstructed only approximately.

### What E0 can already answer from the current code

`algorithm/run_metrics.py` is the single definition site, so several E0 items
have answers today:

- **generation throughput** = generated tokens / wall time. Two variants exist
  and are both recorded: client-side aggregate over the whole run, and the
  server counter delta. The post-warmup work adds a third, overlap-weighted over
  a window. All three are in the same file.
- **task latency** = `ttft + decode + tool_wait` summed over a program's turns,
  so it **includes** tool time, router waiting, engine waiting and prefill (the
  first three are not separable inside `ttft`). A second, distinct notion —
  per-LLM-call latency, `wait + decode`, tool time excluded — is used for the
  window metrics. These must not be mixed.
- **cache hit rate** is token-weighted, denominator `vllm:prefix_cache_queries`.
- **KV utilisation** is `vllm:kv_cache_usage_perc`, i.e. the engine's own
  occupancy figure, not a scheduler reservation — which is exactly the
  distinction the plan asks to pin down, and it means the simulator's
  `planning_used` is **not** the comparable quantity; `physical_used` is.
- **preemptions** is the delta of `vllm:num_preemptions` over the run.

The remaining E0 item — what "prefill prompt tokens" means — has a clear answer
on the real side (`prompt_tokens` from the API usage field, i.e. **submitted**,
including cached tokens) and this is the conflation the plan warns about: the
simulator's figure includes cold recomputation. They are not comparable as they
stand, and the fix is the missing per-call cached/uncached split.

## 4. Suggested order of work

1. **Add the run manifest and the per-call cached/uncached token split.** Both
   are small; without them E1 cannot be validated and E0 cannot be closed.
2. **Split `wait_s`** into router-pause / queue / prefill. Everything downstream
   that wants to attribute latency to a mechanism needs this, and today it is one
   opaque number.
3. **E0 on the existing traces** — the answers above are most of it.
4. **E1 replay against experiment 26 or 28**, in exogenous mode. The turn-count
   and decode-length gaps are large enough that this step alone should move the
   simulator by an order of magnitude.
5. **E2 on Qwen3-Coder-30B**, not on a 32B proxy.
6. Then E3, E4, and E5 with the amended pass condition from §1.

One process note: the plan's validation protocol says to fit on one Size run and
validate on the second. There are now three Size runs, and experiment 28's
baseline arm is the outlier (19 preemptions, 70.7% cache hit). Fitting on 28
would bake that in. Use 26 or 27 for integration debugging and hold out the other
two.
