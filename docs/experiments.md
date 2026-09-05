# Experiments

What each folder under `../result/` is. Figures (.png) and tables (.csv)
live inside each one.

Each subfolder = one experiment. Figures (.png) + tables (.csv) live inside.

- **01_prefix_cache_ab/** — single long trajectory, prefix-cache reuse vs drop.
  prefill 11.5x lower with reuse; decode unaffected. (prefill_decode_AB.png, summary.csv)
- **02_long_context_alpha/** — grow-to-max single stream, no cache.
  per-token prefill alpha(n) is U-shaped (~0.12ms min @5-10k, rises to 0.16ms via O(n^2) attn);
  per-token decode rises ~7% with KV. (per_token_vs_ctx.png)
- **03_concurrency32_nocache/** — 32 agents, no prefix cache, max-turns 3.
  prefill 3.7% of time; decode heavy-tailed (CV 0.55). (turns_prompt_histogram.png,
  prefill_decode_dist.png, per_agent.csv)
- **04_concurrency32_cache/** — 32 agents, prefix cache ON, unlimited turns.
  prefix hit 93.1%; KV peak 96% (no preemption); turns bimodal (most <=6, tail at 50);
  total-prompt ~ O(turns^2). (kv_curve_over_time.png, turns_prompt_histogram.png,
  prefill_decode_dist.png, per_agent.csv)

- **05_agent_behavior/** — single-instance transcripts diagnosing why runs end early.
  Found 2 harness bugs: (1) EDIT+SUBMIT in one message dropped the edit (parse priority);
  (2) SEARCH/REPLACE exact-match rejected correct fixes off by indentation (now
  whitespace-tolerant). Model output was complete (finish_reason=stop) & often correct;
  low resolve was scaffolding, not inference.
- **06_concurrency80_cache/** — 80 agents, prefix cache ON, KV OVERSUBSCRIBED.
  KV pinned 100%, running collapses 80->~16, waiting->54, preemptions->153,
  prefix hit collapses 93%->~40% (LRU eviction + recompute thrashing). vLLM 0.11 V1
  default = recompute preemption (no swap; num_cpu_blocks=None). Effective concurrency
  is bounded by KV pool (258k tokens / 16152 blocks), NOT compute.
  (kv_saturation_curve.png, kv_timeseries.csv, results.jsonl, run.log)

- **07_concurrency80_cpuoffload/** — (vLLM 0.11) CPU offload attempt: CRASHED with
  AssertionError in cpu_gpu.py transfer (experimental connector bug). Pre-crash:
  offload working (running stayed 80, preempt 0). Motivated the vLLM upgrade.
- **08_concurrency80_v012_nooffload/** — vLLM 0.12.0, no offload. Same thrash as 06
  (KV 100%, running collapses, preempts). Confirms version alone doesn't change the
  oversubscription baseline.
- **09_concurrency80_v012_offload/** — vLLM 0.12.0 + CPU offload. NO crash (upgrade
  fixed the bug). External (CPU-tier) prefix hit ~30%. Still thrashes at 80-way
  (GPU-bound active concurrency ~25); preempts MORE (220) due to offload+reload
  cycling. (kv_saturation_curve.png)
- **10_offload_comparison/** — COMPARISON.md, final_stats.json, compare_06_vs_09.png.
  Conclusion: at 80-way extreme oversubscription, CPU offload prevents the crash and
  adds a CPU tier (30% external hit) but does NOT raise the GPU-bound active-decode
  ceiling, so it cannot fix thrashing. Offload's net win needs MODERATE
  oversubscription (~48-way) where GPU alone overflows but GPU+CPU fits.
  vLLM 0.23 (latest) needs CUDA-13 driver; host is 12.8 -> 0.12.0 is newest runnable.

## Scheduling experiments (11 onward)

From here the question changes from "how does the server behave under
oversubscription" to "which program should hold KV". Every run from 15 on is an
A/B: arm A a baseline policy, arm B the policy under test, same 80 instances,
vLLM restarted cold before each arm.

- **11_thunderagent_80way/**, **11b_direct_baseline/**, **12_thunderagent_vs_direct/** —
  first ThunderAgent runs: 80-way through the proxy vs straight at vLLM. Establishes
  that proxy-side pause/resume changes the KV picture at all.
- **13_thunderagent_full80_trace/** — the reference 80-way traced run (per-program
  per-turn trace + KV/preempt time series). `algorithm/reproduce_run13.sh` rebuilds it.
- **14_thunder_evict/** — first eviction experiment.
- **15_AB_density/**, **16_AB_density/** — first A/B harness runs. Both aborted early
  (15: 6 and 17 of 80 programs completed; 16: no summary), so they are process
  artifacts, not results.
- **17_AB_density/**, **18_AB_fidelity_dd/** — **INVALID, do not cite.** A config-wiring
  bug built the router before the config was applied, so both arms silently ran the
  default `size` policy regardless of `--policy`. The bug is fixed; these were never
  re-run.
- **19_AB_fidelity_vs_size/** — aborted, 0 of 80 programs completed in either arm.
- **20_dual_price_check/** — sanity check that the dual-descent price actually rises
  under oversubscription.
- **21_AB_size_vs_fidelity/**, **22_AB_size_vs_fidelity/** — size vs fidelity on
  Qwen3-32B, 80/80 completed. 22 is the one with the policy recorded in the decision
  trace; fidelity's tail latency is much worse (p95 +140% post-warmup).
- **23_AB_coder_size_vs_fidelity/** — same comparison on Qwen3-Coder-30B. Both arms
  hit the time limit (62 and 57 of 80 completed).
- **24_AB_size_vs_hazard/**, **25_AB_size_vs_hazard/** — size vs hazard_grade on
  Qwen3-32B. 25's arm B aborted, so only 24 is a usable pair.
- **26/27/28_AB_coder_size_vs_hazard/** — the headline comparison: size vs
  hazard_grade on Qwen3-Coder-30B, three replicates, 80/80 completed in every arm.
  Resolve favours hazard_grade in all three (14/11, 16/12, 14/11); throughput and
  tail-latency deltas are **not** significant at n=3 and change sign between
  replicates. Same config in all three (`scripts/run_replicates.sh coder <n> 1`).

- **30/31_AB_coder_size_vs_bdp/** — size vs `bdp`, the online port of the
  calibrated simulator's Bayesian-SERPT-plus-dual-price policy
  (branch `sim/minimal-calibrated-bdp`). Two replicates, same config as 26-29.
  The two runs **disagree in sign**: 30 has bdp at −6.9% throughput and +17.4%
  p95, 31 has +7.5% and −38.9%. Experiment 31's size arm is the reason — its
  total queue+prefill wait was 10,399 s against 6,000-6,500 s in the other three
  arms, driven by a 77.4% prefix-cache hit rate, the lowest of the four.
  Two things hold in both runs: bdp reaches the highest prefix-cache hit rate of
  any policy tried (85.0% mean vs 76.7% for size) and the fewest preemptions
  (2.0 vs 7.5), while running at size-like concurrency. It buys cache efficiency
  and spends it on admitting less; the net effect on latency is inside the noise.
  These are the first runs whose decision traces carry the policy's own scores
  and whose tapes carry absolute timestamps (`timeline: recorded`).

### Pooled comparison over the coder series

Averaging the six size arms (26-31) gives a baseline with a p95 standard
deviation of 28.3 s, or 22% of its mean. Against it, on the steady window:

| policy | n | throughput | mean latency | p95 | resolved | prefix hit | preemptions |
|---|---|---|---|---|---|---|---|
| size (baseline) | 6 | 625.4 | 29.4 | 129.8 | 13.0/80 | 76.7% | 7.5 |
| hazard family | 4 | 629.9 | 27.1 | 117.8 | 14.8/80 | 78.2% | 8.0 |
| bdp | 2 | 595.2 | 26.3 | 117.3 | 14.0/80 | **85.0%** | **2.0** |

`hazard_grade` (26-28) and `sim0823` (29) are two ports of the same algorithm
from different simulator revisions, so they are pooled as one family.

**The latency difference between the two families is not resolvable at this
sample size**: their p95 differs by 0.4% against a baseline that varies by 22%.
What is outside the noise is bdp's mechanism — +8.3 points of prefix-cache hit
rate and 75% fewer preemptions — which does not convert into an end-to-end win
because it also admits less.

- **33_warmup_steady_metrics/** — cross-run rollup, not a run: post-warmup and
  steady-state throughput and p90/p95 for every arm above. See its own README.

### Reading these folders

`results_<arm>_summary.json` is whole-run (includes the cold ramp and the drain
tail). `steady_metrics.json` splits the timeline into full / post-warmup / steady.
`kv_<arm>.csv` is the 0.5 s server sampling. Raw `tape_<arm>.jsonl` and
`decision_trace_<arm>.jsonl` are not committed — see the top-level README.

Runs before 22 have no `policy` field in their decision trace, and runs before
2026-08-23 have no absolute timestamps in their tape (the timeline is
reconstructed; each `steady_metrics` row says which applied).
