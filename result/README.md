# Experiment results

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
