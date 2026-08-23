# algorithm/

Workspace for integrating program-aware KV scheduling (ThunderAgent-style external
scheduling + the bayes_dual_price value-aware admission from
`final_agentic_serving_project`) into the SWE-bench serving experiments.

## reproduce_run13.sh

Self-contained driver that reproduces `result/13_thunderagent_full80_trace` — the
80-way SWE-bench edit-agent run served through ThunderAgent on a local vLLM 0.12
backend, with per-program per-turn traces + KV/preempt time series + the standard
prefill/decode/turn figure.

```bash
bash algorithm/reproduce_run13.sh              # exact run-13 config (80 workers, 20 turns)
bash algorithm/reproduce_run13.sh 48 12 my_run # 48 workers, 12 turns, -> result/my_run/
```

Pipeline: agents (`run_swebench_eval.py`, edit agent, `X-Session-ID`=program_id)
→ ThunderAgent proxy `:8300` (router=tr, tool-boundary pause/resume, active KV ≤ GPU
capacity, profiling) → vLLM 0.12 backend `:8000` (Qwen3-32B, 40960 ctx, prefix
caching, no CPU offload, flashinfer sampler off).

## run_density.sh

Same pipeline, but launches ThunderAgent with the **current_density** scheduling
policy (`--policy density`): admit/keep the highest value-density
`v = 1/(tau·footprint)` first, evict the lowest. `tau = alpha·footprint + decode_hat`,
`footprint = total_tokens`; `alpha` is offline-fixed from historical prefill/decode
(~0.03), `decode_hat` is the assumed-known decode tokens/turn. Doubles as the A/B
driver (pass `size` to get the baseline).

```bash
bash algorithm/run_density.sh                       # 80-way density -> result/14_density80/
bash algorithm/run_density.sh 80 20 14_size80 size  # size baseline for A/B
bash algorithm/run_density.sh 80 20 my 14d density 0.03 1000   # explicit alpha/decode_hat
```

The density policy lives in ThunderAgent: `config.py` (policy/alpha/decode_hat),
`scheduler/router.py` (`_program_density`, density branches in `_greedy_resume`
admission and `_pause_until_safe` eviction). Default `--policy size` keeps old behavior.

## run_AB_experiment.sh  (record-and-reproduce A/B)

Compares two scheduling policies on the **identical** SWE-bench workload:

- **Pass A** (baseline): real edit-agent through ThunderAgent, `policy=size`, **temp=0**.
  Records `tape_A.jsonl` — every LLM call's full prefill (messages) + decode
  (completion) + decode length.
- **Pass B** (algorithm): real edit-agent, `policy=density` (or future), temp=0. Reads
  `tape_A` and sends `X-Decode-Len` per call so the density policy scores by the
  **known** decode length; records `tape_B.jsonl`.

Both passes run the real agent at temperature 0 → the trajectory reproduces
turn-for-turn, so pass B's decode equals pass A's (verified by diffing tape_A vs
tape_B at the end). The vLLM backend is restarted before each pass (cold KV). The
only difference between arms is the ThunderAgent policy.

Each pass has a wall-clock **time limit** (7th arg, default 1800s). On hit, the agent
run is stopped, its partial `results`+`_summary.json` are kept (results.jsonl is
flushed per-agent; `run_swebench_eval` catches SIGTERM and writes the summary from
finished agents), and the experiment moves on (A done → B). Same limit on both passes =
a fixed time budget per arm (clean goodput comparison: who completes more in the same T).

```bash
bash algorithm/run_AB_experiment.sh                       # 80-way, B=density, 1800s/pass -> result/15_AB_density/
bash algorithm/run_AB_experiment.sh 48 12 my density 0.03 1000 900   # 48 workers, 12 turns, 900s/pass
```

Mechanism files: `ab_tape.py` (env-gated record / known-decode lookup; `AB_RECORD_TAPE`,
`AB_KNOWN_DECODE_TAPE`), `swebench_edit_agent._stream_call` (records each call, injects
`X-Decode-Len`), ThunderAgent `app.get_known_decode` + `Program.known_decode` +
`_program_density` (uses known decode when present, else `decode_hat`).
Outputs per arm: `results_{A,B}.jsonl` (+_summary.json), `kv_{A,B}.csv`,
`tape_{A,B}.jsonl`, `pdt_{A,B}.png`, `kv_compare.png`.

All pipeline code now lives in `algorithm/`. Big external artifacts stay in `../`:
`vllm_dev/` (editable vLLM 0.12), `ThunderAgent/` (proxy), `data/`, `swebench_runs/`
(built testbeds), `result/`, `ids80.txt`.

Core pipeline:
- `swebench_local_harness.py` — Docker-free conda grading harness (`Instance`:
  build/apply_patch/evaluate/reset; `--patch gold` validates an instance).
- `swebench_agent.py` — shared dataclasses (TurnTiming/AgentResult) + config +
  the bash-style ReAct agent (one bash command/turn; the `--agent bash` baseline).
- `swebench_edit_agent.py` — the edit-style ReAct agent used in all experiments
  (Reasoning + OPEN/EDIT/RUN/SUBMIT; program_id via X-Session-ID, release on done).
- `run_swebench_eval.py` — concurrent runner; `--agent bash|edit`; writes per-turn
  `trace` (results.jsonl) + aggregate `*_summary.json`.

Tooling (consolidated):
- `metrics.py` — vLLM /metrics: in-process `MetricsMonitor` (imported by the runner)
  AND a standalone time-series sampler (`python metrics.py <out.csv> [url] [interval]`).
  Replaces the old metrics_monitor.py + kv_sampler.py.
- `plots.py` — one CLI for all figures: `pdt | saturation | kvlog | compare | gantt`.
  Replaces plot_pdt/plot_saturation/plot_kv/plot_compare/make_results.

Utilities / experiments:
- `prebuild96.py` — parallel pre-build of testbeds. `diagnose_agent.py` — run one
  instance + save transcript. `cache_ab_experiment.py` — prefix-cache A/B.
  `concurrency_bench.py` — synthetic fixed-prompt concurrency stress.
- `serve_dev_vllm.sh` — start the vLLM 0.12 backend from `../vllm_dev`.
- `legacy/` — superseded HumanEval pipeline (react_pipeline.py, run_eval.py).

Outputs in `result/<tag>/`: results.jsonl (per-program trace), kv_timeseries.csv,
ta_profiles/step_profiles.csv (per-step prefill_s/decode_s/pause_s/tool_call_s),
vllm_backend.log, thunderagent.log, run.log, pdt.png.

## Next step
Replace ThunderAgent's greedy `_greedy_resume` (size/capacity bin-pack) with the
`bayes_dual_price` value-density ordering (Δrun/q_i) to test whether value-aware
admission beats greedy. vLLM's LRU eviction (`vllm_dev/.../kv_cache_utils.py`
FreeKVCacheBlockQueue) is the later HOLD/EVICT target.
