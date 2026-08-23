# Reproducing the experiments

## Setup

```bash
pip install -r requirements.txt
pip install -e ThunderAgent
scripts/fetch_data.sh          # downloads SWE-bench Verified into data/
```

Hardware: the reported runs used one 140 GB GPU (sm_90). 80-way concurrency at
40960 context needs a large KV pool; smaller cards work with fewer workers.

### What gets downloaded

The repo is ~21 MB. Everything heavy is fetched or built on the target machine:

| what | size | how |
|---|---|---|
| pip deps (torch, vLLM, CUDA libs) | ~16 GB | `pip install -r requirements.txt` |
| model weights, coder preset (`Qwen3-Coder-30B-A3B-Instruct`) | ~57 GB | HuggingFace, on first server start |
| model weights, reasoning preset (`Qwen3-32B`) | ~62 GB | only if you run that preset |
| SWE-bench Verified | ~8 MB | `scripts/fetch_data.sh` |
| 80 testbeds: repo clones + per-instance conda envs | ~9 GB + ~12 GB | `python3 algorithm/prebuild96.py` |

So budget roughly **90 GB of disk for one preset** (140 GB for both), plus a
conda install — SWE-bench's setup scripts need one, and the harness rewrites
their Docker paths to it. It is auto-detected; `AGENT_EXP_CONDA` overrides.

Build the testbeds before the first run, not during it: a build takes minutes
and would otherwise be measured as serving time.

```bash
python3 algorithm/prebuild96.py            # builds the 80 instances in data/ids80.txt
```

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
