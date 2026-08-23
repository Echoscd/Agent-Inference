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

## Getting started

```bash
pip install -r requirements.txt
pip install -e ThunderAgent
scripts/fetch_data.sh                 # SWE-bench Verified -> data/
python3 algorithm/prebuild96.py       # build the 80 conda testbeds first
scripts/run_replicates.sh coder 29 1  # one A/B replicate -> result/29_AB_coder_size_vs_hazard/
```

The repo is ~21 MB; the model weights, pip deps and testbeds it pulls on the
target machine are ~90 GB for one preset. Details, hardware notes and what a
fresh clone can recompute: [docs/reproducing.md](docs/reproducing.md).

## Documentation

| | |
|---|---|
| [docs/reproducing.md](docs/reproducing.md) | setup, downloads, running an A/B |
| [docs/architecture.md](docs/architecture.md) | pipeline, entry points, how a policy plugs in |
| [docs/metrics.md](docs/metrics.md) | every metric definition |
| [docs/experiments.md](docs/experiments.md) | what each `result/` folder is |
| [docs/limitations.md](docs/limitations.md) | what these results do not support |
| [worklog/](worklog/) | weekly TODOs and experiment progress |
| [log/](log/) | dated working notes and analyses |

## Attribution

`ThunderAgent/` is a vendored copy of
[HaoKang-Timmy/ThunderAgent](https://github.com/HaoKang-Timmy/ThunderAgent)
(MIT, Copyright (c) 2026 Hao Kang), with the scheduling-policy layer in
`ThunderAgent/ThunderAgent/scheduling/` and the router changes added for this
work. Its `LICENSE.md` is preserved. Upstream git history is not included.

vLLM 0.12.0 is used unmodified, from pip.
