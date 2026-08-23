# Agentic Serving KV Scheduling Project

This folder is a cleaned, runnable consolidation of the repository.  It keeps
the original simulator model, preserves the algorithms that appear in the code,
reports, and images, and adds the requested memory-retention statistics.

## Contents

- `code/agentic_kv_sim.py`: unified simulator and experiment runner.
- `IMPLEMENTATION.md`: implementation notes for moving the scheduler into a
  vLLM-style GPU serving loop.
- `report/agentic_serving_report.tex`: mathematical report with the model and
  algorithm steps.
- `results/`: generated CSVs, plots, and LaTeX result tables after running.

## Algorithms

Core policies:

- `commit_many`
- `least_rounds`
- `current_density`
- `bayes_grade`
- `dual_descent_current`
- `bayes_dual_price`
- `mu_two_price`
- `lp_dual_descent`
- `lp_selective`

All preserved policies:

- the core policies above
- `bayes_grade_warm_marginal`
- `area_knapsack`
- `lp_update_long`
- `lp_solve_once`
- `lp_infrequent`
- `mpc_short`
- `dual_threshold`

The main commitbase LP+Bayesian algorithm is `bayes_dual_price`.  It implements
the reduced margin

```text
Delta_run = 1 / (tau_i mu_i) - lambda_t qbar_i
Delta_hold = alpha A_i / mu_i - lambda_t A_i
```

with online projected-KV price updates.

## Run

From this folder:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python code/agentic_kv_sim.py --preset quick --out results/quick --policies core
```

To run every preserved policy:

```bash
python code/agentic_kv_sim.py --preset quick --out results/quick_all --policies all
```

The checked-in `RESULTS.md` summarizes the `results/quick_all` run.

Larger matrices:

```bash
python code/agentic_kv_sim.py --preset standard --out results/standard --policies core
python code/agentic_kv_sim.py --preset stress --out results/stress --policies core
```

The presets use decode means in the requested 100-200 range:

- `prefill_heavy`: `E[P]=160`, `E[D]=100`
- `balanced`: `E[P]=120`, `E[D]=150`
- `decode_heavy`: `E[P]=80`, `E[D]=200`

The quick comparison uses memory factors `f in {5,10,15,20}` with
`M = f * qmax_obs`.  It also increases injected program count with memory:
`n={170,320,480,640}`.  The quick prior is `quick10`, whose realized mean
request count is about 10 rounds per program.  The larger standard/stress
presets remain tighter at `f in {1,2,4}`.

## New Metrics

The simulator records both event counts and ratios:

- `preempt_events`: a warm inactive prefix is held in GPU memory while its next
  request is not admitted.
- `preempted_requests_unique`: unique `(program, request-stage)` pairs that were
  held at least once.
- `preempted_request_ratio`: unique preempted requests divided by completed
  requests.
- `memory_delete_events` / `evictions`: warm inactive prefixes removed from GPU
  memory.
- `deleted_from_memory_ratio`: delete events divided by hold-or-delete memory
  decisions.

Active requests remain non-preemptive, matching the original reports.
