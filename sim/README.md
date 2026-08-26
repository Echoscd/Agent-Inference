# Calibrated Agentic-Serving Simulator

Minimal single-worker Qwen3-Coder-30B/vLLM-0.12 simulator for admission-policy
experiments under KV pressure. Every experiment releases 80 agent programs
together at `t=0`.

The project has exactly two baselines:

- `fcfs`: strict first-come, first-served;
- `thunder`: the single-worker ThunderAgent action policy.

`bdp` is the algorithm under evaluation, not a baseline. Oracles in the gap
experiment are diagnostics only.

## Files

- `code/realistic_agentic_sim.py`: workload, service, KV/APC engine, FCFS,
  Thunder, BDP, and the policy interface;
- `code/run_coder_calibration.py`: fixed workload and mechanism validation;
- `code/run_algorithm_experiment.py`: paired FCFS/Thunder/BDP evaluation;
- `code/run_bdp_gap_experiment.py`: context ablation and hindsight-gap
  decomposition;
- `code/run_bdp_ablation.py`: paired logical-mechanism ablation;
- `code/run_service_curve.py`: one paired run with PPT-style load curves,
  program timelines, and same-state BDP/hindsight action audit;
- `code/export_setting.py`: machine-readable configuration export;
- `results/coder_calibration/`: environment calibration evidence;
- `results/algorithm_experiment/`: ordinary policy evaluation;
- `results/bdp_gap_experiment/`: current context ablation and oracle evidence;
- `results/bdp_ablation/`: development ablation and untouched-seed
  confirmation of the minimal action rule;
- `results/service_curve/`: one paired service trace and rendered figures;
- `SIMULATOR_SETTING.md`: canonical mechanism and formula specification;
- `SIMULATOR_CALIBRATION_EXPERIMENTS.md`: calibration and algorithm evidence.

`new_experiment.pptx` is retained only as the original source artifact.

## Setup and validation

```bash
python -m pip install -r requirements.txt
pytest -q
python code/run_coder_calibration.py --out results/coder_calibration
```

The calibration command checks nine workload and service/APC mechanisms. Its
FCFS trace is diagnostic and is not compared with a different public policy.

## Policies

FCFS orders READY requests only by queue arrival time. The queue head blocks
later requests when it cannot fit. FCFS does not inspect request size, future
work, posterior state, or cache value. Engine APC remains enabled.

Thunder implements the paper's single-worker action policy: fixed 100-token
buffer, `2 ** (-tool_age)` Acting decay, Reasoning-before-new shortest-first
restore, Acting-first shortest-first pause, and deferred pause at a tool
boundary. Multiworker placement and migration are outside this simulator.

BDP is Bayesian SERPT plus one KV dual price. For program `i`:

```text
mu_i     = posterior expected remaining work
q_i      = expected KV footprint
v_i      = 1 / mu_i
d_i      = v_i / q_i
lambda   = marginal d_i that fills currently free KV
margin_i = v_i - lambda * q_i
```

Completed turns update the current program's persistent decode scale using one
Bayesian shrinkage rule. The update first assigns each observed turn to the
already calibrated small/large decode mixture probabilistically, so one rare
large turn is not mistaken for a permanently heavy program. The mixture and
prior precision come directly from the workload model; BDP has no fitted
policy parameter. Prefix reuse remains in shared engine APC; BDP reserves no
tool-state KV.

Positive margins are admitted in descending order subject to actual KV and
batch feasibility. Block normalization, exact set packing, and heuristic
tie-breaks are not part of BDP. The K posterior is also conditioned on the
observable 35k context guard. If
`c_i` is predicted context after the current turn and `u_i` the expected future
context increment, the number of future turns used by BDP is

```text
min(prior_future_turns_i, max(0, (35000 - c_i + decode_i) / u_i))
```

This is a direct consequence of the workload guard, not a fitted coefficient.

## Running experiments

```bash
python code/run_algorithm_experiment.py \
  --policies fcfs,thunder,bdp \
  --out results/algorithm_experiment

python code/run_bdp_gap_experiment.py \
  --out results/bdp_gap_experiment

python code/run_bdp_ablation.py \
  --out results/bdp_ablation

python code/run_service_curve.py \
  --out results/service_curve --seed 20262101
```

Batch runners should use the next untouched seeds `20262401--20262420`. Paired
policies always receive identical realized workloads.

## Current frozen result

On the held-out `20262101--20262120` batch:

| Online policy | Mean CT | Improvement vs FCFS | Improvement vs Thunder |
|---|---:|---:|---:|
| FCFS | 517.23 s | -- | -- |
| Thunder | 464.51 s | +10.21% | -- |
| BDP | 418.96 s | +18.87% | +9.58% |

Mixture-aware inference improves mean CT by +0.92% over the old token-sum
posterior (95% CI [0.18%, 1.65%]); its P95 difference is not significant.
Context conditioning adds +1.44%. Exact turn count adds only +0.50% with a
confidence interval crossing zero. The hindsight index reaches 382.82 s,
leaving an 8.59% gap. It uses the same BDP action formula with realized
remaining work; it is not a deployable baseline or a global optimum.

The logical ablation retains only three pieces: a Bayesian remaining-work
estimate, an expected future KV footprint, and one capacity-clearing dual
price. Removing the program objective, dual, or future footprint degrades mean
CT by 20.94%, 3.25%, and 4.05% on development seeds. Removing block
normalization and heuristic tie-breaks changes no result on 20 development and
20 untouched confirmation seeds, so they were deleted.

Action changes are gated on mean, P90, P95, and makespan together. Total-work
ranking, a service-saturation concurrency cap, and forced probing all failed a
fresh five-seed gate and are not present in the shipping policy. Future action
work first builds a constrained global hindsight teacher, then derives a new
Bayesian marginal value of service from active/passive state transitions. It
will replace the SERPT value if validated; it will not wrap another selector
around the existing BDP action.

The service trace shows that the gap is created mainly by repeated decisions
during the early/middle completion ramp, not by the first `t=0` action or by a
generic makespan tail. See
`SIMULATOR_CALIBRATION_EXPERIMENTS.md` for the controlled decomposition.

## Adding an algorithm

Subclass `BDPLikePolicy`, implement a causal estimate/score, and register it in
`POLICIES`. Router-cache objectives require an explicit policy implementation;
they are not hidden in the BDP interface. `WorkEstimate` exposes distributional quantities, never realized
future decode length, remaining tool duration, or final turn count. Do not
label a new candidate as a baseline unless that choice is explicitly fixed.

## Validity boundary

Use the simulator for single-worker, simultaneous-arrival admission-policy
screening. It is not an iteration-level vLLM replica and does not model
multiworker routing, networking, placement, or migration. Real deployment
claims require a paired system A/B test.
