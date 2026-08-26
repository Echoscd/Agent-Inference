# Simulator Calibration and BDP Gap Evidence

## Fixed question and policies

This document asks why causal BDP differs from its hindsight index when 80
programs arrive together at `t=0`, and records only changes supported by paired
experiments.

The only fixed baselines are:

1. `fcfs`: strict first-come, first-served;
2. `thunder`: the single-worker ThunderAgent action policy.

`bdp` is the candidate. The context ablation and all oracles are diagnostics,
not additional baselines.

## Environment calibration

The simulator uses public Qwen3-Coder-30B/vLLM-0.12 evidence for the
80-program SWE-bench workload and policy-independent mechanisms:

| Public evidence | Implemented constraint |
|---|---|
| Coder request tapes | prompt/decode mixtures, program correlation, turns and tool time |
| Prefix reuse/drop microexperiment | APC hashes survive on the free list and are overwritten block-LRU |
| Long-context microexperiment | context-dependent prefill and 7% decode growth at 40960 tokens |
| vLLM configuration | 8192-token chunked prefill and distinct waiting/running states |
| KV traces | incremental block allocation and recompute preemption |

Run:

```bash
python code/run_coder_calibration.py --out results/coder_calibration
```

All nine declared workload and mechanism checks pass. No policy-specific
service rate, hit-rate constant, preemption probability, per-seed correction,
or scenario function is used.

## Final BDP

BDP remains Bayesian SERPT plus one capacity-clearing KV dual price:

```text
mu_i     = posterior expected remaining work
q_i      = expected KV footprint
v_i      = 1 / mu_i
d_i      = v_i / q_i
lambda   = marginal d_i that fills currently free KV
margin_i = v_i - lambda * q_i
```

The generator has a persistent program scale and an independent small/large
mixture on every turn. The old posterior incorrectly treated every observed
decode token as persistent-scale evidence. For completed decode `d_ir`, final
BDP instead uses the already calibrated component likelihoods:

```text
r_ir = P(large component | d_ir, current scale)
z_ir = (1 - r_ir) * d_ir / small_mean + r_ir * d_ir / large_mean

kappa   = 1 / decode_program_cv^2
scale_i = (kappa + sum_r z_ir) / (kappa + n_i)
decode_i = policy_mean_decode * scale_i
```

All quantities come from completed outputs and the existing calibrated
workload distribution. There is no threshold, learned feature, or new policy
coefficient. On development seeds it improved mean CT by 1.13% over the old
token-sum update. On held-out seeds the gain is 0.92% (95% CI [0.18%, 1.65%],
15/20 wins); the P95 difference is not significant.

The K survival posterior was already present, but the old estimator conditioned
it only on stage. The workload also has a 35k next-input context guard, and
current prefix is observable. The final estimator therefore uses:

```text
c_i = prefix_i + current_prompt_i + decode_i
u_i = mean_future_prompt + decode_i

context_future_turns_i = max(0, (35000 - c_i + decode_i) / u_i)
future_turns_i = min(prior_future_turns_i, context_future_turns_i)
```

The `+ decode_i` follows the actual guard: a future prompt must fit before its
decode is generated. The two retained estimator corrections—mixture-aware
scale evidence and context conditioning—add no constructor parameter, cache
objective, or simulator mechanism. Prefix reuse stays in shared engine APC.

## Logical minimum: paired ablation

The shipping form was selected on development seeds `20261301--20261320` by
removing one logical mechanism at a time. Positive numbers below mean the
ablation is worse than full BDP:

| Removed or replaced mechanism | Mean CT degradation | 95% CI | Worse seeds |
|---|---:|---:|---:|
| Remaining-program objective (current turn only) | +20.94% | [18.93%, 23.24%] | 20/20 |
| KV dual price | +3.25% | [2.60%, 3.93%] | 20/20 |
| Future KV footprint (immediate blocks only) | +4.05% | [3.34%, 4.72%] | 20/20 |
| Bayesian decode-scale update | +2.98% | [2.21%, 3.80%] | 20/20 |
| Mixture-aware evidence (naive token sum) | +1.16% | [0.54%, 1.80%] | 14/20 |
| Observable context guard | +0.94% | [0.40%, 1.52%] | 16/20 |
| K survival update | +0.44% | [0.18%, 0.71%] | 15/20 |
| Block normalization and heuristic tie/fallback rules | 0.00% | [0.00%, 0.00%] | 0/20 |

The last row was then confirmed on untouched seeds `20262201--20262220`: mean
CT and all simulated workload outcomes were identical on 20/20 seeds (only
wall-clock policy timing noise differed). Therefore the
shipping action is only:

```text
mu_i     = Bayesian expected remaining program work
q_i      = expected future KV footprint
v_i      = 1 / mu_i
lambda   = marginal (v_i / q_i) that clears free KV
margin_i = v_i - lambda * q_i
```

Admit positive margins in descending order subject to physical KV and batch
feasibility. PID is only a deterministic tie-break; if the worker is idle, the
top feasible program is admitted for liveness. There is no division by
admission blocks, exact set packing, cache objective, hand-tuned threshold,
scenario branch, or multi-level tie heuristic.

## Action-side multi-objective gate

Action changes are not accepted on mean CT alone. A candidate must improve
mean CT while showing no statistically supported regression in P90, P95, or
makespan. Three parameter-free structural interpretations of the service trace
were screened on new action-development seeds `20261401--20261405`:

| Action change | Mean CT | P90 CT | P95 CT | Makespan |
|---|---:|---:|---:|---:|
| Total work under KV pressure | +4.49% | +1.28% | +0.95% | +0.30% |
| Cap concurrency at `prefill_sat + decode_sat` | +23.95% | +22.69% | +20.98% | +16.87% |
| One least-observed probe admission | +6.62% | +2.75% | +2.80% | -0.47% |

Positive values are regressions relative to BDP. The total-work and
concurrency candidates failed on all five mean-CT seeds; the probe candidate
failed on all five mean/P90/P95 seeds, while its makespan change was not
significant. None entered the shipping policy or the registered policy set.

These failures distinguish causes from correlations in the hindsight trace.
Lower hindsight concurrency is a consequence of completing light programs,
not evidence for a batch cap. Likewise, a lower stage among hindsight choices
does not justify penalizing all attained work or forcing blind exploration.

The next action study must not wrap another selector around the existing
`1 / mu - lambda * q` score. It first constructs a constrained clairvoyant
teacher for each realized workload:

```text
minimize    sum_i completion_time_i
subject to  P90 <= P90_BDP
            makespan <= makespan_BDP
```

The teacher is diagnostic and may use expensive global search. Its purpose is
to expose the completion order and state-transition value of a schedule that
actually dominates BDP, rather than treating the mean-only hindsight index as
ground truth. The causal algorithm is then re-derived as a one-program
Bayesian control problem. For program state `s`, active/passive action values
define a marginal value of service:

```text
I(s) = Q_passive(s) - Q_active(s)
action margin = I(s) - lambda * q(s)
```

`I(s)` accounts for completion, observation, tool overlap, and future KV
release through the state transition itself. `lambda` remains the sole online
capacity price. This replaces the current SERPT value rather than adding a
bonus, guard, cap, or second-stage action filter. A clairvoyant version and a
causal posterior version of the same recursion can then be compared directly
with the constrained teacher.

## Diagnostic definitions

```text
no_context_bdp = final BDP with only context conditioning disabled
turns_oracle   = final BDP with exact remaining turn count
latent_oracle  = exact turns plus generator program decode scale
program_mean_oracle = exact turns plus realized program-average decode
hindsight_index = exact remaining turns, prompts, and ordered decodes
```

Every oracle starts from BDP and adds only the named information. The
hindsight index keeps the same BDP action and dual formula. It is not rollout
optimization, a deployable policy, or a proof of the global hindsight optimum.

## Held-out result

The final mixture-aware BDP was evaluated once on the untouched
`20262101--20262120` workloads:

| Online policy | Mean CT | vs FCFS | vs Thunder |
|---|---:|---:|---:|
| FCFS | 517.23 s | -- | -- |
| Thunder | 464.51 s | +10.21% | -- |
| BDP | 418.96 s | +18.87% | +9.58% |

BDP beats both baselines on all 20 paired seeds. Its mean decision time is
0.147 ms.

The isolated context ablation is:

| Metric | BDP gain over no-context BDP | 95% CI | Wins |
|---|---:|---:|---:|
| Mean CT | +1.44% | [0.96%, 1.90%] | 16/20 |
| P90 CT | +0.76% | [0.26%, 1.25%] | 15/20 |
| P95 CT | +0.12% | [-0.42%, 0.71%] | 10/20 |

Thus the mean improvement generalizes and does not cause a statistically
detectable P95 regression.

## Current gap decomposition

| Information available to the same BDP index | Mean CT | Gain vs BDP | 95% CI | Wins |
|---|---:|---:|---:|---:|
| Causal context-aware BDP | 418.96 s | -- | -- | -- |
| Exact remaining turns | 417.02 s | +0.50% | [-0.55%, 1.51%] | 10/20 |
| Exact turns + latent decode scale | 405.96 s | +3.15% | [2.12%, 4.22%] | 18/20 |
| Exact turns + realized program-mean decode | 382.88 s | +8.58% | [7.43%, 9.68%] | 20/20 |
| Exact ordered prompts and decodes | 382.82 s | +8.59% | [7.57%, 9.59%] | 20/20 |

The program-mean and ordered-decode oracles nearly tie because both remain
heuristic indices rather than global optimizers. Exact ordering provides no
material advantage over smoothing future turns with their realized mean.

The hindsight index is not a Pareto upper bound. Relative to BDP across the
same 20 seeds, it improves mean CT by 8.59%, leaves P90 statistically unchanged
at -0.20% (95% CI [-1.64%, 1.20%]), but worsens P95 by 1.31% (95% CI [0.18%,
2.49%] degradation) and makespan by 1.47% (95% CI [0.20%, 2.80%]
degradation). Action work must therefore report all four metrics and cannot use
distance to this index as a single scalar target.

The important conclusion is stable:

1. Context conditioning captures almost all useful K-side opportunity; exact
   K adds only 0.50% and its confidence interval crosses zero.
2. The generator's latent decode scale explains another 3.15%.
3. Realized program-average decode explains about 8.58%, so per-program
   decode heaviness remains the dominant diagnostic gap.
4. Exact ordered decodes do not help beyond their program average under the
   current index.
5. A new dual formula or router cache objective is not supported.

## Service-curve and action-window finding

`run_service_curve.py` records the PPT fields plus router READY, PREFILL,
DECODE, TOOL, instantaneous token rates, cumulative completions, program-state
timelines, scheduler admissions, and a same-state BDP/hindsight action audit.
Detailed program traces are opt-in and do not slow batch experiments.

The paired `20262101` trace shows that hindsight's main benefit is an earlier
completion ramp, not a uniformly shorter tail: it is far ahead through roughly
150--600 s, while the last program can finish later. On the same BDP states,
admission-set Jaccard drops below 0.3 through much of that interval, and BDP's
selected programs have materially more realized remaining decode.

The same-state audit also identified the retained Bayesian fix: in the key
window, hindsight often selects programs with high past decode but low future
decode. The old sum posterior treated a rare large-mixture turn as persistent
heaviness and systematically deferred them. Mixture-aware inference fixes that
mechanism and closes about 0.92 percentage points on held-out seeds. The
remaining gap still requires a better causal signal or a non-clairvoyant action
space; it is not evidence for more set-packing complexity.

## Reuse audit

Router-level BDP cache retention remains removed. Earlier paired ablation
showed that removing it changed mean CT by only +0.25% and increased prefix hit
from 80.60% to 82.24%. Disabling shared engine APC instead worsened mean CT by
17.67%. The minimal implementation keeps only engine APC.

## Data split

- simulator calibration: `20260826--20260828`;
- BDP development: `20261301--20261320`;
- action-side gate: `20261401--20261405`;
- previous frozen test: `20262001--20262020`;
- current one-time frozen final test: `20262101--20262120`;
- minimal-action confirmation: `20262201--20262220`;
- post-cleanup runner smoke: `20262301`;
- next untouched batch evaluation: `20262401--20262420`.

Authoritative artifacts:

- `results/algorithm_experiment/` for FCFS/Thunder/BDP;
- `results/bdp_gap_experiment/raw_results.csv`;
- `results/bdp_gap_experiment/case_summary.csv`;
- `results/bdp_gap_experiment/paired_summary.csv`;
- `results/bdp_gap_experiment/experiment_config.json`;
- `results/bdp_ablation/` for logical ablation and minimal-action confirmation;
- `results/service_curve/` for the paired load/action visualization;

## Validity boundary

These results support policy screening only in the calibrated, single-worker,
all-at-`t=0` batch. They do not establish iteration-level vLLM fidelity,
multiworker behavior, or real-system speedup. Deployment claims require a
paired real-system A/B test with the same FCFS and Thunder baselines.
