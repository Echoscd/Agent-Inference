# Minimal Calibrated Simulator Setting

This file is the canonical specification of the current simulator. Numeric
values are exported to `settings/coder_calibration.json`.

## Scope

- model/backend: Qwen3-Coder-30B-A3B-Instruct, vLLM 0.12;
- one worker and one KV pool;
- 80 agent programs released together at `t=0`;
- 20-turn maximum with a 35k next-input context guard;
- outer router interval 3 s, simulator step and trace sample 0.5 s;
- fixed baselines: `fcfs` and `thunder`; candidate: `bdp`;
- current frozen evaluation: 20 seeds starting at `20262101`;
- next untouched evaluation: 20 seeds starting at `20262401`.

## Workload

| Quantity | Model |
|---|---|
| Natural turns | 70% reach 20; 30% exit uniformly at 3--19 |
| Policy posterior | empirical Coder posterior from 480 public program traces |
| Initial prompt | Gamma, mean 2788, CV 0.42 |
| Later prompt | 74.5% Gamma(184, 1.70); 25.5% Gamma(4930, 0.16) |
| Decode | 93.2% Gamma(162, 1.19); 6.8% Gamma(3619, 0.80) |
| Program decode correlation | mean-one lognormal scale, CV 0.50 |
| Tool time | three pooled Coder lognormal classes |

Every policy in a paired run receives the same realized `ProgramSpec` objects.

## State and engine

```text
READY -> ENGINE_WAITING -> PREFILL -> DECODE -> TOOL -> ... -> DONE
```

- `READY`: paused by the outer router; no physical KV;
- `ENGINE_WAITING`: submitted to vLLM but not scheduled; no physical KV;
- `PREFILL` / `DECODE`: running and holding materialized KV;
- `TOOL`: GPU-free tool execution; the completed prefix may remain reusable in
  vLLM's APC free list;
- `marked_for_pause`: Thunder keeps an in-flight request running and pauses it
  at its next `TOOL` boundary.

The only KV implementation is incremental allocation:

- capacity: 670000 tokens, blocks of 16;
- completed and preempted contexts become contiguous reusable APC prefixes;
- free-list allocation overwrites hashes in block-LRU order and may leave a
  partial reusable prefix;
- pressure uses last-scheduled-first recompute preemption;
- FCFS uses output pad 0; Thunder uses its fixed 100-token buffer; BDP and new
  BDP-like policies default to 1024;
- physical utilization counts referenced running blocks, not free APC hashes
  or router planning reservations.

Cold all-at-once requests share the measured prefill service window and the
8192-token chunk limit. After that burst, sparse tool returns use ordinary FCFS
engine admission. No fitted queue-rate parameter is used.

## Service profile

| Parameter | Value |
|---|---:|
| Single prefill tokens/s | 8333 |
| Prefill saturation | 8 |
| Single decode tokens/s | 85 |
| Decode saturation | 11 |
| Context reference | 40960 |
| Prefill context penalty | 0.33 |
| Decode context penalty | 0.07 |
| Mixed prefill multiplier | 1.0 |
| Mixed decode multiplier | 0.85 |
| Maximum active sequences | 80 |
| Prefill chunk | 8192 tokens |

Aggregate service is concave in batch size and context dependent. Prefill and
decode use separate measured interference multipliers.

## Policy information boundary

Policies receive immutable `ProgramView` and `SystemView` objects.

Observable program information:

- phase, stage, current prefix and revealed prompt;
- cache/admission/reservation blocks and preemption flag;
- tool class and elapsed tool age;
- waiting age;
- the decode lengths of completed turns;
- posterior terminal probability and expected remaining rounds.

Observable system information:

- block capacity and active/ready/waiting/cache program views;
- workload means and output reserve;
- context guard, service curves, router interval, and current simulated time.

Never observable:

- realized next or future decode lengths;
- realized remaining tool duration;
- realized final turn count or future prompts.

`SystemView.estimate_work(program)` exposes:

```text
uncached_prompt
current_work
remaining_work
expected_footprint_tokens
```

These are computed only from current observations and fitted distributions.

## FCFS baseline

`FCFSPolicy` has no parameter and sorts READY requests by `ready_since`, using
PID only to break simultaneous arrivals. Admission stops when the queue head
cannot fit. Tool returns rejoin READY. FCFS does not reserve tool-state KV, but
shared engine APC remains available.

## ThunderAgent baseline

`ThunderPolicy` implements the single-worker action policy from
[ThunderAgent Section 4.3](https://arxiv.org/html/2602.13692v3#S4.SS3). It has
no fitted constructor parameter and uses:

```text
buffer = 100 tokens per program
f(tool_age_seconds) = 2 ** (-tool_age_seconds)
```

At every router tick it restores paused Reasoning continuations before new
programs, shortest context first; pauses Acting programs shortest first after
rechecking non-decayed capacity; and, if needed, defers Reasoning pauses to the
next tool boundary. Multiworker placement and migration are outside scope.

## BDP interface and policy

New candidates subclass `BDPLikePolicy` and may implement:

- `estimate(program, view)`;
- `update_state(view, estimates)`;
- `admission_score(program, estimate, view)`;

The base class owns only physical KV feasibility, active-slot limits, and a
work-conserving fallback. BDP does not retain router cache; reuse is provided
only by engine APC.

For ready program `i`, let `mu_i` be posterior expected remaining work and
`q_i` expected KV footprint:

```text
mu_i = current_work_i + expected_future_turns_i * mean_future_turn_work
q_i  = expected_footprint_i
v_i  = 1 / mu_i
d_i  = v_i / q_i
```

If all expected footprints fit in currently free KV, `lambda = 0`. Otherwise,
`lambda` is the density of the marginal program that fills free KV in the
fractional capacity relaxation. The Lagrangian margin is:

```text
a_i = v_i - lambda * q_i
```

BDP admits positive `a_i` in descending order, with PID only as a deterministic
tie-break, subject to actual admission blocks and batch capacity. A paired
ablation showed that dividing by admission blocks and adding current-work or
smallest-block tie/fallback heuristics changed no result on 20 development and
20 untouched confirmation seeds, so those rules are absent from the shipping
algorithm.

The generator contains both a persistent program scale and an independent
small/large mixture for every turn. For observed turn `d_ir`, BDP first computes
the calibrated large-component responsibility `r_ir`, using the current scale
estimate in both component likelihoods. It then forms scale evidence:

```text
r_ir = P(large component | d_ir, current scale)
z_ir = (1 - r_ir) * d_ir / small_mean + r_ir * d_ir / large_mean

kappa   = 1 / decode_program_cv^2
scale_i = (kappa + sum_r z_ir) / (kappa + n_i)
decode_i = policy_mean_decode * scale_i
```

This uses only completed outputs and the existing calibrated mixture. It adds
no learned coefficient and prevents one rare large turn from being treated as
a permanently heavy program.

Let `L=35000` be the existing next-input context guard, `c_i` the predicted
context after the current turn, and `u_i=mean_prompt+decode_i` the expected
future context increment. The same K posterior is conditioned on its observable
remaining context budget:

```text
context_future_turns_i = max(0, (L - c_i + decode_i) / u_i)
future_turns_i = min(prior_future_turns_i, context_future_turns_i)
```

The `+ decode_i` term follows the harness rule: the next prompt must fit before
its decode is generated. This adds no fitted parameter, state, branch by
scenario, or new predictor. `decode_i` and `future_turns_i` determine `mu_i`
and `q_i`. The price is recomputed from current READY demand at every control
tick and reset when there is none. This is the complete online BDP mechanism:
one Bayesian estimate and one dual price.

## Metrics and artifacts

Primary metrics are mean/P90/P95 program completion time, paired improvement,
makespan, and generated tokens/s. Diagnostics include KV utilization, APC hit
ratio, waiting, service batch sizes, recomputation, preemption, and policy
decision time.

- `run_coder_calibration.py`: nine workload/mechanism checks;
- `run_algorithm_experiment.py`: FCFS/Thunder/BDP paired comparison;
- `run_bdp_gap_experiment.py`: context ablation and nested information-oracle
  decomposition. None is a registered baseline.
- `run_bdp_ablation.py`: orthogonal logical-mechanism ablation and minimal-rule
  confirmation;
- `run_service_curve.py`: opt-in PPT-style system load, per-program state,
  completion-tail, scheduler-action, and same-state hindsight traces;

Implementation-only block counters maintain exact physical KV and APC
occupancy in constant time and are checked by tests.

## Known boundary

The simulator is not an iteration-level vLLM replica. Exact global hash-block
order, multiworker locality, networking, placement, and migration are not
identified by the public data. Algorithms must not claim gains by optimizing
unvalidated observables.
