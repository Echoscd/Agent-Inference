# Simulator Environment and Experimental Setting

This document is the canonical description of the simulator environment. It
defines how a workload is generated, how simulated time advances, what consumes
GPU and KV capacity, what information a scheduling policy may observe, and how a
paired experiment is constructed. Result tables and interpretations are kept
separately in `RESULTS.md`.

## 1. Scope and simulation unit

The simulator models one LLM-serving worker with one shared GPU and one KV-cache
pool. The unit of work is a multi-turn agent program:

```text
scenario
└── program 0 ... program N-1
    └── LLM turn 1 ... turn K
        ├── incremental prompt
        ├── prefill
        ├── decode
        └── asynchronous tool call, unless this is the final turn
```

Time is measured in seconds. Prompt, output, context, and KV capacity are
measured in model tokens. KV allocation is rounded to physical blocks.

All programs in the checked-in batch experiments are released at \(t=0\). The
simulator does not yet model an open arrival process or request routing across
multiple workers.

## 2. Program state machine

Each program follows:

```text
                       nonterminal turn
READY → PREFILL → DECODE ───────────────→ TOOL → READY
                    │
                    └──── final turn ───→ DONE
```

The states have the following meanings:

| State | Meaning | GPU service | Inactive prefix may be retained |
|---|---|---:|---:|
| `READY` | The next LLM turn is available but not admitted | No | Yes |
| `PREFILL` | The admitted prompt/prefix is being materialized | Yes | No |
| `DECODE` | The request is generating output tokens | Yes | No |
| `TOOL` | The program waits for an asynchronous external tool | No | Yes |
| `DONE` | Every turn has completed | No | No |

An admitted LLM turn is non-preemptive at the program level. The scheduler may
release inactive `READY` or `TOOL` caches, but it does not pause an active
`PREFILL` or `DECODE` request. If an active output exceeds its reservation and no
inactive cache can be released, decode progress stalls at the reservation
boundary.

## 3. Workload generation

### 3.1 Number of turns

For each program, the realized number of LLM turns \(K\) is sampled once from a
discrete prior on \(1,\ldots,K_{\max}\). The unnormalized weight at integer
\(k\) is

\[
w_k = k^{a-1}\exp(-k/b),
\qquad
P(K=k)=\frac{w_k}{\sum_j w_j}.
\]

The available priors are:

| Prior | Shape \(a\) | Scale \(b\) | \(K_{\max}\) | Normalized mean |
|---|---:|---:|---:|---:|
| `short` | 2.0 | 2.0 | 30 | 4.083 |
| `swebench9` | 2.5 | 3.5 | 60 | 8.758 |
| `quick10` | 2.5 | 4.0 | 60 | 10.006 |
| `long` | 3.0 | 6.2 | 80 | 18.584 |

The realized \(K\) is hidden from policies. At stage \(j\), a policy receives
only the posterior terminal probability

\[
P(K=j\mid K\ge j)
\]

and posterior expected remaining turns

\[
\mathbb{E}[K-j+1\mid K\ge j].
\]

### 3.2 Prompt and output lengths

Prompt and decode lengths are independently generated from a Gamma
distribution and rounded to a positive integer:

\[
\text{shape}=\frac{1}{\mathrm{CV}^2},
\qquad
\text{scale}=\frac{\mathrm{mean}}{\text{shape}}.
\]

The simulator supports a separate first-turn prompt distribution. This is
important for SWE-bench-like agents, where the repository/task prompt is large
and later turns primarily contain incremental tool output.

The realized decode length is stored in the environment but hidden from every
policy. Policies receive the class-level mean and a distributional reservation
quantile.

### 3.3 Tool calls

Every nonterminal LLM turn is followed by one asynchronous tool interval.
Durations are lognormal:

\[
\sigma = \sqrt{\log(1+\mathrm{CV}^2)},
\qquad
\mu = \log(\mathrm{mean})-\frac{\sigma^2}{2}.
\]

The checked-in tool classes are:

| Class | Mean duration | CV |
|---|---:|---:|
| `fast` | 1.8 s | 0.55 |
| `medium` | 9.0 s | 0.90 |
| `long` | 48.0 s | 1.25 |

Tool mixes are:

| Mix | Fast | Medium | Long |
|---|---:|---:|---:|
| `swe_regular` | 0.68 | 0.27 | 0.05 |
| `swe_mixed` | 0.50 | 0.35 | 0.15 |
| `heavy_tail` | 0.30 | 0.35 | 0.35 |
| `swebench_proxy` | 0.45 | 0.35 | 0.20 |

Tool execution consumes no GPU service. A policy observes the tool class and
elapsed tool age, but not the sampled duration or remaining tool time.

For a tool class with duration \(T\), the hazard-based policies use

\[
P(T\le a+H\mid T>a)
=
\frac{F(a+H)-F(a)}{1-F(a)},
\]

where \(a\) is elapsed tool age and \(H=10\) seconds by default.

## 4. Paired-workload construction

For each scenario and seed, the environment first samples immutable
`ProgramSpec` objects containing every prompt, realized output, tool class, and
tool duration. The exact same objects are then reused for every policy:

```text
seed
  → sample ProgramSpec list once
  → run Thunder on that list
  → run BDP on the same list
  → run proposed policy on the same list
```

This pairing removes workload randomness from policy differences. Policies do
not gain access to the stored future realizations.

## 5. GPU service environment

The GPU is a shared fluid-service model with separate aggregate prefill and
decode curves.

For \(n\) prefill requests,

\[
R_{\mathrm{prefill}}(n,\bar u)
=
\frac{
R_{\mathrm{prefill},1}
\frac{n}{1+(n-1)/s_{\mathrm{prefill}}}
}{
1+c_{\mathrm{prefill}}\bar u/L_{\mathrm{ref}}
}.
\]

For \(n\) decode requests,

\[
R_{\mathrm{decode}}(n,\bar L)
=
\frac{
R_{\mathrm{decode},1}
\frac{n_{\mathrm{eff}}}{1+(n_{\mathrm{eff}}-1)/s_{\mathrm{decode}}}
}{
1+c_{\mathrm{decode}}\bar L/L_{\mathrm{ref}}
},
\qquad
n_{\mathrm{eff}}=\min(n,n_{\max}).
\]

Here \(\bar u\) is mean remaining uncached prefill and \(\bar L\) is mean
materialized decode context.

When prefill and decode coexist, they receive fixed shares of the simulated GPU:

\[
R'_{\mathrm{prefill}}=\rho R_{\mathrm{prefill}},
\qquad
R'_{\mathrm{decode}}=(1-\rho)R_{\mathrm{decode}}.
\]

Aggregate throughput is shared equally among requests in the same phase.

### 5.1 Synthetic service profile

| Parameter | Value |
|---|---:|
| Single-request prefill rate | 7000 tok/s |
| Prefill saturation | 4 |
| Single-request decode rate | 45 tok/s |
| Decode saturation | 16 |
| Context reference | 4096 tokens |
| Prefill context penalty | 0.12 |
| Decode context penalty | 0.30 |
| Prefill share under mixed load | 0.30 |
| Maximum batch | 64 |

This profile is intentionally parametric and was used for the checked-in
synthetic held-out results.

### 5.2 Qwen3-32B/vLLM proxy profile

| Parameter | Value |
|---|---:|
| Single-request prefill rate | 2500 tok/s |
| Prefill saturation | 3 |
| Single-request decode rate | 55 tok/s |
| Decode saturation | 12 |
| Context reference | 32768 tokens |
| Prefill context penalty | 0.35 |
| Decode context penalty | 0.25 |
| Prefill share under mixed load | 0.20 |
| Maximum batch | 64 |

This proxy is anchored only to aggregate values from the real-system summary:
roughly 250--300 aggregate decode tok/s near an active batch of 10--15 and
long-context cold prefill that is substantially more expensive than in the
synthetic profile. It is not a fitted digital twin.

For reference, the proxy predicts approximately:

```text
batch=14, mean context=32k: 296 decode tok/s
16k cold prefill, single request: 7.5 s
32k cold prefill, single request: 17.2 s
```

## 6. KV-cache environment

### 6.1 Block accounting

With block size \(B\),

\[
\operatorname{blocks}(x)=\left\lceil\frac{x}{B}\right\rceil.
\]

An inactive warm prefix occupies

\[
M_i^{\mathrm{cache}}
=
\operatorname{blocks}(\mathrm{prefix}_i).
\]

An admitted request reserves

\[
M_i^{\mathrm{reserve}}
=
\operatorname{blocks}
\left(
\mathrm{prefix}_i+
\mathrm{prompt}_i+
Q_q(D)
\right),
\]

where \(Q_q(D)\) is the configured output-length quantile, Q95 by default.

### 6.2 Planning versus physical usage

The scheduler uses reserved occupancy:

\[
M^{\mathrm{planning}}
=
\sum_{i\in\mathrm{active}}M_i^{\mathrm{reserve}}
+
\sum_{i\in\mathrm{inactive\ warm}}M_i^{\mathrm{cache}}.
\]

Physical occupancy contains only materialized KV:

\[
M^{\mathrm{physical}}
=
\sum_{i\in\mathrm{active}}
\operatorname{blocks}(\mathrm{activeKV}_i)
+
\sum_{i\in\mathrm{inactive\ warm}}M_i^{\mathrm{cache}}.
\]

These quantities can differ substantially during long cold prefills. The
real-proxy trace currently shows reserved KV near capacity while physical KV is
much lower; this is a known mismatch with vLLM's chunked prefill and active
preemption semantics.

### 6.3 Warm and cold admission

For a warm admission,

\[
\mathrm{prefill\ tokens}=\mathrm{incremental\ prompt}.
\]

For a cold admission after eviction,

\[
\mathrm{prefill\ tokens}
=
\mathrm{evicted\ prefix}
+
\mathrm{incremental\ prompt}.
\]

Cache retention is binary: the full prefix is either retained or discarded.
Partial prefix retention is not modeled.

### 6.4 Output-reservation overflow

If actual decode exceeds Q95 reservation:

1. compute the additional required blocks;
2. release inactive caches using the policy's emergency victim order;
3. expand the active reservation if sufficient space becomes available;
4. otherwise stall that decode at its reservation boundary.

Active-request preemption and recomputation are not modeled.

## 7. Simulated-time execution order

The default settings are:

| Parameter | Value |
|---|---:|
| Outer scheduler interval | 5.0 s |
| Base simulation step | 0.5 s |
| Maximum simulated time | 20000 s |
| Work conversion \(\alpha_{\mathrm{work}}\) | 0.25 |

Each loop iteration follows:

```text
1. If now >= next scheduler tick, run the scheduler.
2. If READY exists and no PREFILL/DECODE is active, run the scheduler immediately.
3. Choose dt as the minimum of:
     - configured simulation step,
     - time to next scheduler tick,
     - time to the next tool completion.
4. Integrate utilization and batch-size metrics over dt.
5. Advance every TOOL interval by dt.
6. Advance shared PREFILL and DECODE service by dt.
7. Complete turns, start tools, or mark programs DONE.
8. Advance wall-clock time and check physical/reserved memory safety.
```

The immediate idle-GPU wakeup prevents a stale outer control interval from
leaving the GPU idle when work is ready.

## 8. Scheduler interface and information boundary

The environment constructs a policy-safe `SystemView`. A policy may observe:

- current time and control interval;
- worker capacity and block size;
- current phase, stage, prefix, and revealed prompt;
- warm/cold state, admission blocks, cache blocks, and active reservation;
- tool class and elapsed age;
- posterior terminal probability and expected remaining turns;
- class-level prompt/decode/tool distributions;
- the configured service curves.

It may not observe:

- realized future decode length;
- realized remaining tool duration;
- realized total program length;
- future tool classes or prompts.

`RequestSpec` contains future realizations only for environment progression.
`ProgramView` deliberately omits them. This separation is covered by
`test_policy_view_hides_realized_future`.

## 9. Scheduler-tick decision flow

At a scheduling epoch:

```text
SystemView
   ├── active PREFILL/DECODE reservations: protected
   ├── READY candidates: ordered and admitted subject to capacity
   └── READY/TOOL warm caches: retained or evicted in residual capacity

Plan(admit IDs, keep-cache IDs)
   → evict every inactive cache not selected
   → admit selected READY requests in policy order
   → perform emergency release if reservation accounting exceeds capacity
   → verify physical and planning memory
```

The simulator separates run priority from cache-retention value for the hazard
policies. This is intentional: a request can be urgent to run but not valuable
to cache while its tool is inactive, or vice versa.

## 10. Policies in the environment

### 10.1 Synthetic Thunder approximation

`thunder` uses continuation/size ordering and a high/low utilization hysteresis:

| Parameter | Value |
|---|---:|
| High watermark | 0.95 |
| Target | 0.80 |
| Hysteresis width | 0.10 |
| Tool-cache decay time | 30 s |
| Forced resume | 1800 s |

This is a single-worker simulator approximation, not the real ThunderAgent
router.

### 10.2 Work-conserving Thunder proxy

`thunder_greedy` approximates the Size/Thunder behavior reported by the real
experiments:

- continuation/reasoning candidates before new work;
- smaller contexts first within a group;
- greedy filling with no global price gate;
- smallest inactive prefixes released first under pressure.

The common simulator still applies Q95 reservation, whereas the real
SizePolicy used `peak_pad=0` and allowed vLLM preemption. This remains a known
baseline mismatch.

### 10.3 Dual-price proxy

`dual_price_proxy` implements the presentation's one-turn density

\[
v_i=\frac{1}{\tau_i q_i},
\qquad
\tau_i=0.03q_i+1000,
\]

and admission gate

\[
v_i-\lambda>0.
\]

The scalar price is updated with step parameter \(\eta_0=0.10\). Because the
same \(\lambda\) is subtracted from every request, it changes admission amount
but not relative density ordering.

### 10.4 Student BDP

`student_bdp` uses posterior current and remaining work plus a learned global
price. It uses \(\eta_0=0.10\). In the synthetic held-out matrix its final price
was zero in every scenario, reducing it to posterior ordering plus a positive
cache-hold rule.

### 10.5 Proposed hazard-grade knapsack

For a READY request,

\[
\begin{aligned}
w_i^{\mathrm{current}}
&=
\alpha_{\mathrm{work}}u_i+\mathbb{E}[D],\\
w_i^{\mathrm{remaining}}
&=
w_i^{\mathrm{current}}
+
(\mathbb{E}[K_i^{\mathrm{remaining}}]-1)
(\alpha_{\mathrm{work}}\mathbb{E}[P]+\mathbb{E}[D]),\\
s_i^{\mathrm{run}}
&=
\frac{1}{
w_i^{\mathrm{current}}
w_i^{\mathrm{remaining}}
M_i^{\mathrm{reserve}}
}.
\end{aligned}
\]

Here \(u_i\) is the revealed prompt plus the prefix only when that prefix is
cold.

For an inactive cache,

\[
V_i^{\mathrm{cache}}
=
P_i(\mathrm{return\ within}\ H)
T_i^{\mathrm{cold\ prefill}}
(1+\beta P_i(\mathrm{terminal}))
(1+\gamma A_i),
\]

with default \(H=10\) s and \(\beta=1.5\). The selected
`hazard_grade_knapsack` configuration sets age bonus \(\gamma=0\) and fixed
safety fraction to zero.

Retained caches maximize

\[
\max \sum_i V_i^{\mathrm{cache}}x_i
\quad
\text{subject to}
\quad
\sum_i M_i^{\mathrm{cache}}x_i\le M^{\mathrm{residual}},
\quad
x_i\in\{0,1\}.
\]

The implementation uses exact block-level dynamic programming unless
\(n(C+1)>5{,}000{,}000\), where it falls back to value-density greedy packing.

## 11. Canonical experiment presets

### 11.1 Synthetic held-out matrix

| Dimension | Setting |
|---|---|
| Programs | 64, all released at \(t=0\) |
| Turn prior | `quick10`, mean 10.006 |
| Prompt | mean 120, CV 0.55; same distribution for initial and later turns |
| Decode | mean 150, CV 0.80 |
| Tool mixes | `swe_regular`, `swe_mixed`, `heavy_tail` |
| KV capacities | 40k, 56k, 72k tokens |
| Replicates | 5 per mix/capacity |
| Total scenarios | \(3\times3\times5=45\) |
| Block size | 16 tokens |
| Decode reservation | Q95 |
| Service profile | `synthetic` |
| Control interval / step | 5.0 s / 0.5 s |
| Base seed | 20260711 |

### 11.2 Validation matrix

| Dimension | Setting |
|---|---|
| Programs | 48 |
| Tool mixes | `swe_mixed`, `heavy_tail` |
| KV capacities | 40k, 56k |
| Replicates | 2 |
| Total scenarios | 8 |

Validation scenarios are used for policy selection and must not be merged into
held-out confidence intervals.

### 11.3 Qwen3-32B/vLLM SWE-bench proxy

| Dimension | Setting |
|---|---|
| Programs | 80, all released at \(t=0\) |
| Turn prior | `swebench9`, mean 8.758 |
| Initial prompt | mean 14k, CV 0.35 |
| Incremental prompt | mean 1.8k, CV 0.75 |
| Decode | mean 1.65k, CV 0.80 |
| Tool mix | `swebench_proxy` |
| KV capacities | 360k, 480k, 600k tokens |
| Replicates | 3 per capacity |
| Total scenarios | 9 |
| Block size | 16 tokens |
| Decode reservation | Q95 |
| Service profile | `qwen3_32b_vllm_proxy` |
| Control interval / step | 5.0 s / 0.5 s |
| Base seed | 20260717 |

This proxy matches only aggregate workload and throughput scale. Per-tool
timestamps, uncached-token counters, and physical reusable-prefix blocks were
not available from the presentation.

## 12. Metrics and definitions

Let \(C_i\) be the completion time of program \(i\), \(T=\max_i C_i\), and \(R\)
the number of completed LLM turns.

| Metric | Definition |
|---|---|
| Mean program CT | \(\frac{1}{N}\sum_i C_i\) |
| P90/P95 program CT | Percentile of \(\{C_i\}\) |
| Makespan | \(T\) |
| Programs/hour | \(3600N/T\) |
| Steps/min | \(60R/T\) |
| Queue time | admission time minus time the turn became READY |
| Prefix cache-hit ratio | warm-admitted prefix tokens / all requested prefix tokens |
| Cold recompute tokens | sum of evicted prefixes that must be prefetched again |
| Physical KV utilization | time integral of physical blocks divided by capacity-time |
| Reserved KV utilization | time integral of planning blocks divided by capacity-time |
| Mean active | time-average number of PREFILL + DECODE programs |
| Mean decode batch | time-average number of DECODE programs |
| GPU busy fraction | fraction of makespan with any PREFILL or DECODE work |
| Decode stall | summed time steps in which output progress hits reservation |
| Decision overhead | scheduler policy wall-clock time per epoch |

Paired percentage improvement for a lower-is-better metric \(X\) is

\[
100\frac{X_{\mathrm{baseline}}-X_{\mathrm{policy}}}
{X_{\mathrm{baseline}}}.
\]

Scenario-level means are bootstrapped with 5000 resamples and bootstrap seed 7.

## 13. Trace setting

`code/trace_compare.py` runs policies on the same sampled workload and records:

- every scheduler snapshot;
- every candidate's `ADMIT`, `WAIT`, `KEEP`, or `EVICT` decision;
- warm/cold admission;
- tool start and return;
- prefill completion;
- program completion;
- all phase transitions;
- policy predictions and post-hoc realized tool remaining time.

The post-hoc realization fields are written only after the policy plan has been
computed and are never included in `SystemView`.

The generated files are:

```text
scheduler_snapshots.csv
scheduler_candidates.csv
decision_events.csv
paired_decision_differences.csv
state_transitions.csv
workload_programs.csv
metrics.csv
trace_overview.png
early_decision_raster.png
program_state_heatmap.png
```

## 14. Reproduction commands

Install and test:

```bash
python -m pip install -r requirements.txt
pytest -q
```

Synthetic held-out matrix:

```bash
python code/realistic_agentic_sim.py \
  --preset test \
  --service-profile synthetic \
  --seed0 20260711 \
  --out results/my_test
```

Real-system proxy matrix:

```bash
python code/realistic_agentic_sim.py \
  --preset real_proxy \
  --service-profile qwen3_32b_vllm_proxy \
  --policies thunder_greedy,dual_price_proxy,student_bdp,hazard_grade_knapsack \
  --thunder-policy thunder_greedy \
  --seed0 20260717 \
  --out results/my_real_proxy
```

Paired trace:

```bash
python code/trace_compare.py \
  --preset real_proxy_smoke \
  --service-profile qwen3_32b_vllm_proxy \
  --policies thunder_greedy,hazard_grade_knapsack \
  --seed0 20260717 \
  --early-window 1200 \
  --out results/trace_real_proxy
```

Export machine-readable settings:

```bash
python code/export_setting.py \
  --preset test \
  --service-profile synthetic \
  --seed0 20260711 \
  --out settings/synthetic_heldout.json
```

## 15. Calibration boundary

The simulator is currently suitable for controlled ablation and algorithm
falsification. It is not yet a faithful emulator of ThunderAgent/vLLM.

The largest remaining environment gaps are:

1. one worker and one aggregate KV pool;
2. no active-request preemption and recomputation;
3. binary rather than block-subset prefix retention;
4. Q95 upfront reservation instead of vLLM's exact token-budget behavior;
5. fixed prefill/decode sharing rather than kernel- and token-budget scheduling;
6. parametric service curves rather than per-call GPU fits;
7. proxy tool distributions rather than recorded SWE-bench tool events;
8. no migration, router/network delay, or multiworker locality;
9. all programs released at \(t=0\);
10. no model-dependent changes to the agent's future trajectory caused by
    scheduling delay.

Before using simulator effect sizes as a deployment forecast, the environment
must reproduce the real baseline's KV utilization, active/decode batch,
uncached-prefill time, preemption rate, prefix-hit behavior, queueing, and
makespan within a declared tolerance.
