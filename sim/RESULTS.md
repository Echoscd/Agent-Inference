# Results and interpretation

## 1. What was tested

The final held-out matrix contains **45 paired scenarios**:

- 3 tool mixes: `swe_regular`, `swe_mixed`, and `heavy_tail`;
- 3 KV capacities: 40,000, 56,000, and 72,000 tokens;
- 5 independent seeds per mix/capacity pair;
- 64 programs released at time zero;
- approximately ten LLM rounds per program under the `quick10` prior;
- incremental prompt mean 120 tokens and output mean 150 tokens;
- output coefficient of variation 0.80;
- Q95 distributional output reservation;
- 16-token KV blocks;
- a 5-second outer scheduler tick and 0.5-second simulation step.

The default tool distributions are lognormal class-level models:

| Class | Mean | CV |
|---|---:|---:|
| `fast` | 1.8 s | 0.55 |
| `medium` | 9.0 s | 0.90 |
| `long` | 48.0 s | 1.25 |

The policy observes the tool class and elapsed age, but not the realized remaining
tool time.  It sees an output distribution and quantile, but not the realized
future decode length.

The GPU model uses concave prefill/decode throughput with batch and context
penalties.  These curves are intentionally non-linear, but they are not fitted to
a particular H800/Qwen-32B trace.

## 2. Aggregate held-out results

| Policy | Mean program CT (s) | P90 CT (s) | Makespan (s) | Steps/min | Cache-hit prefix ratio | Mean decision time |
|---|---:|---:|---:|---:|---:|---:|
| **Hazard-grade knapsack** | **236.57** | **444.48** | **793.66** | **55.06** | 0.831 | 0.446 ms |
| Hazard knapsack | 238.98 | 448.32 | 797.23 | 54.71 | 0.782 | 0.424 ms |
| Student BDP | 239.21 | 449.18 | 802.25 | 54.10 | **0.851** | **0.319 ms** |
| Hazard-batch knapsack | 244.89 | 453.24 | 803.94 | 54.02 | 0.830 | 0.506 ms |
| Thunder-style policy | 382.06 | 669.50 | 1002.62 | 42.20 | 0.835 | 0.384 ms |

All policies completed every program with zero recorded memory violations.

The Thunder row is a **single-worker policy approximation inside this revised
simulator**, not the full multiworker Dynamo/ThunderAgent system.  The large gap
against it must not be interpreted as an expected real-GPU gain.

## 3. Proposed policy versus the student's BDP

For `hazard_grade_knapsack`, paired scenario-level improvements relative to BDP
are:

| Metric | Mean paired improvement | Paired bootstrap 95% interval |
|---|---:|---:|
| Mean program completion time | **1.16%** | **[0.67%, 1.76%]** |
| P90 program completion time | **1.07%** | **[0.45%, 1.78%]** |
| Makespan | **1.38%** | **[0.62%, 2.31%]** |
| Steps/min | **1.50%** | **[0.65%, 2.53%]** |

The proposed policy had lower mean completion time in **38 of 45** held-out
scenarios, tied to numerical tolerance in one, and lost in six.

This is encouraging but modest.  It is a much more credible simulator result than
the original 40–50% gains because the new model includes tools, concave GPU
service, hidden output lengths, periodic control, and block-granular capacity.

## 4. Where the gain appears

Mean program-CT improvement over BDP by workload and capacity:

| Tool mix | 40k KV | 56k KV | 72k KV |
|---|---:|---:|---:|
| `swe_regular` | **5.71%** | 1.05% | 0.46% |
| `swe_mixed` | **1.90%** | 0.39% | 0.29% |
| `heavy_tail` | 0.35% | 0.14% | 0.15% |

The policy matters most near tight memory pressure.  At loose capacity, the
retention decision becomes less important and all strong policies converge.

The smaller gain on `heavy_tail` is also informative.  When many tools are long,
all sensible policies can release those prefixes, and tool wall time dominates a
larger share of end-to-end completion time.

## 5. Cache hit is not the objective

Relative to BDP, the proposed policy averaged:

- **2.02 percentage points lower** prefix cache-hit ratio;
- **9.19% more** cold-recompute tokens on average, although this statistic is
  highly heterogeneous across scenarios;
- nevertheless **1.16% lower** mean program completion time.

This is not a contradiction.  BDP's zero price often retains prefixes that are
valuable eventually but not valuable enough *now* to justify blocking ready
programs.  The proposed policy sometimes accepts recomputation to preserve active
GPU progress and reduce the unfinished-program population.

In the held-out matrix, BDP's learned price ended at exactly zero in all 45
scenarios.  Under asynchronous tool returns, its desired ready-run footprint was
usually below current free capacity.  Therefore, the implemented BDP reduced to:

```text
static posterior run ordering + positive hold rule
```

rather than an active dual-control policy.  This is one reason the new retention
rule is the useful change.

## 6. What did not work

The full `hazard_batch_knapsack` policy caps active concurrency at the knee of the
parametric decode service curve.  It was **2.60% worse than BDP** in mean program
completion time on the held-out matrix.

The likely explanation is not that batch awareness has no value.  It is that a
batch cap computed from an uncalibrated aggregate-throughput curve is too crude:

- it ignores the exact prefill/decode mix;
- it treats context distributions through one mean;
- it does not model the engine's token budget and kernel transitions;
- it can leave useful GPU parallelism unused.

Accordingly, the deployment recommendation is to port the hazard/knapsack
retention rule and grade admission first, while keeping Thunder's stable systems
shell.  Add batch configuration control only after fitting the real service
region.

## 7. Robustness checks

A separate representative tight-memory experiment varied one factor at a time.
The proposed mean-CT improvement over BDP was:

### Decode batch saturation parameter

| Saturation parameter | Improvement over BDP |
|---:|---:|
| 8 | 1.56% |
| 16 | 2.79% |
| 32 | 2.13% |

### Outer control interval

| Control interval | Improvement over BDP |
|---:|---:|
| 1 s | 1.91% |
| 5 s | 1.97% |
| 10 s | 18.19% |

The unusually large 10-second result says that retention quality matters more
when the outer loop is stale.  It should not be used as a forecast; the exact
magnitude depends on the synthetic workload and service model.

A low output-reservation quantile can produce long decode stalls after all
inactive caches have already been released.  Q95 is therefore the checked-in
default.  In the real system, reserve-expansion and active-preemption events must
be measured before lowering it.

## 8. Decision overhead

The Python implementation of `hazard_grade_knapsack` averaged **0.446 ms** per
outer decision with an average P95 of **1.266 ms** across the held-out matrix.
This includes exact block-level knapsack reconstruction.  It is far below the
5-second control interval in the simulator.

The code falls back to value-density greedy packing if the dynamic-programming
table would exceed five million cells.  The real implementation should log this
fallback rate.

## 9. What these results establish—and what they do not

The experiments establish that:

1. the original simulator's large BDP gains were not robust to realistic tool and
   GPU dynamics;
2. a prediction-light hazard/knapsack retention rule can produce a small,
   repeatable gain over the student's strongest ordering in the revised model;
3. directly adding an uncalibrated batch cap is harmful;
4. cache-hit ratio alone is not an adequate optimization target.

They do **not** establish that the proposed policy will improve ThunderAgent by
1.16% or 34% on an H800.  The next decisive step is to fit the service and cache
semantics from real traces, reproduce the Thunder baseline within a small error,
and then rerun exactly the same policy code in the emulator and router.
