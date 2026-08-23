# vLLM GPU Implementation Notes

These notes translate the simulator into an implementation plan for a vLLM-like
serving scheduler.

## State Mapping

Represent each agent program as a scheduler-level object:

- `program_id`
- `stage_index`
- generated prefix length `A_i`
- current request lengths `(p_i, d_i)` or estimates while decoding is unknown
- prior/posterior over remaining request count
- warm/cold flag for prefix KV blocks
- committed/order metadata for baselines

In vLLM terms, the warm prefix corresponds to allocated KV blocks in the block
manager.  `HOLD` keeps those blocks allocated while the request is waiting.
`EVICT` releases them.  `ADMIT` creates or resumes the sequence group and
reserves the estimated peak footprint.

## Decision Epoch

Run the policy at request-boundary events:

1. Completed decode step reveals whether an agent program terminates.
2. If it continues, update `A_i <- A_i + p_i + d_i` and reveal or estimate the
   next `(p_i, d_i)`.
3. Compute free KV capacity after protecting active sequence groups.
4. Score waiting programs.
5. Admit positive/selected runs subject to hard current KV feasibility.
6. Use residual memory for `HOLD`.
7. Evict unheld inactive prefixes.

Do not preempt active decode/prefill in the first GPU implementation.  The
simulator deliberately treats active requests as non-preemptive.

## Bayes-Dual-Price Hook

For each waiting program:

```text
tau_i = alpha * (p_i + 1[cold] * A_i) + d_i
mu_i  = tau_i + (E[R_i | K_i >= j_i] - 1) * E[alpha P + D]
q_i   = A_i + p_i + d_i
qbar_i = q_i + 0.5 * (E[R_i | K_i >= j_i] - 1) * E[P + D]
Delta_run = 1 / (tau_i * mu_i) - lambda_t * qbar_i
Delta_hold = alpha * A_i / mu_i - lambda_t * A_i
```

Admit positive `Delta_run` requests by decreasing `Delta_run / q_i`.  Then hold
positive `Delta_hold` prefixes by decreasing `Delta_hold / A_i`.

Update the price after computing desired projected demand:

```text
lambda_{t+1} = max(0, lambda_t + eta_t * (D_t / M_t - 1))
eta_t = eta0 * scale / sqrt(t + 1)
```

`scale` should be initialized from the median value-density in the first
decision epoch.  This keeps the price magnitude tied to the local workload.

## LP Policies

The full campaign LP should not run synchronously on every GPU scheduler tick.
The practical path is:

1. Use `bayes_dual_price` or `lp_dual_descent` as the online path.
2. Optionally run `lp_selective` in a background control thread.
3. Re-solve only when aggregate drift is large:
   `unfinished count`, waiting count, total waiting peak, total held prefix,
   average stage, warm share.
4. Feed updated prices back into the fast scheduler.

The Python LP implementation is a reference, not the intended production hot
path.

## Memory Accounting

Production needs two separate counters:

- current reserved KV blocks for active sequences;
- held inactive prefix blocks.

The hard feasibility check is:

```text
sum_active q_i + sum_held A_i <= M
```

The simulator uses realized peak `q_i`; vLLM should use conservative estimates
while decode lengths are uncertain, then release surplus blocks when a request
finishes.

## Metrics to Export

Add scheduler metrics:

- program completion latency and request latency;
- warm admits, cold admits, recompute service estimate;
- `preempt_events`: held warm prefix not admitted;
- unique preempted request count;
- eviction/delete count;
- deleted-from-memory ratio;
- memory violations or admission rejections;
- scheduler decision wall time.

These are enough to compare whether an algorithm wins by better ordering or by
excessive deletion/recomputation.
