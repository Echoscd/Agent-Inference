# Changes from the original simulator

The original simulator made several assumptions that strongly favored policies
which admit many concurrent requests and retain many prefixes.  The revised
simulator changes the following items.

| Dimension | Original simulator | Revised simulator | Why it matters |
|---|---|---|---|
| Tool calls | Consecutive LLM turns become ready immediately | Every nonterminal turn enters a stochastic tool phase | Creates the actual retain-versus-evict decision |
| Tool information | No tool state | Tool class and elapsed age are observable; realized remaining time is hidden | Supports distributional hazard control without point prediction |
| Decode service | Every decode-ready sequence receives one token per time unit | Concave aggregate throughput varies with batch size and mean context | Admitting more sequences no longer creates linear GPU throughput |
| Prefill | All prefills progress concurrently without meaningful contention | Concave prefill throughput shares GPU time with decode | Cold starts and admission bursts slow ongoing decoding |
| Output length | Exact `d_i` is known to the scheduler | Policy sees a class distribution; admission reserves a quantile | Removes an oracle input unavailable in deployment |
| KV footprint | Continuous-token peak rectangle | Block rounding, distributional active reservation, and dynamic reserve expansion | Creates fragmentation/headroom and rare long-output pressure |
| Scheduling cadence | Reoptimization at every completion event | Periodic 5-second outer control loop, with an idle-GPU wakeup | Captures stale outer-loop decisions |
| Cache control | Exact binary HOLD/EVICT at every event | Binary block-level retention at control ticks plus emergency release | Closer to logical pause/resume behavior |
| Baseline | `commit_many` | Thunder-style phase-aware size/hysteresis policy | Makes the systems baseline structurally comparable |
| Objective/metrics | Mostly completion time and synthetic throughput | Program CT, P90, makespan, steps/min, cache hits, recomputation, KV utilization, decode batch, stalls, and decision overhead | Exposes throughput/cache/latency tradeoffs |
| Arrival pattern | All programs at time zero | Still all at time zero for paired SWE-bench-style batch experiments | Retained intentionally; open arrivals remain future work |
| Workers | One aggregate KV pool | One worker | Still a limitation; multiworker locality is not yet modeled |

## Assumptions intentionally retained

- Active LLM requests are non-preemptive.
- Cache retention is binary rather than partial.
- Tool execution consumes no GPU service.
- All programs in a paired experiment arrive at time zero.
- Service curves are parametric rather than trace-fitted.

These retained assumptions are listed explicitly so that a real-system mismatch
cannot be mistaken for an algorithm failure.
