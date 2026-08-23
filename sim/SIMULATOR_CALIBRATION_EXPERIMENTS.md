# Minimal Experiments for Calibrating the Simulator to the Real System

## 1. Goal

The simulator must reproduce the real **Size-policy baseline** before it is used
to predict the effect of `hazard_grade`. The purpose of the experiments below is
to determine whether the simulator--system gap comes from:

1. an incorrect workload and dataset fit;
2. an incorrect GPU service model;
3. incorrect KV-cache, preemption, or scheduler mechanisms; or
4. an incorrect definition of a reported metric.

The first calibration target is deliberately narrow:

- model: `qwen3-coder-30b` (not a generic Qwen3-32B proxy);
- backend: vLLM 0.12.0;
- prefix caching: enabled;
- dataset: the exact same 80 SWE-bench Verified task IDs from
  `sympy`, `pytest`, `pylint`, `requests`, `flask`, and `seaborn`;
- all 80 programs released at time zero;
- maximum 20 turns;
- the same agent, prompt template, tool implementation, generation settings,
  GPU type/count, tensor parallelism, vLLM flags, and Size policy;
- one worker/backend configuration and one KV capacity.

Do not start with a capacity sweep or another dataset. A one-setting replay is
enough to identify the dominant mismatch. Sweeps are useful only after this
setting is reproduced.

## 2. Evidence that calibration is currently necessary

The presentation reports the following two-run averages:

| Quantity | Real Size | Real `hazard_grade` |
|---|---:|---:|
| Mean task latency | 447.5 s | 471.0 s |
| Makespan | 1191.0 s | 1069.5 s |
| Generation throughput | 393.3 tok/s | 432.8 tok/s |
| Task rounds | 14.75 | 15.20 |
| Total decode length per task | 5.85k tokens | 5.80k tokens |
| Reported prefill prompt tokens per task | 237.7k | 250.0k |
| Average KV utilization | 54.5% | 65.0% |
| Prefix-cache hit rate | 77.85% | 78.75% |
| Average running programs | 23 | 28 |

The checked-in `real_proxy_smoke` trace is on a very different operating point:

| Quantity | Simulator `thunder_greedy` | Simulator `hazard_grade_knapsack` |
|---|---:|---:|
| Mean task latency | 9494.9 s | 7572.8 s |
| Makespan | 18264.9 s | 15469.9 s |
| Effective generation throughput | 69.7 tok/s | 82.3 tok/s |
| Task rounds | 9.475 | 9.475 |
| Total decode length per task | 15.91k tokens | 15.91k tokens |
| Prefix-cache hit rate | 14.0% | 34.0% |
| Average active programs | 12.0 | 11.7 |

The most important contradiction is not just the absolute error. On real
hardware, `hazard_grade` increases mean task latency in both repetitions
(+9.6% and +1.3%) while improving throughput and usually makespan. The current
simulator predicts a 20.2% reduction in mean task latency. The simulator is
therefore missing a trade-off that is active in the real system.

The apparent agreement in total "prefill tokens" must not be treated as a fit.
The simulator number includes cold recomputation, while the presentation may
report submitted prompt tokens. These quantities have different meanings and
must be separated in the trace.

## 3. Measurements required from every real run

Export one row per LLM call and one row per engine scheduling step. Use stable
`task_id`, `program_id`, and `turn_id` keys.

### Per LLM call

- program release, turn-ready, router-admit, engine-start, first-token, and
  final-token timestamps;
- tool-start and tool-end timestamps;
- total input tokens, cached input tokens, uncached input tokens, and output
  tokens;
- cumulative context before and after the call;
- warm/cold admission and the reason for every cache miss;
- prefill time, decode time, scheduler/queue time, and tool time;
- finish reason and whether the 20-turn limit censored the program;
- preemption time, mode, lost KV blocks, and recomputed tokens.

### Per engine step or short time bin

- running, waiting, prefill, and decode sequence counts;
- tokens scheduled for prefill and decode;
- `num_batched_tokens`, engine token budget, and chunked-prefill events;
- physical KV blocks used, reusable prefix blocks, reserved blocks if such a
  concept exists, and total usable blocks;
- prefix-cache queries, hits, and the exact denominator used for hit rate;
- priority/order chosen by the router and by vLLM;
- GPU busy time or utilization, if available.

Also save a run manifest containing the exact 80 task IDs, code commit, random
seeds, model revision, tokenizer revision, GPU configuration, vLLM arguments,
agent/tool configuration, and policy parameters. Without this manifest, two
repetitions cannot be assumed to be paired.

## 4. Minimal experiment sequence

Each experiment answers one question. Stop and repair the first failed layer
before interpreting later policy results.

### E0 — Metric-definition audit (no new performance run)

Use the existing logs to define exactly:

- whether "prefill prompt tokens" means submitted, uncached, or actually
  computed tokens;
- whether generation throughput is output tokens divided by makespan, GPU-busy
  time, or decode-active time;
- whether task latency includes tool time, router waiting, engine waiting, and
  prefill;
- whether KV utilization is physical occupancy, allocator occupancy, or a
  scheduler reservation;
- whether cache hit rate is token-weighted or request-weighted;
- what `resolved`, `running`, `waiting`, and a vLLM preemption count mean.

**Pass condition:** every real metric has a single formula that can be computed
unchanged from both the real trace and simulator trace.

### E1 — Workload trace and empirical replay

Run the Size policy once with the exact presentation setting and export the
measurements above. Replace independent Gamma/lognormal proxy sampling with a
trace replay for calibration. The replay must preserve:

- the empirical turn count, including censoring at 20 turns;
- per-turn input and output lengths;
- task and turn heterogeneity;
- correlations among context length, output length, tool duration, and turn
  index;
- observed tool completion/release times.

First run the simulator in **exogenous replay mode**: use the recorded turn-ready
times, so GPU/KV behavior can be tested without fitting the tool model. Only
after the engine model passes should tool times be generated by the simulator.

**Parameters validated:** turn-count distribution; first-turn and later-turn
prompt distributions; output distribution; tool-time distribution; correlations;
task heterogeneity; all-at-zero release assumption.

**Pass condition:** task-level and turn-level means and P50/P90/P95 values are
within 5% of the trace, and the empirical distributions pass visual ECDF/QQ
checks. In particular, the replay should be near 14.6--15.3 turns and
5.5k--6.1k total output tokens per task, not the current 9.475 turns and 15.9k
output tokens.

### E2 — GPU service microbenchmark

Remove tools and policy decisions. Use fixed requests on the same model and
vLLM configuration. Measure the following small grid:

- context/prompt length: the real trace P25, P50, and P90;
- active batch: 1, 8, 16, and 32 (include the observed knee near 20--30);
- workload mode: prefill only, decode only, and a fixed mixed prefill/decode
  case;
- cache state: cold and prefix-cache hit.

Fix output length to a small constant for prefill tests and to 256 or 512 tokens
for decode tests. Record TTFT, inter-token latency, aggregate token throughput,
and per-sequence throughput. Repeat each point at least five times after warm-up.

Fit a monotone measured surface or lookup table. Do not first force the data into
the current two saturation equations. In particular, test whether a constant
`mixed_prefill_share = 0.20` can explain the mixed measurements.

**Parameters validated:** single-request prefill/decode rates; batch saturation;
context penalties; maximum useful batch; mixed prefill/decode interference;
chunked-prefill token budget.

**Pass condition:** held-out service points have at most 10% median error and
15% P90 error for TTFT and throughput.

### E3 — KV capacity, prefix caching, and preemption threshold

Use identical programs with one shared prefix. Hold context and output lengths
fixed, then increase concurrency until KV reaches its limit. Repeat with prefix
caching off and on.

Measure:

- total usable KV blocks and block size;
- physical occupancy as tokens are prefetched and decoded;
- whether vLLM admits work incrementally or reserves the full output;
- the exact condition for preemption;
- whether preemption swaps, recomputes, or discards prefixes;
- whether prefixes are retained partially or only as an all-or-nothing object;
- how a cache hit changes TTFT and physical KV occupancy.

Compare these observations with four current assumptions: Q95 full admission
reservation, non-preemptive active requests, binary full-prefix retention, and
stalling at a reservation boundary.

**Mechanisms validated:** planning versus physical KV; incremental allocation;
active preemption/recomputation; partial prefix reuse; APC eviction semantics.

**Pass condition:** the simulator predicts the concurrency of the first
preemption within one request, physical KV utilization within 5 percentage
points, and preemption/recomputation counts within 10%.

### E4 — Closed-loop Size baseline replay

Replay the exact E1 workload using the fitted E2 service surface and E3 KV
mechanisms. Use the real Size policy, `peak_pad=0`, the same control timing, and
the exact real capacity. Do not use `thunder_greedy` as the final baseline unless
decision-by-decision comparison shows it is equivalent.

Add mechanisms in this order and record the reduction in validation error after
each addition:

1. empirical workload replay;
2. measured prefill and decode service surfaces;
3. chunked prefill and the real engine token budget;
4. incremental KV allocation and active preemption;
5. real prefix-cache block retention/eviction;
6. real router/engine scheduling cadence and ordering.

For each step, report the error in task completion-time distribution, makespan,
throughput, running/waiting time series, KV-utilization time series, cache-hit
rate, and preemption count. The mechanism that produces the largest reduction
in out-of-sample error is the main mechanism-level difference.

**Pass condition:** on a repetition not used for fitting, mean and P90 task
latency, makespan, generation throughput, average active count, cache-hit rate,
and average KV utilization are each within 10%; time-series shapes have no long
systematic bias; and the tail-drain period is reproduced.

### E5 — Policy holdout test

Freeze all fitted workload, service, and KV parameters. Replay
`hazard_grade` without using its outcome for calibration. Compare policy deltas
against Size, not only absolute metrics.

The real system currently shows this qualitative trade-off:

- higher generation throughput under `hazard_grade`;
- generally shorter makespan;
- slightly or substantially higher mean task latency;
- higher average running concurrency;
- similar cache-hit rate;
- only 5--8 vLLM preemptions per run;
- only 15--17 retention-knapsack activations in the hazard runs.

**Pass condition:** the simulator reproduces the direction of all five main
policy deltas (mean latency, makespan, throughput, average running count, and
cache-hit rate), and their magnitudes are within a bootstrap 95% interval formed
by task-level resampling. Failure here means the policy/router mechanism is
still wrong even if the Size baseline fits.

## 5. Mechanisms and parameters that must be explicitly validated

| Group | Current assumption | Required validation |
|---|---|---|
| Model | profile named `qwen3_32b_vllm_proxy` | Use the exact `qwen3-coder-30b` model/revision and hardware |
| Turn process | discretized Gamma prior with mean 8.758 | Fit the censored empirical per-task turn distribution near 15 turns |
| Prompts | independent first/later Gamma draws | Separate submitted, cached, uncached, and recomputed tokens; preserve turn/task correlation |
| Decode | independent Gamma, mean 1650 per turn | Fit per-turn conditional output so total output is near 5.8k per task |
| Tools | three-class lognormal proxy | Fit observed tool durations and their dependence on tool type/task/turn; first use exogenous ready times |
| Prefill service | one concave aggregate curve | Measure TTFT/throughput versus uncached tokens, batch, context, and cache state |
| Decode service | one mean-context saturation curve | Measure ITL/throughput versus batch and the full context distribution |
| Mixed service | fixed 20%/80% shares | Measure vLLM token-budget and chunked-prefill interference directly |
| KV admission | full prefix + prompt + Q95 output reserved at admission | Test whether the real engine allocates incrementally and how the router estimates headroom |
| Active work | program-level non-preemptive | Reproduce vLLM last-first preemption and recomputation |
| Cache retention | binary full-prefix keep/evict | Measure block-level/partial reuse and APC eviction behavior |
| Scheduling | 5 s outer tick plus idle wake-up | Match real event triggers, router cadence, vLLM step cadence, and ordering |
| Capacity | 360k/480k/600k token proxy | Derive usable physical blocks from the actual deployment; do not fit capacity to latency |
| Arrivals | all programs at time zero | Retain only if confirmed by real timestamps |

## 6. Fitting and validation protocol

Use a layered fit rather than one end-to-end optimizer:

1. **Metric layer:** fix definitions; fit nothing.
2. **Workload layer:** estimate empirical distributions/conditional tables from
   call traces.
3. **Service layer:** fit only to E2 microbenchmarks.
4. **KV/mechanism layer:** select mechanisms and thresholds only from E3.
5. **Closed-loop layer:** use one Size run for integration debugging, not for
   freely retuning all parameters.
6. **Validation layer:** evaluate on the second Size repetition.
7. **Policy holdout:** evaluate both hazard repetitions with every parameter
   frozen.

Never fit service parameters, KV capacity, or tool times to cancel an end-to-end
latency error. Such compensation can match one aggregate number while predicting
the wrong policy effect.

For every stage, publish a table with real value, simulated value, signed error,
absolute percentage error, and confidence interval. Also publish task-level
completion-time ECDFs and aligned KV/running/waiting time series. Aggregate means
alone cannot reveal a wrong tail-drain or queueing mechanism.

## 7. Decision rule for improving the simulator

After E4, rank candidate mechanisms by the held-out error reduction obtained
when each is added to the previous accepted model. Implement a mechanism in the
canonical simulator when:

1. a direct microbenchmark or trace demonstrates that it exists;
2. it reduces held-out Size-baseline error materially;
3. the improvement cannot be obtained only by refitting unrelated parameters;
4. it improves or preserves the policy-delta prediction in E5.

The likely first changes, based on the current evidence, are empirical workload
replay, measured vLLM service surfaces, incremental/chunked prefill, and real
active-preemption/KV semantics. The exact order must be decided by the measured
error reductions, not by plausibility alone.

