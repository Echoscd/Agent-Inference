# Production implementation guide

## Recommended first policy

Port **`hazard_grade_knapsack`**, not the full batch-capped variant.

At each outer scheduler tick on worker `w`:

1. Protect active reasoning requests and their distributional output headroom.
2. Score each ready LLM turn using

   ```text
   current_work  = alpha * uncached_prompt + E[decode]
   remaining_work = current_work
                    + (E[remaining_rounds] - 1)
                      * (alpha * E[incremental_prompt] + E[decode])
   run_score = 1 / (current_work * remaining_work * reserved_KV_blocks)
   ```

   Use an age term only as a starvation backstop, not as the main ordering.
3. Greedily admit positive feasible requests by `run_score`.  Do not use a single
   cluster-wide `lambda` gate in the first implementation.
4. For each warm inactive prefix `i`, compute

   ```text
   return_prob_i = P(tool returns within H
                     | tool has not returned by elapsed age,
                       tool class)
   cold_cost_i   = fitted cold-prefill seconds(prefix_tokens_i)
   cache_value_i = return_prob_i * cold_cost_i
                   * (1 + beta * terminal_probability_i)
   ```

   A ready program uses `return_prob_i = 1`.
5. Let the remaining worker KV capacity be the cache budget.  Select retained
   prefixes by a 0/1 knapsack in KV-block units.
6. If an active output exceeds its reservation between ticks, release prefixes in
   increasing `cache_value_i / blocks_i` order.

No step requires an individual next-tool completion prediction or an individual
final decode-length prediction.

## Mapping to the supplied scheduler package

The supplied RAR has `scheduling/base.py`, `density.py`, `dual_descent.py`,
`fidelity.py`, and `factory.py`.  A production port needs the following changes.

### 1. Extend program state

Add or expose:

```text
program_id
backend_url
status / phase
step_count
prefix or total cached blocks
tool_type
tool_start_timestamp
ready_since_timestamp
posterior terminal probability
posterior expected remaining rounds
last prompt/cached-token counters
```

The policy must not receive realized future output length from an offline trace.
Use a model-level or coarse-class distribution instead.

### 2. Add a new policy file

Create `scheduling/hazard_grade.py`.  It should provide:

```text
admission_score(program, backend)
cache_value(program, backend, now)
select_cache_victims(backend, required_blocks)
admits(program)
peak_pad(program)
on_epoch(backends, waiting)
```

The current `SchedulingPolicy` interface assumes admission and eviction are two
sides of one `sort_key`.  That is too restrictive here: run priority and cache
retention value are different quantities.  Add a separate cache/victim interface
rather than forcing both into one key.

### 3. Per-worker capacity

Run the knapsack independently per backend using physical KV blocks.  Do not sum
free memory across workers for admission feasibility.  Migration can be added
later as an explicit cost.

### 4. Tool distribution estimator

Start with coarse classes that are visible at execution time, such as:

```text
shell-short
file/read-write
repository search
unit test
build / integration test
other
```

For each class, maintain an empirical survival table over elapsed-time bins.  The
online query is just

```text
P(T <= age + H | T > age, class).
```

Use minimum sample thresholds and fall back to a pooled distribution when a class
is sparse.

### 5. Output reservation

Use a model/workload quantile such as Q90–Q95 of output tokens, not the realized
future decode length.  Export reserve-expansion and emergency-eviction counters.
A low quantile can create long decode stalls when no inactive cache remains to
release.

## Deployment sequence

1. **Shadow mode:** compute decisions and metrics without changing Thunder.
2. **Victim-selection only:** keep Thunder admission, replace only cache eviction.
3. **Hybrid admission:** enable the grade ordering while retaining Thunder's
   hysteresis, forced resume, and non-preemptive boundaries.
4. **Service-aware batching:** add only after fitting the GPU batch service curve.

This sequence makes every observed gain or regression attributable to one module.

## Required metrics

Log per worker and decision epoch:

```text
active reserved blocks
inactive held blocks
physical used blocks
program phase, stage, prefix size, tool class, tool age
run score, cache value, admit/hold/evict decision
warm/cold admit and cached-prefix tokens
cold-prefill GPU time
batch size and context distribution
program completion time, P90/P95, steps/min
scheduler wall time and forced-resume age
```

The real-system success criterion should be paired completion-time/throughput
improvement at the memory-pressure knee, not cache-hit ratio alone.
