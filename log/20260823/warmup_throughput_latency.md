# 26 / 27 / 28: throughput and latency after warmup

Three replicates of the same experiment. Same model, same 80 SWE-bench
instances, same settings — only the run differs. Arm A is the `size` baseline,
arm B is `hazard_grade`.

Numbers come from each run's `steady_metrics.json`, regenerated with
`algorithm/warmup_metrics.py`. Latency here is **per LLM call** (queue + prefill
+ decode); tool execution time is not included.

## What warmup means here

All 80 agents start at the same instant. For the first half minute the server is
not in the state we want to measure:

- the KV cache starts empty and has to fill up,
- the prefix cache has nothing in it yet, so no request can reuse anything,
- the number of running requests is still climbing from zero.

Anything measured during that ramp describes a cold server, not a busy one. So
we cut it off and only measure what happens afterwards.

**The cut-off point:** we watch KV cache utilisation, sampled twice a second.
Warmup ends the first time it reaches 90% of the highest level that run ever
hits. That is roughly 32–34 seconds into these runs.

We use "90% of this run's own peak" rather than a fixed number like "90% full"
on purpose. Different policies settle at different KV levels, and a fixed
threshold would make a policy that deliberately holds less cache look like it
warms up more slowly, which would compare the wrong thing.

Everything after that point is kept, including the tail at the end of the run
where only a few agents are left. Dropping that tail too is a separate view
(`steady` in the JSON); it is not used in this document.


## Experiment 26

| | window | throughput (tok/s) | avg latency (s) | p90 (s) | p95 (s) | calls counted |
|---|---|---|---|---|---|---|
| A `size` | whole run | 491.7 | 28.6 | 52.5 | 97.2 | 1167 |
|  | after warmup | 481.2 | 27.2 | 53.3 | 96.6 | 884 |
| B `hazard_grade` | whole run | 432.2 | 29.7 | 52.9 | 124.6 | 1221 |
|  | after warmup | 420.5 | 27.9 | 55.1 | 127.2 | 939 |

Warmup ended at 32.9 s (arm A) and 34.0 s (arm B).
After warmup, B vs A: throughput -12.6%, avg +2.6%, p90 +3.4%, p95 +31.7%.

## Experiment 27

| | window | throughput (tok/s) | avg latency (s) | p90 (s) | p95 (s) | calls counted |
|---|---|---|---|---|---|---|
| A `size` | whole run | 459.6 | 30.3 | 65.6 | 144.7 | 1193 |
|  | after warmup | 449.8 | 32.4 | 72.5 | 159.0 | 918 |
| B `hazard_grade` | whole run | 436.0 | 30.0 | 58.3 | 124.3 | 1211 |
|  | after warmup | 425.9 | 29.2 | 59.1 | 125.9 | 935 |

Warmup ended at 31.9 s (arm A) and 33.5 s (arm B).
After warmup, B vs A: throughput -5.3%, avg -9.9%, p90 -18.5%, p95 -20.8%.

## Experiment 28

| | window | throughput (tok/s) | avg latency (s) | p90 (s) | p95 (s) | calls counted |
|---|---|---|---|---|---|---|
| A `size` | whole run | 499.9 | 31.9 | 62.3 | 146.6 | 1177 |
|  | after warmup | 490.4 | 29.5 | 61.1 | 128.4 | 897 |
| B `hazard_grade` | whole run | 495.3 | 27.7 | 55.7 | 109.6 | 1207 |
|  | after warmup | 483.7 | 26.8 | 57.5 | 101.7 | 927 |

Warmup ended at 33.5 s (arm A) and 32.4 s (arm B).
After warmup, B vs A: throughput -1.4%, avg -9.2%, p90 -5.9%, p95 -20.8%.

## All three side by side (after warmup)

| metric | 26 | 27 | 28 | mean | spread (sd) |
|---|---|---|---|---|---|
| throughput, B vs A | -12.6% | -5.3% | -1.4% | **-6.4%** | 4.7 pts |
| avg latency, B vs A | +2.6% | -9.9% | -9.2% | **-5.5%** | 5.7 pts |
| p90 latency, B vs A | +3.4% | -18.5% | -5.9% | **-7.0%** | 9.0 pts |
| p95 latency, B vs A | +31.7% | -20.8% | -20.8% | **-3.3%** | 24.7 pts |

Absolute numbers, after warmup:

| | 26 A | 26 B | 27 A | 27 B | 28 A | 28 B |
|---|---|---|---|---|---|---|
| throughput (tok/s) | 481.2 | 420.5 | 449.8 | 425.9 | 490.4 | 483.7 |
| avg latency (s) | 27.2 | 27.9 | 32.4 | 29.2 | 29.5 | 26.8 |
| p90 (s) | 53.3 | 55.1 | 72.5 | 59.1 | 61.1 | 57.5 |
| p95 (s) | 96.6 | 127.2 | 159.0 | 125.9 | 128.4 | 101.7 |

## Reading this

**Dropping warmup barely moves these numbers.** Throughput falls about 2% in
every arm, because the ramp is only ~33 s out of a ~1000 s run and the server is
already generating during it. The latency percentiles move a little in both
directions. Warmup is not what is distorting this comparison.

**The three replicates disagree with each other.** On p95 the difference between
the two policies is +31.7%, then −20.8%, then −20.8%. The spread across
replicates is far larger than the average difference, on every metric. Run-to-run
variance is bigger than the effect being measured, so **these three runs do not
show a throughput or latency difference between `size` and `hazard_grade`**.

Two things to keep in mind before reading more into the table:

- The two arms do not run identical work. Both use temperature 0 and should
  repeat turn for turn, but batch-level variance in the server makes them drift,
  so roughly 15% of (program, turn) pairs differ between arms.
- A call's latency includes queue time, so it partly reflects how loaded the
  server was at that moment, which the policy itself influences.

One thing that *is* consistent across all three: arm B resolves more instances
(14 vs 11, 16 vs 12, 14 vs 11) and never takes longer in wall clock. That points
at goodput rather than raw token throughput, but three runs is not enough to
call it.

## How to regenerate

    for d in result/2[6-8]*/; do python3 algorithm/warmup_metrics.py --out "$d" "$d"; done

Each run folder then holds `steady_metrics.json` with all three windows
(`full`, `warm`, `steady`) and p50/p90/p95/p99 for call latency, queue wait and
decode time.
