# Decision traces

`result/<run>/decision_trace_<arm>.jsonl` — one record per scheduler tick
(3 s), written by the Router when `TA_DECISION_TRACE` is set. This is the only
artifact that explains *why* a policy behaved as it did rather than only what
the aggregate metrics ended up being.

Tracked for the coder A/B series (26–29), ~1.3 MB per arm. The Qwen3-32B series
(17–25) is excluded: longer runs, 5–10 MB per trace.

## One record

```json
{"t": 121.95, "tick": 40, "policy": "sim0823", "lambda": null,
 "free": 8200, "n_reasoning": 57, "n_waiting": 21,
 "reasoning": [{"pid": "...", "P": 14683, "D": 142,
                "status": "reasoning", "step": 16, "marked": false}, ...],
 "waiting":   [ ... same shape ... ],
 "admitted":  ["sympy__sympy-13091"],
 "evicted":   [{"pid": "...", "reason": "capacity"}]}
```

**The two task lists are the state *before* this tick's decisions; `admitted`,
`evicted` and `free` are *after*.** A record is therefore "the situation, then
what was done about it".

| field | meaning |
|---|---|
| `t`, `tick` | seconds since the run began; tick index |
| `policy` | the policy actually in force — check this first, experiments 17/18 are invalid because it silently fell back to the default |
| `lambda` | dual price; only `dual_descent` / `fidelity` set it, otherwise `null` |
| `free` | KV tokens left across all backends after the decisions. Negative means over capacity, which triggers eviction |
| `reasoning` | tasks currently admitted and holding KV. The name is historical: a task in this list may have `status: "acting"` |
| `waiting` | the paused pool — candidates holding no KV |
| `admitted` / `evicted` | program ids chosen this tick; evictions carry a reason (`capacity`, `proactive`) |

Per task:

| field | meaning |
|---|---|
| `P` | current KV footprint in tokens (`total_tokens`), grows with turns |
| `D` | known decode length from the A-arm tape (`X-Decode-Len`); **`0` means unknown** and the policy falls back to `decode_hat` |
| `status` | `reasoning` = waiting on generation; `acting` = running a tool between turns |
| `step` | turn index, capped at 20 |
| `marked` | already marked to pause once it becomes `acting` — i.e. on its way out |

Only `acting` tasks can be evicted safely (no generation in flight); a
`reasoning` task is marked and pauses when it next comes to rest.

Traces written from 2026-09-04 additionally carry each task's `sort_key`,
`evict_key` and `cache_value`, so the ranking the policy computed can be read
directly. Runs 26–29 predate that and show state without scores.

## Reading them

```bash
python3 algorithm/decisions.py summary  <trace>       # admissions, evictions, churn
python3 algorithm/decisions.py timeline <trace>       # ticks that made a decision
python3 algorithm/decisions.py timeline <trace> --all # every tick
python3 algorithm/decisions.py tick     <trace> 40    # one tick's candidate table
python3 algorithm/decisions.py program  <trace> <pid> # one program's life
```

## What they showed in run 29

`summary` on the sim0823 arm: 155 admissions over 80 programs, 24 evictions over
17 programs, all for `capacity`. One program (`pytest-dev__pytest-7432`) was
evicted 4 times and readmitted 6 times, and `psf__requests-1766` was evicted at
tick 35 and readmitted at tick 38 — nine seconds later, having discarded its KV
in between. Churn of that kind is invisible in the summary metrics and is worth
checking before reading a throughput difference as a policy effect.
