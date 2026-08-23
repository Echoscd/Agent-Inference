# Quick Experiment Results

Command used:

```bash
python code/agentic_kv_sim.py --preset quick --out results/quick_all --policies all
```

Setting:

- profile `balanced`: `E[P]=120`, `E[D]=150`
- request-count prior `quick10`: realized mean rounds about `10`
- token CV `0.35`
- memory factors `f in {5,10,15,20}`
- injected programs increase with memory: `n={170,320,480,640}`

All preserved policies completed all programs with `memory_violations = 0`.

## Load Check

The workload no longer becomes slack at large `f`; larger memory factors receive
larger program populations.

| f | n | mean rounds/program | qmax | M | mean active across policies |
|---:|---:|---:|---:|---:|---:|
| 5 | 170 | 9.92 | 9529 | 47645 | 14.11 |
| 10 | 320 | 10.55 | 9529 | 95290 | 28.20 |
| 15 | 480 | 10.10 | 9529 | 142935 | 44.53 |
| 20 | 640 | 10.05 | 9529 | 190580 | 58.52 |

The requested active-program range is now covered by the quick sweep: average
active load rises from about `14` at `f=5` to about `59` at `f=20`.  Some weak
policies intentionally remain underfilled because their original admission rules
are conservative; the competitive policies keep about 48 active programs on
average across the full matrix.

## Ranking

Sorted by mean program completion time across the full quick matrix:

| Policy | Avg active | Mean rounds | Mean CT | P90 CT | Mean improvement vs commit_many | Preempt events | Delete ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bayes_dual_price` | 47.74 | 10.15 | 6405.77 | 13914.93 | 47.53% | 2090.0 | 0.194 |
| `dual_descent_current` | 48.22 | 10.15 | 6865.40 | 14807.15 | 43.75% | 670.5 | 0.530 |
| `lp_dual_descent` | 47.83 | 10.15 | 7232.52 | 15898.70 | 40.79% | 34.2 | 0.963 |
| `mpc_short` | 48.81 | 10.15 | 7323.24 | 16238.38 | 40.02% | 0.0 | 1.000 |
| `lp_update_long` | 47.63 | 10.15 | 7765.87 | 17196.12 | 36.39% | 0.0 | 1.000 |
| `bayes_grade_warm_marginal` | 47.47 | 10.15 | 8822.99 | 19610.62 | 27.74% | 165.0 | 0.873 |
| `lp_selective` | 41.55 | 10.15 | 9450.61 | 21317.41 | 22.11% | 1.2 | 0.998 |
| `current_density` | 47.34 | 10.15 | 9641.80 | 21663.90 | 21.00% | 172.0 | 0.884 |
| `area_knapsack` | 48.04 | 10.15 | 9919.02 | 21899.87 | 18.84% | 152.0 | 0.902 |
| `bayes_grade` | 48.46 | 10.15 | 10019.82 | 21767.85 | 17.97% | 217.8 | 0.875 |
| `commit_many` | 50.75 | 10.15 | 12220.25 | 23565.97 | 0.00% | 1097.2 | 0.643 |
| `least_rounds` | 42.31 | 10.15 | 16524.04 | 39850.79 | -35.19% | 175.5 | 0.935 |
| `mu_two_price` | 11.01 | 10.15 | 22284.52 | 50924.54 | -82.19% | 3255.5 | 0.097 |
| `lp_infrequent` | 2.03 | 10.15 | 379514.67 | 519831.71 | -3022.53% | 3.5 | 0.997 |
| `lp_solve_once` | 1.15 | 10.15 | 439401.86 | 798558.29 | -3519.93% | 0.0 | 1.000 |
| `dual_threshold` | 1.12 | 10.15 | 495776.06 | 859026.69 | -3984.94% | 0.0 | 1.000 |

## Takeaways

Increasing the number of injected programs with `f` changes the conclusion.
The earlier `f=10` equality was a slack-load artifact; it is gone.

Under the new `f={5,10,15,20}` sustained-load setting, the best mean-completion
policy is the commitbase `bayes_dual_price`.  It improves mean completion time
by `47.53%` relative to `commit_many` and is `11.43%` lower mean CT than
`lp_dual_descent` on the aggregate table.

The tradeoff is memory retention pressure.  `bayes_dual_price` has many more
held-prefix/preempt events (`2090.0` on average) and a much lower delete ratio
(`0.194`) than `lp_dual_descent` (`34.2` preempt events, delete ratio `0.963`).
For vLLM implementation, BDP is the strongest latency candidate in this memory
range, while LP dual descent is the more conservative memory-pressure policy.

The parameters now match the requested scale: prefill mean is in the 100-200
range (`120`), decode mean is `150`, and realized average rounds per program
are about `10`.
