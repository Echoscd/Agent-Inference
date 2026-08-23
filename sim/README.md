# Realistic Agentic-Serving Simulator v2

This package implements and evaluates a prediction-light scheduling policy for
agentic LLM serving.  It is a replacement experiment scaffold for the original
`agentic_kv_sim.py`, not a patch that preserves its idealized execution model.

The main deployment candidate is **`hazard_grade_knapsack`**:

1. Keep the student's useful posterior program-work ordering for ready LLM turns.
2. Remove the global BDP admission price, which remained zero throughout the
   revised held-out experiments.
3. Value an inactive KV prefix using the probability that its tool returns within
   a short horizon, conditioned on elapsed tool age.
4. Pack retained prefixes by an exact block-level 0/1 knapsack.
5. Use only class-level distributions and an output-length quantile.  A policy
   never receives the realized next decode length or remaining tool duration.

On 45 held-out synthetic scenarios, `hazard_grade_knapsack` improved mean program
completion time by **1.16%** relative to the student's BDP policy, with a paired
bootstrap 95% interval of **[0.67%, 1.76%]**, and won 38 of 45 scenarios.  This is
small but statistically stable simulator evidence.  It is not yet a prediction of
GPU improvement because the service curves have not been fitted to ThunderAgent
traces.

## Files

- `SIMULATOR_SETTING.md`: canonical environment specification: workload
  generation, program state machine, tool timing, GPU service, KV accounting,
  scheduling cadence, policy information boundary, metrics, and presets.
- `settings/synthetic_heldout.json` and `settings/real_proxy.json`:
  machine-readable settings generated from the simulator code, separate from
  measured result files.
- `code/export_setting.py`: regenerates the machine-readable setting files
  after an environment or policy-default change.
- `code/realistic_agentic_sim.py`: simulator, policies, experiment runner, and
  summary generation.
- `code/run_sensitivity.py`: one-factor robustness checks for decode saturation
  and scheduler control interval.
- `tests/test_simulator.py`: non-clairvoyance, memory-safety, reproducibility, and
  knapsack tests.
- `RESULTS.md`: experiment design, tables, interpretation, and limitations.
- `IMPLEMENTATION_GUIDE.md`: concrete path from this simulator to the supplied
  Thunder scheduler.
- `CHANGES_FROM_ORIGINAL.md`: assumption-by-assumption comparison with the
  student's original simulator.
- `results/validation_final/`: validation scenarios used to choose the
  implementation-first hybrid.
- Long runs write `raw_checkpoint.csv` after every completed policy/scenario pair.
- `results/test_heldout/`: complete held-out data, paired tables, summaries, and
  plots.
- `results/sensitivity/`: service-curve and control-interval robustness results.

## Installation and tests

```bash
python -m pip install -r requirements.txt
pytest -q
```

The checked-in version passes all fifteen tests.

## Quick run

```bash
python code/realistic_agentic_sim.py \
  --preset smoke \
  --out results/my_smoke
```

The complete semantics and exact canonical matrices are documented in
`SIMULATOR_SETTING.md`. To inspect a configuration without running an
experiment:

```bash
python code/export_setting.py \
  --preset real_proxy \
  --service-profile qwen3_32b_vllm_proxy \
  --seed0 20260717 \
  --out settings/real_proxy.json
```

A larger run:

```bash
python code/realistic_agentic_sim.py \
  --preset test \
  --seed0 20260711 \
  --out results/my_test
```

Robustness checks:

```bash
python code/run_sensitivity.py
```

## Real-system proxy profile

The original presets and checked-in held-out results remain unchanged.  A
separate proxy profile maps the aggregate Qwen3-32B/vLLM experiment summary to a
long-context SWE-bench-like workload:

```bash
python code/realistic_agentic_sim.py \
  --preset real_proxy_smoke \
  --service-profile qwen3_32b_vllm_proxy \
  --policies thunder_greedy,dual_price_proxy,student_bdp,hazard_grade_knapsack \
  --thunder-policy thunder_greedy \
  --out results/real_proxy_smoke
```

The proxy uses 80 programs, an approximately 14k-token initial task/repository
prompt, 1.8k incremental prompt tokens, 1.65k decode tokens per turn, an
approximately nine-turn prior, and a slower long-context prefill/decode service
curve.  `thunder_greedy` approximates the work-conserving Size/Thunder policy
described in the real experiment summary; it does not inherit the synthetic
Thunder high/low admission gate.

For a nine-scenario capacity sweep, replace `real_proxy_smoke` with `real_proxy`.
The capacities are 360k, 480k, and 600k KV tokens with three seeds each.

This profile is deliberately labelled a proxy, not a calibrated digital twin.
The presentation contains aggregate token, concurrency, and throughput metrics,
but not per-call prefill GPU time, tool timestamps, reusable-prefix blocks, or
router decisions.  Replace the proxy constants with fits from those raw traces
before interpreting effect sizes as deployment forecasts.

### Decision-level trace visualization

Generate paired scheduler traces on the exact same workload instance:

```bash
python code/trace_compare.py \
  --preset real_proxy_smoke \
  --service-profile qwen3_32b_vllm_proxy \
  --policies thunder_greedy,hazard_grade_knapsack \
  --seed0 20260717 \
  --early-window 1200 \
  --out results/trace_real_proxy
```

The output contains a full scheduler/KV overview, an early admit/evict event
raster, a per-program state heatmap, and CSV files for scheduler snapshots,
decision events, phase transitions, metrics, and workload attributes.  The trace
recorder observes realized events only after decisions and does not expose them
to policies.

## Policies

| Policy | Admission | Tool/ready cache retention |
|---|---|---|
| `thunder` | Continuations and smaller contexts first, with high/low hysteresis | Acting caches released first using decayed effective size and shortest-first order |
| `thunder_greedy` | Work-conserving Size/Thunder proxy; greedily fills active reservations | Passive smallest-prefix-first eviction proxy |
| `dual_price_proxy` | Real-slide one-turn density minus a scalar admission price | Passive density ordering; Q95 simulator reservation retained |
| `student_bdp` | Student formula `1/(tau*mu) - lambda*qbar` | Student hold margin `alpha*A/mu - lambda*A` |
| `hazard_knapsack` | Thunder-like greedy admission | Residual-return value plus exact knapsack |
| `hazard_grade_knapsack` | Posterior current-work × remaining-work density, no global price | Residual-return value plus exact knapsack |
| `hazard_batch_knapsack` | Same grade but capped at a fitted decode-service knee | Residual-return value plus exact knapsack |

`hazard_batch_knapsack` is retained as an ablation.  Its uncalibrated batch cap
was **2.60% worse** than BDP in held-out mean completion time, so it should not be
ported to the GPU before the service curve is measured.

## Non-clairvoyance

`RequestSpec` stores realized output and tool duration for the simulator only.
Policies receive a separate `ProgramView` that deliberately omits those fields.
They can observe:

- current prefix and revealed incremental prompt size;
- stage and posterior remaining-round statistics;
- tool class and elapsed tool age;
- class-level output and tool distributions;
- block capacity and fitted GPU service curves.

This separation is tested in `test_policy_view_hides_realized_future`.

## Calibration boundary

The revised simulator is suitable for algorithm falsification and controlled
ablation.  It is not yet a digital twin.  Before using its effect sizes as a GPU
forecast, fit the prefill/decode service functions, cache-release semantics, and
tool distributions from the exact Qwen-32B/ThunderAgent deployment.
