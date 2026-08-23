# Machine-readable simulator settings

These JSON files describe the simulator environment and experiment matrix, not
the measured results. They are generated directly from
`code/realistic_agentic_sim.py` through `code/export_setting.py`.

- `synthetic_heldout.json`: the 45-scenario held-out matrix reported in
  `RESULTS.md`.
- `real_proxy.json`: the nine-scenario Qwen3-32B/vLLM SWE-bench proxy matrix.

Regenerate them after changing a workload, service curve, policy default, or
simulation cadence:

```bash
python code/export_setting.py \
  --preset test \
  --service-profile synthetic \
  --seed0 20260711 \
  --out settings/synthetic_heldout.json

python code/export_setting.py \
  --preset real_proxy \
  --service-profile qwen3_32b_vllm_proxy \
  --seed0 20260717 \
  --out settings/real_proxy.json
```

The JSON intentionally records known model gaps so a proxy setting cannot be
mistaken for a calibrated real-system configuration.
