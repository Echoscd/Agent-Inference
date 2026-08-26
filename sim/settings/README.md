# Generated setting

`coder_calibration.json` is the only machine-readable simulator setting. It is
generated from code and should not be edited manually.

```bash
python code/export_setting.py \
  --preset coder_calibration \
  --service-profile qwen3_coder_30b_vllm_012 \
  --seed0 20260826 \
  --control-interval 3 \
  --out settings/coder_calibration.json
```
