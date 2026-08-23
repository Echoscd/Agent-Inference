#!/bin/bash
# Start vLLM 0.12 from the in-project editable copy (vllm_dev/).
# Edits to vllm_dev/vllm/**.py take effect on restart (pure-Python, no recompile).
# Usage: bash serve_dev_vllm.sh [offload]   # pass "offload" to enable CPU KV offload
EXP=${AGENT_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$EXP"
# vLLM 0.12.0 is installed from pip (see requirements.txt); set VLLM_SRC only
# if you want to run against a local source checkout instead.
[ -n "${VLLM_SRC:-}" ] && export PYTHONPATH="$VLLM_SRC"
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
ARGS="--model Qwen/Qwen3-32B --served-model-name qwen3-32b --port 8000 \
  --tensor-parallel-size 1 --max-model-len 40960 --max-num-seqs 256 \
  --enable-prefix-caching --trust-remote-code"
if [ "$1" = "offload" ]; then
  ARGS="$ARGS --kv-transfer-config {\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":2000,\"block_size\":256}}"
fi
exec python3 -m vllm.entrypoints.openai.api_server $ARGS
