#!/bin/bash
# A/B for the 2026-08-23 simulator port: A = size (the same baseline as
# experiments 26/27/28), B = sim0823. Config is byte-identical to 26 so
# the size arm is comparable to the three existing replicates.
set -uo pipefail
EXP=${AGENT_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$EXP"

export MODEL_PATH="Qwen/Qwen3-Coder-30B-A3B-Instruct"
export SERVED_NAME="qwen3-coder-30b"
export HZ_ARGS="--hz-decode-mean 300 --hz-prompt-mean 1600 --hz-decode-reserve 1024 --hz-prior coder16"

echo "A=size  B=sim0823  model=$MODEL_PATH  HZ_ARGS=$HZ_ARGS  start=$(date)"
bash algorithm/run_AB_experiment.sh 80 20 29_AB_coder_size_vs_sim0823 sim0823 0.03 1000 6000
echo "FINISHED at $(date)"
