#!/bin/bash
# Run N replicates of one A/B scheduling experiment.
#
# Replaces four near-identical launcher scripts (run_hazard_ab{,_coder,_coder_28,
# _coder_5x}.sh) that differed only in model, hazard params and replicate numbers.
#
# Usage:
#   scripts/run_replicates.sh <preset> <first_n> <count> [policy_B]
#
#   preset    reasoning | coder     which served model + hazard prior to use
#   first_n   result folder number of the first replicate (result/<n>_AB_...)
#   count     how many replicates to run, sequentially (they share one GPU)
#   policy_B  test-arm policy (default hazard_grade). Arm A is always size.
#
# Examples:
#   scripts/run_replicates.sh coder 29 4                 # -> result/29..32_AB_coder_size_vs_hazard
#   scripts/run_replicates.sh reasoning 40 2 fidelity    # A=size vs B=fidelity on Qwen3-32B
#
# Each replicate is a full A/B: cold vLLM + arm A (size), cold vLLM + arm B.
# Expect ~40 min per replicate for the coder preset on one 140 GB GPU.
set -uo pipefail
EXP=${AGENT_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$EXP"

PRESET=${1:?preset required: reasoning | coder}
FIRST=${2:?first replicate number required}
COUNT=${3:-1}
POLICY_B=${4:-hazard_grade}

case "$PRESET" in
  reasoning)
    export MODEL_PATH="Qwen/Qwen3-32B"
    export SERVED_NAME="qwen3-32b"
    export HZ_ARGS=""                      # hazard_grade defaults suit the CoT regime
    TAG_KIND="size_vs_${POLICY_B%%_*}"
    ;;
  coder)
    export MODEL_PATH="Qwen/Qwen3-Coder-30B-A3B-Instruct"
    export SERVED_NAME="qwen3-coder-30b"
    # retuned for the coder regime (short decode, ~16 turns), measured off exp23's tape
    export HZ_ARGS="--hz-decode-mean 300 --hz-prompt-mean 1600 --hz-decode-reserve 1024 --hz-prior coder16"
    TAG_KIND="coder_size_vs_${POLICY_B%%_*}"
    ;;
  *) echo "unknown preset '$PRESET' (want: reasoning | coder)"; exit 2 ;;
esac

# Fixed across every replicate so runs stay comparable: 80 workers, 20 turns,
# alpha 0.03, decode_hat 1000, 6000s per arm.
WORKERS=80 MAX_TURNS=20 ALPHA=0.03 DECODE_HAT=1000 LIMIT=6000

echo "preset=$PRESET model=$MODEL_PATH policy_B=$POLICY_B replicates=$FIRST..$((FIRST+COUNT-1))"
echo "HZ_ARGS=${HZ_ARGS:-<defaults>}   start=$(date)"
for i in $(seq 0 $((COUNT - 1))); do
  N=$((FIRST + i))
  TAG="${N}_AB_${TAG_KIND}"
  echo "############################################################"
  echo "# REPLICATE $N -> result/$TAG   $(date)"
  echo "############################################################"
  bash algorithm/run_AB_experiment.sh "$WORKERS" "$MAX_TURNS" "$TAG" "$POLICY_B" \
       "$ALPHA" "$DECODE_HAT" "$LIMIT"
  echo "REPLICATE $N finished at $(date)"
done
echo "ALL DONE  $(date)"
