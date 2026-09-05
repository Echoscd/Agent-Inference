#!/bin/bash
# Wait for experiment 30 to finish, then run its replicate as experiment 31.
# Same config in both, so the pair measures run-to-run variance rather than a
# configuration difference.
set -uo pipefail
EXP=${AGENT_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$EXP"

echo "waiting for experiment 30 to finish ... $(date)"
while pgrep -f "run_bdp_30.sh" > /dev/null; do sleep 20; done
echo "experiment 30 process gone at $(date); waiting for the GPU to drain"

# The harness cleans up its own processes, but wait for the memory to be
# released before starting a fresh cold backend.
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-1}" -lt 2000 ] && { echo "GPU free (${used} MiB) after $((i*10))s"; break; }
  sleep 10
done
sleep 15

echo "starting experiment 31 at $(date)"
bash scripts/run_bdp_31.sh
