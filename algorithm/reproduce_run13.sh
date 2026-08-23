#!/bin/bash
# =============================================================================
# Reproduce result/13_thunderagent_full80_trace : 80-way SWE-bench edit-agent
# served through ThunderAgent (program-level scheduling) on a local vLLM 0.12
# backend, recording per-program per-turn traces (wait/decode/tool), KV/preempt
# time series, and the standard prefill/decode/turn figure.
#
# Pipeline:
#   agents (run_swebench_eval, edit agent, X-Session-ID=program_id)
#        -> ThunderAgent proxy :8300 (router=tr, program pause/resume at tool
#           boundary, keeps active KV <= GPU capacity, profiling on)
#        -> vLLM 0.12 backend :8000 (Qwen3-32B, 40960 ctx, prefix caching, no
#           CPU offload; flashinfer sampler disabled)
#
# Usage:  bash algorithm/reproduce_run13.sh [N_WORKERS] [MAX_TURNS] [OUTDIR_TAG]
# Defaults reproduce run 13 exactly (80 workers, 20 turns).
# =============================================================================
set -uo pipefail

# ---- config -----------------------------------------------------------------
# Repo root = the directory containing this script's parent (algorithm/ or scripts/).
# Nothing here is tied to a machine-specific path; override with AGENT_EXP_ROOT.
EXP=${AGENT_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# vLLM 0.12.0 comes from pip (requirements.txt). Set VLLM_SRC to run against a
# local source checkout instead; empty means "use the installed package".
VLLM_SRC=${VLLM_SRC:-}
TA_DIR=$EXP/ThunderAgent                          # ThunderAgent package dir
MODEL_PATH="Qwen/Qwen3-32B"
SERVED_NAME="qwen3-32b"
BACKEND_PORT=8000
TA_PORT=8300
MAX_MODEL_LEN=40960
MAX_NUM_SEQS=256

WORKERS=${1:-80}
MAX_TURNS=${2:-20}
TAG=${3:-13_thunderagent_full80_trace}
OUTDIR=$EXP/result/$TAG
IDS_FILE=${IDS_FILE:-$EXP/data/ids80.txt}                           # 80 pre-built testbed instances

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "$OUTDIR/ta_profiles"

cleanup() {
  echo "[cleanup] stopping eval/sampler/ThunderAgent/vLLM ..."
  ps -eo pid,cmd | grep -E "run_swebench_eval|kv_sampler|ThunderAgent|vllm.entrypoints|EngineCore" \
    | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
}
wait_ready() {  # $1=logfile $2=pid $3=label
  for i in $(seq 1 150); do
    grep -q "Application startup complete" "$1" 2>/dev/null && { echo "[$3] READY ~$((i*5))s"; return 0; }
    kill -0 "$2" 2>/dev/null || { echo "[$3] DIED"; tail -6 "$1"; return 1; }
    sleep 5
  done; echo "[$3] timeout"; return 1
}

cleanup; sleep 4

# ---- 1. vLLM 0.12 backend (from the editable vllm_dev copy) ------------------
echo "[1/5] starting vLLM backend on :$BACKEND_PORT ..."
${VLLM_SRC:+PYTHONPATH=$VLLM_SRC} nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --port $BACKEND_PORT \
    --tensor-parallel-size 1 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS \
    --enable-prefix-caching --trust-remote-code > "$OUTDIR/vllm_backend.log" 2>&1 &
BACKEND_PID=$!
wait_ready "$OUTDIR/vllm_backend.log" $BACKEND_PID "backend" || { cleanup; exit 1; }

# ---- 2. ThunderAgent proxy (program scheduler, tr mode, profiling) ----------
echo "[2/5] starting ThunderAgent on :$TA_PORT ..."
cd "$TA_DIR"
nohup python3 -m ThunderAgent --port $TA_PORT --backends http://localhost:$BACKEND_PORT \
    --router tr --backend-type vllm --metrics --scheduler-interval 3 \
    --profile --profile-dir "$OUTDIR/ta_profiles" > "$OUTDIR/thunderagent.log" 2>&1 &
TA_PID=$!
cd "$EXP"
sleep 12
grep -qi "Application startup complete" "$OUTDIR/thunderagent.log" && echo "[ThunderAgent] READY" \
  || { echo "[ThunderAgent] not ready"; tail -8 "$OUTDIR/thunderagent.log"; }

# ---- 3. KV/preempt sampler on the backend -----------------------------------
echo "[3/5] starting backend KV sampler ..."
PYTHONUNBUFFERED=1 nohup python3 algorithm/metrics.py "$OUTDIR/kv_timeseries.csv" > "$OUTDIR/kv_sampler.log" 2>&1 &
SAMPLER_PID=$!

# ---- 4. run 80-way SWE-bench edit-agent THROUGH ThunderAgent ----------------
echo "[4/5] running $WORKERS-way edit-agent (max_turns=$MAX_TURNS) via ThunderAgent ..."
IDS=$(cat "$IDS_FILE")
AGENT_BASE_URL=http://localhost:$TA_PORT/v1 PYTHONUNBUFFERED=1 \
  python3 algorithm/run_swebench_eval.py --instance-ids "$IDS" --workers $WORKERS --max-turns $MAX_TURNS \
    --agent edit --model "$SERVED_NAME" --output "$OUTDIR/results.jsonl" 2>&1 \
    | tee "$OUTDIR/run.log" | grep -E "\[done\]|REPORT|resolved" || true

kill -9 $SAMPLER_PID 2>/dev/null

# ---- 5. standard plots -------------------------------------------------------
echo "[5/5] plotting ..."
python3 algorithm/plots.py pdt "$OUTDIR/results.jsonl" "$OUTDIR/pdt.png" "ThunderAgent ${WORKERS}-way" 2>/dev/null | tail -1
python3 algorithm/plots.py concurrency "$OUTDIR/concurrency.png" "${WORKERS}-way=$OUTDIR/kv_timeseries.csv" 2>/dev/null | tail -1

echo "[done] artifacts in $OUTDIR :"
ls -1 "$OUTDIR"
echo "(backend + ThunderAgent left running; run 'cleanup' or kill api_server/ThunderAgent to free GPU)"
