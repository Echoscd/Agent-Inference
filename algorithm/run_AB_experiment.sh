#!/bin/bash
# =============================================================================
# A/B serving experiment on the SAME SWE-bench workload.
#
#   Pass A (baseline): real edit-agent through ThunderAgent, policy=size, temp=0.
#                      Records tape_A.jsonl = every LLM call's full prefill
#                      (messages) + decode (completion) + decode length.
#   Pass B (algorithm): real edit-agent through ThunderAgent, policy=$POLICY
#                      (default density), temp=0. Reads tape_A to send X-Decode-Len
#                      per call (the policy's KNOWN decode), and records tape_B.
#
# Both passes run the REAL agent at temperature 0, so the trajectory is
# deterministic and reproduces turn-for-turn -> pass B's decode equals pass A's
# (verified at the end by comparing tape_A vs tape_B). The only difference between
# the two arms is the ThunderAgent scheduling policy. The vLLM backend is RESTARTED
# before each pass so both start with a cold KV / prefix cache (fair comparison).
#
# Each pass has a wall-clock TIME LIMIT (default 1800s): when a pass hits it, the
# agent run is stopped, its partial results + summary are kept, and the experiment
# moves on (A done -> B). Both passes use the same limit, so it doubles as a fixed
# time budget per arm (a clean goodput comparison: who completes more in the same T).
#
# Usage:  bash algorithm/run_AB_experiment.sh [N_WORKERS] [MAX_TURNS] [TAG] [POLICY] [ALPHA] [DECODE_HAT] [TIME_LIMIT_S]
#   bash algorithm/run_AB_experiment.sh                          # 80 workers, B=density, 1800s/pass -> result/15_AB_density/
#   bash algorithm/run_AB_experiment.sh 48 12 my density 0.03 1000 900   # 48 workers, 12 turns, 900s/pass
#
# Per-arm overrides via env vars (defaults: A=size/base, B=POLICY/base):
#   POLICY_A = pass-A policy (default size). Pass-B policy = the 4th positional arg.
#   Policies: size | density | dual_descent | fidelity (fidelity is a policy now).
#   Fidelity vs dual_descent (isolate the fidelity gaps):
#     POLICY_A=dual_descent \
#       bash algorithm/run_AB_experiment.sh 80 20 fid fidelity 0.03 1000 6000
#       -> A=dual_descent  vs  B=fidelity
# =============================================================================
set -uo pipefail

# ---- config -----------------------------------------------------------------
# Repo root = the directory containing this script's parent (algorithm/ or scripts/).
# Nothing here is tied to a machine-specific path; override with AGENT_EXP_ROOT.
EXP=${AGENT_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# vLLM 0.12.0 comes from pip (requirements.txt). Set VLLM_SRC to run against a
# local source checkout instead; empty means "use the installed package".
VLLM_SRC=${VLLM_SRC:-}
TA_DIR=$EXP/ThunderAgent
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-32B"}
SERVED_NAME=${SERVED_NAME:-"qwen3-32b"}
BACKEND_PORT=8000
TA_PORT=8300
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

WORKERS=${1:-80}
MAX_TURNS=${2:-20}
POLICY=${4:-density}          # pass-B policy (size | density | ...future)
ALPHA=${5:-0.03}              # density: prefill-token cost vs one decode token
DECODE_HAT=${6:-1000}         # density: fallback decode when a call has no known decode
TIME_LIMIT_S=${7:-6000}        # wall-clock limit per pass; on hit -> stop, keep partial, go to B

# Per-arm overrides via env vars (defaults preserve A=size/base, B=POLICY/base):
POLICY_A=${POLICY_A:-size}     # pass-A policy (baseline arm); pass-B policy = $POLICY

# Auto-number the output folder: scan result/ for the highest NN_ prefix and use
# the next one, so each run lands in a fresh result/<NN>_AB_<policy>/ and never
# overwrites a previous run. Pass an explicit 3rd arg (TAG) to override.
_next_num() {
  local m=0 n d
  for d in "$EXP"/result/*/; do
    n=$(basename "$d"); n=${n%%_*}
    [[ $n =~ ^[0-9]+$ ]] && (( 10#$n > m )) && m=$((10#$n))
  done
  printf "%02d" $((m + 1))
}
TAG=${3:-$(_next_num)_AB_${POLICY}}
OUTDIR=$EXP/result/$TAG
IDS_FILE=${IDS_FILE:-$EXP/data/ids80.txt}

export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "$OUTDIR"

# all relative paths below (algorithm/run_swebench_eval.py, algorithm/metrics.py,
# algorithm/plots.py) are resolved from EXP -> make the script cwd-independent so it
# works whether launched from agent-experiment/ or from inside algorithm/.
cd "$EXP" || { echo "cannot cd $EXP"; exit 1; }

cleanup() {
  ps -eo pid,cmd | grep -E "run_swebench_eval|algorithm/metrics.py|ThunderAgent|vllm.entrypoints|EngineCore" \
    | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
}
wait_ready() {  # $1=logfile $2=pid $3=label
  for i in $(seq 1 150); do
    grep -q "Application startup complete" "$1" 2>/dev/null && { echo "[$3] READY ~$((i*5))s"; return 0; }
    kill -0 "$2" 2>/dev/null || { echo "[$3] DIED"; tail -6 "$1"; return 1; }
    sleep 5
  done; echo "[$3] timeout"; return 1
}

# run_pass <label> <policy> <scheduler> <results.jsonl> <kv.csv> <ta.log> <backend.log>
# env in:  AB_RECORD_TAPE / AB_KNOWN_DECODE_TAPE (exported by caller)
run_pass() {
  local LABEL=$1 POL=$2 RES=$3 KV=$4 TALOG=$5 BLOG=$6
  echo "================= PASS $LABEL  (policy=$POL) ================="
  cleanup; sleep 4

  echo "[$LABEL 1/3] starting cold vLLM backend ..."
  ${VLLM_SRC:+PYTHONPATH=$VLLM_SRC} nohup python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" --port $BACKEND_PORT \
      --tensor-parallel-size 1 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS \
      --enable-prefix-caching --trust-remote-code > "$BLOG" 2>&1 &
  wait_ready "$BLOG" $! "backend-$LABEL" || { cleanup; return 1; }

  echo "[$LABEL 2/3] starting ThunderAgent (policy=$POL alpha=$ALPHA decode_hat=$DECODE_HAT) ..."
  ( cd "$TA_DIR" && TA_DECISION_TRACE="$OUTDIR/decision_trace_$LABEL.jsonl" \
      nohup python3 -m ThunderAgent --port $TA_PORT \
      --backends http://localhost:$BACKEND_PORT --router tr --backend-type vllm --metrics \
      --scheduler-interval 3 --policy "$POL" --alpha "$ALPHA" --decode-hat "$DECODE_HAT" \
      ${HZ_ARGS:-} \
      > "$TALOG" 2>&1 & )
  sleep 12
  grep -qi "Application startup complete" "$TALOG" && echo "[ThunderAgent-$LABEL] READY" \
    || { echo "[ThunderAgent-$LABEL] not ready"; tail -8 "$TALOG"; }

  PYTHONUNBUFFERED=1 nohup python3 algorithm/metrics.py "$KV" > "$OUTDIR/kv_sampler_$LABEL.log" 2>&1 &
  local SAMPLER_PID=$!

  echo "[$LABEL 3/3] running $WORKERS-way edit-agent (temp=0, time limit ${TIME_LIMIT_S}s) ..."
  AGENT_BASE_URL=http://localhost:$TA_PORT/v1 PYTHONUNBUFFERED=1 \
    timeout --signal=TERM --kill-after=30s "${TIME_LIMIT_S}s" \
    python3 algorithm/run_swebench_eval.py --instance-ids "$(cat "$IDS_FILE")" \
      --workers $WORKERS --max-turns $MAX_TURNS --agent edit --model "$SERVED_NAME" \
      --output "$RES" > "$OUTDIR/run_$LABEL.log" 2>&1
  local rc=$?
  if [ $rc -eq 124 ]; then
    echo "[$LABEL] HIT TIME LIMIT ${TIME_LIMIT_S}s -> stopping pass; partial results + summary kept."
  fi
  grep -E "\[done\]|REPORT|resolved|time-limit" "$OUTDIR/run_$LABEL.log" | tail -6 || true

  kill -9 $SAMPLER_PID 2>/dev/null
  cleanup; sleep 4
}

# ---- Pass A: baseline arm (policy=$POLICY_A), record tape_A ----------------------
unset AB_KNOWN_DECODE_TAPE
export AB_RECORD_TAPE="$OUTDIR/tape_A.jsonl"
: > "$AB_RECORD_TAPE"
run_pass A "$POLICY_A" "$OUTDIR/results_A.jsonl" "$OUTDIR/kv_A.csv" "$OUTDIR/ta_A.log" "$OUTDIR/backend_A.log"

# ---- Pass B: test arm (policy=$POLICY), use tape_A decode ------------------------
export AB_KNOWN_DECODE_TAPE="$OUTDIR/tape_A.jsonl"
export AB_RECORD_TAPE="$OUTDIR/tape_B.jsonl"
: > "$AB_RECORD_TAPE"
run_pass B "$POLICY" "$OUTDIR/results_B.jsonl" "$OUTDIR/kv_B.csv" "$OUTDIR/ta_B.log" "$OUTDIR/backend_B.log"

# ---- verify decode reproduced + compare metrics ---------------------------------
echo "================= VERIFY + COMPARE ================="
python3 - "$OUTDIR/tape_A.jsonl" "$OUTDIR/tape_B.jsonl" <<'PY'
import json, sys
def load(p):
    d={}
    for l in open(p):
        l=l.strip()
        if not l: continue
        try: r=json.loads(l)
        except Exception: continue
        d[(r["iid"], r["turn"])]=(r["gen_tokens"], r["completion"])
    return d
A=load(sys.argv[1]); B=load(sys.argv[2])
common=set(A)&set(B)
len_match=sum(1 for k in common if A[k][0]==B[k][0])
txt_match=sum(1 for k in common if A[k][1]==B[k][1])
print(f"calls: A={len(A)}  B={len(B)}  common(iid,turn)={len(common)}")
print(f"  decode-LENGTH identical : {len_match}/{len(common)} ({100*len_match/max(1,len(common)):.1f}%)")
print(f"  decode-TEXT   identical : {txt_match}/{len(common)} ({100*txt_match/max(1,len(common)):.1f}%)")
onlyA=set(A)-set(B); onlyB=set(B)-set(A)
if onlyA or onlyB:
    print(f"  trajectory drift: only-in-A={len(onlyA)} only-in-B={len(onlyB)} (temp=0 should make this 0)")
PY

echo "--- plots (from TAPES = full population incl. unfinished programs) ---"
python3 algorithm/plots.py pdt   "$OUTDIR/tape_A.jsonl" "$OUTDIR/pdt_A.png"   "A size ${WORKERS}-way (all programs)" 2>/dev/null | tail -1
python3 algorithm/plots.py pdt   "$OUTDIR/tape_B.jsonl" "$OUTDIR/pdt_B.png"   "B ${POLICY} ${WORKERS}-way (all programs)" 2>/dev/null | tail -1
python3 algorithm/plots.py gantt "$OUTDIR/tape_A.jsonl" "$OUTDIR/gantt_A.png" 0 2>/dev/null | tail -1
python3 algorithm/plots.py gantt "$OUTDIR/tape_B.jsonl" "$OUTDIR/gantt_B.png" 0 2>/dev/null | tail -1
python3 algorithm/plots.py compare "$OUTDIR/kv_compare.png" "A size=$OUTDIR/kv_A.csv" "B $POLICY=$OUTDIR/kv_B.csv" 2>/dev/null | tail -1
python3 algorithm/plots.py concurrency "$OUTDIR/concurrency.png" "A size=$OUTDIR/kv_A.csv" "B $POLICY=$OUTDIR/kv_B.csv" 2>/dev/null | tail -1
# gantt companion: concurrency + KV utilization over time in one figure (explains gantt grey/gaps)
python3 algorithm/plots.py conckv "$OUTDIR/conc_kv.png" "A size=$OUTDIR/kv_A.csv" "B $POLICY=$OUTDIR/kv_B.csv" 2>/dev/null | tail -1

# Post-warmup / steady-state throughput + p90/p95 call latency, from the TAPES
# (full population) and the KV sampler csv. The results_*_summary.json numbers are
# whole-run: they include the cold ramp and the drain tail. -> steady_metrics.{json,csv}
echo "--- post-warmup steady-state metrics (throughput, p90/p95 call latency) ---"
python3 algorithm/warmup_metrics.py --out "$OUTDIR" "$OUTDIR" 2>&1 | tail -4

echo "[done] A/B artifacts in $OUTDIR :"
ls -1 "$OUTDIR"
echo "Compare results_A/_summary.json vs results_B/_summary.json for preemptions / goodput / resolve."
