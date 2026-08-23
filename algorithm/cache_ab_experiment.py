"""
A/B experiment: prefix/KV cache reuse vs no reuse, on ONE long agent trajectory.

Step 1 (capture): run the thinking edit-agent once on a long instance and record
the EXACT messages sent at every turn (the growing conversation). Save to JSON.

Step 2 (replay): re-send that identical per-turn request sequence and measure, per
turn, prefill latency (client TTFT, no concurrency -> ~= prefill) and decode time.
  - Condition A : server has --enable-prefix-caching  -> turn k reuses the cached
                  prefix from earlier turns (hit ~100%); prefill only the new tokens.
  - Condition B : server started WITHOUT prefix caching -> every turn re-prefills
                  the whole growing context from scratch (hit 0%).

Same requests, only the cache condition differs -> isolates the effect of cache
reuse on total prefill vs total decode.

Usage:
  python cache_ab_experiment.py capture --instance-id <id> --max-turns 6
  python cache_ab_experiment.py replay  --label A         # on cache-ON server
  # (restart server without --enable-prefix-caching)
  python cache_ab_experiment.py replay  --label B         # on cache-OFF server
"""
import json
import re
import sys
import time
import argparse
import urllib.request

import swebench_edit_agent as A
import paths

TRAJ_FILE = "cache_ab_traj.json"
RES_FILE  = "cache_ab_results.json"


def _prefix_counters():
    try:
        t = urllib.request.urlopen("http://localhost:8000/metrics", timeout=5).read().decode()
        q = float(re.search(r'vllm:prefix_cache_queries_total\S*\s+([\d.e+\-]+)', t).group(1))
        h = float(re.search(r'vllm:prefix_cache_hits_total\S*\s+([\d.e+\-]+)', t).group(1))
        return q, h
    except Exception:
        return 0.0, 0.0


# ── capture ────────────────────────────────────────────────────────────────
def capture(args):
    inst = next(json.loads(l) for l in open(args.data)
                if json.loads(l)["instance_id"] == args.instance_id)
    A.MODEL = args.model

    turns = []
    orig = A._stream_call
    def cap(messages):
        turns.append([dict(m) for m in messages])   # snapshot of full context this turn
        return orig(messages)
    A._stream_call = cap

    print(f"capturing trajectory: {args.instance_id} (model={A.MODEL}, max_turns={args.max_turns})")
    r = A.run_agent(inst, verbose=True, max_turns=args.max_turns)
    A._stream_call = orig

    json.dump({"instance_id": args.instance_id, "model": args.model, "turns": turns},
              open(TRAJ_FILE, "w"))
    print(f"\ncaptured {len(turns)} turns -> {TRAJ_FILE}")
    print(f"(agent: turns={r.turns} gen={r.total_gen_tokens} resolved={r.resolved})")


# ── replay ───────────────────────────────────────────────────────────────────
def replay(args):
    traj = json.load(open(TRAJ_FILE))
    turns = traj["turns"]
    A.MODEL = traj["model"]
    print(f"replay [{args.label}] : {len(turns)} turns, model={A.MODEL}")

    q0, h0 = _prefix_counters()
    tot_prefill = tot_decode = tot_gen = 0
    rows = []
    for i, msgs in enumerate(turns, 1):
        _, t = A._stream_call(msgs)
        tot_prefill += t.ttft_s
        tot_decode  += t.decode_s
        tot_gen     += t.gen_tokens
        rows.append({"turn": i, "prompt_tokens": t.prompt_tokens,
                     "prefill_ms": t.ttft_s * 1000, "decode_s": t.decode_s,
                     "gen_tokens": t.gen_tokens})
        print(f"  turn {i:2d}: prompt={t.prompt_tokens:6d}  "
              f"prefill(TTFT)={t.ttft_s*1000:7.0f}ms  decode={t.decode_s:6.1f}s  gen={t.gen_tokens}")
    q1, h1 = _prefix_counters()
    dq, dh = q1 - q0, h1 - h0
    hit = (dh / dq) if dq > 0 else 0.0

    print(f"\n  [{args.label}] TOTAL prefill = {tot_prefill:.2f}s   "
          f"TOTAL decode = {tot_decode:.2f}s   gen = {tot_gen} tok")
    print(f"  [{args.label}] prefix-cache hit rate this replay = {hit:.1%} "
          f"({int(dh)}/{int(dq)} tokens)")

    # accumulate results for A/B comparison
    try:
        res = json.load(open(RES_FILE))
    except FileNotFoundError:
        res = {}
    res[args.label] = {"total_prefill_s": tot_prefill, "total_decode_s": tot_decode,
                       "total_gen": tot_gen, "hit_rate": hit, "rows": rows}
    json.dump(res, open(RES_FILE, "w"))

    if "A" in res and "B" in res:
        a, b = res["A"], res["B"]
        print("\n" + "=" * 64)
        print("  A/B COMPARISON  (A = cache reuse / ~100% hit, B = cache dropped / 0% hit)")
        print("=" * 64)
        print(f"  {'metric':<22}{'A (reuse)':>14}{'B (drop)':>14}{'B/A':>10}")
        print(f"  {'prefix hit rate':<22}{a['hit_rate']*100:>13.1f}%{b['hit_rate']*100:>13.1f}%{'-':>10}")
        print(f"  {'total prefill (s)':<22}{a['total_prefill_s']:>14.2f}{b['total_prefill_s']:>14.2f}"
              f"{b['total_prefill_s']/max(a['total_prefill_s'],1e-9):>9.1f}x")
        print(f"  {'total decode (s)':<22}{a['total_decode_s']:>14.2f}{b['total_decode_s']:>14.2f}"
              f"{b['total_decode_s']/max(a['total_decode_s'],1e-9):>9.1f}x")
        ta = a['total_prefill_s'] + a['total_decode_s']
        tb = b['total_prefill_s'] + b['total_decode_s']
        print(f"  {'prefill+decode (s)':<22}{ta:>14.2f}{tb:>14.2f}{tb/max(ta,1e-9):>9.1f}x")
        print("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--instance-id", default="pylint-dev__pylint-6903")
    c.add_argument("--data", default=paths.SWEBENCH_DATA)
    c.add_argument("--model", default="qwen3-32b")
    c.add_argument("--max-turns", type=int, default=6)
    r = sub.add_parser("replay")
    r.add_argument("--label", required=True, choices=["A", "B"])
    args = ap.parse_args()
    (capture if args.cmd == "capture" else replay)(args)
