"""
Batch SWE-bench runner for inference-acceleration experiments.

Drives the multi-turn agent over N instances (concurrently), while a metrics
monitor samples server-side throughput + KV-cache. Reports BOTH:

  - quality signal : resolve rate (Docker-free conda harness, official grading)
  - inference cost : TTFT, decode time, gen tokens/s, tool-wait, turns, server
                     throughput, KV-cache peak/avg

Use it to compare an acceleration change (e.g. spec-decoding / quant) against a
baseline on the SAME instance subset: relative resolve rate must not drop while
throughput/latency improve.

Usage:
  python run_swebench_eval.py --n 4 --workers 2 --repos psf/requests
  python run_swebench_eval.py --instance-ids psf__requests-1142,pallets__flask-...
"""
import os
import signal
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from metrics import MetricsMonitor
import run_metrics as rm
import paths

# repos that build quickly / reliably under conda (good for a fast smoke batch)
EASY_REPOS = ["psf/requests", "pallets/flask", "mwaskom/seaborn", "pylint-dev/pylint"]


def load_instances(data, n, repos, instance_ids, start):
    rows = [json.loads(l) for l in open(data)]
    if instance_ids:
        want = set(instance_ids.split(","))
        return [r for r in rows if r["instance_id"] in want]
    if repos:
        repo_set = set(repos.split(","))
        rows = [r for r in rows if r["repo"] in repo_set]
    rows = rows[start:]
    return rows[:n] if n else rows


def _fmt(n, unit=""):
    if n >= 1e6: return f"{n/1e6:.2f}M{unit}"
    if n >= 1e3: return f"{n/1e3:.1f}k{unit}"
    return f"{n:.0f}{unit}"


def report(results, wall_s: float, monitor_summary):
    n = len(results)
    resolved = sum(1 for r in results if r.resolved)
    gen_tok  = sum(r.total_gen_tokens for r in results)
    pmt_tok  = sum(r.total_prompt_tokens for r in results)
    dec_s    = sum(r.total_decode_s for r in results)
    wait_s   = sum(r.total_wait_s for r in results)
    turns    = [r.turns for r in results]
    latencies = sorted(r.total_ttft_s + r.total_decode_s + r.total_wait_s for r in results)

    print("\n" + "=" * 72)
    print("  SWE-BENCH AGENT EVALUATION REPORT")
    print("=" * 72)

    print(f"\n{'Quality (resolve)':-<40}")
    print(f"  resolved:        {resolved}/{n} = {resolved/n:.1%}" if n else "  (no instances)")
    print(f"  submitted:       {sum(1 for r in results if r.submitted)}/{n}")
    print(f"  produced patch:  {sum(1 for r in results if r.patch.strip())}/{n}")
    print(f"  avg turns:       {sum(turns)/n:.1f}  (max {max(turns) if turns else 0})")

    print(f"\n{'Inference cost (client-side)':-<40}")
    print(f"  wall time:       {wall_s:.1f}s")
    print(f"  gen tokens:      {_fmt(gen_tok)}   ({gen_tok/dec_s:.0f} t/s decode)" if dec_s else "  gen tokens: 0")
    print(f"  prompt tokens:   {_fmt(pmt_tok)}")
    print(f"  decode time:     {dec_s:.1f}s")
    print(f"  tool wait:       {wait_s:.1f}s  (real inter-turn execution)")
    if latencies:
        print(f"  latency:        avg {sum(latencies)/len(latencies):.1f}s  "
              f"p90 {_percentile(latencies, 90):.1f}s  p95 {_percentile(latencies, 95):.1f}s  "
              f"p99 {_percentile(latencies, 99):.1f}s")
    print(f"  avg TTFT:        {sum(r.total_ttft_s for r in results)/max(sum(turns),1)*1000:.0f}ms / turn")

    print(f"\n{'Server-side (vLLM metrics)':-<40}")
    if monitor_summary and getattr(monitor_summary, 'snapshots', None):
        s = monitor_summary
        print(f"  prompt thruput:  {s.prompt_throughput:.1f} t/s")
        print(f"  gen thruput:     {s.gen_throughput:.1f} t/s")
        print(f"  concurrency:     max {s.max_running} / avg {s.avg_running:.1f} running reqs")
        print(f"  KV peak/avg:     {s.kv_peak_perc:.1f}% / {s.kv_avg_perc:.1f}%")
        print(f"  prefix-cache:    hit rate {s.prefix_cache_hit_rate:.1%}  "
              f"({_fmt(s.prefix_cache_hits)} hits / {_fmt(s.prefix_cache_queries)} queries, tokens)")
        print(f"  KV curve:        {s.ascii_sparkline(width=50)}")
    else:
        print("  (metrics endpoint unavailable)")

    print(f"\n{'Per-instance':-<40}")
    print(f"  {'instance':<28}{'resolved':>9}{'turns':>6}{'gen':>7}{'dec(s)':>8}{'wait(s)':>8}")
    for r in sorted(results, key=lambda x: x.instance_id):
        print(f"  {r.instance_id:<28}{('YES' if r.resolved else r.status):>9}"
              f"{r.turns:>6}{_fmt(r.total_gen_tokens):>7}{r.total_decode_s:>8.1f}{r.total_wait_s:>8.1f}")
    print("=" * 72 + "\n")


# Percentiles and the distribution block come from run_metrics -- the single
# source of truth shared with warmup_metrics/plots (one formula, one p95).
_percentile = rm.pctl
_stats = rm.dist


def write_summary(path, results, wall_s, mon, args):
    """Aggregate stats JSON: agent length/turns, throughput, KV utilization,
    prefix-cache, concurrency, preemptions, resolve quality."""
    n = len(results)
    gen = [r.total_gen_tokens for r in results]
    pmt = [r.total_prompt_tokens for r in results]
    turns = [r.turns for r in results]
    dec = [r.total_decode_s for r in results]
    ttft = [r.total_ttft_s for r in results]
    tool_wait = [r.total_wait_s for r in results]
    latency = [r.total_ttft_s + r.total_decode_s + r.total_wait_s for r in results]
    has_mon = bool(mon and getattr(mon, "snapshots", None))
    summary = {
        "config": {
            "agent": getattr(args, "agent", None), "model": getattr(args, "model", None),
            "workers": args.workers, "max_turns": args.max_turns,
            "prefix_caching": getattr(args, "prefix_caching", None), "completed": n,
        },
        "quality": {
            "resolved": sum(1 for r in results if r.resolved),
            "resolve_rate": round(sum(1 for r in results if r.resolved) / n, 4) if n else 0,
            "submitted": sum(1 for r in results if r.submitted),
            "produced_patch": sum(1 for r in results if r.patch.strip()),
            "errored": sum(1 for r in results if r.error),
        },
        "agent": {
            "turns": _stats(turns),
            "decode_length_tok": _stats(gen),          # generated (decode) tokens / agent
            "prompt_tokens": _stats(pmt),              # total prompt tokens / agent (sum over turns)
            "avg_decode_s": round(sum(dec) / n, 1) if n else 0,
            "latency_s": _stats(latency),          # per-agent end-to-end: TTFT + decode + tool wait
            "average_latency_s": round(sum(latency) / n, 1) if n else 0,
            "p90_latency_s": round(_percentile(sorted(latency), 90), 1) if n else 0,
            "p95_latency_s": round(_percentile(sorted(latency), 95), 1) if n else 0,
            "p99_latency_s": round(_percentile(sorted(latency), 99), 1) if n else 0,
            "decode_s": _stats(dec),              # per-agent generation time
            "ttft_s": _stats(ttft),               # per-agent summed TTFT / queue+prefill time
            "tool_wait_s": _stats(tool_wait),     # per-agent tool/build/eval wait time
            "total_gen_tokens": sum(gen), "total_prompt_tokens": sum(pmt),
        },
        "throughput": {
            "wall_s": round(wall_s, 1),
            "aggregate_gen_tps": round(sum(gen) / wall_s, 1) if wall_s else 0,
            "aggregate_prompt_tps": round(sum(pmt) / wall_s, 1) if wall_s else 0,
            "server_gen_tps": round(mon.gen_throughput, 1) if has_mon else None,
            "server_prompt_tps": round(mon.prompt_throughput, 1) if has_mon else None,
        },
        "kv_cache": {
            "peak_perc": round(mon.kv_peak_perc, 1) if has_mon else None,
            "avg_perc": round(mon.kv_avg_perc, 1) if has_mon else None,
        },
        "prefix_cache": {
            "hit_rate": round(mon.prefix_cache_hit_rate, 4) if has_mon else None,
            "queries_tok": mon.prefix_cache_queries if has_mon else None,
            "hits_tok": mon.prefix_cache_hits if has_mon else None,
        },
        "concurrency": {
            "max_running": mon.max_running if has_mon else None,
            "avg_running": round(mon.avg_running, 1) if has_mon else None,
            "max_waiting": mon.max_waiting if has_mon else None,
        },
        "preemptions": mon.preemptions if has_mon else None,
    }
    json.dump(summary, open(path, "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.SWEBENCH_DATA)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--repos", default=",".join(EASY_REPOS),
                    help="comma-separated repo filter (default: easy-to-build repos)")
    ap.add_argument("--instance-ids", default=None, help="explicit comma-separated instance ids")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    ap.add_argument("--output", default="swebench_results.jsonl")
    ap.add_argument("--agent", choices=["bash", "edit"], default="edit",
                    help="agent style: 'edit' = long reasoning + file edits (decode-heavy)")
    ap.add_argument("--model", default=None, help="override served model name")
    args = ap.parse_args()

    if args.agent == "edit":
        import swebench_edit_agent as A
    else:
        import swebench_agent as A
    if args.model:
        A.MODEL = args.model
    run_agent = A.run_agent
    print(f"agent={args.agent}  model={A.MODEL}")

    insts = load_instances(args.data, args.n, args.repos, args.instance_ids, args.start)
    print(f"Running {len(insts)} instances @ {args.workers} workers, max_turns={args.max_turns}")
    for i in insts:
        print(f"  - {i['instance_id']:<30} {i['repo']} {i['version']}")

    monitor = MetricsMonitor(interval=0.5, metrics_url=args.metrics_url)
    monitor.start()
    t0 = time.perf_counter()

    def rec(r):
        return {
            "instance_id": r.instance_id, "resolved": r.resolved, "status": r.status,
            "turns": r.turns, "submitted": r.submitted,
            "gen_tokens": r.total_gen_tokens, "prompt_tokens": r.total_prompt_tokens,
            "decode_s": r.total_decode_s, "ttft_s": r.total_ttft_s,
            "tool_wait_s": r.total_wait_s, "gen_tps": r.gen_tps,
            "build_s": r.build_s, "eval_s": r.eval_s, "error": r.error,
            # per-turn trace: each LLM call's wait (TTFT = proxy-pause+queue+prefill) vs
            # decode time. Under ThunderAgent a paused turn shows a large wait_s.
            "total_decode_s": round(r.total_decode_s, 2),
            "trace": [
                {"turn": i + 1,
                 "wait_s": round(t.ttft_s, 3),      # time before first token: proxy pause + prefill
                 "decode_s": round(t.decode_s, 3),  # generation (running/decoding) time
                 "gen_tokens": t.gen_tokens,
                 "prompt_tokens": t.prompt_tokens,
                 "tool_wait_s": round(t.wait_s, 3),
                 "waited": t.ttft_s > 1.0}           # heuristic: paused/queued before running
                for i, t in enumerate(r.timings)
            ],
        }

    # incremental write: each agent's record is flushed as soon as it finishes,
    # so a mid-run stop still leaves complete data for every completed agent.
    results = []
    out_f = open(args.output, "w")

    # time-limit support: if killed by `timeout` (SIGTERM) or Ctrl-C (SIGINT),
    # write a summary from whatever finished so far, then exit hard. results.jsonl
    # is already flushed per-agent, so completed agents are never lost.
    def _on_signal(signum, frame):
        try:
            out_f.flush()
            wall = time.perf_counter() - t0
            try:
                mon = monitor.stop()
            except Exception:
                mon = None
            sp = args.output.rsplit(".", 1)[0] + "_summary.json"
            write_summary(sp, results, wall, mon, args)
            print(f"\n[time-limit] signal {signum}: wrote partial summary for "
                  f"{len(results)} completed agents -> {sp}", flush=True)
        finally:
            os._exit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_agent, inst, None, False, args.max_turns): inst for inst in insts}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            out_f.write(json.dumps(rec(r)) + "\n"); out_f.flush()
            print(f"  [done] {r.instance_id:<30} {'RESOLVED' if r.resolved else r.status}"
                  f"  turns={r.turns} gen={r.total_gen_tokens} wait={r.total_wait_s:.0f}s"
                  + (f"  err={r.error}" if r.error else ""))
    out_f.close()

    wall_s = time.perf_counter() - t0
    summary = monitor.stop()
    print(f"raw results -> {args.output}")

    # write aggregate stats JSON next to the per-agent results
    summary_path = args.output.rsplit(".", 1)[0] + "_summary.json"
    write_summary(summary_path, results, wall_s, summary, args)
    print(f"summary stats -> {summary_path}")
    report(results, wall_s, summary)


if __name__ == "__main__":
    main()
