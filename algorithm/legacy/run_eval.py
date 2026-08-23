"""
Batch evaluation of HumanEval using the ReAct pipeline.

Reports:
  - pass@1 accuracy
  - Token throughput (prompt / decode tokens/s)
  - KV cache usage curve  →  kv_cache.csv + kv_cache.png
  - Per-task timing table for a sampled subset

Usage:
    python run_eval.py [--n 20] [--output results.jsonl]
                       [--timing-samples 10] [--workers 4]
"""
import json
import argparse
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from react_pipeline import solve, SolveResult
from metrics_monitor import MetricsMonitor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_separator(char="─", width=72):
    print(char * width)

def _fmt(n, unit="") -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M{unit}"
    if n >= 1_000:
        return f"{n/1e3:.1f}k{unit}"
    return f"{n}{unit}"

def _print_report(
    results: list[SolveResult],
    timed_results: list[SolveResult],
    monitor_summary,
    wall_time: float,
):
    passed = sum(1 for r in results if r.success)
    total  = len(results)

    print()
    _print_separator("═")
    print("  EVALUATION REPORT")
    _print_separator("═")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print(f"\n{'Accuracy':─<40}")
    print(f"  pass@1 = {passed}/{total} = {passed/total:.1%}")

    iter_counts = [r.iterations for r in results]
    avg_iters = sum(iter_counts) / len(iter_counts) if iter_counts else 0
    print(f"  Avg iterations per problem: {avg_iters:.2f}")
    print(f"  Max iterations hit:         {sum(1 for r in results if not r.success and r.iterations == 5)}")

    # ── Throughput (from metrics monitor) ─────────────────────────────────────
    print(f"\n{'Throughput (server-side)':─<40}")
    print(f"  Wall time:           {wall_time:.1f}s")
    if monitor_summary and monitor_summary.snapshots:
        print(f"  Prompt tokens:       {_fmt(monitor_summary.total_prompt_tokens)} "
              f"  ({monitor_summary.prompt_throughput:.1f} tok/s)")
        print(f"  Gen tokens:          {_fmt(monitor_summary.total_gen_tokens)} "
              f"  ({monitor_summary.gen_throughput:.1f} tok/s)")
    else:
        print("  (metrics endpoint not available — start vllm before running)")

    # ── KV cache ──────────────────────────────────────────────────────────────
    print(f"\n{'KV Cache':─<40}")
    if monitor_summary and monitor_summary.snapshots:
        print(f"  Peak:    {monitor_summary.kv_peak_perc:.1f}%")
        print(f"  Average: {monitor_summary.kv_avg_perc:.1f}%")
        print(f"  Samples: {len(monitor_summary.snapshots)}")
        sparkline = monitor_summary.ascii_sparkline(width=60)
        print(f"  Curve:   {sparkline}")
        monitor_summary.save_csv("kv_cache.csv")
        print(f"  Saved:   kv_cache.csv")
        if monitor_summary.save_plot("kv_cache.png"):
            print(f"  Plot:    kv_cache.png")
    else:
        print("  (no data)")

    # ── Per-task timing table ─────────────────────────────────────────────────
    if timed_results:
        print(f"\n{'Sampled Task Timing (client-side, single-request)':─<40}")
        print(f"  Note: TTFT includes scheduling + prefill; decode = remaining time.")
        print()
        hdr = (
            f"  {'Task':<16} {'Iters':>5} {'Status':>6} "
            f"{'Prompt':>7} {'Gen':>5} "
            f"{'TTFT':>8} {'Decode':>8} {'Speed':>9}"
        )
        print(hdr)
        _print_separator(width=72)
        for r in sorted(timed_results, key=lambda x: x.task_id):
            status = "PASS" if r.success else "FAIL"
            has_t  = any(s.timing for s in r.steps)
            if has_t:
                ttft_ms  = r.total_ttft_s * 1000
                dec_s    = r.total_decode_s
                ptok     = r.total_prompt_tokens
                gtok     = r.total_gen_tokens
                speed    = f"{gtok/dec_s:.0f}t/s" if dec_s > 0 else "—"
                print(
                    f"  {r.task_id:<16} {r.iterations:>5} {status:>6} "
                    f"{_fmt(ptok):>7} {_fmt(gtok):>5} "
                    f"{ttft_ms:>6.0f}ms {dec_s:>6.2f}s {speed:>9}"
                )
            else:
                print(f"  {r.task_id:<16} {r.iterations:>5} {status:>6}   (no timing)")

    _print_separator("═")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__)))), "data", "humaneval.jsonl"))
    parser.add_argument("--n",              type=int, default=None,
                        help="Number of problems to evaluate (default: all 164)")
    parser.add_argument("--start",          type=int, default=0)
    parser.add_argument("--output",         default="results.jsonl")
    parser.add_argument("--workers",        type=int, default=4,
                        help="Parallel workers for batch eval")
    parser.add_argument("--timing-samples", type=int, default=10,
                        help="Number of random problems to run with streaming timing (sequential)")
    parser.add_argument("--metrics-url",    default="http://localhost:8000/metrics")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    # Load tasks
    tasks = []
    with open(args.data) as f:
        for line in f:
            tasks.append(json.loads(line))
    tasks = tasks[args.start:]
    if args.n is not None:
        tasks = tasks[:args.n]
    task_map = {t["task_id"]: t for t in tasks}

    # Select timing-sample tasks (stratified: spread across task indices)
    rng = random.Random(args.seed)
    sample_n = min(args.timing_samples, len(tasks))
    timing_task_ids = set(
        t["task_id"] for t in rng.sample(tasks, sample_n)
    )
    print(f"Evaluating {len(tasks)} problems  |  "
          f"{len(timing_task_ids)} timing samples  |  "
          f"{args.workers} workers")
    print(f"Timing samples: {sorted(timing_task_ids)}")
    print()

    # ── Start metrics monitor ─────────────────────────────────────────────────
    monitor = MetricsMonitor(interval=0.5, metrics_url=args.metrics_url)
    monitor.start()
    t_wall_start = time.perf_counter()

    # ── Phase 1: batch eval (no streaming to maximise throughput) ─────────────
    batch_tasks   = [t for t in tasks if t["task_id"] not in timing_task_ids]
    batch_results: list[SolveResult] = []

    print(f"[Phase 1] Batch eval: {len(batch_tasks)} tasks @ {args.workers} workers ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve, t, False, False): t for t in batch_tasks}
        for future in as_completed(futures):
            r = future.result()
            batch_results.append(r)
            status = "✓" if r.success else "✗"
            done = len(batch_results)
            passed_so_far = sum(1 for x in batch_results if x.success)
            print(f"  {status} {r.task_id:20s} ({r.iterations} iter) "
                  f"| running pass@1: {passed_so_far}/{done}")

    # ── Phase 2: timed samples (streaming, sequential for clean timing) ────────
    timed_results: list[SolveResult] = []
    print(f"\n[Phase 2] Timing samples: {sample_n} tasks (sequential, streaming) ...")
    for tid in sorted(timing_task_ids):
        t = task_map[tid]
        r = solve(t, verbose=False, timed=True)
        timed_results.append(r)
        status = "✓" if r.success else "✗"
        ttft_ms = r.total_ttft_s * 1000
        dec_s   = r.total_decode_s
        print(f"  {status} {tid:20s} TTFT={ttft_ms:.0f}ms  decode={dec_s:.2f}s  "
              f"iters={r.iterations}")

    # ── Stop monitor ──────────────────────────────────────────────────────────
    wall_time = time.perf_counter() - t_wall_start
    monitor_summary = monitor.stop()

    # ── Save raw results ──────────────────────────────────────────────────────
    all_results = batch_results + timed_results
    all_results.sort(key=lambda r: r.task_id)
    with open(args.output, "w") as f:
        for r in all_results:
            record = {
                "task_id":    r.task_id,
                "success":    r.success,
                "iterations": r.iterations,
                "steps": [
                    {
                        "iteration": s.iteration,
                        "passed":    s.passed,
                        "total":     s.total,
                        "error":     s.error,
                        **({"timing": {
                            "ttft_s":        s.timing.ttft_s,
                            "decode_s":      s.timing.decode_s,
                            "total_s":       s.timing.total_s,
                            "prompt_tokens": s.timing.prompt_tokens,
                            "gen_tokens":    s.timing.gen_tokens,
                        }} if s.timing else {}),
                    }
                    for s in r.steps
                ],
            }
            f.write(json.dumps(record) + "\n")
    print(f"\nRaw results → {args.output}")

    # ── Final report ──────────────────────────────────────────────────────────
    _print_report(all_results, timed_results, monitor_summary, wall_time)


if __name__ == "__main__":
    main()
