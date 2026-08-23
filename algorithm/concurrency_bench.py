"""
Concurrency stress test: N concurrent requests, ~fixed prompt length, no prefix
caching. Measures how batching changes aggregate vs per-request decode throughput,
KV usage, TTFT, and queueing.

Each request gets a UNIQUE ~PROMPT_TOK-token prompt (so nothing is shared even if
caching were on), generates MAX_GEN tokens. A metrics monitor samples server-side
running/KV/throughput during the run.
"""
import time
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from metrics import MetricsMonitor
import run_metrics as rm

BASE_URL = "http://localhost:8000/v1"


def build_prompt(idx: int, n_lines: int) -> str:
    # ~15 tokens/line of code-like filler; n_lines tuned to hit the target prompt size.
    head = f"You are reviewing code module #{idx}. Study the following and then write a brief analysis.\n\n"
    lines = [f"var_{idx}_{i} = compute_{i}(arg_{i}) + helper_{i}(state_{i})  # step {i} of module {idx}"
             for i in range(n_lines)]
    tail = f"\n\n# end of module {idx}\nAnalyze the code above and describe what it does, step by step."
    return head + "\n".join(lines) + tail


def one_request(client, model, idx, n_lines, max_gen):
    msgs = [{"role": "user", "content": build_prompt(idx, n_lines)}]
    t0 = time.perf_counter(); ttft = None; ptok = gtok = 0
    stream = client.chat.completions.create(
        model=model, messages=msgs, temperature=0.0, max_tokens=max_gen,
        stream=True, stream_options={"include_usage": True},
    )
    for ch in stream:
        if ch.choices and ch.choices[0].delta.content and ttft is None:
            ttft = time.perf_counter() - t0
        if getattr(ch, "usage", None) is not None:
            ptok = ch.usage.prompt_tokens or 0
            gtok = ch.usage.completion_tokens or 0
    total = time.perf_counter() - t0
    ttft = ttft or total
    return {"ttft": ttft, "total": total, "decode": max(total - ttft, 0.0),
            "ptok": ptok, "gtok": gtok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--prompt-lines", type=int, default=330)   # ~5000 tokens
    ap.add_argument("--max-gen", type=int, default=512)
    args = ap.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    print(f"concurrency={args.n}  prompt~{args.prompt_lines} lines  max_gen={args.max_gen}  model={args.model}")

    mon = MetricsMonitor(interval=0.3)
    mon.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.n) as ex:
        res = list(ex.map(lambda i: one_request(client, args.model, i, args.prompt_lines, args.max_gen),
                          range(args.n)))
    wall = time.perf_counter() - t0
    s = mon.stop()

    gtot = sum(r["gtok"] for r in res)
    ptot = sum(r["ptok"] for r in res)
    ttfts = sorted(r["ttft"] for r in res)
    dec_tps = [r["gtok"] / r["decode"] for r in res if r["decode"] > 0]

    # percentile comes from run_metrics (interpolated, q in 0..100) so this
    # bench reports the same p95 as every other tool in the pipeline.

    print("\n" + "=" * 64)
    print(f"  CONCURRENCY BENCH  (n={args.n}, no prefix cache)")
    print("=" * 64)
    print(f"  wall time:            {wall:.1f}s")
    print(f"  prompt tokens/req:    ~{ptot//max(len(res),1)}")
    print(f"  gen tokens total:     {gtot}   (per req ~{gtot//max(len(res),1)})")
    print(f"\n  -- throughput --")
    print(f"  aggregate decode:     {gtot/wall:.0f} tok/s   (all {args.n} reqs combined)")
    print(f"  per-request decode:   median {statistics.median(dec_tps):.1f} tok/s "
          f"(min {min(dec_tps):.1f}, max {max(dec_tps):.1f})")
    print(f"  single-stream ref:    ~50 tok/s")
    print(f"  batching speedup:     {gtot/wall/50:.1f}x aggregate vs single-stream")
    print(f"\n  -- latency --")
    print(f"  TTFT  p50 {rm.pctl(ttfts,50)*1000:.0f}ms  p95 {rm.pctl(ttfts,95)*1000:.0f}ms  max {ttfts[-1]*1000:.0f}ms")
    print(f"\n  -- server side --")
    print(f"  max running reqs:     {s.max_running}   (avg {s.avg_running:.1f})")
    print(f"  KV peak / avg:        {s.kv_peak_perc:.1f}% / {s.kv_avg_perc:.1f}%")
    print(f"  prefix hit rate:      {s.prefix_cache_hit_rate:.1%}")
    print(f"  server gen thruput:   {s.gen_throughput:.0f} tok/s")
    print("=" * 64)


if __name__ == "__main__":
    main()
