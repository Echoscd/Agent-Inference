# CPU KV Offloading comparison (80-way concurrency, Qwen3-32B, 40960 ctx)

| run | vLLM | offload | dur(s) | KV peak/avg | run max/avg | wait max | preempt | GPU hit | CPU(Ext) hit | gen TP |
|-----|------|---------|--------|-------------|-------------|----------|---------|---------|--------------|--------|
| 06  | 0.11 | no  | 1446 | 100/95% | 80/29 | 54 | 153 | 41% (warm start) | - | 232 |
| 08  | 0.12 | no  | 216  | 100/89% | 80/46 | 67 | 47  | 5% (cold)        | - | 774 |
| 09  | 0.12 | YES | 219  | 100/95% | 80/41 | 50 | 66  | 7% (cold)        | 30% | 770 |

## Solid conclusions
1. Upgrade fixed the crash: vLLM 0.11 + OffloadingConnector dies with AssertionError
   (cpu_gpu.py transfer); vLLM 0.12 + offload runs with 0 crashes.
2. CPU offload works on 0.12: External (CPU-tier) prefix cache hit = 30% in run 09
   -> evicted KV is offloaded to CPU and reloaded instead of recomputed. 06/08 (no
   offload) have no such tier.

## Confounds (do NOT over-read cross-run numbers)
- Duration: 06 ran 24min, 08/09 only ~3.6min (08 cut short, 09 still running).
  Preemption COUNTS not comparable across different durations.
- Warm vs cold prefix cache: 06 server was warm (GPU hit decayed 90->41%);
  08/09 cold-started (GPU hit 5-7%). GPU-hit not comparable.
- Only 08 vs 09 are matched (same 0.12, cold, ~3.6min): offload adds 30% External
  hit but preempt/throughput not clearly better.

## Key insight
At 80-way EXTREME oversubscription, GPU(258k tok) + CPU(512k tok) = 770k pool is
still < demand of 80 growing contexts -> offload just adds a tier, still thrashes;
its "avoid recompute" benefit is offset by transfer overhead + pool still too small.
To isolate offload's NET win, test MODERATE oversubscription (~40-48 way) where GPU
alone oversubscribes but GPU+CPU fits -> offload converts would-be recompute
preemptions into offload+fast-reload.

## vLLM version note
Latest (0.23.0) needs CUDA-13 driver; host driver is 12.8 (570.133.20) -> cannot run.
0.12.0 (torch 2.9.0+cu128) is the newest driver-compatible version with the
offloading-connector fix. Boot needs FLASHINFER_DISABLE_VERSION_CHECK=1.
