"""
Background metrics poller for vllm's Prometheus /metrics endpoint.

Collects per-second snapshots of:
  - KV cache usage %
  - Prompt / generation token counters (for throughput)
  - Number of running/waiting requests
"""
import re
import time
import threading
import csv
from dataclasses import dataclass, field
from typing import Optional
import urllib.request

METRICS_URL = "http://localhost:8000/metrics"


@dataclass
class MetricSnapshot:
    elapsed_s: float
    kv_cache_perc: float          # 0.0 – 1.0
    num_running: int
    num_waiting: int
    prompt_tokens_total: float    # cumulative counter
    gen_tokens_total: float       # cumulative counter
    prefix_queries_total: float = 0.0   # cumulative prefix-cache token queries
    prefix_hits_total: float = 0.0      # cumulative prefix-cache token hits
    preemptions_total: float = 0.0      # cumulative request preemptions (recompute)


@dataclass
class MonitorSummary:
    snapshots: list[MetricSnapshot]
    start_wall: float
    end_wall: float

    # ── KV cache ──────────────────────────────────────────────────────────────
    @property
    def kv_peak_perc(self) -> float:
        vals = [s.kv_cache_perc for s in self.snapshots]
        return max(vals) * 100 if vals else 0.0

    @property
    def kv_avg_perc(self) -> float:
        vals = [s.kv_cache_perc for s in self.snapshots]
        return (sum(vals) / len(vals)) * 100 if vals else 0.0

    # ── Throughput ─────────────────────────────────────────────────────────────
    @property
    def total_prompt_tokens(self) -> int:
        if len(self.snapshots) < 2:
            return 0
        return int(self.snapshots[-1].prompt_tokens_total - self.snapshots[0].prompt_tokens_total)

    @property
    def total_gen_tokens(self) -> int:
        if len(self.snapshots) < 2:
            return 0
        return int(self.snapshots[-1].gen_tokens_total - self.snapshots[0].gen_tokens_total)

    @property
    def wall_time_s(self) -> float:
        return self.end_wall - self.start_wall

    @property
    def prompt_throughput(self) -> float:
        return self.total_prompt_tokens / self.wall_time_s if self.wall_time_s > 0 else 0

    @property
    def gen_throughput(self) -> float:
        return self.total_gen_tokens / self.wall_time_s if self.wall_time_s > 0 else 0

    # ── Prefix-cache hit rate ───────────────────────────────────────────────────
    @property
    def prefix_cache_hit_rate(self) -> float:
        """Windowed hit rate over the run: Δhits / Δqueries (tokens)."""
        if len(self.snapshots) < 2:
            return 0.0
        dq = self.snapshots[-1].prefix_queries_total - self.snapshots[0].prefix_queries_total
        dh = self.snapshots[-1].prefix_hits_total - self.snapshots[0].prefix_hits_total
        return (dh / dq) if dq > 0 else 0.0

    @property
    def prefix_cache_queries(self) -> int:
        if len(self.snapshots) < 2:
            return 0
        return int(self.snapshots[-1].prefix_queries_total - self.snapshots[0].prefix_queries_total)

    @property
    def prefix_cache_hits(self) -> int:
        if len(self.snapshots) < 2:
            return 0
        return int(self.snapshots[-1].prefix_hits_total - self.snapshots[0].prefix_hits_total)

    # ── Concurrency (running requests) ──────────────────────────────────────────
    @property
    def max_running(self) -> int:
        return max((s.num_running for s in self.snapshots), default=0)

    @property
    def avg_running(self) -> float:
        vals = [s.num_running for s in self.snapshots]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def max_waiting(self) -> int:
        return max((s.num_waiting for s in self.snapshots), default=0)

    @property
    def preemptions(self) -> int:
        if len(self.snapshots) < 2:
            return 0
        return int(self.snapshots[-1].preemptions_total - self.snapshots[0].preemptions_total)

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["elapsed_s", "kv_cache_perc", "num_running",
                             "num_waiting", "prompt_tokens_total", "gen_tokens_total"])
            for s in self.snapshots:
                writer.writerow([
                    f"{s.elapsed_s:.2f}",
                    f"{s.kv_cache_perc * 100:.2f}",
                    s.num_running, s.num_waiting,
                    int(s.prompt_tokens_total), int(s.gen_tokens_total),
                ])

    def save_plot(self, path: str) -> bool:
        """Save KV cache + throughput curves as PNG. Returns False if matplotlib unavailable."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False

        times = [s.elapsed_s for s in self.snapshots]
        kv    = [s.kv_cache_perc * 100 for s in self.snapshots]
        run   = [s.num_running for s in self.snapshots]

        # Instantaneous generation throughput (tokens/s between snapshots)
        gen_tps = [0.0]
        for i in range(1, len(self.snapshots)):
            dt    = self.snapshots[i].elapsed_s - self.snapshots[i-1].elapsed_s
            dtok  = self.snapshots[i].gen_tokens_total - self.snapshots[i-1].gen_tokens_total
            gen_tps.append(dtok / dt if dt > 0 else 0)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(times, kv, color="steelblue", linewidth=1.5)
        axes[0].fill_between(times, kv, alpha=0.2, color="steelblue")
        axes[0].set_ylabel("KV Cache (%)")
        axes[0].set_ylim(0, max(max(kv) * 1.2, 5))
        axes[0].set_title("vllm Inference Metrics — Qwen3-4B on HumanEval")

        axes[1].plot(times, gen_tps, color="darkorange", linewidth=1.5)
        axes[1].set_ylabel("Gen tokens/s")
        axes[1].set_ylim(bottom=0)

        axes[2].plot(times, run, color="green", linewidth=1.5, drawstyle="steps-post")
        axes[2].set_ylabel("Running reqs")
        axes[2].set_xlabel("Elapsed time (s)")
        axes[2].set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return True

    def ascii_sparkline(self, width: int = 60) -> str:
        """ASCII sparkline of KV cache usage over time."""
        if not self.snapshots:
            return "(no data)"
        vals = [s.kv_cache_perc for s in self.snapshots]
        vmax = max(vals) or 1e-9
        bars = " ▁▂▃▄▅▆▇█"
        step = max(1, len(vals) // width)
        sampled = vals[::step][:width]
        return "".join(bars[min(int(v / vmax * 8), 8)] for v in sampled)


# ── Prometheus text parser ─────────────────────────────────────────────────────

def _parse_gauge(text: str, name: str) -> Optional[float]:
    m = re.search(rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([\d.e+\-]+)', text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _parse_counter_total(text: str, name: str) -> float:
    """Counters are exposed as <name>_total or just <name> depending on vllm version."""
    for suffix in ("_total", ""):
        v = _parse_gauge(text, name + suffix)
        if v is not None:
            return v
    return 0.0


def _fetch_snapshot(t0: float) -> Optional[MetricSnapshot]:
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=2) as resp:
            text = resp.read().decode()
    except Exception:
        return None

    kv       = _parse_gauge(text, "vllm:kv_cache_usage_perc") or 0.0
    running  = int(_parse_gauge(text, "vllm:num_requests_running") or 0)
    waiting  = int(_parse_gauge(text, "vllm:num_requests_waiting") or 0)
    prompt   = _parse_counter_total(text, "vllm:prompt_tokens")
    gen      = _parse_counter_total(text, "vllm:generation_tokens")
    pq       = _parse_counter_total(text, "vllm:prefix_cache_queries")
    ph       = _parse_counter_total(text, "vllm:prefix_cache_hits")
    pre      = _parse_counter_total(text, "vllm:num_preemptions")

    return MetricSnapshot(
        elapsed_s=time.perf_counter() - t0,
        kv_cache_perc=kv,
        num_running=running,
        num_waiting=waiting,
        prompt_tokens_total=prompt,
        gen_tokens_total=gen,
        prefix_queries_total=pq,
        prefix_hits_total=ph,
        preemptions_total=pre,
    )


# ── Monitor class ──────────────────────────────────────────────────────────────

class MetricsMonitor:
    """
    Usage:
        mon = MetricsMonitor(interval=0.5)
        mon.start()
        ... run inference ...
        summary = mon.stop()
        summary.save_csv("kv_cache.csv")
        summary.save_plot("kv_cache.png")
    """

    def __init__(self, interval: float = 0.5, metrics_url: str = METRICS_URL):
        global METRICS_URL
        METRICS_URL = metrics_url
        self._interval = interval
        self._snapshots: list[MetricSnapshot] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._t0: float = 0.0
        self._start_wall: float = 0.0

    def start(self) -> None:
        self._stop_event.clear()
        self._snapshots = []
        self._t0 = time.perf_counter()
        self._start_wall = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> MonitorSummary:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        return MonitorSummary(
            snapshots=list(self._snapshots),
            start_wall=self._start_wall,
            end_wall=time.time(),
        )

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            snap = _fetch_snapshot(self._t0)
            if snap is not None:
                self._snapshots.append(snap)
            self._stop_event.wait(self._interval)


# ── standalone sampler (replaces kv_sampler.py) ───────────────────────────────
# Usage: python metrics.py <out.csv> [metrics_url] [interval_s]
# Polls vLLM /metrics and logs t_s,kv_perc,running,waiting,prefix_hit_cum,preemptions
if __name__ == "__main__":
    import sys, csv as _csv, time as _time
    out = sys.argv[1] if len(sys.argv) > 1 else "kv_timeseries.csv"
    METRICS_URL = sys.argv[2] if len(sys.argv) > 2 else METRICS_URL
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    t0 = time.perf_counter()
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["t_s", "kv_perc", "running", "waiting", "prefix_hit_cum", "preemptions"])
        while True:
            s = _fetch_snapshot(t0)
            if s is not None:
                hit = (s.prefix_hits_total / s.prefix_queries_total * 100) if s.prefix_queries_total else 0.0
                w.writerow([f"{s.elapsed_s:.2f}", f"{s.kv_cache_perc*100:.2f}", s.num_running,
                            s.num_waiting, f"{hit:.2f}", int(s.preemptions_total)])
                f.flush()
            _time.sleep(interval)
