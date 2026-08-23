"""Single source of truth for every metric an agent-serving run produces.

Before this module the same quantities were computed in four places with three
different percentile formulas and two independent tape-timeline reconstructions.
Everything now goes through:

    RunArtifacts   one arm of one run: locates + parses tape / kv csv / results.jsonl
      .calls       list[Call]   -- the ONLY tape parser and timeline reconstruction
      .kv          KvSeries     -- the ONLY kv csv parser, owns the warmup rule
      .programs    list[Program]-- results.jsonl (finishers only; .complete says so)

    Window         a time slice (full / warm / steady); owns the windowing rules
      .throughput(calls)  overlap-weighted token rate
      .latency(calls)     percentiles over calls starting inside it

    RunMetrics     RunArtifacts + windows -> one nested dict / one flat row

    pctl / dist    the ONLY percentile implementation

Call timing semantics (client-side, from swebench_edit_agent._stream_call):
  wait_s  = TTFT: ThunderAgent pause + vLLM queue + prefill (not separable)
  decode_s= first token -> stream end
  tool_wait_s = tool execution after the call returned (NOT part of call latency)

Absolute time: tapes written before 2026-08-23 have no timestamps, so a call's
start is reconstructed as the per-program cumsum of wait+decode+tool_wait with
all programs starting at t=0. Newer tapes carry `t_start_s` (seconds since the
agent run began) and it is used verbatim when present -- `RunArtifacts.timeline`
reports which of the two applied.
"""
import csv
import json
import os
from dataclasses import dataclass, field

# ── percentiles ───────────────────────────────────────────────────────────────
# One implementation. Linear interpolation between order statistics, q in 0..100
# (numpy's default 'linear' method). Everything that reports a pXX uses this.


def pctl(vals, q):
    """Interpolated percentile of an UNSORTED iterable. Empty -> 0."""
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def dist(vals, nd=1):
    """Standard distribution block used in every summary."""
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return {"min": 0, "median": 0, "p90": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0}
    return {
        "min": round(xs[0], nd),
        "median": round(pctl(xs, 50), nd),
        "p90": round(pctl(xs, 90), nd),
        "p95": round(pctl(xs, 95), nd),
        "p99": round(pctl(xs, 99), nd),
        "mean": round(sum(xs) / len(xs), nd),
        "max": round(xs[-1], nd),
    }


# ── the three record types ────────────────────────────────────────────────────
@dataclass
class Call:
    """One LLM call, with an absolute position on the run's timeline."""
    iid: str
    turn: int
    start: float          # request sent
    decode_start: float   # first token out  (= start + wait_s)
    end: float            # stream finished  (= decode_start + decode_s)
    wait_s: float
    decode_s: float
    tool_wait_s: float
    gen_tokens: int
    prompt_tokens: int

    @property
    def latency_s(self) -> float:
        """End-to-end LLM call latency. Tool execution is NOT part of it."""
        return self.wait_s + self.decode_s


@dataclass
class Program:
    """One agent program, from results.jsonl (only programs that returned)."""
    iid: str
    resolved: bool
    status: str
    turns: int
    submitted: bool
    gen_tokens: int
    prompt_tokens: int
    decode_s: float
    ttft_s: float
    tool_wait_s: float
    error: str = ""

    @property
    def latency_s(self) -> float:
        """Program end-to-end wall time. Tool execution IS part of it."""
        return self.ttft_s + self.decode_s + self.tool_wait_s


@dataclass
class KvSeries:
    """The 0.5s vLLM /metrics sampling (metrics.py -> kv_<arm>.csv)."""
    t: list
    kv_perc: list
    running: list
    waiting: list
    prefix_hit_cum: list
    preemptions: list

    WARM_FRAC = 0.9   # warmup ends when KV first reaches this fraction of its run peak

    @property
    def wall_s(self):
        return self.t[-1] if self.t else 0.0

    @property
    def peak_kv(self):
        return max(self.kv_perc) if self.kv_perc else 0.0

    def warm_edges(self):
        """(t_warm, t_drain): first / last sample at >= WARM_FRAC * peak KV.

        Single-sample crossing (no dwell requirement) and the peak is taken over
        the WHOLE run, so this is an offline, non-causal rule. Sensitivity across
        WARM_FRAC 0.7..0.95 is ~1% on throughput, ~2% on p95.
        """
        if not self.t:
            return 0.0, 0.0
        thr = self.WARM_FRAC * self.peak_kv
        hits = [t for t, k in zip(self.t, self.kv_perc) if k >= thr]
        return (hits[0], hits[-1]) if hits else (0.0, self.wall_s)

    def stats(self):
        if not self.t:
            return {}
        return {
            "kv_peak_perc": round(self.peak_kv, 1),
            "kv_avg_perc": round(sum(self.kv_perc) / len(self.kv_perc), 1),
            "max_running": int(max(self.running)),
            "avg_running": round(sum(self.running) / len(self.running), 1),
            "max_waiting": int(max(self.waiting)),
            "preemptions": int(self.preemptions[-1] - self.preemptions[0]),
            "prefix_hit_rate": round(self.prefix_hit_cum[-1] / 100.0, 4),
        }


# ── window ────────────────────────────────────────────────────────────────────
@dataclass
class Window:
    """A slice of the run timeline. Owns how metrics are attributed to a slice."""
    name: str
    t0: float
    t1: float

    @property
    def duration(self):
        return max(self.t1 - self.t0, 1e-9)

    def started_in(self, calls):
        """Calls whose REQUEST falls inside. Latency uses this: a call's queue
        wait reflects the congestion at the moment it entered the system."""
        return [c for c in calls if self.t0 <= c.start <= self.t1]

    def gen_tokens(self, calls):
        """Overlap-weighted: a call's decode interval is credited in proportion
        to how much of it falls inside, so edge-straddling calls are neither
        double counted nor dropped."""
        tot = 0.0
        for c in calls:
            span = c.end - c.decode_start
            if span <= 0:
                if self.t0 <= c.decode_start <= self.t1:
                    tot += c.gen_tokens
                continue
            ov = max(0.0, min(c.end, self.t1) - max(c.decode_start, self.t0))
            if ov > 0:
                tot += c.gen_tokens * ov / span
        return tot

    def metrics(self, calls):
        started = self.started_in(calls)
        gen = self.gen_tokens(calls)
        prompt = sum(c.prompt_tokens for c in started)
        lat = [c.latency_s for c in started]
        return {
            "window_s": round(self.duration, 1),
            "t0_s": round(self.t0, 1), "t1_s": round(self.t1, 1),
            "calls_started": len(started),
            "gen_tokens_in_window": round(gen, 1),
            "gen_tps": round(gen / self.duration, 1),
            "prompt_tps": round(prompt / self.duration, 1),
            "calls_per_s": round(len(started) / self.duration, 3),
            "latency_s": {k: round(pctl(lat, q), 1) for k, q in
                          (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "latency_mean_s": round(sum(lat) / len(lat), 1) if lat else None,
            "latency_max_s": round(max(lat), 1) if lat else None,
            "wait_s": {k: round(pctl([c.wait_s for c in started], q), 1) for k, q in
                       (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "decode_s": {k: round(pctl([c.decode_s for c in started], q), 1) for k, q in
                         (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
        }


# ── artifacts ─────────────────────────────────────────────────────────────────
class RunArtifacts:
    """One arm (A or B) of one run directory. Parses each file exactly once."""

    def __init__(self, run_dir, arm):
        self.run_dir = run_dir.rstrip("/")
        self.arm = arm
        self.run = os.path.basename(self.run_dir)
        self._calls = None
        self._kv = None
        self._programs = None
        self.timeline = None   # "recorded" | "reconstructed", set when calls are parsed

    def path(self, kind):
        return {
            "tape": f"{self.run_dir}/tape_{self.arm}.jsonl",
            "kv": f"{self.run_dir}/kv_{self.arm}.csv",
            "results": f"{self.run_dir}/results_{self.arm}.jsonl",
            "summary": f"{self.run_dir}/results_{self.arm}_summary.json",
            "trace": f"{self.run_dir}/decision_trace_{self.arm}.jsonl",
        }[kind]

    def has(self, *kinds):
        return all(os.path.exists(self.path(k)) for k in kinds)

    # -- tape: the only parser + the only timeline reconstruction ---------------
    @property
    def calls(self):
        if self._calls is not None:
            return self._calls
        per_prog = {}
        with open(self.path("tape")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                per_prog.setdefault(d["iid"], []).append(d)
        recorded = any("t_start_s" in d for ds in per_prog.values() for d in ds)
        self.timeline = "recorded" if recorded else "reconstructed"
        calls = []
        for iid, turns in per_prog.items():
            turns.sort(key=lambda d: d.get("turn", 0))
            t = 0.0
            for d in turns:
                wait = float(d.get("wait_s") or 0.0)
                dec = float(d.get("decode_s") or 0.0)
                tool = float(d.get("tool_wait_s") or 0.0)
                start = float(d["t_start_s"]) if recorded and d.get("t_start_s") is not None else t
                calls.append(Call(iid=iid, turn=d.get("turn"), start=start,
                                  decode_start=start + wait, end=start + wait + dec,
                                  wait_s=wait, decode_s=dec, tool_wait_s=tool,
                                  gen_tokens=int(d.get("gen_tokens") or 0),
                                  prompt_tokens=int(d.get("prompt_tokens") or 0)))
                t = start + wait + dec + tool
        calls.sort(key=lambda c: c.start)
        self._calls = calls
        return calls

    @property
    def kv(self):
        if self._kv is not None:
            return self._kv
        cols = {k: [] for k in ("t_s", "kv_perc", "running", "waiting",
                                "prefix_hit_cum", "preemptions")}
        with open(self.path("kv")) as f:
            for r in csv.DictReader(f):
                for k in cols:
                    cols[k].append(float(r.get(k) or 0))
        self._kv = KvSeries(cols["t_s"], cols["kv_perc"], cols["running"],
                            cols["waiting"], cols["prefix_hit_cum"], cols["preemptions"])
        return self._kv

    @property
    def programs(self):
        if self._programs is not None:
            return self._programs
        out = []
        p = self.path("results")
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(Program(
                    iid=d.get("instance_id", ""), resolved=bool(d.get("resolved")),
                    status=d.get("status", ""), turns=int(d.get("turns") or 0),
                    submitted=bool(d.get("submitted")),
                    gen_tokens=int(d.get("gen_tokens") or 0),
                    prompt_tokens=int(d.get("prompt_tokens") or 0),
                    decode_s=float(d.get("decode_s") or 0.0),
                    ttft_s=float(d.get("ttft_s") or 0.0),
                    tool_wait_s=float(d.get("tool_wait_s") or 0.0),
                    error=d.get("error") or ""))
        self._programs = out
        return out

    @property
    def policy(self):
        p = self.path("trace")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            first = f.readline().strip()
        try:
            return json.loads(first).get("policy")
        except json.JSONDecodeError:
            return None

    @property
    def config(self):
        p = self.path("summary")
        if not os.path.exists(p):
            return {}
        try:
            return json.load(open(p)).get("config", {})
        except (json.JSONDecodeError, OSError):
            return {}

    # -- windows ---------------------------------------------------------------
    @property
    def wall_s(self):
        """Reconstructed/recorded end of the last call (the calls' own timeline)."""
        return max((c.end for c in self.calls), default=0.0)

    def windows(self):
        """The three standard windows. full = whole timeline, warm = warmup
        dropped, steady = warmup and drain tail dropped."""
        t_warm, t_drain = self.kv.warm_edges()
        return [Window("full", 0.0, self.wall_s),
                Window("warm", t_warm, self.wall_s),
                Window("steady", t_warm, t_drain)]


# ── assembled metrics ─────────────────────────────────────────────────────────
WINDOW_COLS = ("window_s", "gen_tps", "prompt_tps", "calls_per_s")
PCT_COLS = ("p50", "p90", "p95", "p99")


class RunMetrics:
    """RunArtifacts + windows -> one nested dict (json) or one flat row (csv)."""

    def __init__(self, artifacts):
        self.a = artifacts

    def to_dict(self):
        a = self.a
        calls = a.calls
        t_warm, t_drain = a.kv.warm_edges()
        cfg = a.config
        out = {
            "run": a.run, "arm": a.arm, "policy": a.policy,
            "model": cfg.get("model"), "workers": cfg.get("workers"),
            "programs_in_tape": len({c.iid for c in calls}),
            "programs_completed": cfg.get("completed"),
            "calls_total": len(calls),
            "timeline": a.timeline,
            "wall_s_sampler": round(a.kv.wall_s, 1),
            "wall_s_calls": round(a.wall_s, 1),
            "max_kv_perc": a.kv.peak_kv,
            "t_warm_s": round(t_warm, 1), "t_drain_s": round(t_drain, 1),
        }
        out.update(a.kv.stats())
        for w in a.windows():
            out[w.name] = w.metrics(calls)
        return out

    @staticmethod
    def columns():
        head = ["run", "arm", "policy", "model", "programs_in_tape", "calls_total",
                "timeline", "t_warm_s", "t_drain_s", "wall_s_sampler"]
        for w in ("full", "warm", "steady"):
            head += [f"{w}_{c}" for c in WINDOW_COLS]
            head += [f"{w}_lat_{p}" for p in PCT_COLS]
            head += [f"{w}_wait_{p}" for p in ("p90", "p95")]
        return head

    def to_row(self):
        d = self.to_dict()
        row = [d[c] for c in ("run", "arm", "policy", "model", "programs_in_tape",
                              "calls_total", "timeline", "t_warm_s", "t_drain_s",
                              "wall_s_sampler")]
        for w in ("full", "warm", "steady"):
            m = d[w]
            row += [m[c] for c in WINDOW_COLS]
            row += [m["latency_s"][p] for p in PCT_COLS]
            row += [m["wait_s"][p] for p in ("p90", "p95")]
        return row


def calls_from_tape(tape_path):
    """Parse any tape file into Calls, for callers that have a path rather than a
    run dir (plots.py). Same parser and timeline rule as RunArtifacts.calls."""
    d = os.path.dirname(tape_path.rstrip("/")) or "."
    base = os.path.basename(tape_path)
    arm = base[len("tape_"):-len(".jsonl")] if base.startswith("tape_") else "A"
    a = RunArtifacts(d, arm)
    if os.path.abspath(a.path("tape")) != os.path.abspath(tape_path):
        a.path = lambda kind, _p=tape_path: _p if kind == "tape" else ""
    return a.calls


def collect(run_dirs, arms=("A", "B")):
    """-> list[RunMetrics] for every arm that has both a tape and a kv csv."""
    out = []
    for rd in run_dirs:
        for arm in arms:
            a = RunArtifacts(rd, arm)
            if a.has("tape", "kv") and a.calls:
                out.append(RunMetrics(a))
    return out


def write(rows, out_dir, stem="steady_metrics"):
    """Write the json + csv pair that every run folder carries."""
    os.makedirs(out_dir, exist_ok=True)
    dicts = [r.to_dict() for r in rows]
    json.dump(dicts, open(f"{out_dir}/{stem}.json", "w"), indent=2)
    with open(f"{out_dir}/{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RunMetrics.columns())
        for r in rows:
            w.writerow(r.to_row())
    return dicts
