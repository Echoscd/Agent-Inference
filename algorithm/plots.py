"""Unified plotting CLI for the SWE-bench / ThunderAgent experiments.

Subcommands:
  pdt        <results.jsonl> <out.png> [title]    1x3 prefill / decode / program-turn dist
  saturation <kv_csv> <out.png> [title]           KV% + running + waiting + cumulative preemptions vs time
  kvlog      <vllm_log> <out.png> [title]          KV% (+ running) vs time, parsed from a vLLM server log
  compare    <out.png> <labelA=kvcsvA> <labelB=kvcsvB> ...   overlay KV/running/preempt of several runs
  concurrency <out.png> <labelA=kvcsvA> [labelB=kvcsvB] ...   same-moment concurrency (running on GPU) vs time
  conckv     <out.png> <labelA=kvcsvA> [labelB=kvcsvB] ...   concurrency (top) + KV utilization (bottom), gantt companion
  gantt      <results.jsonl|tape.jsonl> <out.png> [n]   per-program waiting/decoding timeline (n=0 -> ALL programs)
  lambda     <out.png> <labelA=ta.log> [labelB=ta.log] ...   dual_descent price lambda + desired/free driver over epochs

results.jsonl records need a per-turn `trace` ({prompt_tokens, gen_tokens, wait_s, decode_s, tool_wait_s}).
kv_csv is produced by metrics.py (t_s,kv_perc,running,waiting,prefix_hit_cum,preemptions).
"""
import sys, os, csv, re, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics as rm


def _load_jsonl(p):
    rows = []
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    return rows


def _load_programs(p):
    """Return program-level rows [{instance_id, turns, trace:[...]}], accepting EITHER
    results.jsonl (already program-level, only finished agents) OR an A/B tape.jsonl
    (one record per LLM call, keyed by iid — covers ALL programs incl. unfinished).
    Using the tape lets pdt/gantt cover the full population, not just finishers."""
    recs = _load_jsonl(p)
    if not recs:
        return []
    if "trace" in recs[0]:
        return recs                                   # results.jsonl
    # tape -> per-program trace, via run_metrics.RunArtifacts so the timeline
    # reconstruction (and the wait/decode/tool ordering the Gantt draws) has one
    # definition shared with warmup_metrics.
    by = {}
    for c in rm.calls_from_tape(p):
        by.setdefault(c.iid, []).append(c)
    progs = []
    for iid, calls in by.items():
        calls.sort(key=lambda c: c.start)
        trace = [{"turn": c.turn if c.turn is not None else i + 1,
                  "wait_s": c.wait_s, "decode_s": c.decode_s,
                  "tool_wait_s": c.tool_wait_s,
                  "gen_tokens": c.gen_tokens, "prompt_tokens": c.prompt_tokens}
                 for i, c in enumerate(calls)]
        progs.append({"instance_id": iid, "turns": len(calls), "trace": trace})
    return progs


def _load_kv(p):
    t, kv, run, wait, hit, pre = [], [], [], [], [], []
    for r in csv.DictReader(open(p)):
        t.append(float(r["t_s"])); kv.append(float(r["kv_perc"]))
        run.append(int(r["running"])); wait.append(int(r["waiting"]))
        hit.append(float(r.get("prefix_hit_cum", 0) or 0)); pre.append(int(r.get("preemptions", 0) or 0))
    return t, kv, run, wait, hit, pre


# ── pdt: 1x3 prefill / decode / turns ─────────────────────────────────────────
def cmd_pdt(args):
    jl, out = args[0], args[1]
    title = args[2] if len(args) > 2 else jl
    rows = _load_programs(jl)
    prefill = [t["prompt_tokens"] for r in rows for t in r.get("trace", [])]
    decode  = [t["gen_tokens"]    for r in rows for t in r.get("trace", [])]
    turns   = [r["turns"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].hist(prefill, bins=40, color="steelblue", edgecolor="white")
    ax[0].set(xlabel="prefill length (prompt tokens / call)", ylabel="request number", title="Prefill length distribution")
    ax[1].hist(decode, bins=40, color="darkorange", edgecolor="white")
    ax[1].set(xlabel="decode length (generated tokens / call)", ylabel="request number", title="Decode length distribution")
    ax[2].hist(turns, bins=range(1, (max(turns) if turns else 1) + 2), color="seagreen", edgecolor="white", align="left")
    ax[2].set(xlabel="program turns (call number)", ylabel="program number", title="Program turn distribution")
    fig.suptitle(f"{title}  ({len(rows)} programs, {len(decode)} calls)")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"saved {out}: {len(rows)} programs, {len(decode)} calls")


# ── saturation: KV / running / waiting / preemptions vs time ──────────────────
def cmd_saturation(args):
    csv_path, out = args[0], args[1]
    title = args[2] if len(args) > 2 else "KV saturation dynamics"
    t, kv, run, wait, hit, pre = _load_kv(csv_path)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].fill_between(t, kv, alpha=0.2, color="steelblue"); ax[0].plot(t, kv, color="steelblue", lw=1.6, label="KV cache usage (%)")
    ax[0].axhline(100, ls="--", color="red", lw=0.8, alpha=0.6); ax[0].set_ylabel("KV cache usage (%)"); ax[0].set_ylim(0, 105)
    a0b = ax[0].twinx(); a0b.plot(t, run, color="green", lw=1.3, label="running"); a0b.plot(t, wait, color="darkorange", lw=1.3, label="waiting")
    a0b.set_ylabel("requests"); a0b.set_ylim(bottom=0); ax[0].set_title("KV saturation -> queueing")
    l1, la1 = ax[0].get_legend_handles_labels(); l2, la2 = a0b.get_legend_handles_labels(); ax[0].legend(l1 + l2, la1 + la2, fontsize=8, loc="center right")
    ax[1].plot(t, pre, color="purple", lw=1.6, label="cumulative preemptions"); ax[1].set_ylabel("preemptions"); ax[1].set_ylim(bottom=0)
    a1b = ax[1].twinx(); a1b.plot(t, hit, color="brown", lw=1.4, label="prefix-cache hit (%)"); a1b.set_ylabel("prefix hit (%)"); a1b.set_ylim(0, 105)
    ax[1].set_xlabel("elapsed time (s)"); ax[1].set_title("preemptions + prefix-cache hit")
    l1, la1 = ax[1].get_legend_handles_labels(); l2, la2 = a1b.get_legend_handles_labels(); ax[1].legend(l1 + l2, la1 + la2, fontsize=8, loc="center right")
    fig.suptitle(f"{title}\npeak KV {max(kv):.0f}%  peak waiting {max(wait)}  preemptions {max(pre)}")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"saved {out}: {len(t)} samples, span {t[-1]:.0f}s, peak KV {max(kv):.1f}%, preemptions {max(pre)}")


# ── kvlog: KV vs time parsed from a vLLM server log ───────────────────────────
def cmd_kvlog(args):
    log, out = args[0], args[1]
    title = args[2] if len(args) > 2 else "KV cache over time"
    pat = re.compile(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}).*?Running: (\d+) reqs.*?GPU KV cache usage: ([\d.]+)%")
    ts, run, kv = [], [], []
    for line in open(log, errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        mo, d, H, M, S = (int(x) for x in m.group(1, 2, 3, 4, 5))
        ts.append(((((mo * 31 + d) * 24 + H) * 60 + M) * 60) + S)
        run.append(int(m.group(6))); kv.append(float(m.group(7)))
    if not ts:
        print("no KV lines parsed"); return
    # longest contiguous run (gap >=120s splits)
    segs, cur = [], [0]
    for i in range(1, len(ts)):
        if ts[i] - ts[i-1] >= 120:
            segs.append(cur); cur = []
        cur.append(i)
    segs.append(cur); seg = max(segs, key=len); t0 = ts[seg[0]]
    T = [ts[i]-t0 for i in seg]; KV = [kv[i] for i in seg]; RUN = [run[i] for i in seg]
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.fill_between(T, KV, alpha=0.25, color="steelblue"); ax1.plot(T, KV, color="steelblue", lw=1.8)
    ax1.axhline(100, ls="--", color="red", lw=0.8, alpha=0.6); ax1.set(xlabel="elapsed time (s)", ylabel="GPU KV cache usage (%)", ylim=(0, 105))
    ax2 = ax1.twinx(); ax2.plot(T, RUN, color="darkorange", lw=1.4, drawstyle="steps-post"); ax2.set_ylabel("running requests"); ax2.set_ylim(bottom=0)
    ax1.set_title(f"{title}\npeak KV {max(KV):.0f}%  peak running {max(RUN)}  {T[-1]:.0f}s")
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}: {len(seg)} points, span {T[-1]:.0f}s, peak KV {max(KV):.1f}%")


# ── compare: overlay KV/running/preempt of several runs ───────────────────────
def cmd_compare(args):
    out = args[0]
    runs = [a.split("=", 1) for a in args[1:]]   # label=kvcsv
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    for i, (label, path) in enumerate(runs):
        t, kv, run, wait, hit, pre = _load_kv(path); c = colors[i % len(colors)]
        ax[0].plot(t, kv, color=c, lw=1.3, label=label)
        ax[1].plot(t, run, color=c, lw=1.3, label=label)
        ax[2].plot(t, pre, color=c, lw=1.5, label=label)
    ax[0].set(ylabel="KV cache usage (%)", ylim=(0, 105), title="KV cache usage"); ax[0].legend(fontsize=9)
    ax[1].set(ylabel="running reqs", title="active running"); ax[1].legend(fontsize=9)
    ax[2].set(ylabel="cumulative preemptions", xlabel="elapsed (s)", title="preemptions"); ax[2].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"saved {out}: {len(runs)} runs")


# ── gantt: per-program waiting/decoding timeline ──────────────────────────────
def cmd_gantt(args):
    jl, out = args[0], args[1]
    # N = how many programs to show; pass 0 or a value >= #programs to draw ALL.
    N = int(args[2]) if len(args) > 2 else 20
    rows = [r for r in _load_programs(jl) if r.get("trace")]
    rows.sort(key=lambda r: sum(t["wait_s"] + t["decode_s"] + t["tool_wait_s"] for t in r["trace"]))
    if N <= 0 or len(rows) <= N:
        sel = rows                                                   # draw ALL programs
    else:
        idx = [int(i * (len(rows) - 1) / (N - 1)) for i in range(N)]  # evenly subsample N
        sel = [rows[i] for i in idx]
    M = len(sel)
    # figure height scales with the ACTUAL number of programs drawn (taller per row
    # so 80 bars stay readable), not the requested N.
    row_h = 0.46
    fig, ax = plt.subplots(figsize=(18, max(5, M * row_h)))
    span = max((sum(t["wait_s"] + t["decode_s"] + t["tool_wait_s"] for t in r["trace"]) for r in sel), default=1)
    for row, r in enumerate(sel):
        t = 0.0
        for tn in r["trace"]:
            ax.broken_barh([(t, tn["wait_s"])], (row-0.42, 0.84), facecolors="lightgray"); t += tn["wait_s"]
            ax.broken_barh([(t, tn["decode_s"])], (row-0.42, 0.84), facecolors="steelblue"); t += tn["decode_s"]
            ax.broken_barh([(t, max(tn["tool_wait_s"], 0))], (row-0.42, 0.84), facecolors="seagreen"); t += tn["tool_wait_s"]
        ax.text(-span*0.015, row, str(row+1), ha="right", va="center", fontsize=8, fontweight="bold")
        ax.text(t+span*0.008, row, f"{r['turns']}t", ha="left", va="center", fontsize=6, color="gray")
    ax.legend(handles=[Patch(color="lightgray", label="waiting (pause + queue)"),
                       Patch(color="steelblue", label="decoding (running)"),
                       Patch(color="seagreen", label="tool")], loc="lower right", fontsize=9)
    ax.set_yticks([]); ax.set_ylim(-0.8, M-0.2); ax.set_xlim(left=-span*0.04)
    ax.set_xlabel("wall-clock time (s, all programs start at 0)")
    ax.set_title(f"{M} programs progress over time (waiting vs decoding); number = program index i")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"saved {out}: {M} programs")


# ── concurrency: same-moment concurrency (running on GPU) over time ───────────
def cmd_concurrency(args):
    import statistics as _st
    out = args[0]
    runs = [a.split("=", 1) for a in args[1:]]   # label=kvcsv (one or more)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for i, (label, path) in enumerate(runs):
        t, kv, run, wait, hit, pre = _load_kv(path)
        if not run:
            continue
        c = colors[i % len(colors)]
        ax.plot(t, run, color=c, lw=1.4, label=f"{label}  (mean {_st.mean(run):.0f}, median {_st.median(run):.0f}, max {max(run)})")
        ax.axhline(_st.mean(run), color=c, ls=":", lw=0.8, alpha=0.5)
    ax.set(xlabel="elapsed time (s)", ylabel="concurrency = running requests on GPU",
           title="Same-moment concurrency over time")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"saved {out}: {len(runs)} run(s)")


# ── conckv: concurrency + KV utilization over time in one paired figure ───────
def cmd_conckv(args):
    """Companion to `gantt`: same-moment concurrency (top) + KV utilization (bottom)
    over elapsed time, for one or more runs overlaid. Explains gantt gaps = the grey
    waiting/TTFT (held + queued) that low concurrency / reserved-KV headroom produces."""
    import statistics as _st
    out = args[0]
    runs = [a.split("=", 1) for a in args[1:]]   # label=kvcsv (one or more)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for i, (label, path) in enumerate(runs):
        t, kv, run, wait, hit, pre = _load_kv(path)
        if not run:
            continue
        c = colors[i % len(colors)]
        ax0.plot(t, run, color=c, lw=1.3,
                 label=f"{label}  (mean {_st.mean(run):.1f}, median {_st.median(run):.0f}, max {max(run)})")
        ax0.axhline(_st.mean(run), color=c, ls=":", lw=0.9, alpha=0.6)
        ax1.plot(t, kv, color=c, lw=1.3,
                 label=f"{label}  (mean {_st.mean(kv):.1f}%, median {_st.median(kv):.0f}%, max {max(kv):.0f}%)")
        ax1.axhline(_st.mean(kv), color=c, ls=":", lw=0.9, alpha=0.6)
    ax0.set(ylabel="concurrency = running reqs on GPU", title="Same-moment concurrency over time")
    ax0.set_ylim(bottom=0); ax0.grid(alpha=0.3); ax0.legend(fontsize=9)
    ax1.axhline(100, color="k", ls="--", lw=0.8, alpha=0.5)
    ax1.set(xlabel="elapsed time (s)", ylabel="KV cache utilization %", title="KV cache utilization over time")
    ax1.set_ylim(0, 105); ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"saved {out}: {len(runs)} run(s)")


# ── lambda: dual_descent KV price over time, parsed from a ThunderAgent ta_*.log ──
_LAM_RE = re.compile(
    r"dual_descent: epoch=(\d+) lambda=([\d.eE+-]+) desired=([\d.]+) free=(\d+) waiting=(\d+)"
)


def _load_lambda(path):
    """Parse dual_descent epoch lines from a ta log -> (epoch, lam, desired, free, waiting) lists."""
    ep, lam, des, free, wait = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = _LAM_RE.search(line)
            if not m:
                continue
            ep.append(int(m.group(1))); lam.append(float(m.group(2)))
            des.append(float(m.group(3))); free.append(float(m.group(4))); wait.append(int(m.group(5)))
    return ep, lam, des, free, wait


def cmd_lambda(args):
    import statistics as _st
    out = args[0]
    runs = [a.split("=", 1) for a in args[1:]]   # label=ta.log (one or more)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    any_data = False
    for i, (label, path) in enumerate(runs):
        ep, lam, des, free, wait = _load_lambda(path)
        if not ep:
            print(f"  [warn] no 'dual_descent: epoch=' lines in {path} (policy not dual_descent/fidelity, or run pre-logging)")
            continue
        any_data = True
        c = colors[i % len(colors)]
        ax0.plot(ep, lam, color=c, lw=1.5,
                 label=f"{label}  (final {lam[-1]:.2e}, max {max(lam):.2e}, mean {_st.mean(lam):.2e})")
        # driver signal: desired/free  (>1 pushes lambda up, <1 pulls it toward 0)
        ratio = [d / max(1.0, fr) for d, fr in zip(des, free)]
        ax1.plot(ep, ratio, color=c, lw=1.2, alpha=0.85, label=label)
    ax0.set(ylabel="dual price  lambda", title="dual_descent KV price (lambda) over scheduler epochs")
    ax0.grid(alpha=0.3); ax0.legend(fontsize=9)
    ax1.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.6, label="neutral (desired=free)")
    ax1.set(xlabel="scheduler epoch (~scheduler-interval each)",
            ylabel="desired / free", title="price driver:  lambda rises while desired/free > 1")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"saved {out}: {len(runs)} run(s)" + ("" if any_data else "  [NO lambda data found]"))


CMDS = {"pdt": cmd_pdt, "saturation": cmd_saturation, "kvlog": cmd_kvlog, "compare": cmd_compare,
        "gantt": cmd_gantt, "concurrency": cmd_concurrency, "conckv": cmd_conckv, "lambda": cmd_lambda}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__); sys.exit(1)
    CMDS[sys.argv[1]](sys.argv[2:])
