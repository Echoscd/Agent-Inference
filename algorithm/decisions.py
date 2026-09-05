#!/usr/bin/env python3
"""Print what a scheduling policy actually did, tick by tick.

Reads `decision_trace_<arm>.jsonl`, which the Router writes once per scheduler
tick (default every 3 s) when TA_DECISION_TRACE is set.

    decisions.py timeline <trace> [--all]      one line per tick
    decisions.py tick     <trace> <tick>       the full candidate table for one tick
    decisions.py program  <trace> <pid>        one program's life across the run
    decisions.py summary  <trace>              who was admitted/evicted, and how often

Traces written before 2026-09-04 carry each task's state (P, D, status, step,
marked) but not the policy's scores, so the candidate table shows what the
policy chose without the numbers behind it. Newer traces also carry `sort_key`,
`evict_key` and `cache_value`, and those columns then appear automatically.
"""
import json
import sys


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _fmt_scores(task):
    bits = []
    for k, label in (("sort_key", "sort"), ("evict_key", "evict"),
                     ("cache_value", "value"), ("density", "dens")):
        v = task.get(k)
        if v is None:
            continue
        x = v[0] if isinstance(v, list) and v else v
        if isinstance(x, (int, float)):
            bits.append(f"{label}={x:.4g}")
    return "  ".join(bits)


def cmd_timeline(ticks, show_all=False):
    print(f"{'tick':>5}{'t_s':>9}{'free_KV':>11}{'reason':>8}{'wait':>6}  decisions")
    for r in ticks:
        acts = []
        if r["admitted"]:
            acts.append(f"admit {len(r['admitted'])}: " + ", ".join(r["admitted"][:4])
                        + ("…" if len(r["admitted"]) > 4 else ""))
        if r["evicted"]:
            ev = [e["pid"] if isinstance(e, dict) else str(e) for e in r["evicted"]]
            why = {e.get("reason") for e in r["evicted"] if isinstance(e, dict)}
            acts.append(f"EVICT {len(ev)} ({'/'.join(sorted(w for w in why if w))}): "
                        + ", ".join(ev[:4]) + ("…" if len(ev) > 4 else ""))
        if not acts and not show_all:
            continue
        print(f"{r['tick']:>5}{r['t']:>9.1f}{r['free']:>11,}{r['n_reasoning']:>8}"
              f"{r['n_waiting']:>6}  {' | '.join(acts)}")


def cmd_tick(ticks, want):
    rec = next((r for r in ticks if r["tick"] == want), None)
    if rec is None:
        print(f"no tick {want} (range {ticks[0]['tick']}..{ticks[-1]['tick']})")
        return
    adm = set(rec["admitted"])
    ev = {e["pid"]: e.get("reason", "") for e in rec["evicted"] if isinstance(e, dict)}
    print(f"tick {rec['tick']}  t={rec['t']}s  policy={rec['policy']}"
          f"  free_KV={rec['free']:,}  lambda={rec['lambda']}")
    print(f"running={rec['n_reasoning']}  waiting={rec['n_waiting']}"
          f"  admitted={len(adm)}  evicted={len(ev)}\n")
    for group in ("waiting", "reasoning"):
        tasks = rec.get(group) or []
        if not tasks:
            continue
        # rank by the policy's own admission key when the trace has it
        keyed = [t for t in tasks if isinstance(t.get("sort_key"), list) and t["sort_key"]]
        if keyed:
            tasks = sorted(tasks, key=lambda t: (t.get("sort_key") or [0])[0], reverse=True)
            note = " (sorted by the policy's admission key, best first)"
        else:
            tasks = sorted(tasks, key=lambda t: -t.get("P", 0))
            note = " (no scores in this trace; sorted by footprint)"
        print(f"--- {group}: {len(tasks)}{note}")
        print(f"  {'':2}{'program':<30}{'P tok':>9}{'D':>7}{'step':>6}{'mark':>6}  outcome / scores")
        for t in tasks[:25]:
            pid = t["pid"]
            mark = "ADMIT" if pid in adm else ("EVICT:" + ev[pid] if pid in ev else "")
            print(f"  {'':2}{pid:<30}{t.get('P', 0):>9,}{t.get('D', 0):>7}"
                  f"{t.get('step', 0):>6}{'Y' if t.get('marked') else '':>6}"
                  f"  {mark:<14}{_fmt_scores(t)}")
        if len(tasks) > 25:
            print(f"  {'':2}… {len(tasks) - 25} more")
        print()


def cmd_program(ticks, pid):
    print(f"program {pid}\n")
    print(f"{'tick':>5}{'t_s':>9}  {'state':<10}{'P tok':>9}{'step':>6}  event")
    prev = None
    for r in ticks:
        where = state = None
        for g in ("reasoning", "waiting"):
            for t in r.get(g) or []:
                if t["pid"] == pid:
                    where, state = g, t
                    break
            if where:
                break
        ev = next((e for e in r["evicted"]
                   if isinstance(e, dict) and e["pid"] == pid), None)
        event = ""
        if pid in r["admitted"]:
            event = "ADMITTED"
        if ev:
            event = f"EVICTED ({ev.get('reason', '')})"
        if not event and where == prev:
            continue                      # only print transitions and events
        prev = where
        print(f"{r['tick']:>5}{r['t']:>9.1f}  {(where or '-'):<10}"
              f"{(state or {}).get('P', 0):>9,}{(state or {}).get('step', 0):>6}  {event}")


def cmd_summary(ticks):
    from collections import Counter
    adm, ev, why = Counter(), Counter(), Counter()
    for r in ticks:
        adm.update(r["admitted"])
        for e in r["evicted"]:
            if isinstance(e, dict):
                ev[e["pid"]] += 1
                why[e.get("reason", "?")] += 1
    print(f"policy={ticks[0]['policy']}  ticks={len(ticks)}  span={ticks[-1]['t']:.0f}s")
    print(f"admissions={sum(adm.values())} over {len(adm)} programs   "
          f"evictions={sum(ev.values())} over {len(ev)} programs")
    print(f"eviction reasons: {dict(why)}\n")
    print("most-evicted programs:")
    for pid, c in ev.most_common(10):
        print(f"  {c:>3}x evicted, {adm.get(pid, 0):>3}x admitted   {pid}")
    if not ev:
        print("  (none)")
    print("\nmost-readmitted programs:")
    for pid, c in adm.most_common(5):
        print(f"  {c:>3}x admitted, {ev.get(pid, 0):>3}x evicted   {pid}")


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    cmd, path = argv[1], argv[2]
    ticks = load(path)
    if not ticks:
        print(f"no records in {path}")
        return 1
    if cmd == "timeline":
        cmd_timeline(ticks, "--all" in argv)
    elif cmd == "tick":
        cmd_tick(ticks, int(argv[3]))
    elif cmd == "program":
        cmd_program(ticks, argv[3])
    elif cmd == "summary":
        cmd_summary(ticks)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
