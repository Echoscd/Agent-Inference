#!/usr/bin/env python3
"""Emit steady_metrics.{json,csv} for one or more A/B run folders.

Thin CLI over run_metrics: all definitions (warmup rule, percentiles, timeline
reconstruction, overlap-weighted throughput) live in run_metrics.py.

    warmup_metrics.py [--out DIR] RUN_DIR [RUN_DIR ...]

--out defaults to the first RUN_DIR, so the A/B harness drops each run's metrics
into the run's own result folder. Pass a shared --out to build a cross-run rollup.

Why not just read results_*_summary.json: those numbers are whole-run, so they
include the cold ramp (80 programs launch at once, KV fills from empty) and the
drain tail. This reports them on the full / warm / steady windows instead, off
the tape, which -- unlike results.jsonl -- also covers programs that never
returned.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics as rm


def main(argv):
    argv = list(argv[1:])
    out_dir = None
    if argv and argv[0] == "--out":
        out_dir, argv = argv[1], argv[2:]
    if not argv:
        print(__doc__)
        return []
    out_dir = out_dir or argv[0]

    rows = rm.collect(argv)
    for r in rows:
        d = r.to_dict()
        print(f"[ok] {d['run']}/{d['arm']} policy={d['policy']} calls={d['calls_total']} "
              f"warm@{d['t_warm_s']}s drain@{d['t_drain_s']}s timeline={d['timeline']}")
    for rd in argv:
        for arm in ("A", "B"):
            if not rm.RunArtifacts(rd, arm).has("tape", "kv"):
                print(f"[skip] {os.path.basename(rd.rstrip('/'))}/{arm} (no tape/kv)")
    rm.write(rows, out_dir)
    print(f"\nwrote {out_dir}/steady_metrics.json and .csv  ({len(rows)} run-arms)")
    return rows


if __name__ == "__main__":
    main(sys.argv)
