#!/usr/bin/env python3
"""Pre-build the conda testbeds for a list of SWE-bench instances, in parallel.

Building a repo takes minutes and is pure setup, so do it once up front: a
concurrency run that builds on the fly measures the build, not the serving.

    python3 algorithm/prebuild96.py [ids_file] [-o built_ids_file]

ids_file defaults to data/ids80.txt (the instance list every 80-way run uses).
Accepts one id per line or comma-separated. Writes the ids that built OK, so a
failed build drops out of the workload instead of failing mid-experiment.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swebench_local_harness import Instance
import paths


def build_one(rows, iid):
    try:
        o = Instance(rows[iid])
        if o.is_built():
            return iid, "already"
        return iid, f"built_{o.build():.0f}s"
    except Exception as e:
        return iid, f"FAIL:{str(e)[:60]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids_file", nargs="?", default=paths.IDS80)
    ap.add_argument("-o", "--out", default=None,
                    help="where to write the successfully built ids "
                         "(default: <ids_file>.built)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--data", default=paths.SWEBENCH_DATA)
    args = ap.parse_args()

    raw = open(args.ids_file).read()
    ids = [x.strip() for x in raw.replace(",", "\n").split("\n") if x.strip()]
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in open(args.data)}

    ok = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for iid, st in ex.map(lambda i: build_one(rows, i), ids):
            print(iid, st, flush=True)
            if st.startswith("built") or st == "already":
                ok.append(iid)

    out = args.out or args.ids_file + ".built"
    open(out, "w").write("\n".join(ok))
    print(f"\nSUCCESS {len(ok)}/{len(ids)} -> {out}")
    return 0 if len(ok) == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
