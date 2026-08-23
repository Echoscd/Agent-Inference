#!/bin/bash
# Download the benchmark data the experiments read. Not vendored in the repo:
# SWE-bench Verified is ~8 MB of JSONL and is redistributed by its authors.
#
#   scripts/fetch_data.sh
#
# Writes data/swebench_verified.jsonl. data/ids80.txt (the 80 instance ids used
# by every 80-way run) IS tracked in the repo, so runs stay comparable.
set -euo pipefail
EXP=${AGENT_EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT="$EXP/data/swebench_verified.jsonl"
mkdir -p "$EXP/data"

if [ -s "$OUT" ]; then
  echo "already present: $OUT ($(wc -l < "$OUT") instances)"
  exit 0
fi

python3 - "$OUT" <<'PY'
import json, sys
from datasets import load_dataset
out = sys.argv[1]
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
with open(out, "w") as f:
    for r in ds:
        f.write(json.dumps(dict(r)) + "\n")
print(f"wrote {out}: {len(ds)} instances")
PY
