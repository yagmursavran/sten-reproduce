#!/usr/bin/env bash
# Fig. 14 (1 GPU): run every configuration in its own Python process, so GPU memory from one
# configuration cannot slow down the next (in a single process the step time kept growing from
# configuration to configuration). Each process also times dense PyTorch as its own reference.
#
# Usage: bash experiments/run_fig14.sh <output-folder> [extra arguments for masked_training_overhead.py]
# Example: bash experiments/run_fig14.sh results/fig14
set -u
OUT=${1:?give an output folder, e.g. results/fig14}
shift
mkdir -p "$OUT"
for cfg in fixed-50% fixed-3:6 fixed-3:6:8 new-50% new-3:6 new-3:6:8; do
  name=$(echo "$cfg" | tr ':%' '-p')
  python -W ignore experiments/masked_training_overhead.py --configs "dense,$cfg" \
      --out "$OUT/part_$name.csv" "$@" 2>&1 | tee "$OUT/part_$name.log"
done
python - "$OUT" <<'PY'
import csv, glob, os, sys
out = sys.argv[1]
order = ["fixed-50p", "fixed-3-6", "fixed-3-6-8", "new-50p", "new-3-6", "new-3-6-8"]
rows = []
for name in order:
    path = os.path.join(out, f"part_{name}.csv")
    if not os.path.exists(path):
        print("missing", path); continue
    part = list(csv.DictReader(open(path)))
    if not rows:
        rows.append(part[0])                   # dense from the first process
    rows += [r for r in part if r["config"] != "dense"]
with open(os.path.join(out, "merged.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("merged:", [r["config"] for r in rows])
PY
