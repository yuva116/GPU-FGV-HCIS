#!/usr/bin/env bash
# Downloads 6 SNAP graphs, runs the benchmark, and generates the report.
# Usage: scripts/run_benchmarks.sh [build_dir]   (run from the project root)
set -euo pipefail
BUILD=${1:-build}
DIR=datasets/snap
mkdir -p "$DIR" results

URLS=(
  https://snap.stanford.edu/data/soc-Epinions1.txt.gz
  https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
  https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
  https://snap.stanford.edu/data/web-NotreDame.txt.gz
  https://snap.stanford.edu/data/web-Stanford.txt.gz
  https://snap.stanford.edu/data/roadNet-PA.txt.gz
)
for u in "${URLS[@]}"; do
  f="$DIR/$(basename "${u%.gz}")"
  [ -f "$f" ] || { curl -fsSL "$u" | gunzip > "$f"; }
done

"$BUILD"/run_benchmark results/results.csv "$DIR"/*.txt
python3 report/spanner_report.py results/results.csv results
