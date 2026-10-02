#!/usr/bin/env bash
# Downloads 9 SNAP graphs, runs the benchmark, and generates the report.
# Usage: scripts/run_benchmarks.sh [build_dir]   (run from the project root)
set -euo pipefail
BUILD=${1:-build}
DIR=datasets/snap
mkdir -p "$DIR" results

URLS=(
  # 5 large graphs
  https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
  https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
  https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
  https://snap.stanford.edu/data/wiki-topcats.txt.gz
  https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
  # 4 additional graphs
  https://snap.stanford.edu/data/web-BerkStan.txt.gz
  https://snap.stanford.edu/data/web-Google.txt.gz
  https://snap.stanford.edu/data/roadNet-CA.txt.gz
)

FILES=()
for u in "${URLS[@]}"; do
  f="$DIR/$(basename "${u%.gz}")"
  # -s: an empty file counts as missing; download to .tmp first so an interrupted
  # download never leaves a half-written file that later runs would skip.
  [ -s "$f" ] || { echo "downloading $u"; curl -fL "$u" | gunzip > "$f.tmp" && mv "$f.tmp" "$f"; }
  FILES+=("$f")
done

"$BUILD"/run_benchmark results/results.csv "${FILES[@]}"
python3 report/spanner_report.py results/results.csv results
  