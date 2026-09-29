"""Performance report: table (like Table 2 of the paper) + plots.

Usage: python3 spanner_report.py results.csv [out_dir]
Algorithms in the CSV: FGV (baseline), CS (compact spanner), FGV-CS (hybrid).
Outputs: table.md, table.csv, spanner_size_vs_stretch.png, build_time_vs_stretch.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ALGOS = [("FGV", "o-"), ("CS", "s--"), ("FGV-CS", "^-")]


def build_table(df):
    df = df.assign(pct=100.0 * df.spanner_edges / df.edges)
    p = df.pivot_table(index=["graph", "stretch"], columns="algorithm", values="pct").reset_index()
    out = p[["graph", "stretch"]].copy()
    for a, _ in ALGOS:
        if a in p:
            out[f"{a} size %"] = p[a]
    for a in ("CS", "FGV-CS"):
        if a in p and "FGV" in p:
            out[f"{a} reduction %"] = 100.0 * (p["FGV"] - p[a]) / p["FGV"]
    return out.sort_values(["graph", "stretch"]).round(2)


def plot(df, column, ylabel, fname, out, logy=False):
    graphs = sorted(df.graph.unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for ax, g in zip(axes.flat, graphs):
        d = df[df.graph == g]
        for algo, style in ALGOS:
            s = d[d.algorithm == algo].sort_values("stretch")
            if len(s):
                ax.plot(s.stretch, s[column], style, label=algo)
        ax.set_title(g)
        ax.set_xlabel("Stretch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(d.stretch.unique()))
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in list(axes.flat)[len(graphs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def main(csv, out="."):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    df["size_pct"] = 100.0 * df.spanner_edges / df.edges

    table = build_table(df)
    table.to_csv(out / "table.csv", index=False)
    (out / "table.md").write_text(table.to_markdown(index=False))
    print(table.to_string(index=False))

    plot(df, "size_pct", "Spanner size (% of graph edges)", "spanner_size_vs_stretch.png", out)
    plot(df, "time_ms", "Build time (ms)", "build_time_vs_stretch.png", out, logy=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
