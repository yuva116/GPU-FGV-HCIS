"""Regenerates the tables/figures of "Scalable algorithms for compact spanners on real world graphs" (ICS'23).

Usage: python3 spanner_report.py results.csv [out_dir] [--raw]

Input CSV (written by run_benchmark):
    graph,group,vertices,edges,algorithm,stretch,spanner_edges,time_ms
    algorithms: MPVX5-B MPVX5-CS FGV5-B FGV5-CS FGV3-B FGV3-CS (+ CS5 CS3)
    group     : snap | synthetic | sweep

Outputs (out_dir):
    table1.md/.csv        graph features            (paper Table 1)
    table2.md/.csv/.tex   spanner sizes + reduction (paper Table 2)
    figure5.png/.pdf      spanner sizes on the benchmark graphs (paper Fig. 5)
    figure6.png/.pdf      RMAT (a,b) heatmaps for FGV5-CS      (paper Fig. 6)
    runtime.png           extra: build time per algorithm (not in the paper)
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd

# Row order and display names exactly as in the paper's Table 2.
SNAP_ORDER = ["com-Orkut", "soc-LiveJournal1", "com-LiveJournal", "wiki-topcats", "soc-Pokec", "cit-Patents",
              "web-BerkStan", "wiki-Talk", "web-Google", "roadNet-CA", "amazon0505", "web-Stanford", "roadNet-PA",
              "web-NotreDame", "com-DBLP", "com-Amazon", "amazon0302", "soc-Epinions1"]
SYNTH_ORDER = ["Graph500", "CAHepPh", "WEBNotreDame"]
# (baseline, compact) pairs shown as column groups in Table 2
PAIRS = [("MPVX5-B", "MPVX5-CS"), ("FGV5-B", "FGV5-CS"), ("FGV3-B", "FGV3-CS")]
# Figure 5: all graphs downloaded by scripts/run_benchmarks.sh and the three baseline/CS pairs
FIG5_GRAPHS = [("com-Orkut", "com-Orkut"), ("soc-LiveJournal1", "soc-LiveJournal1"),
               ("com-LiveJournal", "com-LiveJournal1"), ("wiki-topcats", "wiki-Topcats"),
               ("soc-Pokec", "soc-Pokec"), ("web-BerkStan", "web-BerkStan"),
               ("web-Google", "web-Google"), ("roadNet-CA", "roadNet-CA")]
FIG5_PAIRS = [("MPVX5-B", "MPVX5-CS", "MPVX"), ("FGV5-B", "FGV5-CS", "FGV5"), ("FGV3-B", "FGV3-CS", "FGV3")]
FIG5_PAIR_COLORS = {"MPVX": "#2878a5", "FGV5": "#d17a22", "FGV3": "#42805c"}
PLOT_ALGO_LABELS = {
    "MPVX5-B": "MILLER (t=5)",
    "MPVX5-CS": "PATHAK-MILLER (t=5)",
    "FGV5-B": "FORSTER (t=5)",
    "FGV5-CS": "PATHAK-FORSTER (t=5)",
    "FGV3-B": "FORSTER (t=3)",
    "FGV3-CS": "PATHAK-FORSTER (t=3)",
}
BLUE = "#1f6fa5"


# The paper's "edges" are adjacency entries (2 x undirected edges; e.g. web-NotreDame 2,180,216 = 2 x 1,090,108),
# and spanner size % is measured against that, so keeping every edge shows up as ~50%.
# PAPER_CONVENTION=True reproduces that; --raw uses plain undirected edge counts (all-edges = 100%).
PAPER_CONVENTION = True


# Legacy run_benchmark CSV (no "group" column; algorithms FGV / CS / FGV-CS) -> paper-style names.
LEGACY_GRAPH = {"com-orkut.ungraph": "com-Orkut", "com-lj.ungraph": "com-LiveJournal",
                "soc-pokec-relationships": "soc-Pokec", "soc-pokec": "soc-Pokec",
                "com-orkut": "com-Orkut", "com-livejournal": "com-LiveJournal"}
LEGACY_ALGO = {("FGV", 5): "FGV5-B", ("FGV-CS", 5): "FGV5-CS", ("CS", 5): "CS5",
               ("FGV", 3): "FGV3-B", ("FGV-CS", 3): "FGV3-CS", ("CS", 3): "CS3"}


def normalize_legacy(df):
    """Accepts the older CSV (graph,vertices,edges,algorithm,stretch,spanner_edges,time_ms)."""
    df = df.copy()
    if "graph" not in df.columns and "dgraph" in df.columns:
        df = df.rename(columns={"dgraph": "graph"})
    df["graph"] = df["graph"].map(lambda g: LEGACY_GRAPH.get(g, LEGACY_GRAPH.get(g.lower(), g)))
    df["algorithm"] = [LEGACY_ALGO.get((a, s), a if str(a).startswith("MPVX") else None)
                       for a, s in zip(df.algorithm, df.stretch)]
    df = df.dropna(subset=["algorithm"])          # drops stretch-7 rows (not used by the report)
    df["group"] = "snap"
    return df[["graph", "group", "vertices", "edges", "algorithm", "stretch", "spanner_edges", "time_ms"]]


def load(csv):
    df = pd.read_csv(csv)
    if "group" not in df.columns:
        df = normalize_legacy(df)
    df["pct"] = 100.0 * df.spanner_edges / (df.edges * (2 if PAPER_CONVENTION else 1))
    return df


def pct_table(df):
    """graph x algorithm -> spanner size as % of original edges (mean over repetitions)."""
    return df.pivot_table(index="graph", columns="algorithm", values="pct", aggfunc="mean")


def reduction(base, cs):
    return 100.0 * (base - cs) / base


def plot_algo_label(algorithm):
    return PLOT_ALGO_LABELS.get(algorithm, algorithm)


# ------------------------------------------------------------------ Table 1 / Table 2
def table1(df, out):
    g = df.drop_duplicates("graph")[["graph", "group", "vertices", "edges"]].copy()
    g = g[g.group != "sweep"]
    if PAPER_CONVENTION:
        g["edges"] = 2 * g.edges                      # paper lists 2 x undirected edges
        g["Avg Degree"] = (2.0 * g.edges / g.vertices).round(2)   # ...and degree = 2 x that / V (matches Table 1)
    else:
        g["Avg Degree"] = (2.0 * g.edges / g.vertices).round(2)
    order = {n: i for i, n in enumerate(SNAP_ORDER + SYNTH_ORDER)}
    g = g.assign(o=g.graph.map(order).fillna(99)).sort_values("o").drop(columns=["o", "group"])
    g.columns = ["Input Graph", "Vertices", "Edges", "Avg Degree"]
    g.to_csv(out / "table1.csv", index=False)
    (out / "table1.md").write_text(g.to_markdown(index=False))
    return g


def table2(df, out):
    p = pct_table(df[df.group != "sweep"])
    rows = []
    extras = [g for g in p.index if g not in SNAP_ORDER + SYNTH_ORDER]   # graphs not in the paper (e.g. test graphs)
    for name in SNAP_ORDER + SYNTH_ORDER + extras:
        if name not in p.index:
            continue
        r = {"Input Graph": name}
        for b, c in PAIRS:
            bv = p.at[name, b] if b in p.columns else np.nan
            cv = p.at[name, c] if c in p.columns else np.nan
            r[f"{b}"], r[f"{c}"] = bv, cv
            r[f"Reduction ({c.split('-')[0]})"] = reduction(bv, cv)
        rows.append(r)
    t = pd.DataFrame(rows)

    # averages quoted in Sec. 5.3.3/5.3.4 (SNAP graphs only)
    extra_rows = []
    for label, names in (("Average (SNAP)", SNAP_ORDER), ("Average (Figure 5)", [g for g, _ in FIG5_GRAPHS])):
        sub = t[t["Input Graph"].isin(names)]
        if len(sub):
            extra_rows.append({"Input Graph": label, **{c: sub[c].mean() for c in t.columns[1:]}})
    full = pd.concat([t, pd.DataFrame(extra_rows)], ignore_index=True) if extra_rows else t

    full.round(2).to_csv(out / "table2.csv", index=False)
    (out / "table2.md").write_text(full.round(2).to_markdown(index=False, floatfmt=".2f"))
    write_latex(t, out / "table2.tex")
    return full


def write_latex(t, path):
    fm = lambda x: "--" if pd.isna(x) else f"{x:.2f}\\%"
    L = [r"\begin{table*}[t]", r"\centering\small",
         r"\begin{tabular}{l||rrr|rrr|rrr}", r"\hline",
         r"Input Graph & \multicolumn{3}{c|}{Spanner size (MPVX5)} & \multicolumn{3}{c|}{Spanner size (FGV5)}"
         r" & \multicolumn{3}{c}{Spanner size (FGV3)}\\",
         r" & B & CS & Reduction & B & CS & Reduction & B & CS & Reduction \\ \hline"]
    for i, r in t.iterrows():
        if r["Input Graph"] == SYNTH_ORDER[0]:
            L.append(r"\hline")
        cells = [fm(r[k]) for k in t.columns[1:]]
        L.append(r["Input Graph"] + " & " + " & ".join(cells) + r" \\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Spanner sizes as \% of original graph sizes and reduction of the compact spanner algorithms "
          r"over the baselines for SNAP graphs (top 18) and synthetic graphs (bottom 3).}", r"\end{table*}"]
    path.write_text("\n".join(L))


# ------------------------------------------------------------------ Figure 5
def figure5(df, out):
    p = pct_table(df[df.group != "sweep"])
    if not any(g in p.index for g, _ in FIG5_GRAPHS):
        print("none of the Figure 5 graphs are in the results; skipping figure 5")
        return
    fig, ax = plt.subplots(figsize=(16, 7))
    bw, pos, centers = 0.8, 0.0, []
    plt.rcParams["hatch.linewidth"] = 0.6
    legend_handles = []
    legend_labels = []
    for baseline, compact, pair in FIG5_PAIRS:
        for algorithm in (baseline, compact):
            legend_handles.append(plt.Rectangle(
                (0, 0), 1, 1, facecolor=FIG5_PAIR_COLORS[pair], edgecolor="black",
                hatch="//////" if algorithm.endswith("-CS") else None))
            legend_labels.append(plot_algo_label(algorithm))

    for gname, _ in FIG5_GRAPHS:
        start = pos
        for baseline, compact, pair in FIG5_PAIRS:
            bv = p.at[gname, baseline] if gname in p.index and baseline in p else np.nan
            cv = p.at[gname, compact] if gname in p.index and compact in p else np.nan
            color = FIG5_PAIR_COLORS[pair]
            ax.bar(pos, bv, bw, color=color, edgecolor="black", linewidth=0.8)
            ax.bar(pos + bw, cv, bw, facecolor=color, edgecolor="black", hatch="//////", linewidth=0.8)
            pos += 2 * bw + 0.35
        centers.append((start + pos - 0.35) / 2)
        pos += 0.9
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in FIG5_GRAPHS], rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=19)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Spanner Size (% of edges)", fontsize=20)
    ax.tick_params(axis="y", labelsize=18)

    ax.legend(handles=legend_handles, labels=legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=3, fontsize=19)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.40, top=0.70)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure5.{ext}", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 6
def figure6(df, out):
    s = df[df.group == "sweep"].copy()
    if s.empty:
        print("no RMAT sweep rows; skipping figure 6")
        return
    m = s.graph.str.extract(r"rmat_a([\d.]+)_b([\d.]+)").astype(float)
    s["a"], s["b"] = m[0], m[1]
    p = s.pivot_table(index=["a", "b"], columns="algorithm", values="pct", aggfunc="mean")
    size = p["FGV5-CS"].unstack("b").sort_index(ascending=False)          # a=0.5 on top, like the paper
    red = reduction(p["FGV5-B"], p["FGV5-CS"]).unstack("b").sort_index(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, title, cmap in ((axes[0], size, "(a) % of original graph size", "Blues"),
                                  (axes[1], red, "(b) % reduction over baseline", "Blues")):
        im = ax.imshow(data.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels([f"{v:g}" for v in data.columns])
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels([f"{v:g}" for v in data.index])
        ax.set_xlabel("b=c")
        ax.set_ylabel("a")
        ax.set_title(title, y=-0.28, fontsize=10)
        lo, hi = np.nanmin(data.values), np.nanmax(data.values)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.values[i, j]
                dark = (v - lo) / (hi - lo + 1e-9) > 0.55
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if dark else "black")
    fig.suptitle("Figure 6: Heatmaps for spanner sizes obtained using PATHAK-FORSTER (t=5) on RMAT graphs "
                 "obtained by varying a and b", y=0.01, va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure6.{ext}", dpi=200)
    plt.close(fig)


def dense_log_yticks(ax):
    """Log y-axis with a tick at every 1..9 x 10^k, labelled at 1, 2 and 5 so the scale can be read accurately."""
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=30))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=30))
    def label(x, _):
        m = x / 10 ** np.floor(np.log10(x))
        return f"{x:g}" if np.isclose(m, [1, 2, 5]).any() else ""
    ax.yaxis.set_major_formatter(FuncFormatter(label))
    ax.yaxis.set_minor_formatter(FuncFormatter(label))
    ax.grid(axis="y", which="major", alpha=0.35)
    ax.grid(axis="y", which="minor", alpha=0.15)


# ------------------------------------------------------------------ extra: runtime
def runtime(df, out):
    d = df[df.group != "sweep"]
    t = d.pivot_table(index="graph", columns="algorithm", values="time_ms", aggfunc="mean")
    names = [g for g in SNAP_ORDER + SYNTH_ORDER if g in t.index]
    algos = [a for a in ["MPVX5-B", "MPVX5-CS", "FGV5-B", "FGV5-CS", "FGV3-B", "FGV3-CS"] if a in t.columns]
    if not names or not algos:
        return
    fig, ax = plt.subplots(figsize=(16, 7))
    w = 0.8 / len(algos)
    for i, algorithm in enumerate(algos):
        ax.bar(np.arange(len(names)) + i * w, t.loc[names, algorithm], w,
               label=plot_algo_label(algorithm))
    ax.set_yscale("log")
    ax.set_ylabel("Time (ms)", fontsize=20)
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=19)
    ax.tick_params(axis="y", which="both", labelsize=18)
    dense_log_yticks(ax)
    ax.legend(ncol=3, fontsize=19, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.40, top=0.70)
    fig.savefig(out / "runtime.png", dpi=150)
    fig.savefig(out / "runtime.pdf")
    plt.close(fig)


def sizes_all(df, out):
    """Extra: spanner size on every dataset present in the results."""
    p = pct_table(df[df.group != "sweep"])
    names = [g for g in SNAP_ORDER + SYNTH_ORDER if g in p.index]
    algos = [a for a in ["MPVX5-B", "MPVX5-CS", "FGV5-B", "FGV5-CS", "FGV3-B", "FGV3-CS"] if a in p.columns]
    if not names or not algos:
        return
    fig, ax = plt.subplots(figsize=(13, 4.5))
    w = 0.8 / len(algos)
    for i, a in enumerate(algos):
        ax.bar(np.arange(len(names)) + i * w, p.loc[names, a], w, label=plot_algo_label(a))
    ax.set_ylabel("Spanner size (% of graph edges)")
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "sizes_all.png", dpi=150)
    plt.close(fig)


def main(csv, out="."):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(csv)
    graphs = [g for g in SNAP_ORDER + SYNTH_ORDER if g in set(df.graph)] + \
             sorted(set(df.graph) - set(SNAP_ORDER + SYNTH_ORDER))
    print(f"datasets in results ({len(graphs)}): {', '.join(graphs)}\n")
    print(table1(df, out).to_string(index=False), "\n")
    print(table2(df, out).round(2).to_string(index=False))
    figure5(df, out)
    figure6(df, out)
    runtime(df, out)
    sizes_all(df, out)
    print(f"\nwrote outputs to {out}/ ({'paper' if PAPER_CONVENTION else 'raw undirected'} edge convention)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    args = [a for a in sys.argv[1:] if a != "--raw"]
    if "--raw" in sys.argv:
        PAPER_CONVENTION = False
    main(args[0], args[1] if len(args) > 1 else ".")
