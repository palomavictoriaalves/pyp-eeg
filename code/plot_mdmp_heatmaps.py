"""Heatmaps of MDM adjacency matrices by group and condition.

Rows : EO-PRE, EO-POST, EC-PRE, EC-POST
Cols : Active, Passive, Control
Cell : n_nodes × n_nodes matrix — colour = proportion of subjects with causal edge

Input : results/mdmp*/mdmp_edges_long.csv  (produced by calc_mdmp.py)
Output: results/plots/mdmp_heatmaps/*.png
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import config

# ---------------------------------------------------------------------------
# Paths / canonical orders
# ---------------------------------------------------------------------------
OUT_DIR = config.PLOTS_DIR / "mdmp_heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS      = config.GROUPS_ORDER   # ["Active", "Passive", "Control"]
STATES      = config.VS_ORDER       # ["EO", "EC"]
SESSIONS    = config.SESS_ORDER     # ["PRE", "POST"]
BANDS_ORDER = config.BANDS_ORDER
NODES_ORDER = config.ROIS_ORDER

ROW_ORDER = [
    (STATES[0], SESSIONS[0]),   # EO-PRE
    (STATES[0], SESSIONS[1]),   # EO-POST
    (STATES[1], SESSIONS[0]),   # EC-PRE
    (STATES[1], SESSIONS[1]),   # EC-POST
]

# Short labels for node axes
NODE_ABBREV = {
    "Prefrontal":       "PFr",
    "Frontal":          "Fr",
    "Frontocentral":    "FCe",
    "Central":          "Ce",
    "Temporo-parietal": "TP",
    "Centro-parietal":  "CP",
    "Parietal":         "Pa",
    "Occipital":        "Oc",
}

# Subject → group mapping
SUBJECT_GROUP: dict[str, str] = {}
for _sid in config.GROUP_ACTIVE:
    SUBJECT_GROUP[str(_sid).zfill(2)] = "Active"
for _sid in config.GROUP_PASSIVE:
    SUBJECT_GROUP[str(_sid).zfill(2)] = "Passive"
for _sid in config.GROUP_CONTROL:
    SUBJECT_GROUP[str(_sid).zfill(2)] = "Control"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_input_dirs() -> list[Path]:
    """Auto-detect mdmp output directories under results/."""
    candidates = sorted(config.RESULTS_DIR.glob("mdmp*"))
    found = [d for d in candidates if d.is_dir() and (d / "mdmp_edges_long.csv").exists()]
    if not found:
        raise SystemExit(
            "No mdmp_edges_long.csv found under results/mdmp*/. "
            "Run calc_mdmp.py first."
        )
    return found


def load_edges(input_dirs: list[Path]) -> pd.DataFrame:
    """Load and concatenate mdmp_edges_long.csv from all input directories."""
    frames = []
    for d in input_dirs:
        csv = d / "mdmp_edges_long.csv"
        frames.append(pd.read_csv(csv))
        print(f"Loaded: {csv}  ({len(frames[-1])} rows)")
    return pd.concat(frames, ignore_index=True)


def freq_matrix(df_slice: pd.DataFrame, nodes: list[str]) -> tuple[np.ndarray, int]:
    """Compute edge-frequency matrix (n × n) for a group/condition slice.

    Cell [i, j] = proportion of subjects that have a causal edge  node_i → node_j.
    Diagonal is set to NaN (no self-loops in a DAG).
    """
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    mat = np.zeros((n, n), dtype=float)

    n_subs = df_slice["subject"].nunique()
    if n_subs == 0:
        return mat, 0

    for (parent, child), grp in df_slice.groupby(["parent", "child"]):
        pi, ci = node_idx.get(parent), node_idx.get(child)
        if pi is not None and ci is not None:
            mat[pi, ci] = grp["subject"].nunique() / n_subs

    np.fill_diagonal(mat, np.nan)
    return mat, n_subs


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------
def render_panel(df: pd.DataFrame, band: str, metric: str, args) -> plt.Figure | None:
    sub = df[(df["band"] == band) & (df["metric"] == metric)].copy()
    if sub.empty:
        print(f"[WARN] No data for band={band}, metric={metric}.")
        return None

    # Restrict to nodes present in data, preserving canonical order
    all_nodes_in_data = set(sub["parent"].unique()) | set(sub["child"].unique())
    nodes = [n for n in NODES_ORDER if n in all_nodes_in_data]
    if not nodes:
        print(f"[WARN] No recognisable nodes for band={band}, metric={metric}.")
        return None

    n = len(nodes)
    abbrevs = [NODE_ABBREV.get(nd, nd[:4]) for nd in nodes]

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharex=True, sharey=True)

    for row_idx, (state, session) in enumerate(ROW_ORDER):
        for col_idx, group in enumerate(GROUPS):
            ax = axes[row_idx, col_idx]

            d_slice = sub[
                (sub["visual_state"] == state) &
                (sub["session"]      == session) &
                (sub["group"]        == group)
            ]

            if d_slice.empty:
                ax.set_axis_off()
                continue

            mat, n_subs = freq_matrix(d_slice, nodes)

            ax.imshow(
                mat,
                aspect="equal",
                origin="upper",
                cmap=args.cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )

            # Annotate cells with frequency values
            if args.annotate:
                for i in range(n):
                    for j in range(n):
                        if i != j and np.isfinite(mat[i, j]) and mat[i, j] > 0:
                            ax.text(
                                j, i, f"{mat[i, j]:.2f}",
                                ha="center", va="center",
                                fontsize=6, color="black",
                            )

            # Axis ticks — only on outermost subplots (sharex/sharey handles the rest)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(abbrevs, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(abbrevs, fontsize=8)

            # Column title (group name + n)
            if row_idx == 0:
                ax.set_title(f"{group}  (n={n_subs})", fontsize=11, pad=8)

            # Row label — condition
            if col_idx == 0:
                ax.set_ylabel(f"{state}-{session}", fontsize=11, labelpad=6)

            # Axis labels on edges
            if row_idx == 3:
                ax.set_xlabel("Child node (target)", fontsize=9)

    # Shared y-axis label (parent node)
    fig.text(
        0.01, 0.5, "Parent node (source)",
        va="center", ha="center",
        rotation="vertical", fontsize=11,
    )

    # Shared colorbar
    cax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=args.cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Edge frequency\n(proportion of subjects)", rotation=90, fontsize=10)

    tag = "REL" if metric == "power_rel" else "ABS"
    unit_str = "Relative Power" if metric == "power_rel" else "Absolute Power"
    fig.suptitle(
        f"MDM Adjacency Heatmap — {band} — {unit_str}\n"
        "Colour = proportion of subjects with causal edge  (parent → child)",
        fontsize=13, y=0.995,
    )

    plt.subplots_adjust(
        left=0.10, right=0.91, top=0.94, bottom=0.07,
        wspace=0.08, hspace=0.28,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="MDM adjacency heatmaps by group and condition."
    )
    p.add_argument(
        "--input-dirs", default=None,
        help="Comma-separated paths to mdmp output dirs. Auto-detects if omitted.",
    )
    _default_metrics = ",".join(config.MDMP_METRICS_TO_RUN)
    p.add_argument(
        "--metrics", default=_default_metrics,
        help=f"Comma-separated metrics to plot (default: {_default_metrics}).",
    )
    p.add_argument("--bands", nargs="*", default=None)
    p.add_argument("--cmap", default=config.CMAP_ADJACENCY)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--annotate", action="store_true",
        help="Print frequency values inside each cell.",
    )
    args = p.parse_args()

    # Resolve input directories
    if args.input_dirs:
        input_dirs = [Path(d.strip()) for d in args.input_dirs.split(",")]
    else:
        input_dirs = find_input_dirs()
    print(f"Input dirs: {[str(d) for d in input_dirs]}")

    df = load_edges(input_dirs)

    # Normalise columns
    df["subject"]      = df["subject"].astype(str).str.zfill(2)
    df["session"]      = (
        df["session"].astype(str).str.upper()
        .map({"PRE": "PRE", "POS": "POST", "POST": "POST"})
    )
    df["visual_state"] = df["visual_state"].astype(str).str.upper()
    df["group"]        = df["subject"].map(SUBJECT_GROUP)
    df = df[df["group"].notna()].copy()

    bands   = args.bands or BANDS_ORDER
    metrics = [m.strip() for m in args.metrics.split(",")]
    print(f"Bands: {bands} | Metrics: {metrics}")

    for metric in metrics:
        for band in bands:
            fig = render_panel(df, band, metric, args)
            if fig is None:
                continue
            tag  = "REL" if metric == "power_rel" else "ABS"
            stem = f"mdmp_heat_{band}_{tag}.png"
            out_png = OUT_DIR / stem
            fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
