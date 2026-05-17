"""Plot directed MDMP networks from MDMP CSV outputs.

Reads one or more MDMP output directories containing:
  - mdmp_edges_long.csv
  - mdmp_delta_by_node.csv

Writes:
  results/plots/mdmp_networks/
    groups/<metric>/<band>/*.png
    individual/<metric>/<band>/*.png
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

import config
from calc_mdmp import VALID_METRICS, parse_metrics


DEFAULT_INPUT_CANDIDATES = (
    config.RESULTS_DIR / "mdmp",
)

SUBJECT_TO_GROUP = {
    **{str(s).zfill(2): "Active" for s in getattr(config, "GROUP_ACTIVE", set())},
    **{str(s).zfill(2): "Passive" for s in getattr(config, "GROUP_PASSIVE", set())},
    **{str(s).zfill(2): "Control" for s in getattr(config, "GROUP_CONTROL", set())},
}

GROUPS_ORDER = list(getattr(config, "GROUPS_ORDER", ["Active", "Passive", "Control"])) + ["Unknown"]
SESS_ORDER = list(getattr(config, "SESS_ORDER", ["PRE", "POST"]))
VS_ORDER = list(getattr(config, "VS_ORDER", ["EO", "EC"]))
BANDS_ORDER = list(getattr(config, "BANDS_ORDER", []))
METRIC_ORDER = ["power_rel", "power_abs"]

ROI_POSITIONS = {
    "Prefrontal":       ( 0.00,  1.00),
    "Frontal":          ( 0.55,  0.70),
    "Frontocentral":    ( 0.55,  0.25),
    "Central":          ( 0.00,  0.10),
    "Temporo-parietal": (-0.95, -0.05),
    "Centro-parietal":  ( 0.00, -0.30),
    "Parietal":         (-0.55, -0.62),
    "Occipital":        ( 0.00, -1.00),
}

ROI_ABBREV: Dict[str, str] = {
    "Prefrontal":       "PFr",
    "Frontal":          "Fr",
    "Frontocentral":    "FCe",
    "Central":          "Ce",
    "Temporo-parietal": "TP",
    "Centro-parietal":  "CP",
    "Parietal":         "Pa",
    "Occipital":        "Oc",
}


def _abbrev(node: str) -> str:
    return ROI_ABBREV.get(node, node[:3])

CONTEXT_COLS = ("metric", "method", "subject", "group", "session", "visual_state", "band")
GLOBAL_CONTEXT_COLS = (
    "metric",
    "method",
    "vts_method",
    "align_method",
    "group",
    "session",
    "visual_state",
    "band",
)
RUN_ID_COLS = ("subject", "session", "visual_state", "band", "metric", "method")


def parse_csv_list(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def normalize_subject(value: object) -> str:
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if text.lower().startswith("sub-"):
        text = text[4:]
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(2)
    return text or "NA"


def normalize_group(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    upper = text.upper()
    if upper.startswith("ACT"):
        return "Active"
    if upper.startswith("PAS"):
        return "Passive"
    if upper.startswith("CON"):
        return "Control"
    return text


def slugify(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "na"
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        elif ch in {" ", "/", "\\"}:
            out.append("_")
        else:
            out.append("-")
    return "".join(out).strip("_-").lower() or "na"


def resolve_input_dirs(raw: str) -> List[Path]:
    if raw:
        candidates = [Path(p).expanduser().resolve() for p in parse_csv_list(raw)]
    else:
        candidates = [path.resolve() for path in DEFAULT_INPUT_CANDIDATES]

    found = []
    seen = set()
    for candidate in candidates:
        if not candidate.exists():
            continue

        search_dirs = [candidate]
        if candidate.is_dir():
            search_dirs.extend(sorted(path for path in candidate.rglob("*") if path.is_dir()))

        for directory in search_dirs:
            edges = directory / "mdmp_edges_long.csv"
            delta = directory / "mdmp_delta_by_node.csv"
            if edges.exists() and delta.exists():
                resolved = directory.resolve()
                if resolved not in seen:
                    found.append(resolved)
                    seen.add(resolved)
    return found


def load_mdmp_tables(input_dirs: Sequence[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    edges_frames: List[pd.DataFrame] = []
    delta_frames: List[pd.DataFrame] = []

    for directory in input_dirs:
        edges_path = directory / "mdmp_edges_long.csv"
        delta_path = directory / "mdmp_delta_by_node.csv"
        if not edges_path.exists() or not delta_path.exists():
            print(f"[plot-mdmp] skip: missing files in {directory}")
            continue

        e = pd.read_csv(edges_path)
        d = pd.read_csv(delta_path)
        e["_source_dir"] = str(directory)
        d["_source_dir"] = str(directory)
        edges_frames.append(e)
        delta_frames.append(d)
        print(f"[plot-mdmp] loaded: {directory}")

    if not edges_frames or not delta_frames:
        raise FileNotFoundError(
            "No MDMP CSV pair found. Expected mdmp_edges_long.csv + "
            "mdmp_delta_by_node.csv in at least one input directory."
        )

    edges = pd.concat(edges_frames, ignore_index=True).drop_duplicates()
    delta = pd.concat(delta_frames, ignore_index=True).drop_duplicates()
    return edges, delta


def clean_table_columns(edges: pd.DataFrame, delta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "parent" not in edges.columns or "child" not in edges.columns:
        raise ValueError("edges CSV must contain columns: parent, child")
    if "node" not in delta.columns or "df_hat" not in delta.columns:
        raise ValueError("delta CSV must contain columns: node, df_hat")

    edges = edges.copy()
    delta = delta.copy()

    for col in ("parent", "child", "method", "metric", "session", "visual_state", "band"):
        if col in edges.columns:
            edges[col] = edges[col].astype(str).str.strip()
    for col in ("node", "method", "metric", "session", "visual_state", "band"):
        if col in delta.columns:
            delta[col] = delta[col].astype(str).str.strip()

    if "metric" not in edges.columns:
        edges["metric"] = "unknown"
    if "metric" not in delta.columns:
        delta["metric"] = "unknown"
    if "method" not in edges.columns:
        edges["method"] = "mdmp"
    if "method" not in delta.columns:
        delta["method"] = "mdmp"

    if "subject" in edges.columns:
        edges["subject"] = edges["subject"].map(normalize_subject)
    if "subject" in delta.columns:
        delta["subject"] = delta["subject"].map(normalize_subject)

    if "session" in edges.columns:
        edges["session"] = edges["session"].str.upper()
    if "session" in delta.columns:
        delta["session"] = delta["session"].str.upper()

    if "visual_state" in edges.columns:
        edges["visual_state"] = edges["visual_state"].str.upper()
    if "visual_state" in delta.columns:
        delta["visual_state"] = delta["visual_state"].str.upper()

    if "group" in edges.columns:
        edges["group"] = edges["group"].map(normalize_group)
    if "group" in delta.columns:
        delta["group"] = delta["group"].map(normalize_group)

    if "group" not in edges.columns:
        edges["group"] = "Unknown"
    if "group" not in delta.columns:
        delta["group"] = "Unknown"

    if "subject" in edges.columns:
        mapped = edges["subject"].map(SUBJECT_TO_GROUP).fillna("Unknown")
        edges["group"] = np.where(edges["group"] == "Unknown", mapped, edges["group"])
    if "subject" in delta.columns:
        mapped = delta["subject"].map(SUBJECT_TO_GROUP).fillna("Unknown")
        delta["group"] = np.where(delta["group"] == "Unknown", mapped, delta["group"])

    delta["df_hat"] = pd.to_numeric(delta["df_hat"], errors="coerce")

    return edges, delta


def sort_for_reporting(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "metric" in out.columns:
        out["metric"] = pd.Categorical(out["metric"], METRIC_ORDER, ordered=True)
    if "group" in out.columns:
        out["group"] = pd.Categorical(out["group"], GROUPS_ORDER, ordered=True)
    if "session" in out.columns:
        out["session"] = pd.Categorical(out["session"], SESS_ORDER, ordered=True)
    if "visual_state" in out.columns:
        out["visual_state"] = pd.Categorical(out["visual_state"], VS_ORDER, ordered=True)
    if "band" in out.columns and BANDS_ORDER:
        out["band"] = pd.Categorical(out["band"], BANDS_ORDER, ordered=True)

    sort_cols = [c for c in CONTEXT_COLS if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def filter_by_context(df: pd.DataFrame, context: Mapping[str, object]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, value in context.items():
        if col in df.columns:
            mask &= df[col] == value
    return df.loc[mask].copy()


def valid_subject_mask(df: pd.DataFrame) -> pd.Series:
    if "subject" not in df.columns:
        return pd.Series(False, index=df.index)
    subjects = df["subject"].astype(str).str.strip()
    return subjects.notna() & ~subjects.isin({"", "NA", "nan", "None"})


def median_context_mask(df: pd.DataFrame) -> pd.Series:
    markers = [col for col in ("vts_method", "align_method", "n_subjects") if col in df.columns]
    if markers:
        mask = pd.Series(False, index=df.index)
        for col in markers:
            mask |= df[col].notna()
        return mask
    return ~valid_subject_mask(df)


def build_positions(nodes: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    for node in nodes:
        if node in ROI_POSITIONS:
            positions[node] = ROI_POSITIONS[node]

    unknown = [node for node in nodes if node not in positions]
    if unknown:
        radius = 1.35
        angles = np.linspace(0.0, 2.0 * math.pi, num=len(unknown), endpoint=False)
        for node, angle in zip(sorted(unknown), angles):
            positions[node] = (radius * math.cos(angle), radius * math.sin(angle))

    return positions


def _spread_curvatures(
    edge_rows: List[Tuple[str, str, float]],
    positions: Dict[str, Tuple[float, float]],
    default_rad: float = 0.25,
    min_sep: float = 0.16,
    bidir_rad: float = 0.38,
) -> Dict[Tuple[str, str], float]:
    """Assign a unique arc3 curvature to every edge so no two arcs visually overlap.

    Bidirectional pairs always get opposite signs with large separation.
    Unidirectional edges leaving the same source are sorted by destination
    angle and spread into a fan with guaranteed minimum curvature separation.
    Every edge gets a non-zero curvature so no arc is a straight line.
    """
    edge_set = {(s, d) for s, d, _ in edge_rows}
    curvatures: Dict[Tuple[str, str], float] = {}

    # Bidirectional pairs: large, strictly opposite curvature
    for src, dst, _ in edge_rows:
        if (dst, src) in edge_set and (src, dst) not in curvatures:
            sign = 1 if src < dst else -1
            curvatures[(src, dst)] =  sign * bidir_rad
            curvatures[(dst, src)] = -sign * bidir_rad

    # Unidirectional edges: group by source, sort by angle to destination
    by_src: Dict[str, list] = defaultdict(list)
    for src, dst, _ in edge_rows:
        if (src, dst) not in curvatures:
            sx, sy = positions[src]
            dx, dy = positions[dst]
            angle = math.atan2(dy - sy, dx - sx)
            by_src[src].append((angle, src, dst))

    for src, items in by_src.items():
        items_sorted = sorted(items)
        n = len(items_sorted)
        if n == 1:
            _, s, d = items_sorted[0]
            curvatures[(s, d)] = default_rad
        else:
            half = max(min_sep * (n - 1) / 2, 0.20)
            half = min(half, 0.55)
            rads = list(np.linspace(-half, half, n))
            # Ensure no curvature is near zero
            rads = [
                math.copysign(max(abs(r), 0.12), r if r != 0.0 else 1.0)
                for r in rads
            ]
            for (_, s, d), r in zip(items_sorted, rads):
                curvatures[(s, d)] = r

    return curvatures


def _shrink(size_pts2: float, mutation_scale: float) -> float:
    """Shrink in typographic points that clears the node circle and arrowhead."""
    return math.sqrt(size_pts2 / math.pi) + mutation_scale * 0.55 + 4


def _draw_legend(
    ax: plt.Axes,
    fig: plt.Figure,
    cbar,
    nodes: list,
    in_str: dict,
    out_str: dict,
    weighted: bool,
) -> None:
    """Draw node in/out legend and strength table aligned to the colorbar bottom."""
    header = f"{'node':<6}  {'in':>5}  {'out':>5}"
    sep    = "-" * len(header)
    rows   = [
        f"{_abbrev(n):<6}  {format_strength(in_str[n], weighted):>5}  {format_strength(out_str[n], weighted):>5}"
        for n in nodes
    ]
    text = "\n".join([header, sep] + rows)

    fig.canvas.draw()
    ax_pos   = ax.get_position()
    cbar_pos = cbar.ax.get_position()

    legend_x = ax_pos.x1 + 0.06
    legend_y = cbar_pos.y0 - 0.077

    fig.text(
        legend_x, legend_y, text,
        transform=fig.transFigure,
        fontsize=7.5, family="monospace", color="black",
        va="bottom", ha="right", multialignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#aaaaaa", linewidth=0.8, alpha=0.93),
        zorder=10,
    )


def format_strength(value: float, weighted: bool) -> str:
    if weighted:
        return f"{value:.2f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def footer_legend_text(weighted: bool) -> str:
    in_label = "incoming strength" if weighted else "incoming links"
    out_label = "outgoing strength" if weighted else "outgoing links"
    return f"node = ROI; in = {in_label}; out = {out_label}"


def draw_network(
    edges_df: pd.DataFrame,
    node_dfhat: Mapping[str, float],
    title: str,
    subtitle: str,
    out_path: Path,
    weighted: bool,
    dpi: int,
    edge_color: str,
    edge_alpha: float,
    edge_width_min: float,
    edge_width_max: float,
    arrow_size_min: float,
    arrow_size_max: float,
) -> None:
    nodes = sorted(
        set(edges_df["parent"]).union(set(edges_df["child"])).union(set(node_dfhat.keys()))
    )
    if not nodes:
        return

    positions = build_positions(nodes)

    in_strength = {node: 0.0 for node in nodes}
    out_strength = {node: 0.0 for node in nodes}
    for row in edges_df.itertuples(index=False):
        parent = str(row.parent)
        child = str(row.child)
        weight = float(row.weight)
        if parent not in out_strength:
            out_strength[parent] = 0.0
        if child not in in_strength:
            in_strength[child] = 0.0
        out_strength[parent] += weight
        in_strength[child] += weight

    total_strength = {node: in_strength.get(node, 0.0) + out_strength.get(node, 0.0) for node in nodes}
    max_strength = max(total_strength.values()) if total_strength else 1.0
    max_strength = max(max_strength, 1e-9)

    node_sizes = {node: 900.0 + 1800.0 * (total_strength[node] / max_strength) for node in nodes}
    node_values = [float(node_dfhat.get(node, np.nan)) for node in nodes]
    node_values = [0.5 if np.isnan(v) else float(v) for v in node_values]

    fig, ax = plt.subplots(figsize=(12.8, 9.2))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    edge_rows = [
        (str(r.parent), str(r.child), float(r.weight))
        for r in edges_df.itertuples(index=False)
        if str(r.parent) != str(r.child)
        and str(r.parent) in positions
        and str(r.child) in positions
    ]
    curvatures = _spread_curvatures(edge_rows, positions)
    max_w = max((w for _, _, w in edge_rows), default=1.0) or 1.0

    for parent, child, weight in edge_rows:
        intensity = min(max(weight / max_w, 0.0), 1.0)
        linewidth = float(edge_width_min) + (float(edge_width_max) - float(edge_width_min)) * intensity
        mutation_scale = float(arrow_size_min) + (float(arrow_size_max) - float(arrow_size_min)) * intensity
        rad = curvatures.get((parent, child), 0.25)

        arrow = FancyArrowPatch(
            positions[parent],
            positions[child],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=edge_color,
            alpha=float(edge_alpha),
            shrinkA=0,
            shrinkB=_shrink(node_sizes[child], mutation_scale),
            zorder=2,
        )
        ax.add_patch(arrow)

    xy = np.array([positions[node] for node in nodes], dtype=float)
    node_norm = Normalize(vmin=0.0, vmax=1.0)
    node_cmap = plt.get_cmap(config.CMAP_NETWORK)

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=[node_sizes[n] for n in nodes],
        c=node_values,
        cmap=node_cmap,
        norm=node_norm,
        edgecolors="white",
        linewidths=1.6,
        zorder=3,
    )

    for node in nodes:
        x, y = positions[node]
        ax.text(x, y, _abbrev(node), ha="center", va="center",
                fontsize=9, fontweight="bold", color="black", zorder=4)

    scalar = ScalarMappable(norm=node_norm, cmap=node_cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("df_hat", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    _draw_legend(ax, fig, cbar, nodes, in_strength, out_strength, weighted)

    ax.set_title(title, fontsize=13.5, pad=16, color="black")
    footer = f"{subtitle}\n{footer_legend_text(weighted)}"
    ax.text(0.01, 0.02, footer, transform=ax.transAxes,
            fontsize=9, color="black", ha="left", va="bottom")

    min_x, max_x = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    min_y, max_y = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    pad = 0.45
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot-mdmp] saved: {out_path}")


def plot_individual_networks(
    edges: pd.DataFrame,
    delta: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    edge_color: str,
    edge_alpha: float,
    edge_width_min: float,
    edge_width_max: float,
    arrow_size_min: float,
    arrow_size_max: float,
) -> int:
    if "subject" not in edges.columns:
        print("[plot-mdmp] skip individual: 'subject' column not found.")
        return 0

    key_cols = [col for col in CONTEXT_COLS if col in edges.columns and col != "group"]
    # Keep group in filenames/titles when available, but not as identity key.
    group_available = "group" in edges.columns
    count = 0

    grouped = edges.groupby(key_cols, dropna=False, sort=True, observed=True)
    for key, run_edges_raw in grouped:
        values = key if isinstance(key, tuple) else (key,)
        context = {col: val for col, val in zip(key_cols, values)}

        value_cols = [col for col in ("median_coef", "abs_median_coef") if col in run_edges_raw.columns]
        run_edges = (
            run_edges_raw[["parent", "child", *value_cols]]
            .dropna(subset=["parent", "child"])
            .drop_duplicates()
            .copy()
        )
        if run_edges.empty:
            continue
        if "abs_median_coef" in run_edges.columns:
            run_edges["weight_raw"] = pd.to_numeric(run_edges["abs_median_coef"], errors="coerce")
        elif "median_coef" in run_edges.columns:
            run_edges["weight_raw"] = pd.to_numeric(run_edges["median_coef"], errors="coerce").abs()
        else:
            run_edges["weight_raw"] = np.nan

        max_weight = float(run_edges["weight_raw"].max(skipna=True))
        if not np.isfinite(max_weight) or max_weight <= 0.0:
            run_edges["weight"] = 1.0
            weighted = False
            subtitle = "Edge weight = 1 (edge present in this subject/run)."
        else:
            run_edges["weight"] = run_edges["weight_raw"].fillna(0.0) / max_weight
            weighted = True
            subtitle = (
                "Edge weight = normalized absolute median of the smoothed dynamic "
                "coefficient for this subject/run."
            )

        run_delta = filter_by_context(delta, context)
        node_dfhat = (
            run_delta.groupby("node", as_index=False)["df_hat"].mean()
            if not run_delta.empty
            else pd.DataFrame(columns=["node", "df_hat"])
        )
        node_map = {
            str(row.node): float(row.df_hat)
            for row in node_dfhat.itertuples(index=False)
            if pd.notna(row.df_hat)
        }

        subject = context.get("subject", "NA")
        metric = context.get("metric", "unknown")
        session = context.get("session", "NA")
        state = context.get("visual_state", "NA")
        band = context.get("band", "NA")
        method = context.get("method", "mdmp")
        group = "Unknown"
        if group_available:
            group_rows = run_edges_raw["group"].dropna().astype(str)
            if not group_rows.empty:
                group = str(group_rows.iloc[0])

        title = (
            "MDMP individual network | "
            f"sub-{subject} | {group} | {session} | {state} | {band} | {metric} ({method})"
        )
        filename = (
            f"mdmp_sub-{slugify(subject)}_{slugify(group)}_{slugify(session)}_"
            f"{slugify(state)}_{slugify(band)}_{slugify(metric)}_{slugify(method)}.png"
        )
        out_path = out_dir / "individual" / slugify(metric) / slugify(band) / filename
        draw_network(
            edges_df=run_edges,
            node_dfhat=node_map,
            title=title,
            subtitle=subtitle,
            out_path=out_path,
            weighted=weighted,
            dpi=dpi,
            edge_color=edge_color,
            edge_alpha=edge_alpha,
            edge_width_min=edge_width_min,
            edge_width_max=edge_width_max,
            arrow_size_min=arrow_size_min,
            arrow_size_max=arrow_size_max,
        )
        count += 1

    return count


def plot_global_networks(
    edges: pd.DataFrame,
    delta: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    edge_color: str,
    edge_alpha: float,
    edge_width_min: float,
    edge_width_max: float,
    arrow_size_min: float,
    arrow_size_max: float,
) -> int:
    key_cols = [col for col in GLOBAL_CONTEXT_COLS if col in edges.columns]
    if not key_cols:
        print("[plot-mdmp] skip global: no global context columns found.")
        return 0

    count = 0
    grouped = edges.groupby(key_cols, dropna=False, sort=True, observed=True)
    for key, run_edges_raw in grouped:
        values = key if isinstance(key, tuple) else (key,)
        context = {col: val for col, val in zip(key_cols, values)}

        value_cols = [col for col in ("median_coef", "abs_median_coef") if col in run_edges_raw.columns]
        run_edges = (
            run_edges_raw[["parent", "child", *value_cols]]
            .dropna(subset=["parent", "child"])
            .drop_duplicates()
            .copy()
        )
        if run_edges.empty:
            continue
        if "abs_median_coef" in run_edges.columns:
            run_edges["weight_raw"] = pd.to_numeric(run_edges["abs_median_coef"], errors="coerce")
        elif "median_coef" in run_edges.columns:
            run_edges["weight_raw"] = pd.to_numeric(run_edges["median_coef"], errors="coerce").abs()
        else:
            run_edges["weight_raw"] = np.nan

        max_weight = float(run_edges["weight_raw"].max(skipna=True))
        if not np.isfinite(max_weight) or max_weight <= 0.0:
            run_edges["weight"] = 1.0
            weighted = False
            subtitle = "Edge weight = 1 (edge present in median VTS network)."
        else:
            run_edges["weight"] = run_edges["weight_raw"].fillna(0.0) / max_weight
            weighted = True
            subtitle = (
                "Structure recalculated on median VTS; edge width = normalized "
                "|median smoothed coefficient|."
            )

        run_delta = filter_by_context(delta, context)
        node_dfhat = (
            run_delta.groupby("node", as_index=False)["df_hat"].mean()
            if not run_delta.empty
            else pd.DataFrame(columns=["node", "df_hat"])
        )
        node_map = {
            str(row.node): float(row.df_hat)
            for row in node_dfhat.itertuples(index=False)
            if pd.notna(row.df_hat)
        }

        n_subjects = "NA"
        if "n_subjects" in run_edges_raw.columns:
            n_values = run_edges_raw["n_subjects"].dropna()
            if not n_values.empty:
                n_subjects = str(int(float(n_values.iloc[0])))
        if n_subjects != "NA":
            subtitle = f"{subtitle} n={n_subjects} subjects."

        metric = context.get("metric", "unknown")
        session = context.get("session", "NA")
        state = context.get("visual_state", "NA")
        band = context.get("band", "NA")
        method = context.get("method", "mdmp")
        group = context.get("group", "Unknown")

        title = (
            "MDMP global median network | "
            f"{group} | {session} | {state} | {band} | {metric} ({method})"
        )
        filename = (
            f"mdmp_median_{slugify(group)}_{slugify(session)}_{slugify(state)}_"
            f"{slugify(band)}_{slugify(metric)}_{slugify(method)}.png"
        )
        out_path = out_dir / "groups" / slugify(metric) / slugify(band) / filename
        draw_network(
            edges_df=run_edges[["parent", "child", "weight"]],
            node_dfhat=node_map,
            title=title,
            subtitle=subtitle,
            out_path=out_path,
            weighted=weighted,
            dpi=dpi,
            edge_color=edge_color,
            edge_alpha=edge_alpha,
            edge_width_min=edge_width_min,
            edge_width_max=edge_width_max,
            arrow_size_min=arrow_size_min,
            arrow_size_max=arrow_size_max,
        )
        count += 1

    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MDMP directed networks from CSVs generated by code/calc_mdmp.py."
        )
    )
    parser.add_argument(
        "--input-dirs",
        type=str,
        default="",
        help=(
            "Comma-separated MDMP directories. Each must contain mdmp_edges_long.csv "
            "and mdmp_delta_by_node.csv. If omitted, auto-detects results/mdmp "
            "recursively, including results/mdmp/power_rel and "
            "results/mdmp/individual/power_rel."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.PLOTS_DIR / "mdmp_networks",
        help="Output directory for individual network figures.",
    )
    _default_metrics = ",".join(config.MDMP_METRICS_TO_RUN)
    parser.add_argument(
        "--metrics",
        type=str,
        default=_default_metrics,
        help=f"Comma-separated metrics to plot (default: {_default_metrics}).",
    )
    parser.add_argument(
        "--metric",
        choices=VALID_METRICS,
        default=getattr(config, "MDMP_METRIC", "power_rel"),
        help="Single metric fallback used when --metrics is empty.",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip individual networks from mdmp_edges_long.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--edge-color",
        type=str,
        default="#000000",
        help="Edge/arrow color (matplotlib color).",
    )
    parser.add_argument(
        "--edge-alpha",
        type=float,
        default=0.75,
        help="Edge transparency (0-1).",
    )
    parser.add_argument(
        "--edge-width-min",
        type=float,
        default=0.45,
        help="Minimum edge width.",
    )
    parser.add_argument(
        "--edge-width-max",
        type=float,
        default=1.30,
        help="Maximum edge width.",
    )
    parser.add_argument(
        "--arrow-size-min",
        type=float,
        default=7.0,
        help="Minimum arrow head size.",
    )
    parser.add_argument(
        "--arrow-size-max",
        type=float,
        default=11.0,
        help="Maximum arrow head size.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    metrics = parse_metrics(raw_metrics=args.metrics, fallback_metric=args.metric)
    input_dirs = resolve_input_dirs(args.input_dirs)
    if not input_dirs:
        raise FileNotFoundError(
            "No input directories found. Pass --input-dirs with directories that contain "
            "mdmp_edges_long.csv and mdmp_delta_by_node.csv, or run code/calc_mdmp.py first."
        )

    edges, delta = load_mdmp_tables(input_dirs)
    edges, delta = clean_table_columns(edges, delta)
    edges = edges[edges["metric"].isin(metrics)].copy()
    delta = delta[delta["metric"].isin(metrics)].copy()

    if edges.empty:
        raise ValueError("No rows left after metric filtering. Check --metrics and MDMP CSVs.")

    edges = sort_for_reporting(edges)
    delta = sort_for_reporting(delta)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    available_metrics = sorted(edges["metric"].dropna().astype(str).unique().tolist())
    print(f"[plot-mdmp] metrics in scope: {available_metrics}")

    individual_edge_mask = valid_subject_mask(edges)
    individual_delta_mask = valid_subject_mask(delta)
    global_edge_mask = median_context_mask(edges)
    global_delta_mask = median_context_mask(delta)
    has_subject_rows = bool(individual_edge_mask.any())
    has_global_rows = bool(global_edge_mask.any())
    individual_n = 0
    global_n = 0

    if has_subject_rows and not args.skip_individual:
        individual_n = plot_individual_networks(
            edges=edges.loc[individual_edge_mask].copy(),
            delta=delta.loc[individual_delta_mask].copy(),
            out_dir=out_dir,
            dpi=int(args.dpi),
            edge_color=str(args.edge_color),
            edge_alpha=float(args.edge_alpha),
            edge_width_min=float(args.edge_width_min),
            edge_width_max=float(args.edge_width_max),
            arrow_size_min=float(args.arrow_size_min),
            arrow_size_max=float(args.arrow_size_max),
        )

    if has_global_rows:
        global_n = plot_global_networks(
            edges=edges.loc[global_edge_mask].copy(),
            delta=delta.loc[global_delta_mask].copy(),
            out_dir=out_dir,
            dpi=int(args.dpi),
            edge_color=str(args.edge_color),
            edge_alpha=float(args.edge_alpha),
            edge_width_min=float(args.edge_width_min),
            edge_width_max=float(args.edge_width_max),
            arrow_size_min=float(args.arrow_size_min),
            arrow_size_max=float(args.arrow_size_max),
        )

    print(f"[plot-mdmp] individual plots: {individual_n}")
    print(f"[plot-mdmp] global plots: {global_n}")
    print(f"[plot-mdmp] output root: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
