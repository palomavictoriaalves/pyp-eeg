"""Plot directed MDMP networks for individual runs and group-level summaries.

Reads one or more MDMP output directories containing:
  - mdmp_edges_long.csv
  - mdmp_delta_by_node.csv

Writes:
  results/plots/mdmp_networks/
    individual/<metric>/<band>/*.png
    group/<metric>/<band>/*.png
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
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


DEFAULT_INPUT_CANDIDATES = (
    config.RESULTS_DIR / "mdmp",
    config.RESULTS_DIR / "mdmp_rel",
    config.RESULTS_DIR / "mdmp_abs",
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
    "Prefrontal": (0.0, 1.00),
    "Frontal": (0.55, 0.70),
    "Frontocentral": (0.55, 0.25),
    "Central": (0.0, 0.10),
    "Temporo-parietal": (-0.95, -0.05),
    "Centro-parietal": (0.0, -0.30),
    "Parietal": (-0.55, -0.62),
    "Occipital": (0.0, -1.00),
}

CONTEXT_COLS = ("metric", "method", "subject", "group", "session", "visual_state", "band")
GROUP_CONTEXT_COLS = ("metric", "method", "group", "session", "visual_state", "band")
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
        return [Path(p).expanduser().resolve() for p in parse_csv_list(raw)]

    found = []
    for candidate in DEFAULT_INPUT_CANDIDATES:
        edges = candidate / "mdmp_edges_long.csv"
        delta = candidate / "mdmp_delta_by_node.csv"
        if edges.exists() and delta.exists():
            found.append(candidate.resolve())
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


def format_strength(value: float, weighted: bool) -> str:
    if weighted:
        return f"{value:.2f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


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

    node_sizes = [900.0 + 1800.0 * (total_strength[node] / max_strength) for node in nodes]
    node_values = [float(node_dfhat.get(node, np.nan)) for node in nodes]
    node_values = [0.5 if np.isnan(v) else float(v) for v in node_values]

    # Per-node shrink radius (points) = sqrt(area/π) + padding
    node_shrink = {
        node: math.sqrt(node_sizes[i] / math.pi) + 4
        for i, node in enumerate(nodes)
    }

    fig, ax = plt.subplots(figsize=(12.8, 9.2))
    ax.set_facecolor("#f4f6f8")

    edge_pairs = {(str(r.parent), str(r.child)) for r in edges_df.itertuples(index=False)}
    for row in edges_df.itertuples(index=False):
        parent = str(row.parent)
        child = str(row.child)
        weight = float(row.weight)
        if parent == child:
            continue
        if parent not in positions or child not in positions:
            continue

        has_reverse = (child, parent) in edge_pairs
        if has_reverse:
            rad = 0.25 if parent < child else -0.25
        else:
            rad = 0.03

        intensity = min(max(weight, 0.0), 1.0)
        linewidth = float(edge_width_min) + (float(edge_width_max) - float(edge_width_min)) * intensity
        mutation_scale = float(arrow_size_min) + (float(arrow_size_max) - float(arrow_size_min)) * intensity

        arrow = FancyArrowPatch(
            positions[parent],
            positions[child],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=edge_color,
            alpha=float(edge_alpha),
            shrinkA=node_shrink.get(parent, 18),
            shrinkB=node_shrink.get(child, 18),
            zorder=1,
        )
        ax.add_patch(arrow)

    xy = np.array([positions[node] for node in nodes], dtype=float)
    node_norm = Normalize(vmin=0.0, vmax=1.0)
    node_cmap = plt.get_cmap(config.CMAP_NETWORK)

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_sizes,
        c=node_values,
        cmap=node_cmap,
        norm=node_norm,
        edgecolors="white",
        linewidths=1.6,
        zorder=3,
    )

    for node in nodes:
        x, y = positions[node]
        in_txt = format_strength(in_strength.get(node, 0.0), weighted=weighted)
        out_txt = format_strength(out_strength.get(node, 0.0), weighted=weighted)
        ax.text(
            x,
            y,
            f"{node}\n(in={in_txt}, out={out_txt})",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#111827",
            zorder=4,
        )

    scalar = ScalarMappable(norm=node_norm, cmap=node_cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("df_hat")

    ax.set_title(title, fontsize=13.5, pad=16)
    ax.text(
        0.01,
        0.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
        ha="left",
        va="bottom",
    )

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

        run_edges = run_edges_raw[["parent", "child"]].dropna().drop_duplicates().copy()
        if run_edges.empty:
            continue
        run_edges["weight"] = 1.0

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
        subtitle = "Edge weight = 1 (edge present in this subject/run)."
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
            weighted=False,
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


def plot_group_networks(
    edges: pd.DataFrame,
    delta: pd.DataFrame,
    out_dir: Path,
    min_edge_freq: float,
    max_edges: int,
    dpi: int,
    edge_color: str,
    edge_alpha: float,
    edge_width_min: float,
    edge_width_max: float,
    arrow_size_min: float,
    arrow_size_max: float,
) -> int:
    if "group" not in edges.columns:
        print("[plot-mdmp] skip group: 'group' column not found.")
        return 0

    key_cols = [col for col in GROUP_CONTEXT_COLS if col in edges.columns]
    run_cols = [col for col in RUN_ID_COLS if col in edges.columns]
    count = 0

    grouped = edges.groupby(key_cols, dropna=False, sort=True, observed=True)
    for key, block in grouped:
        values = key if isinstance(key, tuple) else (key,)
        context = {col: val for col, val in zip(key_cols, values)}

        if run_cols:
            n_runs = int(block[run_cols].drop_duplicates().shape[0])
        else:
            n_runs = 1
        n_runs = max(n_runs, 1)

        keep_cols = ["parent", "child"] + run_cols
        edge_presence = block[keep_cols].dropna(subset=["parent", "child"]).drop_duplicates()
        edge_counts = (
            edge_presence.groupby(["parent", "child"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        if edge_counts.empty:
            continue
        edge_counts["weight"] = edge_counts["count"] / float(n_runs)
        edge_counts = edge_counts.sort_values("weight", ascending=False).reset_index(drop=True)

        selected = edge_counts.loc[edge_counts["weight"] >= float(min_edge_freq)].copy()
        if selected.empty:
            selected = edge_counts.head(min(len(edge_counts), int(max_edges))).copy()
        if len(selected) > int(max_edges):
            selected = selected.head(int(max_edges)).copy()

        group_delta = filter_by_context(delta, context)
        node_dfhat = (
            group_delta.groupby("node", as_index=False)["df_hat"].mean()
            if not group_delta.empty
            else pd.DataFrame(columns=["node", "df_hat"])
        )
        node_map = {
            str(row.node): float(row.df_hat)
            for row in node_dfhat.itertuples(index=False)
            if pd.notna(row.df_hat)
        }

        metric = context.get("metric", "unknown")
        group = context.get("group", "Unknown")
        session = context.get("session", "NA")
        state = context.get("visual_state", "NA")
        band = context.get("band", "NA")
        method = context.get("method", "mdmp")

        title = (
            "MDMP group network | "
            f"{group} | {session} | {state} | {band} | {metric} ({method})"
        )
        subtitle = (
            f"Edge weight = frequency across subjects/runs (n={n_runs}); "
            f"plot threshold >= {float(min_edge_freq):.2f}."
        )
        filename = (
            f"mdmp_group-{slugify(group)}_{slugify(session)}_{slugify(state)}_"
            f"{slugify(band)}_{slugify(metric)}_{slugify(method)}.png"
        )
        out_path = out_dir / "group" / slugify(metric) / slugify(band) / filename
        draw_network(
            edges_df=selected[["parent", "child", "weight"]],
            node_dfhat=node_map,
            title=title,
            subtitle=subtitle,
            out_path=out_path,
            weighted=True,
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
            "Plot MDMP directed networks (individual + group) from mdmp_edges_long.csv "
            "and mdmp_delta_by_node.csv."
        )
    )
    parser.add_argument(
        "--input-dirs",
        type=str,
        default="",
        help=(
            "Comma-separated MDMP directories. Each must contain mdmp_edges_long.csv "
            "and mdmp_delta_by_node.csv. If omitted, auto-detects results/mdmp, "
            "results/mdmp_rel, results/mdmp_abs."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.PLOTS_DIR / "mdmp_networks",
        help="Output directory for network figures.",
    )
    _default_metrics = ",".join(config.MDMP_METRICS_TO_RUN)
    parser.add_argument(
        "--metrics",
        type=str,
        default=_default_metrics,
        help=f"Comma-separated metrics to plot (default: {_default_metrics}).",
    )
    parser.add_argument(
        "--min-group-edge-freq",
        type=float,
        default=0.30,
        help=(
            "Minimum edge frequency for group plots (0-1). "
            "If no edge survives, top edges are kept."
        ),
    )
    parser.add_argument(
        "--max-group-edges",
        type=int,
        default=24,
        help="Maximum number of edges per group-level graph.",
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

    input_dirs = resolve_input_dirs(args.input_dirs)
    if not input_dirs:
        raise FileNotFoundError(
            "No input directories found. Pass --input-dirs with directories that contain "
            "mdmp_edges_long.csv and mdmp_delta_by_node.csv."
        )

    edges, delta = load_mdmp_tables(input_dirs)
    edges, delta = clean_table_columns(edges, delta)

    requested_metrics = parse_csv_list(args.metrics)
    if requested_metrics:
        edges = edges[edges["metric"].isin(requested_metrics)].copy()
        delta = delta[delta["metric"].isin(requested_metrics)].copy()

    if edges.empty:
        raise ValueError(
            "No rows left after metric filtering. Check --metrics and MDMP source CSVs."
        )

    edges = sort_for_reporting(edges)
    delta = sort_for_reporting(delta)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    available_metrics = sorted(edges["metric"].dropna().astype(str).unique().tolist())
    print(f"[plot-mdmp] metrics in scope: {available_metrics}")

    individual_n = plot_individual_networks(
        edges=edges,
        delta=delta,
        out_dir=out_dir,
        dpi=int(args.dpi),
        edge_color=str(args.edge_color),
        edge_alpha=float(args.edge_alpha),
        edge_width_min=float(args.edge_width_min),
        edge_width_max=float(args.edge_width_max),
        arrow_size_min=float(args.arrow_size_min),
        arrow_size_max=float(args.arrow_size_max),
    )
    group_n = plot_group_networks(
        edges=edges,
        delta=delta,
        out_dir=out_dir,
        min_edge_freq=float(args.min_group_edge_freq),
        max_edges=int(args.max_group_edges),
        dpi=int(args.dpi),
        edge_color=str(args.edge_color),
        edge_alpha=float(args.edge_alpha),
        edge_width_min=float(args.edge_width_min),
        edge_width_max=float(args.edge_width_max),
        arrow_size_min=float(args.arrow_size_min),
        arrow_size_max=float(args.arrow_size_max),
    )

    print(f"[plot-mdmp] individual plots: {individual_n}")
    print(f"[plot-mdmp] group plots: {group_n}")
    print(f"[plot-mdmp] output root: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
