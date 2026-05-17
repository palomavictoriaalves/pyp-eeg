"""Plot MDMP static and dynamic heatmaps.

Reads one or more MDMP output directories containing mdmp_edges_long.csv and
writes one ROI x ROI adjacency matrix per subject/session/visual state/band.
Optionally refits MDM models to produce dynamic heatmap GIFs and static frame
panels from the time-series CSV.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

import config
from calc_mdmp import (
    DEFAULT_GROUP_COLS,
    DEFAULT_NODE_COL,
    VALID_METRICS,
    normalize_reason,
    parse_csv_columns,
    parse_delta_grid,
    parse_metrics,
    prepare_run_matrix,
    resolve_time_col,
)
from plot_mdmp_networks import (
    METRIC_ORDER,
    RUN_ID_COLS,
    clean_table_columns,
    load_mdmp_tables,
    parse_csv_list,
    resolve_input_dirs,
    slugify,
)

try:
    from mdmp import MDM, plot_idag
except Exception as exc:
    MDM = None
    plot_idag = None
    MDM_IMPORT_ERROR = exc
else:
    MDM_IMPORT_ERROR = None


OUT_DIR = config.PLOTS_DIR / "mdmp_heatmaps"
NODES_ORDER = list(getattr(config, "ROIS_ORDER", []))
NODE_ABBREV = {
    "Prefrontal": "PFr",
    "Frontal": "Fr",
    "Frontocentral": "FCe",
    "Central": "Ce",
    "Temporo-parietal": "TP",
    "Centro-parietal": "CP",
    "Parietal": "Pa",
    "Occipital": "Oc",
}
SUBJECT_TO_GROUP = {
    **{str(s).zfill(2): "Active" for s in getattr(config, "GROUP_ACTIVE", set())},
    **{str(s).zfill(2): "Passive" for s in getattr(config, "GROUP_PASSIVE", set())},
    **{str(s).zfill(2): "Control" for s in getattr(config, "GROUP_CONTROL", set())},
}


def sort_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "metric" in out.columns:
        out["metric"] = pd.Categorical(out["metric"], METRIC_ORDER, ordered=True)
    if "session" in out.columns:
        out["session"] = pd.Categorical(out["session"], config.SESS_ORDER, ordered=True)
    if "visual_state" in out.columns:
        out["visual_state"] = pd.Categorical(out["visual_state"], config.VS_ORDER, ordered=True)
    if "band" in out.columns and getattr(config, "BANDS_ORDER", None):
        out["band"] = pd.Categorical(out["band"], config.BANDS_ORDER, ordered=True)
    return out.sort_values([c for c in RUN_ID_COLS if c in out.columns]).reset_index(drop=True)


def make_matrix(block: pd.DataFrame, nodes: Sequence[str], value_col: str) -> np.ndarray:
    idx = {node: i for i, node in enumerate(nodes)}
    mat = np.zeros((len(nodes), len(nodes)), dtype=float)
    for row in block.itertuples(index=False):
        parent = str(row.parent)
        child = str(row.child)
        if parent not in idx or child not in idx or parent == child:
            continue
        if value_col == "edge":
            value = 1.0
        else:
            value = getattr(row, value_col, np.nan)
            value = float(value) if pd.notna(value) else np.nan
        mat[idx[parent], idx[child]] = value
    np.fill_diagonal(mat, np.nan)
    return mat


def render_heatmap(
    block: pd.DataFrame,
    context: Mapping[str, object],
    nodes: Sequence[str],
    value_col: str,
    cmap: str,
    annotate: bool,
    dpi: int,
    out_dir: Path,
) -> Path:
    mat = make_matrix(block, nodes, value_col=value_col)
    abbrevs = [NODE_ABBREV.get(node, node[:4]) for node in nodes]

    if value_col == "edge":
        vmin, vmax = 0.0, 1.0
        color_label = "Edge present"
        title_value = "binary adjacency"
    else:
        finite = np.abs(mat[np.isfinite(mat)])
        vmax = float(finite.max()) if finite.size else 1.0
        vmax = max(vmax, 1e-9)
        vmin = -vmax if value_col == "median_coef" else 0.0
        color_label = value_col
        title_value = value_col.replace("_", " ")

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(mat, aspect="equal", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(nodes)))
    ax.set_yticks(range(len(nodes)))
    ax.set_xticklabels(abbrevs, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(abbrevs, fontsize=9)
    ax.set_xlabel("Child node (target)", fontsize=10)
    ax.set_ylabel("Parent node (source)", fontsize=10)

    if annotate:
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j or not np.isfinite(mat[i, j]) or mat[i, j] == 0:
                    continue
                label = f"{mat[i, j]:.2f}" if value_col != "edge" else "1"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")

    subject = context.get("subject", "NA")
    subject_text = str(subject).strip()
    is_global = "subject" not in context or pd.isna(subject) or subject_text in {"", "NA", "nan", "None"}
    group = context.get("group", "Unknown")
    session = context.get("session", "NA")
    state = context.get("visual_state", "NA")
    band = context.get("band", "NA")
    metric = context.get("metric", "unknown")
    method = context.get("method", "mdmp")

    if is_global:
        ax.set_title(
            "MDMP global median adjacency heatmap | "
            f"{group} | {session} | {state} | {band} | {metric} ({method})\n"
            f"Cell value = {title_value}",
            fontsize=11,
            pad=12,
        )
    else:
        ax.set_title(
            "MDMP individual adjacency heatmap | "
            f"sub-{subject} | {group} | {session} | {state} | {band} | {metric} ({method})\n"
            f"Cell value = {title_value}",
            fontsize=11,
            pad=12,
        )
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(color_label)

    if is_global:
        filename = (
            f"mdmp_heat_median_{slugify(group)}_{slugify(session)}_"
            f"{slugify(state)}_{slugify(band)}_{slugify(metric)}_{slugify(method)}.png"
        )
        out_path = out_dir / "groups" / slugify(metric) / slugify(band) / filename
    else:
        filename = (
            f"mdmp_heat_sub-{slugify(subject)}_{slugify(group)}_{slugify(session)}_"
            f"{slugify(state)}_{slugify(band)}_{slugify(metric)}_{slugify(method)}.png"
        )
        out_path = out_dir / "individual" / slugify(metric) / slugify(band) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_param_matrix(
    model: Any,
    t: int,
    distribution: str = "smoo",
) -> np.ndarray:
    """Build an N x N dynamic parameter matrix at timepoint t."""
    if distribution == "smoo":
        mt_list = model.Smoo["smt"]
    else:
        mt_list = model.Filt["mt"]

    n = model.adj_mat.shape[0]
    mat = np.zeros((n, n), dtype=float)

    for node in range(n):
        mt_node = mt_list[node]
        if mt_node.ndim == 1:
            mt_node = mt_node.reshape(1, -1)
        if t >= mt_node.shape[1]:
            continue

        mat[node, node] = float(mt_node[0, t])
        parents = np.where(model.adj_mat[:, node] > 0)[0]
        for p_offset, parent in enumerate(parents):
            param_idx = p_offset + 1
            if param_idx < mt_node.shape[0]:
                mat[parent, node] = float(mt_node[param_idx, t])

    return mat


def frame_indices_from_config(
    frame_range: Sequence[int],
    frame_count: int,
    total_timepoints: int,
) -> List[int]:
    """Return integer frame indices, inclusive over configured range."""
    if total_timepoints <= 0:
        return []
    if len(frame_range) != 2:
        raise ValueError("frame range must contain exactly two values: (start, end)")

    start = max(int(frame_range[0]), 0)
    end = min(int(frame_range[1]), total_timepoints - 1)
    if end < start:
        return []

    count = max(int(frame_count), 1)
    available = end - start + 1
    if count >= available:
        return list(range(start, end + 1))

    values = np.linspace(start, end, num=count)
    indices = [int(round(v)) for v in values]
    deduped: List[int] = []
    for idx in indices:
        idx = min(max(idx, start), end)
        if idx not in deduped:
            deduped.append(idx)
    return deduped


def render_frame_panel(
    model: Any,
    node_names: Sequence[str],
    frame_indices: Sequence[int],
    subject: str,
    group: str,
    session: str,
    state: str,
    band: str,
    metric: str,
    method: str,
    nbf: int,
    distribution: str,
    cmap: str,
    dpi: int,
    out_path: Path,
) -> None:
    """Render a configurable static panel of dynamic heatmap frames."""
    frames = [build_param_matrix(model, t, distribution=distribution) for t in frame_indices]
    if not frames:
        return

    abbrevs = [NODE_ABBREV.get(n, n[:4]) for n in node_names]
    all_vals = np.concatenate([f.flatten() for f in frames])
    finite_vals = all_vals[np.isfinite(all_vals)]
    abs_max = float(np.abs(finite_vals).max()) if finite_vals.size else 1.0
    abs_max = max(abs_max, 1e-9)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    n_cols = int(getattr(config, "MDMP_HEATMAP_FRAME_COLUMNS", 5))
    n_cols = max(n_cols, 1)
    n_rows = int(np.ceil(len(frames) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.6, n_rows * 2.6 + 1.0),
        squeeze=False,
    )

    for ax in axes.flatten():
        ax.set_visible(False)

    for idx, (mat, t) in enumerate(zip(frames, frame_indices)):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        ax.set_visible(True)
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal", origin="upper")
        n = len(node_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(abbrevs, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticklabels(abbrevs, fontsize=6.5)
        ax.set_title(f"t = {t} s", fontsize=8, pad=4)
        if row == n_rows - 1:
            ax.set_xlabel("Child", fontsize=7)
        if col == 0:
            ax.set_ylabel("Parent", fontsize=7)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, fraction=0.02)
    cbar.set_label("Smoothed coefficient", fontsize=9)

    dist_label = "smoothed" if distribution == "smoo" else "filtered"
    fig.suptitle(
        f"MDM dynamic heatmap | sub-{subject} | {group} | {session} | {state} | {band} | {metric} ({method})\n"
        f"{dist_label} posterior estimates; diagonal = intercept; off-diagonal = connection coefficient\n"
        f"(burn-in nbf={nbf}; frames {min(frame_indices)}-{max(frame_indices)})",
        fontsize=9,
        y=1.01,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [frames] saved: {out_path}")


def plot_static_heatmaps(
    input_dirs: Sequence[Path],
    out_dir: Path,
    metrics: Sequence[str],
    bands: Optional[Sequence[str]],
    value_col: str,
    cmap: str,
    annotate: bool,
    dpi: int,
) -> int:
    if not input_dirs:
        raise FileNotFoundError("No MDMP input directories found.")

    edges, delta = load_mdmp_tables(input_dirs)
    edges, _ = clean_table_columns(edges, delta)

    if metrics:
        edges = edges[edges["metric"].isin(metrics)].copy()
    if bands:
        edges = edges[edges["band"].isin(bands)].copy()
    if edges.empty:
        raise ValueError("No MDMP edge rows left after filtering.")
    if value_col != "edge" and value_col not in edges.columns:
        raise ValueError(
            f"Column '{value_col}' was not found. Re-run code/calc_mdmp.py "
            "with the updated pipeline to export median coefficients."
        )

    edges = sort_key(edges)
    key_order = (
        "subject",
        "group",
        "session",
        "visual_state",
        "band",
        "metric",
        "method",
        "vts_method",
        "align_method",
    )
    key_cols = [col for col in key_order if col in edges.columns]

    count = 0
    for key, block in edges.groupby(key_cols, dropna=False, sort=True, observed=True):
        values = key if isinstance(key, tuple) else (key,)
        context = {col: value for col, value in zip(key_cols, values)}
        present_nodes = set(block["parent"].dropna().astype(str)) | set(block["child"].dropna().astype(str))
        nodes = [node for node in NODES_ORDER if node in present_nodes]
        nodes.extend(sorted(present_nodes.difference(nodes)))
        if not nodes:
            continue
        out_path = render_heatmap(
            block=block,
            context=context,
            nodes=nodes,
            value_col=value_col,
            cmap=cmap,
            annotate=bool(annotate),
            dpi=int(dpi),
            out_dir=out_dir,
        )
        print(f"[plot-mdmp-heatmaps] saved: {out_path}")
        count += 1

    print(f"[plot-mdmp-heatmaps] static plots: {count}")
    print(f"[plot-mdmp-heatmaps] static output root: {out_dir}")
    return count


def plot_dynamic_heatmaps(
    input_csv: Path,
    output_dir: Path,
    metric_col: str,
    group_cols: Sequence[str],
    node_col: str,
    time_col: str,
    method: str,
    min_t: int,
    min_nodes: int,
    nbf: int,
    delta: Optional[np.ndarray],
    max_runs: Optional[int],
    frame_range: Sequence[int],
    frame_count: int,
    distribution: str,
    gif_fps: int,
    gif_dpi: int,
    frame_dpi: int,
    cmap: str,
) -> int:
    if MDM is None or plot_idag is None:
        raise ImportError(
            "Could not import the local 'mdmp' API with MDM and plot_idag. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from MDM_IMPORT_ERROR

    gifs_root = output_dir / slugify(metric_col) / "gifs"
    frames_root = output_dir / slugify(metric_col) / "frames"
    gifs_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    required = set(group_cols) | {node_col, metric_col, time_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in input CSV: {missing}")

    min_t_effective = max(int(min_t), int(nbf) + 1)
    grouped = df.groupby(list(group_cols), dropna=False, sort=True)
    total = int(grouped.ngroups)
    print(
        f"[plot-mdmp-heatmaps] dynamic metric={metric_col} runs={total} "
        f"min_t={min_t_effective} min_nodes={min_nodes}"
    )

    n_ok = n_skip = n_fail = 0
    for run_idx, (group_key, run_df) in enumerate(grouped, start=1):
        if max_runs is not None and run_idx > max_runs:
            break

        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        context: Dict[str, Any] = {col: val for col, val in zip(group_cols, key_tuple)}
        subject = str(context.get("subject", "NA")).strip()
        if subject.endswith(".0"):
            subject = subject[:-2]
        if subject.isdigit():
            subject = subject.zfill(2)

        group = SUBJECT_TO_GROUP.get(subject, "Unknown")
        session = str(context.get("session", "NA")).upper()
        state = str(context.get("visual_state", "NA")).upper()
        band = str(context.get("band", "NA"))
        print(f"[plot-mdmp-heatmaps] dynamic run {run_idx}/{total}: sub-{subject} | {group} | {session} | {state} | {band}")

        try:
            wide, info = prepare_run_matrix(
                run_df=run_df,
                time_col=time_col,
                node_col=node_col,
                metric_col=metric_col,
            )
        except Exception as exc:
            print(f"  [fail] prepare_matrix: {normalize_reason(exc)}")
            n_fail += 1
            continue

        if info["n_timepoints"] < min_t_effective:
            print(f"  [skip] insufficient_timepoints: {info['n_timepoints']} < {min_t_effective}")
            n_skip += 1
            continue
        if info["n_nodes_ready"] < int(min_nodes):
            print(f"  [skip] insufficient_nodes: {info['n_nodes_ready']} < {int(min_nodes)}")
            n_skip += 1
            continue

        t0 = time.perf_counter()
        try:
            model = MDM(
                wide,
                method=method,
                nbf=int(nbf),
                delta=delta,
                verbose=False,
            )
        except Exception as exc:
            print(f"  [fail] fit_error: {normalize_reason(exc)}")
            n_fail += 1
            continue

        elapsed = round(time.perf_counter() - t0, 2)
        node_names = list(wide.columns)
        print(
            f"  [ok] nodes={info['n_nodes_ready']} timepoints={info['n_timepoints']} "
            f"edges={int(np.sum(model.adj_mat))} fit={elapsed}s"
        )

        stem = (
            f"mdmp_dynamic_sub-{slugify(subject)}_{slugify(group)}_"
            f"{slugify(session)}_{slugify(state)}_{slugify(band)}_{slugify(metric_col)}_{slugify(method)}"
        )
        band_slug = slugify(band)

        gif_path = gifs_root / band_slug / f"{stem}.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plot_idag(
                mdm_object=model,
                output_gif=str(gif_path),
                fps=gif_fps,
                width=6,
                height=6,
                dpi=gif_dpi,
                distribution=distribution,
            )
            plt.close("all")
            print(f"  [gif] saved: {gif_path}")
        except Exception as exc:
            print(f"  [gif-warn] {normalize_reason(exc)}")

        indices = frame_indices_from_config(frame_range, frame_count, total_timepoints=wide.shape[0])
        frames_path = frames_root / band_slug / f"{stem}_frames.png"
        render_frame_panel(
            model=model,
            node_names=node_names,
            frame_indices=indices,
            subject=subject,
            group=group,
            session=session,
            state=state,
            band=band,
            metric=metric_col,
            method=method,
            nbf=nbf,
            distribution=distribution,
            cmap=cmap,
            dpi=frame_dpi,
            out_path=frames_path,
        )

        n_ok += 1

    print(f"[plot-mdmp-heatmaps] dynamic done: ok={n_ok} skipped={n_skip} failed={n_fail}")
    return n_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot MDMP static adjacency heatmaps and optional dynamic GIF/frame heatmaps.")
    parser.add_argument("--input-dirs", default="", help="Comma-separated MDMP output directories.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--metrics", default=",".join(config.MDMP_METRICS_TO_RUN))
    parser.add_argument("--metric", choices=VALID_METRICS, default=getattr(config, "MDMP_METRIC", "power_rel"))
    parser.add_argument("--bands", nargs="*", default=None)
    parser.add_argument("--value-col", choices=("edge", "median_coef", "abs_median_coef"), default="edge")
    parser.add_argument("--cmap", default=config.CMAP_ADJACENCY)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--input-csv", type=Path, default=getattr(config, "MDMP_INPUT_CSV", config.TS_DIR / "ts_power_long.csv"))
    parser.add_argument("--group-cols", type=str, default=",".join(str(v) for v in getattr(config, "MDMP_GROUP_COLS", DEFAULT_GROUP_COLS)))
    parser.add_argument("--node-col", default=getattr(config, "MDMP_NODE_COL", DEFAULT_NODE_COL))
    parser.add_argument("--time-col", default=getattr(config, "MDMP_TIME_COL", ""))
    parser.add_argument("--method", default=getattr(config, "MDMP_METHOD", "hc"))
    parser.add_argument("--min-t", type=int, default=int(getattr(config, "MDMP_MIN_T", 20)))
    parser.add_argument("--min-nodes", type=int, default=int(getattr(config, "MDMP_MIN_NODES", 3)))
    parser.add_argument("--nbf", type=int, default=int(getattr(config, "MDMP_NBF", 15)))
    parser.add_argument("--delta-grid", default="")
    parser.add_argument("--max-runs", type=int, default=getattr(config, "MDMP_MAX_RUNS", None))
    parser.add_argument("--dynamic-output-dir", type=Path, default=getattr(config, "MDMP_HEATMAP_DYNAMIC_OUTPUT_DIR", OUT_DIR / "dynamic"))
    parser.add_argument("--plot-dynamic", action="store_true", default=None, dest="dynamic_enabled")
    parser.add_argument("--no-dynamic", action="store_false", dest="dynamic_enabled")
    parser.add_argument("--plot-gifs", action="store_true", dest="dynamic_enabled", help=argparse.SUPPRESS)
    parser.add_argument("--no-gifs", action="store_false", dest="dynamic_enabled", help=argparse.SUPPRESS)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--frame-range", nargs=2, type=int, default=None, metavar=("START", "END"))
    parser.add_argument("--distribution", choices=("filt", "smoo"), default=getattr(config, "MDMP_HEATMAP_DYNAMIC_DISTRIBUTION", "smoo"))
    parser.add_argument("--gif-fps", type=int, default=int(getattr(config, "MDMP_HEATMAP_GIF_FPS", 10)))
    parser.add_argument("--gif-dpi", type=int, default=int(getattr(config, "MDMP_HEATMAP_GIF_DPI", 100)))
    parser.add_argument("--frame-dpi", type=int, default=int(getattr(config, "MDMP_HEATMAP_FRAME_DPI", 250)))
    parser.add_argument("--dynamic-cmap", default=getattr(config, "MDMP_HEATMAP_DYNAMIC_CMAP", "RdBu_r"))
    args = parser.parse_args()

    metrics = parse_metrics(raw_metrics=args.metrics, fallback_metric=args.metric)

    if not args.skip_static:
        plot_static_heatmaps(
            input_dirs=resolve_input_dirs(args.input_dirs),
            out_dir=args.out_dir.resolve(),
            metrics=metrics,
            bands=args.bands,
            value_col=args.value_col,
            cmap=args.cmap,
            annotate=bool(args.annotate),
            dpi=int(args.dpi),
        )

    dynamic_enabled = (
        bool(getattr(config, "MDMP_HEATMAP_DYNAMIC_ENABLED", False))
        if args.dynamic_enabled is None
        else bool(args.dynamic_enabled)
    )

    if dynamic_enabled:
        input_csv = args.input_csv.resolve()
        if not input_csv.exists():
            raise FileNotFoundError(f"input CSV not found: {input_csv}")
        preview = pd.read_csv(input_csv, nrows=5)
        time_col = resolve_time_col(preview, explicit_time_col=args.time_col)
        group_cols = parse_csv_columns(args.group_cols)
        delta = parse_delta_grid(raw_grid=args.delta_grid, fallback=getattr(config, "MDMP_DELTA_GRID", ()))
        frame_range = (
            tuple(args.frame_range)
            if args.frame_range is not None
            else tuple(getattr(config, "MDMP_HEATMAP_FRAME_RANGE", (0, 9)))
        )
        frame_count = (
            int(args.frame_count)
            if args.frame_count is not None
            else int(getattr(config, "MDMP_HEATMAP_FRAME_COUNT", 10))
        )

        for metric in metrics:
            plot_dynamic_heatmaps(
                input_csv=input_csv,
                output_dir=args.dynamic_output_dir.resolve(),
                metric_col=metric,
                group_cols=group_cols,
                node_col=args.node_col,
                time_col=time_col,
                method=args.method,
                min_t=args.min_t,
                min_nodes=args.min_nodes,
                nbf=args.nbf,
                delta=delta,
                max_runs=args.max_runs,
                frame_range=frame_range,
                frame_count=frame_count,
                distribution=args.distribution,
                gif_fps=args.gif_fps,
                gif_dpi=args.gif_dpi,
                frame_dpi=args.frame_dpi,
                cmap=args.dynamic_cmap,
            )
    else:
        print("[plot-mdmp-heatmaps] dynamic disabled: MDMP_HEATMAP_DYNAMIC_ENABLED=False")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
