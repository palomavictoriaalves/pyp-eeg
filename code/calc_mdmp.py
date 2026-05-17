"""Learn dynamic MDMP networks from ROI power time series."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import numpy as np
import pandas as pd

try:
    from mdmp import MDM, compute_vts
except Exception as exc:
    MDM = None
    compute_vts = None
    MDM_IMPORT_ERROR = exc
else:
    MDM_IMPORT_ERROR = None

import config


DEFAULT_GROUP_COLS = ("subject", "session", "visual_state", "band")
DEFAULT_GLOBAL_GROUP_COLS = ("group", "session", "visual_state", "band")
DEFAULT_NODE_COL = "region"
TIME_CANDIDATES = ("t_sec", "time_s")
VALID_METRICS = ("power_rel", "power_abs")


def parse_csv_columns(value: str) -> List[str]:
    cols = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not cols:
        raise ValueError("group columns cannot be empty")
    return cols


def parse_metrics(raw_metrics: str, fallback_metric: str) -> List[str]:
    if raw_metrics:
        metrics = [chunk.strip() for chunk in raw_metrics.split(",") if chunk.strip()]
    else:
        metrics = [str(fallback_metric).strip()] if str(fallback_metric).strip() else []

    if not metrics:
        raise ValueError("at least one metric must be provided")

    normalized: List[str] = []
    seen = set()
    for metric in metrics:
        if metric not in VALID_METRICS:
            raise ValueError(f"invalid metric '{metric}'. Valid options: {VALID_METRICS}")
        if metric not in seen:
            normalized.append(metric)
            seen.add(metric)
    return normalized


def parse_delta_grid(raw_grid: str, fallback: Sequence[float]) -> Optional[np.ndarray]:
    if raw_grid:
        pieces = [piece.strip() for piece in raw_grid.split(",") if piece.strip()]
        values = [float(piece) for piece in pieces]
    else:
        values = [float(v) for v in fallback] if fallback else []

    if not values:
        return None

    delta = np.array(values, dtype=float)
    if np.any(delta < 0.0) or np.any(delta > 1.0):
        raise ValueError("all delta values must be between 0 and 1")

    return np.sort(np.unique(delta))


def normalize_reason(error: Exception, max_len: int = 250) -> str:
    text = str(error).strip() or error.__class__.__name__
    return text[:max_len]


def extract_edge_medians(model: Any, node_names: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """Return median smoothed coefficient for each learned parent -> child edge."""
    medians: Dict[Tuple[str, str], float] = {}
    smoothed = getattr(model, "Smoo", {}) or {}
    filtered = getattr(model, "Filt", {}) or {}
    smt = smoothed.get("smt", {}) if isinstance(smoothed, dict) else {}
    row_names = filtered.get("row_names", {}) if isinstance(filtered, dict) else {}

    for child_idx, child_name in enumerate(node_names):
        values = smt.get(child_idx)
        names = row_names.get(child_idx, [])
        if values is None or not names:
            continue

        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)

        for param_idx, param_name in enumerate(names):
            if param_idx >= values.shape[0] or "->" not in str(param_name):
                continue
            parent, child = str(param_name).split("->", 1)
            if child != str(child_name):
                child = str(child_name)
            series = values[param_idx, :]
            finite = series[np.isfinite(series)]
            medians[(parent, child)] = float(np.median(finite)) if finite.size else np.nan

    return medians


def resolve_time_col(df: pd.DataFrame, explicit_time_col: str) -> str:
    if explicit_time_col:
        if explicit_time_col not in df.columns:
            raise ValueError(f"time column '{explicit_time_col}' was not found in input CSV")
        return explicit_time_col

    for candidate in TIME_CANDIDATES:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"no time column found. Expected one of {TIME_CANDIDATES}; "
        "or pass --time-col explicitly."
    )



def prepare_run_matrix(
    run_df: pd.DataFrame,
    time_col: str,
    node_col: str,
    metric_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    base = run_df[[time_col, node_col, metric_col]].copy()
    base[time_col] = pd.to_numeric(base[time_col], errors="coerce")
    base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce")
    base[node_col] = base[node_col].astype(str).str.strip()
    base = base.dropna(subset=[time_col, node_col])
    base = base[base[node_col] != ""]

    grouped = (
        base.groupby([time_col, node_col], as_index=False, dropna=False)[metric_col]
        .mean()
        .sort_values([time_col, node_col])
    )
    wide = grouped.pivot(index=time_col, columns=node_col, values=metric_col).sort_index()
    wide.columns.name = None

    n_nodes_raw = int(wide.shape[1])
    if n_nodes_raw > 0:
        wide = wide.dropna(axis=1, how="all")
        wide = wide.interpolate(method="linear", axis=0, limit_direction="both")
        wide = wide.ffill().bfill()
        wide = wide.replace([np.inf, -np.inf], np.nan)
        wide = wide.dropna(axis=1, how="any")
        wide = wide.dropna(axis=0, how="any")

    info = {
        "n_rows_in": int(len(run_df)),
        "n_rows_used": int(len(grouped)),
        "n_timepoints": int(wide.shape[0]),
        "n_nodes_raw": n_nodes_raw,
        "n_nodes_ready": int(wide.shape[1]),
    }
    return wide, info


def context_to_text(context: Dict[str, Any], group_cols: Sequence[str]) -> str:
    return ", ".join(f"{col}={context.get(col)}" for col in group_cols)


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


def node_order_for_wides(wides: Sequence[pd.DataFrame], node_order: Sequence[str]) -> List[str]:
    common = set(wides[0].columns)
    for wide in wides[1:]:
        common &= set(wide.columns)
    ordered = [node for node in node_order if node in common]
    ordered.extend(sorted(common.difference(ordered)))
    return ordered


def unique_columns(columns: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for col in columns:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def run(
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
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"input CSV is empty: {input_csv}")

    required_cols = set(group_cols) | {node_col, metric_col, time_col}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in input: {missing}")

    runs_summary: List[Dict[str, Any]] = []
    edges_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    grouped_runs = df.groupby(list(group_cols), dropna=False, sort=True)
    total_runs = int(grouped_runs.ngroups)
    min_t_effective = max(int(min_t), int(nbf) + 1)
    print(
        f"[mdmp] input rows={len(df)}, runs={total_runs}, metric={metric_col}, "
        f"min_t={min_t_effective}, min_nodes={min_nodes}"
    )

    for run_idx, (group_key, run_df) in enumerate(grouped_runs, start=1):
        if max_runs is not None and run_idx > max_runs:
            break

        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        context = {col: value for col, value in zip(group_cols, key_tuple)}
        context["group"] = (
            str(run_df["group"].dropna().iloc[0])
            if "group" in run_df.columns and not run_df["group"].dropna().empty
            else "Unknown"
        )
        summary = {
            **context,
            "metric": metric_col,
            "method": method,
            "time_col": time_col,
            "n_rows_in": int(len(run_df)),
            "n_rows_used": 0,
            "n_timepoints": 0,
            "n_nodes_raw": 0,
            "n_nodes_ready": 0,
            "n_edges": 0,
            "fit_seconds": np.nan,
            "status": "",
            "reason": "",
        }

        context_str = context_to_text(context, group_cols)
        print(f"[mdmp] run {run_idx}/{total_runs}: {context_str}")

        try:
            wide, info = prepare_run_matrix(
                run_df=run_df,
                time_col=time_col,
                node_col=node_col,
                metric_col=metric_col,
            )
            summary.update(info)
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"prepare_matrix_error: {normalize_reason(exc)}"
            runs_summary.append(summary)
            print(f"  [fail] {summary['reason']}")
            continue

        if summary["n_timepoints"] < min_t_effective:
            summary["status"] = "skipped"
            summary["reason"] = (
                f"insufficient_timepoints: {summary['n_timepoints']} < {min_t_effective}"
            )
            runs_summary.append(summary)
            print(f"  [skip] {summary['reason']}")
            continue

        if summary["n_nodes_ready"] < int(min_nodes):
            summary["status"] = "skipped"
            summary["reason"] = (
                f"insufficient_nodes: {summary['n_nodes_ready']} < {int(min_nodes)}"
            )
            runs_summary.append(summary)
            print(f"  [skip] {summary['reason']}")
            continue

        fit_start = time.perf_counter()
        try:
            assert MDM is not None
            model = MDM(
                wide,
                method=method,
                nbf=int(nbf),
                delta=delta,
                verbose=False,
                show_progress=False,
            )
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"fit_error: {normalize_reason(exc)}"
            summary["fit_seconds"] = round(time.perf_counter() - fit_start, 4)
            runs_summary.append(summary)
            print(f"  [fail] {summary['reason']}")
            continue

        fit_seconds = round(time.perf_counter() - fit_start, 4)
        node_names = list(wide.columns)
        adj = np.asarray(model.adj_mat, dtype=int)
        if adj.shape != (len(node_names), len(node_names)):
            summary["status"] = "failed"
            summary["reason"] = (
                f"adjacency_shape_mismatch: got {adj.shape}, expected "
                f"({len(node_names)}, {len(node_names)})"
            )
            summary["fit_seconds"] = fit_seconds
            runs_summary.append(summary)
            print(f"  [fail] {summary['reason']}")
            continue

        edges = np.argwhere(adj == 1)
        edge_medians = extract_edge_medians(model, node_names)
        for parent_idx, child_idx in edges:
            parent_name = node_names[int(parent_idx)]
            child_name = node_names[int(child_idx)]
            median_coef = edge_medians.get((parent_name, child_name), np.nan)
            edges_rows.append(
                {
                    **context,
                    "metric": metric_col,
                    "method": method,
                    "parent": parent_name,
                    "child": child_name,
                    "edge": 1,
                    "median_coef": median_coef,
                    "abs_median_coef": abs(median_coef) if np.isfinite(median_coef) else np.nan,
                }
            )

        df_hat = np.asarray(model.DF.get("DF_hat", []), dtype=float).reshape(-1)
        for idx, node_name in enumerate(node_names):
            delta_rows.append(
                {
                    **context,
                    "metric": metric_col,
                    "method": method,
                    "node": node_name,
                    "df_hat": float(df_hat[idx]) if idx < len(df_hat) else np.nan,
                }
            )

        summary["status"] = "success"
        summary["reason"] = ""
        summary["n_edges"] = int(len(edges))
        summary["fit_seconds"] = fit_seconds
        runs_summary.append(summary)
        print(
            f"  [ok] nodes={summary['n_nodes_ready']} "
            f"timepoints={summary['n_timepoints']} edges={summary['n_edges']}"
        )

    group_cols_list = list(group_cols)
    summary_cols = group_cols_list + [
        "group",
        "metric",
        "method",
        "time_col",
        "n_rows_in",
        "n_rows_used",
        "n_timepoints",
        "n_nodes_raw",
        "n_nodes_ready",
        "n_edges",
        "fit_seconds",
        "status",
        "reason",
    ]
    edges_cols = group_cols_list + [
        "group",
        "metric",
        "method",
        "parent",
        "child",
        "edge",
        "median_coef",
        "abs_median_coef",
    ]
    delta_cols = group_cols_list + ["group", "metric", "method", "node", "df_hat"]

    summary_df = pd.DataFrame(runs_summary, columns=summary_cols)
    edges_df = pd.DataFrame(edges_rows, columns=edges_cols)
    delta_df = pd.DataFrame(delta_rows, columns=delta_cols)
    skipped_df = summary_df.loc[summary_df["status"] == "skipped"].copy()

    out_summary = output_dir / "mdmp_runs_summary.csv"
    out_edges = output_dir / "mdmp_edges_long.csv"
    out_delta = output_dir / "mdmp_delta_by_node.csv"
    out_skipped = output_dir / "mdmp_skipped_runs.csv"

    summary_df.to_csv(out_summary, index=False)
    edges_df.to_csv(out_edges, index=False)
    delta_df.to_csv(out_delta, index=False)
    skipped_df.to_csv(out_skipped, index=False)

    print(f"[mdmp] summary: {out_summary}")
    print(f"[mdmp] edges:   {out_edges}")
    print(f"[mdmp] deltas:  {out_delta}")
    print(f"[mdmp] skipped: {out_skipped}")
    print(
        f"[mdmp] runs={len(summary_df)}, "
        f"success={(summary_df['status'] == 'success').sum()}, "
        f"skipped={(summary_df['status'] == 'skipped').sum()}, "
        f"failed={(summary_df['status'] == 'failed').sum()}"
    )
    return 0


def fit_global_median_context(
    context: Dict[str, Any],
    block: pd.DataFrame,
    metric_col: str,
    time_col: str,
    node_col: str,
    method: str,
    min_t: int,
    min_nodes: int,
    min_subjects: int,
    nbf: int,
    delta: Optional[np.ndarray],
    align_method: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], List[str]]:
    summary: Dict[str, Any] = {
        **context,
        "metric": metric_col,
        "method": method,
        "vts_method": "median",
        "align_method": align_method,
        "time_col": time_col,
        "n_rows_in": int(len(block)),
        "n_subjects_available": 0,
        "n_subjects_used": 0,
        "n_subjects_skipped": 0,
        "n_timepoints": 0,
        "n_nodes_ready": 0,
        "n_edges": 0,
        "fit_seconds": np.nan,
        "status": "",
        "reason": "",
    }
    edges_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    vts_df: Optional[pd.DataFrame] = None

    min_t_effective = max(int(min_t), int(nbf) + 1)
    subject_wides: List[pd.DataFrame] = []
    subject_ids: List[str] = []
    skipped_subjects: List[str] = []

    for subject, subject_df in block.groupby("subject", dropna=False, sort=True):
        summary["n_subjects_available"] += 1
        subject_id = normalize_subject(subject)
        try:
            wide, info = prepare_run_matrix(
                run_df=subject_df,
                time_col=time_col,
                node_col=node_col,
                metric_col=metric_col,
            )
        except Exception:
            skipped_subjects.append(subject_id)
            continue
        if info["n_timepoints"] < min_t_effective or info["n_nodes_ready"] < int(min_nodes):
            skipped_subjects.append(subject_id)
            continue
        subject_wides.append(wide)
        subject_ids.append(subject_id)

    summary["n_subjects_used"] = len(subject_wides)
    summary["n_subjects_skipped"] = len(skipped_subjects)

    if len(subject_wides) < int(min_subjects):
        summary["status"] = "skipped"
        summary["reason"] = f"insufficient_subjects: {len(subject_wides)} < {int(min_subjects)}"
        return summary, pd.DataFrame(), pd.DataFrame(), None, subject_ids

    nodes = node_order_for_wides(subject_wides, getattr(config, "ROIS_ORDER", []))
    if len(nodes) < int(min_nodes):
        summary["status"] = "skipped"
        summary["reason"] = f"insufficient_common_nodes: {len(nodes)} < {int(min_nodes)}"
        return summary, pd.DataFrame(), pd.DataFrame(), None, subject_ids

    aligned_inputs = [wide[nodes].copy() for wide in subject_wides]
    try:
        assert compute_vts is not None
        vts_result = compute_vts(aligned_inputs, method="median", align_method=align_method)
        vts_df = pd.DataFrame(np.asarray(vts_result.vts_data, dtype=float), columns=nodes)
        vts_df = vts_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    except Exception as exc:
        summary["status"] = "failed"
        summary["reason"] = f"vts_error: {normalize_reason(exc)}"
        return summary, pd.DataFrame(), pd.DataFrame(), None, subject_ids

    summary["n_timepoints"] = int(vts_df.shape[0])
    summary["n_nodes_ready"] = int(vts_df.shape[1])
    if summary["n_timepoints"] < min_t_effective:
        summary["status"] = "skipped"
        summary["reason"] = f"insufficient_vts_timepoints: {summary['n_timepoints']} < {min_t_effective}"
        return summary, pd.DataFrame(), pd.DataFrame(), vts_df, subject_ids

    fit_start = time.perf_counter()
    try:
        assert MDM is not None
        model = MDM(
            vts_df,
            method=method,
            nbf=int(nbf),
            delta=delta,
            verbose=False,
            show_progress=False,
        )
    except Exception as exc:
        summary["status"] = "failed"
        summary["reason"] = f"fit_error: {normalize_reason(exc)}"
        summary["fit_seconds"] = round(time.perf_counter() - fit_start, 4)
        return summary, pd.DataFrame(), pd.DataFrame(), vts_df, subject_ids

    summary["fit_seconds"] = round(time.perf_counter() - fit_start, 4)
    adj = np.asarray(model.adj_mat, dtype=int)
    if adj.shape != (len(nodes), len(nodes)):
        summary["status"] = "failed"
        summary["reason"] = (
            f"adjacency_shape_mismatch: got {adj.shape}, expected ({len(nodes)}, {len(nodes)})"
        )
        return summary, pd.DataFrame(), pd.DataFrame(), vts_df, subject_ids

    edge_medians = extract_edge_medians(model, nodes)
    for parent_idx, child_idx in np.argwhere(adj == 1):
        parent = nodes[int(parent_idx)]
        child = nodes[int(child_idx)]
        median_coef = edge_medians.get((parent, child), np.nan)
        edges_rows.append(
            {
                **context,
                "metric": metric_col,
                "method": method,
                "vts_method": "median",
                "align_method": align_method,
                "n_subjects": len(subject_wides),
                "subjects": ";".join(subject_ids),
                "parent": parent,
                "child": child,
                "edge": 1,
                "median_coef": median_coef,
                "abs_median_coef": abs(median_coef) if np.isfinite(median_coef) else np.nan,
            }
        )

    df_hat = np.asarray(model.DF.get("DF_hat", []), dtype=float).reshape(-1)
    for idx, node_name in enumerate(nodes):
        delta_rows.append(
            {
                **context,
                "metric": metric_col,
                "method": method,
                "vts_method": "median",
                "align_method": align_method,
                "n_subjects": len(subject_wides),
                "node": node_name,
                "df_hat": float(df_hat[idx]) if idx < len(df_hat) else np.nan,
            }
        )

    summary["status"] = "success"
    summary["n_edges"] = len(edges_rows)
    return summary, pd.DataFrame(edges_rows), pd.DataFrame(delta_rows), vts_df, subject_ids


def run_global_median(
    input_csv: Path,
    output_dir: Path,
    metric_col: str,
    group_cols: Sequence[str],
    node_col: str,
    time_col: str,
    method: str,
    min_t: int,
    min_nodes: int,
    min_subjects: int,
    nbf: int,
    delta: Optional[np.ndarray],
    align_method: str,
    max_runs: Optional[int],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"input CSV is empty: {input_csv}")

    required_cols = set(group_cols) | {"subject", node_col, metric_col, time_col}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in input: {missing}")

    summaries: List[Dict[str, Any]] = []
    edges_frames: List[pd.DataFrame] = []
    delta_frames: List[pd.DataFrame] = []
    vts_meta_rows: List[Dict[str, Any]] = []

    grouped = df.groupby(list(group_cols), dropna=False, sort=True)
    total = int(grouped.ngroups)
    min_t_effective = max(int(min_t), int(nbf) + 1)
    print(
        f"[mdmp] global median input rows={len(df)}, contexts={total}, metric={metric_col}, "
        f"min_subjects={min_subjects}, min_t={min_t_effective}, min_nodes={min_nodes}"
    )

    for idx, (group_key, block) in enumerate(grouped, start=1):
        if max_runs is not None and idx > max_runs:
            break

        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        context = {col: value for col, value in zip(group_cols, key_tuple)}
        if "group" in context:
            context["group"] = normalize_group(context["group"])
        if "session" in context:
            context["session"] = str(context["session"]).upper()
        if "visual_state" in context:
            context["visual_state"] = str(context["visual_state"]).upper()

        print(f"[mdmp] global run {idx}/{total}: {context_to_text(context, group_cols)}")
        summary, edges_df, delta_df, vts_df, subject_ids = fit_global_median_context(
            context=context,
            block=block,
            metric_col=metric_col,
            time_col=time_col,
            node_col=node_col,
            method=method,
            min_t=min_t,
            min_nodes=min_nodes,
            min_subjects=min_subjects,
            nbf=nbf,
            delta=delta,
            align_method=align_method,
        )
        summaries.append(summary)
        if not edges_df.empty:
            edges_frames.append(edges_df)
        if not delta_df.empty:
            delta_frames.append(delta_df)
        if vts_df is not None:
            vts_meta_rows.append(
                {
                    **context,
                    "metric": metric_col,
                    "method": method,
                    "vts_method": "median",
                    "align_method": align_method,
                    "n_subjects": len(subject_ids),
                    "subjects": ";".join(subject_ids),
                    "n_timepoints": int(vts_df.shape[0]),
                    "n_nodes": int(vts_df.shape[1]),
                    "nodes": ";".join(str(c) for c in vts_df.columns),
                }
            )

        if summary["status"] == "success":
            print(
                f"  [ok] subjects={summary['n_subjects_used']} "
                f"timepoints={summary['n_timepoints']} nodes={summary['n_nodes_ready']} "
                f"edges={summary['n_edges']}"
            )
        else:
            print(f"  [{summary['status']}] {summary['reason']}")

    summary_cols = unique_columns(
        [
            *group_cols,
            "metric",
            "method",
            "vts_method",
            "align_method",
            "time_col",
            "n_rows_in",
            "n_subjects_available",
            "n_subjects_used",
            "n_subjects_skipped",
            "n_timepoints",
            "n_nodes_ready",
            "n_edges",
            "fit_seconds",
            "status",
            "reason",
        ]
    )
    edges_cols = unique_columns(
        [
            *group_cols,
            "metric",
            "method",
            "vts_method",
            "align_method",
            "n_subjects",
            "subjects",
            "parent",
            "child",
            "edge",
            "median_coef",
            "abs_median_coef",
        ]
    )
    delta_cols = unique_columns(
        [
            *group_cols,
            "metric",
            "method",
            "vts_method",
            "align_method",
            "n_subjects",
            "node",
            "df_hat",
        ]
    )
    vts_cols = unique_columns(
        [
            *group_cols,
            "metric",
            "method",
            "vts_method",
            "align_method",
            "n_subjects",
            "subjects",
            "n_timepoints",
            "n_nodes",
            "nodes",
        ]
    )

    summary_df = pd.DataFrame(summaries, columns=summary_cols)
    edges_df = (
        pd.concat(edges_frames, ignore_index=True)
        if edges_frames
        else pd.DataFrame(columns=edges_cols)
    )
    delta_df = (
        pd.concat(delta_frames, ignore_index=True)
        if delta_frames
        else pd.DataFrame(columns=delta_cols)
    )
    vts_df = pd.DataFrame(vts_meta_rows, columns=vts_cols)
    skipped_df = summary_df.loc[summary_df["status"] == "skipped"].copy()

    out_summary = output_dir / "mdmp_runs_summary.csv"
    out_edges = output_dir / "mdmp_edges_long.csv"
    out_delta = output_dir / "mdmp_delta_by_node.csv"
    out_skipped = output_dir / "mdmp_skipped_runs.csv"
    out_vts = output_dir / "mdmp_vts_metadata.csv"

    summary_df.to_csv(out_summary, index=False)
    edges_df.to_csv(out_edges, index=False)
    delta_df.to_csv(out_delta, index=False)
    skipped_df.to_csv(out_skipped, index=False)
    vts_df.to_csv(out_vts, index=False)

    print(f"[mdmp] summary: {out_summary}")
    print(f"[mdmp] edges:   {out_edges}")
    print(f"[mdmp] deltas:  {out_delta}")
    print(f"[mdmp] vts:     {out_vts}")
    print(f"[mdmp] skipped: {out_skipped}")
    print(
        f"[mdmp] contexts={len(summary_df)}, "
        f"success={(summary_df['status'] == 'success').sum() if not summary_df.empty else 0}, "
        f"skipped={(summary_df['status'] == 'skipped').sum() if not summary_df.empty else 0}, "
        f"failed={(summary_df['status'] == 'failed').sum() if not summary_df.empty else 0}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    group_cols_cfg = getattr(config, "MDMP_GROUP_COLS", DEFAULT_GROUP_COLS)
    if isinstance(group_cols_cfg, (list, tuple)):
        group_cols_default = ",".join(str(v).strip() for v in group_cols_cfg if str(v).strip())
    else:
        group_cols_default = str(group_cols_cfg).strip() or ",".join(DEFAULT_GROUP_COLS)

    global_group_cols_cfg = getattr(config, "MDMP_GLOBAL_GROUP_COLS", DEFAULT_GLOBAL_GROUP_COLS)
    if isinstance(global_group_cols_cfg, (list, tuple)):
        global_group_cols_default = ",".join(
            str(v).strip() for v in global_group_cols_cfg if str(v).strip()
        )
    else:
        global_group_cols_default = (
            str(global_group_cols_cfg).strip() or ",".join(DEFAULT_GLOBAL_GROUP_COLS)
        )

    metrics_cfg = getattr(config, "MDMP_METRICS_TO_RUN", ())
    if isinstance(metrics_cfg, str):
        metrics_default = metrics_cfg.strip()
    elif isinstance(metrics_cfg, (list, tuple)):
        metrics_default = ",".join(str(v).strip() for v in metrics_cfg if str(v).strip())
    else:
        metrics_default = ""

    parser = argparse.ArgumentParser(
        description=(
            "Run MDMP: Bayesian Network Modeling for Dynamic Multivariate "
            "Time Series from results/timeseries/ts_power_long.csv"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=getattr(config, "MDMP_INPUT_CSV", config.TS_DIR / "ts_power_long.csv"),
        help="Input long-format time-series CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=getattr(config, "MDMP_OUTPUT_DIR", config.RESULTS_DIR / "mdmp"),
        help="Base directory for MDMP outputs. Metrics are written under <output-dir>/<metric>/.",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        default=group_cols_default,
        help="Legacy individual-run grouping columns.",
    )
    parser.add_argument(
        "--global-group-cols",
        type=str,
        default=global_group_cols_default,
        help="Comma-separated grouping columns for median VTS outputs.",
    )
    parser.add_argument(
        "--node-col",
        type=str,
        default=getattr(config, "MDMP_NODE_COL", DEFAULT_NODE_COL),
        help="Column used as graph nodes.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=getattr(config, "MDMP_TIME_COL", ""),
        help="Time column name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=metrics_default,
        help=(
            "Comma-separated metrics to run (power_rel,power_abs). "
            "If empty, falls back to --metric."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=VALID_METRICS,
        default=getattr(config, "MDMP_METRIC", "power_rel"),
        help="Legacy single metric fallback used when --metrics is empty.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=getattr(config, "MDMP_METHOD", "hc"),
        help="MDMP structure learning method.",
    )
    parser.add_argument(
        "--min-t",
        type=int,
        default=int(getattr(config, "MDMP_MIN_T", 20)),
        help="Minimum number of time points per grouped run.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=int(getattr(config, "MDMP_MIN_NODES", 3)),
        help="Minimum number of nodes per grouped run.",
    )
    parser.add_argument(
        "--nbf",
        type=int,
        default=int(getattr(config, "MDMP_NBF", 15)),
        help="Burn-in used by mdmp.MDM.",
    )
    parser.add_argument(
        "--delta-grid",
        type=str,
        default="",
        help="Comma-separated discount factors. Defaults to config.MDMP_DELTA_GRID.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=getattr(config, "MDMP_MAX_RUNS", None),
        help="Optional cap for number of grouped runs (debug/smoke test).",
    )
    parser.add_argument(
        "--min-subjects",
        type=int,
        default=int(getattr(config, "MDMP_MIN_SUBJECTS", 2)),
        help="Minimum subjects required for each median VTS context.",
    )
    parser.add_argument(
        "--align-method",
        choices=("truncate", "interpolate"),
        default=getattr(config, "MDMP_ALIGN_METHOD", "truncate"),
        help="Alignment method passed to mdmp.compute_vts.",
    )
    parser.add_argument(
        "--ignore-enabled-flag",
        action="store_true",
        default=bool(getattr(config, "MDMP_IGNORE_ENABLED_FLAG", False)),
        help="Run even when config.MDMP_ENABLED is False.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if MDM is None or compute_vts is None:
        raise ImportError(
            "Could not import the local 'mdmp' API with MDM and compute_vts. "
            "Install dependencies with 'pip install -r requirements.txt'."
        ) from MDM_IMPORT_ERROR

    if not getattr(config, "MDMP_ENABLED", True) and not args.ignore_enabled_flag:
        print("MDMP is disabled by config.MDMP_ENABLED=False. Exiting without processing.")
        return 0

    group_cols = parse_csv_columns(args.group_cols)
    global_group_cols = parse_csv_columns(args.global_group_cols)
    metrics = parse_metrics(raw_metrics=args.metrics, fallback_metric=args.metric)

    delta = parse_delta_grid(
        raw_grid=args.delta_grid,
        fallback=getattr(config, "MDMP_DELTA_GRID", ()),
    )
    if str(args.method).lower() == "hc":
        print(
            "[mdmp] note: method=hc uses internal continuous local delta optimization "
            "for DAG search; --delta-grid primarily affects DF_hat/filter outputs."
        )

    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(
            f"input CSV not found: {input_csv}. "
            "Run 'python code/calc_timeseries_power.py' first."
        )

    preview_df = pd.read_csv(input_csv, nrows=5)
    time_col = resolve_time_col(preview_df, explicit_time_col=args.time_col)
    output_base = args.output_dir.resolve()

    for metric in metrics:
        metric_output = output_base / "groups" / metric
        print(f"[mdmp] metric={metric} -> group_median_output_dir={metric_output}")
        run_global_median(
            input_csv=input_csv,
            output_dir=metric_output,
            metric_col=metric,
            group_cols=global_group_cols,
            node_col=args.node_col,
            time_col=time_col,
            method=args.method,
            min_t=int(args.min_t),
            min_nodes=int(args.min_nodes),
            min_subjects=int(args.min_subjects),
            nbf=int(args.nbf),
            delta=delta,
            align_method=args.align_method,
            max_runs=args.max_runs,
        )

        individual_output = output_base / "individual" / metric
        print(f"[mdmp] metric={metric} -> individual_output_dir={individual_output}")
        run(
            input_csv=input_csv,
            output_dir=individual_output,
            metric_col=metric,
            group_cols=group_cols,
            node_col=args.node_col,
            time_col=time_col,
            method=args.method,
            min_t=int(args.min_t),
            min_nodes=int(args.min_nodes),
            nbf=int(args.nbf),
            delta=delta,
            max_runs=args.max_runs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
