"""Learn dynamic MDMP networks from ROI power time series."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from mdmp import MDM
except Exception as exc:
    MDM = None
    MDM_IMPORT_ERROR = exc
else:
    MDM_IMPORT_ERROR = None

import config


DEFAULT_GROUP_COLS = ("subject", "session", "visual_state", "band")
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
        for parent_idx, child_idx in edges:
            edges_rows.append(
                {
                    **context,
                    "metric": metric_col,
                    "method": method,
                    "parent": node_names[int(parent_idx)],
                    "child": node_names[int(child_idx)],
                    "edge": 1,
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
    edges_cols = group_cols_list + ["group", "metric", "method", "parent", "child", "edge"]
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


def build_arg_parser() -> argparse.ArgumentParser:
    group_cols_cfg = getattr(config, "MDMP_GROUP_COLS", DEFAULT_GROUP_COLS)
    if isinstance(group_cols_cfg, (list, tuple)):
        group_cols_default = ",".join(str(v).strip() for v in group_cols_cfg if str(v).strip())
    else:
        group_cols_default = str(group_cols_cfg).strip() or ",".join(DEFAULT_GROUP_COLS)

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
        help="Directory for MDMP outputs when running a single metric.",
    )
    parser.add_argument(
        "--output-dir-rel",
        type=Path,
        default=getattr(config, "MDMP_OUTPUT_DIR_REL", config.RESULTS_DIR / "mdmp_rel"),
        help="Output directory for metric=power_rel when running multiple metrics.",
    )
    parser.add_argument(
        "--output-dir-abs",
        type=Path,
        default=getattr(config, "MDMP_OUTPUT_DIR_ABS", config.RESULTS_DIR / "mdmp_abs"),
        help="Output directory for metric=power_abs when running multiple metrics.",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        default=group_cols_default,
        help="Comma-separated grouping columns (default: subject,session,visual_state,band).",
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
        "--ignore-enabled-flag",
        action="store_true",
        default=bool(getattr(config, "MDMP_IGNORE_ENABLED_FLAG", False)),
        help="Run even when config.MDMP_ENABLED is False.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if MDM is None:
        raise ImportError(
            "Could not import 'mdmp'. Install it with "
            "'pip install mdmp' or "
            "'pip install git+https://github.com/maods2/mdmp.git'."
        ) from MDM_IMPORT_ERROR

    if not getattr(config, "MDMP_ENABLED", True) and not args.ignore_enabled_flag:
        print("MDMP is disabled by config.MDMP_ENABLED=False. Exiting without processing.")
        return 0

    group_cols = parse_csv_columns(args.group_cols)
    metrics = parse_metrics(raw_metrics=args.metrics, fallback_metric=args.metric)

    if len(metrics) > 1:
        output_by_metric = {
            "power_rel": args.output_dir_rel.resolve(),
            "power_abs": args.output_dir_abs.resolve(),
        }
        selected_paths = [output_by_metric[m] for m in metrics]
        if len(set(selected_paths)) != len(selected_paths):
            raise ValueError(
                "output-dir-rel and output-dir-abs must be different when running multiple metrics."
            )
    else:
        output_by_metric = {metrics[0]: args.output_dir.resolve()}

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

    for metric in metrics:
        metric_output = output_by_metric[metric]
        print(f"[mdmp] metric={metric} -> output_dir={metric_output}")
        run(
            input_csv=input_csv,
            output_dir=metric_output,
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
