#!/usr/bin/env python3
"""Orchestrate the EEG pipeline from the repository root."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


class StepError(RuntimeError):
    """Raised when a pipeline step fails."""


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    group: str
    argv_factory: Callable[[Path, str], Optional[list[str]]]


def static_step(key: str, label: str, group: str, script_relpath: str, *extra_args: str) -> Step:
    script_path = Path(script_relpath)

    def _argv(root: Path, python_exe: str) -> list[str]:
        return [python_exe, str((root / script_path).resolve()), *extra_args]

    return Step(key=key, label=label, group=group, argv_factory=_argv)


def detect_mdmp_input_dirs(root: Path) -> list[Path]:
    """Return MDMP output directories that contain the expected CSVs."""
    base = root / "results" / "mdmp"
    found: list[Path] = []
    seen: set[Path] = set()

    if not base.exists():
        return found

    candidates = [base, *sorted(path for path in base.rglob("*") if path.is_dir())]
    for directory in candidates:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        edges = directory / "mdmp_edges_long.csv"
        delta = directory / "mdmp_delta_by_node.csv"
        if edges.exists() and delta.exists():
            found.append(resolved)
            seen.add(resolved)
    return found


def mdmp_network_step() -> Step:
    def _argv(root: Path, python_exe: str) -> Optional[list[str]]:
        input_dirs = detect_mdmp_input_dirs(root)
        if not input_dirs:
            return None
        csv_dirs = ",".join(str(path) for path in input_dirs)
        return [
            python_exe,
            str((root / "code" / "plot_mdmp_networks.py").resolve()),
            "--input-dirs",
            csv_dirs,
        ]

    return Step(
        key="mdmp_networks",
        label="Generate MDMP network plots",
        group="mdmp",
        argv_factory=_argv,
    )



def mdmp_heatmaps_step() -> Step:
    def _argv(root: Path, python_exe: str) -> Optional[list[str]]:
        input_dirs = detect_mdmp_input_dirs(root)
        input_csv = root / "results" / "timeseries" / "ts_power_long.csv"
        if not input_dirs and not input_csv.exists():
            return None
        argv = [
            python_exe,
            str((root / "code" / "plot_mdmp_heatmaps.py").resolve()),
        ]
        if input_dirs:
            argv.extend(["--input-dirs", ",".join(str(path) for path in input_dirs)])
        if input_csv.exists():
            argv.extend(["--input-csv", str(input_csv.resolve())])
        return argv

    return Step(
        key="mdmp_heatmaps",
        label="Generate MDMP static/dynamic heatmaps",
        group="mdmp",
        argv_factory=_argv,
    )


def build_steps() -> list[Step]:
    return [
        static_step("preprocess", "Preprocess EEG recordings", "core", "code/preprocess.py"),
        static_step("power", "Compute power tables", "core", "code/calc_power.py"),
        static_step("timeseries_csv", "Extract ROI x band time-series CSV", "core", "code/calc_timeseries_power.py"),
        static_step("timeseries", "Generate block-resolved time-series plots", "core", "code/plot_timeseries.py"),
        static_step("stats", "Compute paired statistics", "core", "code/calc_stats_power.py"),
        static_step("topomaps", "Generate EO/EC topomaps", "optional_plots", "code/plot_topomaps_grid.py"),
        static_step("mirror", "Generate mirror plots", "optional_plots", "code/plot_mirror.py"),
        static_step("mdmp", "Run MDMP inference", "mdmp", "code/calc_mdmp.py"),
        mdmp_network_step(),
        mdmp_heatmaps_step(),
    ]


def create_parser(steps: list[Step]) -> argparse.ArgumentParser:
    step_keys = [step.key for step in steps]
    parser = argparse.ArgumentParser(
        description="Run the Pyp-EEG workflow from the repository root.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only the core analytical steps and skip optional plots and MDMP.",
    )
    parser.add_argument(
        "--skip-optional-plots",
        action="store_true",
        help="Skip optional plotting scripts while keeping the core steps.",
    )
    parser.add_argument(
        "--no-mdmp",
        action="store_true",
        help="Skip MDMP inference and MDMP-derived plots.",
    )
    parser.add_argument(
        "--from-step",
        choices=step_keys,
        default=None,
        help="Start execution from the selected step.",
    )
    parser.add_argument(
        "--to-step",
        choices=step_keys,
        default=None,
        help="Stop execution after the selected step.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of step keys to run, in canonical order.",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated step keys to skip.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps even if a step fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without executing them.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List all available step keys and exit.",
    )
    return parser


def parse_csv_set(raw: Optional[str]) -> set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def filter_steps(steps: list[Step], args: argparse.Namespace) -> list[Step]:
    selected = steps

    if args.core_only:
        selected = [step for step in selected if step.group == "core"]
    else:
        if args.skip_optional_plots:
            selected = [step for step in selected if step.group != "optional_plots"]
        if args.no_mdmp:
            selected = [step for step in selected if step.group != "mdmp"]

    only_keys = parse_csv_set(args.only)
    if only_keys:
        unknown = only_keys.difference({step.key for step in steps})
        if unknown:
            raise ValueError(f"Unknown step key(s) in --only: {sorted(unknown)}")
        selected = [step for step in selected if step.key in only_keys]

    skip_keys = parse_csv_set(args.skip)
    if skip_keys:
        unknown = skip_keys.difference({step.key for step in steps})
        if unknown:
            raise ValueError(f"Unknown step key(s) in --skip: {sorted(unknown)}")
        selected = [step for step in selected if step.key not in skip_keys]

    if args.from_step:
        start_idx = next(i for i, step in enumerate(selected) if step.key == args.from_step)
        selected = selected[start_idx:]

    if args.to_step:
        end_idx = next(i for i, step in enumerate(selected) if step.key == args.to_step)
        selected = selected[: end_idx + 1]

    return selected


def prepare_runtime_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    cache_root = root / "results" / ".runtime_cache"
    mpl_dir = cache_root / "matplotlib"
    xdg_dir = cache_root / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    env.setdefault("XDG_CACHE_HOME", str(xdg_dir.resolve()))
    return env


def list_steps(steps: list[Step]) -> None:
    print("Available steps:")
    for step in steps:
        print(f"  {step.key:17s} [{step.group}] {step.label}")


def run_step(step: Step, index: int, total: int, root: Path, env: dict[str, str], dry_run: bool) -> None:
    argv = step.argv_factory(root, PYTHON)
    header = f"[{index}/{total}] {step.key} - {step.label}"

    if not argv:
        print(f"{header} -> skipped (required inputs not found)")
        return

    printable = " ".join(argv)
    print(header)
    print(f"  command: {printable}")

    if dry_run:
        return

    started = time.perf_counter()
    completed = subprocess.run(argv, cwd=root, env=env, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise StepError(f"Step '{step.key}' failed with exit code {completed.returncode}.")
    print(f"  finished in {elapsed:.1f}s")


def main() -> int:
    os.chdir(ROOT)
    steps = build_steps()
    parser = create_parser(steps)
    args = parser.parse_args()

    if args.list_steps:
        list_steps(steps)
        return 0

    try:
        selected_steps = filter_steps(steps, args)
    except (StopIteration, ValueError) as exc:
        parser.error(str(exc))

    if not selected_steps:
        print("No steps selected. Use --list-steps to inspect available keys.")
        return 0

    env = prepare_runtime_env(ROOT)
    print(f"Repository root: {ROOT}")
    print(f"Python executable: {PYTHON}")
    print(f"Selected steps: {[step.key for step in selected_steps]}")

    failures: list[str] = []
    total = len(selected_steps)
    for index, step in enumerate(selected_steps, start=1):
        try:
            run_step(step, index=index, total=total, root=ROOT, env=env, dry_run=args.dry_run)
        except StepError as exc:
            print(f"  ERROR: {exc}")
            failures.append(step.key)
            if not args.continue_on_error:
                return 1

    if failures:
        print(f"Completed with failures: {failures}")
        return 1

    print("Pipeline execution completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
