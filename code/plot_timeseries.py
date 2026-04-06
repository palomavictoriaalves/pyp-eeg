"""
Time-series analysis of the four experimental conditions (EO block 1, EC block 1,
EO block 2, EC block 2) by band x ROI, split by session (PRE/POST) and group.

Loads *_desc-preproc_clean_raw.fif, reconstructs the four blocks from
annotations (visual_state:EO / visual_state:EC in order of occurrence),
computes sliding-window power, and generates a 4 x 3 grid
(conditions x groups) with mean +- 95% CI.
Exports a backup CSV and one PNG figure per band x metric x ROI.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import t, wilcoxon

import config

# Paths
PROCESSED_DIR = config.PROCESSED_DIR
PLOTS_DIR     = config.PLOTS_DIR
OUT_DIR       = PLOTS_DIR / "timeseries"
CSV_DIR       = OUT_DIR / "csv"
FIG_DIR       = OUT_DIR / "figs"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
BANDS         = config.BANDS
PSD_FMIN      = config.PSD_FMIN
PSD_FMAX      = config.PSD_FMAX
WELCH_SEG_SEC = config.WELCH_SEG_SEC
WELCH_OVERLAP = config.WELCH_OVERLAP
GROUPS_ORDER  = config.GROUPS_ORDER
ROI_CHANNELS  = config.ROI_CHANNELS
TS_WIN_SEC    = config.TS_WIN_SEC
TS_STEP_SEC   = config.TS_STEP_SEC
ABS_SCALE     = float(getattr(config, "POWER_ABS_SCALE", 1.0))
if not np.isfinite(ABS_SCALE) or ABS_SCALE <= 0:
    ABS_SCALE = 1.0
ALPHA_FDR     = config.TS_FDR_ALPHA
MARK_SIG      = config.TS_MARK_SIG
GENERATE_PLOTS = config.TS_GENERATE_PLOTS

GROUP_ACTIVE  = set(config.GROUP_ACTIVE)
GROUP_PASSIVE = set(config.GROUP_PASSIVE)
GROUP_CONTROL = set(config.GROUP_CONTROL)

# Order and labels for the four conditions
CONDITION_ORDER = ["EO_1", "EC_1", "EO_2", "EC_2"]
CONDITION_LABELS = {
    "EO_1": "EO - Block 1",
    "EC_1": "EC - Block 1",
    "EO_2": "EO - Block 2",
    "EC_2": "EC - Block 2",
}

PALETTE_SESS = config.PALETTE_SESS   # {"PRE": "#72B2E7", "POST": "#F28E2B"}

# Original recording onset for each condition, derived from BLOCKS_WITH_STATE.
# Used to restore each block to its original recording-time position.
_state_counter: dict = {}
BLOCK_ORIGINAL_ONSET: dict = {}
for _state, (_t0, _t1) in config.BLOCKS_WITH_STATE:
    _state_counter[_state] = _state_counter.get(_state, 0) + 1
    BLOCK_ORIGINAL_ONSET[f"{_state}_{_state_counter[_state]}"] = float(_t0)
# Example: {"EO_1": 15.0, "EC_1": 150.0, "EO_2": 285.0, "EC_2": 420.0}

TIME_WINDOWS = getattr(config, "TS_FIXED_X_WINDOWS", {})
GENERATE_CONTINUOUS = bool(getattr(config, "TS4_GENERATE_CONTINUOUS_PLOTS", True))
GENERATE_EO_COMBINED = bool(getattr(config, "TS4_GENERATE_EO_COMBINED_PLOTS", True))

CONDITION_SPANS = []
_state_counter_spans: dict = {}
for _state, (_t0, _t1) in config.BLOCKS_WITH_STATE:
    _state_counter_spans[_state] = _state_counter_spans.get(_state, 0) + 1
    _cond = f"{_state}_{_state_counter_spans[_state]}"
    if _cond in CONDITION_ORDER:
        CONDITION_SPANS.append((_cond, float(_t0), float(_t1)))
CONDITION_SPANS.sort(key=lambda x: x[1])


def get_condition_spans():
    if CONDITION_SPANS:
        return CONDITION_SPANS
    spans = []
    for cond in CONDITION_ORDER:
        if cond in TIME_WINDOWS:
            t0, t1 = TIME_WINDOWS[cond]
            spans.append((cond, float(t0), float(t1)))
    spans.sort(key=lambda x: x[1])
    return spans


def get_state_spans(state: str):
    st = str(state).upper().strip()
    spans = [(cond, t0, t1) for cond, t0, t1 in get_condition_spans()
             if cond.upper().startswith(f"{st}_")]
    spans.sort(key=lambda x: x[1])
    return spans


# Helpers
def parse_subject_session(stem: str):
    s = stem.upper()
    m = re.search(r"SUB[-_]?(\d{2,})", s) or re.search(r"(\d{2,})", s)
    sub = (m.group(1) if m else stem).zfill(2)
    if re.search(r"(SES[-_]?PRE|[-_]PRE)(?:[-_]|$)", s):
        sess = "PRE"
    elif re.search(r"(SES[-_]?POST|[-_](POST|POS))(?:[-_]|$)", s):
        sess = "POST"
    else:
        sess = ""
    return sub, sess


def map_subject_to_group(sub: str) -> str:
    sid = str(sub).zfill(2)
    if sid in GROUP_ACTIVE:  return GROUPS_ORDER[0]
    if sid in GROUP_PASSIVE: return GROUPS_ORDER[1]
    if sid in GROUP_CONTROL: return GROUPS_ORDER[2]
    return "Unknown"


def extract_blocks_from_raw(raw: mne.io.BaseRaw):
    """Extract individual block segments from visual-state annotations.

    Returns a list of (condition_key, raw_segment, time_anchor), where:
    - condition_key: "EO_1", "EC_1", "EO_2", or "EC_2"
    - raw_segment: cropped Raw object preserving concatenated-file timing
    - time_anchor: original block onset in recording time, in seconds
    """
    state_counts: dict = {}
    result = []
    sf  = float(raw.info["sfreq"])
    eps = 1.0 / sf

    visual_annots = [
        a for a in raw.annotations
        if str(a["description"]).startswith("visual_state:")
    ]
    if not visual_annots:
        return result

    # Some concatenated files retain first_samp != 0, which can shift
    # annotation onsets (e.g. 15, 135, 255, 375). Normalize to start at 0.
    onset_base = min(float(a["onset"]) for a in visual_annots)

    for annot in visual_annots:
        desc = str(annot["description"])
        state = desc.split(":")[1].strip()
        state_counts[state] = state_counts.get(state, 0) + 1
        block_num = state_counts[state]
        condition = f"{state}_{block_num}"

        t0_concat = float(annot["onset"]) - onset_base
        t1 = t0_concat + float(annot["duration"])
        t1_safe = min(t1, float(raw.times[-1]) - eps)

        # Original onset used to restore recording-time coordinates.
        original_onset = BLOCK_ORIGINAL_ONSET.get(condition, t0_concat)
        time_anchor = original_onset

        if t1_safe > t0_concat:
            seg = raw.copy().crop(tmin=t0_concat, tmax=t1_safe)
            result.append((condition, seg, time_anchor))

    return result


def rois_from_channels(ch_names):
    rois = {}
    for name, arr in ROI_CHANNELS.items():
        got = [c for c in arr if c in ch_names]
        if len(got) >= 2:
            rois[name] = got
    rois["All"] = list(ch_names)
    return rois


def mean_and_ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if x.size == 1:
        return m, np.nan, np.nan
    sd  = float(np.std(x, ddof=1))
    se  = sd / np.sqrt(x.size)
    tc  = t.ppf(0.975, x.size - 1)
    return m, float(m - tc * se), float(m + tc * se)


def fdr_bh_mask(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool)
    order  = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not np.any(passed):
        return np.zeros_like(p, dtype=bool)
    crit = thresh[np.max(np.where(passed)[0])]
    return p <= crit


def compute_series_for_segment(seg: mne.io.BaseRaw, band_tuple):
    """Compute sliding-window power for one individual block segment."""
    fmin_b, fmax_b = band_tuple
    seg = seg.copy().pick("eeg")
    sf      = float(seg.info["sfreq"])
    fmax_eff = min(PSD_FMAX, sf / 2.0 - 1.0)

    overlap = max(0.0, min(TS_WIN_SEC - TS_STEP_SEC, TS_WIN_SEC - 1e-3))
    epochs  = mne.make_fixed_length_epochs(
        seg, duration=TS_WIN_SEC, overlap=overlap,
        preload=True, verbose="ERROR",
    )
    if len(epochs) == 0:
        return np.array([]), np.array([]), np.array([]), []

    # MNE preserves absolute sample indices (first_samp). Convert them to
    # segment-relative time to avoid artificial x-axis offsets.
    times = (epochs.events[:, 0] - float(seg.first_samp)) / sf

    nseg = int(round(WELCH_SEG_SEC * sf))
    spec = epochs.compute_psd(
        method="welch", fmin=PSD_FMIN, fmax=fmax_eff,
        n_fft=nseg, n_per_seg=nseg,
        n_overlap=int(round(WELCH_OVERLAP * nseg)),
        verbose="ERROR",
    )
    psds, freqs = spec.get_data(return_freqs=True)

    band_mask  = (freqs >= fmin_b)   & (freqs <= fmax_b)
    total_mask = (freqs >= PSD_FMIN) & (freqs <= fmax_eff)

    abs_band = psds[:, :, band_mask].mean(axis=2)
    total    = psds[:, :, total_mask].mean(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_band = abs_band / np.where(total > 0, total, np.nan)
    abs_band = abs_band * ABS_SCALE

    return times, abs_band, rel_band, epochs.ch_names


# Data collection
def collect_all_series() -> pd.DataFrame:
    """Load concatenated FIFs, reconstruct the four blocks, and compute power time series."""

    # Only concatenated files, excluding explicit EO/EC derivatives.
    fifs = sorted(PROCESSED_DIR.rglob("*_desc-preproc_clean_raw.fif"))
    fifs = [f for f in fifs
            if "_EO_" not in f.stem.upper() and "_EC_" not in f.stem.upper()]
    print(f"Concatenated files found: {len(fifs)}")

    rows = []
    for f in fifs:
        sub, sess = parse_subject_session(f.stem)
        if not sub or sess not in {"PRE", "POST"}:
            continue
        grp = map_subject_to_group(sub)
        print(f"  Reading: {f.name}")

        raw = mne.io.read_raw_fif(f, preload=True, verbose="WARNING")
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass

        blocks = extract_blocks_from_raw(raw)
        if not blocks:
            print(f"    No annotated blocks found in {f.name}; skipping.")
            continue

        for condition, seg, time_anchor in blocks:
            if condition not in CONDITION_ORDER:
                continue  # Ignore extra blocks (> 2 per visual state).

            for band_name, band_tuple in BANDS.items():
                times, abs_by_ch, rel_by_ch, chs = compute_series_for_segment(
                    seg, band_tuple
                )
                if times.size == 0:
                    continue

                # Restore original recording-time coordinates.
                times_orig = times + time_anchor

                rois = rois_from_channels(chs)
                for roi_name, roi_list in rois.items():
                    idx = [chs.index(ch) for ch in roi_list if ch in chs]
                    if len(idx) < 2 and roi_name != "All":
                        continue
                    abs_roi = abs_by_ch[:, idx].mean(axis=1)
                    rel_roi = rel_by_ch[:, idx].mean(axis=1)
                    for tval, a, r in zip(times_orig, abs_roi, rel_roi):
                        rows.append(dict(
                            subject=sub, group=grp, session=sess,
                            condition=condition,
                            time_s=float(tval), band=band_name,
                            metric="abs", roi=roi_name, value=float(a),
                        ))
                        rows.append(dict(
                            subject=sub, group=grp, session=sess,
                            condition=condition,
                            time_s=float(tval), band=band_name,
                            metric="rel", roi=roi_name, value=float(r),
                        ))

    return pd.DataFrame.from_records(rows)


# PRE/POST alignment by subject
def align_sessions_per_subject(df_panel: pd.DataFrame):
    subs = sorted(df_panel.subject.astype(str).unique(), key=lambda s: int(s))
    ts_common = None
    PRE_list, POST_list = [], []

    for sub in subs:
        dsub   = df_panel[df_panel.subject.astype(str) == sub]
        d_pre  = dsub[dsub.session == "PRE"].sort_values("time_s")
        d_post = dsub[dsub.session == "POST"].sort_values("time_s")
        n = int(min(len(d_pre), len(d_post)))
        if n == 0:
            continue

        tt   = 0.5 * (d_pre.time_s.values[:n] + d_post.time_s.values[:n])
        pre  = d_pre.value.values[:n]
        post = d_post.value.values[:n]

        if ts_common is None:
            ts_common = tt
            PRE_list  = [pre]
            POST_list = [post]
        else:
            m = min(len(ts_common), len(tt))
            ts_common = ts_common[:m]
            PRE_list  = [x[:m] for x in PRE_list]
            POST_list = [y[:m] for y in POST_list]
            PRE_list.append(pre[:m])
            POST_list.append(post[:m])

    if ts_common is None:
        return None, None, None
    return ts_common, np.vstack(PRE_list), np.vstack(POST_list)


def build_session_matrices(df_panel: pd.DataFrame):
    """Build subject x time matrices for PRE/POST without truncating the timeline."""
    if df_panel.empty:
        return None, None, None

    d = (df_panel.groupby(["subject", "session", "time_s"], as_index=False)["value"]
         .mean())
    d["subject"] = d["subject"].astype(str)
    d["time_key"] = d["time_s"].astype(float).round(6)

    def _sub_sort_key(s):
        return int(s) if str(s).isdigit() else s

    subjects = sorted(d["subject"].unique(), key=_sub_sort_key)
    times = np.sort(d["time_key"].unique().astype(float))
    if times.size == 0 or len(subjects) == 0:
        return None, None, None

    sub_to_i = {s: i for i, s in enumerate(subjects)}
    time_to_i = {float(t): i for i, t in enumerate(times)}

    pre = np.full((len(subjects), len(times)), np.nan, dtype=float)
    post = np.full((len(subjects), len(times)), np.nan, dtype=float)

    for row in d.itertuples(index=False):
        i = sub_to_i[str(row.subject)]
        j = time_to_i[float(row.time_key)]
        if row.session == "PRE":
            pre[i, j] = float(row.value)
        elif row.session == "POST":
            post[i, j] = float(row.value)

    return times, pre, post


def remap_concat_state_time(df_panel: pd.DataFrame, spans):
    """Concatenate blocks from the same state on one continuous axis (e.g. EO1 + EO2)."""
    if df_panel.empty or not spans:
        return pd.DataFrame(columns=df_panel.columns)

    chunks = []
    offset = 0.0
    for cond, t0, t1 in spans:
        dcond = df_panel[df_panel["condition"] == cond].copy()
        span_len = max(0.0, float(t1) - float(t0))
        if dcond.empty:
            offset += span_len
            continue
        # Ensure that only samples from the current block enter the concatenation.
        dcond = dcond[(dcond["time_s"] >= float(t0)) & (dcond["time_s"] <= float(t1))]
        if dcond.empty:
            offset += span_len
            continue
        dcond["time_s"] = dcond["time_s"].astype(float) - float(t0) + offset
        chunks.append(dcond)
        offset += span_len

    if not chunks:
        return pd.DataFrame(columns=df_panel.columns)
    return pd.concat(chunks, ignore_index=True)


# Plotting
def plot_4blocks_grid(df_all: pd.DataFrame, band: str, metric: str, roi: str):
    n_rows = len(CONDITION_ORDER)
    n_cols = len(GROUPS_ORDER)

    fig = plt.figure(figsize=(13.2, 3.8 * n_rows), facecolor="white")
    gs  = gridspec.GridSpec(nrows=n_rows, ncols=n_cols,
                            figure=fig, wspace=0.26, hspace=0.42)
    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(gs[i, j])
            ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.7)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            axes[i, j] = ax

    # Column titles (groups)
    for j, grp in enumerate(GROUPS_ORDER):
        axes[0, j].set_title(grp, fontsize=13, pad=12)

    # Row labels (conditions), placed at the left side of the figure.
    for i, cond in enumerate(CONDITION_ORDER):
        bbox = axes[i, 0].get_position()
        fig.text(
            0.065, (bbox.y0 + bbox.y1) / 2,
            CONDITION_LABELS[cond],
            va="center", ha="right", fontsize=11, fontweight="bold",
        )

    legend_handles, legend_labels = None, None
    row_ymins = [np.inf]  * n_rows
    row_ymaxs = [-np.inf] * n_rows

    col_pre  = PALETTE_SESS["PRE"]
    col_post = PALETTE_SESS["POST"]

    for i, cond in enumerate(CONDITION_ORDER):
        for j, grp in enumerate(GROUPS_ORDER):
            ax = axes[i, j]
            d = df_all[
                (df_all.condition == cond)  &
                (df_all.group     == grp)   &
                (df_all.band      == band)  &
                (df_all.metric    == metric) &
                (df_all.roi       == roi)
            ]
            if d.empty:
                ax.axis("off")
                continue

            ts, PRE, POST = align_sessions_per_subject(d)
            if ts is None:
                ax.axis("off")
                continue

            # Restrict to the original condition interval.
            t_min, t_max = TIME_WINDOWS.get(cond, (ts.min(), ts.max()))
            mask = (ts >= t_min) & (ts <= t_max)
            ts   = ts[mask]
            PRE  = PRE[:,  mask]
            POST = POST[:, mask]
            if ts.size == 0:
                ax.axis("off")
                continue

            mean_pre  = np.nanmean(PRE,  axis=0)
            mean_post = np.nanmean(POST, axis=0)
            ci_pre    = np.array([mean_and_ci95(PRE[:,  k]) for k in range(PRE.shape[1])])
            ci_post   = np.array([mean_and_ci95(POST[:, k]) for k in range(POST.shape[1])])

            h1, = ax.plot(ts, mean_pre,  linewidth=2.0, color=col_pre,  label="PRE")
            ax.fill_between(ts, ci_pre[:, 1],  ci_pre[:, 2],
                            alpha=0.18, color=col_pre)
            h2, = ax.plot(ts, mean_post, linewidth=2.0, color=col_post, label="POST")
            ax.fill_between(ts, ci_post[:, 1], ci_post[:, 2],
                            alpha=0.18, color=col_post)

            if legend_handles is None:
                legend_handles = [h1, h2]
                legend_labels  = ["PRE", "POST"]

            ax.set_xlim(t_min, t_max)
            ax.set_xlabel("Recording Time (s)", fontsize=10)
            if j == 0:
                ylabel = (
                    "Relative Power"
                    if metric == "rel"
                    else "Absolute Power (uV^2/Hz)"
                )
                ax.set_ylabel(ylabel, fontsize=10)

            # FDR significance marking for PRE vs POST at each time window.
            if MARK_SIG:
                pvals = []
                for k in range(PRE.shape[1]):
                    a = POST[:, k]
                    b = PRE[:,  k]
                    m = np.isfinite(a) & np.isfinite(b)
                    if np.sum(m) < 3:
                        pvals.append(1.0)
                        continue
                    try:
                        p = wilcoxon(a[m] - b[m], zero_method="wilcox").pvalue
                    except Exception:
                        p = 1.0
                    pvals.append(float(p))

                sigmask = fdr_bh_mask(np.array(pvals), alpha=ALPHA_FDR)
                y_top  = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])
                dy     = np.nanmax([mean_pre, mean_post]) - np.nanmin([mean_pre, mean_post])
                y_mark = y_top + 0.06 * (dy if np.isfinite(dy) and dy > 0 else 1.0)
                ax.plot(ts, np.where(sigmask, y_mark, np.nan),
                        linewidth=3.0, color=config.COLOR_FDR, solid_capstyle="butt")

                y_min_ax = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
                y_max_ax = max(y_mark, np.nanmax([ci_pre[:, 2], ci_post[:, 2]]))
            else:
                y_min_ax = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
                y_max_ax = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])

            row_ymins[i] = min(row_ymins[i], y_min_ax)
            row_ymaxs[i] = max(row_ymaxs[i], y_max_ax)

    # Synchronize the y-axis within each row for cross-group comparability.
    for i in range(n_rows):
        ymin, ymax = row_ymins[i], row_ymaxs[i]
        if not (np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin):
            continue
        pad = 0.04 * (ymax - ymin)
        for j in range(n_cols):
            if axes[i, j].has_data():
                axes[i, j].set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(
        f"Four-Block Time Series - {band} - {metric.upper()} - ROI: {roi}",
        fontsize=15, y=0.995,
    )
    plt.subplots_adjust(left=0.15, right=0.86, bottom=0.05, top=0.95)

    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper right", bbox_to_anchor=(0.985, 0.985),
            frameon=True, fontsize=11,
        )

    roi_slug = roi.replace(" ", "_").replace("-", "")
    fname = FIG_DIR / f"ts_{band}_{metric}_{roi_slug}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_continuous_timeline(df_all: pd.DataFrame, band: str, metric: str, roi: str):
    """Plot the full continuous timeline, marking EO_1/EC_1/EO_2/EC_2 blocks."""
    n_rows = len(GROUPS_ORDER)
    fig = plt.figure(figsize=(15.5, 3.9 * n_rows), facecolor="white")
    gs = gridspec.GridSpec(nrows=n_rows, ncols=1, figure=fig, hspace=0.28)
    axes = np.empty((n_rows,), dtype=object)

    for i in range(n_rows):
        share_ax = axes[0] if i > 0 else None
        ax = fig.add_subplot(gs[i, 0], sharex=share_ax)
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        axes[i] = ax

    col_pre = PALETTE_SESS["PRE"]
    col_post = PALETTE_SESS["POST"]
    legend_handles, legend_labels = None, None

    spans = get_condition_spans()
    if spans:
        x_min = min(t0 for _, t0, _ in spans)
        x_max = max(t1 for _, _, t1 in spans)
    else:
        x_min = float(df_all["time_s"].min())
        x_max = float(df_all["time_s"].max())

    ymins, ymaxs = [], []

    for i, grp in enumerate(GROUPS_ORDER):
        ax = axes[i]
        ax.set_title(grp, fontsize=13, pad=10)

        d = df_all[
            (df_all.group == grp) &
            (df_all.band == band) &
            (df_all.metric == metric) &
            (df_all.roi == roi)
        ]
        if d.empty:
            ax.axis("off")
            continue

        ts, PRE, POST = build_session_matrices(d)
        if ts is None:
            ax.axis("off")
            continue

        mask = (ts >= x_min) & (ts <= x_max)
        ts = ts[mask]
        PRE = PRE[:, mask]
        POST = POST[:, mask]
        if ts.size == 0:
            ax.axis("off")
            continue

        mean_pre = np.nanmean(PRE, axis=0)
        mean_post = np.nanmean(POST, axis=0)
        ci_pre = np.array([mean_and_ci95(PRE[:, k]) for k in range(PRE.shape[1])])
        ci_post = np.array([mean_and_ci95(POST[:, k]) for k in range(POST.shape[1])])

        # Alternate background shading to improve block readability.
        for idx, (_, t0, t1) in enumerate(spans):
            shade = 0.07 if idx % 2 == 0 else 0.03
            ax.axvspan(t0, t1, color="black", alpha=shade, zorder=0)

        h1, = ax.plot(ts, mean_pre, linewidth=2.0, color=col_pre, label="PRE")
        ax.fill_between(ts, ci_pre[:, 1], ci_pre[:, 2], alpha=0.18, color=col_pre)
        h2, = ax.plot(ts, mean_post, linewidth=2.0, color=col_post, label="POST")
        ax.fill_between(ts, ci_post[:, 1], ci_post[:, 2], alpha=0.18, color=col_post)

        if legend_handles is None:
            legend_handles = [h1, h2]
            legend_labels = ["PRE", "POST"]

        if MARK_SIG:
            pvals = []
            for k in range(PRE.shape[1]):
                a = POST[:, k]
                b = PRE[:, k]
                m = np.isfinite(a) & np.isfinite(b)
                if np.sum(m) < 3:
                    pvals.append(1.0)
                    continue
                try:
                    p = wilcoxon(a[m] - b[m], zero_method="wilcox").pvalue
                except Exception:
                    p = 1.0
                pvals.append(float(p))

            sigmask = fdr_bh_mask(np.array(pvals), alpha=ALPHA_FDR)
            y_top = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])
            dy = np.nanmax([mean_pre, mean_post]) - np.nanmin([mean_pre, mean_post])
            y_mark = y_top + 0.06 * (dy if np.isfinite(dy) and dy > 0 else 1.0)
            ax.plot(ts, np.where(sigmask, y_mark, np.nan),
                    linewidth=3.0, color=config.COLOR_FDR, solid_capstyle="butt")
            y_min_ax = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
            y_max_ax = max(y_mark, np.nanmax([ci_pre[:, 2], ci_post[:, 2]]))
        else:
            y_min_ax = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
            y_max_ax = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])

        ymins.append(y_min_ax)
        ymaxs.append(y_max_ax)

        # Block boundaries and labels.
        for i_span, (cond, t0, t1) in enumerate(spans):
            if i_span > 0:
                ax.axvline(t0, color="black", linewidth=1.2, linestyle="--", alpha=0.65)
            y_for_label = y_max_ax + 0.02 * (y_max_ax - y_min_ax if y_max_ax > y_min_ax else 1.0)
            ax.text((t0 + t1) / 2.0, y_for_label, cond.replace("_", ""),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xlim(x_min, x_max)
        if i == n_rows - 1:
            ax.set_xlabel("Recording Time (s)", fontsize=11)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        ylabel = (
            "Relative Power"
            if metric == "rel"
            else "Absolute Power (uV^2/Hz)"
        )
        ax.set_ylabel(ylabel, fontsize=10)

    if ymins and ymaxs:
        ymin = np.nanmin(ymins)
        ymax = np.nanmax(ymaxs)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.10 * (ymax - ymin)
            for ax in axes:
                if ax.has_data():
                    ax.set_ylim(ymin - 0.04 * (ymax - ymin), ymax + pad)

    fig.suptitle(
        f"Continuous Time Series - {band} - {metric.upper()} - ROI: {roi}",
        fontsize=16, y=0.995,
    )
    plt.subplots_adjust(left=0.09, right=0.88, bottom=0.06, top=0.94)

    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper right", bbox_to_anchor=(0.985, 0.98),
            frameon=True, fontsize=11,
        )

    roi_slug = roi.replace(" ", "_").replace("-", "")
    fname = FIG_DIR / f"ts_continuous_{band}_{metric}_{roi_slug}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_eo_combined_timeline(df_all: pd.DataFrame, band: str, metric: str, roi: str):
    """Plot a 3 x 1 layout combining only EO_1 + EO_2 on a continuous axis."""
    spans = get_state_spans("EO")
    if not spans:
        return

    total_len = float(sum(max(0.0, t1 - t0) for _, t0, t1 in spans))
    if total_len <= 0:
        return

    n_rows = len(GROUPS_ORDER)
    fig = plt.figure(figsize=(15.5, 3.9 * n_rows), facecolor="white")
    gs = gridspec.GridSpec(nrows=n_rows, ncols=1, figure=fig, hspace=0.28)
    axes = np.empty((n_rows,), dtype=object)

    for i in range(n_rows):
        share_ax = axes[0] if i > 0 else None
        ax = fig.add_subplot(gs[i, 0], sharex=share_ax)
        ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.7)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        axes[i] = ax

    col_pre = PALETTE_SESS["PRE"]
    col_post = PALETTE_SESS["POST"]
    legend_handles, legend_labels = None, None
    ymins, ymaxs = [], []

    for i, grp in enumerate(GROUPS_ORDER):
        ax = axes[i]
        ax.set_title(grp, fontsize=13, pad=10)

        d = df_all[
            (df_all.group == grp) &
            (df_all.band == band) &
            (df_all.metric == metric) &
            (df_all.roi == roi) &
            (df_all.condition.isin([s[0] for s in spans]))
        ]
        if d.empty:
            ax.axis("off")
            continue

        d = remap_concat_state_time(d, spans)
        if d.empty:
            ax.axis("off")
            continue

        ts, PRE, POST = build_session_matrices(d)
        if ts is None:
            ax.axis("off")
            continue

        mask = (ts >= 0.0) & (ts <= total_len)
        ts = ts[mask]
        PRE = PRE[:, mask]
        POST = POST[:, mask]
        if ts.size == 0:
            ax.axis("off")
            continue

        mean_pre = np.nanmean(PRE, axis=0)
        mean_post = np.nanmean(POST, axis=0)
        ci_pre = np.array([mean_and_ci95(PRE[:, k]) for k in range(PRE.shape[1])])
        ci_post = np.array([mean_and_ci95(POST[:, k]) for k in range(POST.shape[1])])

        # Alternate EO1/EO2 shading on the concatenated axis.
        offset = 0.0
        for idx, (_, t0, t1) in enumerate(spans):
            seg_len = max(0.0, float(t1) - float(t0))
            shade = 0.07 if idx % 2 == 0 else 0.03
            ax.axvspan(offset, offset + seg_len, color="black", alpha=shade, zorder=0)
            offset += seg_len

        h1, = ax.plot(ts, mean_pre, linewidth=2.0, color=col_pre, label="PRE")
        ax.fill_between(ts, ci_pre[:, 1], ci_pre[:, 2], alpha=0.18, color=col_pre)
        h2, = ax.plot(ts, mean_post, linewidth=2.0, color=col_post, label="POST")
        ax.fill_between(ts, ci_post[:, 1], ci_post[:, 2], alpha=0.18, color=col_post)
        if legend_handles is None:
            legend_handles = [h1, h2]
            legend_labels = ["PRE", "POST"]

        if MARK_SIG:
            pvals = []
            for k in range(PRE.shape[1]):
                a = POST[:, k]
                b = PRE[:, k]
                m = np.isfinite(a) & np.isfinite(b)
                if np.sum(m) < 3:
                    pvals.append(1.0)
                    continue
                try:
                    p = wilcoxon(a[m] - b[m], zero_method="wilcox").pvalue
                except Exception:
                    p = 1.0
                pvals.append(float(p))
            sigmask = fdr_bh_mask(np.array(pvals), alpha=ALPHA_FDR)
            y_top = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])
            dy = np.nanmax([mean_pre, mean_post]) - np.nanmin([mean_pre, mean_post])
            y_mark = y_top + 0.06 * (dy if np.isfinite(dy) and dy > 0 else 1.0)
            ax.plot(ts, np.where(sigmask, y_mark, np.nan),
                    linewidth=3.0, color=config.COLOR_FDR, solid_capstyle="butt")
            y_min_ax = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
            y_max_ax = max(y_mark, np.nanmax([ci_pre[:, 2], ci_post[:, 2]]))
        else:
            y_min_ax = np.nanmin([ci_pre[:, 1], ci_post[:, 1]])
            y_max_ax = np.nanmax([ci_pre[:, 2], ci_post[:, 2]])

        ymins.append(y_min_ax)
        ymaxs.append(y_max_ax)

        # EO1 | EO2 boundary on the concatenated axis.
        offset = 0.0
        for idx, (cond, t0, t1) in enumerate(spans):
            seg_len = max(0.0, float(t1) - float(t0))
            if idx > 0:
                ax.axvline(offset, color="black", linewidth=1.2, linestyle="--", alpha=0.65)
            y_for_label = y_max_ax + 0.02 * (y_max_ax - y_min_ax if y_max_ax > y_min_ax else 1.0)
            ax.text(offset + 0.5 * seg_len, y_for_label, cond.replace("_", ""),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
            offset += seg_len

        ax.set_xlim(0.0, total_len)
        if i == n_rows - 1:
            ax.set_xlabel("Concatenated EO Time (s)", fontsize=11)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        ylabel = (
            "Relative Power"
            if metric == "rel"
            else "Absolute Power (uV^2/Hz)"
        )
        ax.set_ylabel(ylabel, fontsize=10)

    if ymins and ymaxs:
        ymin = np.nanmin(ymins)
        ymax = np.nanmax(ymaxs)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = 0.10 * (ymax - ymin)
            for ax in axes:
                if ax.has_data():
                    ax.set_ylim(ymin - 0.04 * (ymax - ymin), ymax + pad)

    fig.suptitle(
        f"Combined EO Time Series - {band} - {metric.upper()} - ROI: {roi}",
        fontsize=16, y=0.995,
    )
    plt.subplots_adjust(left=0.09, right=0.88, bottom=0.06, top=0.94)

    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper right", bbox_to_anchor=(0.985, 0.98),
            frameon=True, fontsize=11,
        )

    roi_slug = roi.replace(" ", "_").replace("-", "")
    fname = FIG_DIR / f"ts_eo_combined_{band}_{metric}_{roi_slug}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# Main
def main():
    print("Processed-data directory:", PROCESSED_DIR.resolve())

    print("\nExtracting four-block time series...")
    df = collect_all_series()
    if df.empty:
        print("No data extracted. Check PROCESSED_DIR and the derivative filenames.")
        return

    csv_out = CSV_DIR / "ts_all_bands_rois_relabs.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nCSV saved: {csv_out}")
    print(f"Conditions found: {sorted(df.condition.unique())}")
    print(f"Subjects: {len(df.subject.unique())}  |  "
          f"Sessions: {sorted(df.session.unique())}  |  "
          f"Bands: {sorted(df.band.unique())}")

    if GENERATE_PLOTS:
        bands   = [b for b in BANDS if b in df.band.unique()]
        metrics = ["rel", "abs"]
        rois    = sorted(df.roi.unique())
        total_per_mode = len(bands) * len(metrics) * len(rois)
        modes = 1 + int(GENERATE_CONTINUOUS) + int(GENERATE_EO_COMBINED)
        total = total_per_mode * modes
        print(f"\nGenerating {total} figures...")
        for band in bands:
            for metric in metrics:
                for roi in rois:
                    plot_4blocks_grid(df, band, metric, roi)
                    if GENERATE_CONTINUOUS:
                        plot_continuous_timeline(df, band, metric, roi)
                    if GENERATE_EO_COMBINED:
                        plot_eo_combined_timeline(df, band, metric, roi)

    print("\nDone.")


if __name__ == "__main__":
    main()
