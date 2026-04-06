"""Sliding-window ROI×Band power time series (ABS/REL) for EO/EC and PRE/POST."""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import mne
import config

# ---------------------------------------------------------------------
# Paths / parameters
# ---------------------------------------------------------------------
PROCESSED_DIR = config.PROCESSED_DIR
OUT_DIR = config.TS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = config.BANDS
REGIONS = config.REGIONS
BANDS_ORDER = config.BANDS_ORDER
ROIS_ORDER = config.ROIS_ORDER
SESS_ORDER = config.SESS_ORDER
VS_ORDER = config.VS_ORDER

PSD_FMIN = config.PSD_FMIN
PSD_FMAX = config.PSD_FMAX
WELCH_SEG_SEC = config.WELCH_SEG_SEC
WELCH_OVERLAP = config.WELCH_OVERLAP
TS_WIN_SEC = config.TS_WIN_SEC
TS_STEP_SEC = config.TS_STEP_SEC
ABS_SCALE = float(getattr(config, "POWER_ABS_SCALE", 1.0))
if not np.isfinite(ABS_SCALE) or ABS_SCALE <= 0:
    ABS_SCALE = 1.0

GROUP_ACTIVE = set(config.GROUP_ACTIVE)
GROUP_PASSIVE = set(config.GROUP_PASSIVE)
GROUP_CONTROL = set(config.GROUP_CONTROL)
ACTIVE_CHANNELS = set(config.ACTIVE_CHANNELS)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_subject_session_state(stem: str):
    s = stem.upper()
    m = re.search(r"SUB[-_]?(\d{2,})", s)
    sub = m.group(1).zfill(2) if m else stem

    if re.search(r"(SES[-_]?PRE|[-_]PRE)(?:[-_]|$)", s):
        sess = "PRE"
    elif re.search(r"(SES[-_]?POST|[-_]POS[T]?)(?:[-_]|$)", s):
        sess = "POST"
    else:
        sess = ""

    if re.search(r"(^|[-_])EC($|[-_])", s):
        state = "EC"
    elif re.search(r"(^|[-_])EO($|[-_])", s):
        state = "EO"
    else:
        state = ""

    return sub, sess, state


def infer_group_from_subject(subj_num: str) -> str:
    s = str(subj_num).zfill(2)
    if s in GROUP_ACTIVE:
        return "Active"
    if s in GROUP_PASSIVE:
        return "Passive"
    if s in GROUP_CONTROL:
        return "Control"
    return "Unknown"


def pick_roi_channel_indices(raw: mne.io.BaseRaw, ch_list):
    inc = [ch for ch in ch_list if (not ACTIVE_CHANNELS or ch in ACTIVE_CHANNELS)]
    return mne.pick_channels(raw.ch_names, include=inc, ordered=False)


def load_state_raw(f: Path):
    try:
        return mne.io.read_raw_fif(f, preload=True, verbose="ERROR")
    except Exception as e:
        print(f"  WARNING: cannot open {f.name}: {e}")
    return None

# ---------------------------------------------------------------------
# Discover candidate FIF files
# ---------------------------------------------------------------------
concat_files = sorted(PROCESSED_DIR.rglob("*_clean_raw.fif"))

bases = []
seen = set()
for f in concat_files:
    sub, sess, state = parse_subject_session_state(f.stem)
    if not sub or not state or not sess:
        continue
    key = (f.parent, f.stem)
    if key in seen:
        continue
    seen.add(key)
    bases.append({
        "dir_eeg": f.parent,
        "path": f,
        "sub_num": sub,
        "sess_pp": sess,
        "state": state
    })

if not bases:
    print(f"No matching FIF files found under {PROCESSED_DIR}/**/eeg/")
    raise SystemExit

_sess_rank = {s: i for i, s in enumerate(SESS_ORDER)}
bases.sort(key=lambda x: (x["sub_num"], _sess_rank.get(x["sess_pp"], 99)))

# ---------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------
rows = []

for entry in bases:
    f = entry["path"]
    sub = entry["sub_num"]
    sess = entry["sess_pp"]
    state = entry["state"]
    grp = infer_group_from_subject(sub)

    raw = load_state_raw(f)
    if raw is None:
        print(f"Missing {state} for {f.stem}; skipping.")
        continue

    raw.pick("eeg")
    sf = float(raw.info["sfreq"])
    fmax_eff = min(float(PSD_FMAX), sf / 2.0 - 1.0)

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(TS_WIN_SEC),
        overlap=max(0.0, float(TS_WIN_SEC) - float(TS_STEP_SEC)),
        preload=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        print(f"Not enough windows for {f.stem}; skipping.")
        continue

    epoch_n_samples = len(epochs.times)
    n_per_seg = min(epoch_n_samples, max(16, int(round(float(WELCH_SEG_SEC) * sf))))
    n_overlap = int(round(float(WELCH_OVERLAP) * n_per_seg))
    n_overlap = min(max(n_overlap, 0), max(n_per_seg - 1, 0))
    spec = epochs.compute_psd(
        method="welch",
        fmin=float(PSD_FMIN),
        fmax=fmax_eff,
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        verbose="ERROR",
    )
    psds, freqs = spec.get_data(return_freqs=True)

    total_mask = (freqs >= float(PSD_FMIN)) & (freqs <= fmax_eff)
    band_masks = {b: ((freqs >= lo) & (freqs <= hi)) for b, (lo, hi) in BANDS.items()}

    # MNE preserves absolute sample indices via `first_samp`; normalize them so
    # exported times always start at 0 within each EO/EC derivative file.
    starts = (epochs.events[:, 0] - float(raw.first_samp)) / sf
    centers = starts + float(TS_WIN_SEC) / 2.0

    for roi in ROIS_ORDER:
        chs = REGIONS[roi]
        picks = pick_roi_channel_indices(raw, chs)
        if picks.size == 0:
            print(f"  WARNING: ROI '{roi}' has no matching channels in {f.stem}; skipping.")
            continue

        total = psds[:, picks][:, :, total_mask].mean(axis=2)
        total_mean = np.where(np.isfinite(total), total, np.nan).mean(axis=1)

        for band in BANDS_ORDER:
            mask = band_masks[band]
            band_abs = psds[:, picks][:, :, mask].mean(axis=2)
            band_abs_mean = np.nanmean(band_abs, axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                band_rel = np.divide(band_abs_mean, total_mean)

            for t, p_abs, p_rel in zip(centers, band_abs_mean, band_rel):
                rows.append({
                    "subject": sub,
                    "group": grp,
                    "session": sess,
                    "visual_state": state,
                    "region": roi,
                    "band": band,
                    "t_sec": float(t),
                    "power_abs": float(p_abs * ABS_SCALE) if np.isfinite(p_abs) else np.nan,
                    "power_rel": float(p_rel) if np.isfinite(p_rel) else np.nan,
                    "source_file": f.stem,
                })

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------
df = pd.DataFrame(rows).sort_values(
    ["band", "region", "group", "subject", "visual_state", "session", "t_sec"]
)
out_csv = OUT_DIR / "ts_power_long.csv"
df.to_csv(out_csv, index=False)

params = dict(
    BANDS=BANDS,
    REGIONS=REGIONS,
    TS_WIN_SEC=float(TS_WIN_SEC),
    TS_STEP_SEC=float(TS_STEP_SEC),
    PSD_FMIN=float(PSD_FMIN),
    PSD_FMAX=float(PSD_FMAX),
    WELCH_SEG_SEC=float(WELCH_SEG_SEC),
    WELCH_OVERLAP=float(WELCH_OVERLAP),
    POWER_ABS_SCALE=float(ABS_SCALE),
)
(OUT_DIR / "readme_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

print(f"\nTime series saved to: {out_csv}")
