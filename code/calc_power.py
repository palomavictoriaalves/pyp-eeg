"""Compute band × ROI power (absolute/relative)"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne

import config

# --- paths / params -----------------------------------------------------------
PROCESSED_DIR = config.PROCESSED_DIR
POWER_DIR     = config.POWER_DIR
POWER_DIR.mkdir(parents=True, exist_ok=True)

BANDS      = config.BANDS
REGIONS    = config.REGIONS
ACTIVE_CHS = set(config.ACTIVE_CHANNELS)

WELCH_SEG = config.WELCH_SEG_SEC
WELCH_OVL = config.WELCH_OVERLAP
PSD_FMIN  = config.PSD_FMIN
PSD_FMAX  = config.PSD_FMAX

EXPORT_REL  = config.EXPORT_RELATIVE
STD_DUR_SEC = config.STANDARDIZE_DURATION_SEC  # seconds or None
ABS_SCALE   = float(getattr(config, "POWER_ABS_SCALE", 1.0))
if not np.isfinite(ABS_SCALE) or ABS_SCALE <= 0:
    ABS_SCALE = 1.0

GROUP_ACTIVE  = set(config.GROUP_ACTIVE)
GROUP_PASSIVE = set(config.GROUP_PASSIVE)
GROUP_CONTROL = set(config.GROUP_CONTROL)

# Canonical orders (fallback to dict order if missing)
BANDS_ORDER = getattr(config, "BANDS_ORDER", list(BANDS.keys()))
ROIS_ORDER  = getattr(config, "ROIS_ORDER", list(REGIONS.keys()))
SESS_ORDER  = getattr(config, "SESS_ORDER", ["PRE", "POST"])
VS_ORDER    = getattr(config, "VS_ORDER", ["EO", "EC"])

print("\n=== calc_power.py (BIDS-aware) ===")
print(f"Reading from : {PROCESSED_DIR}")
print(f"Saving to    : {POWER_DIR}")
print(f"ABS scale    : x{ABS_SCALE:g}")

# ---- helpers -----------------------------------------------------------------
_BIDS_CONCAT_RE = re.compile(
    r"(?P<sub>sub-\d+)"
    r"(?:_(?P<ses>ses-[a-z0-9]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_desc-(?P<desc>[^_]+)"
    r"_clean_raw\.fif$",
    re.IGNORECASE,
)

def _subject_num(sub_tag: str) -> str:
    m = re.search(r"sub-(\d+)", sub_tag, re.IGNORECASE)
    return m.group(1).zfill(2) if m else sub_tag

def _sess_prepost(ses: str | None) -> str:
    if not ses:
        return ""
    s = ses.lower()
    if s.endswith("pre"):
        return "PRE"
    if s.endswith("post"):
        return "POST"
    return s.upper()

def _infer_group(sub: str) -> str:
    s = str(sub).zfill(2)
    if s in GROUP_ACTIVE:
        return "Active"
    if s in GROUP_PASSIVE:
        return "Passive"
    if s in GROUP_CONTROL:
        return "Control"
    return "Unknown"

def _pick_roi(raw: mne.io.BaseRaw, ch_list):
    inc = [ch for ch in ch_list if (not ACTIVE_CHS or ch in ACTIVE_CHS)]
    return mne.pick_channels(raw.ch_names, include=inc, ordered=False)

def _psd_welch(raw: mne.io.BaseRaw, fmin, fmax, seg_sec, overlap):
    sf = float(raw.info["sfreq"])
    fmax_eff = min(float(fmax), sf / 2.0 - 1.0)
    n_per_seg = max(16, int(round(seg_sec * sf)))
    n_overlap = int(round(overlap * n_per_seg))
    spec = raw.compute_psd(
        method="welch",
        picks="eeg",
        fmin=float(fmin),
        fmax=fmax_eff,
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        verbose="ERROR",
    )
    return spec.get_data(return_freqs=True)

def _roi_band_abs(psds, freqs, picks, band_rng):
    if picks.size == 0:
        return np.nan
    lo, hi = band_rng
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.nan
    return float(np.mean(psds[picks][:, mask]))

def _open_state_raw(dir_eeg: Path, base: str, state: str):
    direct = dir_eeg / f"{base}_desc-preproc_{state}_clean_raw.fif"
    if direct.exists():
        try:
            return mne.io.read_raw_fif(direct, preload=True, verbose="ERROR")
        except Exception as e:
            print(f"  > WARNING: cannot open {direct.name}: {e}")

    concat = dir_eeg / f"{base}_desc-preproc_clean_raw.fif"
    if concat.exists():
        try:
            raw_all = mne.io.read_raw_fif(concat, preload=True, verbose="ERROR")
            if not raw_all.annotations or len(raw_all.annotations) == 0:
                return None
            sf = float(raw_all.info.get("sfreq", 250.0))
            eps = 1.0 / sf
            pieces = []
            for desc, onset, dur in zip(
                raw_all.annotations.description,
                raw_all.annotations.onset,
                raw_all.annotations.duration,
            ):
                if desc == f"visual_state:{state}" and dur > 0:
                    t0 = float(onset)
                    t1 = min(float(onset + dur), float(raw_all.times[-1]) - eps)
                    if t1 > t0:
                        pieces.append(raw_all.copy().crop(tmin=t0, tmax=t1))
            if not pieces:
                return None
            return (
                mne.concatenate_raws(pieces, verbose="ERROR")
                if len(pieces) > 1
                else pieces[0]
            )
        except Exception as e:
            print(f"  > WARNING: cannot crop {concat.name}: {e}")
    return None

def _manifest_duration(dir_eeg: Path, base: str, state: str) -> float:
    mpath = dir_eeg / f"{base}_desc-preproc_blocks_manifest.csv"
    if not mpath.exists():
        return np.nan
    try:
        dfm = pd.read_csv(mpath)
        use = (dfm.get("visual_state", "").astype(str) == state) & (dfm.get("used", 0) == 1)
        return float(dfm.loc[use, "duration_s"].sum())
    except Exception:
        return np.nan

# ---- discover concatenated FIFs ----------------------------------------------
pairs = []   # {"dir": Path, "base": str, "sub": "01", "sess": "PRE|POST|''"}
seen = set()

for f in PROCESSED_DIR.rglob("*_desc-preproc_clean_raw.fif"):
    m = _BIDS_CONCAT_RE.search(f.name)
    if not m:
        continue
    d = m.groupdict()
    base = "_".join(
        [d["sub"]]
        + ([d["ses"]] if d.get("ses") else [])
        + [f"task-{d['task']}"]
        + ([f"run-{d['run']}"] if d.get("run") else [])
    )
    key = (f.parent, base)
    if key in seen:
        continue
    seen.add(key)
    pairs.append(
        {
            "dir": f.parent,
            "base": base,
            "sub": _subject_num(d["sub"]),
            "sess": _sess_prepost(d.get("ses")),
        }
    )

print(f"\nDetected subject/session sets: {len(pairs)}")

# Stable ordering by subject, then session order
_sess_rank = {s: i for i, s in enumerate(SESS_ORDER)}
pairs.sort(key=lambda x: (x["sub"], _sess_rank.get(x["sess"], 99)))

# ---- main --------------------------------------------------------------------
rows = []
roi_cov_rows = []

for item in pairs:
    sub, sess, base, dir_eeg = item["sub"], item["sess"], item["base"], item["dir"]
    group = _infer_group(sub)

    for state in VS_ORDER:
        print(f"\nPower: sub={sub} sess={sess or '-'} state={state} group={group}")
        raw = _open_state_raw(dir_eeg, base, state)
        if raw is None:
            print(f"  > NOTE: missing {state} for {base}. Skipping.")
            continue

        # Optional duration cap per state
        if isinstance(STD_DUR_SEC, (int, float)) and STD_DUR_SEC and STD_DUR_SEC > 0:
            T = float(STD_DUR_SEC)
            if float(raw.times[-1]) > T:
                sf = float(raw.info["sfreq"])
                raw.crop(tmin=0.0, tmax=T - 1.0 / sf)

        raw.pick("eeg")
        psds, freqs = _psd_welch(raw, PSD_FMIN, PSD_FMAX, WELCH_SEG, WELCH_OVL)

        total_mask = (freqs >= float(PSD_FMIN)) & (freqs <= min(float(PSD_FMAX), freqs[-1]))
        total_power = np.mean(psds[:, total_mask], axis=1)

        dur_s = _manifest_duration(dir_eeg, base, state)

        for region in ROIS_ORDER:
            chs = REGIONS[region]
            picks = _pick_roi(raw, chs)
            roi_cov_rows.append(
                {
                    "subject": sub,
                    "group": group,
                    "session": sess,
                    "visual_state": state,
                    "region": region,
                    "n_channels_present": int(picks.size),
                }
            )

            rel_den = np.nanmean(total_power[picks]) if (EXPORT_REL and picks.size > 0) else np.nan

            for band in BANDS_ORDER:
                br = BANDS[band]
                p_abs = _roi_band_abs(psds, freqs, picks, br)
                p_abs_scaled = float(p_abs * ABS_SCALE) if np.isfinite(p_abs) else np.nan
                p_rel = (
                    float(p_abs / rel_den)
                    if (EXPORT_REL and np.isfinite(rel_den) and rel_den > 0 and np.isfinite(p_abs))
                    else np.nan
                )

                rows.append(
                    {
                        "subject": sub,
                        "group": group,
                        "session": sess,
                        "visual_state": state,
                        "region": region,
                        "band": band,
                        "power_abs": p_abs_scaled,
                        "power_rel": p_rel,
                        "duration_s": dur_s,
                        "base_file": f"{base}",
                    }
                )

# ---- export ------------------------------------------------------------------
if not rows:
    print("\nNo results produced.")
else:
    df_long = pd.DataFrame(rows).sort_values(
        ["subject", "session", "visual_state", "region", "band"]
    )
    out_long = POWER_DIR / "power_long_by_region_EO_EC.xlsx"
    df_long.to_excel(out_long, index=False)
    print(f"Saved: {out_long}")

    def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        wide = df.pivot_table(
            index=["subject", "group", "session", "visual_state", "base_file"],
            columns=["region", "band"],
            values=value_col,
            aggfunc="mean",
        )
        # Reorder multiindex columns deterministically: ROI then Band
        ordered_cols = [(r, b) for r in ROIS_ORDER for b in BANDS_ORDER if (r, b) in wide.columns]
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))
        wide.columns = [f"{r}__{b}" for r, b in wide.columns]
        return wide.reset_index()

    wide_abs = _pivot(df_long, "power_abs")
    out_abs = POWER_DIR / "power_wide_abs_EO_EC.csv"
    wide_abs.to_csv(out_abs, index=False)
    print(f"Saved: {out_abs}")

    if EXPORT_REL:
        wide_rel = _pivot(df_long, "power_rel")
        out_rel = POWER_DIR / "power_wide_rel_EO_EC.csv"
        wide_rel.to_csv(out_rel, index=False)
        print(f"Saved: {out_rel}")

    if roi_cov_rows:
        df_cov = pd.DataFrame(roi_cov_rows).sort_values(
            ["subject", "session", "visual_state", "region"]
        )
        out_cov = POWER_DIR / "roi_coverage_EO_EC.csv"
        df_cov.to_csv(out_cov, index=False)
        print(f"Saved: {out_cov}")

    print("\nPower computation (EO/EC) finished.")
