"""2×3 panel (EO/EC × Active/Passive/Control) per band; POST−PRE topomaps.
Exports REL (Δ%) and ABS (Δ dB) with per-channel FDR masking.
"""

from pathlib import Path
import re
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ttest_1samp
from mne.stats import fdr_correction

import config

PROCESSED_DIR = config.PROCESSED_DIR
OUT_REL = config.PLOTS_DIR / "topomaps_grid_rel"
OUT_ABS = config.PLOTS_DIR / "topomaps_grid_abs"
OUT_REL.mkdir(parents=True, exist_ok=True)
OUT_ABS.mkdir(parents=True, exist_ok=True)

BANDS = config.BANDS
BANDS_ORDER = config.BANDS_ORDER
PSD_FMIN = config.PSD_FMIN
PSD_FMAX = config.PSD_FMAX
WELCH_SEG_SEC = config.WELCH_SEG_SEC
WELCH_OVERLAP = config.WELCH_OVERLAP

GROUPS = config.GROUPS_ORDER
STATES = config.VS_ORDER
GROUP_ACTIVE = set(config.GROUP_ACTIVE)
GROUP_PASSIVE = set(config.GROUP_PASSIVE)
GROUP_CONTROL = set(config.GROUP_CONTROL)

def infer_group_from_subject(sub: str) -> str:
    sid = str(sub).zfill(2)
    if sid in GROUP_ACTIVE:  return GROUPS[0]
    if sid in GROUP_PASSIVE: return GROUPS[1]
    if sid in GROUP_CONTROL: return GROUPS[2]
    return "Unknown"

_BIDS_DERIV_RE = re.compile(
    r"(?P<sub>sub-\d+)"
    r"(?:_(?P<ses>ses-[a-z0-9]+))?"
    r"_task-(?P<task>[^_]+)"
    r"(?:_run-(?P<run>[^_]+))?"
    r"_desc-(?P<desc>[^_]+)"
    r"_(?P<what>clean_raw|EO_clean_raw|EC_clean_raw|blocks_manifest)\.(?P<ext>fif|csv)$",
    re.IGNORECASE,
)

def session_label_prepost(ses: str | None) -> str:
    if not ses: return ""
    s = ses.lower()
    if s.endswith("pre"):  return "PRE"
    if s.endswith("post"): return "POST"
    return s.upper()

def subject_two_digit(sub: str) -> str:
    m = re.search(r"sub-(\d+)", sub, re.IGNORECASE)
    return m.group(1).zfill(2) if m else sub

def parse_bids_from_name(path: Path):
    m = _BIDS_DERIV_RE.search(path.name)
    if not m:
        return None
    d = m.groupdict()
    parts = [d["sub"]]
    if d.get("ses"): parts.append(d["ses"])
    parts.append(f"task-{d['task']}")
    if d.get("run"): parts.append(f"run-{d['run']}")
    base = "_".join(parts) + f"_desc-{d['desc']}"
    return {
        "sub": d["sub"],
        "ses": d.get("ses"),
        "task": d["task"],
        "run": d.get("run"),
        "desc": d["desc"],
        "what": d["what"],
        "ext": d["ext"],
        "base": base,
    }

def load_state_raw(dir_eeg: Path, base: str, state: str):
    """Prefer <base>_{EO|EC}_clean_raw.fif; otherwise crop state from <base>_clean_raw.fif."""
    direct = dir_eeg / f"{base}_{state}_clean_raw.fif"
    if direct.exists():
        try:
            return mne.io.read_raw_fif(direct, preload=True, verbose="ERROR").pick("eeg")
        except Exception as e:
            print(f"WARNING: cannot open {direct.name}: {e}")

    concat = dir_eeg / f"{base}_clean_raw.fif"
    if concat.exists():
        try:
            raw_all = mne.io.read_raw_fif(concat, preload=True, verbose="ERROR")
            if not raw_all.annotations or len(raw_all.annotations) == 0:
                return None
            sf = float(raw_all.info.get("sfreq", 250.0))
            eps = 1.0 / sf
            pieces = []
            for desc, onset, dur in zip(raw_all.annotations.description,
                                        raw_all.annotations.onset,
                                        raw_all.annotations.duration):
                if desc == f"visual_state:{state}" and dur > 0:
                    t0 = float(onset)
                    t1 = min(float(onset + dur), float(raw_all.times[-1]) - eps)
                    if t1 > t0:
                        pieces.append(raw_all.copy().crop(tmin=t0, tmax=t1))
            if not pieces:
                return None
            raw = mne.concatenate_raws(pieces, verbose="ERROR") if len(pieces) > 1 else pieces[0]
            return raw.pick("eeg")
        except Exception as e:
            print(f"WARNING: cannot crop {concat.name}: {e}")
    return None

def psd_band_by_channel(raw, fmin_band, fmax_band):
    """Welch PSD per channel; returns (abs, rel, channel_names)."""
    raw = raw.copy().pick("eeg")
    sf = float(raw.info["sfreq"])
    fmax_eff = min(PSD_FMAX, sf / 2.0 - 1.0)
    n_per_seg = max(16, int(round(WELCH_SEG_SEC * sf)))
    n_overlap = int(round(WELCH_OVERLAP * n_per_seg))
    spec = raw.compute_psd(
        method="welch",
        fmin=PSD_FMIN, fmax=fmax_eff,
        n_fft=n_per_seg, n_per_seg=n_per_seg, n_overlap=n_overlap,
        picks="eeg", verbose="ERROR",
    )
    psds, freqs = spec.get_data(return_freqs=True)
    band_mask = (freqs >= fmin_band) & (freqs <= fmax_band)
    total_mask = (freqs >= PSD_FMIN) & (freqs <= fmax_eff)
    abs_band = psds[:, band_mask].mean(axis=1)
    total = psds[:, total_mask].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_band = abs_band / np.where(total > 0, total, np.nan)
    return abs_band, rel_band, raw.ch_names

def make_info(ch_names, sfreq=250.0):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1020")
    return info

# --- collect PRE/POST bases
concat_files = sorted(PROCESSED_DIR.rglob("*_desc-preproc_clean_raw.fif"))
bank = {}  # (sub_num, state) -> {"PRE": (dir_eeg, base), "POST": (...)}
for f in concat_files:
    info = parse_bids_from_name(f)
    if not info:
        continue
    dir_eeg = f.parent
    base = info["base"]
    sub = subject_two_digit(info["sub"])
    sess = session_label_prepost(info.get("ses"))
    if sess not in {"PRE", "POST"}:
        continue
    for state in STATES:
        bank.setdefault((sub, state), {})[sess] = (dir_eeg, base)

pairs = [((sub, state), sesmap) for (sub, state), sesmap in bank.items() if "PRE" in sesmap and "POST" in sesmap]
print(f"Valid PRE/POST pairs: {len(pairs)}")

# --- cache per subject/state/band
cache = {}
for (sub, state), sesmap in pairs:
    for sess in ("PRE", "POST"):
        dir_eeg, base = sesmap[sess]
        raw = load_state_raw(dir_eeg, base, state)
        if raw is None:
            cache[(sub, state, sess, None)] = None
            continue
        try:
            raw.set_montage("standard_1020", on_missing="ignore")
        except Exception:
            pass
        for band, (fmin_b, fmax_b) in BANDS.items():
            abs_v, rel_v, chs = psd_band_by_channel(raw, fmin_b, fmax_b)
            cache[(sub, state, sess, band)] = {"abs": abs_v, "rel": rel_v, "ch": chs}

def compute_post_minus_pre_deltas(metric: str):
    """metric: 'rel' -> Δ% ; 'abs' -> Δ dB."""
    results = {st: {b: [] for b in BANDS} for st in STATES}
    for (sub, state), sesmap in pairs:
        grp = infer_group_from_subject(sub)
        for band in BANDS.keys():
            pre = cache.get((sub, state, "PRE", band))
            post = cache.get((sub, state, "POST", band))
            if pre is None or post is None:
                continue
            ch_pre, ch_post = pre["ch"], post["ch"]
            common = [ch for ch in ch_pre if ch in ch_post]
            if len(common) < 8:
                continue
            idx_pre = [ch_pre.index(ch) for ch in common]
            idx_post = [ch_post.index(ch) for ch in common]

            if metric == "rel":
                pre_rel = pre["rel"][idx_pre]
                post_rel = post["rel"][idx_post]
                with np.errstate(divide="ignore", invalid="ignore"):
                    delta = 100.0 * (post_rel - pre_rel) / np.where(pre_rel != 0, pre_rel, np.nan)
            else:
                post_db = 10 * np.log10(post["abs"][idx_post])
                pre_db = 10 * np.log10(pre["abs"][idx_pre])
                delta = post_db - pre_db

            results[state][band].append({"sub": sub, "group": grp, "ch": common, "delta": delta})
    return results

FIGSIZE = (13.0, 7.8)
GRID_WSPACE = 0.12
GRID_HSPACE = 0.30
CBAR_WIDTH = 0.045
TITLE_Y = 0.99
MARGINS = dict(left=0.08, right=0.95, top=0.90, bottom=0.06)
COL_TITLE_PAD = 10

def plot_topomap_grid_for_band(band: str, results, metric: str, out_dir: Path):
    """2×3 grid; one-sample t vs 0 on Δ with FDR q=.05."""
    row_data, ch_common = {}, {}

    for state in STATES:
        lst = results[state][band]
        if not lst:
            row_data[state] = None
            ch_common[state] = []
            continue
        inter = set(lst[0]["ch"])
        for d in lst[1:]:
            inter &= set(d["ch"])
        inter = [ch for ch in lst[0]["ch"] if ch in inter]
        ch_common[state] = inter

        vals, stacks = {}, {}
        for grp in GROUPS:
            ds = [d for d in lst if d["group"] == grp]
            if not ds or len(inter) < 8:
                vals[grp] = None
                stacks[grp] = None
                continue
            mats = []
            for d in ds:
                idx = [d["ch"].index(ch) for ch in inter]
                mats.append(d["delta"][idx])
            X = np.vstack(mats)
            vals[grp] = np.nanmean(X, axis=0)
            stacks[grp] = X
        row_data[state] = dict(vals=vals, stacks=stacks)

    all_vals = []
    for state in STATES:
        rd = row_data.get(state)
        if not rd:
            continue
        for grp in GROUPS:
            vec = rd["vals"].get(grp) if rd["vals"] else None
            if vec is not None:
                all_vals.append(np.abs(vec))
    if not all_vals:
        print(f"[{band}] Not enough data.")
        return

    vmax = float(np.nanpercentile(np.concatenate(all_vals), 95))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.concatenate(all_vals)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    vmax_cap = 30.0 if metric == "rel" else 2.0
    vmax = min(vmax, vmax_cap)
    vlim = (-vmax, vmax)

    fig = plt.figure(figsize=FIGSIZE)
    gs = gridspec.GridSpec(
        nrows=2, ncols=4, figure=fig,
        width_ratios=[1, 1, 1, CBAR_WIDTH / (3 + CBAR_WIDTH)],
        wspace=GRID_WSPACE, hspace=GRID_HSPACE,
    )
    axes = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j])
    cax = fig.add_subplot(gs[:, 3])

    for j, grp in enumerate(GROUPS):
        axes[0, j].set_title(grp, fontsize=13, pad=COL_TITLE_PAD)

    im = None
    for i, state in enumerate(STATES):
        rd = row_data.get(state)
        inter = ch_common.get(state, [])
        for j, grp in enumerate(GROUPS):
            ax = axes[i, j]
            if not rd or rd["vals"].get(grp) is None or len(inter) < 8:
                ax.axis("off")
                continue
            vec = rd["vals"][grp]
            X = rd["stacks"][grp]
            tval, p = ttest_1samp(X, popmean=0.0, axis=0, nan_policy="omit")
            rej, _ = fdr_correction(p, alpha=0.05)
            mask = rej.astype(bool)
            info = make_info(inter, sfreq=250.0)
            im, _ = mne.viz.plot_topomap(
                vec, info, cmap=config.CMAP_TOPO, vlim=vlim,
                contours=0, sensors=True, axes=ax, show=False,
                mask=mask, mask_params=dict(marker="o", markersize=5, alpha=0.0),
            )

    fig.text(0.06, 0.72, STATES[0], va="center", ha="left", fontsize=13)
    fig.text(0.06, 0.26, STATES[1], va="center", ha="left", fontsize=13)

    cb = fig.colorbar(im, cax=cax)
    if metric == "rel":
        cb.set_label("Δ% (POST−PRE) — Increase (red) / Decrease (blue)")
    else:
        cb.set_label("Δ dB (POST−PRE) — Increase (red) / Decrease (blue)")

    metric_title = "Relative Difference (POST−PRE, %)" if metric == "rel" else "Absolute Difference (POST−PRE, dB)"
    fig.suptitle(f"Topomaps — {band}  |  {metric_title}  |  FDR q=.05", fontsize=17, y=TITLE_Y)

    plt.subplots_adjust(**MARGINS)
    tag = "relDiff" if metric == "rel" else "absDiff"
    fname = out_dir / f"topogrid_{tag}_{band}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

# --- run (REL & ABS)
results_rel = compute_post_minus_pre_deltas(metric="rel")
results_abs = compute_post_minus_pre_deltas(metric="abs")

for band in BANDS_ORDER:
    plot_topomap_grid_for_band(band, results_rel, metric="rel", out_dir=OUT_REL)
    plot_topomap_grid_for_band(band, results_abs, metric="abs", out_dir=OUT_ABS)

print("Topomaps generated (REL% and ABS dB, with FDR masking).")
