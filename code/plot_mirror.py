"""Mirror plots per BAND×ROI with individual points (no PRE↔POST links).
Reads : results/power/power_long_by_region_EO_EC.xlsx
Writes: results/plots/mirror_by_region_rel|abs/*.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

import config

# --- paths / config -----------------------------------------------------------
POWER_DIR = config.POWER_DIR
PLOTS_DIR = config.PLOTS_DIR
OUT_REL = PLOTS_DIR / "mirror_by_region_rel"
OUT_ABS = PLOTS_DIR / "mirror_by_region_abs"
OUT_REL.mkdir(parents=True, exist_ok=True)
OUT_ABS.mkdir(parents=True, exist_ok=True)

DATA_PATH = POWER_DIR / "power_long_by_region_EO_EC.xlsx"

BANDS_ORDER = config.BANDS_ORDER
ROIS_ORDER = config.ROIS_ORDER
GROUPS = config.GROUPS_ORDER
SESS = config.SESS_ORDER
STATES = config.VS_ORDER
PALETTE_VS = config.PALETTE_VS

# --- visual params ------------------------------------------------------------
Y_JITTER_STD = 0.035
X_PAD_FRAC = 0.12
RNG_SEED = 12345

SHOW_POINTS = True
POINT_SIZE = 28
POINT_ALPHA = 0.7
LABEL_OUTLIERS = True
OUTLIER_Z = 2.0
POINT_EDGE = "#000000"

# --- load ---------------------------------------------------------------------
if DATA_PATH.exists():
    df = pd.read_excel(DATA_PATH)
else:
    ts_csv = config.TS_DIR / "ts_power_long.csv"
    if not ts_csv.exists():
        raise FileNotFoundError(
            f"Neither {DATA_PATH} nor {ts_csv} found. "
            "Run calc_timeseries_power.py first."
        )
    print(f"[mirror] {DATA_PATH.name} not found — aggregating from {ts_csv.name}")
    _raw = pd.read_csv(ts_csv)
    df = (
        _raw.groupby(
            ["subject", "group", "session", "visual_state", "region", "band"],
            as_index=False,
        )
        .agg(power_abs=("power_abs", "mean"), power_rel=("power_rel", "mean"))
    )

need_base = {"subject", "group", "session", "visual_state", "region", "band"}
need_measures = {"power_rel", "power_abs"}
miss = (need_base | need_measures).difference(df.columns)
if miss:
    raise ValueError(f"Missing columns in {DATA_PATH.name}: {miss}")

df = df[df["group"].isin(GROUPS)].copy()
df["band"] = pd.Categorical(df["band"], BANDS_ORDER, ordered=True)
df["region"] = pd.Categorical(df["region"], ROIS_ORDER, ordered=True)
df["group"] = pd.Categorical(df["group"], GROUPS, ordered=True)
df["session"] = df["session"].str.upper().map({"PRE": "PRE", "POS": "POST", "POST": "POST"})
df["session"] = pd.Categorical(df["session"], SESS, ordered=True)
df["visual_state"] = pd.Categorical(df["visual_state"].str.upper(), STATES, ordered=True)
df["subject"] = df["subject"].astype(str).str.zfill(2)

# --- error functions ----------------------------------------------------------
def _se(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.nan if x.size == 0 else float(np.std(x, ddof=1) / np.sqrt(x.size))

try:
    from math import sqrt
    from scipy.stats import t

    def _ci95(x):
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        n = x.size
        if n < 2:
            return np.nan
        return float(t.ppf(0.975, n - 1) * (np.std(x, ddof=1) / sqrt(n)))

    ERR_LABEL = "ci95"
    err_func = _ci95
except Exception:
    ERR_LABEL = "se"
    err_func = _se
    print("SciPy not found: using SE instead of 95% CI.")

# --- axis formatter -----------------------------------------------------------
_sci_fmt = ScalarFormatter(useMathText=True)
_sci_fmt.set_powerlimits((-3, 3))

def sci_abs(x, pos):
    try:
        return _sci_fmt.format_data(abs(float(x)))
    except Exception:
        return ""

# --- summaries ----------------------------------------------------------------
def build_summary(df_use: pd.DataFrame, value_col: str):
    return (
        df_use.groupby(["band", "region", "group", "session", "visual_state"], as_index=False)[value_col]
        .agg(mean="mean", **{ERR_LABEL: err_func}, n="count")
    )

# --- plotting -----------------------------------------------------------------
def plot_mirror_with_points(band: str, region: str, out_dir: Path,
                            df_base: pd.DataFrame, summary: pd.DataFrame,
                            value_col: str):

    sub_bar = summary[(summary["band"] == band) & (summary["region"] == region)].copy()
    if sub_bar.empty:
        print(f"No data for {band} / {region}")
        return

    sub_bar["group"] = pd.Categorical(sub_bar["group"], GROUPS, ordered=True)
    sub_bar["session"] = pd.Categorical(sub_bar["session"], SESS, ordered=True)
    sub_bar["visual_state"] = pd.Categorical(sub_bar["visual_state"], STATES, ordered=True)

    sub_pts = df_base[(df_base["band"] == band) & (df_base["region"] == region)].copy()

    y_pos = {g: i for i, g in enumerate(GROUPS)}
    vshift = {"EO": +0.17, "EC": -0.17}
    rng = np.random.default_rng(RNG_SEED)

    fig, ax = plt.subplots(figsize=(12, 7))

    def signed_value(sess, val):
        return -val if sess == "PRE" else val

    for _, row in sub_bar.iterrows():
        g, s, st = row["group"], row["session"], row["visual_state"]
        y = y_pos[g] + vshift[st]
        x = signed_value(s, float(row["mean"]))
        err = float(row[ERR_LABEL]) if np.isfinite(row[ERR_LABEL]) else 0.0
        ax.barh(y, x, height=0.30, color=PALETTE_VS[st], edgecolor="k", alpha=0.95, zorder=1)
        if err > 0:
            ax.errorbar(x, y, xerr=err, fmt="none", ecolor="k", elinewidth=1, capsize=3, zorder=2)

    if SHOW_POINTS:
        pts = sub_pts.copy()
        pts["x_signed"] = pts.apply(lambda r: signed_value(r["session"], float(r[value_col])), axis=1)
        if LABEL_OUTLIERS:
            def _z(v):
                v = v.to_numpy(dtype=float)
                sd = v.std(ddof=1)
                return (v - v.mean()) / (sd if sd > 0 else np.nan)
            pts["_z"] = pts.groupby(["group", "visual_state", "session"])["x_signed"].transform(_z)
        for (g, st, s), chunk in pts.groupby(["group", "visual_state", "session"]):
            y_base = y_pos[g] + vshift[st]
            y_vals = y_base + rng.normal(0, Y_JITTER_STD, size=len(chunk))
            ax.scatter(
                chunk["x_signed"].values, y_vals,
                s=POINT_SIZE, c=PALETTE_VS[st], alpha=POINT_ALPHA,
                edgecolors=POINT_EDGE, linewidths=0.5, zorder=4
            )
            if LABEL_OUTLIERS and "_z" in chunk:
                for _, r in chunk.iterrows():
                    if np.isfinite(r["_z"]) and abs(r["_z"]) >= OUTLIER_Z:
                        ax.text(r["x_signed"], y_base + 0.08, str(r["subject"]),
                                fontsize=8, ha="center", va="bottom", color="#444", zorder=5)

    ax.set_yticks([y_pos[g] for g in GROUPS])
    ax.set_yticklabels(GROUPS)
    ax.axvline(0, color="k", lw=1)

    err_series = sub_bar[ERR_LABEL].fillna(0.0)
    x_from_mean = np.nanmax(np.abs(sub_bar["mean"])) if not sub_bar["mean"].isna().all() else 0.0
    x_from_mean_err = np.nanmax(np.abs(sub_bar["mean"] + err_series)) if not err_series.isna().all() else x_from_mean
    x_from_pts = np.nanmax(np.abs(sub_pts[value_col].to_numpy(dtype=float))) if not sub_pts.empty else 0.0
    xmax_local = float(np.nanmax([x_from_mean, x_from_mean_err, x_from_pts]))
    if not np.isfinite(xmax_local) or xmax_local <= 0:
        xmax_local = 1.0
    pad = xmax_local * X_PAD_FRAC
    ax.set_xlim(-(xmax_local + pad), (xmax_local + pad))

    ax.margins(x=0.02)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(-3, 3))
    ax.xaxis.set_major_formatter(FuncFormatter(sci_abs))

    unit = "Relative Power (within-ROI proportion)" if value_col == "power_rel" else r"Absolute Power ($\mu V^2/Hz$)"
    err_title = "95% CI" if ERR_LABEL == "ci95" else "SE"
    ax.set_xlabel(unit)
    ax.set_title(f"{'Relative' if value_col == 'power_rel' else 'Absolute'} Power — {band} — {region} ({err_title})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    from matplotlib.lines import Line2D
    cond_handles = [Line2D([0], [0], color=PALETTE_VS[st], lw=8, label=st) for st in STATES]
    sess_handles = [
        Line2D([0], [0], color="k", lw=6, label="PRE (left)"),
        Line2D([0], [0], color="k", lw=6, label="POST (right)", alpha=0.4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#777", markeredgecolor="#000",
               markersize=7, lw=0, label="Individuals"),
    ]
    leg1 = ax.legend(handles=cond_handles, title="Condition", loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=sess_handles, title="Session", loc="upper right", frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"mirror_{band}_{region.replace(' ', '_').replace('/', '-')}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# --- run ----------------------------------------------------------------------
for VALUE_COL, out_dir in [("power_rel", OUT_REL), ("power_abs", OUT_ABS)]:
    summary = build_summary(df, VALUE_COL)
    for band in BANDS_ORDER:
        for region in ROIS_ORDER:
            plot_mirror_with_points(
                band, region, out_dir,
                df_base=df, summary=summary,
                value_col=VALUE_COL
            )

print("Mirror plots generated for REL and ABS.")
