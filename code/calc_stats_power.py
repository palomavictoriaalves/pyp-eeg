"""Paired stats for ROI×Band power tables (REL/ABS).
- Configurable numeric precision
- Explicit outputs for EC vs EO and POST vs PRE
- delta_pct = % change relative to correct baseline (EO or PRE)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy import stats
import config

# ---- Precision control ----
PRECISION_MODE = "none"  # 'none' | 'fixed' | 'auto'
DECIMALS_MAP = {}
MAX_DECIMALS_AUTO = 6
FLOAT_FORMAT_NONE = "%.17g"

ALPHA = 0.05

POWER_DIR = config.POWER_DIR
RESULTS_DIR = config.RESULTS_DIR
OUT_ROOT = (RESULTS_DIR / "stats").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

REL_PATH = POWER_DIR / "power_wide_rel_EO_EC.csv"
ABS_PATH = POWER_DIR / "power_wide_abs_EO_EC.csv"

# ---- Precision helpers ----
def _infer_decimals(series: pd.Series, max_decimals: int = 6) -> int:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size <= 1:
        return 0
    x = np.unique(x)
    if x.size <= 1:
        return 0
    diffs = np.diff(np.sort(x))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return min(6, max_decimals)
    step = float(np.min(diffs))
    dec = int(np.ceil(max(0.0, -np.log10(step))))
    return int(min(dec, max_decimals))

def apply_precision(df: pd.DataFrame) -> pd.DataFrame:
    if PRECISION_MODE == "none":
        return df
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if PRECISION_MODE == "auto":
        for c in num_cols:
            out[c] = out[c].round(_infer_decimals(out[c], MAX_DECIMALS_AUTO))
    elif PRECISION_MODE == "fixed":
        for c, n in DECIMALS_MAP.items():
            if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].round(int(n))
    return out

# ---- Statistical helpers ----
def cohen_dz(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    if a.size < 3:
        return np.nan
    d = a - b
    sd = float(np.std(d, ddof=1))
    return float(np.mean(d) / sd) if sd > 0 else np.nan

def shapiro_p(diff):
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    if diff.size < 3:
        return np.nan
    try:
        return float(stats.shapiro(diff).pvalue)
    except Exception:
        return np.nan

def _wilcoxon_p(a, b):
    try:
        return float(stats.wilcoxon(a - b, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def choose_test(a, b, p_shapiro, alpha=ALPHA, force_test=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if force_test == "t_paired":
        try:
            _, pval = stats.ttest_rel(a, b, nan_policy="omit")
            return "t_paired", float(pval)
        except Exception:
            return "t_paired", np.nan
    if force_test == "wilcoxon":
        return "wilcoxon", _wilcoxon_p(a, b)

    if np.isnan(p_shapiro):
        return "wilcoxon", _wilcoxon_p(a, b)
    if p_shapiro >= alpha:
        try:
            _, pval = stats.ttest_rel(a, b, nan_policy="omit")
            return "t_paired", float(pval)
        except Exception:
            return "t_paired", np.nan
    return "wilcoxon", _wilcoxon_p(a, b)

def p_to_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

def discover_roi_band_columns(df, bands=None, regions=None):
    out = []
    pat = re.compile(r"^(.+?)__([A-Za-z]+)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            reg, band = m.group(1), m.group(2)
            if (regions is None or reg in regions) and (bands is None or band in bands):
                out.append((c, reg, band))
    return out

def normalize_base_columns(df):
    if "session" in df.columns:
        df["session"] = df["session"].astype(str).str.upper().replace({"POS": "POST"})
    if "visual_state" in df.columns:
        df["visual_state"] = df["visual_state"].astype(str).str.upper()
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str).str.zfill(2)
    if "group" not in df.columns:
        df["group"] = "ALL"
    return df

def longify(df, bands=None, regions=None):
    metrics = discover_roi_band_columns(df, bands=bands, regions=regions)
    if not metrics:
        return None
    value_cols = [m[0] for m in metrics]
    idv = ["subject", "group", "session", "visual_state"]
    if "base_file" in df.columns:
        idv.append("base_file")
    long = df.melt(id_vars=idv, value_vars=value_cols, var_name="roi_band", value_name="value")
    rb = long["roi_band"].str.split("__", n=1, expand=True)
    long["region"] = rb[0]
    long["band"] = rb[1]
    return long

def paired_arrays(df_block, pair_col, level_A, level_B):
    piv = df_block.pivot_table(index="subject", columns=pair_col, values="value", aggfunc="mean")
    if level_A not in piv.columns or level_B not in piv.columns:
        return None
    sub = piv.dropna(subset=[level_A, level_B], how="any")
    if sub.empty:
        return None
    A = sub[level_A].values
    B = sub[level_B].values
    return A, B, len(sub)

def paired_summary(df_block, pair_col, level_A, level_B, force_test=None):
    paired = paired_arrays(df_block, pair_col, level_A, level_B)
    if paired is None:
        return None
    A, B, n = paired
    mean_A = float(np.mean(A)) if A.size else np.nan
    mean_B = float(np.mean(B)) if B.size else np.nan
    delta = float(mean_A - mean_B) if np.isfinite(mean_A) and np.isfinite(mean_B) else np.nan
    p_norm = shapiro_p(A - B)
    test_used, p_val = choose_test(A, B, p_norm, alpha=ALPHA, force_test=force_test)
    dz = cohen_dz(A, B)
    return dict(mean_A=mean_A, mean_B=mean_B, delta=delta,
                normality_test="Shapiro-Wilk", normality_p=p_norm,
                test_used=test_used, p_value=p_val,
                sig=p_to_stars(p_val), d=dz, n=n)

def pick_global_test_mode(long: pd.DataFrame, alpha: float = ALPHA) -> str:
    """Choose one global test family for the full analysis table.

    If any paired comparison violates normality, or if normality is undefined,
    Wilcoxon is used for all rows. Otherwise, a paired t-test is used
    throughout the table.
    """
    pvals = []

    for _, d in long.groupby(["group", "session", "region", "band"], dropna=False):
        paired = paired_arrays(d, pair_col="visual_state", level_A="EC", level_B="EO")
        if paired is None:
            continue
        A, B, _ = paired
        pvals.append(shapiro_p(A - B))

    for _, d in long.groupby(["group", "visual_state", "region", "band"], dropna=False):
        paired = paired_arrays(d, pair_col="session", level_A="POST", level_B="PRE")
        if paired is None:
            continue
        A, B, _ = paired
        pvals.append(shapiro_p(A - B))

    if not pvals:
        return "wilcoxon"

    pvals = np.asarray(pvals, float)
    has_non_normal = np.any(~np.isfinite(pvals)) or np.any(pvals < alpha)
    return "wilcoxon" if has_non_normal else "t_paired"

# ---- Runner ----
def run_for_table(path_csv: Path, tag: str):
    if not path_csv.exists():
        print(f"[{tag}] file not found: {path_csv.name} — skipping.")
        return
    df = pd.read_csv(path_csv)
    df = normalize_base_columns(df)
    long = longify(df)
    if long is None:
        print(f"[{tag}] no ROI__Band columns found in {path_csv.name}.")
        return
    global_test_mode = pick_global_test_mode(long, alpha=ALPHA)
    print(f"[{tag}] global test mode: {global_test_mode}")

    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # EC vs EO
    rows = []
    for (g, sess, reg, band), d in long.groupby(["group", "session", "region", "band"], dropna=False):
        res = paired_summary(
            d, pair_col="visual_state", level_A="EC", level_B="EO",
            force_test=global_test_mode,
        )
        if res is None:
            continue
        base = res["mean_B"]
        delta_pct = (res["delta"] / base * 100.0) if (base and np.isfinite(base) and base != 0) else np.nan
        rows.append(dict(group=g, session=sess, region=reg, band=band, compare="EC-EO",
                         mean_EC=res["mean_A"], mean_EO=res["mean_B"],
                         delta_EC_minus_EO=res["delta"], delta_pct=delta_pct,
                         normality_test=res["normality_test"], normality_p=res["normality_p"],
                         test_used=res["test_used"], p_value=res["p_value"],
                         sig=res["sig"], d=res["d"], n=res["n"]))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = apply_precision(out).sort_values(["group", "session", "region", "band"])
        float_fmt = FLOAT_FORMAT_NONE if PRECISION_MODE == "none" else None
        out.to_csv(out_dir / f"stats_{tag}_EOvsEC.csv", index=False, float_format=float_fmt)
        print(f"[{tag}] EC vs EO saved.")
    else:
        print(f"[{tag}] EC vs EO: no results.")

    # POST vs PRE
    rows = []
    for (g, state, reg, band), d in long.groupby(["group", "visual_state", "region", "band"], dropna=False):
        res = paired_summary(
            d, pair_col="session", level_A="POST", level_B="PRE",
            force_test=global_test_mode,
        )
        if res is None:
            continue
        base = res["mean_B"]
        delta_pct = (res["delta"] / base * 100.0) if (base and np.isfinite(base) and base != 0) else np.nan
        rows.append(dict(group=g, visual_state=state, region=reg, band=band, compare="POST-PRE",
                         mean_POST=res["mean_A"], mean_PRE=res["mean_B"],
                         delta_POST_minus_PRE=res["delta"], delta_pct=delta_pct,
                         normality_test=res["normality_test"], normality_p=res["normality_p"],
                         test_used=res["test_used"], p_value=res["p_value"],
                         sig=res["sig"], d=res["d"], n=res["n"]))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = apply_precision(out).sort_values(["group", "visual_state", "region", "band"])
        float_fmt = FLOAT_FORMAT_NONE if PRECISION_MODE == "none" else None
        out.to_csv(out_dir / f"stats_{tag}_POSTvsPRE.csv", index=False, float_format=float_fmt)
        print(f"[{tag}] POST vs PRE saved.")
    else:
        print(f"[{tag}] POST vs PRE: no results.")

# ---- Main ----
if __name__ == "__main__":
    run_for_table(REL_PATH, "rel")
    run_for_table(ABS_PATH, "abs")
    print("\nMinimal statistics completed for REL and ABS.")
