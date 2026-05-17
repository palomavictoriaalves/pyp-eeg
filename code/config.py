"""
Global configuration
"""

from pathlib import Path

# === Paths ====================================================================
SCRIPT_PATH   = Path(__file__).resolve()
PROJECT_ROOT  = SCRIPT_PATH.parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR   = PROJECT_ROOT / "results"
PLOTS_DIR     = RESULTS_DIR / "plots"
POWER_DIR     = RESULTS_DIR / "power"
TS_DIR        = RESULTS_DIR / "timeseries"
PROCESSED_DIR = RESULTS_DIR / "processed"

# === EEG processing ===========================================================
FILTER_LOW  = 0.5      # Hz
FILTER_HIGH = 50.0     # Hz
NOTCH_HZ    = 60       # Hz

# EO/EC blocks (seconds from recording start)
BLOCKS_WITH_STATE = [
    ("EO", (15, 135)),
    ("EC", (150, 270)),
    ("EO", (285, 405)),
    ("EC", (420, 540)),
]

# === Welch / PSD ==============================================================
WELCH_SEG_SEC = 4.0    # s
WELCH_OVERLAP = 0.5
PSD_FMIN      = 0.5    # Hz
PSD_FMAX      = 50.0   # Hz

# === Power export =============================================================
EXPORT_RELATIVE          = True
STANDARDIZE_DURATION_SEC = None  # e.g. 120.0
POWER_ABS_SCALE          = 1e12  # scale PSD from V^2/Hz to uV^2/Hz by default

# === Study design =============================================================
GROUP_ACTIVE  = {'01', '05', '07', '10', '15', '16', '19'}
GROUP_PASSIVE = {'03', '04', '06', '08', '11', '13', '21'}
GROUP_CONTROL = {'02', '09', '12', '14', '17', '18', '22'}

SUBJECTS_ORDER     = sorted(GROUP_ACTIVE | GROUP_PASSIVE | GROUP_CONTROL)
GROUP_ACTIVE_ORDER = sorted(GROUP_ACTIVE)
GROUP_PASSIVE_ORDER = sorted(GROUP_PASSIVE)
GROUP_CONTROL_ORDER = sorted(GROUP_CONTROL)

# Channels 
ACTIVE_CHANNELS = {
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "C3", "Cz", "C4",
    "FT9", "T7", "T8", "FT10", "TP9", "TP10",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
}

# === Bands & ROIs =============================================================
BANDS = {
    #"Delta": (0.1, 3.5),
    #"Theta": (4.0, 7.9),
    "Alpha": (8.0, 12.9),
    #"Beta":  (13.0, 30.0),
    #"Gamma": (30.1, 50.0),
}

REGIONS = {
    "Prefrontal":       ["Fp1", "Fp2"],
    "Frontal":          ["F7", "F3", "Fz", "F4", "F8"],
    "Frontocentral":    ["FC5", "FC1", "FC2", "FC6"],
    "Central":          ["C3", "Cz", "C4"],
    "Temporo-parietal": ["FT9", "T7", "T8", "FT10", "TP9", "TP10"],
    "Centro-parietal":  ["CP5", "CP1", "CP2", "CP6"],
    "Parietal":         ["P7", "P3", "Pz", "P4", "P8"],
    "Occipital":        ["O1", "Oz", "O2"],
}

# === Canonical orders / palettes =============================================
GROUPS        = ["Active", "Passive", "Control"]
SESSIONS      = ["PRE", "POST"]
VISUAL_STATES = ["EO", "EC"]

BANDS_ORDER  = list(BANDS.keys())
ROIS_ORDER   = list(REGIONS.keys())
GROUPS_ORDER = GROUPS
SESS_ORDER   = SESSIONS
VS_ORDER     = VISUAL_STATES

PALETTE_SESS  = {"PRE": "#72B2E7", "POST": "#F28E2B"}
PALETTE_VS    = {"EO": "#7FB3D5", "EC": "#1F618D"}
PALETTE_GROUP = {"Active": "#E07B39", "Passive": "#4878CF", "Control": "#6ACC65"}

COLOR_FDR = "#27AE60"   # significance markers in timeseries plots

# Colormaps — change here to propagate across all plots
CMAP_POWER     = "viridis"   # timeseries power heatmaps
CMAP_NETWORK   = "viridis"   # MDM network node coloring (df_hat)
CMAP_ADJACENCY = "YlGnBu"    # MDM adjacency heatmaps (edge frequency)
CMAP_TOPO      = "RdBu_r"    # topomaps (diverging, POST-PRE delta)

# === Time-series ==============================================================
TS_WIN_SEC        = 4.0
TS_STEP_SEC       = 1.0
TS_FDR_ALPHA      = 0.05
TS_MARK_SIG       = True
TS_GENERATE_PLOTS = True
TS4_GENERATE_CONTINUOUS_PLOTS = True
TS4_GENERATE_EO_COMBINED_PLOTS = True

# X-axis windows in seconds.
# State-level outputs use keys "EO"/"EC" via `results/timeseries/ts_power_long.csv`.
# `plot_timeseries.py` reconstructs block-level conditions
# ("EO_1", "EC_1", "EO_2", "EC_2") from the concatenated file annotations and
# places each block on its original recording-time axis.
TS_FIXED_X_WINDOWS = {
    "EO":   (0.0, 240.0),    # concatenated EO file time (EO1 + EO2), starting at 0 s
    "EC":   (0.0, 240.0),    # concatenated EC file time (EC1 + EC2), starting at 0 s
    "EO_1": (15.0, 135.0),   # first EO block in original recording time
    "EC_1": (150.0, 270.0),  # first EC block in original recording time
    "EO_2": (285.0, 405.0),  # second EO block in original recording time
    "EC_2": (420.0, 540.0),  # second EC block in original recording time
}

ROI_CHANNELS = REGIONS
# === MDMP =====================================================================
MDMP_ENABLED = True

# Run one or both metrics in a single call to code/calc_mdmp.py
MDMP_METRICS_TO_RUN = ("power_rel", "power_abs")

# Legacy fallback used only when MDMP_METRICS_TO_RUN is empty
MDMP_METRIC = "power_abs"

MDMP_INPUT_CSV = TS_DIR / "ts_power_long.csv"
MDMP_OUTPUT_DIR = RESULTS_DIR / "mdmp"

MDMP_GROUP_COLS = ("subject", "session", "visual_state", "band")
MDMP_GLOBAL_GROUP_COLS = ("group", "session", "visual_state", "band")
MDMP_NODE_COL = "region"
MDMP_TIME_COL = ""   # empty = auto-detect from input

MDMP_METHOD = "hc"
MDMP_MIN_T = 20
MDMP_MIN_NODES = 3
MDMP_MIN_SUBJECTS = 2
MDMP_NBF = 15
MDMP_DELTA_GRID = ()
MDMP_ALIGN_METHOD = "truncate"
MDMP_MAX_RUNS = None
MDMP_IGNORE_ENABLED_FLAG = False

# MDMP heatmap outputs.
# Static adjacency heatmaps are always generated by code/plot_mdmp_heatmaps.py
# unless the CLI is called with --skip-static.
# When enabled, dynamic output generates both GIFs and frame panels.
MDMP_HEATMAP_DYNAMIC_ENABLED = True
MDMP_HEATMAP_FRAME_COUNT = 10
MDMP_HEATMAP_FRAME_RANGE = (0, 9)  # inclusive timepoint/second range
MDMP_HEATMAP_FRAME_COLUMNS = 5
MDMP_HEATMAP_DYNAMIC_OUTPUT_DIR = PLOTS_DIR / "mdmp_heatmaps" / "dynamic"
MDMP_HEATMAP_DYNAMIC_DISTRIBUTION = "smoo"
MDMP_HEATMAP_DYNAMIC_CMAP = "RdBu_r"
MDMP_HEATMAP_GIF_FPS = 10
MDMP_HEATMAP_GIF_DPI = 100
MDMP_HEATMAP_FRAME_DPI = 250
