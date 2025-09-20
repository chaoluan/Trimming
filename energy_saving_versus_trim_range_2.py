from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# =========================
folder = Path(r"D:\Chao Luan\64 channel transmitter\Resonance distribution"
              r"\Resonance distribution\64_ch_transmitter")

# Search window (nm)
wmin, wmax = 1536.0, 1539.0
power_in_db = False

# Target wavelength (nm)
target_center_nm = 1537.4021

# Sweep trimming range (pm)
trim_start_pm = 0
trim_stop_pm  = 1000
trim_step_pm  = 10

# Outputs
save_csv  = True
csv_name  = "ratio_energy_saved_vs_trim.csv"
make_plot = True

# =========================
# PLOT STYLE — tweak these to change look & feel
# =========================
FIG_SIZE        = (8, 6)      # inches
SAVE_DPI        = 600
TRANSPARENT_BG  = True

AXES_LINEWIDTH  = 2.0         # axis box (spines) thickness
GRID_ON         = True
GRID_ALPHA      = 0.30
GRID_STYLE      = "--"

FONT_BASE       = 18
FONT_LABEL      = 18
FONT_LEGEND     = 18

TICK_MAJOR_SIZE = 6
TICK_MINOR_SIZE = 3
TICK_WIDTH      = 1.6         # tick line thickness

LINE_WIDTH      = 2.4
LINE_COLOR      = None        # set e.g. "tab:blue" or None to use default cycle

SCATTER_SIZE    = 56
SCATTER_ALPHA   = 0.9
SCATTER_EDGE    = "none"      # e.g. "k" for black edge

SHOW_BASELINE_POINT = True    # highlight the 0-pm point
BASELINE_MARKER_SIZE = 90

LEGEND_LOC      = "best"
XLABEL          = "Trimming range (pm)"
YLABEL          = "Relative energy saving"
TITLE           = None        # e.g. "Energy saved ratio vs trimming"

X_LIMITS        = None        # e.g. (0, 1000) or None for auto
Y_LIMITS        = (-0.02, 1.02)

# Apply global rcParams
plt.rcParams.update({
    "figure.figsize":     FIG_SIZE,
    "font.size":          FONT_BASE,
    "axes.labelsize":     FONT_LABEL,
    "legend.fontsize":    FONT_LEGEND,
    "axes.linewidth":     AXES_LINEWIDTH,
    "xtick.major.size":   TICK_MAJOR_SIZE,
    "ytick.major.size":   TICK_MAJOR_SIZE,
    "xtick.minor.size":   TICK_MINOR_SIZE,
    "ytick.minor.size":   TICK_MINOR_SIZE,
    "xtick.major.width":  TICK_WIDTH,
    "ytick.major.width":  TICK_WIDTH,
    "xtick.minor.width":  TICK_WIDTH * 0.8,
    "ytick.minor.width":  TICK_WIDTH * 0.8,
})

# =========================
# 1) LOAD .MAT & EXTRACT RESONANCES
# =========================
mat_files = sorted(folder.glob("*.mat"))
if not mat_files:
    raise FileNotFoundError(f"No .mat files found in {folder}")

res_wl = []
for fp in mat_files:
    data = loadmat(fp, squeeze_me=True)
    wl_key = next((k for k in data if "wav" in k.lower()), None)
    pw_key = next((k for k in data if "pow" in k.lower()), None)
    if wl_key is None or pw_key is None:
        print(f"[WARN] Skip {fp.name}: missing wavelength/power key")
        continue

    wl = np.asarray(data[wl_key]).ravel()
    pw = np.asarray(data[pw_key]).ravel()
    if power_in_db:
        pw = 10 * np.log10(np.clip(pw, 1e-12, None))

    mask = (wl >= wmin) & (wl <= wmax)
    if not mask.any():
        print(f"[WARN] Skip {fp.name}: no samples in [{wmin},{wmax}] nm")
        continue

    wl_sub, pw_sub = wl[mask], pw[mask]
    lam0 = float(wl_sub[int(np.argmin(pw_sub))])  # resonance valley
    res_wl.append(lam0)

if not res_wl:
    raise RuntimeError("No valid resonances found within the window.")
res_wl = np.array(res_wl, dtype=float)

# =========================
# 2) FILTER: ONLY λ < target (red-shift cases)
# =========================
mask_below = res_wl < target_center_nm
if not np.any(mask_below):
    print("[INFO] No below-target resonances; nothing to compute.")
    if save_csv:
        pd.DataFrame(columns=["trim_cap_pm","ratio_energy_saved"]).to_csv(folder / csv_name, index=False)
    raise SystemExit(0)

res_below = res_wl[mask_below]

# =========================
# 3) INITIAL MEAN & INITIAL GAP (denominator)
# =========================
mean_before_nm = float(np.mean(res_below))
initial_gap_pm = max((target_center_nm - mean_before_nm) * 1000.0, 0.0)  # pm (clamped)
print(f"[INFO] Considered devices: {len(res_below)}")
print(f"[INFO] Mean before trim : {mean_before_nm:.6f} nm")
print(f"[INFO] Target          : {target_center_nm:.6f} nm")
print(f"[INFO] Initial gap     : {initial_gap_pm:.3f} pm (denominator)")

# Guard: if initial gap is 0, ratio is undefined → all ratios = 0
if initial_gap_pm == 0:
    print("[INFO] Initial gap is zero; all ratios set to 0.")
    trim_caps_pm = np.arange(trim_start_pm, trim_stop_pm + trim_step_pm, trim_step_pm, dtype=float)
    out = pd.DataFrame({"trim_cap_pm": trim_caps_pm, "ratio_energy_saved": np.zeros_like(trim_caps_pm)})
    if save_csv:
        out.to_csv(folder / csv_name, index=False)
        print(f"[INFO] Saved CSV: {folder / csv_name}")
    if make_plot:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot(out["trim_cap_pm"], out["ratio_energy_saved"], linewidth=LINE_WIDTH, label="Relative energy saving", color=LINE_COLOR)
        ax.scatter(out["trim_cap_pm"], out["ratio_energy_saved"], s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                   edgecolors=SCATTER_EDGE, label="data points")
        if GRID_ON: ax.grid(alpha=GRID_ALPHA, linestyle=GRID_STYLE)
        ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
        if TITLE: ax.set_title(TITLE)
        if X_LIMITS: ax.set_xlim(*X_LIMITS)
        if Y_LIMITS: ax.set_ylim(*Y_LIMITS)
        for s in ax.spines.values(): s.set_linewidth(AXES_LINEWIDTH)
        ax.tick_params(width=TICK_WIDTH)
        ax.legend(loc=LEGEND_LOC)
        fig.tight_layout()
        fig.savefig(folder / "ratio_energy_saved_vs_trim.svg", format="svg", dpi=SAVE_DPI, transparent=TRANSPARENT_BG)
        print(f"[INFO] Saved: {folder / 'ratio_energy_saved_vs_trim.svg'}")
    raise SystemExit(0)

# Precompute per-device red need (nm)
delta_nm = target_center_nm - res_below  # >0

# =========================
# 4) SWEEP TRIM: compute ratio of energy saved
# =========================
trim_caps_pm = np.arange(trim_start_pm, trim_stop_pm + trim_step_pm, trim_step_pm, dtype=float)
rows = []

for cap_pm in trim_caps_pm:
    cap_nm = cap_pm / 1000.0

    # Apply trimming rule per device
    within = delta_nm <= cap_nm                   # can hit target
    final_lambda_nm = res_below.copy()
    final_lambda_nm[within]  = target_center_nm
    final_lambda_nm[~within] = res_below[~within] + cap_nm

    # Mean after & gap(C) in pm
    mean_after_nm = float(np.mean(final_lambda_nm))
    gap_pm = max((target_center_nm - mean_after_nm) * 1000.0, 0.0)

    # Ratio of energy saved = 1 - gap(C)/initial_gap
    ratio_saved = 1.0 - (gap_pm / initial_gap_pm)
    ratio_saved = float(np.clip(ratio_saved, 0.0, 1.0))

    rows.append({
        "trim_cap_pm":        float(cap_pm),
        "ratio_energy_saved": ratio_saved
    })

summary_df = pd.DataFrame(rows)

# Preview
print("\n=== Ratio of energy saved vs trimming (preview) ===")
print(summary_df.head(10).to_string(index=False))
print("...")

# Save CSV
if save_csv:
    out_path = folder / csv_name
    out_df = summary_df.copy()
    out_df["ratio_energy_saved"] = out_df["ratio_energy_saved"].round(6)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved CSV: {out_path}")

# =========================
# 5) PLOT (ratio only) — with full style control
# =========================
if make_plot:
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Line
    ax.plot(summary_df["trim_cap_pm"], summary_df["ratio_energy_saved"],
            linewidth=LINE_WIDTH, label="Energy saved ratio", color=LINE_COLOR)

    # Data points
    x = summary_df["trim_cap_pm"].to_numpy()
    y = summary_df["ratio_energy_saved"].to_numpy()
    ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
               edgecolors=SCATTER_EDGE, label="data points")

   
    # Axes cosmetics
    if GRID_ON: ax.grid(alpha=GRID_ALPHA, linestyle=GRID_STYLE)
    ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
    if TITLE: ax.set_title(TITLE)
    if X_LIMITS: ax.set_xlim(*X_LIMITS)
    if Y_LIMITS: ax.set_ylim(*Y_LIMITS)

    # Spine & tick thickness
    for s in ax.spines.values():
        s.set_linewidth(AXES_LINEWIDTH)
    ax.tick_params(which="both", width=TICK_WIDTH)

    # Legend
    ax.legend(loc=LEGEND_LOC)

    # Save
    fig.tight_layout()
    fig.savefig(folder / "ratio_energy_saved_vs_trim.svg", format="svg",
                dpi=SAVE_DPI, transparent=TRANSPARENT_BG)
    print(f"[INFO] Saved: {folder / 'ratio_energy_saved_vs_trim.svg'}")

    # Optional second export name to match your previous habit
    fig.savefig("energy_saving.svg", dpi=SAVE_DPI, transparent=TRANSPARENT_BG)
    plt.show()
