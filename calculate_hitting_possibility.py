from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm, gaussian_kde
import math

# ================================================================
# USER CONFIGURATION
# ================================================================
folder = Path(r"D:\Chao Luan\64 channel transmitter\Resonance distribution"
              r"\Resonance distribution\64_ch_transmitter")

# Spectral processing
wmin, wmax = 1536.0, 1539.0          # nm window for searching minima
power_in_db = False                  # convert power trace to dB?

# Target / spec
target_center_nm       = 1537.2   # target wavelength
pretrim_half_span_pm   = 10          # direct acceptance ± (pm)
trim_range_pm          = 100          # max red-shift tuning range (pm) (only upward shift)

# Probability estimation mode ('empirical' or 'gaussian')
mode_pre  = "empirical"
mode_post = "empirical"

# Channel / redundancy sweeps
k_max                = 64            # maximum channel count for sweep
r_max_plot           = 64            # maximum redundancy to show in Plot 2
r_search_max         = 500           # cap when solving required r
r_fixed              = 8             # (kept in case you still want old fig1 style)
k_fixed              = 64            # channel count used in Plot 2 (system yield vs r)
target_system_yield  = 0.99          # desired all-k success probability

# Normalization flag for Figures 3 & 4
normalize_fig3_fig4 = True   # Set False for absolute y-axes

# Output toggles
save_resonance_csv   = True
save_summary_csv     = True
summary_csv_name     = "redundancy_requirements.csv"
show_distribution_plots = True

# Histogram binning (for diagnostic distribution plot)
bin_width = 0.02  # nm

# ================================================================
# 1. LOAD DATA & EXTRACT RESONANCES
# ================================================================
mat_files = sorted(folder.glob("*.mat"))
if not mat_files:
    raise FileNotFoundError(f"No .mat files found in {folder}")

trace_idx, resonance_wls, spectra_cache = [], [], []
for idx, fp in enumerate(mat_files, start=1):
    data = loadmat(fp, squeeze_me=True)
    wl_key = next((k for k in data if "wav" in k.lower()), None)
    pw_key = next((k for k in data if "pow" in k.lower()), None)
    if wl_key is None or pw_key is None:
        print(f"Skipping {fp.name}: wavelength or power array missing.")
        continue

    wl = np.asarray(data[wl_key]).ravel()
    pw = np.asarray(data[pw_key]).ravel()
    if power_in_db:
        pw = 10 * np.log10(np.clip(pw, 1e-12, None))

    mask = (wl >= wmin) & (wl <= wmax)
    if not mask.any():
        print(f"Skipping {fp.name}: no samples in window {wmin}-{wmax} nm.")
        continue

    wl_sub, pw_sub = wl[mask], pw[mask]
    lam0 = wl_sub[np.argmin(pw_sub)]

    trace_idx.append(idx)
    resonance_wls.append(lam0)
    spectra_cache.append((wl_sub, pw_sub, lam0, pw_sub.min()))

res_wl = np.array(resonance_wls)
print(f"Loaded {len(res_wl)} valid resonances.")

if save_resonance_csv:
    pd.DataFrame({"trace_idx": trace_idx, "lambda_nm": res_wl}).to_csv(
        folder / "resonance_summary.csv", index=False)
    print("Resonance summary written.")

# ================================================================
# 2. SINGLE RING PROBABILITIES (PRE & POST TRIM)
# ================================================================
mu_hat, sigma_hat = norm.fit(res_wl)

pre_half_nm = pretrim_half_span_pm / 1000.0
pre_low  = target_center_nm - pre_half_nm
pre_high = target_center_nm + pre_half_nm

trim_range_nm = trim_range_pm / 1000.0
salvage_low  = target_center_nm - trim_range_nm - pre_half_nm  # lowest salvageable

in_direct   = (res_wl >= pre_low) & (res_wl <= pre_high)
in_salvage  = (res_wl >= salvage_low) & (res_wl < pre_low)
in_post     = in_direct | in_salvage
p_pre_emp   = in_direct.mean()
p_post_emp  = in_post.mean()

p_pre_gauss  = norm.cdf(pre_high, mu_hat, sigma_hat) - norm.cdf(pre_low,  mu_hat, sigma_hat)
p_post_gauss = norm.cdf(pre_high, mu_hat, sigma_hat) - norm.cdf(salvage_low, mu_hat, sigma_hat)

p_pre  = p_pre_emp  if mode_pre.lower()  == "empirical" else p_pre_gauss
p_post = p_post_emp if mode_post.lower() == "empirical" else p_post_gauss

print("\n=== Single-Ring Success Probabilities ===")
print(f"Target λ = {target_center_nm:.6f} nm")
print(f"Direct spec window (pre):      [{pre_low:.6f}, {pre_high:.6f}] nm  (±{pretrim_half_span_pm} pm)")
print(f"Red-shift trim range: +{trim_range_pm} pm -> salvage [{salvage_low:.6f}, {pre_low:.6f})")
print(f"Pre-trim  p_single (empirical={p_pre_emp:.4%}, gaussian={p_pre_gauss:.4%}) -> using {mode_pre}: {p_pre:.4%}")
print(f"Post-trim p_single (empirical={p_post_emp:.4%}, gaussian={p_post_gauss:.4%}) -> using {mode_post}: {p_post:.4%}")

# # ================================================================
# # 3. HELPER FUNCTIONS
# # ================================================================
# def per_channel_success(p_single: float, r: int) -> float:
#     if r <= 0:
#         return 0.0
#     return 1 - (1 - p_single)**r

# def system_success(p_single: float, r: int, k: int) -> float:
#     return per_channel_success(p_single, r)**k

# def required_redundancy(p_single: float, k: int, system_target: float) -> int:
#     if p_single <= 0:
#         return math.inf
#     if p_single >= 1:
#         return 1
#     p_chan_target = system_target**(1.0 / k)
#     r_real = math.log(1 - p_chan_target) / math.log(1 - p_single)
#     r_int = max(1, math.ceil(r_real))
#     return min(r_int, r_search_max)

# # ================================================================
# # 4. NEW FIGURE 1: MULTI-CURVE (LOG Y) SUCCESS vs k
# # ================================================================
# k_vals = np.arange(1, k_max + 1)

# # Case 1: No trim, 1 ring / channel
# sys_case1 = p_pre ** k_vals

# # Case 2: With trim, 1 ring / channel
# sys_case2 = p_post ** k_vals

# # Case 3: With trim, redundancy r = k (i.e., r grows with channel count)
# # System success = [1 - (1 - p_post)^k]^k
# sys_case3 = (1 - (1 - p_post)**k_vals) ** k_vals

# # Case 4: No trim, redundancy r = k
# sys_case4 = (1 - (1 - p_pre)**k_vals) ** k_vals

# plt.figure(figsize=(7,4))
# plt.plot(k_vals, sys_case1, 'o-',  label='No trim, r=1')
# plt.plot(k_vals, sys_case2, 's--', label='Trim, r=1')
# plt.plot(k_vals, sys_case3, 'd-',  label='Trim, r=k')
# plt.plot(k_vals, sys_case4, 'x--', label='No trim, r=k')
# plt.axhline(target_system_yield, color='k', ls=':', label=f'Target {target_system_yield:.2%}')
# plt.yscale('log')
# plt.xlabel('Number of channels k')
# plt.ylabel('System success probability (log scale)')
# plt.title('System success vs channel count (various redundancy & trimming cases)')
# plt.grid(alpha=.3, which='both', ls=':')
# plt.legend()
# plt.tight_layout()

# ================================================================
# 5. (Optional) OLD STYLE FIGURE (fixed r_fixed) — comment out if not needed
# ================================================================
# sys_pre_fixed_r  = [system_success(p_pre,  r_fixed, k) for k in k_vals]
# sys_post_fixed_r = [system_success(p_post, r_fixed, k) for k in k_vals]
# plt.figure(figsize=(7,4))
# plt.plot(k_vals, sys_pre_fixed_r,  'o-', label=f'Pre-trim (r={r_fixed})')
# plt.plot(k_vals, sys_post_fixed_r, 's--', label=f'Post-trim (r={r_fixed})')
# plt.axhline(target_system_yield, color='k', ls=':', label=f'Target sys yield={target_system_yield:.2%}')
# plt.xlabel('Number of channels k')
# plt.ylabel('System success probability')
# plt.title('System success vs channel count (fixed redundancy)')
# plt.grid(alpha=.3, ls=':')
# plt.legend()
# plt.tight_layout()

# ================================================================
# 6. FIGURE 2: SYSTEM SUCCESS vs r (fixed k_fixed)
# # ================================================================
# r_vals = np.arange(1, r_max_plot + 1)
# sys_pre_vs_r  = [system_success(p_pre,  r, k_fixed) for r in r_vals]
# sys_post_vs_r = [system_success(p_post, r, k_fixed) for r in r_vals]

# plt.figure(figsize=(7,4))
# plt.plot(r_vals, sys_pre_vs_r,  'o-', label=f'Pre-trim (k={k_fixed})')
# plt.plot(r_vals, sys_post_vs_r, 's--', label=f'Post-trim (k={k_fixed})')
# plt.axhline(target_system_yield, color='k', ls=':', label=f'Target sys yield={target_system_yield:.2%}')
# plt.xlabel('Redundancy per channel r')
# plt.ylabel('System success probability')
# plt.title(f'System success vs redundancy (k={k_fixed})')
# plt.grid(alpha=.3, ls=':')
# plt.legend()
# plt.tight_layout()
# ================================================================
# 6.  NEW  FIGURE 2 –  EXPECTED PASSING CHANNELS vs REDUNDANCY
#                     (64‑channel design, each channel has r rings)
# ================================================================
# n_channels = 64
# r_vals = np.arange(1, r_max_plot + 1)

# # Per‑channel pass probabilities for each redundancy value
# p_chan_pre  = [per_channel_success(p_pre,  r) for r in r_vals]
# p_chan_post = [per_channel_success(p_post, r) for r in r_vals]

# # Expected number of channels that meet the 1537.4 nm spec
# exp_pass_pre  = [n_channels * p for p in p_chan_pre]
# exp_pass_post = [n_channels * p for p in p_chan_post]

# plt.figure(figsize=(7, 4))
# plt.plot(r_vals, exp_pass_pre,  'o-', label='Untrimmed rings')
# plt.plot(r_vals, exp_pass_post, 's--', label='Trimmed rings')

# plt.xlabel('Redundancy per channel  r')
# plt.ylabel('Expected # of channels\nwithin 1537.4 nm spec (out of 64)')
# plt.title('Expected usable WDM channels vs redundancy')
# plt.grid(alpha=.3, ls=':')
# plt.legend()
# plt.tight_layout()

# # ‑‑‑‑‑‑ print a quick numeric table (optional) ‑‑‑‑‑‑
# table = pd.DataFrame({
#     'r': r_vals,
#     'exp_channels_pre':  exp_pass_pre,
#     'exp_channels_post': exp_pass_post
# })
# print("\n=== Expected passing channels (out of 64) ===")
# print(table.head(15).to_string(index=False))

# # ================================================================
# # 7. FIGURES 3 & 4: REQUIRED r VS k + TOTAL RINGS (normalize option)
# # ================================================================
# req_r_pre  = []
# req_r_post = []
# sys_yield_pre  = []
# sys_yield_post = []
# total_rings_pre  = []
# total_rings_post = []

# for k in k_vals:
#     rpre  = required_redundancy(p_pre,  k, target_system_yield)
#     rpost = required_redundancy(p_post, k, target_system_yield)
#     req_r_pre.append(rpre)
#     req_r_post.append(rpost)
#     sys_yield_pre.append(system_success(p_pre,  rpre,  k) if np.isfinite(rpre) else np.nan)
#     sys_yield_post.append(system_success(p_post, rpost, k) if np.isfinite(rpost) else np.nan)
#     total_rings_pre.append(k * rpre  if np.isfinite(rpre) else np.nan)
#     total_rings_post.append(k * rpost if np.isfinite(rpost) else np.nan)

# if normalize_fig3_fig4:
#     max_req = max([x for x in req_r_pre + req_r_post if np.isfinite(x)] or [1])
#     max_tot = max([x for x in total_rings_pre + total_rings_post if np.isfinite(x)] or [1])

#     req_r_pre_norm  = [x / max_req  if np.isfinite(x) else np.nan for x in req_r_pre]
#     req_r_post_norm = [x / max_req  if np.isfinite(x) else np.nan for x in req_r_post]
#     total_rings_pre_norm  = [x / max_tot if np.isfinite(x) else np.nan for x in total_rings_pre]
#     total_rings_post_norm = [x / max_tot if np.isfinite(x) else np.nan for x in total_rings_post]

#     plt.figure(figsize=(8, 6))
#     plt.plot(k_vals, req_r_pre_norm,  'o-', label='pre-trim')
#     plt.plot(k_vals, req_r_post_norm, 's--', label='post-trim')
#     plt.xlabel('Passive WDM channels', fontsize=18)
#     plt.ylabel('Design redundancy', fontsize=18)
    
#     # Set axis number (tick label) font size
#     plt.tick_params(axis='both', labelsize=18)
    
#     plt.grid(alpha=0.5)
#     plt.legend(fontsize=14)
#     plt.tight_layout()
#     plt.savefig('redundancy.svg', format='svg', transparent=True, dpi=600)


#     plt.figure(figsize=(7,4))
#     plt.plot(k_vals, total_rings_pre_norm,  'o-', label='Total rings pre-trim')
#     plt.plot(k_vals, total_rings_post_norm, 's--', label='Total rings post-trim')
#     plt.xlabel('Number of channels k')
#     plt.ylabel('Normalized total rings (÷ max)')
#     plt.title('Normalized total microrings vs channel count')
#     plt.grid(alpha=.3, ls=':')
#     plt.legend()
#     plt.tight_layout()

#     print(f"[Normalization] max required r = {max_req}, max total rings = {max_tot}")
# else:
#     plt.figure(figsize=(7,4))
#     plt.plot(k_vals, req_r_pre,  'o-', label='Required r (pre-trim)')
#     plt.plot(k_vals, req_r_post, 's--', label='Required r (post-trim)')
#     plt.xlabel('Number of channels k')
#     plt.ylabel('Required redundancy r')
#     plt.title(f'Required redundancy vs k for system yield ≥ {target_system_yield:.2%}')
#     plt.grid(alpha=.3, ls=':')
#     plt.legend()
#     plt.tight_layout()

#     plt.figure(figsize=(7,4))
#     plt.plot(k_vals, total_rings_pre,  'o-', label='Total rings pre-trim')
#     plt.plot(k_vals, total_rings_post, 's--', label='Total rings post-trim')
#     plt.xlabel('Number of channels k')
#     plt.ylabel('Total rings = k × r')
#     plt.title('Total microrings needed vs channel count')
#     plt.grid(alpha=.3, ls=':')
#     plt.legend()
#     plt.tight_layout()

# # ================================================================
# # 8. SUMMARY TABLE
# # ================================================================
# summary_df = pd.DataFrame({
#     "k_channels": k_vals,
#     "p_single_pre":  [p_pre]*len(k_vals),
#     "p_single_post": [p_post]*len(k_vals),
#     "required_r_pre": req_r_pre,
#     "required_r_post": req_r_post,
#     "system_yield_pre":  sys_yield_pre,
#     "system_yield_post": sys_yield_post,
#     "total_rings_pre":  total_rings_pre,
#     "total_rings_post": total_rings_post
# })

# print("\n=== Redundancy Requirement Summary (first 15 rows) ===")
# print(summary_df.head(15).to_string(index=False))

# if save_summary_csv:
#     summary_df.to_csv(folder / summary_csv_name, index=False)
#     print(f"\nSummary table written → {folder / summary_csv_name}")

# # ================================================================
# # 9. OPTIONAL DISTRIBUTION PLOTS
# # ================================================================
# if show_distribution_plots:
#     fig_sc, ax_sc = plt.subplots(figsize=(7,4))
#     ax_sc.scatter(trace_idx, res_wl, s=25, color='tab:blue')
#     ax_sc.axhspan(salvage_low, pre_low, color='green', alpha=.15, label='Salvage (tune up)')
#     ax_sc.axhspan(pre_low, pre_high, color='orange', alpha=.25, label='Direct spec')
#     ax_sc.axhline(target_center_nm, color='k', ls='--')
#     ax_sc.set(xlabel="Trace index", ylabel="λ₀ (nm)",
#               title=f"Resonances @ {target_center_nm} nm")
#     ax_sc.grid(alpha=.3, ls=':')
#     ax_sc.legend(fontsize=8)
#     fig_sc.tight_layout()

#     fig_hist, ax_hist = plt.subplots(figsize=(7,4))
#     edges = np.arange(res_wl.min(), res_wl.max()+bin_width, bin_width)
#     counts, bins, _ = ax_hist.hist(res_wl, bins=edges,
#                                    alpha=.7, color='tab:blue',
#                                    edgecolor='black', linewidth=.5)
#     kde = gaussian_kde(res_wl)
#     x_smooth = np.linspace(bins[0], bins[-1], 400)
#     bw = (bins[1] - bins[0]) if len(bins) > 1 else 1
#     ax_hist.plot(x_smooth, kde(x_smooth)*len(res_wl)*bw, color='black', lw=1.1, label='KDE')
#     pdf_y = norm.pdf(x_smooth, mu_hat, sigma_hat)*len(res_wl)*bw
#     ax_hist.plot(x_smooth, pdf_y, '--', color='red', lw=1.3, label='Gaussian fit')
#     ax_hist.axvspan(salvage_low, pre_low,  color='green', alpha=.15)
#     ax_hist.axvspan(pre_low, pre_high,     color='orange', alpha=.25)
#     ax_hist.axvline(target_center_nm, color='k', ls='--')
#     ax_hist.set(xlabel='Resonant wavelength λ₀ (nm)', ylabel='Count',
#                 title='λ₀ Distribution (direct vs salvageable)')
#     ax_hist.grid(alpha=.3, ls=':')
#     ax_hist.legend(fontsize=8)
#     fig_hist.tight_layout()

# plt.show()

# # ================================================================
# # 10. NOTES
# # ================================================================
# # - New Figure 1: Four scenarios on one log-scale plot.
# # - Case r=k illustrates aggressive redundancy scaling (area/power tradeoff).
# # - Figures 3 & 4 optionally normalized to compare shape, not magnitude.
# # - Single-ring probabilities p_pre, p_post control all curves.
# # ================================================================
# # 6b.  NEW FIGURE – Max passive‑WDM channels at fixed redundancy r_fixed_channels
# # ================================================================
# r_fixed_channels = 10      # <<–– fixed redundancy per channel
# k_search = np.arange(1, 1501)   # search up to 1500 channels; enlarge if needed

# # System success for each k with fixed redundancy
# sys_fixed_trimmed   = [system_success(p_post, r_fixed_channels, k) for k in k_search]
# sys_fixed_untrimmed = [system_success(p_pre,  r_fixed_channels, k) for k in k_search]

# # Find the largest k values meeting the target yield
# try:
#     max_k_trimmed   = k_search[np.where(np.array(sys_fixed_trimmed)   >= target_system_yield)][-1]
# except IndexError:
#     max_k_trimmed   = 0
# try:
#     max_k_untrimmed = k_search[np.where(np.array(sys_fixed_untrimmed) >= target_system_yield)][-1]
# except IndexError:
#     max_k_untrimmed = 0

# print(f"\n=== Max passive‑WDM channels @ r={r_fixed_channels} ===")
# print(f"Trimmed   devices : {max_k_trimmed} channels")
# print(f"Untrimmed devices : {max_k_untrimmed} channels")

# # ---------- Plot ----------
# plt.figure(figsize=(7, 4))
# plt.plot(k_search, sys_fixed_trimmed,   'o-',  label=f'Trimmed, r={r_fixed_channels}')
# plt.plot(k_search, sys_fixed_untrimmed, 's--', label=f'Untrimmed, r={r_fixed_channels}')
# plt.axhline(target_system_yield, color='k', ls=':', label=f'Target {target_system_yield:.0%}')

# plt.yscale('log')
# plt.xlabel('Number of passive WDM channels k')
# plt.ylabel('System success probability (log scale)')
# plt.title(f'System success vs k (fixed redundancy r={r_fixed_channels})')

# # Annotate max‑k points
# plt.axvline(max_k_trimmed,   color='tab:blue',  ls=':', lw=1)
# plt.axvline(max_k_untrimmed, color='tab:orange', ls=':', lw=1)
# plt.text(max_k_trimmed,   target_system_yield*0.7,
#          f'  max k = {max_k_trimmed}',   color='tab:blue')
# plt.text(max_k_untrimmed, target_system_yield*0.5,
#          f'  max k = {max_k_untrimmed}', color='tab:orange')

# plt.grid(alpha=.3, which='both', ls=':')
# plt.legend()
# plt.tight_layout()
