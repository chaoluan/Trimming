import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ───────────────────────────────────
# FIGURE STYLE — adjust here
# ───────────────────────────────────
FIG_SIZE          = (8, 6)
FONT_SIZE_BASE    = 18
FONT_SIZE_LABEL   = 18
FONT_SIZE_TITLE   = 18
FONT_SIZE_LEGEND  = 12
GRID_ALPHA        = 0.30
GRID_STYLE        = '-'
COLOR_PRE         = 'tab:orange'
COLOR_POST        = 'tab:blue'
MARKER_PRE        = 'o'
MARKER_POST       = 's'
LEGEND_LOC        = 'best'

plt.rcParams.update({
    'figure.figsize'  : FIG_SIZE,
    'font.size'       : FONT_SIZE_BASE,
    'axes.labelsize'  : FONT_SIZE_LABEL,
    'axes.titlesize'  : FONT_SIZE_TITLE,
    'legend.fontsize' : FONT_SIZE_LEGEND,
    'xtick.labelsize' : FONT_SIZE_BASE,
    'ytick.labelsize' : FONT_SIZE_BASE,
})

# ───────────────────────────────────
# INPUT PARAMETERS
# ───────────────────────────────────
K        = 64          # total designed WDM channels
p_pre    = 0.0468      # per‑ring success prob (untrimmed)
p_post   = 0.14        # per‑ring success prob (trimmed)
r_max    = 100          # max redundancy shown on x‑axis

# ───────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────
def p_channel(p_single: float, r: int) -> float:
    return 1 - (1 - p_single)**r

def expected_channels(p_single: float, r: int, K: int = 64) -> float:
    return K * p_channel(p_single, r)

def min_r_for_target(p_single: float, n_target: int,
                     K: int = 64, r_cap: int = 64) -> int:
    for r in range(1, r_cap + 1):
        if expected_channels(p_single, r, K) >= n_target:
            return r
    return np.inf

# ───────────────────────────────────
# FIGURE 1 — dots only
# ───────────────────────────────────
r_vals   = np.arange(1, r_max + 1)
exp_pre  = [expected_channels(p_pre,  r, K) for r in r_vals]
exp_post = [expected_channels(p_post, r, K) for r in r_vals]

plt.figure()
plt.plot(r_vals, exp_pre,  marker=MARKER_PRE , linestyle='None',
         color=COLOR_PRE , label='Untrimmed')
plt.plot(r_vals, exp_post, marker=MARKER_POST, linestyle='None',
         color=COLOR_POST, label='Trimmed')
plt.xlabel('Redundancy per channel')
plt.ylabel(f'Passive WDM channels')
# plt.title('Expected usable WDM channels vs redundancy')
plt.grid(alpha=GRID_ALPHA, linestyle=GRID_STYLE)
plt.legend(loc=LEGEND_LOC)
plt.tight_layout()
plt.savefig('expected_usable_channels.svg', format='svg', transparent=True, dpi=600)

# ───────────────────────────────────
# FIGURE 2 — dots only
# ───────────────────────────────────
n_vals      = np.arange(1, K + 1)
req_r_pre   = [min_r_for_target(p_pre,  n, K, r_max) for n in n_vals]
req_r_post  = [min_r_for_target(p_post, n, K, r_max) for n in n_vals]

plt.figure()
plt.plot(n_vals, req_r_pre,  marker=MARKER_PRE , linestyle='None',
         color=COLOR_PRE , label='Untrimmed')
plt.plot(n_vals, req_r_post, marker=MARKER_POST, linestyle='None',
         color=COLOR_POST, label='Trimmed')
plt.xlabel('Passive WDM channels')
plt.ylabel('Design redundancy')
# plt.title('Redundancy required vs target channels')
plt.grid(alpha=GRID_ALPHA, linestyle=GRID_STYLE)
plt.legend(loc=LEGEND_LOC)
plt.tight_layout()
plt.savefig('required_redundancy.svg', format='svg', transparent=True, dpi=600)

# ───────────────────────────────────
# OPTIONAL quick table preview
# ───────────────────────────────────
table = pd.DataFrame({
    'r': r_vals,
    'expected_channels_pre':  np.round(exp_pre,  2),
    'expected_channels_post': np.round(exp_post, 2)
})
print("\nFirst 10 rows of expected channel table:")
print(table.head(10).to_string(index=False))

plt.show()








