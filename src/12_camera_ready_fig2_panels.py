"""Camera-ready Fig. 2b (MII by network, real values) and Fig. 2f (per-subject weights = Table II)."""
import json
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ['/System/Library/Fonts/Supplemental/Times New Roman.ttf', '/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf']:
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)  # falls back to the default serif font elsewhere
plt.rcParams.update({'font.family': 'Times New Roman', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8})
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs' / 'camera_ready' / 'fig2'
OUT.mkdir(parents=True, exist_ok=True)
d = json.load(open(ROOT / 'runs' / 'camera_ready' / 'fig2_data.json'))
DPI = 600

# ---- b: MII by network ----
nets = ['Vis', 'Som', 'DAN', 'VAN', 'Lim', 'FPN', 'DMN']
cols = ['F0F9FC', '5DBA8A', 'D4EFEA', 'A8DED1', '449C61', '7DCBB2', '267145']
m = [d['MII'][n]['mean'] for n in nets]
sd = [d['MII'][n]['sd'] for n in nets]
fig, ax = plt.subplots(figsize=(1066 / DPI, 793 / DPI), dpi=DPI)
y = np.arange(len(nets))
ax.barh(y, m, xerr=sd, color=['#' + c for c in cols], edgecolor='#9AA0A6', linewidth=0.4, height=0.78,
        error_kw=dict(ecolor='#333333', elinewidth=0.6, capsize=1.2, capthick=0.6))
for yi, (v, s) in enumerate(zip(m, sd)):
    ax.text(v + s + 0.015, yi, f'{v:.3f}' if nets[yi] == 'DMN' else f'{v:.2f}', va='center', fontsize=5.2, fontweight='bold')
ax.set_yticks(y, nets, fontsize=6.2)
ax.invert_yaxis()
ax.set_xlim(0, 0.9)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ['0.0', '0.2', '0.4', '0.6', '0.8'], fontsize=5.2)
ax.set_xlabel('Mean MII', fontsize=6.2, labelpad=1.5)
ax.tick_params(length=1.8, width=0.5, pad=1.5)
ax.grid(axis='x', color='#E6E6E6', linewidth=0.4)
ax.set_axisbelow(True)
fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.2)
fig.savefig(OUT / 'p_b_new.png', dpi=DPI, facecolor='white')

# ---- f: per-subject observed weights (Table II) ----
subs = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
W = np.array([d['weights'][s] for s in subs])
fig, ax = plt.subplots(figsize=(1428 / DPI, 850 / DPI), dpi=DPI)
x = np.arange(4); bw = 0.26
for j, (lab, c) in enumerate(zip(['Visual', 'Audio', 'Language'], ['267145', '5DBA8A', 'A8DED1'])):
    ax.bar(x + (j - 1) * bw, W[:, j], bw, color='#' + c, label=lab)
ax.axhline(1 / 3, color='#58D68D', linestyle=(0, (1, 1.5)), linewidth=1.0, label='Uniform (33.3%)')
ax.set_ylim(0.30, 0.36)
ax.set_yticks([0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36])
ax.set_yticklabels([f'{v:.2f}' for v in [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36]], fontsize=5.2)
ax.set_xticks(x, ['S1', 'S2', 'S3', 'S5'], fontsize=5.8)
ax.set_xlabel('Subject', fontsize=6.2, labelpad=1)
ax.set_ylabel('Attention Weight', fontsize=6.2, labelpad=1.5)
ax.tick_params(length=1.8, width=0.5, pad=1.5)
ax.legend(ncol=2, fontsize=4.9, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.16),
          handlelength=1.6, columnspacing=1.0, borderaxespad=0)
fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.2)
fig.savefig(OUT / 'p_f_new.png', dpi=DPI, facecolor='white')
print('ok')
