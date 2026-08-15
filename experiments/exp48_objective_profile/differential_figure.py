"""The differential-deposition test, as a figure: WHERE does newly
accreted stellar mass land? Between adjacent epochs a galaxy grows; this
plots the fraction of that NEW mass falling beyond 50 kpc (and beyond
100 kpc), for the truth and for each model."""
import sys, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'experiments/exp48_objective_profile')
sys.path.insert(0,'experiments/exp47_compact_defect'); sys.path.insert(0,'experiments/exp46_highz_ridge')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import objective as ob
from hongshao import plotting
from pathlib import Path

plotting.set_style()
FIG = Path('experiments/exp48_objective_profile/figures')
ob._w_init(None)
gals, e = ob._W['gals'], ob._W['e']
R, n = e.R, len(gals)
data = np.stack([g['data'] for g in gals])
pop = np.load(ob.POP_NPZ)
logms = np.array([pop['logms'][g['row']] for g in gals])

s = dict(np.load('experiments/exp48_objective_profile/outputs/screen.npz', allow_pickle=True))
f = dict(np.load('experiments/exp48_objective_profile/outputs/factorial.npz', allow_pickle=True))
WIN = 'q=density|res=log|rad=rms|w=uniform|gal=mean'
models = {'baseline objective': s['theta::baseline'],
          'density-log objective': f[f'theta::{WIN}']}
cols = {'baseline objective': '#0072B2', 'density-log objective': '#D55E00'}

built = {}
for lab, th in models.items():
    c = np.full((n, 5, len(R)), np.nan)
    for i, g in enumerate(gals):
        o = ob._W['s2'].model_cogs(th, g, [0,1,2,3,4], '1ch-mof')
        if o is not None: c[i] = o
    built[lab] = c

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
pairs = ['z0.7→0.4', 'z1.0→0.7', 'z1.5→1.0', 'z2.0→1.5']
x = np.arange(4)
for panel, (bidx, ttl) in enumerate([(0, 'beyond 50 kpc'), (1, 'beyond 100 kpc')]):
    ax = axes[panel]
    ed3, rows0 = e.differential(built['baseline objective'], data, logms)
    ax.plot(x, [rows0[('data', 2, k)][bidx] for k in range(4)], 'k-o',
            lw=2.4, ms=7, label='TRUTH', zorder=5)
    for lab, c in built.items():
        _, rws = e.differential(c, data, logms)
        ax.plot(x, [rws[('model', 2, k)][bidx] for k in range(4)],
                '--s', color=cols[lab], lw=2.0, ms=6, label=lab)
    if panel == 0:
        ax.axhspan(0.30, 0.45, color='0.85', zorder=0)
        ax.text(0.05, 0.455, 'accepted band (gate)', fontsize=7.5, color='0.35')
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel(f'fraction of NEW stellar mass landing {ttl}')
    ax.set_title(f'{ttl}  (massive tercile)', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
fig.suptitle('exp48 — the differential-deposition GATE: where does newly '
             'accreted mass land?', fontsize=11)
fig.tight_layout(rect=(0,0,1,0.93))
plotting.save_fig(fig, str(FIG / 'exp48_differential'))
print('wrote', FIG / 'exp48_differential')
