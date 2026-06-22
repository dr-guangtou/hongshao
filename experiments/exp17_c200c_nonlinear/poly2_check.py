"""exp17 addendum — Is the poly-2 gain justified, and what is its form?

Two questions about the exp17 poly-2 result (0.0808 vs linear DiffMAH+c200c
0.0839, +3.7% CRPS): (1) what degree-2 terms does it actually use? (2) is the
gain real and worth the extra degrees of freedom (5 -> ~21 params per aperture),
or could a few interpretable terms capture it?

Approach:
  - FORM: fit poly-2 on all data with standardized design columns (so each
    coefficient is the effect in dex per 1 sigma of that term), rank the degree-2
    terms by |coef| per aperture, and write out the dominant ones.
  - JUSTIFY:
    (a) multi-seed 5-fold CV -> distribution of the linear->poly-2 improvement
        (is it consistent, or one lucky split? CV already penalizes useless DOF).
    (b) forward selection of *shared* degree-2 terms -> CRPS vs #terms: how few
        terms recover most of the +3.7% (parsimony / DOF justification).
    (c) BIC (per aperture) for linear vs the parsimonious form vs full poly-2.

Run: PYTHONPATH=. uv run python experiments/exp17_c200c_nonlinear/poly2_check.py
"""
# %% setup
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import load_mah, load_cosmic_time, peak_history          # noqa: E402
from hongshao.metrics import crps_gaussian, aic_bic                             # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                     # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
TNAMES = ["<10", "10-30", "30-50", "50-100"]
FEAT = ["logmp", "logtc", "early", "late", "c200c"]

t = Table.read(TABLE); t = t[t["use"]]
aper = np.asarray(t["logmstar_aper"], float)
M0 = np.asarray(t["logmh_z0p4"], float)
idx = np.asarray(t["index"])
X = np.column_stack([np.asarray(t[c], float) for c in
                     ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late", "c_200c")])


def _ann(a_o, a_i):
    return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))


Y = np.column_stack([aper[:, 0], _ann(aper[:, 1], aper[:, 0]),
                     _ann(aper[:, 2], aper[:, 1]), _ann(aper[:, 4], aper[:, 2])])
g = np.isfinite(Y).all(1) & np.isfinite(M0) & np.isfinite(X).all(1)
Y = Y[g]; X = X[g]; N = len(Y)
Xs = (X - X.mean(0)) / X.std(0)          # standardized features
print(f"poly2 check on n={N}")

# --- build the degree-2 term library (label -> column) -----------------------
terms = {}                                # name -> (N,) standardized column
for i in range(5):
    terms[FEAT[i]] = Xs[:, i]
LINEAR = list(FEAT)
NONLIN = []
for i in range(5):
    terms[f"{FEAT[i]}^2"] = Xs[:, i] ** 2
    NONLIN.append(f"{FEAT[i]}^2")
for i, j in combinations(range(5), 2):
    terms[f"{FEAT[i]}*{FEAT[j]}"] = Xs[:, i] * Xs[:, j]
    NONLIN.append(f"{FEAT[i]}*{FEAT[j]}")
# standardize every constructed column too, so coefficients are comparable
col = {k: (v - v.mean()) / v.std() for k, v in terms.items()}


def design(names):
    return np.column_stack([np.ones(N)] + [col[n] for n in names])


def cv_crps(names, seeds=(0,), k=5):
    A = design(names); out = []
    for s in seeds:
        order = np.random.default_rng(s).permutation(N)
        pred = np.full((N, 4), np.nan); sig = np.full((N, 4), np.nan)
        for fold in np.array_split(order, k):
            tr = np.setdiff1d(np.arange(N), fold)
            for jj in range(4):
                beta, *_ = np.linalg.lstsq(A[tr], Y[tr, jj], rcond=None)
                pred[fold, jj] = A[fold] @ beta
                sig[fold, jj] = (Y[tr, jj] - A[tr] @ beta).std()
        out.append(crps_gaussian(Y, pred, sig).mean())
    return np.array(out)


# %% ---- (1) the fitted poly-2 form ------------------------------------------
print("\n[1] poly-2 fitted form (standardized; coef = dex per 1 sigma of the term)")
A_full = design(LINEAR + NONLIN)
for j, tn in enumerate(TNAMES):
    beta, *_ = np.linalg.lstsq(A_full, Y[:, j], rcond=None)
    names = ["1"] + LINEAR + NONLIN
    nl = [(names[k], beta[k]) for k in range(1 + len(LINEAR), len(names))]
    nl.sort(key=lambda kv: -abs(kv[1]))
    top = "  ".join(f"{n}={c:+.3f}" for n, c in nl[:4])
    print(f"  {tn:>7s}: dominant degree-2 terms: {top}")


# %% ---- (2a) is the gain consistent across CV splits? -----------------------
seeds = tuple(range(10))
lin = cv_crps(LINEAR, seeds)
full = cv_crps(LINEAR + NONLIN, seeds)
impr = 100 * (lin - full) / lin
print(f"\n[2a] 10-seed CV: linear={lin.mean():.4f}±{lin.std():.4f}  "
      f"poly2={full.mean():.4f}±{full.std():.4f}  "
      f"improvement {impr.mean():+.2f}%±{impr.std():.2f}% "
      f"(positive in {int((impr > 0).sum())}/10 splits)")


# %% ---- (2b) forward selection of shared degree-2 terms ----------------------
chosen, path = [], []
remaining = list(NONLIN)
base = cv_crps(LINEAR, seeds).mean()
path.append(("linear", base))
for step in range(8):
    best, best_c = None, np.inf
    for cand in remaining:
        c = cv_crps(LINEAR + chosen + [cand], seeds).mean()
        if c < best_c:
            best, best_c = cand, c
    chosen.append(best); remaining.remove(best); path.append((best, best_c))
    print(f"  +{best:14s} -> CV CRPS {best_c:.4f}  ({100*(base-best_c)/base:+.2f}% vs linear)")

full_c = full.mean()
# parsimonious model = fewest terms within 0.0003 dex of full poly-2
n_par = next((i for i, (_, c) in enumerate(path) if c <= full_c + 0.0003), len(path) - 1)
par_terms = chosen[:max(n_par, 1)] if n_par >= 1 else []
print(f"\n[2b] full poly-2 CRPS={full_c:.4f}; parsimonious set reaching it: "
      f"{LINEAR} + {par_terms} ({len(par_terms)} extra terms)")


# %% ---- (2c) BIC: linear vs parsimonious vs full (per aperture, summed) -----
def total_bic(names):
    A = design(names); tot = 0.0
    for j in range(4):
        beta, *_ = np.linalg.lstsq(A, Y[:, j], rcond=None)
        rss = float(np.sum((Y[:, j] - A @ beta) ** 2))
        tot += aic_bic(rss, N, A.shape[1] + 1)[1]      # k = n_params + noise var
    return tot


bic_lin = total_bic(LINEAR)
bic_par = total_bic(LINEAR + par_terms)
bic_full = total_bic(LINEAR + NONLIN)
print(f"\n[2c] summed BIC (lower=better):  linear={bic_lin:.0f}  "
      f"parsimonious={bic_par:.0f}  full poly-2={bic_full:.0f}")
print(f"     BIC prefers: {'parsimonious' if bic_par < min(bic_lin, bic_full) else ('full' if bic_full < bic_lin else 'linear')}")


# %% ---- FIGURE: forward-selection path --------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
labels = [p[0] for p in path]
vals = [p[1] for p in path]
ax.plot(range(len(vals)), vals, "-o", color=OKABE_ITO[2], ms=6)
ax.axhline(base, ls="--", color=OKABE_ITO[7], lw=1, label=f"linear ({base:.4f})")
ax.axhline(full_c, ls=":", color=OKABE_ITO[5], lw=1.2, label=f"full poly-2, 15 terms ({full_c:.4f})")
ax.axvline(len(par_terms), color="0.7", lw=1)
for i, lab in enumerate(labels):
    ax.annotate(lab, (i, vals[i]), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7, rotation=30)
ax.set_xlabel("number of degree-2 terms added (forward selection)")
ax.set_ylabel("CV CRPS [dex]")
ax.set_title(f"exp17 — most of the poly-2 gain is in ~{len(par_terms)} interpretable terms")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp17_poly2_selection")
print(f"\nwrote figure -> {FIGDIR}/exp17_poly2_selection")
