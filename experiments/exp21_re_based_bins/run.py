"""exp21 — An Re-based emulator: aperture/outskirt masses in half-mass-radius units.

The graduated emulator predicts stellar masses in *fixed physical* apertures.
A fixed kpc aperture mixes physically different regions across the size-mass
range, so here we bin the profile in **effective-radius (Re) units**: Re = the
half-mass radius within 120 kpc (the observational "total"), read off each
galaxy's 1-D curve of growth.

Two questions, both about the outskirts:
  (1) **Finer Re outskirt bins.** Six bins `<0.5/0.5-1/1-2/2-4/4-6/6-9 Re`
      (replacing the earlier coarse `4Re-120kpc`). 6-9 Re reaches ~100 kpc for
      the median galaxy; for larger galaxies it runs past the 120 kpc
      observational comfort zone, so we require the 9 Re edge to lie within the
      measured CoG (<=148 kpc) and note the (mass-correlated) selection.
  (2) **Does a richer MAH help the Re outskirts?** The outskirts may track
      *recent* accretion, which the smooth DiffMAH rolling-power-law could blur.
      Compare feature sets DiffMAH(4)+c_200c vs MAH-PCA(8)+c_200c vs raw-MAH(18)
      +c_200c, per Re bin (exp13 did this for the *kpc* 50-100 bin and found
      DiffMAH(4) already at the ceiling; the finer Re outer bins are a new test).

Independent of the graduated emulator: a local copy of the exp19 CV machinery,
generalized to n bins. `hongshao/emulator.py` is left untouched.

Run (fast):  EXP21_NMAX=700 PYTHONPATH=. uv run python experiments/exp21_re_based_bins/run.py
Full:        PYTHONPATH=. uv run python experiments/exp21_re_based_bins/run.py
"""
# %% setup
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import (COG_RAD_KPC, load_mah, load_cosmic_time,           # noqa: E402
                               peak_history)
from hongshao.metrics import crps_gaussian, interval_coverage                    # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                      # noqa: E402
from hongshao.provenance import write_manifest                                   # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
NMAX = int(os.environ.get("EXP21_NMAX", 0))
RIDGE = float(os.environ.get("EXP21_RIDGE", 2.0))
TOTAL_KPC = 120.0                                  # observational "total" -> defines Re
RE_EDGES = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 9.0])    # bin edges in Re (6 bins)
RE_NAMES = ["<0.5Re", "0.5-1Re", "1-2Re", "2-4Re", "4-6Re", "6-9Re"]
KPC_NAMES = ["<10", "10-30", "30-50", "50-100"]

t = Table.read(TABLE); t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
cog = np.asarray(t["logmstar_cog"], float)         # log10 M(<R) on COG_RAD_KPC, (N,24)
aper = np.asarray(t["logmstar_aper"], float)
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)
M0 = np.asarray(t["logmh_z0p4"], float)
idx = np.asarray(t["index"])
R = COG_RAD_KPC

# raw + M0-normalized log-MAH on a fixed cosmic-time grid (exp06/09/11/13 construction)
mah = load_mah(); tsnap = load_cosmic_time(); tg = np.linspace(2.2, 9.0, 18)
ms_raw = np.full((len(t), 18), np.nan)
for r, i in enumerate(idx):
    sn, lmp = peak_history(mah[int(i)])
    if lmp is None:
        continue
    tt = tsnap[sn.astype(int)]
    if tt[0] <= tg[0] and tt[-1] >= tg[-1]:
        ms_raw[r] = np.interp(tg, tt, lmp)


# %% ---- Re and the Re-based targets -----------------------------------------
def cum_mass_at(r_kpc):
    """Linear M(<r) for each galaxy by interpolating its log10 CoG (N,) -> (N,)."""
    return np.array([10.0 ** np.interp(rr, R, c) for c, rr in zip(cog, r_kpc)])


def annuli_from_edges(edges_kpc):
    """(N,K) bin masses from K cumulative edges: bin0 cumulative, rest annuli."""
    mcum = np.column_stack([cum_mass_at(edges_kpc[:, k]) for k in range(edges_kpc.shape[1])])
    out = np.empty_like(mcum)
    out[:, 0] = np.log10(np.clip(mcum[:, 0], 1.0, None))
    for k in range(1, mcum.shape[1]):
        out[:, k] = np.log10(np.clip(mcum[:, k] - mcum[:, k - 1], 1.0, None))
    return out, mcum


log_total = np.array([np.interp(TOTAL_KPC, R, c) for c in cog])      # log10 M(<120)
Re = np.array([np.interp(lt - np.log10(2.0), c, R) for c, lt in zip(cog, log_total)])
re_edges = Re[:, None] * RE_EDGES[None, :]                           # (N,6) in kpc
Y_re, mcum_re = annuli_from_edges(re_edges)


def _ann(a_o, a_i):
    return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))


Y_kpc = np.column_stack([aper[:, 0], _ann(aper[:, 1], aper[:, 0]),
                         _ann(aper[:, 2], aper[:, 1]), _ann(aper[:, 4], aper[:, 2])])

# common mask: finite targets/features, monotone masses, and 9Re within the
# measured CoG (so the outer bins are real data, not CoG extrapolation)
mono = np.all(np.diff(mcum_re, axis=1) > 0, axis=1)
within_data = re_edges[:, -1] <= R[-1]
g = (np.isfinite(Y_re).all(1) & np.isfinite(Y_kpc).all(1) & np.isfinite(dmah).all(1)
     & np.isfinite(c200) & np.isfinite(M0) & np.isfinite(ms_raw).all(1)
     & np.isfinite(Re) & (Re > R[0]) & mono & within_data)
Y_re, Y_kpc = Y_re[g], Y_kpc[g]
Re_g, raw = Re[g], ms_raw[g]
N = len(Y_re)

# feature sets (all share +c_200c, the established secondary axis, exp16-19)
shape = raw - M0[g][:, None]                        # M0-normalized MAH shape
_, _, Vt = np.linalg.svd(shape - shape.mean(0), full_matrices=False)
pcs = (shape - shape.mean(0)) @ Vt.T               # MAH-PCA scores
X_dmc = np.column_stack([dmah[g], c200[g]])                          # DiffMAH(4)+c200c
X_pca = np.column_stack([M0[g], pcs[:, :8], c200[g]])               # MAH-PCA(8)+c200c
X_raw = np.column_stack([raw, c200[g]])                             # raw-MAH(18)+c200c
frac_lost = 1.0 - g.sum() / np.isfinite(Re).sum()
print(f"exp21: Re-based emulator on n={N} (dropped {100*frac_lost:.0f}% where 9Re>148 kpc); "
      f"Re median={np.median(Re_g):.1f} kpc, 9Re median={9*np.median(Re_g):.0f} kpc")


# %% ---- emulator CV machinery (local copy of exp19, generalized to nb bins) --
def fit_logvar(r, Z, ridge=RIDGE):
    A = np.column_stack([np.ones(len(r)), Z]); r2 = r ** 2

    def nll(gm):
        s = A @ gm
        return 0.5 * np.sum(s + r2 * np.exp(-s)) + 0.5 * ridge * np.sum(gm[1:] ** 2)

    def grad(gm):
        s = A @ gm; w = 0.5 * (1 - r2 * np.exp(-s)); o = A.T @ w; o[1:] += ridge * gm[1:]
        return o
    return minimize(nll, np.r_[np.log(max(r2.mean(), 1e-6)), np.zeros(Z.shape[1])],
                    jac=grad, method="L-BFGS-B").x


def cv_emulator(Y, Xmean, Xvar, k=5, seed=0):
    n, nb = Y.shape
    order = np.random.default_rng(seed).permutation(n)
    MU = np.full((n, nb), np.nan); SIG = np.full((n, nb), np.nan)
    FULL = np.full((n, nb, nb), np.nan)
    Am = np.column_stack([np.ones(n), Xmean])
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        r_tr = np.empty((len(tr), nb))
        for j in range(nb):
            beta, *_ = np.linalg.lstsq(Am[tr], Y[tr, j], rcond=None)
            MU[fold, j] = Am[fold] @ beta
            r_tr[:, j] = Y[tr, j] - Am[tr] @ beta
        mx, sx = Xvar[tr].mean(0), Xvar[tr].std(0)
        Ztr = (Xvar[tr] - mx) / sx; Zte = (Xvar[fold] - mx) / sx
        sig_tr = np.empty((len(tr), nb)); sig_te = np.empty((len(fold), nb))
        for j in range(nb):
            gm = fit_logvar(r_tr[:, j], Ztr)
            sig_tr[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(tr)), Ztr]) @ gm))
            sig_te[:, j] = np.exp(0.5 * (np.column_stack([np.ones(len(fold)), Zte]) @ gm))
        SIG[fold] = sig_te
        Rc = np.corrcoef((r_tr / sig_tr).T)
        for n_i, i in enumerate(fold):
            D = np.diag(sig_te[n_i]); FULL[i] = D @ Rc @ D
    return MU, SIG, FULL


def joint_nll(Y, MU, FULL):
    resid = Y - MU; out = np.empty(len(Y))
    for i in range(len(Y)):
        _, ld = np.linalg.slogdet(FULL[i])
        out[i] = 0.5 * (Y.shape[1] * np.log(2 * np.pi) + ld
                        + resid[i] @ np.linalg.solve(FULL[i], resid[i]))
    return float(out.mean())


def cond_cov_gap(Y, MU, SIG):
    gaps = []
    for j in range(Y.shape[1]):
        s = SIG[:, j]; edges = np.quantile(s, [1/3, 2/3]); bins = np.digitize(s, edges)
        cov = [np.mean(np.abs(Y[bins == b, j] - MU[bins == b, j]) <= SIG[bins == b, j])
               for b in (0, 2)]
        gaps.append(abs(cov[1] - cov[0]))
    return float(np.mean(gaps))


def feature_cv(Y, X, k=5, seed=0):
    """Per-bin linear-mean homoscedastic CV (matches exp13); CRPS, R^2 per bin."""
    n, nb = Y.shape
    A = np.column_stack([np.ones(n), X])
    order = np.random.default_rng(seed).permutation(n)
    pred = np.full((n, nb), np.nan); sig = np.full((n, nb), np.nan)
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        for j in range(nb):
            beta, *_ = np.linalg.lstsq(A[tr], Y[tr, j], rcond=None)
            pred[fold, j] = A[fold] @ beta
            sig[fold, j] = (Y[tr, j] - A[tr] @ beta).std()
    crps = crps_gaussian(Y, pred, sig).mean(0)
    r2 = 1.0 - ((Y - pred) ** 2).mean(0) / Y.var(0)
    return crps, r2


# %% ---- (1) the Re emulator (DiffMAH+c200c, 6 bins) + kpc reference ----------
MU, SIG, FULL = cv_emulator(Y_re, X_dmc, X_dmc)
crps_re = crps_gaussian(Y_re, MU, SIG).mean(0)
_, cov_re = interval_coverage(Y_re, MU, SIG)
nll_re = joint_nll(Y_re, MU, FULL); gap_re = cond_cov_gap(Y_re, MU, SIG)
MUk, SIGk, FULLk = cv_emulator(Y_kpc, X_dmc, X_dmc)
crps_kpc = crps_gaussian(Y_kpc, MUk, SIGk).mean(0)
print(f"\n[1] Re emulator (DiffMAH+c200c, 6 bins): CRPS {crps_re.mean():.4f}  "
      f"NLL {nll_re:+.3f}  cov {'/'.join(f'{c:.2f}' for c in cov_re)}  condgap {gap_re:.3f}")
print(f"    kpc reference (same sample): CRPS {crps_kpc.mean():.4f}")


# %% ---- (2) does a richer MAH help the Re outskirts? ------------------------
SETS = {"DiffMAH(4)+c200c": X_dmc, "MAH-PCA(8)+c200c": X_pca, "raw-MAH(18)+c200c": X_raw}
fc = {name: feature_cv(Y_re, X) for name, X in SETS.items()}
print("\n[2] richer-MAH test — per-bin CV CRPS [dex] (lower=better):")
print(f"  {'feature set':20s} " + " ".join(f"{nm:>9s}" for nm in RE_NAMES) + f"  {'mean':>7s}")
for name, (crps, r2) in fc.items():
    print(f"  {name:20s} " + " ".join(f"{c:9.4f}" for c in crps) + f"  {crps.mean():7.4f}")
base = fc["DiffMAH(4)+c200c"][0]
print("  --- richer-MAH gain over DiffMAH (CRPS reduction; + = richer helps) ---")
for name in ("MAH-PCA(8)+c200c", "raw-MAH(18)+c200c"):
    d = base - fc[name][0]
    print(f"  {name:20s} " + " ".join(f"{x:+9.4f}" for x in d)
          + f"  {d.mean():+7.4f}  ({100*d.mean()/base.mean():+.1f}%)")
print("  (exp13: for the kpc 50-100 bin, richer MAH added only +0.9% — DiffMAH at ceiling)")


# %% ---- FIGURE ---------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.3))
medRe = np.median(Re_g)
axA.plot(R, np.median(cog[g], axis=0), "-", color="0.3", lw=2, label="median CoG")
for frac in RE_EDGES:
    axA.axvline(frac * medRe, color=OKABE_ITO[2], ls=":", lw=1)
axA.axvline(TOTAL_KPC, color="k", ls="--", lw=1, label="120 kpc (total)")
axA.axvline(medRe, color=OKABE_ITO[1], ls="-", lw=1.5, label=f"Re={medRe:.0f} kpc")
axA.set_xscale("log"); axA.set_xlabel("R [kpc]"); axA.set_ylabel(r"$\log M_*(<R)$")
axA.set_title("A. Six Re-based bins on the CoG"); axA.legend(fontsize=8)

x = np.arange(6); w = 0.26
for i, (name, (crps, r2)) in enumerate(fc.items()):
    axB.bar(x + (i - 1) * w, crps, w, color=OKABE_ITO[[7, 2, 5][i]], label=name)
axB.set_xticks(x); axB.set_xticklabels(RE_NAMES, rotation=30, fontsize=7, ha="right")
axB.set_ylabel("CV CRPS [dex]"); axB.set_xlabel("Re-based bin")
axB.set_title("B. MAH richness vs Re bin"); axB.legend(fontsize=7)

axC.axhline(0, color="k", lw=0.8)
for name, c in [("MAH-PCA(8)+c200c", OKABE_ITO[2]), ("raw-MAH(18)+c200c", OKABE_ITO[5])]:
    axC.plot(x, 100 * (base - fc[name][0]) / base, "-o", color=c, ms=5, label=name)
axC.set_xticks(x); axC.set_xticklabels(RE_NAMES, rotation=30, fontsize=7, ha="right")
axC.set_ylabel("CRPS gain over DiffMAH [%]"); axC.set_xlabel("Re-based bin")
axC.set_title("C. Does richer MAH help the outskirts?"); axC.legend(fontsize=7)
fig.suptitle(f"exp21 — Re-based emulator (n={N}); 6 bins, mean CRPS {crps_re.mean():.4f}, "
             f"NLL {nll_re:+.3f}; richer MAH gain (mean) "
             f"{100*(base.mean()-fc['raw-MAH(18)+c200c'][0].mean())/base.mean():+.1f}%", fontsize=10)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp21_re_emulator")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"bin": RE_NAMES, "crps_dmc": crps_re.tolist(),
            "crps_diffmah_c200c": fc["DiffMAH(4)+c200c"][0].tolist(),
            "crps_mahpca_c200c": fc["MAH-PCA(8)+c200c"][0].tolist(),
            "crps_rawmah_c200c": fc["raw-MAH(18)+c200c"][0].tolist(),
            "r2_dmc": fc["DiffMAH(4)+c200c"][1].tolist()})
st.write(OUTDIR / "re_emulator_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "frac_dropped_9Re_gt_148kpc": float(frac_lost),
    "re_edges_in_Re": RE_EDGES.tolist(), "total_kpc": TOTAL_KPC,
    "re_median_kpc": float(np.median(Re_g)),
    "crps_re_dmc": float(crps_re.mean()), "nll_re_dmc": float(nll_re),
    "condgap_re_dmc": float(gap_re), "cov_re": cov_re.tolist(),
    "crps_kpc_same_sample": float(crps_kpc.mean()),
    "mahpca_gain_pct": float(100 * (base.mean() - fc["MAH-PCA(8)+c200c"][0].mean()) / base.mean()),
    "rawmah_gain_pct": float(100 * (base.mean() - fc["raw-MAH(18)+c200c"][0].mean()) / base.mean()),
    "rawmah_gain_outer_two": (base[-2:] - fc["raw-MAH(18)+c200c"][0][-2:]).tolist()})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
print(f"\n[verdict] Re emulator (6 bins): CRPS {crps_re.mean():.4f}, NLL {nll_re:+.3f}; "
      f"richer MAH over DiffMAH: PCA "
      f"{100*(base.mean()-fc['MAH-PCA(8)+c200c'][0].mean())/base.mean():+.1f}%, raw "
      f"{100*(base.mean()-fc['raw-MAH(18)+c200c'][0].mean())/base.mean():+.1f}% "
      f"(outer 6-9Re bin: {base[-1]-fc['raw-MAH(18)+c200c'][0][-1]:+.4f} dex)")
