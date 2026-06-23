"""exp21 — An Re-based emulator: aperture/outskirt masses in half-mass-radius units.

The graduated emulator (hongshao/emulator.py) predicts stellar masses in *fixed
physical* apertures (<10, 10-30, 30-50, 50-100 kpc). But a fixed kpc aperture
means different things for a compact vs an extended galaxy — "<10 kpc" is the
whole inner galaxy for one and a sub-core slice for another. A more physical,
observationally-motivated coordinate is the **effective radius Re**: define Re
from the 1-D curve of growth as the half-mass radius *within 120 kpc* (the
practical "total" — light beyond 120 kpc is hard to measure), then bin the
profile in Re units.

This experiment: (1) define Re per galaxy from the CoG, (2) build five Re-based
bins tiling 0->120 kpc, (3) train the same probabilistic emulator (linear mean
on DiffMAH+c_200c + heteroscedastic full covariance) on these masses, and (4)
ask whether Re-normalization changes how predictable the profile is from halo
assembly -- in particular whether the secondary (c_200c) signal sharpens.

Independent of the graduated emulator: this reuses the *approach* (a local copy
of the exp19 CV machinery, generalized to n bins) but does NOT touch the library.

Run (fast):  EXP21_NMAX=600 PYTHONPATH=. uv run python experiments/exp21_re_based_bins/run.py
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
from hongshao.tng_data import COG_RAD_KPC                                        # noqa: E402
from hongshao.metrics import crps_gaussian, interval_coverage                   # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                     # noqa: E402
from hongshao.provenance import write_manifest                                  # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
NMAX = int(os.environ.get("EXP21_NMAX", 0))
RIDGE = float(os.environ.get("EXP21_RIDGE", 2.0))
TOTAL_KPC = 120.0                                  # the observational "total" aperture
RE_EDGES = np.array([0.5, 1.0, 2.0, 4.0])          # inner bin edges in Re; outer = TOTAL_KPC
RE_NAMES = ["<0.5Re", "0.5-1Re", "1-2Re", "2-4Re", "4Re-120kpc"]
KPC_NAMES = ["<10", "10-30", "30-50", "50-100"]
VAR_FEAT = ["logmp", "logtc", "early", "late", "c200c"]

t = Table.read(TABLE); t = t[t["use"]]
if NMAX:
    t = t[:NMAX]
cog = np.asarray(t["logmstar_cog"], float)         # log10 M(<R) on COG_RAD_KPC, (N,24)
aper = np.asarray(t["logmstar_aper"], float)       # the 7 fixed kpc apertures
dmah = np.column_stack([np.asarray(t[c], float) for c in
                        ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late")])
c200 = np.asarray(t["c_200c"], float)
R = COG_RAD_KPC


# %% ---- Re and the Re-based targets -----------------------------------------
def cum_mass_at(r_kpc):
    """Linear M(<r) for each galaxy by interpolating its log10 CoG (N,) -> (N,)."""
    return np.array([10.0 ** np.interp(rr, R, c) for c, rr in zip(cog, r_kpc)])


def annuli_from_edges(edges_kpc):
    """Build (N, K) bin masses from K cumulative edges: bin0 cumulative, rest annuli."""
    mcum = np.column_stack([cum_mass_at(edges_kpc[:, k]) for k in range(edges_kpc.shape[1])])
    out = np.empty_like(mcum)
    out[:, 0] = np.log10(np.clip(mcum[:, 0], 1.0, None))
    for k in range(1, mcum.shape[1]):
        out[:, k] = np.log10(np.clip(mcum[:, k] - mcum[:, k - 1], 1.0, None))
    return out, mcum


log_total = np.array([np.interp(TOTAL_KPC, R, c) for c in cog])      # log10 M(<120)
Re = np.array([np.interp(lt - np.log10(2.0), c, R) for c, lt in zip(cog, log_total)])
edge_kpc = np.minimum(Re[:, None] * RE_EDGES[None, :], TOTAL_KPC - 1.0)
re_edges = np.column_stack([edge_kpc, np.full(len(Re), TOTAL_KPC)])  # (N,5)
Y_re, mcum_re = annuli_from_edges(re_edges)

# kpc reference targets (exactly the graduated-emulator definition, exp19)
def _ann(a_o, a_i):
    return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))


Y_kpc = np.column_stack([aper[:, 0], _ann(aper[:, 1], aper[:, 0]),
                         _ann(aper[:, 2], aper[:, 1]), _ann(aper[:, 4], aper[:, 2])])

# common finite/monotone mask
mono = np.all(np.diff(re_edges, axis=1) > 0, axis=1) & np.all(np.diff(mcum_re, axis=1) > 0, axis=1)
g = (np.isfinite(Y_re).all(1) & np.isfinite(Y_kpc).all(1) & np.isfinite(dmah).all(1)
     & np.isfinite(c200) & np.isfinite(Re) & (Re > R[0]) & (Re < TOTAL_KPC) & mono)
Y_re, Y_kpc = Y_re[g], Y_kpc[g]
X_dm = dmah[g]
X_dmc = np.column_stack([dmah[g], c200[g]])
Re = Re[g]
N = len(Y_re)
print(f"exp21: Re-based emulator on n={N};  Re[kpc] median={np.median(Re):.1f} "
      f"(5/95 pct {np.percentile(Re,5):.1f}/{np.percentile(Re,95):.1f})")


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


def r2_per_bin(Y, MU):
    return 1.0 - ((Y - MU) ** 2).mean(0) / Y.var(0)


def evaluate(name, Y, Xmean, Xvar, names):
    MU, SIG, FULL = cv_emulator(Y, Xmean, Xvar)
    crps = crps_gaussian(Y, MU, SIG).mean(0)
    _, cov = interval_coverage(Y, MU, SIG)
    nll = joint_nll(Y, MU, FULL); gap = cond_cov_gap(Y, MU, SIG)
    r2 = r2_per_bin(Y, MU)
    print(f"  {name:28s} CRPS={crps.mean():.4f}  NLL={nll:+.3f}  "
          f"cov={'/'.join(f'{c:.2f}' for c in cov)}  condgap={gap:.3f}")
    return dict(crps=crps, cov=cov, nll=nll, gap=gap, r2=r2, MU=MU, SIG=SIG, names=names)


# %% ---- run: Re emulator (DiffMAH, DiffMAH+c200c) + kpc reference ------------
print("\n[Re-based emulator] (5 bins, lower CRPS/NLL/gap = better)")
RE_dm = evaluate("Re | DiffMAH", Y_re, X_dm, X_dm, RE_NAMES)
RE_dmc = evaluate("Re | DiffMAH+c200c", Y_re, X_dmc, X_dmc, RE_NAMES)
print("\n[kpc reference] (4 bins; the graduated emulator's definition)")
KPC_dmc = evaluate("kpc | DiffMAH+c200c", Y_kpc, X_dmc, X_dmc, KPC_NAMES)

print("\n[per-bin skill] CRPS [dex] and R^2 of the mean from DiffMAH+c_200c:")
print("  Re-based:")
for j, nm in enumerate(RE_NAMES):
    print(f"    {nm:12s} CRPS={RE_dmc['crps'][j]:.4f}  R2={RE_dmc['r2'][j]:+.3f}  "
          f"c200c gain (DiffMAH->+c200c CRPS) {RE_dm['crps'][j]-RE_dmc['crps'][j]:+.4f}")
print("  kpc reference:")
for j, nm in enumerate(KPC_NAMES):
    print(f"    {nm:12s} CRPS={KPC_dmc['crps'][j]:.4f}  R2={KPC_dmc['r2'][j]:+.3f}")

c200_gain_re = 100 * (RE_dm["crps"].mean() - RE_dmc["crps"].mean()) / RE_dm["crps"].mean()
print(f"\n  c_200c overall gain in Re bins: {c200_gain_re:+.1f}% CRPS "
      f"(kpc reference was +4.7%, exp19)")


# %% ---- FIGURE ---------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.0, 4.3))
# A: the Re binning scheme on the median CoG
axA.plot(R, np.median(cog[g], axis=0), "-", color="0.3", lw=2, label="median CoG")
medRe = np.median(Re)
for frac in RE_EDGES:
    axA.axvline(frac * medRe, color=OKABE_ITO[2], ls=":", lw=1)
axA.axvline(TOTAL_KPC, color="k", ls="--", lw=1, label="120 kpc (total)")
axA.axvline(medRe, color=OKABE_ITO[1], ls="-", lw=1.5, label=f"Re={medRe:.0f} kpc")
axA.set_xscale("log"); axA.set_xlabel("R [kpc]"); axA.set_ylabel(r"$\log M_*(<R)$")
axA.set_title("A. Re-based bins on the CoG"); axA.legend(fontsize=8)

# B: per-bin CRPS, Re vs kpc
xr = np.arange(5); xk = np.arange(4)
axB.bar(xr - 0.0, RE_dmc["crps"], 0.6, color=OKABE_ITO[2], label="Re bins")
axB.set_xticks(xr); axB.set_xticklabels(RE_NAMES, rotation=30, fontsize=7, ha="right")
axB.set_ylabel("CV CRPS [dex]"); axB.set_xlabel("Re-based bin")
axB.set_title(f"B. Re emulator skill (mean CRPS {RE_dmc['crps'].mean():.4f})")

# C: c_200c gain per Re-bin (does assembly signal sharpen outward?)
axC.axhline(0, color="k", lw=0.8)
gain = RE_dm["crps"] - RE_dmc["crps"]
axC.bar(xr, gain, 0.6, color=OKABE_ITO[5])
axC.set_xticks(xr); axC.set_xticklabels(RE_NAMES, rotation=30, fontsize=7, ha="right")
axC.set_ylabel(r"CRPS gain from $c_{200c}$ [dex]"); axC.set_xlabel("Re-based bin")
axC.set_title("C. Where does concentration help?")
fig.suptitle(f"exp21 — Re-based emulator (n={N}); mean CRPS {RE_dmc['crps'].mean():.4f}, "
             f"NLL {RE_dmc['nll']:+.3f}, c200c gain {c200_gain_re:+.1f}%", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp21_re_emulator")


# %% ---- save outputs --------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
st = Table({"bin": RE_NAMES, "crps_dmc": RE_dmc["crps"].tolist(),
            "r2_dmc": RE_dmc["r2"].tolist(),
            "crps_c200c_gain": (RE_dm["crps"] - RE_dmc["crps"]).tolist()})
st.write(OUTDIR / "re_emulator_scores.csv", overwrite=True)
write_manifest(OUTDIR, params={
    "n": int(N), "re_edges_in_Re": RE_EDGES.tolist(), "total_kpc": TOTAL_KPC,
    "re_median_kpc": float(np.median(Re)),
    "crps_re_dmc": float(RE_dmc["crps"].mean()), "nll_re_dmc": float(RE_dmc["nll"]),
    "condgap_re_dmc": float(RE_dmc["gap"]),
    "crps_kpc_dmc": float(KPC_dmc["crps"].mean()), "nll_kpc_dmc": float(KPC_dmc["nll"]),
    "c200c_gain_pct_re": float(c200_gain_re),
    "r2_re_per_bin": RE_dmc["r2"].tolist(), "r2_kpc_per_bin": KPC_dmc["r2"].tolist()})
print(f"\nwrote figure -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
print(f"\n[verdict] Re emulator: CRPS {RE_dmc['crps'].mean():.4f}, NLL {RE_dmc['nll']:+.3f}, "
      f"condgap {RE_dmc['gap']:.3f}; c_200c gain {c200_gain_re:+.1f}% "
      f"(vs +4.7% in kpc bins, exp19)")
