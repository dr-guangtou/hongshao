"""exp25 — Build a massive galaxy from its MAH with a smooth deposition kernel.

A physics-inspired forward toy (Lackner-Ostriker / El-Badry spirit, context doc
sec. 6-7), independent of the data-driven emulator. The idea, deliberately
simple (we IGNORE the in-situ/ex-situ distinction):

  Between two snapshots the halo gains dM_h. A fraction eps(z, M_h) of it shows
  up as new stellar mass dM* = eps * dM_h, deposited as a single 2-D Gaussian
  centred at R=0. A mass-normalized Gaussian has no free amplitude:
      Sigma_i(R) = dM*_i / (2 pi sigma_i^2) * exp(-R^2 / 2 sigma_i^2)
  so each epoch contributes ONE number, its width sigma_i. The galaxy is the sum
  of all epochs' Gaussians; the curve of growth is closed-form:
      M*(<R) = sum_i dM*_i * (1 - exp(-R^2 / 2 sigma_i^2)).
  Widths are not fit per snapshot: they follow the halo size at deposition,
      sigma_i = f * R_200c(M_h,i, z_i)^p,
  so recently accreted stars (low z, big halo) land at large radius
  automatically. Efficiency evolves smoothly, eps ~ (1+z)^beta (more efficient
  early), with the normalization fixed by the galaxy's total stellar mass.

Models (nested):
  0  const eps,            sigma = f R_200c        -> 1 shape param  (f)
  1  eps ~ (1+z)^beta,     sigma = f R_200c        -> 2 params       (f, beta)
  2  eps ~ (1+z)^beta,     sigma = f R_200c^p      -> 3 params       (f, beta, p)

Phase 1: the single most massive galaxy in TNG300. Question: can this analytic
MAH->profile map reproduce a real 1-D surface-density profile?

Run: PYTHONPATH=. uv run python experiments/exp25_deposition_kernel/run.py
"""
# %% setup
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import COG_RAD_KPC, load_mah, load_cosmic_time, peak_history   # noqa: E402
from hongshao.profile_emulator import density_from_cog                                # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                           # noqa: E402
from hongshao.provenance import write_manifest                                         # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089)        # TNG cosmology
R = COG_RAD_KPC


# %% ---- forward model -------------------------------------------------------
def r200c_kpc(m_msun, z):
    """Virial radius R_200c (kpc) for halo mass m_msun at redshift z."""
    rho_c = COSMO.critical_density(z).to("Msun/kpc^3").value
    return (3.0 * m_msun / (4.0 * np.pi * 200.0 * rho_c)) ** (1.0 / 3.0)


def deposit_cog(dMstar, sigma, Rgrid):
    """Sum of centred mass-normalized Gaussians -> cumulative M*(<R). (analytic)"""
    return (dMstar[None, :] * (1.0 - np.exp(-Rgrid[:, None] ** 2 / (2.0 * sigma[None, :] ** 2)))).sum(1)


def deposit_sigma(dMstar, sigma, Rgrid):
    """Sum of centred mass-normalized Gaussians -> surface density Sigma(R)."""
    return (dMstar[None, :] / (2.0 * np.pi * sigma[None, :] ** 2)
            * np.exp(-Rgrid[:, None] ** 2 / (2.0 * sigma[None, :] ** 2))).sum(1)


def deposited_mass(dMh, z, beta, Mstar_tot):
    """Stellar mass per epoch from halo increments, eps~(1+z)^beta, mass-normalized."""
    w = (1.0 + z) ** beta
    return Mstar_tot * (w * dMh) / (w * dMh).sum()


# %% ---- the most massive galaxy and its MAH ---------------------------------
t = Table.read(TABLE); t = t[t["use"]]
cog = np.asarray(t["logmstar_cog"], float)
idx = np.asarray(t["index"])
ok = np.isfinite(cog).all(1)
i = int(np.argmax(np.where(ok, cog[:, -1], -np.inf)))
logMstar_tot = float(cog[i, -1])
Mstar_tot = 10.0 ** logMstar_tot
print(f"exp25: most massive galaxy index={idx[i]}  logM*tot={logMstar_tot:.2f}  "
      f"logMh={float(np.asarray(t['logmh_z0p4'], float)[i]):.2f}")

mah = load_mah(); tsnap = load_cosmic_time()
sn, lmp = peak_history(mah[int(idx[i])])
t_gyr = tsnap[sn.astype(int)]
Mh = 10.0 ** lmp
# t -> z via an interpolation table (z_at_value is slow per call)
z_grid = np.linspace(0.0, 20.0, 4000)
t_of_z = COSMO.age(z_grid).to("Gyr").value
z_snap = np.interp(t_gyr, t_of_z[::-1], z_grid[::-1])
dMh = np.clip(np.diff(Mh), 0.0, None)              # positive halo-mass increments
z_dep, Mh_dep, t_dep = z_snap[1:], Mh[1:], t_gyr[1:]
Rv = r200c_kpc(Mh_dep, z_dep)
print(f"  MAH: {len(t_gyr)} snapshots, z={z_snap[0]:.1f}->{z_snap[-1]:.2f}, "
      f"logMh={lmp[0]:.1f}->{lmp[-1]:.1f}; R_200c {Rv.min():.0f}->{Rv.max():.0f} kpc")

logSig_true, mid = density_from_cog(cog[i][None, :], R)
logSig_true = logSig_true[0]


# %% ---- fit the three nested models -----------------------------------------
def fit_model(kind):
    if kind == 0:                                   # const eps, sigma=f R200c
        eps0 = Mstar_tot / dMh.sum()
        dMstar = eps0 * dMh

        def loss(lf):
            m = deposit_cog(dMstar, 10 ** lf * Rv, R)
            return np.sum((np.log10(np.clip(m, 1, None)) - cog[i]) ** 2)
        r = minimize_scalar(loss, bounds=(-3, 0), method="bounded")
        f = 10 ** r.x
        return dict(name="0: const eps", dMstar=dMstar, sigma=f * Rv, par=dict(f=f))
    if kind == 1:                                   # eps~(1+z)^beta, sigma=f R200c
        def loss(p):
            dMstar = deposited_mass(dMh, z_dep, p[1], Mstar_tot)
            m = deposit_cog(dMstar, 10 ** p[0] * Rv, R)
            return np.sum((np.log10(np.clip(m, 1, None)) - cog[i]) ** 2)
        r = minimize(loss, [np.log10(0.02), 2.0], method="Nelder-Mead")
        f, beta = 10 ** r.x[0], r.x[1]
        dMstar = deposited_mass(dMh, z_dep, beta, Mstar_tot)
        return dict(name="1: eps~(1+z)^b", dMstar=dMstar, sigma=f * Rv, par=dict(f=f, beta=beta))
    # kind == 2: + width slope sigma = f R200c^p
    def loss(p):
        dMstar = deposited_mass(dMh, z_dep, p[1], Mstar_tot)
        m = deposit_cog(dMstar, 10 ** p[0] * Rv ** p[2], R)
        return np.sum((np.log10(np.clip(m, 1, None)) - cog[i]) ** 2)
    r = minimize(loss, [np.log10(0.02), 2.0, 1.0], method="Nelder-Mead")
    f, beta, pw = 10 ** r.x[0], r.x[1], r.x[2]
    dMstar = deposited_mass(dMh, z_dep, beta, Mstar_tot)
    return dict(name="2: +width slope", dMstar=dMstar, sigma=f * Rv ** pw,
                par=dict(f=f, beta=beta, p=pw))


fits = []
print("\n[fit] reproduce the BCG curve of growth (lower RMS = better):")
for kind in (0, 1, 2):
    fit = fit_model(kind)
    m_cog = deposit_cog(fit["dMstar"], fit["sigma"], R)
    fit["rms"] = float(np.sqrt(np.mean((np.log10(np.clip(m_cog, 1, None)) - cog[i]) ** 2)))
    fit["cog_model"] = np.log10(np.clip(m_cog, 1, None))
    fits.append(fit)
    pars = "  ".join(f"{k}={v:.3f}" for k, v in fit["par"].items())
    print(f"  Model {fit['name']:18s} CoG RMS={fit['rms']:.3f} dex   ({pars})")

best = fits[1]                                       # Model 1: the 2-param sweet spot
print(f"\n  -> Model 1 builds the BCG from its MAH to {best['rms']:.3f} dex with 2 params "
      f"(width fraction f={best['par']['f']:.3f}, efficiency slope beta={best['par']['beta']:.2f}).")


# %% ---- FIGURE 1: the galaxy built clump by clump ---------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.8))
order = np.argsort(z_dep)[::-1]                      # early (high z) first
cmap = matplotlib.colormaps["cividis"]
znorm = matplotlib.colors.Normalize(vmin=0, vmax=min(6, z_dep.max()))
for j in order:
    sig = np.array([best["sigma"][j]])
    s = deposit_sigma(np.array([best["dMstar"][j]]), sig, mid)
    axA.plot(mid, np.log10(np.clip(s, 1e-3, None)), color=cmap(znorm(z_dep[j])), lw=0.7, alpha=0.7)
axA.plot(mid, logSig_true, "o", color="k", ms=4, label="TNG truth")
axA.plot(mid, np.log10(np.clip(deposit_sigma(best["dMstar"], best["sigma"], mid), 1e-3, None)),
         "-", color=OKABE_ITO[1], lw=2.4, label="model (sum of clumps)")
axA.set_xscale("log"); axA.set_xlabel("R [kpc]"); axA.set_ylabel(r"$\log \Sigma_*$ [$M_\odot/{\rm kpc}^2$]")
axA.set_ylim(np.floor(logSig_true.min() - 0.5), np.ceil(logSig_true.max() + 0.3))
axA.set_title("A. The BCG as a sum of MAH-deposited Gaussians")
axA.legend(fontsize=8, loc="upper right")
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=znorm, cmap=cmap), ax=axA, pad=0.01)
cb.set_label("deposition redshift", fontsize=8)

axB.plot(R, cog[i], "o", color="k", ms=4, label="TNG truth")
for fit, c in zip(fits, [OKABE_ITO[7], OKABE_ITO[1], OKABE_ITO[2]]):
    axB.plot(R, fit["cog_model"], "-", color=c, lw=1.8,
             label=f"Model {fit['name']} ({fit['rms']:.3f} dex)")
axB.set_xscale("log"); axB.set_xlabel("R [kpc]"); axB.set_ylabel(r"$\log M_*(<R)$")
axB.set_title("B. Curve of growth: model vs truth"); axB.legend(fontsize=8, loc="lower right")
fig.suptitle(f"exp25 — building TNG300's most massive galaxy (logM*={logMstar_tot:.2f}) from its "
             f"halo MAH; Model 1 reaches {best['rms']:.3f} dex", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp25_build_galaxy")


# %% ---- FIGURE 2: the ingredients (MAH, deposited mass, widths) -------------
fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.0, 4.2))
a1.plot(z_snap, lmp, "-o", color=OKABE_ITO[0], ms=3)
a1.set_xlabel("redshift"); a1.set_ylabel(r"$\log M_{\rm halo}(z)$")
a1.invert_xaxis(); a1.set_title("A. The halo MAH (input)")
pos = best["dMstar"] > 1.0                          # epochs that actually deposited mass
a2.plot(z_dep[pos], np.log10(best["dMstar"][pos]), "-o", color=OKABE_ITO[2], ms=3)
a2.set_xlabel("deposition redshift"); a2.set_ylabel(r"$\log \Delta M_*$ per epoch")
a2.invert_xaxis(); a2.set_title(f"B. Stellar mass deposited (beta={best['par']['beta']:.2f})")
a3.plot(z_dep, best["sigma"], "-o", color=OKABE_ITO[5], ms=3, label=r"$\sigma=f\,R_{200c}$")
a3.set_xlabel("deposition redshift"); a3.set_ylabel(r"clump width $\sigma$ [kpc]")
a3.invert_xaxis(); a3.set_yscale("log"); a3.set_title("C. Deposition width grows to low z")
a3.legend(fontsize=8)
fig2.suptitle("exp25 — ingredients: the MAH sets how much stellar mass forms when, "
              "and the halo size sets where it lands", fontsize=10)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp25_ingredients")


# %% ---- save + self-check ---------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
write_manifest(OUTDIR, params={
    "galaxy_index": int(idx[i]), "logMstar_tot": logMstar_tot,
    "n_snapshots": int(len(t_gyr)),
    "model0_rms": fits[0]["rms"], "model1_rms": fits[1]["rms"], "model2_rms": fits[2]["rms"],
    "model1_f": float(best["par"]["f"]), "model1_beta": float(best["par"]["beta"])})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
# the toy must reproduce the profile and improve with the physical efficiency term
assert fits[0]["rms"] < 0.4, fits[0]["rms"]
assert fits[1]["rms"] < 0.5 * fits[0]["rms"], (fits[0]["rms"], fits[1]["rms"])
assert fits[1]["rms"] < 0.08, fits[1]["rms"]
assert best["par"]["beta"] > 0, "efficiency should be higher at early times"
print(f"\n[verdict] the MAH builds the BCG: const-eps {fits[0]['rms']:.3f} dex -> "
      f"eps~(1+z)^{best['par']['beta']:.1f} {fits[1]['rms']:.3f} dex -> "
      f"+width-slope {fits[2]['rms']:.3f} dex. Mathematically feasible; 2 params suffice.")
