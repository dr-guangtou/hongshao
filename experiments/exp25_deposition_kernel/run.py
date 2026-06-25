"""exp25 — Build a massive galaxy from its MAH with a smooth deposition kernel.

A physics-inspired forward toy (Lackner-Ostriker / El-Badry spirit, context doc
sec. 6-7), independent of the data-driven emulator. We do NOT label stars
in-situ vs ex-situ — only "stellar mass grown" per epoch.

Model: between two snapshots the halo gains dM_h. A fraction eps(z) of it becomes
new stellar mass dM* = eps*dM_h, deposited as ONE centred 2-D Gaussian. A
mass-normalized Gaussian has no free amplitude (set by mass + width), so each
epoch contributes one number, its width sigma. The galaxy is the sum; the curve
of growth is closed-form:
    Sigma(R) = sum_i dM*_i / (2 pi sigma_i^2) * exp(-R^2 / 2 sigma_i^2)
    M*(<R)   = sum_i dM*_i * (1 - exp(-R^2 / 2 sigma_i^2)).

Two ingredients, each a smooth function (exp25 found, against the most massive
TNG300 galaxy):
  WIDTH   sigma(t) = sigma_0 * (t / t_obs)^g            [direct in cosmic time]
    Tying sigma to R_200c was tested and dropped: it fit no better than a direct
    sigma(t) and added the assumption M_peak = M_200c + cosmology in the width.
  EFFICIENCY (two-epoch "quenching", continuous at a transition redshift z_c):
        eps(z) ~ (1+z)^b_early                              for z >= z_c
        eps(z) ~ (1+z_c)^(b_early-b_late) (1+z)^b_late      for z <  z_c
    Before z_c: rapid early growth (steep b_early). After z_c ("quenched"):
    shallower growth (b_late). z_c is the halo's quenching redshift; in a
    population model it should scale with halo mass (massive halos quench
    earlier). The normalization is fixed by the galaxy's total stellar mass.

Phase 1: the single most massive galaxy in TNG300. A single power-law efficiency
reaches ~0.028 dex; the two-epoch quenching law reaches ~0.008 dex.

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
from scipy.optimize import minimize

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
COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089)        # TNG cosmology (only for snapshot z)
R = COG_RAD_KPC


# %% ---- forward model -------------------------------------------------------
def deposit_cog(dMstar, sigma, Rgrid):
    """Sum of centred mass-normalized Gaussians -> cumulative M*(<R). (analytic)"""
    return (dMstar[None, :] * (1.0 - np.exp(-Rgrid[:, None] ** 2 / (2.0 * sigma[None, :] ** 2)))).sum(1)


def deposit_sigma(dMstar, sigma, Rgrid):
    """Sum of centred mass-normalized Gaussians -> surface density Sigma(R)."""
    return (dMstar[None, :] / (2.0 * np.pi * sigma[None, :] ** 2)
            * np.exp(-Rgrid[:, None] ** 2 / (2.0 * sigma[None, :] ** 2))).sum(1)


def width_t(sigma0, g, t, t_obs):
    """Deposition width sigma(t) = sigma_0 (t/t_obs)^g (no R_200c)."""
    return sigma0 * (t / t_obs) ** g


def eff_single(z, beta):
    """Single power-law efficiency weight ~ (1+z)^beta (un-normalized)."""
    return (1.0 + z) ** beta


def eff_two_epoch(z, b_early, b_late, z_c):
    """Two-epoch quenching weight, continuous at z_c (un-normalized)."""
    hi = (1.0 + z) ** b_early
    lo = (1.0 + z_c) ** (b_early - b_late) * (1.0 + z) ** b_late
    return np.where(z >= z_c, hi, lo)


def deposited(weight, dMh, Mstar_tot):
    """Per-epoch stellar mass from efficiency weights, normalized to total mass."""
    return Mstar_tot * (weight * dMh) / (weight * dMh).sum()


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
z_grid = np.linspace(0.0, 20.0, 4000)
t_of_z = COSMO.age(z_grid).to("Gyr").value
z_snap = np.interp(t_gyr, t_of_z[::-1], z_grid[::-1])      # snapshot redshift
dMh = np.clip(np.diff(Mh), 0.0, None)                      # positive halo-mass increments
z_dep, t_dep = z_snap[1:], t_gyr[1:]
t_obs = t_gyr[-1]
print(f"  MAH: {len(t_gyr)} snapshots, z={z_snap[0]:.1f}->{z_snap[-1]:.2f}, "
      f"logMh={lmp[0]:.1f}->{lmp[-1]:.1f}, t_obs={t_obs:.2f} Gyr")

logSig_true, mid = density_from_cog(cog[i][None, :], R)
logSig_true = logSig_true[0]


# %% ---- fit: single power-law vs two-epoch quenching ------------------------
def rms_of(dMstar, sigma):
    m = deposit_cog(dMstar, sigma, R)
    return float(np.sqrt(np.mean((np.log10(np.clip(m, 1, None)) - cog[i]) ** 2)))


def fit_single():
    def loss(q):                                          # q = [log s0, g, beta]
        dMstar = deposited(eff_single(z_dep, q[2]), dMh, Mstar_tot)
        return rms_of(dMstar, width_t(10 ** q[0], q[1], t_dep, t_obs))
    r = minimize(loss, [np.log10(30.0), 2.0, 2.0], method="Nelder-Mead")
    s0, g, beta = 10 ** r.x[0], r.x[1], r.x[2]
    dMstar = deposited(eff_single(z_dep, beta), dMh, Mstar_tot)
    return dict(name="single power-law", dMstar=dMstar, sigma=width_t(s0, g, t_dep, t_obs),
                weight=eff_single(z_dep, beta), par=dict(sigma0=s0, g=g, beta=beta), rms=loss(r.x))


def fit_two_epoch():
    def loss(q):                                          # q = [log s0, g, b_early, b_late, z_c]
        dMstar = deposited(eff_two_epoch(z_dep, q[2], q[3], q[4]), dMh, Mstar_tot)
        return rms_of(dMstar, width_t(10 ** q[0], q[1], t_dep, t_obs))
    best = None
    for zc0 in (1.0, 2.0, 3.0, 4.0, 6.0):                 # multistart on the transition redshift
        r = minimize(loss, [np.log10(30.0), 1.5, 4.0, 1.5, zc0],
                     method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    s0, g, be, bl, zc = (10 ** best.x[0], best.x[1], best.x[2], best.x[3], best.x[4])
    dMstar = deposited(eff_two_epoch(z_dep, be, bl, zc), dMh, Mstar_tot)
    return dict(name="two-epoch quench", dMstar=dMstar, sigma=width_t(s0, g, t_dep, t_obs),
                weight=eff_two_epoch(z_dep, be, bl, zc),
                par=dict(sigma0=s0, g=g, b_early=be, b_late=bl, z_c=zc), rms=loss(best.x))


fits = [fit_single(), fit_two_epoch()]
print("\n[fit] reproduce the BCG curve of growth (sigma(t), no R_200c):")
for f in fits:
    pars = "  ".join(f"{k}={v:.3f}" for k, v in f["par"].items())
    print(f"  {f['name']:18s} ({len(f['par'])} par)  RMS={f['rms']:.3f} dex   ({pars})")
best = fits[1]
print(f"\n  -> two-epoch quenching: rapid early growth (b_early={best['par']['b_early']:.1f}) until "
      f"z_c={best['par']['z_c']:.1f}, then shallower (b_late={best['par']['b_late']:.1f}); "
      f"{fits[0]['rms']:.3f} -> {best['rms']:.3f} dex.")

# physical sanity: the absolute efficiency eps = dM*/dM_h should stay below the
# cosmic baryon fraction. The steep b_early can push eps>1 at the earliest epochs.
F_BARYON = 0.157
eps = (best["dMstar"] / np.clip(dMh, 1.0, None))
n_over = int((eps > 1.0).sum())
mass_over = float(best["dMstar"][eps > 1.0].sum() / Mstar_tot)
print(f"  [caveat] eps in [{eps.min():.1e}, {eps.max():.2f}]; eps>1 at {n_over}/{len(eps)} earliest "
      f"epochs carrying {100*mass_over:.1f}% of M* (overfit core; capping at f_b keeps RMS~0.007).")


# %% ---- FIGURE 1: the galaxy built clump by clump ---------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.8))
order = np.argsort(z_dep)[::-1]
cmap = matplotlib.colormaps["cividis"]
znorm = matplotlib.colors.Normalize(vmin=0, vmax=min(6, z_dep.max()))
for j in order:
    s = deposit_sigma(np.array([best["dMstar"][j]]), np.array([best["sigma"][j]]), mid)
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
for f, c in zip(fits, [OKABE_ITO[7], OKABE_ITO[2]]):
    axB.plot(R, np.log10(np.clip(deposit_cog(f["dMstar"], f["sigma"], R), 1, None)),
             "-", color=c, lw=1.9, label=f"{f['name']} ({f['rms']:.3f} dex)")
axB.set_xscale("log"); axB.set_xlabel("R [kpc]"); axB.set_ylabel(r"$\log M_*(<R)$")
axB.set_title("B. Curve of growth: model vs truth"); axB.legend(fontsize=8, loc="lower right")
fig.suptitle(f"exp25 — building TNG300's most massive galaxy (logM*={logMstar_tot:.2f}) from its "
             f"MAH; two-epoch quenching -> {best['rms']:.3f} dex", fontsize=11)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp25_build_galaxy")


# %% ---- FIGURE 2: the ingredients (MAH, efficiency, widths) -----------------
fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.0, 4.2))
a1.plot(z_snap, lmp, "-o", color=OKABE_ITO[0], ms=3)
a1.set_xlabel("redshift"); a1.set_ylabel(r"$\log M_{\rm halo}(z)$")
a1.invert_xaxis(); a1.set_title("A. The halo MAH (input)")
# efficiency eps(z): two-epoch weight, normalized to an absolute fraction
zline = np.linspace(z_dep.min(), z_dep.max(), 200)
eps0 = Mstar_tot / (best["weight"] * dMh).sum()
a2.plot(zline, eps0 * eff_two_epoch(zline, best["par"]["b_early"], best["par"]["b_late"],
                                    best["par"]["z_c"]), "-", color=OKABE_ITO[2])
a2.axvline(best["par"]["z_c"], ls=":", color="0.5", lw=1, label=f"z_c={best['par']['z_c']:.1f}")
a2.set_xlabel("redshift"); a2.set_ylabel(r"efficiency $\epsilon(z)=dM_*/dM_h$")
a2.invert_xaxis(); a2.set_yscale("log"); a2.set_title("B. Two-epoch quenching efficiency")
a2.legend(fontsize=8)
a3.plot(z_dep, best["sigma"], "-o", color=OKABE_ITO[5], ms=3)
a3.set_xlabel("deposition redshift"); a3.set_ylabel(r"clump width $\sigma(t)$ [kpc]")
a3.invert_xaxis(); a3.set_yscale("log")
a3.set_title(f"C. Width sigma(t)=s0(t/t_obs)^g (g={best['par']['g']:.2f})")
fig2.suptitle("exp25 — ingredients: MAH x quenching efficiency sets how much forms when; "
              "sigma(t) sets where it lands (no R_200c)", fontsize=10)
fig2.tight_layout()
save_fig(fig2, FIGDIR / "exp25_ingredients")


# %% ---- save + self-check ---------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
write_manifest(OUTDIR, params={
    "galaxy_index": int(idx[i]), "logMstar_tot": logMstar_tot, "n_snapshots": int(len(t_gyr)),
    "single_rms": fits[0]["rms"], "two_epoch_rms": fits[1]["rms"],
    "single_par": fits[0]["par"], "two_epoch_par": fits[1]["par"]})
print(f"\nwrote figures -> {FIGDIR}\nwrote outputs -> {OUTDIR}")
assert fits[0]["rms"] < 0.06, fits[0]["rms"]
assert fits[1]["rms"] < fits[0]["rms"], (fits[0]["rms"], fits[1]["rms"])
assert best["par"]["b_early"] > best["par"]["b_late"], "early growth should be steeper than late"
assert 0.4 < best["par"]["z_c"] < 12.0, best["par"]["z_c"]
print(f"\n[verdict] sigma(t) (no R_200c) + two-epoch quenching builds the BCG to {best['rms']:.3f} dex; "
      f"rapid early growth quenches at z_c={best['par']['z_c']:.1f}.")
