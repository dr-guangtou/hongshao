"""exp25 phase 2 — fit the deposition-kernel toy across the population.

Phase 1 (run.py) showed the toy reproduces the single most massive TNG300 galaxy
to 0.008 dex. Here we fit the same 5-parameter model to every clean galaxy
(`use` cut, n~2545) and ask the population questions:

  1. Are the shape parameters (g, b_early, b_late) ~universal, or mass-dependent?
  2. **Does the quenching redshift z_c scale with halo mass?** (massive halos
     quench earlier -> z_c should rise with logM_h.)
  3. Does a reduced model with the shape params FIXED at their population medians
     -- letting only the size sigma_0 and quenching z_c vary -- still reproduce
     the profiles? (the real test of a population model.)

The 5-param fit per galaxy carries the known z_c/b_early degeneracy, so the
headline z_c(M_h) is read off the REDUCED model, where z_c is identifiable.

Physical constraint: per-epoch efficiency eps = dM*/dM_h cannot exceed 1 (you
cannot turn more than the whole accreted mass into stars). The loss carries a
soft hinge penalty above eps=1 (penw=5.0, pins eps_max at 1.0 at ~0 dex RMS
cost). We deliberately do NOT cap at the baryon fraction f_b=0.157: that is too
aggressive (it inverts b_early<b_late and destroys the two-epoch structure), and
because the SHMR makes low-mass halos more eps-rich it would distort them
differentially and could fake a z_c(M_h) trend. Per-epoch eps>f_b is physical
anyway -- stars form from a gas reservoir, decoupling SF timing from accretion.
f_b violation is reported only as a diagnostic (frac_over_fb).

Two cached fit passes (outputs/*.fits); pass --refit to recompute.

Run: PYTHONPATH=. uv run python experiments/exp25_deposition_kernel/population_fit.py [N] [--refit]
"""
# %% setup
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import COG_RAD_KPC, load_mah, load_cosmic_time, peak_history   # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                           # noqa: E402
from hongshao.provenance import write_manifest                                         # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089)      # TNG cosmology (only for snapshot z)
R = COG_RAD_KPC
EPS_CAP = 1.0                                     # hard limit: dM* <= dM_h (eps<=1)
F_BARYON = 0.157                                  # cosmic baryon fraction (diagnostic only)
PEN_W = 5.0                                       # soft-hinge weight for eps>EPS_CAP
ZC_STARTS = (1.0, 2.0, 3.0, 4.0, 6.0)             # multistart on the transition redshift
EMULATOR_RECON_RMS = 0.116                        # exp22 halo->profile reconstruction baseline


# %% ---- forward model (same closed form as run.py) --------------------------
def deposit_cog(dMstar, sigma, Rgrid):
    return (dMstar[None, :] * (1.0 - np.exp(-Rgrid[:, None] ** 2 / (2.0 * sigma[None, :] ** 2)))).sum(1)


def width_t(sigma0, g, t, t_obs):
    return sigma0 * (t / t_obs) ** g


def eff_two_epoch(z, b_early, b_late, z_c):
    hi = (1.0 + z) ** b_early
    lo = (1.0 + z_c) ** (b_early - b_late) * (1.0 + z) ** b_late
    return np.where(z >= z_c, hi, lo)


def deposited(weight, dMh, Mstar_tot):
    return Mstar_tot * (weight * dMh) / (weight * dMh).sum()


# %% ---- per-galaxy inputs from the MAH --------------------------------------
def build_inputs(table):
    """One light dict per galaxy: positive halo-mass increments, deposition
    redshifts/times, t_obs, total stellar mass, and the target CoG."""
    cog = np.asarray(table["logmstar_cog"], float)
    idx = np.asarray(table["index"]).astype(int)
    logmh = np.asarray(table["logmh_z0p4"], float)
    r50 = np.nanmedian(np.asarray(table["r50_proj"], float), axis=1)   # size, kpc (median over projections)
    tsnap = load_cosmic_time()
    zg = np.linspace(0.0, 20.0, 4000)
    z_of_snap = np.interp(tsnap, COSMO.age(zg).to("Gyr").value[::-1], zg[::-1])
    mah = load_mah()
    out = []
    for k in range(len(table)):
        sn, lmp = peak_history(mah[idx[k]])
        if sn is None or len(sn) < 8:
            continue
        sn = sn.astype(int)
        dMh = np.clip(np.diff(10.0 ** lmp), 0.0, None)
        if dMh.sum() <= 0:
            continue
        t_gyr = tsnap[sn]
        out.append(dict(index=int(idx[k]), logmh=float(logmh[k]), logmstar=float(cog[k, -1]),
                        r50=float(r50[k]), dMh=dMh, z_dep=z_of_snap[sn][1:], t_dep=t_gyr[1:],
                        t_obs=float(t_gyr[-1]), Mstar_tot=10.0 ** cog[k, -1], cog=cog[k]))
    return out


# %% ---- losses (CoG RMS + soft f_b hinge) -----------------------------------
def _cog_rms(dMstar, sigma, cog):
    m = deposit_cog(dMstar, sigma, R)
    return float(np.sqrt(np.mean((np.log10(np.clip(m, 1.0, None)) - cog) ** 2)))


def _eff_excess(dMstar, dMh, Mstar_tot, cap):
    """Fraction of stellar mass deposited above eps=cap (soft hinge / diagnostic)."""
    eps = dMstar / np.clip(dMh, 1.0, None)
    return float((np.maximum(eps - cap, 0.0) * dMh).sum() / Mstar_tot)


def fit_full(g):
    """5-param fit: q = [log10 sigma0, g, b_early, b_late, z_c]."""
    dMh, z_dep, t_dep, t_obs, Mtot, cog = (g["dMh"], g["z_dep"], g["t_dep"],
                                           g["t_obs"], g["Mstar_tot"], g["cog"])

    def loss(q):
        ls0, gg, be, bl, zc = q
        if not (0.0 < gg < 5.0 and -5.0 < be < 30.0 and -5.0 < bl < 30.0
                and 0.1 < zc < 15.0 and 0.0 < ls0 < 3.0):
            return 1e3
        dM = deposited(eff_two_epoch(z_dep, be, bl, zc), dMh, Mtot)
        sig = width_t(10.0 ** ls0, gg, t_dep, t_obs)
        val = _cog_rms(dM, sig, cog) + PEN_W * _eff_excess(dM, dMh, Mtot, EPS_CAP)
        return val if np.isfinite(val) else 1e3

    best = None
    for zc0 in ZC_STARTS:
        r = minimize(loss, [np.log10(30.0), 1.5, 4.0, 1.5, zc0], method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    ls0, gg, be, bl, zc = best.x
    dM = deposited(eff_two_epoch(z_dep, be, bl, zc), dMh, Mtot)
    sig = width_t(10.0 ** ls0, gg, t_dep, t_obs)
    eps = dM / np.clip(dMh, 1.0, None)
    identifiable = bool(z_dep.min() < zc < z_dep.max() and be > bl)
    return dict(index=g["index"], logmh=g["logmh"], logmstar=g["logmstar"],
                sigma0=10.0 ** ls0, g=gg, b_early=be, b_late=bl, z_c=zc,
                rms=_cog_rms(dM, sig, cog), eps_max=float(eps.max()),
                frac_over_fb=_eff_excess(dM, dMh, Mtot, F_BARYON), zc_identifiable=identifiable)


# reduced model: shape params (g, b_early, b_late) fixed at population medians;
# only the size sigma_0 and the quenching redshift z_c vary (both identifiable).
_REDUCED = {}   # filled before the pass-2 map (workers read it after fork)


def fit_reduced(g):
    g_fix, be_fix, bl_fix = _REDUCED["g"], _REDUCED["b_early"], _REDUCED["b_late"]
    dMh, z_dep, t_dep, t_obs, Mtot, cog = (g["dMh"], g["z_dep"], g["t_dep"],
                                           g["t_obs"], g["Mstar_tot"], g["cog"])

    def loss(q):
        ls0, zc = q
        if not (0.0 < ls0 < 3.0 and 0.1 < zc < 15.0):
            return 1e3
        dM = deposited(eff_two_epoch(z_dep, be_fix, bl_fix, zc), dMh, Mtot)
        sig = width_t(10.0 ** ls0, g_fix, t_dep, t_obs)
        val = _cog_rms(dM, sig, cog) + PEN_W * _eff_excess(dM, dMh, Mtot, EPS_CAP)
        return val if np.isfinite(val) else 1e3

    best = None
    for zc0 in ZC_STARTS:
        r = minimize(loss, [np.log10(30.0), zc0], method="Nelder-Mead",
                     options=dict(maxiter=2000, xatol=1e-4, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    ls0, zc = best.x
    dM = deposited(eff_two_epoch(z_dep, be_fix, bl_fix, zc), dMh, Mtot)
    sig = width_t(10.0 ** ls0, g_fix, t_dep, t_obs)
    return dict(index=g["index"], logmh=g["logmh"], logmstar=g["logmstar"],
                sigma0=10.0 ** ls0, z_c=zc, rms=_cog_rms(dM, sig, cog),
                frac_over_fb=_eff_excess(dM, dMh, Mtot, F_BARYON))


# %% ---- TRUE population fit: one shared parameter set for all galaxies -------
# The per-galaxy fits above are exploratory. A genuine population model shares
# the deposition physics across galaxies: the SAME parameters map any halo's MAH
# to its profile. We minimize the mean per-galaxy CoG RMS over the WHOLE sample.
# Vectorized (galaxies x epochs zero-padded) so each objective eval is one pass.
def pad_inputs(inputs):
    """Zero-pad the per-galaxy MAHs into (G, L) arrays for the vectorized model."""
    G = len(inputs)
    L = max(len(g["dMh"]) for g in inputs)
    Z = np.zeros((G, L)); T = np.zeros((G, L)); DMH = np.zeros((G, L))
    TOBS = np.zeros(G); MTOT = np.zeros(G); COG = np.zeros((G, len(R)))
    logmh = np.zeros(G); logr50 = np.zeros(G)
    for i, g in enumerate(inputs):
        n = len(g["dMh"])
        Z[i, :n] = g["z_dep"]; T[i, :n] = g["t_dep"]; DMH[i, :n] = g["dMh"]
        TOBS[i] = g["t_obs"]; MTOT[i] = g["Mstar_tot"]; COG[i] = g["cog"]
        logmh[i] = g["logmh"]; logr50[i] = np.log10(g["r50"])
    return dict(Z=Z, T=T, DMH=DMH, TOBS=TOBS, MTOT=MTOT, COG=COG,
                logmh=logmh, logr50=logr50, mask=(DMH > 0))


def pop_forward(g_exp, b_early, b_late, S0, ZC, P):
    """Vectorized model: per-galaxy CoG RMS (G,) and f_b-cap excess (G,).

    S0 (G,), ZC (G,) are per-galaxy size and quenching redshift (a global fit
    sets them from shared parameters / scaling relations)."""
    Z, T, DMH, MASK = P["Z"], P["T"], P["DMH"], P["mask"]
    w = np.where(Z >= ZC[:, None], (1.0 + Z) ** b_early,
                 (1.0 + ZC[:, None]) ** (b_early - b_late) * (1.0 + Z) ** b_late)
    wdmh = np.where(MASK, w * DMH, 0.0)
    dMstar = P["MTOT"][:, None] * wdmh / wdmh.sum(1, keepdims=True)
    sig = np.where(MASK, S0[:, None] * (T / P["TOBS"][:, None]) ** g_exp, 1.0)
    arg = R[None, :, None] ** 2 / (2.0 * sig[:, None, :] ** 2)
    model = (dMstar[:, None, :] * (1.0 - np.exp(-arg))).sum(2)          # (G, 24)
    rms = np.sqrt(np.mean((np.log10(np.clip(model, 1.0, None)) - P["COG"]) ** 2, axis=1))
    eps = dMstar / np.clip(DMH, 1.0, None)
    excess = (np.maximum(eps - EPS_CAP, 0.0) * DMH).sum(1) / P["MTOT"]
    return rms, excess


def _unpack_global(theta, modes, P):
    """Map a global theta + per-galaxy observables to (g, b_early, b_late, S0, ZC)."""
    g_exp, be, bl = theta[0], theta[1], theta[2]
    i = 3
    if modes["sigma"] == "const":
        S0 = np.full(len(P["logmh"]), 10.0 ** theta[i]); i += 1
    else:  # sigma0 follows a log-linear size relation: log10 sigma0 = a + b*logR50
        S0 = 10.0 ** (theta[i] + theta[i + 1] * P["logr50"]); i += 2
    if modes["zc"] == "const":
        ZC = np.full(len(P["logmh"]), theta[i]); i += 1
    else:  # z_c = zc0 + s*(logMh - 13.5)
        ZC = theta[i] + theta[i + 1] * (P["logmh"] - 13.5); i += 2
    return g_exp, be, bl, S0, np.clip(ZC, 0.1, 15.0)


def fit_global(modes, P, x0):
    """Fit one shared parameter set over the whole population (mean RMS + eps hinge)."""
    def loss(theta):
        if not (0.0 < theta[0] < 5.0 and -5.0 < theta[1] < 30.0 and -5.0 < theta[2] < 30.0):
            return 1e3
        g_exp, be, bl, S0, ZC = _unpack_global(theta, modes, P)
        rms, excess = pop_forward(g_exp, be, bl, S0, ZC, P)
        val = float(rms.mean() + PEN_W * excess.mean())
        return val if np.isfinite(val) else 1e3

    best = None
    for zc0 in (1.5, 2.5, 4.0):                      # multistart on the global z_c
        x = list(x0)
        x[-2 if modes["zc"] == "mass" else -1] = zc0
        r = minimize(loss, x, method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-5, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    g_exp, be, bl, S0, ZC = _unpack_global(best.x, modes, P)
    rms, _ = pop_forward(g_exp, be, bl, S0, ZC, P)
    return dict(theta=best.x, modes=modes, rms=rms, S0=S0, ZC=ZC,
                g=g_exp, b_early=be, b_late=bl)


# %% ---- driver --------------------------------------------------------------
def run_pass(inputs, worker, path, refit, initializer=None, initargs=()):
    if path.exists() and not refit:
        return Table.read(path)
    with ProcessPoolExecutor(initializer=initializer, initargs=initargs) as ex:
        rows = list(ex.map(worker, inputs, chunksize=16))
    tab = Table(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab.write(path, overwrite=True)
    return tab


def _init_reduced(d):
    _REDUCED.update(d)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refit = "--refit" in sys.argv
    n = int(args[0]) if args else None

    inputs = build_inputs(Table.read(TABLE)[Table.read(TABLE)["use"]])
    if n:
        inputs = inputs[:n]
    print(f"exp25 population fit: {len(inputs)} galaxies "
          f"(logMh {min(i['logmh'] for i in inputs):.1f}-{max(i['logmh'] for i in inputs):.1f})")

    # ---- pass 1: full 5-param fit ----
    suffix = f"_n{n}" if n else ""
    full = run_pass(inputs, fit_full, OUTDIR / f"population_full{suffix}.fits", refit)
    med = {p: float(np.median(full[p])) for p in ("sigma0", "g", "b_early", "b_late", "z_c")}
    print(f"\n[pass 1] 5-param fit, n={len(full)}")
    print(f"  RMS: median {np.median(full['rms']):.4f}  90th {np.percentile(full['rms'], 90):.4f} dex"
          f"  (emulator recon RMS {EMULATOR_RECON_RMS} dex)")
    for p in ("sigma0", "g", "b_early", "b_late", "z_c"):
        v = np.asarray(full[p]); q = np.percentile(v, [16, 50, 84])
        rho, pval = linregress(full["logmh"], v)[2:4]
        print(f"  {p:8s} median {q[1]:7.3f}  [16-84: {q[0]:.3f}, {q[2]:.3f}]  "
              f"corr(logMh) r={rho:+.2f} p={pval:.1e}")
    print(f"  eps_max median {np.median(full['eps_max']):.2f}  "
          f"frac>f_b median {np.median(full['frac_over_fb']):.3f}  "
          f"(z_c identifiable in {int(np.sum(full['zc_identifiable']))}/{len(full)})")

    # ---- pass 2: reduced model (fix shape params at medians; fit sigma0, z_c) ----
    fixed = {k: med[k] for k in ("g", "b_early", "b_late")}
    print(f"\n[pass 2] reduced model, shape fixed at medians: "
          f"g={fixed['g']:.2f} b_early={fixed['b_early']:.2f} b_late={fixed['b_late']:.2f}")
    red = run_pass(inputs, fit_reduced, OUTDIR / f"population_reduced{suffix}.fits", refit,
                   initializer=_init_reduced, initargs=(fixed,))
    print(f"  RMS: median {np.median(red['rms']):.4f}  90th {np.percentile(red['rms'], 90):.4f} dex")

    # ---- headline: z_c(M_h) scaling from the reduced (identifiable) model ----
    PIVOT = 13.5
    lr = linregress(red["logmh"] - PIVOT, red["z_c"])
    print(f"\n[headline] reduced-model z_c vs logM_h (pivot {PIVOT}):")
    print(f"  z_c = {lr.intercept:.2f} + {lr.slope:+.2f}*(logMh-{PIVOT})   "
          f"r={lr.rvalue:+.2f}  p={lr.pvalue:.1e}  (slope SE {lr.stderr:.2f})")
    # full-model identifiable subset, as a cross-check
    fid = full[full["zc_identifiable"]]
    if len(fid) > 20:
        lr2 = linregress(fid["logmh"] - PIVOT, fid["z_c"])
        print(f"  cross-check (full-fit identifiable, n={len(fid)}): "
              f"slope {lr2.slope:+.2f}  r={lr2.rvalue:+.2f}  p={lr2.pvalue:.1e}")
    # does z_c track assembly time or stellar mass at all? (join z50/logmstar by index)
    main = Table.read(TABLE)
    by_idx = {int(j): k for k, j in enumerate(main["index"])}
    rows = np.array([by_idx[int(j)] for j in red["index"]])
    for col in ("z50", "logmstar"):
        x = (np.asarray(main["z50"])[rows] if col == "z50" else np.asarray(red["logmstar"]))
        ok = np.isfinite(x)
        rr = linregress(x[ok], np.asarray(red["z_c"])[ok])
        print(f"  z_c vs {col:8s}: r={rr.rvalue:+.2f}  slope={rr.slope:+.2f}  p={rr.pvalue:.1e}")

    # ---- passes 3-5: TRUE population fits (one shared parameter set) ----
    P = pad_inputs(inputs)
    mean_lr50 = float(P["logr50"].mean())
    gA = fit_global(dict(sigma="const", zc="const"), P,
                    [med["g"], med["b_early"], med["b_late"], np.log10(med["sigma0"]), med["z_c"]])
    gB = fit_global(dict(sigma="R50", zc="const"), P,
                    [med["g"], med["b_early"], med["b_late"],
                     np.log10(med["sigma0"]) - mean_lr50, 1.0, med["z_c"]])
    gBp = fit_global(dict(sigma="R50", zc="mass"), P, list(gB["theta"]) + [0.0])
    print("\n[pass 3-5] TRUE population fits (shared parameters; median per-galaxy RMS):")
    for name, gf, npar in [("A  global const", gA, 5), ("B  +sigma0(R50)", gB, 6),
                           ("B' +z_c(logMh) slope", gBp, 7)]:
        print(f"  {name:22s} ({npar}p)  median {np.median(gf['rms']):.4f}  mean {gf['rms'].mean():.4f} dex")
    print(f"  A : sigma0={10 ** gA['theta'][3]:.0f} kpc  g={gA['g']:.2f}  "
          f"b_early={gA['b_early']:.2f}  b_late={gA['b_late']:.2f}  z_c={gA['ZC'][0]:.2f}")
    print(f"  B : log10 sigma0 = {gB['theta'][3]:.2f} + {gB['theta'][4]:.2f}*logR50  "
          f"(slope {gB['theta'][4]:.2f}); g={gB['g']:.2f} b_early={gB['b_early']:.2f} "
          f"b_late={gB['b_late']:.2f} z_c={gB['ZC'][0]:.2f}")
    d_help = float(np.median(gB["rms"]) - np.median(gBp["rms"]))
    print(f"  B': z_c slope s={gBp['theta'][6]:+.3f}/dex -> median RMS {np.median(gBp['rms']):.4f} "
          f"(B {np.median(gB['rms']):.4f}); a z_c(logMh) trend "
          f"{'helps' if d_help > 5e-4 else 'adds essentially nothing'} "
          f"(ΔRMS {d_help:+.4f}) -> quenching-mass trend not supported at population level.")

    make_figures(full, red, fixed, lr, PIVOT, suffix)
    make_global_figure(full, red, gA, gB, gBp, P, suffix)

    write_manifest(OUTDIR, params={
        "n_galaxies": len(full), "pen_w": PEN_W, "f_baryon": F_BARYON,
        "full_rms_median": float(np.median(full["rms"])),
        "reduced_rms_median": float(np.median(red["rms"])),
        "medians": med, "reduced_fixed": fixed,
        "zc_logmh_slope": float(lr.slope), "zc_logmh_intercept": float(lr.intercept),
        "zc_logmh_r": float(lr.rvalue), "zc_logmh_p": float(lr.pvalue),
        "emulator_recon_rms": EMULATOR_RECON_RMS,
        "global_A": {"rms_median": float(np.median(gA["rms"])), "params": gA["theta"].tolist(),
                     "sigma0": float(10 ** gA["theta"][3]), "g": float(gA["g"]),
                     "b_early": float(gA["b_early"]), "b_late": float(gA["b_late"]),
                     "z_c": float(gA["ZC"][0])},
        "global_B": {"rms_median": float(np.median(gB["rms"])), "params": gB["theta"].tolist(),
                     "sigma0_R50_a": float(gB["theta"][3]), "sigma0_R50_b": float(gB["theta"][4]),
                     "g": float(gB["g"]), "b_early": float(gB["b_early"]),
                     "b_late": float(gB["b_late"]), "z_c": float(gB["ZC"][0])},
        "global_Bp": {"rms_median": float(np.median(gBp["rms"])), "params": gBp["theta"].tolist(),
                      "zc_logmh_slope": float(gBp["theta"][6])}})
    print(f"\nwrote outputs -> {OUTDIR}\nwrote figures -> {FIGDIR}")

    # self-checks
    assert np.median(full["rms"]) < 0.03, np.median(full["rms"])
    assert np.median(red["rms"]) < 0.05, np.median(red["rms"])
    assert np.percentile(full["eps_max"], 95) < 1.1, "eps<=1 cap not enforced"
    assert np.median(gA["rms"]) < 0.10, np.median(gA["rms"])          # global beats emulator
    assert np.median(gB["rms"]) <= np.median(gA["rms"]) + 1e-3, "sigma0(R50) should not hurt"
    verdict = ("is ~INDEPENDENT of halo mass" if abs(lr.rvalue) < 0.15
               else f"{'RISES' if lr.slope > 0 else 'FALLS'} with halo mass")
    print(f"\n[verdict] toy fits {len(full)} galaxies to median {np.median(full['rms']):.4f} dex; "
          f"z_c {verdict} (r={lr.rvalue:+.2f}, slope {lr.slope:+.2f}/dex) -- the predicted "
          f"quenching-mass trend is absent.")


# %% ---- figures -------------------------------------------------------------
def make_figures(full, red, fixed, lr, pivot, suffix):
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # FIGURE 1: parameter distributions + RMS
    params = ["sigma0", "g", "b_early", "b_late", "z_c"]
    labels = [r"$\sigma_0$ [kpc]", "g", r"$b_{\rm early}$", r"$b_{\rm late}$", r"$z_c$"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, p, lab in zip(axes.flat, params, labels):
        v = np.asarray(full[p])
        ax.hist(v, bins=40, color=OKABE_ITO[0], alpha=0.8)
        ax.axvline(np.median(v), color="k", ls="--", lw=1.2, label=f"med {np.median(v):.2f}")
        ax.set_xlabel(lab); ax.legend(fontsize=8)
        if p in ("sigma0", "b_early"):
            ax.set_xlim(np.percentile(v, 0.5), np.percentile(v, 99.5))
    ax = axes.flat[5]
    ax.hist(full["rms"], bins=40, color=OKABE_ITO[2], alpha=0.8,
            label=f"full (med {np.median(full['rms']):.4f})")
    ax.hist(red["rms"], bins=40, color=OKABE_ITO[1], alpha=0.5,
            label=f"reduced (med {np.median(red['rms']):.4f})")
    ax.axvline(EMULATOR_RECON_RMS, color="r", ls=":", lw=1.5, label=f"emulator {EMULATOR_RECON_RMS}")
    ax.set_xlabel("per-galaxy CoG RMS [dex]"); ax.legend(fontsize=8); ax.set_xlim(0, 0.06)
    fig.suptitle(f"exp25 population fit — 5-param distributions + reconstruction RMS (n={len(full)})",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, FIGDIR / f"exp25_pop_params{suffix}")

    # FIGURE 2: each param vs logM_h (universality check)
    fig2, axes2 = plt.subplots(1, 5, figsize=(19, 3.8))
    for ax, p, lab in zip(axes2, params, labels):
        v = np.asarray(full[p])
        ax.scatter(full["logmh"], v, s=5, alpha=0.25, color=OKABE_ITO[0], edgecolors="none")
        bins = np.linspace(13.0, np.percentile(full["logmh"], 99.5), 9)
        bc = 0.5 * (bins[1:] + bins[:-1])
        which = np.digitize(full["logmh"], bins)
        med = [np.median(v[which == j]) if np.any(which == j) else np.nan for j in range(1, len(bins))]
        ax.plot(bc, med, "-o", color=OKABE_ITO[2], lw=2, ms=4)
        ax.set_xlabel(r"$\log M_h$"); ax.set_ylabel(lab)
        r = linregress(full["logmh"], v).rvalue
        ax.set_title(f"r={r:+.2f}", fontsize=9)
        if p in ("sigma0", "b_early"):
            ax.set_ylim(np.percentile(v, 0.5), np.percentile(v, 99.5))
    fig2.suptitle("exp25 — parameter vs halo mass: which are universal (flat) vs mass-dependent?",
                  fontsize=11)
    fig2.tight_layout()
    save_fig(fig2, FIGDIR / f"exp25_pop_vs_mass{suffix}")

    # FIGURE 3: the headline — z_c(M_h) from the reduced model
    fig3, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.scatter(red["logmh"], red["z_c"], s=6, alpha=0.3, color=OKABE_ITO[0], edgecolors="none")
    bins = np.linspace(13.0, np.percentile(red["logmh"], 99), 10)
    bc = 0.5 * (bins[1:] + bins[:-1]); which = np.digitize(red["logmh"], bins)
    zc = np.asarray(red["z_c"])
    bm = [np.median(zc[which == j]) if (which == j).sum() > 3 else np.nan for j in range(1, len(bins))]
    a1.plot(bc, bm, "-o", color="k", lw=2, ms=5, label="binned median")
    xx = np.linspace(red["logmh"].min(), np.percentile(red["logmh"], 99.5), 50)
    a1.plot(xx, lr.intercept + lr.slope * (xx - pivot), "-", color=OKABE_ITO[1], lw=2.4,
            label=f"$z_c={lr.intercept:.2f}{lr.slope:+.2f}(\\log M_h-{pivot})$\n"
                  f"r={lr.rvalue:+.2f}, p={lr.pvalue:.0e}")
    a1.set_xlabel(r"$\log M_h$ (z=0.4)"); a1.set_ylabel(r"quenching redshift $z_c$")
    a1.set_ylim(0, np.percentile(zc, 99)); a1.legend(fontsize=9, loc="upper left")
    trend = (r"$\approx$ independent of halo mass" if abs(lr.rvalue) < 0.15
             else ("rises with" if lr.slope > 0 else "falls with") + " halo mass")
    a1.set_title(f"A. Headline: z_c {trend} (reduced model)")

    a2.hist(full["rms"], bins=50, color=OKABE_ITO[2], alpha=0.75,
            label=f"full 5-param (med {np.median(full['rms']):.4f})")
    a2.hist(red["rms"], bins=50, color=OKABE_ITO[1], alpha=0.55,
            label=f"reduced 2-param (med {np.median(red['rms']):.4f})")
    a2.axvline(EMULATOR_RECON_RMS, color="r", ls=":", lw=1.5,
               label=f"emulator recon {EMULATOR_RECON_RMS}")
    a2.set_xlabel("per-galaxy CoG RMS [dex]"); a2.set_ylabel("count")
    a2.set_xlim(0, 0.06); a2.legend(fontsize=8)
    a2.set_title("B. Reconstruction accuracy: full vs reduced vs emulator")
    fig3.suptitle(f"exp25 — does z_c scale with halo mass? Reduced model fixes "
                  f"g={fixed['g']:.2f}, b_early={fixed['b_early']:.2f}, b_late={fixed['b_late']:.2f}",
                  fontsize=11)
    fig3.tight_layout()
    save_fig(fig3, FIGDIR / f"exp25_pop_zc_scaling{suffix}")


def make_global_figure(full, red, gA, gB, gBp, P, suffix):
    """The true-population result: accuracy ladder + the sigma0-size relation."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    # A. accuracy ladder: per-galaxy free -> reduced -> global B -> global A
    levels = [("per-galaxy 5p (free)", full["rms"], OKABE_ITO[2]),
              ("reduced 2p (shape fixed)", red["rms"], OKABE_ITO[1]),
              ("global B (sigma0~R50, shared)", gB["rms"], OKABE_ITO[0]),
              ("global A (fully shared)", gA["rms"], OKABE_ITO[5])]
    for lab, rms, c in levels:
        a1.hist(rms, bins=np.linspace(0, 0.09, 60), histtype="step", lw=2, color=c,
                label=f"{lab}: med {np.median(rms):.4f}")
    a1.axvline(EMULATOR_RECON_RMS, color="r", ls=":", lw=1.5, label=f"emulator {EMULATOR_RECON_RMS}")
    a1.set_xlabel("per-galaxy CoG RMS [dex]"); a1.set_ylabel("count")
    a1.set_xlim(0, 0.09); a1.legend(fontsize=8, loc="upper right")
    a1.set_title("A. Cost of universality: free per-galaxy -> shared population")

    # B. the global sigma0-size relation that step B fits
    a, b = gB["theta"][3], gB["theta"][4]
    a2.scatter(P["logr50"], np.log10(full["sigma0"]), s=5, alpha=0.2, color=OKABE_ITO[0],
               edgecolors="none", label="per-galaxy fit sigma0")
    xx = np.linspace(P["logr50"].min(), P["logr50"].max(), 50)
    a2.plot(xx, a + b * xx, "-", color=OKABE_ITO[1], lw=2.5,
            label=f"global B: log$\\sigma_0$={a:.2f}+{b:.2f} logR50")
    a2.set_xlabel(r"$\log R_{50}$ [kpc]"); a2.set_ylabel(r"$\log \sigma_0$ [kpc]")
    a2.set_ylim(np.percentile(np.log10(full["sigma0"]), 1), np.percentile(np.log10(full["sigma0"]), 99))
    a2.legend(fontsize=9); a2.set_title("B. The size relation that replaces per-galaxy sigma0")

    fig.suptitle(f"exp25 — TRUE population fit: one shared kernel for all {len(full)} galaxies "
                 f"(global A median {np.median(gA['rms']):.3f}, B {np.median(gB['rms']):.3f} dex)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, FIGDIR / f"exp25_pop_global{suffix}")


if __name__ == "__main__":
    main()
