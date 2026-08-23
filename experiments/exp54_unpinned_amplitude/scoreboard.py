"""exp54 Stage 1 — the AMPLITUDE scoreboard.

What accuracy is achievable in predicting a massive galaxy's stellar mass from
its halo alone? Every row is scored on the SAME 5-fold galaxy-level split, at
all five epochs, against the same target, with the exp07 suite (RMS **and**
CRPS **and** interval coverage) rather than RMS alone.

This is model-independent: it defines the target the successor must hit, and it
is unaffected by whatever the deposition model turns out to be.

Target: ``log10 M*(<100 kpc)`` -- the measured anchor decided in plan §4.2,
obtained by log-log interpolation on the 24-point grid (interior, so
``np.interp``'s clamping is harmless; it must never be used to reach 500 kpc).

Row groups:
  A  halo-only statistical baselines -- what a regression can do
  B  the absolute deposition law     -- what the physical model can do
  C  shuffled-MAH controls           -- is it the HISTORY or just its endpoint?

Input conventions fixed in plan §4.4.5: the OFFICIAL DiffMAH is primary (its
four parameters are recovered by re-fitting the official curve, which IS a
DiffMAH model), our own exp10 ``dmah_*`` fit is carried only as a robustness
check, and the catalog ``logMh(z=0.4)`` is reported but not used as an input.

Concentration enters as the DIMENSIONLESS EXCESS ``log10[c200c / c_Diemer19]``
(plan §4.4.2), interpolated in ln(1+z) from the 16 catalog snapshots and set to
zero where the catalog has no valid fit -- "assume this halo is typical for its
mass and redshift", recorded rather than silently imputed.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/scoreboard.py [--rebuild]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink"):
    sys.path.insert(0, str(p))

import stage2_multiepoch as s2                          # noqa: E402
from hongshao.metrics import crps_gaussian, interval_coverage   # noqa: E402

sys.path.insert(0, str(HERE))
from halo import _assert_m200c_is_log10                 # noqa: E402

OUTDIR = HERE / "outputs"
CACHE = OUTDIR / "scoreboard_features.npz"
HS = OUTDIR / "halo_structure_history.npz"
POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
OFFICIAL = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/official_mah.npz"
TABLE = ROOT / "data/processed/tng300_072_z0p4.fits"

Z = [0.4, 0.7, 1.0, 1.5, 2.0]
ANCHOR_SNAP = [72, 59, 50, 40, 33]
NFOLD = 5
R_ANCHOR = 100.0
M_PIV = 13.5
N_SHUFFLE_BINS = 60
RULE = "=" * 96


# --------------------------------------------------------------------------- #
# features                                                                     #
# --------------------------------------------------------------------------- #
def _diemer19(logmh, z):
    """c200c from Diemer & Joyce (2019) at the TNG300 cosmology, h-free input."""
    from colossus.cosmology import cosmology
    from colossus.halo import concentration
    if "tng300" not in cosmology.cosmologies:
        cosmology.addCosmology("tng300", flat=True, H0=67.74, Om0=0.3089,
                               Ob0=0.0486, sigma8=0.8159, ns=0.9667)
    cosmology.setCosmology("tng300")
    out = np.full_like(logmh, np.nan, dtype=float)
    ok = np.isfinite(logmh)
    if ok.any():
        out[ok] = concentration.concentration(
            10.0 ** logmh[ok] * 0.6774, "200c", float(z), model="diemer19")
    return out


def build():
    """Assemble every feature array once and cache it."""
    from astropy.table import Table
    from hongshao.diffmah import fit_mah

    s2._w_init(None)
    gals, e = s2._W["gals"], s2._W["e"]
    n = len(gals)
    pop = np.load(POP)
    rows = np.array([g["row"] for g in gals])
    R = e.R
    lr = np.log10(R)

    # --- target ---------------------------------------------------------- #
    data = pop["data"][rows]
    y = np.array([[np.interp(np.log10(R_ANCHOR), lr, np.log10(d))
                   for d in data[:, k]] for k in range(5)]).T

    # --- official DiffMAH: the curve, and its four parameters ------------- #
    off = np.load(OFFICIAL)
    tg = off["cosmic_time_gyr"]
    o_idx = {int(v): i for i, v in enumerate(off["index"])}
    sel = np.array([o_idx[int(pop["index"][r])] for r in rows])
    curve = off["diffmah_log_mah_fit"][sel]                    # (n, 100)
    lmh_k = curve[:, ANCHOR_SNAP]                              # (n, 5)
    t0 = float(tg[72])
    dm4 = np.full((n, 4), np.nan)
    refit_rms = np.full(n, np.nan)
    for i in range(n):
        yy = curve[i]
        m = np.isfinite(yy)
        if m.sum() < 5:
            continue
        r = fit_mah(tg[m], yy[m], t0, t_min=2.0)
        if np.isfinite(r["logmp"]):
            dm4[i] = [r["logmp"], r["logtc"], r["early"], r["late"]]
            refit_rms[i] = r["rms"]

    # --- our own exp10 fit, the robustness check -------------------------- #
    t = Table.read(TABLE)
    tid = {int(v): i for i, v in enumerate(t["index"])}
    ts = np.array([tid[int(pop["index"][r])] for r in rows])
    ours = np.column_stack([np.asarray(t["dmah_" + k], float)[ts]
                            for k in ("logmp", "logtc", "early", "late")])

    # --- concentration excess over Diemer19, at the anchors and per deposit #
    hs = np.load(HS, allow_pickle=True)
    hs_idx = {int(v): i for i, v in enumerate(hs["index"])}
    hsel = np.array([hs_idx[int(pop["index"][r])] for r in rows])
    c_cat = np.where(hs["GroupFlag"][hsel] == 1, hs["c200c"][hsel], np.nan)
    hs_snap, hs_z = list(hs["snaps"]), hs["z"]
    _assert_m200c_is_log10(hs, pop)          # the column is ALREADY log10 Msun
    m200 = np.where(np.isfinite(c_cat), hs["M200c"][hsel], np.nan)
    excess_grid = np.full_like(c_cat, np.nan)
    for j, zz in enumerate(hs_z):
        excess_grid[:, j] = (np.log10(c_cat[:, j])
                             - np.log10(_diemer19(m200[:, j], zz)))
    # at the five profile epochs
    c_exc_k = np.column_stack([excess_grid[:, hs_snap.index(s)]
                               for s in ANCHOR_SNAP])

    # per-deposit, interpolated in ln(1+z); 0 where the catalog has nothing
    L = max(len(g["mah"]["z"]) for g in gals)
    z_dep = np.zeros((n, L)); dmh = np.zeros((n, L))
    lmh_dep = np.zeros((n, L)); mask = np.zeros((n, 5, L))
    exc_dep = np.zeros((n, L)); exc_have = np.zeros((n, L))
    xg = np.log1p(hs_z)
    order = np.argsort(xg)
    for i, g in enumerate(gals):
        mm = g["mah"]; ln_ = len(mm["z"])
        z_dep[i, :ln_] = mm["z"]; dmh[i, :ln_] = mm["dMh"]
        lmh_dep[i, :ln_] = mm["logMh_full"][1:]
        for k in range(5):
            mask[i, k, :ln_] = mm["snap"] <= ANCHOR_SNAP[k]
        v = excess_grid[i][order]
        good = np.isfinite(v)
        if good.sum() >= 2:
            exc_dep[i, :ln_] = np.interp(np.log1p(mm["z"]),
                                         xg[order][good], v[good])
            lo, hi = xg[order][good][[0, -1]]
            inside = (np.log1p(mm["z"]) >= lo) & (np.log1p(mm["z"]) <= hi)
            exc_have[i, :ln_] = inside
            exc_dep[i, :ln_] *= inside                   # 0 outside the support

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, y=y, lmh_cat=pop["logmh"][rows], lmh_k=lmh_k, dm4=dm4,
        refit_rms=refit_rms, ours=ours, c_exc_k=c_exc_k, fz2=pop["fz2"][rows],
        z_dep=z_dep, dmh=dmh, lmh_dep=lmh_dep, mask=mask, exc_dep=exc_dep,
        exc_have=exc_have)
    print(f"  built and cached {CACHE.name}")
    return dict(np.load(CACHE))


def load(rebuild=False):
    if CACHE.exists() and not rebuild:
        return dict(np.load(CACHE))
    return build()


# --------------------------------------------------------------------------- #
# scoring                                                                      #
# --------------------------------------------------------------------------- #
def folds(n):
    return np.arange(n) % NFOLD


def score_linear(y, feats):
    """CV (residual, sigma) for an OLS fit of ``y`` on ``feats``."""
    n = len(y)
    F = [np.nan_to_num(np.asarray(f, float), nan=0.0) for f in feats]
    fold = folds(n)
    res = np.full(n, np.nan); sig = np.full(n, np.nan)
    for f in range(NFOLD):
        tr, te = fold != f, fold == f
        D = lambda m: np.column_stack([np.ones(m.sum())] + [x[m] for x in F])
        b, *_ = np.linalg.lstsq(D(tr), y[tr], rcond=None)
        res[te] = y[te] - D(te) @ b
        sig[te] = np.std(y[tr] - D(tr) @ b)
    return res, sig


def law_predict(d, par, sub=slice(None), cross=False, conc=False):
    """log10 M*(<100) at the 5 epochs from the absolute deposition law."""
    a0, a_m, a_z = par[0], par[1], par[2]
    dlm = d["lmh_dep"][sub] - M_PIV
    ln1z = np.log1p(d["z_dep"][sub])
    le = a0 + a_m * dlm + a_z * ln1z
    i = 3
    if cross:
        le = le + par[i] * dlm * ln1z; i += 1
    if conc:
        le = le + par[i] * d["exc_dep"][sub]
    w = 10.0 ** le * d["dmh"][sub]
    return np.log10(np.clip(np.einsum("nkl,nl->nk", d["mask"][sub], w), 1e-30, None))


def score_law(d, y, p0, **kw):
    """CV (residual, sigma) for the fitted absolute law."""
    n = len(y)
    def rms(par, sub=slice(None)):
        return float(np.sqrt(np.nanmean((y[sub] - law_predict(d, par, sub, **kw)) ** 2)))
    fold = folds(n)
    res = np.full_like(y, np.nan); sig = np.full_like(y, np.nan)
    par_full = minimize(rms, p0, method="Nelder-Mead",
                        options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12)).x
    for f in range(NFOLD):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        pf = minimize(lambda q: rms(q, tr), par_full, method="Nelder-Mead",
                      options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12)).x
        res[te] = y[te] - law_predict(d, pf, te, **kw)
        sig[te] = np.std(y[tr] - law_predict(d, pf, tr, **kw), axis=0)
    return res, sig, par_full


def report(name, res, sig, note=""):
    """One scoreboard line: per-epoch scatter, CRPS, 90% coverage."""
    res = np.atleast_2d(res.T).T if res.ndim == 2 else res[:, None] * np.ones((1, 5))
    sig = np.atleast_2d(sig.T).T if sig.ndim == 2 else sig[:, None] * np.ones((1, 5))
    sc, cr, cv = [], [], []
    for k in range(5):
        r, s = res[:, k], sig[:, k]
        m = np.isfinite(r) & np.isfinite(s) & (s > 0)
        sc.append(np.std(r[m]))
        cr.append(np.mean(crps_gaussian(r[m], np.zeros(m.sum()), s[m])))
        cv.append(float(interval_coverage(
            r[m], np.zeros(m.sum()), s[m], (0.9,))[1][0]))
    print(f"  {name:<42}" + "".join(f"{v:8.4f}" for v in sc)
          + f"{np.mean(cr):9.4f}{np.mean(cv):9.3f}   {note}")
    return sc


def main(rebuild=False):
    d = load(rebuild)
    y = d["y"]; n = len(y)
    print(f"exp54 STAGE 1 — the amplitude scoreboard   "
          f"(n={n}, target log10 M*(<{R_ANCHOR:.0f} kpc), {NFOLD}-fold CV)")
    ok = np.isfinite(d["refit_rms"])
    print(f"official DiffMAH re-fit: {ok.sum()}/{n} recovered, median rms "
          f"{np.nanmedian(d['refit_rms']):.2e} dex (the official curve IS a "
          f"DiffMAH model)")
    hv = d["exc_have"]
    print(f"concentration excess available for "
          f"{100 * hv[d['dmh'] > 0].mean():.1f}% of deposits; 0 elsewhere\n")
    print(f"  {'row':<42}" + "".join(f"{f'z={z}':>8}" for z in Z)
          + f"{'CRPS':>9}{'cov90':>9}")
    print(f"  {'-' * 92}")

    DM4 = [d["dm4"][:, j] for j in range(4)]
    OURS = [d["ours"][:, j] for j in range(4)]
    CEX = [d["c_exc_k"][:, k] for k in range(5)]

    print("  GROUP A — halo-only statistical baselines")
    report("do-nothing (population sigma)",
           y - y.mean(0), np.tile(y.std(0), (n, 1)))
    for nm, F in [("catalog logMh(z=0.4)  [descendant]", [d["lmh_cat"]]),
                  ("official DiffMAH curve at z_k", None),
                  ("official DiffMAH 4 params", DM4),
                  ("  + concentration excess (z_k)", None),
                  ("  + fz2", None),
                  ("our exp10 DiffMAH 4 params  [robustness]", OURS)]:
        res = np.full_like(y, np.nan); sig = np.full_like(y, np.nan)
        for k in range(5):
            if nm.startswith("official DiffMAH curve"):
                F_k = [d["lmh_k"][:, k]]
            elif nm.startswith("  + concentration"):
                F_k = DM4 + [CEX[k]]
            elif nm.startswith("  + fz2"):
                F_k = DM4 + [CEX[k], d["fz2"]]
            else:
                F_k = F
            res[:, k], sig[:, k] = score_linear(y[:, k], F_k)
        report(nm, res, sig)

    print("\n  GROUP B — the absolute deposition law (fitted, not regressed)")
    for nm, p0, kw in [("3p  a0 + a_M dlogMh + a_z ln(1+z)",
                        [-2.5, -0.19, 0.36], {}),
                       ("4p  + mass x redshift cross",
                        [-2.5, -0.19, 0.36, 0.0], dict(cross=True)),
                       ("4p  + concentration excess",
                        [-2.5, -0.19, 0.36, 0.0], dict(conc=True)),
                       ("5p  + both",
                        [-2.5, -0.19, 0.36, 0.0, 0.0],
                        dict(cross=True, conc=True))]:
        res, sig, par = score_law(d, y, p0, **kw)
        report(nm, res, sig, note=f"par {np.round(par, 3)}")

    print("\n  GROUP C — shuffled-MAH controls (whole MAHs permuted within "
          f"{N_SHUFFLE_BINS} narrow final-mass bins)")
    rng = np.random.default_rng(0)
    perm = np.arange(n)
    for grp in np.array_split(np.argsort(d["lmh_k"][:, 0]), N_SHUFFLE_BINS):
        perm[grp] = rng.permutation(grp)
    ds = dict(d)
    for k in ("z_dep", "dmh", "lmh_dep", "mask", "exc_dep"):
        ds[k] = d[k][perm]
    res, sig, par = score_law(ds, y, [-2.5, -0.19, 0.36])
    report("3p law on SHUFFLED MAHs", res, sig)
    res = np.full_like(y, np.nan); sig = np.full_like(y, np.nan)
    dm4s = d["dm4"][perm]
    for k in range(5):
        res[:, k], sig[:, k] = score_linear(y[:, k], [dm4s[:, j] for j in range(4)])
    report("DiffMAH 4 params on SHUFFLED MAHs", res, sig)
    print("\n  Caveat: shuffling a whole MAH also swaps Mh(z_k), so at high z the")
    print("  control partly re-measures 'epoch-local halo mass matters'.")


if __name__ == "__main__":
    main("--rebuild" in sys.argv)
