"""Reproduce the §3.2 measurements in `2026-08-22-exp54-unpinned-amplitude.md`.

This is a PLANNING PROBE, not exp54. It exists so the eight measurements that
revision 2 of the exp54 plan rests on can be re-derived rather than trusted,
and so the next session can vary them cheaply before committing to a design.
It touches nothing in `hongshao/` and writes nothing.

What it measures, in the plan's section order:

  A  the anchor aperture is statistically free (M<100 / M<148 / M<500)
  B  epoch-local vs descendant conditioning (the 1.7x at z=2)
  C  the shape-fitted window is a bad amplitude model (slope 0.582 at z=0.4)
  D  the plan's 3-parameter efficiency law, 5-fold CV + shuffled-MAH control
  E  the mass x redshift cross-term, and the c200c residual correlation
  F  the free lognormal window buys nothing over the monotone form

Every predictor scatter is 5-fold GALAXY-LEVEL cross-validation on identical
folds. The target is `log10 M*(<100 kpc)`, obtained by log-log interpolation on
the measured 24-point grid -- interior, so `np.interp`'s CLAMPING is harmless
here. It must never be used to reach 500 kpc.

Deposit bookkeeping, stated because an off-by-one is a silent 0.01-0.03 dex
systematic in `a_M`: `dMh = diff(10**logMh_full)`, so deposit `i` spans
`[t_i, t_{i+1}]`, and this module aligns it with `logMh_full[1:]` -- the
POST-deposit halo mass.

Run (~4 min):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof \
    PYTHONPATH=.:experiments/exp38_deposit_rethink \
    uv run python doc/plans/2026-08-22-exp54-evidence-probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import stage2_multiepoch as s2                                    # noqa: E402

Z = [0.4, 0.7, 1.0, 1.5, 2.0]
NFOLD = 5
MPIV = 13.5
R_ANCHOR = 100.0
POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
M500 = ROOT / "experiments/exp35_total_norm/outputs/m500.npz"
#: the adopted exp40 theta's efficiency window -- the SHAPE-fitted one
ADOPTED = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"


def load():
    """Everything the probe needs, as one flat dict of aligned arrays."""
    s2._w_init(None)
    gals, e = s2._W["gals"], s2._W["e"]
    pop = np.load(POP)
    rows = np.array([g["row"] for g in gals])
    n, R = len(gals), e.R
    lr = np.log10(R)
    data = pop["data"][rows]

    def aperture(r_kpc):
        assert r_kpc <= R[-1] + 1e-9, "np.interp CLAMPS -- stay inside the grid"
        return np.array([[np.interp(np.log10(r_kpc), lr, np.log10(d))
                          for d in data[:, k]] for k in range(5)]).T

    # per-deposit arrays, right-padded into one rectangular block
    L = max(len(g["mah"]["z"]) for g in gals)
    z_dep = np.zeros((n, L))
    dmh = np.zeros((n, L))
    lmh_dep = np.zeros((n, L))
    mask = np.zeros((n, 5, L))
    for i, g in enumerate(gals):
        m = g["mah"]
        ln = len(m["z"])
        z_dep[i, :ln] = m["z"]
        dmh[i, :ln] = m["dMh"]
        lmh_dep[i, :ln] = m["logMh_full"][1:]          # POST-deposit mass
        for k in range(5):
            mask[i, k, :ln] = m["snap"] <= e.ANCHOR_SNAP[k]
    assert np.all(lmh_dep[:, :3] > 0), "deposit-time halo mass missing"

    return dict(
        n=n, R=R, gals=gals,
        y=aperture(R_ANCHOR),
        y148=np.log10(data[:, :, -1]),
        y500=np.log10(np.load(M500)["m500"][rows]),
        m100_over_m148=10.0 ** aperture(R_ANCHOR) / data[:, :, -1],
        lmh_k=pop["logmh_zk_diffmah"][rows],
        lmh_desc=pop["logmh"][rows],
        c200c=pop["c200c"][rows], fz2=pop["fz2"][rows],
        z_dep=z_dep, ln1z=np.log1p(z_dep), dmh=dmh,
        dlm=lmh_dep - MPIV, mask=mask,
    )


def folds(n):
    return np.arange(n) % NFOLD


def cv_linear(y, features):
    """5-fold CV residual scatter of an OLS fit of ``y`` on ``features``."""
    ok = np.isfinite(y) & np.all([np.isfinite(f) for f in features], axis=0)
    y, features = y[ok], [f[ok] for f in features]
    fold, res = folds(ok.sum()), []
    for f in range(NFOLD):
        tr, te = fold != f, fold == f

        def design(m):
            return np.column_stack([np.ones(m.sum())] + [x[m] for x in features])
        b, *_ = np.linalg.lstsq(design(tr), y[tr], rcond=None)
        res.append(y[te] - design(te) @ b)
    return float(np.std(np.concatenate(res)))


# --------------------------------------------------------------------------- #
# the efficiency law: log10 eps_i = a0 + a_M*dlm + a_z*ln(1+z) [+ a_Mz cross]  #
# --------------------------------------------------------------------------- #
def law_mass(d, par, sub=slice(None), window=False, cross=False):
    """Predicted log10 M*(<100) at the 5 epochs. Deposits, no renormalization.

    ``window=True`` swaps the monotone ``a_z ln(1+z)`` term for a free
    lognormal in ln(1+z) -- the incumbent kernel's form -- to show it buys
    nothing for the amplitude (§3.2.F).
    """
    a0, a_m = par[0], par[1]
    log_eps = a0 + a_m * d["dlm"][sub]
    i = 2
    if window:
        mu, sig = par[i], max(abs(par[i + 1]), 1e-3)
        shape = np.exp(-((d["ln1z"][sub] - mu) ** 2) / (2.0 * sig ** 2))
        i += 2
    else:
        log_eps = log_eps + par[i] * d["ln1z"][sub]
        shape = 1.0
        i += 1
    if cross:
        log_eps = log_eps + par[i] * d["dlm"][sub] * d["ln1z"][sub]
    w = 10.0 ** log_eps * shape * d["dmh"][sub]
    s = np.einsum("nkl,nl->nk", d["mask"][sub], w)
    return np.log10(np.clip(s, 1e-30, None))


def fit_law(d, p0, **kw):
    def rms(par, sub=slice(None)):
        return float(np.sqrt(np.nanmean((d["y"][sub] - law_mass(d, par, sub, **kw)) ** 2)))
    r = minimize(rms, p0, method="Nelder-Mead",
                 options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12))
    fold, res = folds(d["n"]), np.full_like(d["y"], np.nan)
    for f in range(NFOLD):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        rf = minimize(lambda q: rms(q, tr), r.x, method="Nelder-Mead",
                      options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12))
        res[te] = d["y"][te] - law_mass(d, rf.x, te, **kw)
    return r, res


def _row(label, vals, fmt="{:>9.4f}"):
    print(f"  {label:<34}" + "".join(fmt.format(v) for v in vals))


def main():
    d = load()
    n = d["n"]
    print(f"exp54 planning probe — n={n}, target log10 M*(<{R_ANCHOR:.0f} kpc)\n")

    print("A. the anchor aperture is statistically free "
          "(logMh(z_k) alone, 5-fold CV)")
    _row("epoch", Z, "{:>9}")
    for nm, y in (("M*(<100)", d["y"]), ("M*(<148)", d["y148"]),
                  ("M*(<500) [extrapolated]", d["y500"])):
        _row(nm, [cv_linear(y[:, k], [d["lmh_k"][:, k]]) for k in range(5)])
    _row("M(<100)/M(<148) median", np.median(d["m100_over_m148"], axis=0))

    print("\nB. epoch-local vs descendant conditioning")
    _row("epoch", Z, "{:>9}")
    _row("do-nothing (population sigma)", [np.std(d["y"][:, k]) for k in range(5)])
    _row("descendant logMh(z=0.4)",
         [cv_linear(d["y"][:, k], [d["lmh_desc"]]) for k in range(5)])
    _row("epoch-matched logMh(z_k)",
         [cv_linear(d["y"][:, k], [d["lmh_k"][:, k]]) for k in range(5)])
    _row("+ c200c + fz2",
         [cv_linear(d["y"][:, k], [d["lmh_k"][:, k], d["c200c"], d["fz2"]])
          for k in range(5)])

    print("\nC. the SHAPE-fitted window is a bad amplitude model")
    th = np.load(ADOPTED)["theta_z15"][:12]
    mu_a, sig_a = float(th[3]), float(th[4])
    par = [0.0, 0.0, mu_a, sig_a]
    S = law_mass(d, par, window=True)                  # a0=a_M=0 -> pure window
    print(f"  adopted window: mu={mu_a:.4f}  sig={sig_a:.4f}")
    _row("epoch", Z, "{:>9}")
    _row("log S, slope-one",
         [np.std(d["y"][:, k] - S[:, k] - np.median(d["y"][:, k] - S[:, k]))
          for k in range(5)])
    _row("log S, free slope",
         [cv_linear(d["y"][:, k], [S[:, k]]) for k in range(5)])
    _row("   ... the fitted slope",
         [np.polyfit(S[:, k], d["y"][:, k], 1)[0] for k in range(5)], "{:>9.3f}")

    print("\nD/E/F. the absolute efficiency law")
    base = [-2.5, -0.19, 0.36]
    variants = [("3p  a0 + a_M dlogMh + a_z ln(1+z)", base, {}),
                ("4p  ... free lognormal window", [-1.34, -0.19, 5.18, 2.19],
                 dict(window=True)),
                ("4p  3p + mass x redshift cross", base + [0.0], dict(cross=True))]
    fitted = {}
    for nm, p0, kw in variants:
        r, res = fit_law(d, p0, **kw)
        fitted[nm] = (r, res)
        print(f"\n  {nm}")
        print(f"    par {np.round(r.x, 4)}   in-sample pooled {r.fun:.4f}   "
              f"CV pooled {np.sqrt(np.nanmean(res ** 2)):.4f}")
        _row("  epoch", Z, "{:>9}")
        _row("  CV sigma", [np.std(res[:, k]) for k in range(5)])
        _row("  CV bias", [np.mean(res[:, k]) for k in range(5)], "{:>+9.4f}")
        _row("  CV resid vs logMh(z_k) slope",
             [np.polyfit(d["lmh_k"][:, k], res[:, k], 1)[0] for k in range(5)],
             "{:>9.3f}")
        _row("  CV resid vs c200c corr",
             [np.corrcoef(d["c200c"], res[:, k])[0, 1] for k in range(5)],
             "{:>9.3f}")

    print("\n  shuffled-MAH control on the 3p law: whole MAHs permuted within")
    print("  60 narrow final-halo-mass bins. NOTE this also swaps Mh(z_k), so at")
    print("  high z it partly re-measures 'epoch-local halo mass matters'.")
    rng = np.random.default_rng(0)
    perm = np.arange(n)
    for grp in np.array_split(np.argsort(d["lmh_k"][:, 0]), 60):
        perm[grp] = rng.permutation(grp)
    sh = dict(d, dmh=d["dmh"][perm], dlm=d["dlm"][perm],
              ln1z=d["ln1z"][perm], mask=d["mask"][perm])
    r3 = fitted[variants[0][0]][0]
    ms = law_mass(sh, r3.x)
    _row("epoch", Z, "{:>9}")
    _row("shuffled-MAH sigma",
         [np.std(d["y"][:, k] - ms[:, k] - np.median(d["y"][:, k] - ms[:, k]))
          for k in range(5)])
    _row("real MAH sigma (3p, CV)",
         [np.std(fitted[variants[0][0]][1][:, k]) for k in range(5)])


if __name__ == "__main__":
    main()
