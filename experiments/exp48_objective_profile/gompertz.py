"""exp48 side quest — can a GOMPERTZ function describe a galaxy's CoG?

Motivation (user, 2026-08-14): arXiv:2604.13423 introduces a Gompertzian
template for cosmic reionization, replacing the symmetric tanh sigmoid
because "its earlier stages occur at a slower pace relative to its later
stages". That paper is about the ionization fraction versus redshift, NOT
about galaxy profiles — so this is a transfer of the functional form to a
new context, and it has to earn its place empirically.

A curve of growth is a monotone rise from 0 to the total mass, i.e. a
sigmoid in log radius, so the analogy is natural. The Gompertz sigmoid is

    F(x) = exp(-b exp(-c x))

asymmetric, unlike tanh or the logistic. Two readings of the radius
variable are tested, because the choice is not obvious:

  gompertz_log   x = ln R  ->  M(<R)/Mtot = exp(-b R^-c)
                 large R: 1 - M ~ b R^-c, so Sigma ~ R^-(c+2), a genuine
                 POWER-LAW tail (the property that made Moffat win exp38)
                 small R: M -> 0 super-exponentially -> a central HOLE

  gompertz_lin   x = R     ->  M(<R)/Mtot = exp(-b e^-cR)
                 an exponential-ish outer cutoff, no power-law tail

Both are 2-parameter shapes, so they are compared against two 2-parameter
incumbents on equal terms: Moffat (rc, gamma) and Sersic (R50, n).

Everything is fitted 148-pinned — model and data are each divided by
their own value at the outermost measured radius — which is the project's
standard shape convention and removes the amplitude from the comparison.

Run: PYTHONPATH=. uv run python experiments/exp48_objective_profile/\
gompertz.py {demo|run} [--n N]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammainc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
Z_GRID = (0.4, 0.7, 1.0, 1.5, 2.0)
RMIN = 5.0                      # the project's trusted-radius floor


def cog_model(family, p, R):
    """Unit-ish CoG shape on R; normalization is handled by pinning."""
    if family == "gompertz_log":
        b, c = np.exp(p[0]), np.exp(p[1])
        return np.exp(-b * R ** (-c))
    if family == "gompertz_lin":
        b, c = np.exp(p[0]), np.exp(p[1])
        return np.exp(-b * np.exp(-c * R))
    if family == "loglogistic":
        # symmetric-in-log sigmoid: POWER LAW AT BOTH ENDS.
        # M = X^k/(1+X^k) -> Sigma ~ R^(k-2) inner, R^(-k-2) outer, so
        # k < 2 gives a genuine central CUSP, unlike Gompertz's hole.
        lam, k = np.exp(p[0]), np.exp(p[1])
        X = (R / lam) ** k
        return X / (1.0 + X)
    if family == "richards":
        # generalized logistic in log R, which NESTS both of the above:
        #   nu -> 0  ->  exp(-X^-k)      = gompertz_log (Frechet)
        #   nu  = 1  ->  1/(1 + X^-k)    = loglogistic
        # so nu is a measured asymmetry rather than an assumed one.
        lam, k, nu = np.exp(p[0]), np.exp(p[1]), np.exp(p[2])
        # clip the exponent, not the result: (R/lam)^-k overflows for
        # small R with large k, and inf*nu then poisons the power
        X = np.exp(np.clip(-k * np.log(R / lam), -700.0, 700.0))
        return (1.0 + nu * X) ** (-1.0 / nu)
    if family == "moffat":
        rc, gam = np.exp(p[0]), 1.02 + np.exp(p[1])
        return 1.0 - (1.0 + (R / rc) ** 2) ** (1.0 - gam)
    if family == "sersic":
        from profiles import b_n
        r50, n = np.exp(p[0]), np.clip(np.exp(p[1]), 0.3, 10.0)
        return gammainc(2.0 * n, (R * b_n(n) ** n / r50) ** (1.0 / n))
    raise ValueError(family)


def fit_one(family, R, y, p0):
    """148-pinned least squares on the normalized CoG."""
    def resid(p):
        m = cog_model(family, p, R)
        if not np.isfinite(m).all() or m[-1] <= 0:
            return np.full(len(R), 10.0)
        return m / m[-1] - y

    try:
        r = least_squares(resid, p0, method="lm", max_nfev=4000)
        return r.x
    except Exception:
        return np.asarray(p0, float)


def maxrel(family, p, R, y, rmin=RMIN):
    m = cog_model(family, p, R)
    if not np.isfinite(m).all() or m[-1] <= 0:
        return np.nan
    rel = np.abs(m / m[-1] - y) / np.clip(y, 1e-6, None)
    sel = R >= rmin
    return float(np.max(rel[sel]))


FAMILIES = ("gompertz_log", "loglogistic", "richards", "moffat", "sersic",
            "gompertz_lin")
NPAR = {"gompertz_log": 2, "loglogistic": 2, "richards": 3, "moffat": 2,
        "sersic": 2, "gompertz_lin": 2}
START = {"gompertz_log": [np.log(3.0), np.log(1.0)],
         "gompertz_lin": [np.log(3.0), np.log(0.1)],
         "loglogistic": [np.log(10.0), np.log(1.2)],
         "richards": [np.log(10.0), np.log(1.2), np.log(0.5)],
         "moffat": [np.log(10.0), np.log(0.4)],
         "sersic": [np.log(10.0), np.log(2.0)]}


def cmd_run(nmax=600):
    from hongshao import qa
    from hongshao.tng_data import COG_RAD_KPC as R

    pop = np.load(POP_NPZ)
    data = pop["data"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data), size=min(nmax, len(data)), replace=False)
    r50 = qa._safe_rhalf(data[idx], R, 0.5)

    print(f"exp48 side quest — Gompertz vs the incumbents on REAL CoGs "
          f"(n={len(idx)} galaxies, 148-pinned, max|rel| over R>{RMIN:g} kpc)")
    out = {}
    for j, z in enumerate(Z_GRID):
        y = data[idx, j, :] / data[idx, j, -1:]
        comp = r50[:, j] < 5.0
        for fam in FAMILIES:
            errs = np.array([maxrel(fam, fit_one(fam, R, y[i], START[fam]),
                                    R, y[i]) for i in range(len(idx))])
            out[(fam, j)] = errs
        print(f"\n  z={z}  (compact {int(comp.sum())}/{len(idx)})")
        print(f"    {'family':<14}{'np':>2}{'median':>9}{'90th pct':>10}"
              f"{'compact med':>13}{'extended med':>14}")
        for fam in FAMILIES:
            e = out[(fam, j)]
            mc = np.nanmedian(e[comp]) if comp.sum() else np.nan
            me = np.nanmedian(e[~comp]) if (~comp).sum() else np.nan
            print(f"    {fam:<14}{NPAR[fam]}p{100*np.nanmedian(e):8.2f}%"
                  f"{100*np.nanpercentile(e, 90):9.2f}%"
                  f"{100*mc:12.2f}%{100*me:13.2f}%")
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "gompertz.npz",
             **{f"{f}_{j}": out[(f, j)] for f in FAMILIES
                for j in range(len(Z_GRID))}, idx=idx, r50=r50)
    print(f"\n  wrote {OUTDIR / 'gompertz.npz'}")
    return out, idx, r50, R, data


def demo():
    R = np.geomspace(2.0, 148.0, 24)
    # every family must be a valid CoG: monotone, positive, -> its max
    for fam in FAMILIES:
        m = cog_model(fam, START[fam], R)
        assert np.all(np.diff(m) >= -1e-12), f"{fam} not monotone"
        assert m.min() >= 0.0, fam
    # the recovery test: fit each family to ITSELF and get ~0 error
    for fam in FAMILIES:
        y = cog_model(fam, START[fam], R)
        y = y / y[-1]
        p = fit_one(fam, R, y, [x + 0.3 for x in START[fam]])
        e = maxrel(fam, p, R, y)
        assert e < 1e-3, f"{fam} cannot recover itself: {e:.2e}"
    # richards NESTS the two 2-parameter sigmoids; this is the property
    # that makes it worth a third parameter rather than a separate family
    Rn = np.geomspace(1.0, 100.0, 40)
    gl = cog_model("gompertz_log", [np.log(10.0 ** 1.2), np.log(1.2)], Rn)
    rn = cog_model("richards", [np.log(10.0), np.log(1.2), np.log(1e-6)], Rn)
    assert np.abs(gl - rn).max() < 1e-4, np.abs(gl - rn).max()
    ll = cog_model("loglogistic", [np.log(10.0), np.log(1.2)], Rn)
    r1 = cog_model("richards", [np.log(10.0), np.log(1.2), np.log(1.0)], Rn)
    assert np.abs(ll - r1).max() < 1e-12, np.abs(ll - r1).max()
    # loglogistic has a power-law CORE (a cusp for k<2), unlike Gompertz
    Rc = np.array([0.01, 0.02])
    lc = cog_model("loglogistic", [np.log(10.0), np.log(1.2)], Rc)
    sl = np.diff(np.log(lc))[0] / np.diff(np.log(Rc))[0]
    assert abs(sl - 1.2) < 0.01, f"loglogistic inner CoG slope {sl:.3f} != k"

    # gompertz_log must have a POWER-LAW tail: 1-M ~ b R^-c
    b, c = 3.0, 1.4
    Rt = np.array([300.0, 3000.0])
    miss = 1.0 - np.exp(-b * Rt ** (-c))
    slope = np.diff(np.log(miss))[0] / np.diff(np.log(Rt))[0]
    assert abs(slope + c) < 0.02, f"tail slope {slope:+.3f} != {-c}"
    # ... and a central HOLE, not a cusp: M falls faster than any power law
    Rs = np.array([0.5, 0.7])   # not smaller: exp(-b R^-c) underflows to 0
    inner = np.exp(-b * Rs ** (-c))
    sl_in = np.diff(np.log(inner))[0] / np.diff(np.log(Rs))[0]
    assert sl_in > 5.0 and np.isfinite(sl_in), \
        f"inner log-slope {sl_in:.1f} should far exceed a cusp's 2-3"
    print("demo OK — all four families are monotone CoGs and recover "
          f"themselves to <1e-3; gompertz_log has a power-law tail "
          f"(slope {slope:+.2f} = -c) and a central HOLE (inner log-slope "
          f"{sl_in:.0f}, vs 2-3 for a cusp)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 600
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(n)
    else:
        raise SystemExit(__doc__)
