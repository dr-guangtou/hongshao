"""exp33 step iii — single-epoch feature increments under the standard harness.

Does anything from the recent experiments earn a slot next to the portable
baseline [DiffMAH(4), c200c]? Candidates (all halo-only):
  burst       real-MAH burstiness (growth fraction in >10% steps; exp30) —
              never tested as a mass FEATURE
  t50, fz2    real-MAH formation summaries (exp29 lesson: the smooth DiffMAH
              curve can flatter; do the raw-MAH numbers carry extra signal?)
  acc_rate    late halo accretion rate (catalog)
  alt         REPLACE DiffMAH shape params with [logmh, t50, fz2] (is the
              smooth-fit parameterization itself losing information?)
Every candidate that helps is re-run with the feature SHUFFLED (the exp16
control): a real feature loses its gain, a spurious one keeps it.

Model/targets: the frozen emulator core on the mode-1 kpc masses
[<10, 10-30, 30-50, 50-100]; 5-fold OOF; scores = CRPS + per-target dex.

Run: PYTHONPATH=. uv run python experiments/exp33_single_epoch/features.py
Demo: ... features.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
from astropy.table import Table                                                      # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
from hongshao.emulator import fit as emu_fit                                         # noqa: E402
from hongshao.metrics import crps_gaussian                                           # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
BASE = ["dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late", "c_200c"]
KFOLD, SEED = 5, 0                                # same protocol as run.py


def annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


def folds_of(n):
    order = np.random.default_rng(SEED).permutation(n)
    return np.array_split(order, KFOLD)


def load_features():
    """Baseline features + candidates; restricted to galaxies with a real MAH."""
    t = Table.read(TABLE)
    t = t[t["use"]]
    aper = np.asarray(t["logmstar_aper"], float)
    Y = np.column_stack([aper[:, 0], annulus(aper[:, 1], aper[:, 0]),
                         annulus(aper[:, 2], aper[:, 1]),
                         annulus(aper[:, 4], aper[:, 2])])
    X0 = np.column_stack([np.asarray(t[c], float) for c in BASE])
    logmh = np.asarray(t["logmh_z0p4"], float)
    acc = np.asarray(t["acc_rate"], float)
    n = len(t)
    t50 = np.full(n, np.nan)
    fz2 = np.full(n, np.nan)
    burst = np.full(n, np.nan)
    for i, gi in enumerate(np.asarray(t["index"])):
        mah = real_mah(int(gi))
        if mah is None:
            continue
        Mh, tf = 10.0 ** mah["logMh_full"], mah["t_full"]
        t50[i] = np.interp(0.5 * Mh[-1], Mh, tf)
        fz2[i] = 10.0 ** (np.interp(33, mah["snap_full"], mah["logMh_full"])
                          - np.log10(Mh[-1]))
        s = mah["dMh"] / 10.0 ** mah["logMh_full"][1:]
        burst[i] = mah["dMh"][s > 0.10].sum() / mah["dMh"].sum()
    g = (np.isfinite(Y).all(1) & np.isfinite(X0).all(1) & np.isfinite(logmh)
         & np.isfinite(acc) & np.isfinite(t50) & np.isfinite(fz2)
         & np.isfinite(burst))
    return (Y[g], X0[g], dict(logmh=logmh[g], acc_rate=acc[g], t50=t50[g],
                              fz2=fz2[g], burst=burst[g]))


def cv_crps(X, Y):
    """OOF CRPS + per-target dex scatter for one feature set."""
    n = len(Y)
    mu = np.empty_like(Y)
    sig = np.empty_like(Y)
    for fold in folds_of(n):
        tr = np.setdiff1d(np.arange(n), fold)
        emu = emu_fit(X[tr], Y[tr])
        mu[fold], sig[fold], _ = emu.predict(X[fold])
    return float(crps_gaussian(Y, mu, sig).mean()), np.std(mu - Y, axis=0)


def main():
    rng = np.random.default_rng(7)
    Y, X0, cand = load_features()
    n = len(Y)
    print(f"exp33 step iii — feature increments (mode-1 kpc masses, n={n}, "
          "5-fold OOF)\n")
    sets = {
        "baseline [DiffMAH4+c200c]": X0,
        "+ burst": np.column_stack([X0, cand["burst"]]),
        "+ t50 + fz2": np.column_stack([X0, cand["t50"], cand["fz2"]]),
        "+ acc_rate": np.column_stack([X0, cand["acc_rate"]]),
        "+ all four": np.column_stack([X0, cand["burst"], cand["t50"],
                                       cand["fz2"], cand["acc_rate"]]),
        "alt [logmh,t50,fz2,c200c]": np.column_stack(
            [cand["logmh"], cand["t50"], cand["fz2"], X0[:, 4]]),
    }
    results = {}
    c0 = None
    for name, X in sets.items():
        crps, dex = cv_crps(X, Y)
        results[name] = crps
        if c0 is None:
            c0 = crps
        gain = 100 * (c0 - crps) / c0
        print(f"  {name:>28s}: CRPS {crps:.4f} ({gain:+5.1f}%)  dex "
              + " ".join(f"{d:.3f}" for d in dex))

    # shuffle controls for every candidate that improved by >0.5%
    print("\n  shuffle controls (a real feature loses its gain):")
    helped = [nm for nm, c in results.items()
              if nm.startswith("+") and 100 * (c0 - c) / c0 > 0.5]
    for nm in helped or ["+ burst"]:                    # always control burst
        cols = {"+ burst": ["burst"], "+ t50 + fz2": ["t50", "fz2"],
                "+ acc_rate": ["acc_rate"],
                "+ all four": ["burst", "t50", "fz2", "acc_rate"]}[nm]
        Xs = np.column_stack([X0] + [rng.permutation(cand[c]) for c in cols])
        crps, _ = cv_crps(Xs, Y)
        print(f"  {nm + ' (shuffled)':>28s}: CRPS {crps:.4f} "
              f"({100*(c0-crps)/c0:+5.1f}%)")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = list(results)
    gains = [100 * (c0 - results[nm]) / c0 for nm in names]
    ax.barh(range(len(names)), gains, color="#0072B2")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, c="0.4", lw=0.8)
    ax.set(xlabel="CRPS improvement over baseline [%]",
           title=f"exp33 — single-epoch feature increments (n={n})")
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp33_features")[0])


def demo():
    """Self-check: an informative synthetic feature improves OOF CRPS; its
    shuffle does not."""
    rng = np.random.default_rng(11)
    n = 1200
    X0 = rng.normal(size=(n, 3))
    extra = rng.normal(size=n)
    Y = (X0 @ np.array([[.4, .2, .1], [.1, .3, .2]]).T
         + 0.5 * extra[:, None] + 0.2 * rng.standard_normal((n, 2))) + 10.0
    c_base, _ = cv_crps(X0, Y)
    c_extra, _ = cv_crps(np.column_stack([X0, extra]), Y)
    c_shuf, _ = cv_crps(np.column_stack([X0, rng.permutation(extra)]), Y)
    assert c_extra < 0.8 * c_base, (c_base, c_extra)
    assert abs(c_shuf - c_base) < 0.05 * c_base, (c_base, c_shuf)
    print(f"features.demo OK: base {c_base:.3f} -> +feature {c_extra:.3f} "
          f"(shuffled {c_shuf:.3f} = base)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
