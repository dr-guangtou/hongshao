"""exp33 — consolidate the single-epoch (z=0.4) prediction under the standard QA.

The graduated stack (hongshao.emulator + profile_emulator, frozen spec from
exp14-22) has only ever been scored with CRPS/NLL/1-D coverage. This runs it
through the exp31/32 standardized QA, including the test it has never faced:
is the GENERATIVE claim true in 2-D — do sample() draws reproduce the
observational planes (energy/floor ~ 1), where the mean prediction cannot?

Modes (5-fold out-of-fold, portable features X = [DiffMAH(4), c200c]):
  1  kpc aperture/annulus masses  [<10, 10-30, 30-50, 50-100]      (exp20)
  2  Re-bin masses                [<0.5, 0.5-1, 1-2, 2-4, 4-6, 6-9] (exp21)
  3  the cumulative CoG (PCA-3 compressed)                          (exp22)
  (4 density deferred: its CoG integration needs a predicted central mass —
   a protocol decision, not a rerun.)

Run: PYTHONPATH=. uv run python experiments/exp33_single_epoch/run.py
Demo: ... run.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from astropy.table import Table                                                      # noqa: E402
from hongshao import qa                                                              # noqa: E402
from hongshao.emulator import fit as emu_fit                                         # noqa: E402
from hongshao.profile_emulator import re_targets, fit_profile                        # noqa: E402
from hongshao.metrics import crps_gaussian                                           # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402

FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = COG_RAD_KPC
KPC_NAMES = ["M(<10)", "M(10-30)", "M(30-50)", "M(50-100)"]
RE_EDGES = [0.5, 1, 2, 4, 6, 9]
N_DRAW = 10                                     # generative populations per test
KFOLD, SEED = 5, 0


def annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


def load_sample():
    t = Table.read(TABLE)
    t = t[t["use"]]
    cog = np.asarray(t["logmstar_cog"], float)
    aper = np.asarray(t["logmstar_aper"], float)
    X = np.column_stack([np.asarray(t[c], float) for c in
                         ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late",
                          "c_200c")])
    g = np.isfinite(cog).all(1) & np.isfinite(X).all(1) & np.isfinite(aper).all(1)
    return X[g], cog[g], aper[g]


def folds_of(n):
    order = np.random.default_rng(SEED).permutation(n)
    return np.array_split(order, KFOLD)


def cv_mass(X, Y):
    """OOF (mu, sigma) + N_DRAW OOF sampled populations for a mass-target mode."""
    n, T = Y.shape
    mu = np.empty((n, T))
    sig = np.empty((n, T))
    draws = np.empty((N_DRAW, n, T))
    for fi, fold in enumerate(folds_of(n)):
        tr = np.setdiff1d(np.arange(n), fold)
        emu = emu_fit(X[tr], Y[tr])
        m, s, _ = emu.predict(X[fold])
        mu[fold], sig[fold] = m, s
        draws[:, fold] = emu.sample(X[fold], size=N_DRAW, rng=100 + fi)
    return mu, sig, draws


def cv_cog(X, cog):
    """OOF per-radius (mu, sigma) + N_DRAW sampled CoG populations (n, 24)."""
    n = len(cog)
    prof, anchor = cog[:, :-1], cog[:, -1]
    mu = np.empty((n, 24))
    sig = np.empty((n, 24))
    draws = np.empty((N_DRAW, n, 24))
    for fi, fold in enumerate(folds_of(n)):
        tr = np.setdiff1d(np.arange(n), fold)
        pe = fit_profile(X[tr], prof[tr], anchor[tr], R[:-1])
        mp, sp = pe.predict(X[fold])
        sc_mu, sc_sig, _ = pe.emu.predict(X[fold])
        mu[fold] = np.column_stack([mp, sc_mu[:, 0]])
        sig[fold] = np.column_stack([sp, sc_sig[:, 0]])
        dp = pe.sample(X[fold], size=N_DRAW, rng=200 + fi)          # (S, f, 23)
        da = pe.emu.sample(X[fold], size=N_DRAW, rng=300 + fi)[:, :, 0]
        draws[:, fold] = np.concatenate([dp, da[:, :, None]], axis=2)
    return mu, sig, draws


def plane_report(name, pair_names, truth_log, mean_log, draw_log):
    """Plane fidelity for a mass mode: truth vs mean prediction vs draws.
    All inputs log10 masses; draw_log (S, n, T). Prints full+centered ratios."""
    ix, iy = 0, len(pair_names) - 1
    T = np.column_stack([truth_log[:, ix], truth_log[:, iy]])
    pm = qa.plane_energy(T, np.column_stack([mean_log[:, ix], mean_log[:, iy]]))
    rd = [qa.plane_energy(T, np.column_stack([d[:, ix], d[:, iy]]))
          for d in draw_log]
    full = np.mean([r["energy_ratio"] for r in rd])
    cent = np.mean([r["energy_ratio_centered"] for r in rd])
    print(f"    [{name}] plane {pair_names[ix]} vs {pair_names[iy]} (energy/floor "
          "full | centered):")
    print(f"      mean prediction : {pm['energy_ratio']:5.1f} | "
          f"{pm['energy_ratio_centered']:5.1f}")
    print(f"      sample() draws  : {full:5.1f} | {cent:5.1f}   "
          f"(avg over {len(rd)} drawn populations)")
    return pm, full, cent


def mass_table(name, names, Y, mu, sig):
    crps = crps_gaussian(Y, mu, sig).mean()
    print(f"\n  == mode {name} (n={len(Y)}), CRPS {crps:.4f} ==")
    print(f"    {'target':>10s} |  bias%  | dex scatter")
    for j, nm in enumerate(names):
        re_ = 10.0 ** (mu[:, j] - Y[:, j]) - 1.0
        print(f"    {nm:>10s} | {100*np.median(re_):+6.1f}% | "
              f"{np.std(mu[:, j] - Y[:, j]):.3f}")
    return crps


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    X, cog, aper = load_sample()
    n = len(X)
    print(f"exp33 — single-epoch (z=0.4) consolidation, n={n}, "
          f"features [DiffMAH(4), c200c], {KFOLD}-fold OOF\n")

    # ---- mode 1: kpc masses ----
    Ykpc = np.column_stack([aper[:, 0], annulus(aper[:, 1], aper[:, 0]),
                            annulus(aper[:, 2], aper[:, 1]),
                            annulus(aper[:, 4], aper[:, 2])])
    mu1, sig1, dr1 = cv_mass(X, Ykpc)
    crps1 = mass_table("1 kpc-masses", KPC_NAMES, Ykpc, mu1, sig1)
    plane_report("mode 1", KPC_NAMES, Ykpc, mu1, dr1)

    # ---- mode 2: Re bins ----
    Yre, _, m2 = re_targets(cog, R, RE_EDGES)
    re_names = [f"M(<{RE_EDGES[0]}Re)"] + \
        [f"M({a}-{b}Re)" for a, b in zip(RE_EDGES[:-1], RE_EDGES[1:])]
    mu2, sig2, dr2 = cv_mass(X[m2], Yre[m2])
    mass_table(f"2 Re-masses (n={int(m2.sum())})", re_names, Yre[m2], mu2, sig2)
    plane_report("mode 2", re_names, Yre[m2], mu2, dr2)

    # ---- mode 3: the CoG, through the FULL standard QA ----
    mu3, sig3, dr3 = cv_cog(X, cog)
    print("\n  == mode 3 (CoG profile) — full standardized QA (mean prediction) ==")
    res = qa.evaluate(10.0 ** mu3[:, None, :], 10.0 ** cog[:, None, :], R, [0.4],
                      name="cog-mean", figdir=FIGDIR, verbose=True, figures=True,
                      bin_by=X[:, 0], bin_label="logMh (DiffMAH logmp)")
    print("\n  mode 3 generative planes (draws through the same quantity set):")
    for kx, ky in qa.PLANES:
        tr_ = np.column_stack([np.log10(np.clip(res["truth"][kx][:, 0], 1, None)),
                               np.log10(np.clip(res["truth"][ky][:, 0], 1, None))])
        pm = qa.plane_energy(tr_, np.column_stack(
            [np.log10(np.clip(res["model"][kx][:, 0], 1, None)),
             np.log10(np.clip(res["model"][ky][:, 0], 1, None))]))
        ratios = []
        for s in range(N_DRAW):
            t_, mdl, _, _ = qa.measure_all(10.0 ** dr3[s][:, None, :],
                                           10.0 ** cog[:, None, :], R)
            ratios.append(qa.plane_energy(tr_, np.column_stack(
                [np.log10(np.clip(mdl[kx][:, 0], 1, None)),
                 np.log10(np.clip(mdl[ky][:, 0], 1, None))])))
        f_ = np.mean([r["energy_ratio"] for r in ratios])
        c_ = np.mean([r["energy_ratio_centered"] for r in ratios])
        print(f"    {kx} vs {ky}: mean {pm['energy_ratio']:.1f}|"
              f"{pm['energy_ratio_centered']:.1f}  ->  draws {f_:.1f}|{c_:.1f}")

    np.savez(OUTDIR / "single_epoch.npz", X=X, cog=cog, Ykpc=Ykpc,
             mu1=mu1, sig1=sig1, draws1=dr1, mu3=mu3, sig3=sig3, draws3=dr3,
             m2=m2, Yre=Yre, mu2=mu2, sig2=sig2, draws2=dr2, crps1=crps1)
    print(f"\nwrote {OUTDIR / 'single_epoch.npz'}")


def demo():
    """Self-check: OOF machinery on a synthetic linear-Gaussian problem —
    the mean under-disperses the plane, the draws restore it (ratio near 1)."""
    rng = np.random.default_rng(3)
    n = 1500
    X = rng.normal(size=(n, 5))
    B = rng.normal(size=(2, 6)) * 0.15
    # noise dominates (like the real annuli): the conditional mean is strongly
    # under-dispersed, so the plane test must separate mean from draws
    Y = np.column_stack([np.ones(n), X]) @ B.T + 0.45 * rng.standard_normal((n, 2))
    Y += 10.0                                       # log-mass-like scale
    mu, sig, draws = cv_mass(X, Y)
    assert mu.shape == Y.shape and draws.shape == (N_DRAW, n, 2)
    pm = qa.plane_energy(Y, mu)
    pd = np.mean([qa.plane_energy(Y, d)["energy_ratio_centered"] for d in draws])
    assert pm["energy_ratio_centered"] > 2.0, pm    # mean: visibly under-dispersed
    assert pd < 2.0, pd                             # draws: near the floor
    print(f"run.demo OK: OOF mean centered-ratio {pm['energy_ratio_centered']:.1f} "
          f"(under-dispersed), draws {pd:.1f} (restored)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
