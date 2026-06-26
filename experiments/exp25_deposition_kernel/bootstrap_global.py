"""Bootstrap sampling uncertainties for the exp25 true-population model (global B).

The global fit (population_fit.py) gives point estimates only. Here we resample
the 2540 galaxies with replacement, refit the shared 6-parameter model B
(g, b_early, b_late, z_c + the log-linear sigma0(R50) relation) warm-started from
the full-sample optimum, and report the parameter scatter -> formal sampling
errors on the population parameters. (These are tight at n=2540; the larger real
uncertainty is the g/b_early/b_late basin degeneracy, which a warm-started
bootstrap does NOT capture -- see the README Phase-3 caveat.)

Run: PYTHONPATH=. uv run python experiments/exp25_deposition_kernel/bootstrap_global.py [N_BOOT]
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy.table import Table
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import experiments.exp25_deposition_kernel.population_fit as pf   # noqa: E402

MODES = dict(sigma="R50", zc="const")
NAMES = ["g", "b_early", "b_late", "sigma0_a", "sigma0_b", "z_c"]
_W = {}   # per-worker globals (spawn re-imports, so build inside the worker)


def _loss_on(P, theta):
    if not (0.0 < theta[0] < 5.0 and -5.0 < theta[1] < 30.0 and -5.0 < theta[2] < 30.0):
        return 1e3
    g_exp, be, bl, S0, ZC = pf._unpack_global(theta, MODES, P)
    rms, exc = pf.pop_forward(g_exp, be, bl, S0, ZC, P)
    v = float(rms.mean() + pf.PEN_W * exc.mean())
    return v if np.isfinite(v) else 1e3


def _init(theta0):
    inputs = pf.build_inputs(Table.read(pf.TABLE)[Table.read(pf.TABLE)["use"]])
    _W["P"] = pf.pad_inputs(inputs)
    _W["G"] = len(inputs)
    _W["theta0"] = np.asarray(theta0, float)


def _boot(seed):
    G, P0 = _W["G"], _W["P"]
    idx = np.random.default_rng(seed).integers(0, G, G)
    P = {k: (v[idx] if getattr(v, "shape", (0,))[:1] == (G,) else v) for k, v in P0.items()}
    r = minimize(lambda th: _loss_on(P, th), _W["theta0"], method="Nelder-Mead",
                 options=dict(maxiter=3000, xatol=1e-5, fatol=1e-7))
    return r.x.tolist()


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    inputs = pf.build_inputs(Table.read(pf.TABLE)[Table.read(pf.TABLE)["use"]])
    P = pf.pad_inputs(inputs)
    full = Table.read(pf.OUTDIR / "population_full.fits")
    med = {p: float(np.median(full[p])) for p in ("sigma0", "g", "b_early", "b_late", "z_c")}
    mean_lr50 = float(P["logr50"].mean())
    gB = pf.fit_global(MODES, P, [med["g"], med["b_early"], med["b_late"],
                                  np.log10(med["sigma0"]) - mean_lr50, 1.0, med["z_c"]])
    theta0 = np.asarray(gB["theta"], float)
    print(f"full-sample model-B optimum (median RMS {np.median(gB['rms']):.4f} dex):")
    print("  " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(NAMES, theta0)))

    with ProcessPoolExecutor(initializer=_init, initargs=(theta0,)) as ex:
        boots = np.array(list(ex.map(_boot, range(n_boot))))

    print(f"\nbootstrap sampling uncertainties (N={n_boot}):")
    for i, name in enumerate(NAMES):
        print(f"  {name:9s} {theta0[i]:+7.3f} +/- {boots[:, i].std():.3f}")
    # derived: sigma0 (kpc) at a reference R50=10 kpc (logR50=1)
    s0_ref = 10.0 ** (boots[:, 3] + boots[:, 4] * 1.0)
    print(f"  sigma0(R50=10kpc) = {10**(theta0[3]+theta0[4]):.1f} +/- {s0_ref.std():.1f} kpc")
    print(f"  z_c               = {theta0[5]:.2f} +/- {boots[:,5].std():.2f}")


if __name__ == "__main__":
    main()
