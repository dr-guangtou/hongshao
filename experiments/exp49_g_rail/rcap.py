"""exp49 — does the model care about the Mpc-scale deposit radii at all?

The size law `rc(t_i) = 10**log_rc * (t_i/t_obs)**g` is an unbounded power law.
At the widened optimum (g = 4.35) the largest deposit scale is 938 kpc, and at
forced g = 5.5 it reaches 4464 kpc -- at or beyond the host halo's virial
radius, which no physical deposition process should produce.

Cheap diagnostic before proposing any model change: CAP the deposit scale at a
physical radius and see whether the fit notices.

  - If the loss barely moves, the Mpc tail is unconstrained slack. Capping it
    costs nothing and removes a class of nonsense.
  - If the loss degrades materially, the model is genuinely using those radii
    and a ceiling is a real model change needing its own evaluation.

Two measurements per cap: the loss at the UNCHANGED optimum (immediate impact)
and after a full refit (whether the model recovers).

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp49_g_rail/rcap.py [--dev]
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import stage2_multiepoch as s2                      # noqa: E402
from hongshao.fitting import minimize_loss          # noqa: E402
from hongshao.objective import Objective, reduce_galaxies   # noqa: E402

OUTDIR = HERE / "outputs"
KS = [0, 1, 2, 3]
CAPS = (300.0, 1000.0)
FAIL_LOSS = 4.0
BOX_LO = [1.0, 0.0, 0.0, 0.0, 0.05, 1.05] + [None] * 6
BOX_HI = [3.5, 6.0, 3.0, 3.0, 2.00, 6.00] + [None] * 6
BOUNDS = list(zip(BOX_LO, BOX_HI))

_ORIGINAL_BASIS = s2.basis_mof


def capped_basis(cap):
    """`basis_mof` with the deposit scale ceilinged at `cap` kpc.

    Only the CEILING changes; the floor, the migration term and everything
    else are the incumbent's. `model_cogs` resolves `basis_mof` as a module
    global, so replacing the attribute is enough.
    """
    def basis(th4, gam, ti, t_obs, tk, r):
        rc0 = np.clip(10.0 ** th4[0] * (ti / t_obs) ** th4[1], 1e-4, cap)
        dt = np.clip(tk - ti, 0.0, None)
        fc = np.exp(-dt / (10.0 ** th4[2] * ti))
        rcw = np.clip(rc0 * (tk / ti) ** max(th4[3], 0.0), 1e-4, cap)

        def cog(rc):
            u = 1.0 + (r[:, None] / rc[None, :]) ** 2
            return 1.0 - u ** (1.0 - gam)
        return fc[None, :] * cog(rc0) + (1.0 - fc)[None, :] * cog(rcw)
    return basis


def main(dev=False):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R
    obj = Objective()
    data = np.stack([np.array([g["data"][k] for k in KS], float) for g in gals])

    def loss(theta):
        cogs = np.full((len(gals), len(KS), len(R)), np.nan)
        for i, g in enumerate(gals):
            c = s2.model_cogs(theta, g, KS, "1ch-mof")
            if c is not None:
                cogs[i] = np.asarray(c)
        v = obj.per_galaxy(cogs, data, R)
        return reduce_galaxies(np.where(np.isfinite(v), v, FAIL_LOSS),
                               obj.galaxy)

    fit = np.load(OUTDIR / "widened_fit_production.npz", allow_pickle=True)
    keys = [k.split("::", 1)[1] for k in fit if k.startswith("theta::")]
    best = keys[int(np.argmin([fit[f"loss::{k}"][0] for k in keys]))]
    theta = np.asarray(fit[f"theta::{best}"], float)

    s2.basis_mof = _ORIGINAL_BASIS
    l0 = loss(theta)
    print(f"exp49 deposit-radius cap test  (n={len(gals)})")
    print(f"  uncapped optimum: loss {l0:.8f}, g={theta[1]:.4f}, "
          f"log_rc={theta[0]:.4f}  (largest rc = {10**theta[0]:.0f} kpc)\n")
    print(f"  {'cap':>8}{'loss at same theta':>21}{'delta':>12}"
          f"{'loss refitted':>16}{'delta':>12}{'g refit':>10}")
    out = {}
    for cap in CAPS:
        s2.basis_mof = capped_basis(cap)
        l_same = loss(theta)
        t0 = time.time()
        r = minimize_loss(loss, theta, method="lbfgsb", bounds=BOUNDS,
                          max_evals=4000)
        out[cap] = (l_same, float(r.fun), r.x)
        print(f"  {cap:7.0f}k{l_same:21.8f}{l_same-l0:12.2e}"
              f"{r.fun:16.8f}{r.fun-l0:12.2e}{r.x[1]:10.4f}"
              f"   ({(time.time()-t0)/60:.1f} min, conv={r.success})",
              flush=True)
    s2.basis_mof = _ORIGINAL_BASIS
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"rcap{'_dev' if dev else ''}.npz",
             loss_uncapped=np.array([l0]), theta_uncapped=theta,
             **{f"loss_same::{c:g}": np.array([out[c][0]]) for c in CAPS},
             **{f"loss_refit::{c:g}": np.array([out[c][1]]) for c in CAPS},
             **{f"theta_refit::{c:g}": out[c][2] for c in CAPS})
    print(f"\n  wrote {OUTDIR / 'rcap.npz'}")
    print("\n  Reading: if 'loss refitted' is within ~1e-5 of the uncapped "
          "loss, the\n  Mpc-scale tail is unconstrained slack and a physical "
          "ceiling is free.")


if __name__ == "__main__":
    main("--dev" in sys.argv)
