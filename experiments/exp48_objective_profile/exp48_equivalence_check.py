"""ONE-OFF check gating the objective-module port, exp48 half. Not part of exp48.

The exp38 check verifies only the DEFAULT objective. This one verifies the
other five axes, by asserting that ``hongshao.objective.Objective`` reproduces
``exp48/shootout.py::score_cogs`` — the reference implementation that produced
exp48's measured results — across every configuration exp48 ran, including its
winner (surface density with a log residual) and the twelve-cell factorial.

That is what "fold the duplicate together" has to mean in practice: the library
must give the same numbers as the code whose results are on the record, or the
record no longer applies to it.

Delete after the port is committed; its output belongs in the commit message.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp48_objective_profile/exp48_equivalence_check.py [--dev]
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.objective import Objective            # noqa: E402
import objective as e48                             # noqa: E402  (exp48's copy)

POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
KS = [0, 1, 2, 3]


def main(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    e48._w_init(rows)
    gals, R = e48._W["gals"], e48._W["e"].R
    theta = np.asarray(np.load(E40)["theta_z15"], float)[:12]

    # Build the model profiles ONCE; both scorers then see identical inputs,
    # so any disagreement is the scoring and nothing else.
    keep, cogs, data = [], [], []
    for i, g in enumerate(gals):
        c = e48._W["s2"].model_cogs(theta, g, KS, "1ch-mof")
        if c is not None:
            keep.append(i)
            cogs.append(np.asarray(c))
            data.append(np.array([g["data"][k] for k in KS], float))
    cogs, data = np.array(cogs), np.array(data)
    print(f"n={len(keep)}/{len(gals)} galaxies with usable profiles, "
          f"{len(KS)} epochs (z<=1.5), {len(R)} radii\n")

    # every configuration exp48 ran, plus the full six-axis product as a sweep
    named = dict(e48.screen_configs())
    named.update(dict(e48.factorial_configs()))
    worst_gal, worst_pop, n = 0.0, 0.0, 0
    for name, cfg in sorted(named.items()):
        obj = Objective.from_dict(cfg)
        mine = obj.per_galaxy(cogs, data, R)
        theirs = np.array([_ref(e48, cogs[j], data[j], cfg)
                           for j in range(len(keep))])
        # RELATIVE tolerance: losses span 0.05 to ~14 across these
        # configurations (density x fractional diverges by design, which is
        # exp48's result), so a fixed absolute threshold would be far too
        # tight at the top of that range and meaninglessly loose at the
        # bottom. The residual disagreement is summation order -- einsum here
        # against a plain sum there -- not different arithmetic.
        dg = _rel(mine, theirs)
        dp = _rel(obj(cogs, data, R), e48.reduce_galaxies(theirs, cfg["galaxy"]))
        worst_gal, worst_pop, n = max(worst_gal, dg), max(worst_pop, dp), n + 1
        assert dg < 1e-12 and dp < 1e-12, (name, dg, dp)
        print(f"  {name:<44} per-galaxy {dg:.2e}  population {dp:.2e}  "
              f"loss {obj(cogs, data, R):.6f}")

    # and the exhaustive six-axis product, to catch a combination exp48 never
    # ran (the density-x-fractional interaction is why one-at-a-time screens
    # are not enough)
    axes = itertools.product(("cog", "density"), ("frac", "abs", "log"),
                             ("rms", "absmean", "max"),
                             ("uniform", "perdex", "inner"),
                             ("mean", "median", "cvar20"))
    for q, res, rad, w, gal in axes:
        cfg = dict(quantity=q, residual=res, radial=rad, weight=w, galaxy=gal)
        obj = Objective.from_dict(cfg)
        theirs = np.array([_ref(e48, cogs[j], data[j], cfg)
                           for j in range(len(keep))])
        d = _rel(obj.per_galaxy(cogs, data, R), theirs)
        worst_gal, n = max(worst_gal, d), n + 1
        assert d < 1e-12, (cfg, d)

    print(f"\nEQUIVALENCE OK — {n} configurations, worst per-galaxy "
          f"disagreement {worst_gal:.2e}, worst population {worst_pop:.2e}")


def _rel(a, b):
    """Worst relative disagreement, with a small absolute floor so a loss that
    is legitimately near zero does not blow the ratio up."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-3)))


def _ref(e48, cog, dat, cfg):
    """exp48's scorer, fed one galaxy's already-built profiles.

    Copied VERBATIM from ``exp48/shootout.py::score_cogs`` -- the reference
    implementation whose numbers are on the record. exp48's ``objective.py``
    version cannot be called directly here because it rebuilds the model
    internally, and the point is to compare scoring on identical inputs.
    """
    w = e48._W["wts"][cfg["quantity"]][cfg["weight"]]
    per_epoch = []
    for c, d in zip(cog, dat):
        if cfg["quantity"] == "density":
            dfc = e48._W["density_from_cog"]
            lm = dfc(np.log10(np.clip(c, 1.0, None))[None], e48._W["e"].R)[0][0]
            ld = dfc(np.log10(np.clip(d, 1.0, None))[None], e48._W["e"].R)[0][0]
            m_v, d_v, norm = 10.0 ** lm, 10.0 ** ld, 10.0 ** np.max(ld)
        else:
            m_v, d_v, norm = c, d, d[-1]
        if cfg["residual"] == "frac":
            r = (m_v - d_v) / np.clip(d_v, 1e-30, None)
        elif cfg["residual"] == "abs":
            r = (m_v - d_v) / max(norm, 1e-30)
        else:
            r = (np.log10(np.clip(m_v, 1e-30, None))
                 - np.log10(np.clip(d_v, 1e-30, None)))
        if not np.isfinite(r).all():
            return 4.0
        if cfg["radial"] == "rms":
            per_epoch.append(np.sqrt(np.sum(w * r ** 2)))
        elif cfg["radial"] == "absmean":
            per_epoch.append(np.sum(w * np.abs(r)))
        else:
            per_epoch.append(np.max(np.abs(r)))
    return float(np.mean(per_epoch))


if __name__ == "__main__":
    main("--dev" in sys.argv)
