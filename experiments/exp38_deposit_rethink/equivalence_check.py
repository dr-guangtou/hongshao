"""ONE-OFF check gating the objective-module port. Not part of the experiment.

Asserts that ``stage2_multiepoch.gal_loss``, now routed through
``hongshao.objective.Objective()``, returns exactly what the previous inline
implementation returned — on the real sample, at every epoch, including
galaxies the kernel fails on. Also times both paths, because the whole-
population array call is a different shape of computation from the per-galaxy
loop and which is faster is a measurement, not a guess.

Delete after the port is committed; its output belongs in the commit message.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp38_deposit_rethink/equivalence_check.py [--dev]
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.objective import Objective            # noqa: E402
import stage2_multiepoch as s2                      # noqa: E402

POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
KS_ALL = [0, 1, 2, 3, 4]


def previous_gal_loss(p, g, ks, variant):
    """The implementation this port replaces, verbatim from git history."""
    cogs = s2.model_cogs(p, g, ks, variant)
    if cogs is None:
        return 4.0
    return float(np.mean([np.sqrt(np.mean(((c - g["data"][k]) / g["data"][k])
                                          ** 2))
                          for c, k in zip(cogs, ks)]))


def main(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R
    theta = np.asarray(np.load(E40)["theta_z15"], float)[:12]
    print(f"n={len(gals)} galaxies, {len(R)} radii, adopted 1ch-mof theta")

    # ---- 1. agreement, per galaxy, at every epoch and on the fitted scope --- #
    worst = 0.0
    n_failed = 0
    for ks in ([0], [1], [2], [3], [4], [0, 1, 2, 3], KS_ALL):
        a = np.array([previous_gal_loss(theta, g, ks, "1ch-mof") for g in gals])
        b = np.array([s2.gal_loss(theta, g, ks, "1ch-mof") for g in gals])
        assert np.isfinite(a).all() and np.isfinite(b).all()
        d = np.abs(a - b).max()
        worst = max(worst, d)
        fails = int((b == s2.FAIL_LOSS).sum())
        n_failed += fails
        print(f"  ks={str(ks):<15} max|old-new| = {d:.3e}   "
              f"mean loss {b.mean():.6f}   sentinel hits {fails}")
        assert d < 1e-12, (ks, d)

    # ---- 2. the sentinel path, forced ------------------------------------- #
    # A theta the kernel cannot use must give 4.0 from BOTH paths, not NaN and
    # not a silently dropped galaxy. mu = 12 pushes the star-formation-
    # efficiency peak far outside the sampled redshift range, so the deposited
    # mass underflows for some galaxies but not others -- a MIXED case, which
    # is the one that would expose a wrapper that mishandles only one branch.
    broken = theta.copy()
    broken[3] = 12.0
    a = np.array([previous_gal_loss(broken, g, KS_ALL, "1ch-mof") for g in gals])
    b = np.array([s2.gal_loss(broken, g, KS_ALL, "1ch-mof") for g in gals])
    forced = int((b == s2.FAIL_LOSS).sum())
    print(f"\n  forced-failure theta: {forced}/{len(gals)} galaxies hit the "
          f"sentinel, {len(gals) - forced} still score finitely, "
          f"max|old-new| = {np.abs(a - b).max():.3e}")
    assert 0 < forced < len(gals), f"need a MIXED failure case, got {forced}"
    assert np.abs(a - b).max() < 1e-12
    assert np.isfinite(b).all(), "the sentinel must replace NaN, never leak it"

    # ---- 3. the NaN-drop trap, demonstrated ------------------------------- #
    # Without the sentinel the module reports NaN and reduce_galaxies DROPS it,
    # so a fit that starts failing would see its loss improve. Show it.
    cogs = np.array([s2.model_cogs(theta, g, KS_ALL, "1ch-mof") for g in gals])
    data = np.array([[g["data"][k] for k in KS_ALL] for g in gals])
    obj = Objective()
    poisoned = cogs.copy()
    poisoned[:5] = np.nan
    rep = obj.report(poisoned, data, R)
    print(f"  NaN-drop trap: {rep['n_failed']}/{rep['n_total']} galaxies NaN, "
          f"population value {rep['value']:.6f} vs {obj(cogs, data, R):.6f} "
          f"intact -> the drop {'HIDES' if rep['value'] < obj(cogs, data, R) else 'raises'}"
          f" the failure; the sentinel is what prevents this")
    assert rep["n_failed"] == 5

    # ---- 4. timing: per-galaxy loop vs one whole-population call ----------- #
    t0 = time.time()
    for _ in range(3):
        [s2.gal_loss(theta, g, [0, 1, 2, 3], "1ch-mof") for g in gals]
    t_loop = (time.time() - t0) / 3

    t0 = time.time()
    for _ in range(3):
        [previous_gal_loss(theta, g, [0, 1, 2, 3], "1ch-mof") for g in gals]
    t_prev = (time.time() - t0) / 3

    # the array path, model profiles already built (scoring only)
    c4, d4 = cogs[:, :4], data[:, :4]
    t0 = time.time()
    for _ in range(3):
        obj(c4, d4, R)
    t_array = (time.time() - t0) / 3

    print(f"\n  timing, one full pass over {len(gals)} galaxies (z<=1.5):\n"
          f"    previous inline loop : {t_prev * 1e3:8.1f} ms\n"
          f"    ported loop          : {t_loop * 1e3:8.1f} ms "
          f"({t_loop / t_prev:.2f}x)\n"
          f"    scoring only, one array call (profiles prebuilt): "
          f"{t_array * 1e3:8.1f} ms")

    print(f"\nEQUIVALENCE OK — worst disagreement {worst:.3e} across all "
          f"epoch scopes")


if __name__ == "__main__":
    main("--dev" in sys.argv)
