"""exp57 Stage 0 — what does the model class change COST, and does it nest?

Two questions, both answered by measurement, before anything is fitted.

**1. THE COST, measured not estimated.** In exp54 a deposit's profile is fixed
the moment it is made, so the basis is built once and reused at all five
epochs. With expansion the profile depends on the epoch it is VIEWED at as
well, so the basis is built once per epoch. The technical note guessed this
"costs several times more per evaluation"; a guess is not a schedule. This
measures the real per-evaluation time, because every fitting decision after
this — how many starts, how many candidates, whether a profile likelihood is
affordable — follows from it.

The one structural saving is real and is used: `model.solve_u` depends on the
two radii only through their RATIO, which homologous scaling leaves unchanged,
so the expensive inversion is done once for all epochs and only the radius
lookup is repeated. `expand.cog_scaled` documents why that is exact.

**2. THE NESTING GATE, and an honest correction to how it is stated.** The
first draft of this file demanded that `ExpandingProblem` with the law switched
off reproduce `stage35_time_law.StackedProblem` **bit-for-bit**. It does not,
and it cannot: exp54 forms one matrix product over all five epochs at once,
while expansion forces one product per epoch, so the same sum is accumulated in
a different order. Floating-point addition is not associative, and the measured
disagreement is **1.7e-15 relative — about eight units in the last place.**

That is exactly the situation exp54's own `check_stacking` faced when it
stacked the per-galaxy loop, and this file adopts the same standard: the NaN
pattern must match **exactly** (a different set of galaxies failing is a real
difference, not rounding), the profiles to **1e-12 relative**, and the loss to
**1e-9 absolute**. Demanding bit-equality here would not have been rigour, it
would have been a wrong specification — so the specification is corrected in
the open rather than the tolerance quietly widened.

With the law ON at its nesting values (`A = 0`, `alpha = 0`) the same three
checks must pass, because that is the identity every later comparison rests on:
an enriched fit launched from the incumbent cannot come out worse unless the
optimiser failed.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage0_cost.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import fit as F                                          # noqa: E402
import model as M                                        # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage37_size_epoch as S37                         # noqa: E402

BASE = "gompertz_log-E2-S2"
LOSS_INCUMBENT = 1.874253
#: the nesting gate's tolerances. NOT free parameters: the profiles agree to
#: ~8 ulp and the loss to ~1e-15, so these sit three and six orders of
#: magnitude above what is measured, and a real refactor error moves them by
#: far more than that.
P_TOL, L_TOL = 1e-12, 1e-9
OUT = HERE / "outputs" / "stage0_cost.npz"
RULE = "=" * 96


def build(smoke=False):
    """The exp54 sample, the incumbent, and both problems on it."""
    recs, sel, data, mask, lmh, cuts = S37.build(smoke)
    base = S33.spec_from_label(BASE)
    theta = S37.incumbent()
    slow = S35.StackedProblem(base, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    return recs, data, mask, lmh, base, theta, slow


def main(smoke=False):
    print(f"{RULE}\nexp57 STAGE 0 — the cost of the model class change, and "
          f"the nesting gate\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = build(smoke)
    print(f"  sample   : {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs")
    l0 = slow.loss(theta)
    print(f"  incumbent: {BASE}, loss {l0:.6f}")
    if not smoke and abs(l0 - LOSS_INCUMBENT) > 1e-5:
        raise SystemExit(f"REFUSING TO RUN: exp54's own evaluator gives "
                         f"{l0:.6f}, not {LOSS_INCUMBENT:.6f}.")

    # ------------------------------------------------------------------ #
    print(f"\n  THE NESTING GATE — NaN pattern exact, profiles < {P_TOL:g} "
          f"relative, loss < {L_TOL:g}")
    print(f"    (not bit-for-bit: the per-epoch loop reassociates the sum, "
          f"see the module docstring)")
    pa = slow.predict(theta)

    def gate(name, pr, th):
        pc = pr.predict(th)
        same_nan = np.array_equal(np.isnan(pa), np.isnan(pc))
        both = ~np.isnan(pa)
        rel = float(np.max(np.abs(pa[both] - pc[both])
                           / np.maximum(np.abs(pa[both]), 1.0)))
        lx = pr.loss(th)
        dl = abs(lx - l0)
        ulp = rel / np.finfo(float).eps
        print(f"    {name:<12}: NaN pattern {'exact' if same_nan else 'DIFFERS'}"
              f", profiles {rel:.2e} rel ({ulp:.1f} ulp), loss {lx:.12f} "
              f"(d={dl:.1e})")
        if not same_nan or rel > P_TOL or dl > L_TOL:
            raise SystemExit(
                f"REFUSING TO RUN: {name} does not reproduce exp54's model "
                f"when switched off (NaN {same_nan}, {rel:.2e} relative, "
                f"loss off by {dl:.2e}). Fix the refactor, not the tolerance.")
        return pr

    gate('law=""', X.ExpandingProblem(X.XSpec(base, ""), recs, data, mask,
                                      epochs=(0, 1, 2, 3, 4)), theta)
    results = {}
    for law in ("X1", "X2", "X0", "X3"):
        sp = X.XSpec(base, law)
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        results[law] = gate(f"law={law}", pr, sp.nest(theta))
    print(f"    OK — every law reproduces the incumbent EXACTLY when switched "
          f"off, so no later\n    comparison can be an artefact of the "
          f"refactor.")

    # ------------------------------------------------------------------ #
    print(f"\n  THE COST, measured on {len(recs)} galaxies "
          f"(single-threaded, 10 evaluations each)")
    rng = np.random.default_rng(0)

    def timeit(pr, th, n=10):
        pr.loss(th)                                        # warm the caches
        t0 = time.time()
        for k in range(n):
            pr.loss(th * (1.0 + 1e-5 * k))
        return (time.time() - t0) / n

    t_base = timeit(slow, theta)
    print(f"    exp54 StackedProblem            {t_base * 1000:8.1f} ms")
    rows = [("exp54", t_base, 1.0)]
    for law in ("X1", "X2", "X0", "X3"):
        sp = X.XSpec(base, law)
        pr = results[law]
        # time it AWAY from the nesting point, where the arithmetic is real
        th = sp.nest(theta)
        th[base.n_theta] = 2.0 if law != "X2" else 2.0
        if law == "X2":
            th[base.n_theta + 1] = 1.5
        if law == "X0":
            th[base.n_theta] = 0.3
        t = timeit(pr, th)
        rows.append((law, t, t / t_base))
        print(f"    ExpandingProblem, law={law:<8}  {t * 1000:8.1f} ms   "
              f"{t / t_base:5.2f}x exp54")

    t_x1 = rows[1][1]
    print(f"\n  WHAT THAT BUYS AND COSTS. A Nelder-Mead fit of this model "
          f"converges in about\n  1500 iterations, roughly 2300 evaluations. "
          f"At {t_x1 * 1000:.0f} ms that is "
          f"{2300 * t_x1 / 60:.1f} min per start,\n  so a four-start fit is "
          f"about {4 * 2300 * t_x1 / 60:.0f} min and eight candidates run in "
          f"parallel on ten cores in\n  roughly "
          f"{4 * 2300 * t_x1 / 60:.0f} min of wall clock. A Stage 3.9-style "
          f"profile likelihood at 11 grid\n  points x 2 starts would be "
          f"{11 * 2 * 2300 * t_x1 / 3600:.1f} h per parameter.")

    if not smoke:
        OUT.parent.mkdir(exist_ok=True)
        np.savez(OUT, laws=np.array([r[0] for r in rows]),
                 seconds=np.array([r[1] for r in rows]),
                 ratio=np.array([r[2] for r in rows]),
                 n_galaxies=len(recs), loss=l0)
        print(f"\n  wrote {OUT.name}")


if __name__ == "__main__":
    main("--smoke" in sys.argv)
