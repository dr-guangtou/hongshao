"""Fitting helpers: which optimizer to point at a profile-model loss, and how.

Two functions. ``minimize_loss`` is the entry point; ``initial_simplex`` is the
Nelder-Mead repair it applies automatically and which is also useful alone.

**Why not just call scipy directly.** Measured on the real 1ch-mof loss (100
galaxies, four epochs at z <= 1.5, started from a displaced point with the six
conditioning slopes zeroed, 4000-evaluation budget):

    method                      final loss   evals   time
    L-BFGS-B (finite-diff)        0.152764    1170   38.4 s
    BFGS (finite-diff)            0.152764    1194   39.1 s
    trust-constr (finite-diff)    0.152764    1625   52.5 s
    Nelder-Mead, adaptive=True    0.152764    2259   73.5 s
    Powell                        0.152855    4000  129.7 s
    Nelder-Mead + initial_simplex 0.155685    3531  114.5 s
    Nelder-Mead, scipy defaults   0.159980    4000  131.2 s

Four methods agree on 0.152764 to six decimals, which is good evidence that is
a real local minimum rather than one method's quirk. The plain Nelder-Mead the
project has been using is the WORST of the field: it exhausts its budget 4.7%
above that minimum, while L-BFGS-B gets there in a third of the time despite
paying 13 evaluations per finite-difference gradient.

**This works because the loss is smooth**, which was measured rather than
assumed. At the adopted solution none of the kernel's non-differentiable
guards are active -- 0 of 9900 deposit events at a ``clip`` bound on the
deposit scale, 0 of 39600 event-epochs at the migrated-scale bound, 0 of 100
galaxies at the outer-slope bound, the ``max(q, 0)`` kink inactive, and the box
penalty at 2e-8. Central-difference gradients are then stable to 3-4
significant figures for step sizes from 1e-2 down to 1e-7, so finite
differences are trustworthy here and automatic differentiation, while cheaper,
is not required for correctness.

**The one direction that needs care** is ``g``, the exponent setting how the
deposit scale grows with time: its gradient reads 0.188 at a step of 1e-2 but
converges to 0.0386 for steps at or below 1e-4. That is curvature, not noise --
but it is why ``fd_step`` defaults to 1e-5 and why a coarse step would misread
that direction by a factor of five.

Two cautions for anyone switching an existing fit over.

- **Changing the optimizer changes the answer.** Re-running the adopted fit
  with L-BFGS-B moves the loss only 0.119% (0.153795 -> 0.153612 on all 2397
  galaxies) but drops the gradient norm 39-fold (0.0196 -> 0.0005), and the six
  conditioning slopes move a lot in relative terms -- the deposit scale's
  formation-time slope flips sign, from +0.00016 to -0.01570. The base kernel
  is well determined; those slopes are not. Re-run and re-judge, never swap in
  place.
- **Bounds and the box penalty do the same job.** ``bounds`` here are real
  constraints handled by the optimizer; the exp35/38 kernels instead add a
  quadratic box penalty inside the loss. Use one or the other, not both, or the
  edge is charged twice.

Self-check: ``python -m hongshao.fitting``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# scipy's own rule, for reference in the docs and the self-check below. Not
# imported from scipy: these are function-local constants there, not API.
SCIPY_REL = 0.05
SCIPY_ZERO_STEP = 0.00025

#: Finite-difference step. 1e-5 sits comfortably inside the measured stable
#: plateau (gradients hold from 1e-2 to 1e-7) and is small enough for the
#: high-curvature ``g`` direction, which needs <= 1e-4.
DEFAULT_FD_STEP = 1e-5

METHODS = ("lbfgsb", "bfgs", "neldermead", "powell")


def initial_simplex(p0, floor=0.02, rel=SCIPY_REL):
    """Starting simplex for Nelder-Mead with a meaningful span on every axis.

    ``minimize_loss(..., method="neldermead")`` applies this for you; use it
    directly only when calling scipy yourself:
    ``minimize(loss, p0, method="Nelder-Mead",
               options=dict(initial_simplex=initial_simplex(p0)))``

    Each vertex moves ONE parameter off ``p0`` by ``max(rel * |p0_i|, floor_i)``,
    so a parameter starting at or near zero still gets a real span instead of
    scipy's 0.00025.

    **The trap this fixes.** scipy's Nelder-Mead builds its starting simplex by
    perturbing each parameter RELATIVELY, by 5% of its own value
    (``nonzdelt = 0.05``), with one special case: a parameter that is exactly
    0.0 gets a fixed ``zdelt = 0.00025`` (checked against scipy 1.13.1).
    Nelder-Mead only explores the space its starting simplex spans, and along a
    direction where the loss barely changes it contracts rather than grows -- so
    a parameter handed a span of 0.00025 in a surface where it acts at order one
    comes back where it started, indistinguishable from a genuine measurement
    that it does nothing. This produced a false null in exp47, where three shape
    slopes started at exactly 0 (deliberately, so the new component nested the
    previous model) and came back at ~0.01. **The bug is broader than "exactly
    zero"**: because the step is purely relative, any parameter starting small
    in absolute terms but acting at order unity gets a meaningless span, which
    is how exp48's six conditioning slopes (sitting *near* zero) hit the same
    wall.

    ``floor`` is one number for all parameters, or one per parameter -- use the
    per-parameter form when they have different natural scales, and set each to
    roughly the smallest change in that parameter you would care about.

    Returns the (n+1, n) array scipy expects.
    """
    p0 = np.asarray(p0, float)
    if p0.ndim != 1 or len(p0) == 0:
        raise ValueError(f"p0 must be a 1-D parameter vector, got {p0.shape}")
    if not np.isfinite(p0).all():
        raise ValueError("p0 must be finite")
    floor = np.broadcast_to(np.asarray(floor, float), p0.shape)
    if not np.isfinite(floor).all() or (floor <= 0).any():
        raise ValueError("floor must be finite and positive")
    step = np.maximum(rel * np.abs(p0), floor)
    return np.vstack([p0] + [p0 + step[i] * np.eye(len(p0))[i]
                             for i in range(len(p0))])


def minimize_loss(loss, p0, method="lbfgsb", bounds=None, max_evals=4000,
                  fd_step=DEFAULT_FD_STEP, simplex_floor=0.02, options=None):
    """Minimize ``loss(p)`` with measured defaults. Returns scipy's result.

    ``method`` (see the table in the module docstring for what each cost on the
    real loss):

    - ``"lbfgsb"`` (default) -- quasi-Newton on finite-difference gradients,
      and the only one here that takes real ``bounds``. Fastest and jointly
      most accurate in the benchmark.
    - ``"bfgs"`` -- the same idea unbounded; matched L-BFGS-B almost exactly.
      Use when the problem has no box.
    - ``"neldermead"`` -- derivative-free. ``adaptive=True`` (the Gao-Han
      parameters, which matter above ~5 parameters) and ``initial_simplex`` are
      applied automatically; that combination also reached the minimum, just
      about twice as slowly. Reach for it when the loss is NOT smooth.
    - ``"powell"`` -- derivative-free direction-set. Slowest here and it did not
      quite converge, but it respects bounds without gradients, so it is the
      fallback for a bounded non-smooth problem.

    ``trust-constr`` is deliberately absent: it reached the same minimum but
    needed 39% more evaluations than L-BFGS-B with no capability we use.

    ``bounds`` is a sequence of ``(lo, hi)`` pairs, ``None`` for unbounded on
    either side. Accepted by ``lbfgsb``, ``neldermead`` and ``powell``; passing
    it with ``bfgs`` raises rather than silently ignoring it. Remember the box
    penalty caution in the module docstring.

    ``max_evals`` caps function evaluations. Note ``bfgs`` has no evaluation
    cap in scipy, so it is applied there as an ITERATION cap instead -- the one
    place this argument does not mean the same thing.

    ``options`` is merged over the defaults, so anything here can be overridden.
    """
    p0 = np.asarray(p0, float)
    if p0.ndim != 1 or len(p0) == 0:
        raise ValueError(f"p0 must be a 1-D parameter vector, got {p0.shape}")
    if not np.isfinite(p0).all():
        raise ValueError("p0 must be finite")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not np.isfinite(fd_step) or fd_step <= 0:
        raise ValueError(f"fd_step must be finite and positive, got {fd_step}")
    if fd_step > 1e-4:
        raise ValueError(
            f"fd_step={fd_step} is too coarse: gradients on this class of loss "
            f"are only stable at 1e-4 or below (the deposit-scale growth "
            f"exponent reads 5x too steep at 1e-2). Pass a smaller step, or "
            f"use a derivative-free method.")
    if bounds is not None and method == "bfgs":
        raise ValueError("bfgs does not accept bounds; use lbfgsb")
    if bounds is not None:
        if len(bounds) != len(p0):
            raise ValueError(f"bounds has {len(bounds)} entries for {len(p0)} "
                             f"parameters")
        # scipy only WARNS here and then silently clips the start point, which
        # is how a fit quietly begins somewhere the caller did not choose.
        outside = [(i, p0[i], lo, hi)
                   for i, (lo, hi) in enumerate(bounds)
                   if (lo is not None and p0[i] < lo)
                   or (hi is not None and p0[i] > hi)]
        if outside:
            raise ValueError(
                "p0 lies outside bounds (scipy would clip it silently): "
                + ", ".join(f"p0[{i}]={v:g} not in [{lo}, {hi}]"
                            for i, v, lo, hi in outside))

    if method == "lbfgsb":
        scipy_method = "L-BFGS-B"
        defaults = dict(maxfun=max_evals, maxiter=max_evals, ftol=1e-14,
                        gtol=1e-12, eps=fd_step)
    elif method == "bfgs":
        scipy_method = "BFGS"
        defaults = dict(maxiter=max_evals, gtol=1e-8, eps=fd_step)
    elif method == "neldermead":
        scipy_method = "Nelder-Mead"
        defaults = dict(maxfev=max_evals, maxiter=max_evals, adaptive=True,
                        xatol=3e-4, fatol=1e-8,
                        initial_simplex=initial_simplex(p0, simplex_floor))
    else:
        scipy_method = "Powell"
        defaults = dict(maxfev=max_evals, maxiter=max_evals, xtol=3e-4,
                        ftol=1e-8)

    return minimize(loss, p0, method=scipy_method, bounds=bounds,
                    options={**defaults, **(options or {})})


if __name__ == "__main__":       # self-checks
    # ---- initial_simplex ------------------------------------------------- #
    # 1. shape, and each vertex moves exactly one parameter
    p0 = np.array([0.5, -2.0, 0.0, 0.01])
    sx = initial_simplex(p0)
    assert sx.shape == (5, 4)
    assert np.allclose(sx[0], p0)
    for i in range(4):
        moved = np.flatnonzero(~np.isclose(sx[i + 1], p0))
        assert moved.tolist() == [i], (i, moved)

    # 2. the spans: relative where that is the larger, floored where it is not
    span = np.diag(sx[1:] - p0)
    assert np.isclose(span[1], 0.05 * 2.0)          # |p0|=2.0 -> 5% dominates
    assert np.isclose(span[2], 0.02)                # exactly zero -> the floor
    assert np.isclose(span[3], 0.02)                # NEAR zero -> also the floor
    # which is what scipy would not have done, in both of those cases
    assert span[2] / SCIPY_ZERO_STEP == 80.0
    assert span[3] > SCIPY_REL * 0.01 * 20

    # 3. a per-parameter floor is honoured
    sx2 = initial_simplex(p0, floor=[0.02, 0.02, 0.6, 0.3])
    span2 = np.diag(sx2[1:] - p0)
    assert np.isclose(span2[2], 0.6) and np.isclose(span2[3], 0.3)

    # 4. validation
    for bad in (dict(p0=np.zeros((2, 2))), dict(p0=[1.0, np.nan]),
                dict(p0=[1.0], floor=0.0), dict(p0=[1.0], floor=-1.0)):
        try:
            initial_simplex(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have raised")

    # ---- THE TRAP, reproduced -------------------------------------------- #
    # Two well-scaled parameters plus three weak ones started at exactly 0; the
    # weak ones are worth 1% as much to the loss. scipy's default simplex
    # CONVERGES on the wrong answer. Several well-scaled parameters alongside
    # the weak ones are load-bearing: in a two-parameter toy Nelder-Mead
    # escapes anyway and the trap does not reproduce.
    def toy(p):
        return float(np.sum((p[:2] - 1.0) ** 2)
                     + 0.01 * np.sum((p[2:] - 1.0) ** 2))

    start = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    opts = dict(maxiter=4000, xatol=3e-4, fatol=1e-10)
    default = minimize(toy, start, method="Nelder-Mead", options=opts)
    fixed = minimize(toy, start, method="Nelder-Mead",
                     options=dict(opts, initial_simplex=initial_simplex(start)))
    assert default.success and fixed.success           # BOTH report convergence
    assert np.abs(default.x[2:]).max() < 0.5           # ... one of them is wrong
    assert np.abs(fixed.x[2:] - 1.0).max() < 1e-2
    assert fixed.fun < default.fun / 1e6

    # ---- minimize_loss --------------------------------------------------- #
    # 5. every method solves the toy that defeats raw Nelder-Mead, and the
    #    neldermead route does so WITHOUT the caller knowing about simplexes
    solved = {}
    for m in METHODS:
        r = minimize_loss(toy, start, method=m, max_evals=20000)
        solved[m] = (r.fun, np.abs(r.x - 1.0).max())
        assert np.abs(r.x - 1.0).max() < 5e-2, (m, r.x)
        assert r.fun < 1e-3, (m, r.fun)

    # 6. bounds are real constraints (the toy's optimum at 1.0 is outside, so
    #    every parameter should be driven to the wall), and are refused where
    #    unsupported
    box = [(0.0, 0.6)] * 5
    bounded = minimize_loss(toy, start, method="lbfgsb", bounds=box)
    assert bounded.x.max() <= 0.6 + 1e-9, bounded.x
    assert np.allclose(bounded.x, 0.6, atol=1e-6), bounded.x
    for m in ("neldermead", "powell"):
        b = minimize_loss(toy, start, method=m, bounds=box, max_evals=20000)
        assert b.x.max() <= 0.6 + 1e-6, (m, b.x)

    # 7. a start point outside the bounds is REFUSED, not silently clipped
    try:
        minimize_loss(toy, start, method="lbfgsb", bounds=[(0.0, 0.4)] * 5)
    except ValueError as exc:
        assert "outside bounds" in str(exc)
    else:
        raise AssertionError("p0 outside bounds should have raised")

    # 8. validation, including the too-coarse finite-difference step
    for bad in (dict(method="lm"), dict(fd_step=1e-2), dict(fd_step=0.0),
                dict(fd_step=-1e-5), dict(method="bfgs", bounds=[(0, 1)] * 5),
                dict(bounds=[(0, 1)] * 3), dict(p0=[1.0, np.nan, 0.0, 0.0, 0.0])):
        kw = dict(dict(loss=toy, p0=start), **bad)
        try:
            minimize_loss(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have raised")

    # 9. options override the defaults rather than being ignored
    short = minimize_loss(toy, start, method="neldermead",
                          options=dict(maxfev=25))
    assert short.nfev <= 30, short.nfev

    print("fitting self-check OK\n"
          f"  weak parameters started at 0, true value 1.0:\n"
          f"    scipy default simplex -> {np.round(default.x[2:], 4).tolist()}"
          f"  loss {default.fun:.2e}  (converged={default.success})\n"
          f"    initial_simplex       -> {np.round(fixed.x[2:], 4).tolist()}"
          f"  loss {fixed.fun:.2e}  (converged={fixed.success})\n"
          "  minimize_loss on the same toy (worst parameter error, loss):\n"
          + "".join(f"    {m:<12} {solved[m][1]:.2e}  {solved[m][0]:.2e}\n"
                    for m in METHODS))
