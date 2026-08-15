"""Fitting helpers. Currently one: a starting simplex that is actually usable.

**The trap.** ``scipy.optimize.minimize(..., method="Nelder-Mead")`` builds its
starting simplex by perturbing each parameter in turn off the start point, and
the perturbation is RELATIVE — 5% of the parameter's current value
(``nonzdelt = 0.05``), with one special case: a parameter that is exactly 0.0
gets a fixed ``zdelt = 0.00025`` instead (checked against scipy 1.13.1).

Nelder-Mead only ever explores the space its starting simplex spans; it
reflects, expands and contracts that shape, and along a direction where the
loss barely changes it contracts rather than grows. So a parameter handed a
span of 0.00025 in a loss surface where it only does anything at values of
order one comes back essentially where it started — and that is indistinguishable
from a genuine measurement that the parameter does nothing.

This has produced a false null in this project twice. In exp47 three shape
slopes were started at exactly 0 (deliberately, so the new component nested the
previous model as a special case); scipy spanned them by 0.00025 and they came
back at about 0.01 — a property of the start point, not of the data. The
self-check below reproduces it in five lines: three weakly-influential
parameters started at 0 make Nelder-Mead report CONVERGENCE at 0.28 instead of
the true 1.0, with a final loss 3e8 times worse, identically at every iteration
budget from 400 to 4000. It is a wrong answer, not a truncated run.

**The bug is broader than "exactly zero".** Because the step is purely relative,
any parameter that starts small in absolute terms but acts at order unity gets a
near-meaningless span — no special case needed. exp48's six conditioning slopes
(how the profile's size and the star-formation-efficiency width shift with halo
mass, concentration and formation time) sit NEAR zero rather than at zero and
hit the same wall through the ordinary 5% path.

Hence a relative step with an absolute FLOOR, which covers both cases, and a
floor that may be set per parameter, because parameters with different natural
scales need different ones (exp47 needed 0.3 for a core amplitude, 0.6 for the
shape slopes under test and 0.15 for a log radius).

Nothing in the library calls this. Every existing fit keeps scipy's default
simplex, so no already-published number moves; use it for new fits, and
especially for any fit where a parameter is started at or near zero.

Self-check: ``python -m hongshao.fitting``.
"""
from __future__ import annotations

import numpy as np

# scipy's own rule, for reference in the docs and the self-check below. Not
# imported from scipy: these are function-local constants there, not API.
SCIPY_REL = 0.05
SCIPY_ZERO_STEP = 0.00025


def initial_simplex(p0, floor=0.02, rel=SCIPY_REL):
    """Starting simplex for Nelder-Mead with a meaningful span on every axis.

    Pass the result straight through:
    ``minimize(loss, p0, method="Nelder-Mead",
               options=dict(initial_simplex=initial_simplex(p0)))``

    Each vertex moves ONE parameter off ``p0`` by ``max(rel * |p0_i|, floor_i)``,
    so a parameter starting at or near zero still gets a real span instead of
    scipy's 0.00025.

    ``floor`` is one number for all parameters, or one per parameter — use the
    per-parameter form when they have different natural scales, and set each to
    roughly the smallest change in that parameter you would care about.

    Returns the (n+1, n) array scipy expects. Note that supplying a simplex
    changes what a fit returns, so do not switch an existing, already-reported
    fit to this without re-running it.
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


if __name__ == "__main__":       # self-checks
    from scipy.optimize import minimize

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

    # 5. THE TRAP, reproduced. Two well-scaled parameters plus three weak ones
    #    started at exactly 0; the weak ones are worth 1% as much to the loss.
    #    scipy's default simplex CONVERGES on the wrong answer.
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

    print("fitting self-check OK\n"
          f"  weak parameters started at 0, true value 1.0:\n"
          f"    scipy default simplex -> {np.round(default.x[2:], 4).tolist()}"
          f"  loss {default.fun:.2e}  (converged={default.success})\n"
          f"    initial_simplex       -> {np.round(fixed.x[2:], 4).tolist()}"
          f"  loss {fixed.fun:.2e}  (converged={fixed.success})")
