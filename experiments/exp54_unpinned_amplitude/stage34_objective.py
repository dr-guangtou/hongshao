"""exp54 Stage 3.4 — IS THE CEILING A PROPERTY OF THE MODEL, OR OF THE LOSS?

`stage34_ceiling.py` finds its weights with non-negative least squares, which
minimises the summed squared FRACTIONAL residual of the ABSOLUTE curve of
growth. That is not the production objective: `score_F` renormalises every
profile at 100 kpc, takes an RMS over radii, then a mean over epochs, then a
mean over galaxies. So every number in that module is the production score of
one feasible solution optimised under a SURROGATE -- a low value proves the
basis can reach at least that, a high value proves nothing.

This module removes the caveat. Two facts make it possible to optimise the
production objective EXACTLY rather than on a diagnostic sample:

  * `score_F` is a mean over galaxies of a per-galaxy quantity, so minimising
    it galaxy by galaxy minimises the population score exactly. The same holds
    for `score_A^2`, a mean over galaxy-epochs.
  * Both per-galaxy losses are smooth in the weights and have closed-form
    gradients, so a bound-constrained quasi-Newton solve from the NNLS start
    converges in milliseconds.

WHAT IS COMPUTED, all with `w >= 0` and the same causal design:

  `nnls`         the surrogate optimum -- `stage34_ceiling`'s object, repeated
                 here so the three are read on one page.
  `shape`        the EXACT minimum of the production shape loss.
  `amp`          the EXACT minimum of the production amplitude loss.
  `both`         a per-galaxy `lam_A * amp + lam_F * shape^2`, the closest
                 separable analogue of the fitted loss. Reported as a
                 diagnostic; it is not an exact bound on either score alone.

THE MONOTONE BOUND, done properly. Stage 3.0's basis-free reference is an
isotonic projection of the truth: reasonable, but not the minimum of any
reported quantity, and the epoch at which it places its unavoidable error is a
property of the projection rather than of the physics. `monotone_bound`
optimises the SAME production shape loss over all profiles that are
non-decreasing in time -- the only property every static non-negative
deposition model must have -- so the result is a genuine floor in the reported
units, for any basis whatsoever.

THE TRADE-OFF, which is the point the ceiling most easily overstates.
Monotonicity guarantees that a conflict exists; it does not say which epoch
pays for it. `pareto` re-solves the monotone bound with the z=2 epoch weighted
from 0.2 to 20 and traces the z=2 central deficit against the z=0.4 central
overshoot. A defect that slides along that curve at nearly constant total loss
is a choice the objective is making, not an irreducible error.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage34_objective.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage34_ceiling as C                              # noqa: E402

R = C.R
I100 = C.I100
Z = C.Z
RULE = C.RULE
OUT = HERE / "outputs" / "stage34_objective.npz"
LABEL = "gompertz_log-E2-S2"
OBJECTIVES = ("nnls", "shape", "amp", "both")
#: z=2 weights for the trade-off sweep
PARETO_W = (0.2, 0.5, 1.0, 2.0, 5.0, 20.0)
#: the sweep costs six extra monotone solves per galaxy and only a population
#: median is read off it, so it runs on a halo-mass-stratified subsample
PARETO_N = 250
PARTIAL = HERE / "outputs" / "stage34_objective_partial.npz"


# --------------------------------------------------------------------------- #
# the per-galaxy production losses and their gradients                         #
# --------------------------------------------------------------------------- #
def shape_loss(m, d, use, w_ep=None):
    """(loss, dloss/dm) for the per-galaxy production shape loss.

    `mean over epochs of rms over radii of (F_model - F_truth)/F_truth`, with
    `F = M*(<R)/M*(<100 kpc)`. Averaging this over galaxies IS `score_F`
    before the `L_F_ref` division, so minimising it per galaxy minimises the
    population score exactly.
    """
    ne, nr = m.shape
    ww = np.ones(ne) if w_ep is None else np.asarray(w_ep, float)
    ww = ww * use
    tot = ww.sum()
    g = np.zeros_like(m)
    loss = 0.0
    for k in range(ne):
        if ww[k] <= 0:
            continue
        m100 = max(m[k, I100], 1.0)
        c = d[k, I100] / m100
        e = m[k] * c / d[k] - 1.0
        q = np.sqrt(np.mean(e ** 2))
        loss += ww[k] * q
        if q <= 0:
            continue
        # d e_r / d m_s = (c/d_r) [ delta_rs - m_r delta_{s,100} / m100 ],
        # the second term because renormalising at 100 kpc makes every radius
        # depend on that one aperture
        base = (c / d[k]) * e / (nr * q)
        g[k] += ww[k] * base
        g[k, I100] -= ww[k] * (base * m[k]).sum() / m100
    return loss / tot, g / tot


def amp_loss(m, d, use):
    """(loss, dloss/dm) for the per-galaxy amplitude loss.

    `mean over epochs of [(log10 M*(<100) - log10 truth)/sigma_A(z)]^2`.
    Averaging over galaxy-epochs gives `score_A^2` exactly.
    """
    ne, nr = m.shape
    tot = max(use.sum(), 1)
    g = np.zeros_like(m)
    loss = 0.0
    for k in range(ne):
        if not use[k]:
            continue
        m100 = max(m[k, I100], 1.0)
        a = (np.log10(m100) - np.log10(d[k, I100])) / F.SIGMA_A[k]
        loss += a ** 2
        g[k, I100] += 2 * a / (F.SIGMA_A[k] * np.log(10.0) * m100)
    return loss / tot, g / tot


def _objective(kind, A, d, use, ne, nr, lam_A=1.0, lam_F=1.0, w_ep=None,
               scale=1.0):
    """A closure returning (value, gradient) in the weights, for L-BFGS-B.

    `scale` puts the variables in units of a typical deposit mass. WITHOUT IT
    THE SOLVE SILENTLY DOES NOTHING: the weights are ~1e10 and the gradient of
    a dimensionless loss with respect to them is ~1e-11, so L-BFGS-B's first
    trial step moves the parameters by a part in 1e11 and it declares
    convergence at the starting point. The same trap cost this experiment a
    bounded-variable solve earlier; both are now scaled at the call site.
    """
    d = np.asarray(d, float)

    def f(v):
        w = v * scale
        m = (A @ w).reshape(ne, nr)
        if kind == "shape":
            v, gm = shape_loss(m, d, use, w_ep)
        elif kind == "amp":
            v, gm = amp_loss(m, d, use)
        else:                                            # "both"
            vs, gs = shape_loss(m, d, use, w_ep)
            va, ga = amp_loss(m, d, use)
            v = lam_A * va + lam_F * (vs / F.L_F_REF) ** 2
            gm = lam_A * ga + lam_F * 2 * (vs / F.L_F_REF) * gs / F.L_F_REF
        return float(v), scale * (A.T @ gm.ravel())
    return f


def gauge_fix(w, A, d, use, ne, nr):
    """Rescale `w` to the amplitude-optimal overall normalisation.

    The shape loss is EXACTLY invariant under `w -> s w` (every profile is
    renormalised at 100 kpc), so its minimiser is a ray, not a point, and the
    `score_A` of an arbitrary point on that ray is meaningless. This picks the
    point on the ray that minimises the amplitude loss, in closed form, so the
    two scores can be read side by side.
    """
    m = (A @ w).reshape(ne, nr)
    num = den = 0.0
    for k in range(ne):
        if not use[k] or m[k, I100] <= 0:
            continue
        wt = 1.0 / F.SIGMA_A[k] ** 2
        num += wt * (np.log10(d[k, I100]) - np.log10(m[k, I100]))
        den += wt
    return w * (10.0 ** (num / den) if den > 0 else 1.0)


def solve_objective(kind, A, d, use, w0, ne, nr, n_start=3, seed=0):
    """Minimise a production loss over `w >= 0`, multi-start from NNLS."""
    scale = max(float(np.max(w0)), 1.0)
    f = _objective(kind, A, d, use, ne, nr, scale=scale)
    rng = np.random.default_rng(seed)
    starts = [w0 / scale]
    for _ in range(n_start - 1):
        starts.append(np.clip(w0 / scale + rng.normal(0, 0.25, w0.shape),
                              0.0, None))
    best, best_v, ok = starts[0], f(starts[0])[0], True
    for s in starts:
        r = minimize(f, s, jac=True, method="L-BFGS-B",
                     bounds=[(0.0, None)] * len(s),
                     options=dict(maxiter=2000, ftol=1e-16, gtol=1e-14))
        if r.fun < best_v:
            best, best_v, ok = r.x, float(r.fun), bool(r.success)
    w = best * scale
    if kind == "shape":
        w = gauge_fix(w, A, d, use, ne, nr)
    return w, best_v, ok


# --------------------------------------------------------------------------- #
# the objective-matched monotone bound                                         #
# --------------------------------------------------------------------------- #
def monotone_bound(d, use, kind="both", w_ep=None, n_start=3, seed=0):
    """The best non-decreasing-in-time profile set, under a chosen loss.

    Optimises over an ARBITRARY cumulative profile at every epoch, subject only
    to `M*(<R, t)` never falling with time -- the one property every static
    non-negative deposition model has, whatever its basis. So this is a floor
    for the entire model class, which an isotonic projection of the truth is
    not.

    **WHY `kind` MATTERS MORE THAN IT LOOKS.** Under `kind="shape"` the bound
    is nearly ZERO, and that is not a numerical accident. The production shape
    loss renormalises every epoch at 100 kpc, while monotonicity constrains the
    ABSOLUTE profile -- and any set of shapes can be made monotone by inflating
    the later epochs' amplitudes enough. **Deposition-only monotonicity
    therefore puts almost no floor under the shape loss on its own.** It binds
    only when the amplitude is constrained at the same time, which is what
    `kind="both"` and `kind="absfrac"` do. Stage 3.0's isotonic projection
    scored ~0.29 in `score_F` because it happened to match the amplitude, not
    because 0.29 was unreachable.

      `shape`    the production shape loss alone -- reported to demonstrate the
                 point above, never as a floor.
      `both`     `lam_A * amplitude + lam_F * (shape/L_F_ref)^2` per galaxy,
                 the separable analogue of the fitted loss. The meaningful one.
      `absfrac`  the summed squared fractional residual of the ABSOLUTE curve
                 of growth. Solved EXACTLY by weighted pool-adjacent-violators
                 at each radius, with weights `1/d^2` -- no optimiser, no
                 convergence question, and the same functional the NNLS ceiling
                 minimises, so the two are directly comparable.

    Returns (profiles, loss, converged).
    """
    ne, nr = d.shape
    order = list(range(ne))[::-1]                  # earliest (z=2) first

    if kind == "absfrac":
        out = np.array(d, float)
        ks = [k for k in range(ne) if use[k]]
        if len(ks) >= 2:
            oo = ks[::-1]
            for c in range(nr):
                y = d[oo, c]
                out[oo, c] = C.weighted_pava(y, 1.0 / np.clip(y, 1.0, None) ** 2)
        loss = float(np.mean(((out - d) / np.clip(d, 1.0, None)) ** 2))
        return out, loss, True

    T = np.zeros((ne, ne))
    for s in range(ne):
        T[s, : s + 1] = 1.0
    Tf = np.zeros((ne, ne))
    for s, k in enumerate(order):
        Tf[k] = T[s]
    scale = max(float(d[0, I100]), 1.0)            # see `_objective` on scaling

    def unpack(x):
        return (Tf @ x.reshape(ne, nr)) * scale

    def f(x):
        m = unpack(x)
        if kind == "shape":
            v, gm = shape_loss(m, d, use, w_ep)
        else:
            vs, gs = shape_loss(m, d, use, w_ep)
            va, ga = amp_loss(m, d, use)
            v = va + (vs / F.L_F_REF) ** 2
            gm = ga + 2 * (vs / F.L_F_REF) * gs / F.L_F_REF
        return float(v), scale * (Tf.T @ gm).ravel()

    x0 = np.zeros((ne, nr))
    x0[0] = d[order[0]]
    for s in range(1, ne):
        x0[s] = np.clip(d[order[s]] - d[order[s - 1]], 0.0, None)
    x0 = x0.ravel() / scale
    rng = np.random.default_rng(seed)
    best, best_v, moved = x0, f(x0)[0], False
    for t in range(n_start):
        s0 = x0 if t == 0 else np.clip(x0 * rng.uniform(0.6, 1.6, x0.size),
                                       0.0, None)
        r = minimize(f, s0, jac=True, method="L-BFGS-B",
                     bounds=[(0.0, None)] * x0.size,
                     options=dict(maxiter=4000, ftol=1e-16, gtol=1e-14))
        if r.fun < best_v:
            best, best_v, moved = r.x, float(r.fun), True
    return unpack(best), best_v, moved


# --------------------------------------------------------------------------- #
def main(smoke=False):
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(C.POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    cuts = np.load(C.SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    complete_all = np.isfinite(m200) & (m200 >= cuts[None, :])

    rows = all_rows[::40] if smoke else all_rows
    recs = H.build_records(rows=rows, verbose=False)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    mask = complete_all[sel]

    # the fixed massive-progenitor cohort: same galaxies at every epoch, which
    # is the only sample on which an EVOLUTION statement is well posed
    keep = np.where(mask[:, 4])[0]
    recs = [recs[i] for i in keep]
    data, mask = data[keep], mask[keep]
    n, ne, nr = len(recs), 5, len(R)
    use = np.ones((n, ne), bool)

    spec = S33.spec_from_label(LABEL)
    theta = np.asarray(np.load(C.PEREPOCH, allow_pickle=True)
                       [f"{LABEL}_theta"], float)

    print(f"exp54 STAGE 3.4 — IS THE CEILING A PROPERTY OF THE MODEL OR OF "
          f"THE LOSS?")
    print(f"  basis {LABEL}; the fixed z=2-threshold cohort, {n} galaxies, all "
          f"five epochs")
    print(f"  every rung uses w >= 0 and the same causal design; only the "
          f"objective changes\n")

    MONO = ("mono_shape", "mono_both", "mono_absfrac")
    preds = {k: np.empty((n, ne, nr)) for k in OBJECTIVES + MONO + ("model",)}
    conv = {k: np.zeros(n, bool) for k in OBJECTIVES + MONO}
    pgl = {k: np.zeros(n) for k in MONO}
    pareto = {w: np.empty((n, ne, nr)) for w in PARETO_W}

    lmh = pop["logmh_zk_diffmah"][sel][keep][:, 0]
    if smoke or n <= PARETO_N:
        par_idx = np.arange(n)
    else:                                  # stratified by z=0.4 halo mass
        order = np.argsort(lmh)
        par_idx = order[np.linspace(0, n - 1, PARETO_N).astype(int)]
    par_set = set(int(v) for v in par_idx)
    print(f"  the trade-off sweep runs on {len(par_set)} galaxies, evenly "
          f"spaced in z=0.4 halo mass\n")

    start = 0
    if PARTIAL.exists() and not smoke:
        ck = np.load(PARTIAL, allow_pickle=True)
        if int(ck["n"]) == n and int(ck["done"]) > 0:
            start = int(ck["done"])
            for k in preds:
                preds[k][:start] = ck[f"pred_{k}"][:start]
            for k in conv:
                conv[k][:start] = ck[f"conv_{k}"][:start]
            for k in pgl:
                pgl[k][:start] = ck[f"pgl_{k}"][:start]
            for wz in PARETO_W:
                pareto[wz][:start] = ck[f"par_{wz}"][:start]
            print(f"  resuming from checkpoint at galaxy {start}\n")

    t0 = time.time()
    for i, h in enumerate(recs):
        if i < start:
            continue
        B, em, dmh, good = C.build_basis(spec, theta, h)
        wmod = C.model_weights(spec, theta, h, good)
        A = C._design(B, em)
        d = data[i]
        preds["model"][i] = (A @ wmod).reshape(ne, nr)
        w0 = C._solve(A, d.ravel())
        preds["nnls"][i] = (A @ w0).reshape(ne, nr)
        conv["nnls"][i] = True
        for kind in ("shape", "amp", "both"):
            w, _, ok = solve_objective(kind, A, d, use[i], w0, ne, nr)
            preds[kind][i] = (A @ w).reshape(ne, nr)
            conv[kind][i] = ok
        for mk in MONO:
            preds[mk][i], pgl[mk][i], conv[mk][i] = monotone_bound(
                d, use[i], kind=mk.split("_", 1)[1])
        if i in par_set:
            for wz in PARETO_W:
                w_ep = np.ones(ne)
                w_ep[4] = wz
                pareto[wz][i], _, _ = monotone_bound(d, use[i], kind="both",
                                                     w_ep=w_ep)
        else:
            for wz in PARETO_W:
                pareto[wz][i] = np.nan
        if i and i % 100 == 0:
            print(f"    {i}/{n} in {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            if not smoke:            # long runs checkpoint per unit of work
                np.savez(PARTIAL, n=n, done=i + 1,
                         **{f"pred_{k}": v for k, v in preds.items()},
                         **{f"conv_{k}": v for k, v in conv.items()},
                         **{f"pgl_{k}": v for k, v in pgl.items()},
                         **{f"par_{w}": pareto[w] for w in PARETO_W})
    print(f"    {n} galaxies in {(time.time() - t0) / 60:.1f} min")

    print(f"\n{RULE}\n  THE SAME COURT, THREE OBJECTIVES — score_F "
          f"(lower is better)\n{RULE}")
    print(f"     {'weights chosen by':<34}"
          + "".join(f"{f'z={z}':>9}" for z in Z) + f"{'mean':>9}")
    ORDER = ("nnls", "shape", "amp", "both") + MONO + ("model",)
    NAMES = {"nnls": "the NNLS surrogate (ceiling)",
             "shape": "the EXACT production shape loss",
             "amp": "the EXACT production amplitude loss",
             "both": "a per-galaxy amplitude + shape",
             "mono_shape": "monotone, shape loss  [NOT a floor]",
             "mono_both": "monotone, amplitude + shape",
             "mono_absfrac": "monotone, absolute frac (exact PAVA)",
             "model": "the fitted global law"}
    rows_out = {}
    for k in ORDER:
        sa, sf = C._score_masked(preds[k], data, mask)
        rows_out[k] = (sa, sf)
        print(f"     {NAMES[k]:<38}" + "".join(f"{v:>9.3f}" for v in sf)
              + f"{np.nanmean(sf):>9.3f}")
    print(f"\n     score_A (1.0 = the halo-only regression)")
    print(f"     {'weights chosen by':<34}"
          + "".join(f"{f'z={z}':>9}" for z in Z) + f"{'mean':>9}")
    for k in ORDER:
        sa = rows_out[k][0]
        print(f"     {NAMES[k]:<38}" + "".join(f"{v:>9.3f}" for v in sa)
              + f"{np.nanmean(sa):>9.3f}")
    print(f"\n     L-BFGS-B improved on its start in: "
          + "  ".join(f"{k} {100 * conv[k].mean():.0f} per cent"
                      for k in ("shape", "amp", "both", "mono_both")))
    print(f"\n     WHY THE SHAPE-ONLY MONOTONE BOUND IS ~0, AND WHY THAT "
          f"MATTERS. The production shape loss\n     renormalises every epoch "
          f"at 100 kpc; monotonicity constrains the ABSOLUTE profile. Any set "
          f"of\n     shapes can be made monotone by inflating the later "
          f"epochs, so monotonicity alone puts almost\n     NO floor under "
          f"score_F. Stage 3.0's isotonic projection scored ~0.29 because it "
          f"chose to\n     match the amplitude, not because 0.29 was "
          f"unreachable. The premise binds only when amplitude\n     and "
          f"shape are constrained together — the `mono_both` and "
          f"`mono_absfrac` rows.")

    idx = [0, 3, 6, 11, 18, 23]
    print(f"\n{RULE}\n  DOES THE OBJECTIVE MOVE THE z=2 CENTRAL DEFICIT?"
          f"\n{RULE}")
    print(f"     median 100*(prediction - truth)/truth")
    for k in (0, 4):
        print(f"\n     z = {Z[k]}")
        print(f"       {'weights chosen by':<34}"
              + "".join(f"{R[c]:>9.1f}" for c in idx) + "   kpc")
        for nm in ORDER:
            p = preds[nm]
            print(f"       {nm:<38}" + "".join(
                f"{100 * np.median(p[:, k, c] / data[:, k, c] - 1):>9.1f}"
                for c in idx))

    print(f"\n{RULE}\n  THE TRADE-OFF: MONOTONICITY FIXES THAT A CONFLICT "
          f"EXISTS, NOT WHICH EPOCH PAYS\n{RULE}")
    print(f"     The basis-free monotone bound, re-solved with the z=2 epoch "
          f"weighted from 0.2 to 20.\n     A residual that slides along this "
          f"curve at nearly constant loss is a choice the objective\n     is "
          f"making; one that does not move is structural.")
    i2 = int(np.argmin(np.abs(R - 2.0)))
    par = {}
    ps = np.array(sorted(par_set))
    print(f"     (on the {len(ps)}-galaxy stratified subsample)")
    print(f"\n       {'z=2 weight':>12}{'score_F mean':>15}"
          f"{'z=2 at 2.0 kpc':>18}{'z=0.4 at 2.0 kpc':>20}")
    for wz in PARETO_W:
        p = pareto[wz][ps]
        _, sf = C._score_masked(p, data[ps], mask[ps])
        a = 100 * np.median(p[:, 4, i2] / data[ps][:, 4, i2] - 1)
        b = 100 * np.median(p[:, 0, i2] / data[ps][:, 0, i2] - 1)
        par[wz] = (float(np.nanmean(sf)), a, b)
        print(f"       {wz:>12.1f}{np.nanmean(sf):>15.3f}{a:>17.1f}%"
              f"{b:>19.1f}%")
    print(f"\n     For reference the fitted model is "
          f"{100 * np.median(preds['model'][:, 4, i2] / data[:, 4, i2] - 1):.1f}"
          f"{'%'} at z=2 and "
          f"{100 * np.median(preds['model'][:, 0, i2] / data[:, 0, i2] - 1):.1f}"
          f"{'%'} at z=0.4.")

    if smoke:
        print("\n  smoke run: writing nothing")
        return
    np.savez_compressed(
        OUT, label=LABEL, rows=sel[keep], mask=mask,
        objectives=np.array(OBJECTIVES), pareto_w=np.array(PARETO_W),
        pareto=np.array([[par[w][j] for j in range(3)] for w in PARETO_W]),
        **{f"pred_{k}": v for k, v in preds.items()},
        **{f"pgl_{k}": v for k, v in pgl.items()},
        **{f"sA_{k}": rows_out[k][0] for k in rows_out},
        **{f"sF_{k}": rows_out[k][1] for k in rows_out},
        **{f"conv_{k}": v for k, v in conv.items()},
        par_idx=ps)
    PARTIAL.unlink(missing_ok=True)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
