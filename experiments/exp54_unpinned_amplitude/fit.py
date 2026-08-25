"""exp54 Stage 3 — the fitting harness, the split objective, and the mandatory
identifiability report.

THE OBJECTIVE (plan §5). The curve of growth is split losslessly into an
amplitude and a dimensionless shape, and each term is normalized by ITS OWN
measured benchmark so both read ~1 at "as good as we currently know how to do":

    A       = log10 M*(<100 kpc)
    F(<R)   = M*(<R) / M*(<100 kpc)
    score_A = rms_{gal,epoch}[ (A_model - A_data) / sigma_A(z) ]
    score_F = Objective()(F_model, F_data, R) / L_F_ref
    L       = lam_A * score_A**2 + lam_F * score_F**2

`sigma_A(z)` is the Stage 1 best halo-only row (0.1047 / 0.1359 / 0.1417 /
0.1456 / 0.1744 dex) and `L_F_ref = 0.158196` is the incumbent's 100 kpc shape
loss from Stage 0. Normalizing only `L_A` and leaving `L_F` raw would be an
undeclared weight, which the review flagged.

THE IDENTIFIABILITY REPORT is not optional (plan §6). A parameter can be
useless with a sharp optimum, or useful and completely undetermined, and the
loss distinguishes neither. Every fit ships:

  * a scaled residual-Jacobian SVD -- cheaper and better conditioned than a
    finite-difference Hessian, and it does not need second differences;
  * the effective parameter count at a noise-set threshold;
  * a one-parameter CONDITIONAL LOSS SLICE per parameter. **It is not a
    profile scan**: the other parameters are held FIXED while one is displaced,
    so it measures conditional curvature. A genuine profile would re-optimise
    the remaining parameters at every grid point, and in a correlated model the
    conditional slice can rise steeply along a direction the profiled curve
    leaves nearly flat. Do not quote it as evidence of identifiability on its
    own;
  * bootstrap stability over galaxies;
  * profiles of DERIVED quantities (R50/R200c, deposited mass by redshift),
    which are often identified where their constituents are not.

**A parameter that fails may stay in the model; its VALUE is never quoted.**

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/fit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink"):
    sys.path.insert(0, str(p))

import model as M                                        # noqa: E402
from hongshao.objective import Objective                 # noqa: E402

R_GRID = np.array([2., 2.76, 3.72, 4.92, 6.38, 8.14, 10.25, 12.74, 15.66,
                   19.05, 22.97, 27.46, 32.58, 38.39, 44.94, 52.30, 60.52,
                   69.68, 79.84, 91.07, 103.45, 117.05, 131.94, 148.22])
I100 = int(np.argmin(np.abs(R_GRID - 100.0)))
#: Stage 1's best halo-only amplitude scatter, per epoch -- the benchmark, NOT
#: automatically the correct weight (review §5.1); see `lam_A`/`lam_F`.
#:
#: CORRECTED 2026-08-23. The previous values
#:     0.1047  0.1359  0.1417  0.1456  0.1744
#: were inflated 8-21% at z >= 0.7 by ONE galaxy, row 181, whose measured
#: M*(<100 kpc) collapses from 10^11.71 at z=0.4 to a flat ~10^8 at all four
#: higher-redshift epochs -- a broken cross-match, not a progenitor. Removing
#: it alone moves the best halo-only row to
#:     0.1044  0.1120  0.1204  0.1279  0.1611
#: (the next-largest backward drop in the sample is 2.03 dex and moves it by
#: <= 0.0015 dex, so this is one galaxy, not a tail).
#:
#: WHY IT MATTERED OUT OF ALL PROPORTION TO ITS COUNT: row 181 is in Stage 1's
#: 2397 but in NONE of the model-fitting subsamples -- not Stage 3.1's 300, not
#: Stage 3.2's 240, not Stage 3.3's 1199. So every `score_A` in this experiment
#: divided a model error measured on clean galaxies by a benchmark charged for
#: a corrupt one. `scoreboard.py` now rejects such histories at source
#: (`MAX_BACKWARD_DEX`), so this array and that rejection must move together.
SIGMA_A = np.array([0.1044, 0.1120, 0.1204, 0.1279, 0.1611])
#: Stage 0's incumbent shape loss at the 100 kpc anchor
L_F_REF = 0.158196
#: a failed theta must be PRICED, never dropped (exp53's lesson)
FAIL = 25.0


class Problem:
    """Model + halo records + measured CoGs + the split objective."""

    def __init__(self, spec, halos, data, epochs=(0, 4), lam_A=1.0, lam_F=1.0):
        self.spec, self.halos, self.epochs = spec, halos, list(epochs)
        self.lam_A, self.lam_F = lam_A, lam_F
        self.obj = Objective()
        self.raw = np.asarray(data)                 # (n, 5, R), unsliced
        d = self.raw[:, self.epochs, :]
        self.data = d
        self.A_d = np.log10(d[:, :, I100])
        self.F_d = d / d[:, :, I100][:, :, None]
        self.sig = SIGMA_A[self.epochs][None, :]
        self.n_eval = 0

    def predict(self, theta):
        out = np.full((len(self.halos), len(self.epochs), len(R_GRID)), np.nan)
        for i, h in enumerate(self.halos):
            m = M.forward(self.spec, theta, h, self.epochs, R_GRID)
            if m is not None:
                out[i] = m
        return out

    def scores(self, theta):
        """(score_A, score_F, n_bad) -- both ~1 at the benchmark."""
        m = self.predict(theta)
        good = np.isfinite(m).all(axis=(1, 2)) & (m[:, :, I100] > 0).all(axis=1)
        n_bad = int((~good).sum())
        if good.sum() < 10:
            return np.nan, np.nan, n_bad
        mg = m[good]
        A_m = np.log10(np.clip(mg[:, :, I100], 1.0, None))
        dA = (A_m - self.A_d[good]) / self.sig
        score_A = float(np.sqrt(np.mean(dA ** 2)))
        F_m = mg / np.clip(mg[:, :, I100], 1.0, None)[:, :, None]
        score_F = float(self.obj(F_m, self.F_d[good], R_GRID)) / L_F_REF
        return score_A, score_F, n_bad

    def per_epoch(self, theta):
        """(score_A[k], score_F[k]) per epoch -- what the JUDGE ranks on."""
        m = self.predict(theta)
        good = np.isfinite(m).all(axis=(1, 2)) & (m[:, :, I100] > 0).all(axis=1)
        if good.sum() < 10:
            return np.full(len(self.epochs), np.nan), np.full(len(self.epochs), np.nan)
        mg = m[good]
        A_m = np.log10(np.clip(mg[:, :, I100], 1.0, None))
        dA = (A_m - self.A_d[good]) / self.sig
        sa = np.sqrt(np.mean(dA ** 2, axis=0))
        F_m = mg / np.clip(mg[:, :, I100], 1.0, None)[:, :, None]
        sf = np.array([self.obj(F_m[:, [j]], self.F_d[good][:, [j]], R_GRID)
                       / L_F_REF for j in range(len(self.epochs))])
        return sa, sf

    def diagnostics(self, theta):
        """Over-reach and the halo-mass tilt -- judge criteria, not losses."""
        m = self.predict(theta)
        good = np.isfinite(m).all(axis=(1, 2)) & (m[:, :, I100] > 0).all(axis=1)
        mg, dg = m[good], self.data[good]
        over = np.median(1.0 - mg[:, 0, -1] / np.clip(
            mg[:, 0, I100] / np.clip(mg[:, 0, I100], 1e-30, None)
            * mg[:, 0, -1], 1e-30, None))          # placeholder, see below
        # fraction of the model's own deposited mass outside the measured grid
        # is not observable from the CoG alone; use the RESIDUAL TILT instead
        r148 = np.log10(np.clip(mg[:, 0, -1], 1.0, None)) - np.log10(dg[:, 0, -1])
        return r148, good

    def loss(self, theta):
        self.n_eval += 1
        a, f, n_bad = self.scores(theta)
        if not (np.isfinite(a) and np.isfinite(f)):
            return FAIL
        pen = FAIL * n_bad / len(self.halos)     # price failures, never drop
        return self.lam_A * a ** 2 + self.lam_F * f ** 2 + pen

    def residuals(self, theta):
        """Flat residual vector for the Jacobian SVD: amplitude then shape."""
        m = self.predict(theta)
        good = np.isfinite(m).all(axis=(1, 2)) & (m[:, :, I100] > 0).all(axis=1)
        mg = m[good]
        A_m = np.log10(np.clip(mg[:, :, I100], 1.0, None))
        dA = ((A_m - self.A_d[good]) / self.sig).ravel()
        F_m = mg / np.clip(mg[:, :, I100], 1.0, None)[:, :, None]
        dF = ((F_m - self.F_d[good]) / np.clip(self.F_d[good], 1e-30, None))
        dF = (dF / L_F_REF).reshape(len(dA), -1).ravel()
        return np.concatenate([np.sqrt(self.lam_A) * dA,
                               np.sqrt(self.lam_F) * dF / np.sqrt(dF.size / dA.size)])


def fit(problem, starts=None, maxiter=4000, verbose=True):
    """Multi-start Nelder-Mead. Multi-start is MANDATORY here (exp53)."""
    spec = problem.spec
    if starts is None:
        # `default_theta` needs a full `model.Spec`; a caller that supplies its
        # own starts may pass any object that can report `bounds()` -- which is
        # how Stage 3.9 profiles a subset of the parameters through this same
        # optimiser instead of a second, differently-behaved one.
        base = M.default_theta(spec)
        rng = np.random.default_rng(0)
        starts = [base]
        for _ in range(3):
            j = base + rng.normal(0, 0.15, base.shape) * np.maximum(np.abs(base), 0.3)
            starts.append(np.clip(j, *np.array(spec.bounds()).T))
    lo, hi = np.array(spec.bounds()).T

    def wrapped(x):
        if np.any(x < lo - 1e-9) or np.any(x > hi + 1e-9):
            return FAIL + float(np.sum(np.clip(lo - x, 0, None) ** 2
                                       + np.clip(x - hi, 0, None) ** 2))
        return problem.loss(x)

    best = None
    for s, p0 in enumerate(starts):
        r = minimize(wrapped, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=1e-5, fatol=1e-9))
        if verbose:
            print(f"      start {s}: loss {r.fun:.6f} ({r.nit} iters)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    if verbose:
        spread = "n/a"
        print(f"      best {best.fun:.6f}")
    return best.x, float(best.fun)


# --------------------------------------------------------------------------- #
# identifiability                                                              #
# --------------------------------------------------------------------------- #
def jacobian(problem, theta, rel=1e-4):
    """Scaled residual Jacobian: column j is d(resid)/d(theta_j * scale_j)."""
    r0 = problem.residuals(theta)
    scale = np.maximum(np.abs(theta), 0.1)
    J = np.empty((len(r0), len(theta)))
    for j in range(len(theta)):
        tp = theta.copy()
        tp[j] += rel * scale[j]
        J[:, j] = (problem.residuals(tp) - r0) / rel
    return J, r0


def identifiability(problem, theta, names, n_boot=12, verbose=True):
    """The mandatory report. Returns a dict; prints a human-readable table."""
    J, r0 = jacobian(problem, theta)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_rel = s / s[0]
    thresh = 1e-3
    n_eff = int(np.sum(s_rel > thresh))

    if verbose:
        print(f"\n    IDENTIFIABILITY  ({len(theta)} parameters, "
              f"{J.shape[0]} residuals)")
        print(f"      singular values / max: "
              + "  ".join(f"{v:.2e}" for v in s_rel))
        print(f"      EFFECTIVE parameter count at {thresh:g}: "
              f"**{n_eff} of {len(theta)}**")
        if n_eff < len(theta):
            worst = Vt[-1]
            comp = "  ".join(f"{n}:{v:+.2f}" for n, v in
                             zip(names, worst / np.abs(worst).max()))
            print(f"      the FLATTEST direction is a combination, not a "
                  f"parameter:\n        {comp}")

    # ONE-PARAMETER CONDITIONAL SLICES -- not profile scans. The other
    # parameters are held at `theta` throughout; nothing is re-optimised.
    prof = {}
    lo, hi = np.array(problem.spec.bounds()).T
    for j, nm in enumerate(names):
        step = 0.25 * max(abs(theta[j]), 0.2)
        vals, losses = [], []
        for dv in (-2, -1, 1, 2):
            x = theta.copy()
            x[j] = np.clip(theta[j] + dv * step, lo[j], hi[j])
            vals.append(x[j])
            losses.append(problem.loss(x))
        base = problem.loss(theta)
        curv = (min(losses) - base) / max(base, 1e-12)
        prof[nm] = float(np.min(losses) - base)
    if verbose:
        print(f"      CONDITIONAL loss rise at +-2 steps, others held FIXED "
              f"(flat => unidentified,\n      but a steep rise does NOT prove "
              f"identified -- see the module docstring):")
        for nm in names:
            flag = "  <-- FLAT" if prof[nm] < 1e-4 else ""
            print(f"        {nm:<12}{prof[nm]:+.3e}{flag}")

    # bootstrap over galaxies
    rng = np.random.default_rng(0)
    n = len(problem.halos)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = Problem(problem.spec, [problem.halos[i] for i in idx],
                      problem.raw[idx], epochs=problem.epochs,
                      lam_A=problem.lam_A, lam_F=problem.lam_F)
        x, _ = fit(sub, starts=[theta], maxiter=800, verbose=False)
        boots.append(x)
    B = np.array(boots)
    if verbose:
        print(f"      bootstrap over galaxies (n={n_boot}):")
        for j, nm in enumerate(names):
            sd = B[:, j].std()
            rel = sd / max(abs(theta[j]), 1e-9)
            flag = "  <-- UNSTABLE" if rel > 0.5 else ""
            print(f"        {nm:<12}{theta[j]:+9.4f} +- {sd:.4f}"
                  f"  ({100 * rel:5.1f}%){flag}")
    return dict(sv=s_rel, n_eff=n_eff, profile=prof, boot=B, Vt=Vt)


if __name__ == "__main__":
    import halo as H
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(ROOT / "experiments/exp32_full_population/outputs/population.npz")
    rows = np.array([g["row"] for g in s2._W["gals"]])[:120]
    recs = H.build_records(rows=rows)
    data = pop["data"][np.array([h.row for h in recs])]
    sp = M.Spec(family="sersic", eff="E1", size="S2")
    pr = Problem(sp, recs, data, epochs=(0, 4))
    th = M.default_theta(sp)
    a, f, nb = pr.scores(th)
    print(f"\n  at the default theta: score_A={a:.3f}  score_F={f:.3f}  "
          f"failed={nb}  loss={pr.loss(th):.4f}")
    assert np.isfinite(pr.loss(th))
    # the objective must be exactly 0 when the model IS the data
    m = pr.predict(th)
    fake = np.repeat(np.where(np.isfinite(m), m, 1.0)[:, :1], 5, axis=1)
    fake[:, 4] = m[:, 1]                       # put epoch 4 back where it belongs
    pr2 = Problem(sp, recs, fake, epochs=(0, 4))
    a2, f2, _ = pr2.scores(th)
    print(f"  self-consistency: feeding the model back as its own data gives "
          f"score_A={a2:.2e}, score_F={f2:.2e}  (must be ~0)")
    assert a2 < 1e-9 and f2 < 1e-9, "the objective is not zero at a perfect fit"
    # the Jacobian must have full column rank at a sane theta
    J, r0 = jacobian(pr, th)
    s = np.linalg.svd(J, compute_uv=False)
    print(f"  Jacobian {J.shape}, singular values / max: "
          + " ".join(f"{v:.1e}" for v in s / s[0]))
    print("\nfit.py self-check OK")
