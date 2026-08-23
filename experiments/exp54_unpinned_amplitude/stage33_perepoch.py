"""exp54 Stage 3.3 — the fair-sample fit done WITHOUT the assembly bias.

WHAT THIS REPAIRS. `stage33_refit.py` built its fair sample as the INTERSECTION
of the five per-epoch completeness cuts: galaxies above the cut at every epoch.
That removes the progenitor-selection bias and introduces an assembly bias in
its place, because requiring a halo to be massive at every epoch selects early
assemblers. Measured at fixed z = 0.4 halo mass (13.2-13.6, n = 987), the
intersection has `t50` = 4.14 Gyr against 5.59 for the rest -- 1.45 Gyr
earlier, Mann-Whitney p = 2e-53 -- and `fz2` = 0.350 against 0.175, p = 1e-83.

The repair is to stop taking an intersection. Each epoch is scored on ITS OWN
fair galaxies: 2397 at z = 0.4, then 1781 / 1436 / 1145 / 840 -- **7599
galaxy-epochs against the intersection's 751 x 5 = 3755**, and no galaxy is
required to be massive at an epoch where it is not being scored.

HOW, WITHOUT TOUCHING SHARED CODE. `hongshao.objective.Objective.per_galaxy`
marks a galaxy bad if its residual is non-finite at ANY epoch, so NaN-masking
one epoch would discard the galaxy everywhere. Instead the shape loss is
computed one epoch at a time on that epoch's galaxies and averaged, which is
IDENTICAL to the unmasked computation when the galaxy sets coincide -- mean
over galaxies of a mean over epochs is the mean over epochs of a mean over
galaxies. `MaskedProblem` therefore subclasses `fit.Problem` and is checked
against it: with an all-true mask the two must agree to machine precision, and
the run refuses to proceed if they do not.

WHAT IT ANSWERS that the intersection could not:

  1. **How the model actually does on a fair sample**, epoch by epoch, without
     the early-assembler contamination. The intersection's `score_A` values
     could not be quoted as this.
  2. **Whether the S3 collapse is real.** In `moffat-E5-S2+S3` all three size-
     law conditioning slopes went to ~0 on the intersection
     (`f:logMh` 0.099 -> 0.003, `f:dlogc` -0.338 -> -0.017,
     `f:fform` -0.334 -> +0.004). If S3 was absorbing selection structure they
     should stay near zero here; if it was an artifact of the intersection's
     narrow assembly range they should come back.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage33_perepoch.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402
import stage33 as S33                                    # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
SEL = HERE / "outputs" / "selection.npz"
REFIT = HERE / "outputs" / "stage33_refit.npz"
OUT = HERE / "outputs" / "stage33_perepoch.npz"
PARTIAL = HERE / "outputs" / "stage33_perepoch_partial.npz"

#: the adopted model, and the one whose S3 slopes collapsed
LABELS = ("gompertz_log-E2-S2", "moffat-E5-S2+S3")
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
RULE = "=" * 96


class MaskedProblem(F.Problem):
    """`fit.Problem` with a PER-EPOCH galaxy mask.

    `mask` is (n_galaxies, n_epochs) and says which galaxies count at which
    epoch. Everything else -- the model, the objective, the benchmarks -- is
    inherited unchanged, so the only difference from the parent is which
    residuals enter the two sums.
    """

    def __init__(self, spec, halos, data, mask, **kw):
        super().__init__(spec, halos, data, **kw)
        self.mask = np.asarray(mask, bool)[:, self.epochs]
        if self.mask.shape != (len(halos), len(self.epochs)):
            raise ValueError(f"mask must be (n_galaxies, n_epochs), got "
                             f"{self.mask.shape}")

    def _parts(self, theta):
        """(score_A, score_F, n_bad) under the mask."""
        m = self.predict(theta)
        finite = np.isfinite(m).all(axis=2) & (m[:, :, F.I100] > 0)
        use = self.mask & finite
        n_bad = int((self.mask & ~finite).sum())
        if use.sum() < 10:
            return np.nan, np.nan, n_bad
        A_m = np.log10(np.clip(m[:, :, F.I100], 1.0, None))
        dA = (A_m - self.A_d) / self.sig
        score_A = float(np.sqrt(np.mean(dA[use] ** 2)))
        F_m = m / np.clip(m[:, :, F.I100], 1.0, None)[:, :, None]
        # one epoch at a time, on that epoch's galaxies, then averaged --
        # identical to the unmasked call when the sets coincide
        per_ep = []
        for j in range(len(self.epochs)):
            g = use[:, j]
            if g.sum() < 10:
                continue
            per_ep.append(self.obj(F_m[g][:, [j]], self.F_d[g][:, [j]],
                                   F.R_GRID))
        score_F = float(np.mean(per_ep)) / F.L_F_REF if per_ep else np.nan
        return score_A, score_F, n_bad

    def scores(self, theta):
        return self._parts(theta)

    def per_epoch(self, theta):
        m = self.predict(theta)
        finite = np.isfinite(m).all(axis=2) & (m[:, :, F.I100] > 0)
        use = self.mask & finite
        A_m = np.log10(np.clip(m[:, :, F.I100], 1.0, None))
        dA = (A_m - self.A_d) / self.sig
        F_m = m / np.clip(m[:, :, F.I100], 1.0, None)[:, :, None]
        sa, sf = [], []
        for j in range(len(self.epochs)):
            g = use[:, j]
            sa.append(np.sqrt(np.mean(dA[g, j] ** 2)) if g.sum() else np.nan)
            sf.append(self.obj(F_m[g][:, [j]], self.F_d[g][:, [j]], F.R_GRID)
                      / F.L_F_REF if g.sum() else np.nan)
        return np.array(sa), np.array(sf)

    def loss(self, theta):
        self.n_eval += 1
        a, f, n_bad = self._parts(theta)
        if not (np.isfinite(a) and np.isfinite(f)):
            return F.FAIL
        return (self.lam_A * a ** 2 + self.lam_F * f ** 2
                + F.FAIL * n_bad / max(self.mask.sum(), 1))


def check_equivalence(spec, recs, data):
    """With an all-true mask the subclass MUST reproduce the parent exactly.

    This is the whole warrant for trusting the masked numbers: if the only
    change is which residuals are summed, then turning the mask off has to give
    the parent's answer back.
    """
    theta = M.default_theta(spec)
    p = F.Problem(spec, recs, data, epochs=(0, 1, 2, 3, 4))
    q = MaskedProblem(spec, recs, data,
                      np.ones((len(recs), 5), bool), epochs=(0, 1, 2, 3, 4))
    a0, f0, _ = p.scores(theta)
    a1, f1, _ = q.scores(theta)
    da, df = abs(a1 - a0), abs(f1 - f0)
    print(f"  EQUIVALENCE CHECK (all-true mask must reproduce fit.Problem)")
    print(f"    score_A {a0:.10f} vs {a1:.10f}   |d| = {da:.2e}")
    print(f"    score_F {f0:.10f} vs {f1:.10f}   |d| = {df:.2e}")
    if not (da < 1e-10 and df < 1e-10):
        raise SystemExit(
            "REFUSING TO RUN: the masked problem does not reproduce the parent "
            "with the mask off, so a masked number cannot be compared with any "
            "unmasked one. Fix the subclass, not the tolerance.")
    print(f"    OK — the subclass differs only in which residuals are summed\n")


def main(smoke=False):
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])

    if not SEL.exists():
        raise SystemExit("run selection.py first — the cuts come from there")
    cuts = np.load(SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    fair_all = np.isfinite(m200) & (m200 >= cuts[None, :])

    rows = all_rows[::2] if smoke else all_rows
    if smoke:
        rows = rows[::12]
    recs = H.build_records(rows=rows, verbose=False)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    lmh = pop["logmh_zk_diffmah"][sel]
    mask = fair_all[sel]

    print(f"exp54 STAGE 3.3 — PER-EPOCH fair mask (no intersection, no "
          f"assembly bias)")
    print(f"  SIGMA_A: " + " ".join(f"{v:.4f}" for v in F.SIGMA_A))
    print(f"  cuts    : " + " ".join(f"{c:.2f}" for c in cuts) + "  log10 M200c")
    print(f"  galaxies scored per epoch: " + " ".join(
        f"{int(v)}" for v in mask.sum(0)) + f"   total {int(mask.sum())} "
        f"galaxy-epochs")
    inter = int(mask.all(axis=1).sum())
    print(f"  (the intersection used by stage33_refit.py was {inter} galaxies "
          f"= {5 * inter} galaxy-epochs)\n")

    check_equivalence(S33.spec_from_label(LABELS[0]), recs, data)

    done = {}
    if PARTIAL.exists() and not smoke:
        z = np.load(PARTIAL, allow_pickle=True)
        done = {str(k): v.item() for k, v in z.items()}
        if done:
            print(f"  resuming: {len(done)} fits already complete\n")

    refit = np.load(REFIT, allow_pickle=True) if REFIT.exists() else None
    out = {}
    for label in (LABELS[:1] if smoke else LABELS):
        sp = S33.spec_from_label(label)
        print(f"\n{RULE}\n{label}   {sp.n_theta} parameters\n{RULE}")
        if label in done:
            th, lo = done[label]["theta"], done[label]["loss"]
            print(f"    loss {lo:.5f}  (cached)")
        else:
            pr = MaskedProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
            base = M.default_theta(sp)
            rng = np.random.default_rng(1)
            j = np.clip(base + rng.normal(0, 0.12, base.shape)
                        * np.maximum(np.abs(base), 0.3),
                        *np.array(sp.bounds()).T)
            t0 = time.time()
            th, lo = F.fit(pr, starts=[base, j], maxiter=600 * sp.n_theta,
                           verbose=False)
            print(f"    fitted in {(time.time() - t0) / 60:.1f} min, "
                  f"loss {lo:.5f}", flush=True)
            if not smoke:
                done[label] = dict(theta=th, loss=lo)
                np.savez(PARTIAL, **done)

        print(f"\n    PARAMETERS — three samples")
        names = sp.theta_names
        tf = refit[f"{label}_theta_full"] if refit is not None else None
        ti = refit[f"{label}_theta_fair"] if refit is not None else None
        print(f"      {'parameter':<12}{'full':>11}{'intersect':>11}"
              f"{'per-epoch':>11}{'per-ep - full':>15}")
        for i, nm in enumerate(names):
            a = tf[i] if tf is not None else np.nan
            b = ti[i] if ti is not None else np.nan
            print(f"      {nm:<12}{a:>11.4f}{b:>11.4f}{th[i]:>11.4f}"
                  f"{th[i] - a:>+15.4f}")

        pr = MaskedProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        sa, sf = pr.per_epoch(th)
        print(f"\n    SCORES on each epoch's own fair galaxies "
              f"(1.0 = the full-sample halo-only regression)")
        print(f"      {'':<10}" + "".join(f"{f'z={z}':>10}" for z in Z))
        print(f"      {'n':<10}" + "".join(f"{int(v):>10}" for v in mask.sum(0)))
        print(f"      {'score_A':<10}" + "".join(f"{v:>10.4f}" for v in sa))
        print(f"      {'score_F':<10}" + "".join(f"{v:>10.4f}" for v in sf))

        # and the full-sample theta scored under the same mask, for contrast
        if tf is not None:
            sa2, sf2 = pr.per_epoch(tf)
            print(f"      {'':<10}--- the FULL-fitted theta, same mask ---")
            print(f"      {'score_A':<10}" + "".join(f"{v:>10.4f}" for v in sa2))
            print(f"      {'score_F':<10}" + "".join(f"{v:>10.4f}" for v in sf2))

        pred = pr.predict(th)
        tilts = S33.tilt_report(pred, data, [0, 1, 2, 3, 4], lmh)
        print(f"\n    HALO-MASS TILT of the per-epoch-fitted theta, measured "
              f"on the FULL sample")
        print(f"      {'R [kpc]':>9}" + "".join(f"{f'z={z}':>22}" for z in Z))
        for rr, rowset in tilts.items():
            cells = [f"{v:+.3f} [{a:+.3f},{b:+.3f}]" for v, a, b in rowset]
            print(f"      {rr:>9.0f}" + "".join(f"{c:>22}" for c in cells))
        out[label] = dict(theta=th, loss=lo, sA=sa, sF=sf)

    if smoke:
        print("\n  smoke run: writing nothing")
        return out
    np.savez_compressed(
        OUT, labels=np.array(list(out)), rows=sel, mask=mask, cuts=cuts,
        sigma_a=F.SIGMA_A,
        **{f"{l}_{k}": v[k] for l, v in out.items()
           for k in ("theta", "loss", "sA", "sF")})
    print(f"\n  wrote {OUT}")
    PARTIAL.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
