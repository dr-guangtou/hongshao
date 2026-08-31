"""exp72 Step 2 — refit the exp63 model under the one objective Step 1 cleared.

Step 1 (`errors.py`) swept five candidate objectives at frozen theta and
disqualified four of them without spending a fit:

  chi2_gls / chi2_gls_amp UNFLOORED   one galaxy carried 100% of the population's
                                      loss at z = 0.4
  chi2_diag                           2-5x more concentrated than the production
                                      objective at every floor, and the most
                                      amplitude-dominated objective tested
  frac_sigw                           passes usability, but moves the
                                      amplitude-to-shape ratio from 26 to 41 at
                                      z = 2 -- the wrong direction

**`chi2_gls_amp` with the annulus error floored survives.** It is the only
candidate that keeps real shape sensitivity at z = 2 (0.7 per cent against
0.1-0.2), balances amplitude against shape within a factor 2.5-6 at every epoch,
passes the usability gate at z >= 0.7, and disagrees with the incumbent
objectives about the two real models. This script fits it.

WHAT IS HELD FIXED SO THE COMPARISON IS LIKE FOR LIKE. The same twelve
parameters, the same bounds, the same starts, the same sample rule, the same
engine and the same five epochs jointly as
`exp63/outputs/stage2_fit_joint_kpc_free_sane.npz`. **The objective is the only
thing that changes.** Anything else moving would make the result unreadable.

THE OBJECTIVE, stated exactly. Per galaxy and epoch,

    L = amp_scale[k] * (dlog10 M*(<103 kpc) / SIGMA_A[k])^2
      + sum_j [ (dM_ann,j of the PINNED profile) / max(sigma_ann,j, f sigma_cog,j) ]^2

the first term exp63's own amplitude score, the second a generalised
least-squares chi-square of the shape at fixed `M*(<103 kpc)`. Two constants,
both MEASURED and then FROZEN for the whole fit rather than tuned:

  `amp_scale[k]`  the shape half's median at the NESTED INCUMBENT, so the two
                  halves start equal there. Without it the amplitude half is
                  numerically invisible -- Step 1 measured a pure amplitude
                  error costing exactly 1.000.
  `f`             the floor on the annulus error, as a fraction of the
                  cumulative error at the same radius. Step 1's sweep: f = 0 is
                  unusable, f = 0.03-0.10 works, f = 0.30 over-smooths.

Each epoch's loss is divided by the nested incumbent's value at that epoch, so
every epoch contributes 1 at the null and the five-epoch total starts at 5.000.
That is exp63's own convention for its shells and binned terms, kept so the two
fits' loss numbers are read the same way.

TWO CAVEATS THIS FIT CARRIES, and every number it produces carries with it:
the error model is known to be too small (isophotal only, no projection term:
`doc/tng300_profile_data.md` section 11.1 measured reduced chi-square 3.4
instead of 1.1 at z = 0.4); and at z = 2 only 67 per cent of galaxies have a
positive measured error beyond 103 kpc, so this objective is quietest in the
outskirts of the epoch that fails.

Run (long; the MAIN shell, never a subagent -- a background subagent dies with
its host, and each start checkpoints):

    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. nohup uv run python -u \\
        experiments/exp72_error_objective/fit_gls.py [--smoke] [--floor 0.1] \\
        [--starts 0:6] > experiments/exp72_error_objective/outputs/fit_gls.log 2>&1 &
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402
from errors import (FLOOR_DEFAULT, RULE, load_matched,  # noqa: E402
                    per_galaxy, pin_at, terms)

OUTDIR = HERE / "outputs"
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)
#: the exp63 product this fit is the like-for-like counterpart of
EXP63_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
MAX_EVALS = {"smoke": 60, "full": 3000}


class GlsProblem:
    """One theta, five epochs, under `chi2_gls_amp`.

    Every constant the objective needs is measured once in `__init__` from the
    NESTED INCUMBENT and then frozen: the per-epoch normalisation, and the
    amplitude/shape balance. Nothing here is recomputed from the current theta,
    because an objective whose own scales move with the parameters is not an
    objective.
    """

    def __init__(self, spec2, curves, data, keep, sig_cog, sig_ann, ann_ok,
                 th_nested, floor=FLOOR_DEFAULT):
        self.spec2, self.floor = spec2, floor
        self.R = F.R_GRID
        self.keep = np.asarray(keep, bool)                       # (n, 5)
        self.rows = {k: np.where(self.keep[:, k])[0] for k in EPOCHS}
        self.all_rows = np.unique(np.concatenate(list(self.rows.values())))
        pos = {r: i for i, r in enumerate(self.all_rows)}
        self.index = {k: np.array([pos[r] for r in self.rows[k]]) for k in EPOCHS}
        self.curves = [curves[i] for i in self.all_rows]
        self.data = data[self.all_rows]
        self.sig_cog = sig_cog[self.all_rows]
        self.sig_ann = sig_ann[self.all_rows]
        self.ann_ok = ann_ok[self.all_rows]
        self.mask = self.keep[self.all_rows]
        self.n_eval = 0

        # --- the two frozen constants, measured at the nested incumbent ---- #
        null = M2.predict2(spec2, th_nested, self.curves, self.R, nodes=M2.FULL_NODES)
        shape_ref = terms("chi2_gls", pin_at(null, self.data), self.data,
                          self.sig_cog, self.sig_ann, self.ann_ok,
                          floor_frac=floor)[0].sum(axis=-1)
        self.amp_scale = np.array([
            np.nanmedian(np.where(self.mask[:, k], shape_ref[:, k], np.nan))
            for k in EPOCHS])
        self.ref = np.array([
            np.nanmean(np.where(self.mask[:, k],
                                self._raw(null)[:, k], np.nan)) for k in EPOCHS])

    def _raw(self, cogs):
        return per_galaxy("chi2_gls_amp", cogs, self.data, self.sig_cog,
                          self.sig_ann, self.ann_ok, floor_frac=self.floor,
                          amp_scale=self.amp_scale)[0]

    def per_epoch(self, theta, nodes=M2.FIT_NODES):
        """(5,) each epoch's loss, 1.0 at the nested incumbent by construction."""
        cogs = M2.predict2(self.spec2, theta, self.curves, self.R, nodes=nodes)
        bad = ~(np.isfinite(cogs).all(axis=(1, 2)) & (cogs > 0).all(axis=(1, 2)))
        raw = self._raw(cogs)
        out = np.full(5, np.nan)
        for k in EPOCHS:
            m = self.mask[:, k] & ~bad
            if m.sum() < 10:
                return np.full(5, np.nan), int(bad.sum())
            out[k] = np.nanmean(np.where(m, raw[:, k], np.nan)) / self.ref[k]
        return out, int(bad.sum())

    def loss(self, theta):
        self.n_eval += 1
        per, n_bad = self.per_epoch(theta)
        if not np.isfinite(per).all():
            return F.FAIL * len(EPOCHS)
        # the same failure penalty exp63 uses, so a theta that cannot build a
        # profile is walked away from rather than silently dropped
        return float(per.sum()) + F.FAIL * n_bad / len(self.curves)

    def report(self, theta, label, nodes=M2.FULL_NODES):
        per, n_bad = self.per_epoch(theta, nodes=nodes)
        print(f"  {label}: " + "  ".join(f"z={z}:{v:.4f}" for z, v in zip(ANCHOR_Z, per))
              + f"  -> total {np.nansum(per):.4f}"
              + (f"  ({n_bad} galaxies unbuildable)" if n_bad else ""))
        return per


# --------------------------------------------------------------------------- #
def build(smoke=False, floor=FLOOR_DEFAULT):
    """Everything the fit needs, and the sample printed as the repo rule wants."""
    D = load_matched()
    recs, data, mask_legacy, lmh, _, th_inc, _ = S0.build(smoke)
    fitmask, _ = S2F.fit_masks(data, mask_legacy, lmh, "sane")
    sel = np.array([h.row for h in recs])
    curves = E.build_curves(recs, verbose=False)

    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    good &= fitmask.all(1)
    keep = good[:, None] & D["err_ok"][sel]
    print(f"  THE FITTING SAMPLE with a usable error: "
          + "/".join(f"{int(keep[:, k].sum())}" for k in EPOCHS)
          + f" per epoch, of {int(good.sum())} passing the sample rule alone")

    fz = np.load(EXP63_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    th_nested = np.asarray(fz["theta_nested"], float)
    print(f"  the same spec, bounds and starts as {EXP63_NPZ.name}; "
          f"THE OBJECTIVE IS THE ONLY CHANGE")
    return (recs, data, lmh, curves, sel, spec2, th_inc, th_nested, fz,
            keep, D["sigma_cog"][sel], D["sigma_ann"][sel], D["ann_ok"][sel])


def main(smoke=False, floor=FLOOR_DEFAULT, starts_sel=None):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp72 Step 2 — the exp63 model refitted under chi2_gls_amp "
          f"(floor f = {floor})\n{RULE}\n")
    (recs, data, lmh, curves, sel, spec2, th_inc, th_nested, fz,
     keep, sig_cog, sig_ann, ann_ok) = build(smoke, floor)

    pr = GlsProblem(spec2, curves, data, keep, sig_cog, sig_ann, ann_ok,
                    th_nested, floor=floor)
    print(f"  joint fit at all five epochs, union {len(pr.all_rows)} galaxies")
    print(f"  amplitude/shape balance (frozen, from the nested incumbent): "
          + " ".join(f"{v:.3g}" for v in pr.amp_scale))
    print(f"  per-epoch normalisation (frozen): "
          + " ".join(f"{v:.4g}" for v in pr.ref))

    # --- the gate every basin must pass: the null is exactly 1 per epoch --- #
    l_null = pr.loss(th_nested)
    per_null = pr.report(th_nested, "the null (nested incumbent, exact engine)")
    assert np.allclose(per_null, 1.0, atol=2e-2), per_null
    print(f"  GATE the nested start reproduces the incumbent and each epoch "
          f"normalises to 1.000 (max deviation "
          f"{np.max(np.abs(per_null - 1.0)):.1e})  OK")

    # where exp63's own theta lands under THIS objective, before fitting
    th_exp63 = np.asarray(fz["theta_best"], float)
    pr.report(th_exp63, "exp63's theta, scored under this objective")

    bounds = np.asarray(fz["bounds"], float).tolist()
    rng = np.random.default_rng(72)
    starts = S2F.starts_for(spec2, th_inc, rng)
    starts.append(("exp63", np.asarray(th_exp63)))
    if starts_sel is not None:
        starts = starts[starts_sel[0]:starts_sel[1] + 1]
    print(f"\n  {len(starts)} starts: " + ", ".join(n for n, _ in starts))

    results = []
    partial = OUTDIR / f"fit_gls{tag}.PARTIAL.npz"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    max_evals = MAX_EVALS["smoke" if smoke else "full"]
    for name, p0 in starts:
        t0 = time.time(); pr.n_eval = 0
        l0 = pr.loss(p0)
        r = minimize_loss(pr.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=max_evals, fd_step=1e-5)
        dt = time.time() - t0
        rl = S2F.railed(spec2, r.x, bounds=bounds)
        print(f"  start {name:<11} loss {l0:.6f} -> {r.fun:.6f}  "
              f"({pr.n_eval} evals, {dt / 60:.1f} min)"
              f"{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        results.append(dict(name=name, theta=np.asarray(r.x), loss=float(r.fun),
                            loss0=float(l0), railed=rl))
        np.savez(partial, names=np.array([q["name"] for q in results]),
                 thetas=np.array([q["theta"] for q in results]),
                 losses=np.array([q["loss"] for q in results]),
                 losses0=np.array([q["loss0"] for q in results]),
                 loss_null=l_null, theta_nested=th_nested, floor=floor,
                 amp_scale=pr.amp_scale, ref=pr.ref,
                 theta_names=np.array(spec2.theta_names))

    best = min(results, key=lambda q: q["loss"])
    print(f"\n  best start: {best['name']} loss {best['loss']:.6f} "
          f"(null {l_null:.6f}, {100 * (1 - best['loss'] / l_null):+.1f}%)")
    print("  " + "  ".join(f"{n}={v:+.4f}"
                           for n, v in zip(spec2.theta_names, best["theta"])))
    if best["railed"]:
        print(f"  RAILED at the bound: {', '.join(best['railed'])} — a bound is "
              f"usually the optimiser buying an over-weighted term, so read "
              f"these before the loss")
    pr.report(best["theta"], "best (full nodes)")

    # how many starts agree, which is the only evidence the basin is real
    ls = np.array([q["loss"] for q in results])
    n_agree = int((ls < best["loss"] + 0.01).sum())
    print(f"  {n_agree} of {len(results)} starts land within 0.01 of the best "
          f"loss{'  (a single basin)' if n_agree >= 3 else '  — NOT enough agreement to call the basin settled'}")

    out = OUTDIR / f"fit_gls{tag}.npz"
    np.savez(out, theta_best=best["theta"], loss_best=best["loss"],
             best_name=best["name"], loss_null=l_null, floor=floor,
             amp_scale=pr.amp_scale, ref=pr.ref, bounds=np.array(bounds),
             theta_nested=th_nested, theta_exp63=th_exp63,
             theta_names=np.array(spec2.theta_names),
             extended_family=spec2.extended_family, objective="chi2_gls_amp",
             compact_in_kpc=spec2.compact_in_kpc, fit_epoch=-1, fit_sample="sane",
             n_fit=np.array([len(pr.rows[k]) for k in EPOCHS]),
             fit_rows=pr.all_rows,
             names=np.array([q["name"] for q in results]),
             thetas=np.array([q["theta"] for q in results]),
             losses=ls, losses0=np.array([q["loss0"] for q in results]))
    print(f"\nwrote {out}")
    return best


if __name__ == "__main__":
    a = sys.argv
    fl = float(a[a.index("--floor") + 1]) if "--floor" in a else FLOOR_DEFAULT
    ss = None
    if "--starts" in a:
        lo, hi = a[a.index("--starts") + 1].split(":")
        ss = (int(lo), int(hi))
    main(smoke="--smoke" in a, floor=fl, starts_sel=ss)
