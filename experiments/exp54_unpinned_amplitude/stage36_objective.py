"""exp54 Stage 3.6 — repairing the loss, one cause at a time.

THE DEFECT, MEASURED. The loss we fit splits each galaxy's curve of growth into
an amplitude, `log10 M*(<100 kpc)`, and a dimensionless shape, `M*(<R)` divided
by that same value, compared at 24 radii with fractional residuals. Perturb the
model's mass in one radial shell by +-1% and watch the loss move, and the shape
term turns out to be extraordinarily uneven about where it looks:

    at redshift 0.4, an error in the 132-148 kpc shell must be 175,000 times
    larger than one in the innermost 2 kpc to cost the loss the same amount

The cause is structural, not a weighting accident: dividing by the value at
100 kpc PINS the profile there, so the three shells beyond the pin are nearly
free. That is the blindness Section 10.2 of the technical note asked to repair.

**WHY exp48's WINNER CANNOT BE THE REPAIR.** The obvious candidate was exp48's
best objective, which compares surface-density profiles with a log residual.
Building a surface density means differencing the curve of growth between grid
radii, which gives 23 numbers from 24 radii: the mass inside the innermost
radius is never a difference and is dropped. Worse, adding mass inside that
radius raises every later cumulative value by the same constant, so every
difference is unchanged — the comparison is not merely insensitive there, it is
blind to floating-point noise. `hongshao.objective` now asserts this. That
aperture holds 12% of the stellar mass at redshift 0.4 and **32% at redshift
2**, and it is where the model's largest known defect lives. Adopting it would
trade the outer blindness for a worse central one.

**THE TWO CAUSES, VARIED ONE AT A TIME** (the user's call, and the more careful
design: a repair that changes two things at once cannot say which one worked).

  *the pin*        what divides out the amplitude: `M*(<100 kpc)`, or the
                   galaxy's total inside the last measured radius
  *the quantity*   what is compared: the cumulative profile with fractional
                   residuals, or the SHELL decomposition
                   `[M(<R_0), M(R_0..R_1), ...]` with log residuals

`shells` is new in `hongshao.objective` and is bijective with the curve of
growth, so unlike a density it drops nothing; a log residual then holds a shell
carrying 1% of the mass to the same relative accuracy as one carrying 30%. On
the same perturbation test its worst shell-to-shell ratio is 234 against the
production objective's 175,570.

    cell            pin            quantity              what it changes
    ------------------------------------------------------------------------
    production      M*(<100 kpc)   cumulative, frac      nothing: the CONTROL
    pin-total       total          cumulative, frac      the pin only
    shell-log       M*(<100 kpc)   shells, log           the quantity only
    shell-log-tot   total          shells, log           both
    density-log     M*(<100 kpc)   density, log          exp48's winner

**HOW THEY ARE JUDGED, and why it is not circular.** A model fitted under
objective B will of course score well on objective B. Every cell is therefore
reported on the SAME diagnostics, none of which is any cell's fitting loss for
more than one cell: the production shape error, the amplitude error, the median
error in each of the 24 shells, the outer surface-density slope, the halo-mass
tilt of the outer residual, and the predicted stellar-mass growth. This is the
same discipline Stage 3.5 needed and is the reason its negative result stands.

**THE WEIGHT THAT WOULD OTHERWISE BE UNDECLARED.** The total loss is
`lam_A * score_A^2 + lam_F * score_F^2`, and `score_F` is a raw objective value
divided by a reference so that both terms read about 1 at "as good as we
currently know how to do". The production reference, 0.158196, was measured for
the production objective; the other objectives have raw values on entirely
different scales, so dividing them by the same number would silently change how
hard the fit works on the amplitude. Each cell therefore gets its OWN reference,
fixed so that **the incumbent's fitted parameters score the same `score_F` in
every cell**. The amplitude term is identical in all five cells. Any difference
between cells is then the objective's shape across radius, which is the thing
under test, and not a rescaled weight against the amplitude.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage36_objective.py [--only CELL]
     [--smoke] [--starts 4]
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
import stage35_time_law as S35                           # noqa: E402
from hongshao.objective import Objective                 # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
SEL = HERE / "outputs" / "selection.npz"
OUT = HERE / "outputs" / "stage36_objective.npz"
FITDIR = HERE / "outputs" / "stage36_fits"

#: the shape is carried in solar masses at this pin mass; see `_norm`
_M_REF = 1.0e11
LABEL = "gompertz_log-E2-S2"
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
RULE = "=" * 100

#: (name, pin, Objective) -- `pin` is "anchor" for M*(<100 kpc), "total" for
#: the mass inside the last measured radius
CELLS = (
    ("production",    "anchor", Objective()),
    ("pin-total",     "total",  Objective()),
    ("shell-log",     "anchor", Objective(quantity="shells", residual="log")),
    ("shell-log-tot", "total",  Objective(quantity="shells", residual="log")),
    ("density-log",   "anchor", Objective(quantity="density", residual="log")),
)
CELL = {c[0]: c for c in CELLS}


class ObjectiveProblem(S35.StackedProblem):
    """`StackedProblem` with the shape term's pin and objective swapped.

    The amplitude term is untouched, so it is identical in every cell. Only two
    things vary: which radius divides out the amplitude before the comparison,
    and which `Objective` performs it.
    """

    def __init__(self, spec, halos, data, mask, pin="anchor", obj=None,
                 ref=None, **kw):
        super().__init__(spec, halos, data, mask, **kw)
        if pin not in ("anchor", "total"):
            raise ValueError(f"pin must be 'anchor' or 'total', got {pin!r}")
        self.pin, self.obj = pin, (obj or Objective())
        self.ref = F.L_F_REF if ref is None else float(ref)
        self.G_d = self._norm(self.data)

    def _pin_index(self):
        return F.I100 if self.pin == "anchor" else self.data.shape[-1] - 1

    def _norm(self, cum):
        """The amplitude divided out, but expressed in SOLAR MASSES.

        `hongshao.objective` floors a shell or an annulus at 1 Msun, which is a
        floor when the input is a real mass and a truncation when it is not: a
        profile normalised to 1 has every shell below the floor, so model and
        truth both clip to 1 and the residual collapses to exactly zero. This
        is not hypothetical — it is what the first version of this script did,
        and the reference calculation caught it by returning 0.
        So the shape is written as "the profile this galaxy would have if its
        pin mass were `_M_REF`". Identical to dividing through, since the
        fractional and log residuals are both scale-free, and `check_floor`
        asserts the floor is nowhere near biting.
        """
        return (cum / np.clip(cum[:, :, [self._pin_index()]], 1.0, None)
                * _M_REF)

    def check_floor(self, theta, margin=1e4):
        """The smallest SCORED shell either side must sit far above the floor.

        Only shells the mask admits can matter, so only those are checked. The
        distinction is not academic: row 181's measured profile is flat, and it
        stays in the data array after `sane_history_mask` removes it from the
        mask — checking the array rather than the mask reported a shell of
        2e-4 Msun that nothing was ever going to score.
        """
        sm = self._shells(self._norm(self.predict(theta)))
        sd = self._shells(self.G_d)
        m = self.mask[:, :, None]
        ok = lambda a: a[np.isfinite(a) & (a > 0) & np.broadcast_to(m, a.shape)]
        lo = min(float(np.min(ok(sm))), float(np.min(ok(sd))))
        if lo < margin:
            raise SystemExit(
                f"REFUSING TO RUN: the smallest shell is {lo:.3g} Msun, within "
                f"{margin:g} of the objective's 1 Msun floor, so the "
                f"comparison would be truncating rather than flooring.")
        return lo

    @staticmethod
    def _shells(cum):
        return np.concatenate([cum[:, :, :1], np.diff(cum, axis=2)], axis=2)

    def _parts(self, theta):
        m = self.predict(theta)
        i = self._pin_index()
        finite = np.isfinite(m).all(axis=2) & (m[:, :, F.I100] > 0) \
            & (m[:, :, i] > 0)
        use = self.mask & finite
        n_bad = int((self.mask & ~finite).sum())
        if use.sum() < 10:
            return np.nan, np.nan, n_bad
        A_m = np.log10(np.clip(m[:, :, F.I100], 1.0, None))
        score_A = float(np.sqrt(np.mean(((A_m - self.A_d) / self.sig)[use] ** 2)))
        G_m = self._norm(m)
        per_ep = [self.obj(G_m[use[:, j]][:, [j]], self.G_d[use[:, j]][:, [j]],
                           F.R_GRID)
                  for j in range(len(self.epochs)) if use[:, j].sum() >= 10]
        score_F = float(np.mean(per_ep)) / self.ref if per_ep else np.nan
        return score_A, score_F, n_bad

    def per_epoch(self, theta):
        m = self.predict(theta)
        i = self._pin_index()
        finite = np.isfinite(m).all(axis=2) & (m[:, :, F.I100] > 0) \
            & (m[:, :, i] > 0)
        use = self.mask & finite
        A_m = np.log10(np.clip(m[:, :, F.I100], 1.0, None))
        dA = (A_m - self.A_d) / self.sig
        G_m = self._norm(m)
        sa, sf = [], []
        for j in range(len(self.epochs)):
            g = use[:, j]
            sa.append(np.sqrt(np.mean(dA[g, j] ** 2)) if g.sum() else np.nan)
            sf.append(self.obj(G_m[g][:, [j]], self.G_d[g][:, [j]], F.R_GRID)
                      / self.ref if g.sum() else np.nan)
        return np.array(sa), np.array(sf)


def incumbent_theta():
    """The incumbent's fitted parameters, from the newest fit that exists.

    Prefers a Stage 3.5 fit on the CURRENT sample; falls back to the pre-row-181
    fits, and finally to Stage 3.3. This value only sets the starting point and
    the `score_F` references, both of which are declared, so the fallback is
    safe -- but which one was used must be visible, hence the print.
    """
    for path, key in ((HERE / "outputs" / "stage35_fits" / f"{LABEL}.npz", "theta"),
                      (HERE / "outputs" / "stage35_fits_PRE_ROW181_FIX"
                       / f"{LABEL}.npz", "theta"),
                      (HERE / "outputs" / "stage33_perepoch.npz",
                       f"{LABEL}_theta")):
        if path.exists():
            print(f"  incumbent parameters taken from {path.name}")
            return np.asarray(np.load(path)[key], float)
    raise SystemExit("no incumbent fit found; run stage35_time_law.py first")


def production_problem(spec, recs, data, mask):
    """The incumbent's own problem, used to report every cell on one metric."""
    return S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))


def references(spec, recs, data, mask, theta):
    """Each cell's `score_F` reference, so all five start balanced.

    Fixed so the incumbent's fitted parameters give the same `score_F` in every
    cell as they do under production. Without this, dividing a raw value on a
    different scale by the production reference would be an undeclared change
    to how the fit trades shape against amplitude.
    """
    prod = production_problem(spec, recs, data, mask)
    target = float(np.mean(prod.per_epoch(theta)[1]))
    out = {}
    for name, pin, obj in CELLS:
        pr = ObjectiveProblem(spec, recs, data, mask, pin=pin, obj=obj, ref=1.0,
                              epochs=(0, 1, 2, 3, 4))
        raw = float(np.mean(pr.per_epoch(theta)[1]))
        out[name] = raw / target          # an ABSOLUTE reference, like L_F_REF
    return out, target


# --------------------------------------------------------------------------- #
# the shared, objective-independent diagnostics                                #
# --------------------------------------------------------------------------- #
def shell_errors(pred, data, mask):
    """Median log10(model/truth) in each of the 24 shells, per epoch.

    The diagnostic no cell can game: it is not any of their fitting losses, and
    it is exactly where the blindness was measured.
    """
    def shells(c):
        return np.concatenate([c[:, :, :1], np.diff(c, axis=2)], axis=2)
    sm, sd = shells(pred), shells(data)
    out = np.full((5, data.shape[-1]), np.nan)
    for k in range(5):
        g = mask[:, k] & np.isfinite(sm[:, k, :]).all(1) & (sm[:, k, :] > 0).all(1)
        if g.sum() < 10:
            continue
        out[k] = np.median(np.log10(np.clip(sm[g, k, :], 1.0, None))
                           - np.log10(np.clip(sd[g, k, :], 1.0, None)), axis=0)
    return out


def outer_slope(pred, data, mask, rmin=52.0):
    """d log Sigma / d log R over `rmin`..148 kpc, model and truth, per epoch.

    Section 5.5's diagnostic. The model is known to be too shallow here and the
    loss is known not to feel it, so this is the number the repair is for.
    """
    from hongshao.profile_emulator import density_from_cog
    R = F.R_GRID
    out = np.full((5, 2), np.nan)
    for k in range(5):
        g = mask[:, k] & np.isfinite(pred[:, k, :]).all(1)
        if g.sum() < 10:
            continue
        for c, arr in enumerate((pred, data)):
            ls, mid = density_from_cog(
                np.log10(np.clip(arr[g, k, :], 1.0, None)), R)
            sel = mid >= rmin
            x = np.log10(mid[sel])
            out[k, c] = np.median([np.polyfit(x, y[sel], 1)[0] for y in ls])
    return out


def growth_report(pred, data, mask):
    """Predicted vs true growth in log10 M*(<100 kpc) from z=2 to z=0.4.

    On the galaxies scored at BOTH ends, which is the only set for which growth
    is defined.
    """
    g = mask[:, 0] & mask[:, 4] & np.isfinite(pred[:, [0, 4], :]).all((1, 2))
    i = F.I100
    gm = (np.log10(np.clip(pred[g, 0, i], 1.0, None))
          - np.log10(np.clip(pred[g, 4, i], 1.0, None)))
    gd = (np.log10(data[g, 0, i]) - np.log10(data[g, 4, i]))
    return dict(n=int(g.sum()), med_model=float(np.median(gm)),
                med_truth=float(np.median(gd)),
                sd_model=float(np.std(gm)), sd_truth=float(np.std(gd)),
                med_err=float(np.median(gm - gd)), sd_err=float(np.std(gm - gd)))


def report_cell(name, theta, spec, recs, data, mask, lmh, prod):
    """Every diagnostic, identical for every cell."""
    pr = ObjectiveProblem(spec, recs, data, mask, pin=CELL[name][1],
                          obj=CELL[name][2], epochs=(0, 1, 2, 3, 4))
    pred = pr.predict(theta)
    sa, sf_prod = prod.per_epoch(theta)
    shape_pct = 100.0 * float(np.mean(sf_prod)) * F.L_F_REF
    se = shell_errors(pred, data, mask)
    os_ = outer_slope(pred, data, mask)
    gr = growth_report(pred, data, mask)
    over, tilt, _ = S35.masked_diagnostics(pr, theta, lmh)
    return dict(name=name, theta=theta, sA=sa, sF_prod=sf_prod,
                shape_pct=shape_pct, shells=se, slope=os_, growth=gr,
                over=over, tilt=tilt, pred=pred)


def print_cell(r):
    se, os_ = r["shells"], r["slope"]
    R = F.R_GRID
    print(f"\n    SCORES on the PRODUCTION metric (the same for every cell)")
    print(f"      {'':<12}" + "".join(f"{f'z={z}':>10}" for z in Z))
    print(f"      {'amplitude':<12}" + "".join(f"{v:>10.4f}" for v in r["sA"]))
    print(f"      {'shape':<12}" + "".join(f"{v:>10.4f}" for v in r["sF_prod"]))
    print(f"      profile-shape error, mean over epochs: {r['shape_pct']:.3f}%")
    print(f"\n    MEDIAN log10(model/truth) IN EACH SHELL "
          f"(negative = model too light there)")
    print(f"      {'shell [kpc]':>16}" + "".join(f"{f'z={z}':>10}" for z in Z))
    lo = np.r_[0.0, R[:-1]]
    for j in (0, 3, 7, 11, 15, 19, 23):
        print(f"      {lo[j]:6.1f}-{R[j]:6.1f}  "
              + "".join(f"{se[k, j]:>10.3f}" for k in range(5)))
    print(f"      {'sum >52 kpc':>16}" + "".join(
        f"{np.nanmedian(se[k, R > 52]):>10.3f}" for k in range(5)))
    print(f"\n    OUTER SURFACE-DENSITY SLOPE, 52-148 kpc "
          f"(positive difference = model too shallow)")
    print(f"      {'':<12}" + "".join(f"{f'z={z}':>10}" for z in Z))
    print(f"      {'model':<12}" + "".join(f"{os_[k, 0]:>10.2f}" for k in range(5)))
    print(f"      {'truth':<12}" + "".join(f"{os_[k, 1]:>10.2f}" for k in range(5)))
    print(f"      {'difference':<12}" + "".join(
        f"{os_[k, 0] - os_[k, 1]:>+10.2f}" for k in range(5)))
    g = r["growth"]
    print(f"\n    GROWTH in log10 M*(<100 kpc), z=2 -> z=0.4, on {g['n']} galaxies")
    print(f"      model {g['med_model']:+.3f} +- {g['sd_model']:.3f} dex   "
          f"truth {g['med_truth']:+.3f} +- {g['sd_truth']:.3f} dex   "
          f"per-galaxy error {g['med_err']:+.3f} +- {g['sd_err']:.3f}")
    print(f"    OUTER RESIDUAL at 148 kpc: median {r['over']:+.4f} dex, "
          f"halo-mass tilt {r['tilt']:+.4f} dex/dex")


# --------------------------------------------------------------------------- #
def main(smoke=False, n_starts=4, only=None):
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    cuts = np.load(SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    # The completeness cut is not the only condition on a scored galaxy-epoch:
    # the measured history must also be physical. `sane_history_mask` was in
    # `selection.py`'s diagnostics but not in this fitting path, which let one
    # broken cross-match (row 181) carry 18% of the loss; see that function.
    complete_all = (np.isfinite(m200) & (m200 >= cuts[None, :])
                    & S.sane_history_mask(pop["data"]))

    rows = all_rows[::24] if smoke else all_rows
    recs = H.build_records(rows=rows, verbose=False)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    lmh = pop["logmh_zk_diffmah"][sel][:, 0]
    mask = complete_all[sel]
    spec = S33.spec_from_label(LABEL)
    th_inc = incumbent_theta()
    names = [only] if only else [c[0] for c in CELLS]

    print(f"exp54 STAGE 3.6 — repairing the loss, one cause at a time")
    print(f"  model    : {LABEL}, {spec.n_theta} parameters, held fixed")
    print(f"  sample   : {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs")
    print(f"  cells    : {', '.join(names)}\n")

    prod = production_problem(spec, recs, data, mask)
    # THE GATE. The production cell must reproduce the incumbent's own loss
    # exactly, or its objective is not the one every previous number was fitted
    # under and no cell can be compared with anything already reported.
    refs, target = references(spec, recs, data, mask, th_inc)
    p0 = ObjectiveProblem(spec, recs, data, mask, pin="anchor", obj=Objective(),
                          ref=F.L_F_REF, epochs=(0, 1, 2, 3, 4))
    a, b = p0.loss(th_inc), prod.loss(th_inc)
    print(f"  GATE: the production cell against the incumbent's own problem")
    print(f"    loss {a:.12f} vs {b:.12f}   |d| = {abs(a - b):.2e}")
    if abs(a - b) > 1e-10 or abs(refs["production"] / F.L_F_REF - 1) > 1e-9:
        raise SystemExit(
            "REFUSING TO RUN: the production cell does not reproduce the "
            "incumbent's loss, so no cell here can be compared with any number "
            "already reported. Fix the cell, not the tolerance.")
    lo = p0.check_floor(th_inc)
    print(f"    the smallest shell on either side is {lo:.3g} Msun, "
          f"{lo:.0e} times the objective's 1 Msun floor")
    print(f"    OK — and the reference this script derives for it comes back "
          f"at\n         {refs['production']:.10f} against the production "
          f"{F.L_F_REF:.6f}, so the recipe is consistent\n")
    print(f"  score_F REFERENCES, each fixed so the incumbent scores "
          f"{target:.4f} in every cell:")
    for n in [c[0] for c in CELLS]:
        print(f"    {n:<16}{refs[n]:.6f}"
              + ("   (= the production reference)" if n == "production" else ""))
    print()

    done = {}
    if not smoke:
        FITDIR.mkdir(exist_ok=True)
        for f in sorted(FITDIR.glob("*.npz")):
            z = np.load(f)
            done[str(z["name"])] = dict(theta=z["theta"], loss=float(z["loss"]))
        if done:
            print(f"  resuming: {len(done)} fits already on disk\n")

    results = []
    for name in names:
        _, pin, obj = CELL[name]
        print(f"\n{RULE}\n{name}   pin={pin}   {obj.name}\n{RULE}")
        pr = ObjectiveProblem(spec, recs, data, mask, pin=pin, obj=obj,
                              ref=refs[name], epochs=(0, 1, 2, 3, 4))
        if name in done:
            th, lo_ = np.asarray(done[name]["theta"]), done[name]["loss"]
            print(f"    loss {lo_:.6f}  (cached)")
        else:
            st = S35.starts_for(spec, n_starts) + [np.asarray(th_inc, float)]
            t0 = time.time()
            th, lo_ = F.fit(pr, starts=st, maxiter=600 * spec.n_theta,
                            verbose=True)
            print(f"    fitted in {(time.time() - t0) / 60:.1f} min from "
                  f"{len(st)} starts, loss {lo_:.6f}", flush=True)
            if not smoke:
                FITDIR.mkdir(exist_ok=True)
                np.savez(FITDIR / f"{name}.npz", name=name, theta=th, loss=lo_)
        print(f"\n    PARAMETERS  " + "  ".join(
            f"{n}={v:+.4f}" for n, v in zip(spec.theta_names, th)))
        r = report_cell(name, th, spec, recs, data, mask, lmh, prod)
        r["loss"] = lo_
        print_cell(r)
        results.append(r)

    if smoke or only:
        print(f"\n  {'smoke' if smoke else '--only'} run: writing nothing")
        return results
    np.savez_compressed(
        OUT, names=np.array([r["name"] for r in results]), rows=sel, mask=mask,
        refs=np.array([refs[r["name"]] for r in results]),
        **{f"theta::{r['name']}": r["theta"] for r in results},
        **{f"sA::{r['name']}": r["sA"] for r in results},
        **{f"sF::{r['name']}": r["sF_prod"] for r in results},
        **{f"shells::{r['name']}": r["shells"] for r in results},
        **{f"slope::{r['name']}": r["slope"] for r in results},
        shape_pct=np.array([r["shape_pct"] for r in results]),
        tilt=np.array([r["tilt"] for r in results]),
        over=np.array([r["over"] for r in results]),
        loss=np.array([r["loss"] for r in results]))
    print(f"\n  wrote {OUT}")
    return results


if __name__ == "__main__":
    ns = (int(sys.argv[sys.argv.index("--starts") + 1])
          if "--starts" in sys.argv else 4)
    on = (sys.argv[sys.argv.index("--only") + 1]
          if "--only" in sys.argv else None)
    main(smoke="--smoke" in sys.argv, n_starts=ns, only=on)
