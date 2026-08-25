"""exp54 Stage 3.8 — a second deposit channel with its own radial scale.

Plan: `doc/plans/2026-08-25-exp54-stage38-two-channel.md`.

THE ONE THING THIS STAGE TESTS. Stage 3.7 established that with ONE radial
scale per deposit, the central mass at redshift 2 and the outer mass at
redshift 2 cannot both be right: a free shared size law fitted to reduce the
central error's epoch swing gets it from 0.145 to 0.061 dex, and pays 0.104 dex
of outskirt mass at the same epoch, taking the model from 2% to 23% too light
beyond 52 kpc.

**A second channel with its own scale should break that trade.** If it cannot,
the deposition picture itself is the limit rather than its parameterisation.

WHAT IS ADDED. Each deposit's unit profile becomes a mixture weighted by WHEN
the mass was laid down:

    F_j = (1 - w_j) * F_extended(R50_j)  +  w_j * F_compact(rho * R50_j)

with `F_compact` an exponential (Sersic n = 1, no free shape) at a fraction
`rho < 1` of the extended channel's radius. Both are truncated at the same
boundary and are individually unit-mass, so the mixture is unit-mass and Stage
0's mass-conservation contract is untouched. `w = 0` reproduces the
single-channel model bit for bit, which `model.py` asserts at random thetas AND
random `rho`.

  `P1`  w constant                              2 extra: w0, log_rho
  `P2`  logistic in ln(1+z), transition at z=2  3 extra: w0, w_s, log_rho
  `P3`  free logistic                           4 extra: w0, w_zt, w_s, log_rho

`P1` is the control for the other two: if a CONSTANT compact fraction does as
well as a time-varying one, the time dependence has not earned itself.

TWO THINGS THIS IS NOT, recorded so they are not conflated later. It is not two
components inside every deposit — what varies is the mixing weight with
deposition time. And it is not the redshift-dependent deposit SHAPE (`c_z`)
that exp54 already rejected: that drifted one family's shape parameter, this
switches between two families at two different sizes, which `c_z` could not
express at any parameter value.

WHY AN EXPONENTIAL. A parallel experiment (exp55) finds an exponential core
plus an extended Moffat reproduces measured curves of growth as well as a
three-component decomposition, and connects them to halo growth AT A SINGLE
EPOCH. This model serves five, so that conclusion does not transfer — only the
hint about the inner component's shape. The exponential is the first thing to
try, not a result.

HOW IT IS JUDGED — and the criterion is not the loss. Each rung is fitted
twice: once under the production loss, which answers "would we adopt it", and
once to minimise the central span directly with the loss held within 5% of the
incumbent's, which answers the structural question:

    can the mixture reach a central span near 0.06 dex WHILE the error beyond
    52 kpc stays near the incumbent's -0.001 to -0.008 dex?

Everything is also reported on the NON-DECLINING galaxies separately. Stage 3.4
found that 53% of these objects have their measured mass inside 4.9 kpc FALL
with time, which no deposition-only model can follow, so a compact channel can
only be the right mechanism for the other 47%.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage38_two_channel.py [--only LABEL]
     [--span] [--smoke] [--starts 4]
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
import model as M                                        # noqa: E402
import stage32 as S32                                    # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage36_objective as S36                          # noqa: E402
import stage37_size_epoch as S37                         # noqa: E402

OUT = HERE / "outputs" / "stage38_two_channel.npz"
FITDIR = HERE / "outputs" / "stage38_fits"
BASE = S37.BASE
RUNGS = (f"{BASE}+P1expo", f"{BASE}+P2expo", f"{BASE}+P3expo")
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: Stage 3.4's preregistered radius for "the central mass declined"
DECLINE_R = 4.92
RULE = "=" * 100


def decline_flag(data):
    """Did the measured mass at 4.92 kpc FALL between redshift 2 and 0.4?

    Transcribed from `stage34_ceiling.decline_flag` so the split is the same
    one Stage 3.4 preregistered, not a new one invented here.
    """
    c = int(np.argmin(np.abs(F.R_GRID - DECLINE_R)))
    return data[:, 0, c] < data[:, 4, c]


def report(label, theta, recs, data, mask, lmh, base_spec, prod, dec):
    """Every diagnostic, plus the same split by whether the centre declined."""
    r = S37.report_one(label, theta, recs, data, mask, lmh, base_spec, prod)
    spec = S33.spec_from_label(label)
    pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    err = S37.central_error(pr.predict(theta), data, mask)
    for tag, sel in (("dec", dec), ("nodec", ~dec)):
        cen = np.array([np.nanmedian(err[sel, k]) for k in range(5)])
        r[f"central_{tag}"] = cen
        r[f"span_{tag}"] = float(cen[0] - cen[4])
    if spec.compact:
        w = M.compact_weight(spec, theta, recs[0])
        zz = np.asarray(recs[0].z)
        r["w_at"] = np.array([float(np.interp(z, zz[::-1], w[::-1]))
                              for z in Z])
        r["rho"] = float(10.0 ** theta[-1])
    return r


def show(r, base=None):
    R = F.R_GRID
    print(f"\n    CENTRAL ERROR inside 2 kpc, median log10(model/truth)")
    print(f"      {'':<24}" + "".join(f"{f'z={z}':>9}" for z in Z)
          + f"{'span':>9}")
    for tag, key in (("all galaxies", "central"),
                     ("centre DECLINED (53%)", "central_dec"),
                     ("centre did NOT decline", "central_nodec")):
        v = r[key]
        sp_ = r["span"] if key == "central" else r[f"span_{key.split('_')[-1]}"]
        print(f"      {tag:<24}" + "".join(f"{x:>9.3f}" for x in v)
              + f"{sp_:>9.3f}")
    print(f"      {'beyond 52 kpc':<24}" + "".join(
        f"{np.nanmedian(r['shells'][k, R > 52]):>9.3f}" for k in range(5)))
    if base is not None:
        print(f"      {'  the incumbent was':<24}" + "".join(
            f"{np.nanmedian(base['shells'][k, R > 52]):>9.3f}" for k in range(5)))
    if "w_at" in r:
        print(f"\n      compact fraction w at each epoch's own redshift: "
              + "  ".join(f"{v:.3f}" for v in r["w_at"])
              + f"    size ratio rho = {r['rho']:.3f}")
    print(f"      profile-shape error {r['shape_pct']:.3f}%   tilt "
          f"{r['tilt']:+.4f}   ident {r['ident']:.1e}   loss {r['loss']:.6f}")


def main(labels=None, n_starts=4, smoke=False, span=False, fit_only=False):
    recs, sel, data, mask, lmh, cuts = S37.build(smoke)
    base_spec = S33.spec_from_label(BASE)
    th_inc = S37.incumbent()
    prod = S35.StackedProblem(base_spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    dec = decline_flag(data)
    labels = list(labels or RUNGS)

    print(f"exp54 STAGE 3.8 — a second deposit channel with its own scale")
    print(f"  {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs; "
          f"{100 * dec.mean():.0f}% have a DECLINING centre at "
          f"{DECLINE_R:.2f} kpc")
    print(f"  incumbent loss {prod.loss(th_inc):.6f}")
    print(f"  mode: {'MINIMISE THE SPAN (the structural test)' if span else 'fit the production loss (would we adopt it)'}\n")

    base_r = report(BASE, th_inc, recs, data, mask, lmh, base_spec, prod, dec)
    base_r["loss"] = prod.loss(th_inc)
    print(f"{RULE}\n{BASE}   the incumbent\n{RULE}")
    show(base_r)

    results = [base_r]
    for label in labels:
        spec = S33.spec_from_label(label)
        print(f"\n{RULE}\n{label}   {spec.n_theta} parameters: "
              f"{', '.join(spec.theta_names)}\n{RULE}")
        tag = "span" if span else "loss"
        f_out = FITDIR / f"{label}__{tag}.npz"
        if f_out.exists():
            th = np.asarray(np.load(f_out)["theta"], float)
            print(f"    (cached)")
        elif span:
            th = S37.span_bound(label, n_starts=n_starts, smoke=smoke)[1]
        else:
            pr = S35.StackedProblem(spec, recs, data, mask,
                                    epochs=(0, 1, 2, 3, 4))
            st = S35.starts_for(spec, n_starts) + [S37.assert_lift_nests(
                spec, th_inc, base_spec, recs, data, mask)]
            t0 = time.time()
            th, lo_ = F.fit(pr, starts=st, maxiter=600 * spec.n_theta,
                            verbose=True)
            print(f"    fitted in {(time.time() - t0) / 60:.1f} min, "
                  f"loss {lo_:.6f}", flush=True)
        if not smoke and not f_out.exists():
            FITDIR.mkdir(exist_ok=True)
            np.savez(f_out, label=label, theta=th, mode=tag)
        pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        r = report(label, th, recs, data, mask, lmh, base_spec, prod, dec)
        r["loss"] = pr.loss(th)
        print(f"    PARAMETERS  " + "  ".join(
            f"{n}={v:+.4f}" for n, v in zip(spec.theta_names, th)))
        show(r, base_r)
        results.append(r)

    if smoke or fit_only:
        print(f"\n  {'smoke' if smoke else '--only'}: writing nothing")
        return results

    R = F.R_GRID
    print(f"\n{RULE}\nDOES THE SECOND SCALE BREAK THE TRADE?\n{RULE}")
    print(f"  {'model':<26}{'span':>8}{'span(nodec)':>13}{'>52kpc z=2':>12}"
          f"{'shape%':>9}{'ident':>10}")
    for r in results:
        print(f"  {r['label'].replace(BASE, 'base'):<26}{r['span']:>8.3f}"
              f"{r['span_nodec']:>13.3f}"
              f"{np.nanmedian(r['shells'][4, R > 52]):>12.3f}"
              f"{r['shape_pct']:>9.3f}{r['ident']:>10.1e}")
    print(f"\n  For reference, Stage 3.7's one-scale size law fitted the same "
          f"way reached\n  span 0.061 with the outskirts at redshift 2 at "
          f"-0.112 — that is the trade to beat.")
    np.savez_compressed(
        OUT, labels=np.array([r["label"] for r in results]),
        mode="span" if span else "loss",
        **{f"theta::{r['label']}": r["theta"] for r in results},
        **{f"central::{r['label']}": r["central"] for r in results},
        **{f"central_nodec::{r['label']}": r["central_nodec"] for r in results},
        **{f"shells::{r['label']}": r["shells"] for r in results},
        span=np.array([r["span"] for r in results]),
        span_nodec=np.array([r["span_nodec"] for r in results]),
        shape_pct=np.array([r["shape_pct"] for r in results]),
        ident=np.array([r["ident"] for r in results]),
        loss=np.array([r["loss"] for r in results]))
    print(f"\n  wrote {OUT}")
    return results


if __name__ == "__main__":
    ns = (int(sys.argv[sys.argv.index("--starts") + 1])
          if "--starts" in sys.argv else 4)
    only = (sys.argv[sys.argv.index("--only") + 1]
            if "--only" in sys.argv else None)
    main(labels=[only] if only else None, n_starts=ns,
         smoke="--smoke" in sys.argv, span="--span" in sys.argv,
         fit_only=bool(only))
