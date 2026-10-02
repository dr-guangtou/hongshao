"""exp87 Stage 7 (added 2026-10-02 at the user's question) — CROSS-EPOCH: what
does the z = 0.4 curve of growth know about the halo's mass at EARLIER epochs?

Target: log M200c of the main progenitor at z = 0.7, 1.0, 1.5, 2.0, for the
galaxies selected at z = 0.4. This is a well-posed population: "centrals
above 10^13 at z = 0.4, and the mass their main progenitor had at z_k". There
is no cut on the target, so the predictive is a plain normal; the sample is
the curated five-epoch sample (the progenitor masses exist only there).

Inputs compared, all scored out of fold on the same galaxies:
  nothing                      the sample's own distribution of the progenitor mass
  M*(<148) at z = 0.4          one number
  the z = 0.4 profile          24 shell masses, linear; and + quadratic
  the SAME-EPOCH profile       the curve of growth at z_k itself (the Stage 3 reference)
  both profiles                z = 0.4 and z_k stacked
  true Mh(z = 0.4)             ORACLE: how much of the progenitor mass is just the final mass
  true Mh(z = 0.4) + profile   ORACLE: what the z = 0.4 profile adds about the GROWTH
  mediated                     the z = 0.4 profile's own estimate of Mh(z = 0.4), then
                               the progenitor mass from that estimate alone

The lockbox is not touched (it was scored once, in Stage 6).

Run:  PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage7_cross_epoch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402
import harness as HN                                     # noqa: E402
import methods as M                                      # noqa: E402
import scoring as S                                      # noqa: E402

RULE = "=" * 100


class Mediated(M.Method):
    """The progenitor mass through the profile's own estimate of the z = 0.4
    mass ONLY. The design's last column is the true z = 0.4 mass, used to train
    the estimate and never read for the test galaxies."""
    rung = "L1"

    def fit_predict(self, X_train, y_train, X_test, cut):
        cog, mass = X_train[:, :-1], X_train[:, -1]
        estimate = np.empty(len(mass))
        for a, b in M.inner_folds(len(mass)):
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(a)), cog[a]]), mass[a], rcond=None)[0]
            estimate[b] = cog[b] @ coef[1:] + coef[0]
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(mass)), cog]), mass, rcond=None)[0]
        estimate_test = X_test[:, :-1] @ coef[1:] + coef[0]
        inner = M.DirectLinear("poly", "L0", transform=M._poly_only)
        return inner.fit_predict(estimate[:, None], y_train, estimate_test[:, None], cut)


def quad():
    return M.DirectLinear("linear+pca6-quad", "L2", transform=(lambda: M._LinearPlusQuadratic(6)), ridge_grid=M.RIDGE_GRID)


def ridge():
    return M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)


def main():
    delta = float(json.loads((C.OUTDIR / "gate_mechanics.json").read_text())["delta"])
    print(f"{RULE}\nexp87 STAGE 7 — the z = 0.4 curve of growth and the halo's EARLIER mass (config {C.config_hash()})\n{RULE}")
    s0 = D.load_curated(0, verbose=False)
    shells0, _ = D.feature_set(s0, "shells24")
    mtot0 = s0.logcog[:, [-1]]
    mh0 = s0.targets["mh"]
    out = {}
    for k in range(1, 5):
        sk = D.load_curated(k, verbose=False)
        assert np.array_equal(sk.index, s0.index)
        y = sk.targets["mh"]
        shells_k, _ = D.feature_set(sk, "shells24")
        ok = np.isfinite(y) & np.isfinite(mh0)
        fold = np.where(ok & ~s0.lockbox, s0.fold, -1)
        used = fold >= 0
        designs = {
            "nothing": (mtot0, M.Climatology("mh")),
            "M*(<148) at z=0.4": (mtot0, M.scalar_method("line")),
            "z=0.4 profile (24 shells)": (shells0, ridge()),
            "z=0.4 profile + quadratic": (shells0, quad()),
            "same-epoch profile": (shells_k, ridge()),
            "same-epoch profile + quadratic": (shells_k, quad()),
            "both profiles": (np.column_stack([shells0, shells_k]), ridge()),
            "both profiles + quadratic": (np.column_stack([shells0, shells_k]), quad()),
            "mediated (profile's own Mh(0.4))": (np.column_stack([shells0, mh0]), Mediated("mediated")),
            "ORACLE true Mh(z=0.4)": (mh0[:, None], M.DirectLinear("poly", "L0", transform=M._poly_only)),
            "ORACLE true Mh(0.4) + z=0.4 profile": (np.column_stack([shells0, mh0]), ridge()),
            "ORACLE true Mh(0.4) + profile + quad": (np.column_stack([shells0, mh0]), quad()),
            "ORACLE true Mh(0.4) + same-epoch profile": (np.column_stack([shells_k, mh0]), ridge()),
        }
        per, rows = {}, {}
        for name, (X, method) in designs.items():
            per[name], _ = HN.oof_scores(method, X, y, fold, None)
            rows[name] = HN.summarise_per(per[name], y, used)
        clim = rows["nothing"]["crps"]
        growth = mh0[used] - y[used]
        print(f"\n--- the main progenitor's log M200c at z = {C.ANCHOR_Z[k]}: {used.sum()} development galaxies; its spread {np.std(y[used]):.3f} dex; "
              f"growth to z = 0.4: median {np.median(growth):.3f}, scatter {np.std(growth):.3f} dex ---")
        print(f"    {'input':<42}{'CRPS':>8}{'skill':>8}{'RMSE':>8}{'cov68':>7}{'cov90':>7}")
        for name, r in rows.items():
            print(f"    {name:<42}{r['crps']:>8.4f}{1 - r['crps'] / clim:>8.3f}{r['rmse']:>8.4f}{r['cover68']:>7.2f}{r['cover90']:>7.2f}")

        def paired(a, b, label):
            ok_, boot = S.significant_gain(per[a]["crps"][used], per[b]["crps"][used], delta)
            print(f"    {label}: {100 * -boot['rel']:+.1f}% [{100 * -boot['rel_hi']:+.1f}, {100 * -boot['rel_lo']:+.1f}] {'SIGNIFICANT' if ok_ else 'not significant'}")
            return dict(a=a, b=b, gain=-boot["rel"], lo=-boot["rel_hi"], hi=-boot["rel_lo"], significant=ok_)
        best0 = min(("z=0.4 profile (24 shells)", "z=0.4 profile + quadratic"), key=lambda n: rows[n]["crps"])
        bestk = min(("same-epoch profile", "same-epoch profile + quadratic"), key=lambda n: rows[n]["crps"])
        bestb = min(("both profiles", "both profiles + quadratic"), key=lambda n: rows[n]["crps"])
        besto = min(("ORACLE true Mh(0.4) + z=0.4 profile", "ORACLE true Mh(0.4) + profile + quad"), key=lambda n: rows[n]["crps"])
        comps = [paired(best0, "M*(<148) at z=0.4", "the z = 0.4 PROFILE over the z = 0.4 stellar mass"),
                 paired(best0, "mediated (profile's own Mh(0.4))", "the z = 0.4 profile over its own estimate of the z = 0.4 mass"),
                 paired(bestk, best0, "the SAME-EPOCH profile over the z = 0.4 profile"),
                 paired(bestb, bestk, "adding the z = 0.4 profile to the same-epoch profile"),
                 paired("ORACLE true Mh(z=0.4)", best0, "the TRUE z = 0.4 halo mass over the z = 0.4 profile"),
                 paired(besto, "ORACLE true Mh(z=0.4)", "ORACLE: the z = 0.4 profile added to the true z = 0.4 mass (what it knows about the GROWTH)")]
        out[str(C.ANCHOR_Z[k])] = dict(n=int(used.sum()), rows=rows, comparisons=comps, spread=float(np.std(y[used])), growth_scatter=float(np.std(growth)))
    (C.OUTDIR / "stage7_cross_epoch.json").write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {C.OUTDIR / 'stage7_cross_epoch.json'}")


if __name__ == "__main__" and "--lag" not in sys.argv:
    main()


def lag_scan():
    """At which EARLIER time does the z = 0.4 profile predict the halo mass best?

    Target: the main progenitor's instantaneous log M200c at t(z = 0.4) - tau,
    from the dense history, for tau = 0.5 .. 6 Gyr (plain normal predictive;
    tau = 0 is the truncated Stage 3 problem and is not on this scale).
    Read against the TRUE z = 0.4 halo mass as the only input: the lookback
    where the profile overtakes it, and where the profile's own skill peaks.
    """
    import mah as MAH
    s0 = D.load_curated(0, verbose=False)
    so = np.load(MAH.SO_NPZ, allow_pickle=True)
    t_snap = np.loadtxt(MAH.TIME_TXT)
    snap0 = C.ANCHOR_SNAP[0]
    mass = np.asarray(so["Group_M_Crit200"], float)[s0.flags["population_row"], :snap0 + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.log10(mass * 1e10 / MAH.H_LITTLE)
    taus = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
    t0 = float(t_snap[snap0])
    Y = np.full((len(s0), len(taus)), np.nan)
    for i, row in enumerate(logm):
        ok = np.isfinite(row)
        if ok.sum() >= 5 and ok[-1]:
            Y[i] = np.interp(t0 - taus, t_snap[:snap0 + 1][ok], row[ok])
    shells0, _ = D.feature_set(s0, "shells24")
    mh0 = s0.targets["mh"]
    print(f"\n{RULE}\nTHE LAG SCAN — the halo's mass at t(z = 0.4) - tau from the z = 0.4 inputs (CRPS, dex; skill over knowing nothing)\n{RULE}")
    print(f"    {'tau [Gyr]':>10}{'z':>6}{'n':>6}{'nothing':>9}{'true Mh(0.4)':>14}{'M*(<148)':>10}{'profile':>9}{'profile skill':>15}{'Mh(0.4) skill':>15}{'profile RMSE':>14}")
    scan = []
    for j, tau in enumerate(taus):
        y = Y[:, j]
        ok = np.isfinite(y) & np.isfinite(mh0)
        fold = np.where(ok & ~s0.lockbox, s0.fold, -1)
        used = fold >= 0
        crps = {}
        for name, (X, method) in {"nothing": (mh0[:, None], M.Climatology("mh")), "mh0": (mh0[:, None], M.DirectLinear("poly", "L0", transform=M._poly_only)),
                                  "mtot": (s0.logcog[:, [-1]], M.scalar_method("line")), "profile": (shells0, quad())}.items():
            per, _ = HN.oof_scores(method, X, y, fold, None)
            crps[name] = float(per["crps"][used].mean())
            if name == "profile":
                rmse = float(np.sqrt(np.mean(per["err_mean"][used] ** 2)))
        z = float(np.interp(t0 - tau, t_snap[::-1][::-1], np.arange(len(t_snap))))        # the snapshot index, for the record
        scan.append(dict(tau=float(tau), n=int(used.sum()), rmse=rmse, **crps))
        print(f"    {tau:>10.1f}{'':>6}{used.sum():>6}{crps['nothing']:>9.4f}{crps['mh0']:>14.4f}{crps['mtot']:>10.4f}{crps['profile']:>9.4f}"
              f"{1 - crps['profile'] / crps['nothing']:>15.3f}{1 - crps['mh0'] / crps['nothing']:>15.3f}{rmse:>14.4f}")
        del z
    best = max(scan, key=lambda r: 1 - r["profile"] / r["nothing"])
    cross = next((r["tau"] for r in scan if r["profile"] < r["mh0"]), None)
    print(f"    the profile's skill peaks at tau = {best['tau']:.1f} Gyr (skill {1 - best['profile'] / best['nothing']:.3f}); "
          f"it overtakes the TRUE z = 0.4 halo mass as a predictor from tau = {cross} Gyr on")
    out = json.loads((C.OUTDIR / "stage7_cross_epoch.json").read_text())
    out["lag_scan"] = scan
    (C.OUTDIR / "stage7_cross_epoch.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__" and "--lag" in sys.argv:
    lag_scan()
