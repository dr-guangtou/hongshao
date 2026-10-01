"""exp87 Stage 2 — the selection report. What the sample's selection can and
cannot do to an inverse prediction, measured before the ladder runs.

1. INDEPENDENCE at high redshift. The z >= 0.7 sample is the set of main
   progenitors of z = 0.4 haloes above 10^13, so at fixed progenitor mass y_k
   it favours haloes with large future growth G = log Mh(z=0.4) - log Mh(z_k).
   The population claim "centrals above the completeness cut c_k" is valid
   only if the curve of growth does not know G at fixed y_k. Two readings:
     (a) does the curve of growth improve a cross-validated prediction of G
         beyond y_k alone? (the gain in R^2)
     (b) per principal component of the curve of growth: the partial slope on
         G at fixed y_k, times the mean shift of G that the selection induces
         (exp54's inverse-Mills estimate from the measured completeness), in
         units of the component's forward scatter. TOLERANCE 0.2 (the plan).
   Read on the whole progenitor sample and above c_k. A violation above c_k
   demotes the "complete above c_k" rows to exploratory.
2. PARENT versus CURATED at z = 0.4: the forward relation (curve of growth on
   halo mass) and its resolution on both samples — the size of the optimism
   the curated sample's cuts would have bought.
3. INVERSE-COMPLETENESS WEIGHTS: the Kish effective sample size, over the
   whole progenitor sample and above c_k (why weighting is a sensitivity row
   and not the primary path).

Run:  PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage2_selection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402

RULE = "=" * 100
SEL_NPZ = C.ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
N_COMPONENTS = 4


def cv_r2(X, y):
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), X, y, cv=KFold(5, shuffle=True, random_state=C.SEED))
    return float(1.0 - np.mean((y - pred) ** 2) / np.var(y))


def completeness_curve(epoch):
    """(centres, completeness, plateau) of the sample against the box at one epoch (exp54)."""
    z = np.load(SEL_NPZ, allow_pickle=True)
    return np.asarray(z[f"comp_ctr_{epoch}"], float), np.asarray(z[f"comp_val_{epoch}"], float), float(z["ceilings"][epoch])


def induced_growth_shift(y, growth, epoch):
    """The mean shift of G at each galaxy's mass that keeping only the upper
    fraction C_eff(y) of a normal G distribution would produce: s_G * phi(z_c) / C_eff
    with z_c = Phi^-1(1 - C_eff) (exp54 `predicted_spurious_tilt`, the 'bound' form)."""
    ctr, comp, plateau = completeness_curve(epoch)
    ok = np.isfinite(comp)
    c_eff = np.clip(np.interp(y, ctr[ok], comp[ok] / plateau), 1e-3, 1.0)
    resid = growth - np.polyval(np.polyfit(y, growth, 2), y)
    s_g = float(np.std(resid))
    z_c = ndtri(np.clip(1.0 - c_eff, 1e-9, 1 - 1e-9))
    shift = np.where(c_eff < 1.0, s_g * np.exp(-0.5 * z_c * z_c) / np.sqrt(2 * np.pi) / c_eff, 0.0)
    return shift, c_eff, s_g


def forward_fit(logcog, y, n_components=N_COMPONENTS):
    """Standardise + PCA the curve of growth, regress each component on (1, y): returns
    (scores, slope per component, residual scatter per component, resolution)."""
    z = PCA(n_components, random_state=C.SEED).fit_transform(StandardScaler().fit_transform(logcog))
    B = np.column_stack([np.ones(len(y)), y - np.median(y)])
    coef = np.linalg.lstsq(B, z, rcond=None)[0]
    resid = z - B @ coef
    cov = np.cov(resid.T)
    resolution = float(1.0 / np.sqrt(coef[1] @ np.linalg.inv(cov) @ coef[1]))
    return z, coef[1], resid.std(0), resolution


def main():
    print(f"{RULE}\nexp87 STAGE 2 — the selection report (config {C.config_hash()})\n{RULE}\n")
    report = {"config": C.config_hash(), "independence": {}, "weights": {}}

    print("1. DOES THE CURVE OF GROWTH KNOW THE HALO'S FUTURE GROWTH AT FIXED PROGENITOR MASS?  (tolerance: shift < "
          f"{C.INDEPENDENCE_TOL} forward scatters per component)")
    worst_above = 0.0
    for k in range(1, 5):
        s = D.load_curated(k, verbose=False)
        y, g = s.targets["mh"], s.targets["future_growth"]
        ok = np.isfinite(y) & np.isfinite(g) & ~s.lockbox
        for label, rows in (("all progenitors", ok), (f"above c_k = {C.COMPLETE_CUTS[k]}", ok & s.flags["complete"])):
            yy, gg, xx = y[rows], g[rows], s.logcog[rows]
            base = np.column_stack([yy, yy ** 2])
            r2_y, r2_yx = cv_r2(base, gg), cv_r2(np.column_stack([base, xx]), gg)
            z, slope_y, sd, _ = forward_fit(xx, yy)
            design = np.column_stack([np.ones(len(yy)), yy - np.median(yy), (yy - np.median(yy)) ** 2, gg])
            beta_g = np.linalg.lstsq(design, z, rcond=None)[0][3]
            shift, c_eff, s_g = induced_growth_shift(yy, gg, k)
            standardized = np.abs(beta_g) * float(np.mean(shift)) / sd
            # what the shift means for the inferred mass: the component shift divided by its slope on y, for the leading component
            dy = float(beta_g[0] * np.mean(shift) / slope_y[0]) if slope_y[0] != 0 else np.nan
            key = f"z{C.ANCHOR_Z[k]}|{label}"
            report["independence"][key] = dict(n=int(rows.sum()), r2_from_mass=r2_y, r2_with_cog=r2_yx, slope_on_growth=beta_g.tolist(),
                                               mean_growth_shift=float(np.mean(shift)), scatter_growth=s_g, mean_c_eff=float(np.mean(c_eff)),
                                               standardized_shift=standardized.tolist(), implied_mass_shift=dy)
            if label.startswith("above"):
                worst_above = max(worst_above, float(standardized.max()))
            print(f"   z = {C.ANCHOR_Z[k]}, {label:<22} n {rows.sum():>4}: R^2 of G from mass {r2_y:+.3f}, with the curve of growth {r2_yx:+.3f} "
                  f"(gain {r2_yx - r2_y:+.3f}); selection keeps {100 * np.mean(c_eff):.0f}% on average, shifting G by {np.mean(shift):+.3f} dex "
                  f"(scatter of G {s_g:.3f}); per-component shift / forward scatter " + " / ".join(f"{v:.3f}" for v in standardized)
                  + f"; implied shift of the inferred mass {dy:+.3f} dex")
    report["independence_passed_above_cut"] = bool(worst_above < C.INDEPENDENCE_TOL)
    print(f"   VERDICT above the completeness cuts: worst component shift {worst_above:.3f} forward scatters "
          f"({'within' if worst_above < C.INDEPENDENCE_TOL else 'OUTSIDE'} the tolerance {C.INDEPENDENCE_TOL}) -> the 'complete above c_k' rows are "
          f"{'a population claim under the stated assumption' if worst_above < C.INDEPENDENCE_TOL else 'EXPLORATORY'}")

    print("\n2. PARENT versus CURATED at z = 0.4: the forward relation and its resolution")
    parent, curated = D.load_parent(verbose=False), D.load_curated(0, verbose=False)
    rows = {"parent (complete)": (parent.logcog[~parent.lockbox], parent.targets["mh"][~parent.lockbox]),
            "curated, input-only cuts": (curated.logcog[~curated.lockbox], curated.targets["mh"][~curated.lockbox]),
            "curated, the 2356 fitting sample": (curated.logcog[~curated.lockbox & curated.flags["fit2356"]],
                                                 curated.targets["mh"][~curated.lockbox & curated.flags["fit2356"]])}
    report["forward"] = {}
    for label, (xx, yy) in rows.items():
        _, slope, sd, resolution = forward_fit(xx, yy)
        i148 = xx[:, -1]
        b = np.polyfit(yy, i148, 1)
        scatter = float(np.std(i148 - np.polyval(b, yy)))
        report["forward"][label] = dict(n=int(len(yy)), resolution=resolution, slope_m148=float(b[0]), scatter_m148=scatter,
                                        median_m148_at_13p3=float(np.polyval(b, 13.3)))
        print(f"   {label:<34} n {len(yy):>4}: forward resolution (4 components, flat prior) {resolution:.3f} dex; "
              f"M*(<148) on Mh: slope {b[0]:.3f}, scatter {scatter:.3f} dex, value at Mh = 13.3: {np.polyval(b, 13.3):.3f}")
    drop = parent.flags["mah_declined"] & ~parent.lockbox
    yy, m148 = parent.targets["mh"][~parent.lockbox], parent.logcog[~parent.lockbox, -1]
    resid = m148 - np.polyval(np.polyfit(yy, m148, 2), yy)
    d = drop[~parent.lockbox]
    report["declined_offset"] = float(np.mean(resid[d]) - np.mean(resid[~d]))
    print(f"   the haloes the curated sample drops for a declined accretion history ({int(d.sum())} of {len(d)}) sit "
          f"{report['declined_offset']:+.3f} dex in M*(<148) at fixed halo mass relative to the rest: the curated cut depends on the input at fixed target")

    print("\n3. INVERSE-COMPLETENESS WEIGHTS: the effective sample size (Kish)")
    for k in range(1, 5):
        s = D.load_curated(k, verbose=False)
        y = s.targets["mh"]
        ok = np.isfinite(y) & ~s.lockbox
        ctr, comp, plateau = completeness_curve(k)
        good = np.isfinite(comp)
        c = np.clip(np.interp(y[ok], ctr[good], comp[good]), 1e-4, None)
        w = 1.0 / c
        above = s.flags["complete"][ok]
        n_eff_all = float(w.sum() ** 2 / (w ** 2).sum())
        n_eff_above = float(w[above].sum() ** 2 / (w[above] ** 2).sum())
        report["weights"][f"z{C.ANCHOR_Z[k]}"] = dict(n=int(ok.sum()), n_eff_all=n_eff_all, n_above=int(above.sum()), n_eff_above=n_eff_above,
                                                      max_weight=float(w.max()))
        print(f"   z = {C.ANCHOR_Z[k]}: whole sample n {ok.sum()} -> effective {n_eff_all:.0f} (largest weight {w.max():.0f}); "
              f"above c_k n {above.sum()} -> effective {n_eff_above:.0f}")
    (C.OUTDIR / "stage2_selection.json").write_text(json.dumps(report, indent=1))
    print(f"\n  STAGE 2 done; wrote {C.OUTDIR / 'stage2_selection.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
