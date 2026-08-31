"""exp71 part 2 — where does the future-growth channel come from?

`leakage.py` established C18's answer, and turned up something its own question
did not ask for. The residual of every model tested depends on FUTURE GROWTH at
fixed halo mass (partial correlation +0.19 to +0.21 at 103 kpc, every epoch,
outside the shuffle null) -- but the MEASURED stellar mass does not. TNG300's
own M*(<103 kpc) at fixed log10 M200c(z_k) has essentially no dependence on how
much the halo grows afterwards: partial correlation +0.003 at z = 2, and never
above +0.09.

So the channel is not the simulation's. It is manufactured by the model, and
that changes what C18's answer means. If it were the simulation's, an
epoch-local model would be structurally unable to reproduce it and reporting
the selection would be the only honest response. If it is the model's, it is a
defect with an address, and closing it would remove the sample's sensitivity to
the selection altogether.

THREE CANDIDATE ADDRESSES, and the measurement that separates them:

  A. **DiffMAH mass mismatch.** The model reads a smooth four-parameter fit to
     the whole assembly history, not the catalog mass. At fixed CATALOG mass at
     z_k the DiffMAH mass varies, and because the fit was made to the whole
     curve -- the future included -- that variation can carry future growth.
     Test: regress the offset (DiffMAH minus catalog) on G at fixed catalog
     mass. If the offset carries the growth, and controlling for it removes the
     residual's dependence, this is the whole story.

  B. **The past genuinely predicts the future, and the simulation disagrees.**
     Haloes that will grow are not a random subset of haloes of that mass: they
     are the ones still on the rising part of their history. A model that
     integrates the pre-z_k curve legitimately reads that shape, and if
     TNG300's stellar mass does NOT read it, the model is over-reading real
     information. Test: how much of G at fixed Mh(z_k) does the REAL (catalog)
     pre-z_k mass history predict? Ten catalog snapshots exist before z = 2.

  C. **DiffMAH's pre-z_k curve knows the future in a way the real one does
     not.** The same test run on the DiffMAH curve. If DiffMAH's past predicts
     G far better than the catalog's past does, the model is reading a
     smoothing artefact and A and C are two faces of one problem.

The separator is the pair of R-squared values in section 2: the same target
(future growth at fixed halo mass), the same number of predictors, one set
measured and one set fitted.

Everything is frozen and nothing is refitted. Run after `leakage.py`:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp71_c18_selection_leakage/channel_origin.py [--smoke]
"""
from __future__ import annotations

import sys
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
import selection as SEL                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
from leakage import (ANCHOR_Z, EPOCHS, FIT_NPZ, OUTDIR, R_LEAK,  # noqa: E402
                     RULE, residual_cum)

#: how many pre-z_k history points each predictor set gets. Both sets get the
#: SAME number, because R-squared rises with predictors whatever they contain.
#: Three is what the catalog can supply at z = 2 -- only snapshots 17, 21 and 25
#: lie before the z = 2 anchor with a halo mass measured for most of our
#: galaxies (93%, 99%, 100%; snapshot 13 and earlier fall to 74% and below,
#: because the progenitor is not yet a resolved FoF-group central).
N_HIST = 3
#: a snapshot is usable as a predictor only if this fraction of the galaxies
#: have a valid central-halo mass there; a column that is missing for half the
#: sample would select on the very thing being measured
MIN_COVER = 0.90
#: cross-validation folds for every R-squared quoted here: an in-sample R^2
#: with four predictors on 2000 rows is not large, but the comparison between
#: two predictor sets is the whole point and it must not be one they can win by
#: fitting noise differently
N_FOLD = 5


def cv_r2(y, X, n_fold=N_FOLD, seed=0):
    """K-fold cross-validated R-squared of a linear fit of y on X (n, p).

    Out-of-fold predictions only, so a predictor set cannot buy R-squared by
    fitting noise. Returns NaN if there is not enough data.
    """
    y = np.asarray(y, float)
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 10 * (X.shape[1] + 2):
        return np.nan
    y, X = y[ok], X[ok]
    D = np.column_stack([np.ones(len(y)), X])
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % n_fold
    pred = np.empty(len(y))
    for f in range(n_fold):
        tr, te = fold != f, fold == f
        beta = np.linalg.lstsq(D[tr], y[tr], rcond=None)[0]
        pred[te] = D[te] @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-300)


def partial_slope(y, g, covars, mask):
    """(partial correlation, slope dy/dg, n) of y on g controlling for `covars`.

    The generalisation of `selection.partial_growth` from one covariate to
    several: both y and g are stripped of their linear dependence on ALL the
    covariates, and what remains is correlated. `covars` is a list of (n,)
    arrays.
    """
    C = np.column_stack([np.asarray(c, float) for c in covars])
    ok = (mask & np.isfinite(y) & np.isfinite(g) & np.isfinite(C).all(axis=1))
    n = int(ok.sum())
    if n < 30 + C.shape[1]:
        return np.nan, np.nan, n
    D = np.column_stack([np.ones(n), C[ok]])
    ry = y[ok] - D @ np.linalg.lstsq(D, y[ok], rcond=None)[0]
    rg = g[ok] - D @ np.linalg.lstsq(D, g[ok], rcond=None)[0]
    if rg.std() < 1e-12:
        return np.nan, np.nan, n
    return (float(np.corrcoef(ry, rg)[0, 1]),
            float(np.polyfit(rg, ry, 1)[0]), n)


def smoothed_catalog_past(hs, k, n_coef=N_HIST, min_cover=MIN_COVER, min_pts=4):
    """A SMOOTH summary of the measured pre-z_k history: (n, n_coef).

    The control the comparison in section 2 needs. Three raw catalog snapshots
    carry halo-finder noise -- mergers counted or not, a boundary crossed --
    that a four-parameter fit to the whole curve does not, so a DiffMAH set
    could win on noise suppression alone rather than on knowing the future.
    This gives the catalog the same advantage: a polynomial of the same order
    in log time, fitted to EVERY usable pre-z_k snapshot of that galaxy, with
    its coefficients as the predictors. Same predictor count, same smoothness,
    and strictly no data from after z_k.
    """
    snaps = list(hs["snaps"])
    m_all = np.asarray(hs["M200c"], float)
    ok_all = (np.asarray(hs["GroupFlag"]) == 1) & np.isfinite(m_all) & (m_all > 0)
    cover = ok_all.mean(axis=0)
    anchor = E.ANCHOR_SNAP[k]
    cols = [i for i, s in enumerate(snaps) if s < anchor and cover[i] >= min_cover]
    if len(cols) < min_pts:
        return None, 0
    lt = np.log10(np.array([E.T_SNAP[snaps[i]] for i in cols]))
    lt0 = np.log10(E.T_SNAP[anchor])
    x = lt - lt0
    out = np.full((m_all.shape[0], n_coef), np.nan)
    for g in range(m_all.shape[0]):
        v = ok_all[g, cols]
        if v.sum() < max(min_pts, n_coef):
            continue
        out[g] = np.polyfit(x[v], m_all[g, cols][v], n_coef - 1)
    return out, len(cols)


def history_columns(hs, curves, k, n_hist=N_HIST, min_cover=MIN_COVER):
    """The pre-z_k halo history, measured and fitted, as matched predictor sets.

    Returns (z_used, real (n, n_hist), diffmah (n, n_hist)). Both are log10
    halo mass at the SAME cosmic times -- the `n_hist` catalog snapshots
    immediately before this epoch's anchor whose mass is measured for at least
    `min_cover` of the galaxies -- so the only difference between the two sets
    is measured against fitted. Earlier snapshots exist but are dropped: before
    z ~ 5 the main progenitor is often not yet a resolved FoF-group central,
    and a column missing for half the sample would select on assembly history,
    which is the very thing under test.
    """
    snaps = list(hs["snaps"])
    m_all = np.asarray(hs["M200c"], float)
    ok_all = (np.asarray(hs["GroupFlag"]) == 1) & np.isfinite(m_all) & (m_all > 0)
    cover = ok_all.mean(axis=0)
    anchor = E.ANCHOR_SNAP[k]
    usable = [s for i, s in enumerate(snaps) if s < anchor and cover[i] >= min_cover]
    if len(usable) < n_hist:
        return None, None, None
    take = usable[-n_hist:]                       # the n_hist closest to the anchor

    cols = [snaps.index(s) for s in take]
    real = np.where(ok_all[:, cols], m_all[:, cols], np.nan)
    lt = np.log10(np.array([E.T_SNAP[s] for s in take]))
    dm = np.array([E.log_mah(lt, hc) for hc in curves])
    return np.asarray(hs["z"])[cols], real, dm


def main(smoke=False):
    print(f"{RULE}\nexp71 part 2 — where does the future-growth channel come "
          f"from?\n{RULE}\n")

    recs, data, mask_legacy, lmh_dm, _, _, _ = S0.build(smoke)
    fitmask, complete = S2F.fit_masks(data, mask_legacy, lmh_dm, "sane")
    curves = E.build_curves(recs, verbose=False)
    sel_rows = np.array([h.row for h in recs])

    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    theta = np.asarray(fz["theta_best"], float)
    cache = OUTDIR / f"pred_exp63_{len(recs)}.npy"
    if cache.exists() and not smoke:
        pred = np.load(cache)
        print(f"  reusing the frozen prediction {cache.name}")
    else:
        pred = M2.predict2(spec2, theta, curves, F.R_GRID)

    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat_all = SEL.sample_masses(hs)
    lmh_cat = lmh_cat_all[sel_rows]
    growth = lmh_cat[:, 0][:, None] - lmh_cat

    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    good &= np.isfinite(pred).all(axis=(1, 2)) & (pred > 0).all(axis=(1, 2))
    good &= fitmask.all(1)
    keep = good[:, None] & np.isfinite(lmh_cat)

    lab = f"M(<{R_LEAK:g})"
    c_idx = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    y_res = residual_cum(pred, data, R_LEAK)
    y_mod = np.log10(np.clip(pred[:, :, c_idx], 1.0, None))
    y_tru = np.log10(np.clip(data[:, list(EPOCHS), c_idx], 1.0, None))
    print(f"  quantity {lab}; {int(good.sum())} galaxies; theta frozen at "
          f"{FIT_NPZ.name}")

    # ---- 1. the three quantities, side by side --------------------------- #
    print(f"\n{RULE}\n1. WHO READS FUTURE GROWTH — the same partial slope for "
          f"the model, the truth,\n   and their difference, all at fixed "
          f"log10 M200c(z_k). dex per dex of growth.\n{RULE}")
    print(f"  {'epoch':>6}{'model':>12}{'truth':>12}{'residual':>12}"
          f"{'model - truth':>16}{'n':>8}")
    for k in range(1, 5):
        _, s_m, n = SEL.partial_growth(y_mod[:, k], lmh_cat[:, k], growth[:, k], keep[:, k])
        _, s_t, _ = SEL.partial_growth(y_tru[:, k], lmh_cat[:, k], growth[:, k], keep[:, k])
        _, s_r, _ = SEL.partial_growth(y_res[:, k], lmh_cat[:, k], growth[:, k], keep[:, k])
        print(f"  {ANCHOR_Z[k]:>6.1f}{s_m:>+12.4f}{s_t:>+12.4f}{s_r:>+12.4f}"
              f"{s_m - s_t:>+16.4f}{n:>8d}")
    print(f"\n  The residual column is the leakage `leakage.py` measured. It is "
          f"the model column\n  almost exactly, because the truth column is "
          f"near zero: the channel is the model's.")

    # ---- 2. does the PAST predict the FUTURE? ---------------------------- #
    print(f"\n{RULE}\n2. DOES THE PAST PREDICT THE FUTURE? — cross-validated "
          f"R-squared of future\n   growth G_k on the pre-z_k halo history at "
          f"fixed log10 M200c(z_k), for the\n   MEASURED history and the "
          f"FITTED (DiffMAH) one. Same target, same number of\n   predictors, "
          f"{N_FOLD}-fold out-of-fold predictions, so the only difference is "
          f"measured\n   against fitted.\n{RULE}")
    logmp = np.array([hc.logmp for hc in curves])       # DiffMAH's PEAK mass
    def _f(v):
        return "     —" if not np.isfinite(v) else f"{v:6.3f}"
    print(f"  {'epoch':>6}{'pre-z_k snapshots at z =':>28}{'R2 catalog':>12}"
          f"{'R2 catalog smoothed':>21}{'R2 DiffMAH':>12}"
          f"{'R2 DiffMAH logmp alone':>24}{'n':>8}")
    r2 = np.full((5, 4), np.nan)
    hist = {}
    for k in range(1, 5):
        t_used, real, dm = history_columns(hs, curves, k)
        if real is None:
            continue
        real = real[sel_rows]
        smooth, n_snap = smoothed_catalog_past(hs, k)
        smooth = smooth[sel_rows] if smooth is not None else None
        hist[k] = (real, dm, smooth)
        ok = keep[:, k] & np.isfinite(growth[:, k]) & np.isfinite(real).all(1)
        if smooth is not None:
            ok &= np.isfinite(smooth).all(1)
        # strip the epoch's own halo mass from BOTH sides first, so what is
        # predicted is growth AT FIXED MASS and the predictors carry only the
        # SHAPE of the history, not its normalisation
        D0 = np.column_stack([np.ones(int(ok.sum())), lmh_cat[ok, k]])
        g_res = growth[ok, k] - D0 @ np.linalg.lstsq(D0, growth[ok, k], rcond=None)[0]
        def strip(X):
            return X - D0 @ np.linalg.lstsq(D0, X, rcond=None)[0]
        r2[k] = [cv_r2(g_res, strip(real[ok])),
                 cv_r2(g_res, strip(smooth[ok])) if smooth is not None else np.nan,
                 cv_r2(g_res, strip(dm[ok])),
                 cv_r2(g_res, strip(logmp[ok][:, None]))]
        print(f"  {ANCHOR_Z[k]:>6.1f}"
              + ("  ".join(f"{t:.2f}" for t in t_used)).rjust(28)
              + _f(r2[k, 0]).rjust(12) + _f(r2[k, 1]).rjust(21)
              + _f(r2[k, 2]).rjust(12) + _f(r2[k, 3]).rjust(24)
              + f"{int(ok.sum()):>8d}")
    print(f"\n  A large R-squared for the CATALOG past means future growth is "
          f"genuinely readable\n  from the measured history, and a model that "
          f"reads the history is entitled to it.\n  A DiffMAH R-squared much "
          f"larger than the catalog one means the smooth fit carries\n  "
          f"information the measured history does not, which the model would "
          f"then be reading\n  as if it were physical. The middle column is "
          f"the control for the obvious objection:\n  three raw snapshots "
          f"carry halo-finder noise a smooth curve does not, so it gives the\n"
          f"  catalog the same smoothness — a polynomial of the same order in "
          f"log time fitted to\n  EVERY usable pre-z_k snapshot, with no data "
          f"from after z_k. (At z = 2 only three\n  usable snapshots lie "
          f"before the anchor, so the smooth fit is the raw points and the\n"
          f"  column is left blank.) The last column is the mechanism in one "
          f"number: DiffMAH's\n  `logmp` is the halo's PEAK mass, a property "
          f"of the whole history, and it enters the\n  curve at every time — "
          f"so a curve read before z_k still carries it.")

    # ---- 3. controlling for the candidates ------------------------------- #
    print(f"\n{RULE}\n3. WHAT REMOVES IT — the residual's slope on future "
          f"growth, controlling for more\n   and more. If a control removes "
          f"the slope, that control IS the channel.\n{RULE}")
    print(f"  each cell is  dy/dG  (R2 of G on that control) — see the warning "
          f"below the table")
    print(f"  {'epoch':>6}" + "".join(f"{h:>20}" for h in (
        "M200c(z_k)", "+ DiffMAH Mh", "+ catalog past", "+ DiffMAH past")) + f"{'n':>8}")
    ctrl = np.full((5, 4), np.nan)
    ctrl_r2 = np.full((5, 4), np.nan)
    for k in range(1, 5):
        base = [lmh_cat[:, k]]
        sets = [base, base + [lmh_dm[:, k]]]
        # ONE mask for all four columns, so the numbers in a row are comparable:
        # the catalog past is missing for a few per cent of galaxies and those
        # would otherwise drop out of one column only
        common = keep[:, k].copy()
        if k in hist:
            real, dm, _ = hist[k]
            common &= np.isfinite(real).all(1) & np.isfinite(dm).all(1)
            sets += [base + [real[:, j] for j in range(real.shape[1])],
                     base + [lmh_dm[:, k]] + [dm[:, j] for j in range(dm.shape[1])]]
        row, rr, n = [], [], 0
        for cs in sets:
            _, sl, n = partial_slope(y_res[:, k], growth[:, k], cs, common)
            row.append(sl)
            C = np.column_stack([np.asarray(c, float) for c in cs])
            m2 = common & np.isfinite(growth[:, k]) & np.isfinite(C).all(axis=1)
            rr.append(cv_r2(growth[m2, k], C[m2]))
        ctrl[k, :len(row)] = row
        ctrl_r2[k, :len(rr)] = rr
        print(f"  {ANCHOR_Z[k]:>6.1f}"
              + "".join((f"{v:+.4f} ({r:.2f})" if np.isfinite(v) else "—").rjust(20)
                        for v, r in zip(ctrl[k], ctrl_r2[k]))
              + f"{n:>8d}")
    print(f"\n  Reading it: column 1 is the leakage as measured. Column 2 adds "
          f"the halo mass the\n  model actually reads. Columns 3 and 4 add "
          f"{N_HIST} points of the pre-z_k history.\n"
          f"\n  WARNING, and it is why this table is corroboration and not the "
          f"argument. A partial\n  slope is measured on the part of G the "
          f"control does NOT explain. Where the bracketed\n  R-squared "
          f"approaches 1 that part is a thin remainder, and the slope on it is "
          f"both\n  poorly determined and free to move in either direction — "
          f"so a column-4 number LARGER\n  than column 1 is not evidence "
          f"against the leak, it is evidence that the control has\n  almost "
          f"nothing left to work with. Section 2 is the clean test: it compares "
          f"two\n  predictor sets on the SAME target with the same count, "
          f"rather than conditioning on\n  a near-determining variable.")

    # ---- 4. the DiffMAH offset ------------------------------------------- #
    print(f"\n{RULE}\n4. IS IT THE DIFFMAH OFFSET? — the fitted minus the "
          f"measured halo mass at z_k,\n   against future growth at fixed "
          f"measured mass. A smooth fit made to the WHOLE\n   history can put "
          f"the future into the past through this offset alone.\n{RULE}")
    print(f"  {'epoch':>6}{'median offset':>16}{'scatter':>10}"
          f"{'partial r with G':>19}{'d offset / dG':>16}{'n':>8}")
    off = lmh_dm - lmh_cat
    for k in range(1, 5):
        ok = keep[:, k] & np.isfinite(off[:, k])
        r, sl, n = SEL.partial_growth(off[:, k], lmh_cat[:, k], growth[:, k], keep[:, k])
        print(f"  {ANCHOR_Z[k]:>6.1f}{np.nanmedian(off[ok, k]):>+16.4f}"
              f"{np.nanstd(off[ok, k]):>10.4f}{r:>+19.4f}{sl:>+16.4f}{n:>8d}")
    print(f"\n  An offset that reads growth is not by itself a problem; it "
          f"becomes one only if\n  controlling for it removes the residual's "
          f"slope, which section 3 column 2 tests.")

    np.savez(OUTDIR / ("channel_origin_smoke.npz" if smoke else "channel_origin.npz"),
             anchor_z=np.array(ANCHOR_Z), r2=r2, ctrl=ctrl, ctrl_r2=ctrl_r2,
             offset_median=np.array([np.nanmedian(off[keep[:, k], k]) for k in range(5)]))
    print(f"\nwrote {OUTDIR / ('channel_origin_smoke.npz' if smoke else 'channel_origin.npz')}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
