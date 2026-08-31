"""exp71 — C18: is the z=2 halo-mass dependence the z=0.4 progenitor selection?

THE QUESTION, stated without jargon. Every galaxy in this project was picked at
redshift 0.4 and then followed backwards. So the "z = 2 sample" is not a sample
of z = 2 galaxies: it is the set of main progenitors of a z = 0.4 selection. At
z = 2 a halo of a given mass is in the sample only if it goes on to grow enough
to clear the z = 0.4 threshold. At the massive end that condition is free --
every heavy z = 2 halo clears it -- but at the light end it keeps only the
fast-growing minority. Any statement of the form "the model is worse for light
haloes at z = 2" therefore mixes a statement about the model with a statement
about which light haloes we chose to look at.

exp63 Stage 5i raised this concretely. The adopted exp63 mean
(`stage2_fit_joint_kpc_free_sane.npz`, one theta at all five epochs) is within a
few per cent of the truth on the whole fitting sample at z = 2, but on the
mh-complete subset -- the massive progenitors, the ones the selection did NOT
filter -- it is far off (M*(50-100 kpc): -1.6% on the whole sample against
-28.4% on the subset). One theta cannot do both. Either the model needs a second
variable at high redshift, which is a modelling step, or the difference is the
selection speaking, which is a reporting statement.

WHAT SETTLES IT, AND WHY NO FIT IS NEEDED

A selection can only bias a residual through a variable that (a) the selection
truncates and (b) the residual depends on. Here that variable is FUTURE GROWTH

    G_k = log10 Mh(z=0.4) - log10 Mh(z_k)

-- how much the halo grows AFTER the epoch being scored. It is exactly what the
z = 0.4 selection cuts on, and exactly what a model reading only the halo's
history up to z_k cannot see. Three measurements, in order:

1. **The phenomenon.** The halo-mass tilt of the residual (dex of stellar-mass
   error per dex of halo mass) on the fitting sample and on the mh-complete
   subset, at several radii and all five epochs. This is the thing to explain.

2. **The channel.** The partial correlation of the residual with G_k at fixed
   log10 Mh(z_k), and the slope dy/dG in dex per dex. If this is null the
   selection cannot be producing the residual through this channel whatever the
   completeness curve looks like, and C18's answer is "no". Run against a
   permutation null first.

3. **The size.** `selection.predicted_spurious_tilt` predicts the tilt the
   selection ALONE would produce, from the measured completeness curve and the
   measured dy/dG -- with no reference at all to the tilt we observe. If the
   prediction accounts for the observed tilt, the mass dependence is the
   selection; if it accounts for a small part of it, the rest is the model's.

Nothing here refits anything. Theta is frozen at the exp63 joint fit. The
nested incumbent (compact share exactly zero) is carried through every table as
the control: if the leakage is the same for both, it is a property of the
sample, not of the exp63 fit.

THE SAMPLE is the repo rule (CLAUDE.md): all galaxies selected at z = 0.4 with a
sane halo and stellar history at every epoch, `selection.fitting_sample_mask`,
2356 of 2397. The mh-complete subset is the after-fit check, never the fit.

DEFINITIONS (never assume the reader has them)

* **residual** y = log10[M*_model(<R)] - log10[M*_truth(<R)], per galaxy, per
  epoch. Negative means the model puts LESS stellar mass there than TNG300 did.
* **halo-mass tilt** = slope of y against log10 M200c across the population at
  one radius and one epoch. Zero means the model is equally right for light and
  heavy haloes.
* **completeness** = (our galaxies) / (all TNG300 central haloes) in a 0.1 dex
  bin of log10 M200c at one epoch. 1.0 = we keep every halo of that mass.
* **mh-complete subset** = the progenitors above the per-epoch halo mass at
  which completeness reaches 60% of the z = 0.4 plateau (exp54 Stage 3.4).

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp71_c18_selection_leakage/leakage.py [--smoke] [--no-fig]
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

import fit as F                                          # noqa: E402  (must precede halo/model)
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402

RULE = "=" * 96
THIN = "-" * 96
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)

#: the exp63 product this experiment scores, theta FROZEN
FIT_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
#: exp54 Stage 3.4's measured selection function: completeness curves, the
#: per-epoch mh-complete cuts, and each epoch's high-mass plateau
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"

#: radii the tilt and the leakage are reported at. The last entry is the
#: 52-103 kpc SHELL, not a cumulative aperture: it is the quantity whose
#: whole-sample and mh-complete numbers diverged most in exp63 Stage 5i.
R_REPORT = (2.0, 4.92, 10.25, 52.30, 103.45, 148.22)
SHELL = (52.30, 103.45)
#: the radius whose dy/dG feeds the spurious-tilt prediction, matching exp54
R_LEAK = 103.45
#: permutation null for the partial correlation: G shuffled inside halo-mass
#: bins, so the null keeps the mass distribution and destroys only the pairing
N_PERM, PERM_BINS = 200, 12


# --------------------------------------------------------------------------- #
# residuals
# --------------------------------------------------------------------------- #
def residual_cum(pred, data, r_kpc):
    """(n, 5) residual log10(model/truth) of the cumulative M*(<r_kpc)."""
    c = int(np.argmin(np.abs(F.R_GRID - r_kpc)))
    return (np.log10(np.clip(pred[:, :, c], 1.0, None))
            - np.log10(np.clip(data[:, list(EPOCHS), c], 1.0, None)))


def residual_shell(pred, data, r_in, r_out):
    """(n, 5) residual of the mass in the SHELL r_in < R < r_out.

    A shell can be non-positive in either the model or the measurement (the
    stored curve of growth is not strictly increasing everywhere), so those
    galaxy-epochs come back NaN and drop out of every statistic that uses them.
    """
    a = int(np.argmin(np.abs(F.R_GRID - r_in)))
    b = int(np.argmin(np.abs(F.R_GRID - r_out)))
    m = pred[:, :, b] - pred[:, :, a]
    t = data[:, list(EPOCHS), b] - data[:, list(EPOCHS), a]
    ok = (m > 0) & (t > 0)
    return np.where(ok, np.log10(np.where(ok, m, 1.0)) - np.log10(np.where(ok, t, 1.0)), np.nan)


def residual_table(pred, data):
    """{label: (n, 5) residual} for every reported radius and the shell."""
    out = {f"M(<{r:g})": residual_cum(pred, data, r) for r in R_REPORT}
    out[f"M({SHELL[0]:g}-{SHELL[1]:g})"] = residual_shell(pred, data, *SHELL)
    return out


# --------------------------------------------------------------------------- #
# the permutation null for the leakage test
# --------------------------------------------------------------------------- #
def perm_null_partial(y, lmh, growth, mask, n_perm=N_PERM, n_bins=PERM_BINS, seed=0):
    """Null distribution of the partial correlation of `y` with `growth`.

    `growth` is shuffled WITHIN equal-count bins of `lmh`, so the null keeps
    the joint distribution of mass and growth in the population and destroys
    only which galaxy has which growth. Returns (mean, 2.5%, 97.5%) of the
    null partial correlation and the same for the null slope dy/dG.
    """
    ok = mask & np.isfinite(y) & np.isfinite(lmh) & np.isfinite(growth)
    if ok.sum() < 50:
        return (np.nan,) * 6
    idx = np.where(ok)[0]
    edges = np.quantile(lmh[idx], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    which = np.clip(np.searchsorted(edges, lmh[idx], side="right") - 1, 0, n_bins - 1)
    rng = np.random.default_rng(seed)
    rs, ss = [], []
    g = growth.copy()
    for _ in range(n_perm):
        gp = g.copy()
        for b in range(n_bins):
            sub = idx[which == b]
            if len(sub) > 1:
                gp[sub] = g[rng.permutation(sub)]
        r, s, _ = SEL.partial_growth(y, lmh, gp, mask)
        rs.append(r); ss.append(s)
    rs, ss = np.array(rs, float), np.array(ss, float)
    return (float(np.nanmean(rs)), float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5)),
            float(np.nanmean(ss)), float(np.nanpercentile(ss, 2.5)), float(np.nanpercentile(ss, 97.5)))


# --------------------------------------------------------------------------- #
def main(smoke=False, make_fig=True):
    print(f"{RULE}\nexp71 — C18: is the z=2 halo-mass dependence the z=0.4 "
          f"progenitor selection?\n{RULE}\n")

    # ---- 0. sample, model, frozen theta --------------------------------- #
    recs, data, mask_legacy, lmh_dm, _, _, _ = S0.build(smoke)
    fitmask, complete = S2F.fit_masks(data, mask_legacy, lmh_dm, "sane")
    curves = E.build_curves(recs, verbose=False)
    sel_rows = np.array([h.row for h in recs])

    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    theta = np.asarray(fz["theta_best"], float)
    theta_nested = np.asarray(fz["theta_nested"], float)
    print(f"  model    : {FIT_NPZ.name}, THETA FROZEN "
          f"(loss {float(fz['loss_best']):.4f} vs null {float(fz['loss_null']):.4f})")
    print("  theta    : " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(spec2.theta_names, theta)))
    print(f"  control  : the NESTED INCUMBENT — the same code with the compact "
          f"share set to exactly zero")
    print(f"  epochs   : z = " + ", ".join(f"{z}" for z in ANCHOR_Z))

    # halo masses. Two different quantities, used for two different jobs:
    #   lmh_cat  the CATALOG log10 M200c -- what the completeness function and
    #            therefore the selection are measured in
    #   lmh_dm   the DiffMAH log10 Mh   -- what the model itself reads
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[sel_rows]                     # (n, 5)
    growth = lmh_cat[:, 0][:, None] - lmh_cat                     # (n, 5), G_k
    print(f"  regressor: catalog log10 M200c for the tilt and the leakage "
          f"(the selection function is measured in it); DiffMAH log10 Mh "
          f"reported alongside at {R_LEAK:g} kpc")

    sn = np.load(SEL_NPZ, allow_pickle=True)
    cuts, ceilings = sn["cuts"], sn["ceilings"]
    curves_comp = [(sn[f"comp_ctr_{k}"], sn[f"comp_val_{k}"]) for k in range(5)]
    print(f"\n  selection function from {SEL_NPZ.parent.parent.name}/"
          f"{SEL_NPZ.name} (exp54 Stage 3.4, unchanged)")
    print(f"    mh-complete cut  log10 M200c > " + "  ".join(f"{c:.2f}" for c in cuts))
    print(f"    epoch's own high-mass plateau " + "  ".join(f"{c:.3f}" for c in ceilings))

    # predictions
    cogs = {}
    for name, th in (("exp63 joint", theta), ("nested incumbent", theta_nested)):
        cache = OUTDIR / f"pred_{name.split()[0]}_{len(recs)}.npy"
        if cache.exists() and not smoke:
            cogs[name] = np.load(cache)
            print(f"  reusing cached prediction {cache.name}")
        else:
            cogs[name] = M2.predict2(spec2, th, curves, F.R_GRID)
            if not smoke:
                np.save(cache, cogs[name])

    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    good &= fitmask.all(1)
    keep = good[:, None] & np.isfinite(lmh_cat)                   # (n, 5) per-epoch
    comp = keep & np.asarray(complete, bool)
    print(f"\n  THE FITTING SAMPLE: {int(good.sum())} galaxies with a finite "
          f"catalog mass at " + "/".join(f"{int(keep[:, k].sum())}" for k in range(5))
          + " per epoch")
    print(f"  the mh-complete AFTER-FIT CHECK keeps "
          + "/".join(f"{int(comp[:, k].sum())}" for k in range(5)) + " per epoch")

    res = {m: residual_table(cogs[m], data) for m in cogs}
    labels = list(res["exp63 joint"].keys())

    # ---- 1. the phenomenon ---------------------------------------------- #
    print(f"\n{RULE}\n1. THE PHENOMENON — the residual's halo-mass tilt, "
          f"fitting sample vs mh-complete\n   tilt = dex of log10(model/truth) "
          f"per dex of log10 M200c; 0 = equally right at every halo mass"
          f"\n{RULE}")
    tilts = {}
    for m in cogs:
        print(f"\n  {m}")
        print(f"  {'quantity':<16}{'epoch':>6}"
              f"{'tilt, fitting sample':>34}{'tilt, mh-complete':>34}")
        tilts[m] = {}
        for lab in labels:
            y = res[m][lab]
            rowset = []
            for k in range(5):
                a = SEL.tilt(y[:, k], lmh_cat[:, k], keep[:, k], seed=k)
                b = SEL.tilt(y[:, k], lmh_cat[:, k], comp[:, k], seed=100 + k)
                rowset.append((a, b))
                print(f"  {lab if k == 0 else '':<16}{ANCHOR_Z[k]:>6.1f}"
                      f"   {a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}] n={a[3]:<5d}"
                      f"   {b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}] n={b[3]:<5d}")
            tilts[m][lab] = rowset

    print(f"\n{THIN}\n  and the medians the tilt is made of — median "
          f"log10(model/truth), dex\n{THIN}")
    for m in cogs:
        print(f"\n  {m}")
        print(f"  {'quantity':<16}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for lab in labels:
            y = res[m][lab]
            for nm, msk in (("fitting sample", keep), ("mh-complete", comp)):
                print(f"  {lab if nm.startswith('fit') else '':<16}{nm:<16}"
                      + "".join(f"{np.nanmedian(np.where(msk[:, k], y[:, k], np.nan)):>+10.4f}"
                                for k in range(5)))

    # ---- 2. the channel -------------------------------------------------- #
    print(f"\n{RULE}\n2. THE CHANNEL — does the residual depend on FUTURE "
          f"GROWTH at fixed halo mass?\n   G_k = log10 Mh(z=0.4) - log10 "
          f"Mh(z_k). Both y and G are stripped of their linear\n   dependence "
          f"on log10 M200c(z_k) first, so what is correlated is the part of "
          f"each\n   that halo mass cannot explain. z=0.4 has no future growth "
          f"and is left blank.\n{RULE}")
    leak, null = {}, {}
    for m in cogs:
        print(f"\n  {m}")
        print(f"  {'quantity':<16}{'epoch':>6}{'partial r':>11}{'dy/dG':>9}{'n':>7}"
              f"   |{'null mean r':>12}{'null 95% of r':>22}   {'outside?':>9}")
        leak[m], null[m] = {}, {}
        for lab in labels:
            rows, nulls = [], []
            for k in range(5):
                r, sl, n = SEL.partial_growth(res[m][lab][:, k], lmh_cat[:, k],
                                              growth[:, k], keep[:, k])
                rows.append((r, sl, n))
                if k == 0:
                    nulls.append((np.nan,) * 6)
                    print(f"  {lab:<16}{ANCHOR_Z[k]:>6.1f}{'—':>11}{'—':>9}{n:>7}"
                          f"   |  (G is identically zero at the selection epoch)")
                    continue
                nul = perm_null_partial(res[m][lab][:, k], lmh_cat[:, k],
                                        growth[:, k], keep[:, k], seed=k)
                nulls.append(nul)
                out = "yes" if (r > nul[2] or r < nul[1]) else "no"
                print(f"  {'':<16}{ANCHOR_Z[k]:>6.1f}{r:>+11.4f}{sl:>+9.4f}{n:>7}"
                      f"   |{nul[0]:>+12.4f}"
                      + f"[{nul[1]:+.4f}, {nul[2]:+.4f}]".rjust(22)
                      + f"{out:>12}")
            leak[m][lab] = rows
            null[m][lab] = nulls

    print(f"\n  A null partial correlation would mean the selection cannot "
          f"produce the residual\n  through this channel, whatever the "
          f"completeness curve shows. The null reshuffles G\n  inside "
          f"equal-count halo-mass bins, so it keeps every shared dependence on "
          f"halo mass\n  the linear stripping leaves behind (which is why its "
          f"mean is not exactly zero) and\n  destroys only which galaxy has "
          f"which growth.")

    # The channel WITHOUT any model in it. The residual y = log10(model/truth)
    # can only depend on future growth because the TRUTH does: nothing in the
    # model reads anything after z_k. Measuring the truth's own dependence
    # states the fact in a form no fit can be blamed for, and it is the reason
    # any epoch-local model must inherit the residual.
    print(f"\n{THIN}\n  THE SAME CHANNEL WITH NO MODEL IN IT — the measured "
          f"stellar mass's own dependence\n  on future growth at fixed halo "
          f"mass. d log10 M*(truth) / dG in dex per dex.\n{THIN}")
    print(f"  {'quantity':<16}{'epoch':>6}{'partial r':>12}"
          f"{'d log10 M* / dG':>18}{'model dy/dG':>14}{'n':>8}")
    truth_leak = {}
    for lab, r_kpc in ((f"M(<{r:g})", r) for r in R_REPORT):
        c = int(np.argmin(np.abs(F.R_GRID - r_kpc)))
        yt = np.log10(np.clip(data[:, list(EPOCHS), c], 1.0, None))
        rows = []
        for k in range(1, 5):
            r, sl, n = SEL.partial_growth(yt[:, k], lmh_cat[:, k], growth[:, k], keep[:, k])
            rows.append((r, sl, n))
            print(f"  {lab if k == 1 else '':<16}{ANCHOR_Z[k]:>6.1f}{r:>+12.4f}"
                  f"{sl:>+18.4f}{leak['exp63 joint'][lab][k][1]:>+14.4f}{n:>8d}")
        truth_leak[lab] = rows
    print(f"\n  The model column is the residual's slope from the table above. "
          f"The truth's own\n  slope is the thing the selection truncates; the "
          f"residual inherits whatever part of\n  it the model fails to "
          f"reproduce from the halo history before z_k alone.")

    # DiffMAH cross-check: the model reads DiffMAH mass, the selection is
    # measured in catalog mass; if the leakage were an artefact of using one
    # in place of the other it would not survive the swap.
    print(f"\n  cross-check at {R_LEAK:g} kpc with DiffMAH log10 Mh as the "
          f"regressor instead of the catalog mass")
    print(f"  {'model':<18}{'epoch':>6}{'partial r':>12}{'dy/dG':>10}")
    for m in cogs:
        for k in range(1, 5):
            r, sl, _ = SEL.partial_growth(res[m][f"M(<{R_LEAK:g})"][:, k],
                                          lmh_dm[:, k], growth[:, k], keep[:, k])
            print(f"  {m if k == 1 else '':<18}{ANCHOR_Z[k]:>6.1f}{r:>+12.4f}{sl:>+10.4f}")

    # ---- 3. the size ----------------------------------------------------- #
    print(f"\n{RULE}\n3. THE SIZE — the tilt the SELECTION ALONE would "
          f"produce, predicted from the\n   measured completeness curve and "
          f"the measured dy/dG, with no reference to the\n   tilt we observe. "
          f"Restricting the mass range moves a slope for reasons that have\n"
          f"   nothing to do with selection, so the comparison is made on the "
          f"SHIFT between\n   the two samples as well as on the full-sample "
          f"tilt itself.\n{RULE}")
    pred_tilt = {}
    for m in cogs:
        print(f"\n  {m} — dy/dG taken at {R_LEAK:g} kpc, tilt quoted at the "
              f"same radius, dex per dex")
        print(f"  {'epoch':>6}{'observed full':>15}{'observed compl':>15}"
              f"{'obs shift':>11}{'pred (bound)':>14}{'pred (corr)':>13}"
              f"{'pred full':>11}{'s_G':>8}")
        pred_tilt[m] = np.full((5, 4), np.nan)
        lab = f"M(<{R_LEAK:g})"
        for k in range(5):
            dydg = leak[m][lab][k][1]
            if not np.isfinite(dydg):
                print(f"  {ANCHOR_Z[k]:>6.1f}   (no future growth at the selection epoch)")
                continue
            (plain, corr), s_g = SEL.predicted_spurious_tilt(
                lmh_cat[:, k], curves_comp[k], float(ceilings[k]), dydg,
                growth[:, k], [keep[:, k], comp[:, k]])
            obs_f, obs_r = tilts[m][lab][k][0][0], tilts[m][lab][k][1][0]
            pred_tilt[m][k] = [plain[0] - plain[1], corr[0] - corr[1], plain[0], corr[0]]
            print(f"  {ANCHOR_Z[k]:>6.1f}{obs_f:>+15.4f}{obs_r:>+15.4f}"
                  f"{obs_f - obs_r:>+11.4f}{plain[0] - plain[1]:>+14.4f}"
                  f"{corr[0] - corr[1]:>+13.4f}{plain[0]:>+11.4f}{s_g:>8.3f}")
        print(f"    `pred (bound)` takes the growth scatter from the "
              f"mh-complete subsample and so\n    understates it wherever the "
              f"sample is truncated; `pred (corr)` divides the\n    truncation "
              f"out with the standard-normal formula and so overstates it in "
              f"the far\n    tail. They bracket what selection alone can do. "
              f"`pred full` is the predicted\n    spurious tilt ON THE FITTING "
              f"SAMPLE — compare it with `observed full`.")

    # the same prediction for every reported quantity, at z = 2 only
    print(f"\n{THIN}\n  every reported quantity at z = 2.0: how much of the "
          f"observed tilt does the\n  selection alone predict?\n{THIN}")
    frac = {}
    for m in cogs:
        print(f"\n  {m}")
        print(f"  {'quantity':<16}{'obs full':>11}{'obs compl':>11}{'obs shift':>11}"
              f"{'pred shift (bound..corr)':>28}{'pred full (bound..corr)':>27}")
        frac[m] = {}
        for lab in labels:
            k = 4
            dydg = leak[m][lab][k][1]
            if not np.isfinite(dydg):
                continue
            (plain, corr), _ = SEL.predicted_spurious_tilt(
                lmh_cat[:, k], curves_comp[k], float(ceilings[k]), dydg,
                growth[:, k], [keep[:, k], comp[:, k]])
            obs_f, obs_r = tilts[m][lab][k][0][0], tilts[m][lab][k][1][0]
            frac[m][lab] = (obs_f, obs_r, plain[0], corr[0], plain[0] - plain[1], corr[0] - corr[1])
            print(f"  {lab:<16}{obs_f:>+11.4f}{obs_r:>+11.4f}{obs_f - obs_r:>+11.4f}"
                  + f"{plain[0] - plain[1]:+.4f} .. {corr[0] - corr[1]:+.4f}".rjust(28)
                  + f"{plain[0]:+.4f} .. {corr[0]:+.4f}".rjust(27))

    # ---- 4. what is left after the selection is removed ------------------ #
    print(f"\n{RULE}\n4. WHAT SURVIVES — the tilt with the leakage channel "
          f"regressed out.\n   The residual is stripped of its linear "
          f"dependence on future growth G_k, and the\n   halo-mass tilt is "
          f"remeasured on what is left. This is not the same thing as\n   "
          f"section 3 (it removes the WHOLE growth dependence, not the part "
          f"the completeness\n   curve attributes to selection) and it is the "
          f"aggressive version of the correction.\n{RULE}")
    resid_tilt = {}
    for m in cogs:
        print(f"\n  {m}")
        print(f"  {'quantity':<16}{'epoch':>6}{'tilt raw':>12}{'tilt | G':>12}"
              f"{'removed':>10}")
        resid_tilt[m] = {}
        for lab in labels:
            rows = []
            for k in range(1, 5):
                y = res[m][lab][:, k]
                ok = keep[:, k] & np.isfinite(y) & np.isfinite(growth[:, k])
                yg = np.full_like(y, np.nan)
                D = np.column_stack([np.ones(int(ok.sum())), growth[ok, k]])
                yg[ok] = y[ok] - D @ np.linalg.lstsq(D, y[ok], rcond=None)[0]
                t_raw = tilts[m][lab][k][0][0]
                t_res = SEL.tilt(yg, lmh_cat[:, k], keep[:, k], seed=k)[0]
                rows.append((t_raw, t_res))
                print(f"  {lab if k == 1 else '':<16}{ANCHOR_Z[k]:>6.1f}"
                      f"{t_raw:>+12.4f}{t_res:>+12.4f}{t_raw - t_res:>+10.4f}")
            resid_tilt[m][lab] = rows

    # ---- save ------------------------------------------------------------ #
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / ("leakage_smoke.npz" if smoke else "leakage.npz")
    np.savez(
        out, labels=np.array(labels), models=np.array(list(cogs)),
        anchor_z=np.array(ANCHOR_Z), cuts=cuts, ceilings=ceilings,
        n_keep=np.array([[int(keep[:, k].sum()) for k in range(5)],
                         [int(comp[:, k].sum()) for k in range(5)]]),
        **{f"tilt_{m.split()[0]}_{lab}": np.array([[list(a), list(b)] for a, b in tilts[m][lab]])
           for m in cogs for lab in labels},
        **{f"leak_{m.split()[0]}_{lab}": np.array(leak[m][lab], float)
           for m in cogs for lab in labels},
        **{f"null_{m.split()[0]}_{lab}": np.array(null[m][lab], float)
           for m in cogs for lab in labels},
        **{f"truthleak_{lab}": np.array(v, float) for lab, v in truth_leak.items()},
        **{f"predtilt_{m.split()[0]}": pred_tilt[m] for m in cogs},
        **{f"restilt_{m.split()[0]}_{lab}": np.array(resid_tilt[m][lab], float)
           for m in cogs for lab in labels},
    )
    print(f"\nwrote {out}")
    if make_fig:
        figure(res, tilts, pred_tilt, frac["exp63 joint"], keep, comp,
               lmh_cat, growth, labels, smoke)
    return tilts, leak, pred_tilt


# --------------------------------------------------------------------------- #
def figure(res, tilts, pred_tilt, frac_z2, keep, comp, lmh_cat, growth, labels, smoke):
    """Four panels: the phenomenon, the channel, the prediction, the remainder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import save_fig, set_style
    from hongshao.qa import _tex
    set_style()
    m0 = "exp63 joint"
    lab0 = f"M(<{R_LEAK:g})"
    #: per quantity at z = 2: the predicted spurious full-sample tilt,
    #: (lower bound, truncation-corrected)
    pred_tilt_q = {l: (v[2], v[3]) for l, v in frac_z2.items()}
    cols = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) the phenomenon: median residual against halo mass at z=2
    a = ax[0, 0]
    k = 4
    y = res[m0][lab0][:, k]
    ok = keep[:, k] & np.isfinite(y)
    edges = np.quantile(lmh_cat[ok, k], np.linspace(0, 1, 9))
    ctr = 0.5 * (edges[:-1] + edges[1:])
    med = [np.median(y[ok & (lmh_cat[:, k] >= edges[i]) & (lmh_cat[:, k] < edges[i + 1])])
           if (ok & (lmh_cat[:, k] >= edges[i]) & (lmh_cat[:, k] < edges[i + 1])).sum() > 10
           else np.nan for i in range(8)]
    a.plot(ctr, med, "o-", color=cols[4], lw=2, label="z = 2.0")
    for j, kk in enumerate((1, 2, 3)):
        yy = res[m0][lab0][:, kk]
        oo = keep[:, kk] & np.isfinite(yy)
        e2 = np.quantile(lmh_cat[oo, kk], np.linspace(0, 1, 9))
        c2 = 0.5 * (e2[:-1] + e2[1:])
        m2 = [np.median(yy[oo & (lmh_cat[:, kk] >= e2[i]) & (lmh_cat[:, kk] < e2[i + 1])])
              if (oo & (lmh_cat[:, kk] >= e2[i]) & (lmh_cat[:, kk] < e2[i + 1])).sum() > 10
              else np.nan for i in range(8)]
        a.plot(c2, m2, "-", color=cols[kk], lw=1.3, alpha=0.8, label=f"z = {ANCHOR_Z[kk]}")
    a.axhline(0, color="0.4", lw=0.8)
    a.set_xlabel(_tex(r"log$_{10}$ M$_{200c}$ at that epoch"))
    a.set_ylabel(_tex(r"median log$_{10}$(model / truth), M$_*$(<103 kpc)"))
    a.set_title("(a) the phenomenon: the residual tilts with halo mass", fontsize=11)
    a.legend(fontsize=8, frameon=False)

    # (b) the channel: residual against future growth at fixed mass, z=2
    b = ax[0, 1]
    for kk in (1, 2, 3, 4):
        yy = res[m0][lab0][:, kk]
        ok2 = keep[:, kk] & np.isfinite(yy) & np.isfinite(growth[:, kk])
        n = int(ok2.sum())
        D = np.column_stack([np.ones(n), lmh_cat[ok2, kk]])
        ry = yy[ok2] - D @ np.linalg.lstsq(D, yy[ok2], rcond=None)[0]
        rg = growth[ok2, kk] - D @ np.linalg.lstsq(D, growth[ok2, kk], rcond=None)[0]
        e = np.quantile(rg, np.linspace(0, 1, 9))
        c = 0.5 * (e[:-1] + e[1:])
        mm = [np.median(ry[(rg >= e[i]) & (rg < e[i + 1])]) if ((rg >= e[i]) & (rg < e[i + 1])).sum() > 10
              else np.nan for i in range(8)]
        b.plot(c, mm, "o-", color=cols[kk], lw=1.6, ms=4, label=f"z = {ANCHOR_Z[kk]}")
    b.axhline(0, color="0.4", lw=0.8); b.axvline(0, color="0.4", lw=0.8)
    b.set_xlabel(_tex(r"future growth G at fixed M$_{200c}$ [dex]"))
    b.set_ylabel(_tex(r"residual at fixed M$_{200c}$ [dex]"))
    b.set_title("(b) the channel: the residual reads future growth", fontsize=11)
    b.legend(fontsize=8, frameon=False)

    # (c) the size: observed, and observed with the predicted spurious tilt
    #     taken off. If the corrected band reaches the mh-complete points, the
    #     selection accounts for the whole difference between the two samples.
    c = ax[1, 0]
    zz = np.arange(1, 5)
    obs = np.array([tilts[m0][lab0][k][0][0] for k in zz])
    obs_c = np.array([tilts[m0][lab0][k][1][0] for k in zz])
    c.plot(zz, obs, "o-", color="#0072B2", lw=2, label="observed, fitting sample")
    c.plot(zz, obs_c, "s--", color="#56B4E9", lw=1.8, ms=7,
           label="observed, mh-complete (no selection acts here)")
    c.fill_between(zz, obs - pred_tilt[m0][zz, 2], obs - pred_tilt[m0][zz, 3],
                   color="#D55E00", alpha=0.28,
                   label="fitting sample with the predicted\nspurious tilt removed")
    c.axhline(0, color="0.4", lw=0.8)
    c.set_xticks(zz); c.set_xticklabels([f"{ANCHOR_Z[k]}" for k in zz])
    c.set_xlabel("redshift")
    c.set_ylabel(_tex("halo-mass tilt of M$_*$(<103 kpc) [dex / dex]"))
    c.set_title("(c) the size: correcting for selection alone", fontsize=11)
    c.legend(fontsize=8, frameon=False, loc="best")

    # (d) the two samples' z = 2 tilt, quantity by quantity
    d = ax[1, 1]
    xs = np.arange(len(labels))
    raw = [tilts[m0][l][4][0][0] for l in labels]
    comp_t = [tilts[m0][l][4][1][0] for l in labels]
    d.bar(xs - 0.2, raw, 0.4, color="#0072B2", label="fitting sample")
    d.bar(xs + 0.2, comp_t, 0.4, color="#56B4E9", label="mh-complete")
    for j, l in enumerate(labels):
        lo, hi = pred_tilt_q.get(l, (np.nan, np.nan))
        d.plot([xs[j] - 0.2, xs[j] - 0.2], [raw[j] - lo, raw[j] - hi],
               color="#D55E00", lw=3, solid_capstyle="butt",
               label=("with the predicted spurious tilt removed" if j == 0 else None))
    d.axhline(0, color="0.4", lw=0.8)
    d.set_xticks(xs)
    d.set_xticklabels([_tex(l) for l in labels], rotation=30, ha="right", fontsize=8)
    d.set_ylabel("halo-mass tilt at z = 2 [dex / dex]")
    d.set_title("(d) every quantity at z = 2", fontsize=11)
    d.legend(fontsize=8, frameon=False)

    fig.suptitle("exp71 / C18 — the z = 2 halo-mass dependence against the "
                 "z = 0.4 progenitor selection", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    save_fig(fig, FIGDIR / ("exp71_c18_smoke" if smoke else "exp71_c18"))
    plt.close(fig)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, make_fig="--no-fig" not in sys.argv)
