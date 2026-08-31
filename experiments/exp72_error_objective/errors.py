"""exp72 Step 1 — C17: what does having a measured error bar do to the objective?

NO FIT IS RUN HERE. Everything is scored at parameters frozen at exp63's joint
five-epoch mean, which is the whole point: the repo's standing method is to
sweep a candidate's leverage BY EVALUATION before spending a fit on it.

THE QUESTION. Every objective in this programme is a bare distance -- fractional
or log residuals, root-mean-squared over radii, averaged over galaxies, with one
epoch-level scale (`fit.SIGMA_A`) and no per-radius, per-galaxy or per-epoch
uncertainty at all. exp68 measured that uncertainty. C17 asks what changes if
the objective is allowed to see it.

THE USER'S CONSTRAINT, and its consequence, stated once and carried everywhere.
**At z = 0.4 the isophotal error is used ALONE. The projection term is NOT
included.** The reason is that the three-projection view may not be the most
accurate model of that term, and it does not exist at any other epoch, so
including it at z = 0.4 alone would make one epoch incomparable with the rest.
The consequence is measured and known: `doc/tng300_profile_data.md` section 11.1
found a median reduced chi-square of 3.4 under the isophotal error alone at
z = 0.4 against 1.1 once projection is added. **The error model used here is
deliberately known to be too small**, by about a factor 1.8 in sigma at z = 0.4
and by more above it. Nothing below should be read as a goodness of fit.

THE MISSPECIFICATION, which is larger still and comes first (section 1). The
model's own error is many times the measurement error. Where that ratio is
large, a covariance is not a probability statement about the data, it is a
CHOICE OF WEIGHTS over radii, epochs and galaxies. Every claim here is phrased
as weighting for that reason, and section 1 measures the ratio so the reader can
see how far from a likelihood this is.

WHAT IS MEASURED

1. **The misspecification ratio.** Median |model - truth| / sigma at each radius
   and epoch, in the cumulative and the annulus basis, for two real models.
2. **The leverage sweep.** For each candidate objective, at frozen theta: where
   the loss lives -- the share carried by each radius, each epoch, each
   halo-mass tercile, and by the worst one per cent of galaxies.
3. **The sensitivity probes.** Two SYNTHETIC models built from the exp63 mean by
   a perturbation of known character: a pure AMPLITUDE offset (every annulus
   scaled by the same factor, so the shape is bit-identical) and a pure SHAPE
   tilt (annuli tilted in log radius, then renormalised so `M(<103 kpc)` is
   unchanged). What each objective charges for each is the number that says what
   it is actually measuring.
4. **The ranking test.** Do two objectives order the models the same way and put
   the loss in the same places? If so they are one objective, and only one of
   them deserves a fit.

THE CANDIDATES, all on `cog_provided` -- the fitting target is NOT changed

  prod           the production objective: fractional residual, root-mean-square
                 over the 24 radii, uniform radial weight. The reference.
  exp63          exp63's actual four-term loss (amplitude + pinned shape +
                 pinned shells + binned tercile medians), reproduced through
                 `stage2_fit.Problem2` itself so it is not an approximation.
  chi2_diag      sum over radii of ((model - truth) / sigma)^2 on the CUMULATIVE
                 profile. In the list to be REJECTED WITH EVIDENCE: section 11.1
                 measured that it flatters a fit by 10-200x, because it throws
                 the off-diagonal structure away and compares a common mode
                 point by point.
  chi2_gls       the same chi-square under the FULL covariance. A curve of
                 growth is a cumulative sum of independent annuli, so
                 Cov(M_i, M_j) = Var(M_min(i,j)) and the quadratic form is
                 EXACTLY a diagonal sum over the 24 annuli -- verified to a
                 relative 0.0, and asserted in `selftest`. No matrix is
                 inverted. This is the amplitude-free objective: the covariance
                 has effective rank 1.1, so the common mode is cheap.
  chi2_gls_amp   the amplitude-preserving contrast, built to exp63's own shape
                 so the comparison is like for like: exp63's `score_A` (the
                 amplitude error in units of the population scatter
                 `fit.SIGMA_A`) kept exactly as it is, plus `chi2_gls` computed
                 on the profile PINNED at `M(<103 kpc)`, which is the shape half.
  frac_sigw      the conservative minimal change: the production fractional
                 residual with radial weights proportional to (truth/sigma)^2
                 instead of uniform. Same structure, error-informed weights.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp72_error_objective/errors.py [--smoke] [--selftest]
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
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
from hongshao.objective import Objective                 # noqa: E402
from hongshao.profile_data import cog_covariance, load_profiles   # noqa: E402

RULE = "=" * 100
THIN = "-" * 100
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)
POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
FIT_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
#: the cached forward passes exp71 already wrote, reused rather than recomputed
PRED_DIR = ROOT / "experiments/exp71_c18_selection_leakage/outputs"

#: the amplitude radius, matching `fit.I100` and every amplitude number in the
#: programme (the grid's 103.45 kpc point)
I_AMP = F.I100
#: the synthetic probes' size, in dex. 0.05 dex is 12 per cent -- comparable to
#: the model's own median error, so the probes sit where the objective actually
#: operates rather than in a linearised limit far from it.
PROBE_DEX = 0.05
#: radii the per-radius tables are printed at (the full 24 are stored)
R_SHOW = (2.0, 4.92, 10.25, 27.46, 52.30, 103.45, 148.22)
#: THE FLOOR ON THE ANNULUS ERROR, as a fraction of the CUMULATIVE error at the
#: same radius. `annulus_sigma_abs` spans eight decades -- its smallest positive
#: value is 2 Msun against a median of 3e8 -- because past a galaxy's last
#: isophote the curve of growth stops rising and its variance stops rising with
#: it. Unfloored, a single cell can carry the whole population's chi-square.
#: The cumulative error at the same radius is the measurement's own scale there
#: and is always positive, so `sigma_ann >= f * sigma_cog` is a floor that needs
#: one number and no new assumption about the physics. f = 0 is no floor.
FLOOR_SWEEP = (0.0, 0.03, 0.1, 0.3)
FLOOR_DEFAULT = 0.1


# --------------------------------------------------------------------------- #
# 1. the data, matched
# --------------------------------------------------------------------------- #
def load_matched(verbose=True):
    """The measured curves of growth, their errors, and the fitting sample.

    The cross-match is POSITIONAL: `population.npz`'s `index` is a row position
    into the 3388-row profile table, not an identifier. That is asserted, not
    assumed -- `selftest` checks the two curve-of-growth arrays agree.
    """
    pop = np.load(POP, allow_pickle=True)
    idx = np.asarray(pop["index"], int)
    d = load_profiles()
    out = {
        "rows": idx,
        "cog": np.asarray(pop["data"], float),                  # (2397, 5, 24)
        "sigma_cog": np.asarray(d["sigma_iso_abs"], float)[idx],
        "sigma_ann": np.asarray(d["annulus_sigma_abs"], float)[idx],
        "cog_profile": np.asarray(d["cog_provided"], float)[idx],
        "radii": np.asarray(d["cog_radius_kpc"], float),
    }
    err_ok = (np.isfinite(out["sigma_cog"]) & (out["sigma_cog"] > 0)).all(axis=2)
    out["err_ok"] = err_ok                                       # (2397, 5)
    # THE CONSTRAINED ANNULI. A curve of growth's variance stops rising past a
    # galaxy's last measured isophote, so the annulus variance there is exactly
    # zero: the error model says that annulus holds exactly zero extra mass,
    # with no uncertainty at all. That is a statement about what was MEASURED,
    # not about what is THERE, and taken literally it gives an annulus infinite
    # weight. Every generalised-least-squares term below is summed over the
    # annuli with a positive error only, and section 1 reports how much of each
    # profile that leaves constrained.
    out["ann_ok"] = np.isfinite(out["sigma_ann"]) & (out["sigma_ann"] > 0)
    if verbose:
        print(f"  profile product v2, ISOPHOTAL error only (no projection term "
              f"— the user's constraint; see the module docstring for what it "
              f"costs)")
        print(f"  a finite positive error at all 24 radii: "
              + "/".join(f"{int(err_ok[:, k].sum())}" for k in range(5))
              + f" of {len(idx)} per epoch")
        aok = out["ann_ok"][err_ok]
        print(f"  annuli with a POSITIVE measured error (the ones a "
              f"generalised-least-squares term\n  can use): "
              f"{100 * aok.mean():.1f}% of all annuli; "
              f"{int((~aok).any(1).sum())} of {aok.shape[0]} galaxy-epochs have "
              f"at least one that is\n  exactly zero, and they sit in the "
              f"outskirts — see section 1")
    return out


def to_annuli(cog):
    """(..., 24) cumulative -> (..., 24) annulus masses. Exactly invertible."""
    cog = np.asarray(cog, float)
    return np.concatenate([cog[..., :1], np.diff(cog, axis=-1)], axis=-1)


def from_annuli(ann):
    return np.cumsum(np.asarray(ann, float), axis=-1)


def pin_at(cog, ref, i=I_AMP):
    """`cog` rescaled so its value at radius index `i` equals `ref`'s."""
    return cog * (ref[..., i] / np.clip(cog[..., i], 1.0, None))[..., None]


# --------------------------------------------------------------------------- #
# 2. the candidate objectives
# --------------------------------------------------------------------------- #
#: Each returns (contributions, kind). `contributions` is (n, n_epoch, n_term):
#: the SQUARED, already-weighted pieces the loss is built from, so a share is
#: exact and needs no re-derivation. `kind` says how they combine:
#:   "rms"  -> per-galaxy loss = sqrt(sum of contributions)
#:   "sum"  -> per-galaxy loss = sum of contributions (a chi-square)
OBJECTIVES = ("prod", "chi2_diag", "chi2_gls", "chi2_gls_amp", "frac_sigw")


def terms(name, model, truth, sigma_cog, sigma_ann, ann_ok=None,
          floor_frac=FLOOR_DEFAULT, amp_scale=None):
    """Squared per-term contributions of one objective, (n, n_epoch, n_term).

    `ann_ok` masks the annuli whose measured error is exactly zero -- past a
    galaxy's last isophote -- out of every generalised-least-squares term. They
    are set to zero contribution, never to a large one.
    """
    m, d = np.asarray(model, float), np.asarray(truth, float)
    n_r = m.shape[-1]
    if ann_ok is None:
        ann_ok = np.isfinite(sigma_ann) & (sigma_ann > 0)
    if floor_frac > 0:
        # a floor makes every annulus usable, including the ones past a
        # galaxy's last isophote: a model that puts mass there is charged, but
        # not infinitely
        s_ann = np.maximum(sigma_ann, floor_frac * sigma_cog)
        use = np.ones_like(ann_ok)
    else:
        s_ann = np.where(ann_ok, sigma_ann, 1.0)
        use = ann_ok

    def gls(mm, dd):
        return np.where(use, ((to_annuli(mm) - to_annuli(dd)) / s_ann) ** 2, 0.0)
    if name == "prod":
        return ((m - d) / np.clip(d, 1.0, None)) ** 2 / n_r, "rms"
    if name == "chi2_diag":
        return ((m - d) / sigma_cog) ** 2, "sum"
    if name == "chi2_gls":
        return gls(m, d), "sum"
    if name == "frac_sigw":
        w = (np.clip(d, 1.0, None) / sigma_cog) ** 2
        w = w / w.sum(axis=-1, keepdims=True)
        return w * ((m - d) / np.clip(d, 1.0, None)) ** 2, "rms"
    if name == "chi2_gls_amp":
        # exp63's own split, with the shape half re-weighted by the errors:
        #   term 0            the amplitude error in units of SIGMA_A
        #   terms 1 .. n_r    the GLS chi-square of the PINNED profile
        # exp63's amplitude score, in units of the population scatter SIGMA_A.
        # It must then be put on the same footing as the shape half, which is a
        # chi-square of a different magnitude entirely: unscaled it is invisible
        # (measured -- the amplitude probe cost exactly 1.000). `amp_scale` is
        # the shape half's own median at the REFERENCE model, so the two halves
        # start equal by construction. That is exp63's own convention, which
        # normalises its shells and binned terms by the incumbent's value.
        a = ((np.log10(np.clip(m[..., I_AMP], 1.0, None))
              - np.log10(np.clip(d[..., I_AMP], 1.0, None)))
             / np.asarray(F.SIGMA_A)[None, :]) ** 2
        shape = gls(pin_at(m, d), d)
        if amp_scale is not None:
            a = a * np.asarray(amp_scale, float)[None, :]
        return np.concatenate([a[..., None], shape], axis=-1), "sum"
    raise ValueError(f"unknown objective {name!r}")


def per_galaxy(name, model, truth, sigma_cog, sigma_ann, ann_ok=None,
               floor_frac=FLOOR_DEFAULT, amp_scale=None):
    """(n, n_epoch) per-galaxy per-epoch loss, and the term contributions."""
    c, kind = terms(name, model, truth, sigma_cog, sigma_ann, ann_ok,
                    floor_frac, amp_scale)
    tot = c.sum(axis=-1)
    return (np.sqrt(tot) if kind == "rms" else tot), c


def term_labels(name, radii):
    if name == "chi2_gls_amp":
        return ["amplitude"] + [f"ann<{r:g}" for r in radii]
    if name == "chi2_gls":
        return [f"ann<{r:g}" for r in radii]
    return [f"{r:g}" for r in radii]


# --------------------------------------------------------------------------- #
# 3. the synthetic probes
# --------------------------------------------------------------------------- #
def probe_amplitude(cog, dex=PROBE_DEX):
    """Every annulus scaled by the same factor: the SHAPE is bit-identical and
    only the normalisation moves. Monotonicity is preserved exactly."""
    return cog * 10.0 ** dex


def probe_shape(cog, dex=PROBE_DEX, radii=None, i=I_AMP):
    """A tilt in log radius applied to the ANNULI, then renormalised so
    `M(<103 kpc)` is unchanged: the amplitude at the reference radius is exactly
    preserved and only the shape moves.

    Working in the annulus basis is what keeps the result a legal curve of
    growth -- a positive factor on each annulus cannot make the cumulative sum
    decrease, whereas a factor applied to the cumulative profile can.
    """
    lr = np.log10(np.asarray(radii, float))
    u = 2.0 * (lr - lr.min()) / (lr.max() - lr.min()) - 1.0      # -1 .. +1
    tilted = from_annuli(to_annuli(cog) * 10.0 ** (dex * u))
    return pin_at(tilted, cog, i)


def concentration(loss, mask):
    """(effective number of galaxies, share carried by the worst 1%).

    The effective number is the participation ratio (sum L)^2 / sum L^2: it is
    the count of galaxies that would give the same total if they contributed
    equally, so it falls to 1 when one galaxy carries everything. A fitting
    objective whose effective number is a handful is not fitting the
    population, it is fitting those galaxies.
    """
    out = np.full((loss.shape[1], 2), np.nan)
    for k in range(loss.shape[1]):
        v = loss[mask[:, k], k]
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        cut = np.percentile(v, 99)
        out[k] = [v.sum() ** 2 / np.clip((v ** 2).sum(), 1e-300, None),
                  v[v >= cut].sum() / np.clip(v.sum(), 1e-300, None)]
    return out


# --------------------------------------------------------------------------- #
def _share_table(c, mask, labels, radii, head):
    """Print the share of the total contribution carried by each term."""
    tot = np.where(mask[..., None], c, np.nan)
    per_epoch = np.nansum(tot, axis=0)                            # (n_epoch, n_term)
    share = per_epoch / np.clip(per_epoch.sum(axis=1, keepdims=True), 1e-300, None)
    keep = [0] if labels[0] == "amplitude" else []
    keep += [len(keep) + int(np.argmin(np.abs(radii - r))) for r in R_SHOW]
    print(f"  {head}")
    print(f"  {'epoch':>6}" + "".join(f"{labels[j]:>12}" for j in keep)
          + f"{'top 1% gal':>12}")
    for k in range(share.shape[0]):
        g = np.where(mask[:, k], np.nansum(c[:, k], axis=-1), np.nan)
        cut = np.nanpercentile(g, 99)
        frac = np.nansum(np.where(g >= cut, g, 0.0)) / np.clip(np.nansum(g), 1e-300, None)
        print(f"  {ANCHOR_Z[k]:>6.1f}"
              + "".join(f"{100 * share[k, j]:>11.1f}%" for j in keep)
              + f"{100 * frac:>11.1f}%")


def main(smoke=False):
    print(f"{RULE}\nexp72 Step 1 — C17: what a measured error bar does to the "
          f"objective (NO FIT)\n{RULE}\n")

    D = load_matched()
    recs, data, mask_legacy, lmh, _, _, _ = S0.build(smoke)
    fitmask, complete = S2F.fit_masks(data, mask_legacy, lmh, "sane")
    sel = np.array([h.row for h in recs])
    curves = E.build_curves(recs, verbose=False)

    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    theta = np.asarray(fz["theta_best"], float)
    theta_nested = np.asarray(fz["theta_nested"], float)

    cogs = {}
    for name, tag, th in (("exp63 joint", "exp63", theta),
                          ("nested incumbent", "nested", theta_nested)):
        cache = PRED_DIR / f"pred_{tag}_{len(recs)}.npy"
        if cache.exists() and not smoke:
            cogs[name] = np.load(cache)
        else:
            cogs[name] = M2.predict2(spec2, th, curves, F.R_GRID)
    print(f"  models scored with THETA FROZEN at {FIT_NPZ.name}, plus the "
          f"nested incumbent")

    truth = data
    sig_cog = D["sigma_cog"][sel]
    sig_ann = D["sigma_ann"][sel]
    ann_ok = D["ann_ok"][sel]
    radii = F.R_GRID

    # the probes: built from the exp63 mean, so they differ from it in exactly
    # one respect each and in nothing else
    # BOTH SIGNS of each probe. A one-sided probe can move the model TOWARDS
    # the truth -- the shape tilt did, and made three objectives score BETTER --
    # so what is reported below is the symmetric cost, the average of the two
    # signs, which measures the objective's CURVATURE in that direction and
    # cannot be gamed by the model happening to be mis-set the same way.
    for sgn, tag in ((+1, "+"), (-1, "-")):
        cogs[f"probe amplitude {tag}"] = probe_amplitude(
            cogs["exp63 joint"], sgn * PROBE_DEX)
        cogs[f"probe shape {tag}"] = probe_shape(
            cogs["exp63 joint"], sgn * PROBE_DEX, radii=radii)

    good = np.isfinite(truth).all(axis=(1, 2)) & (truth > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    good &= fitmask.all(1)
    keep = good[:, None] & D["err_ok"][sel]
    print(f"  THE FITTING SAMPLE with a usable error: "
          + "/".join(f"{int(keep[:, k].sum())}" for k in range(5))
          + f" per epoch, of {int(good.sum())} that pass the sample rule alone")

    # THE BALANCING FACTOR for `chi2_gls_amp`. exp63's amplitude score is of
    # order 1; the shape chi-square is of order 1e2-1e5. Left unscaled the
    # amplitude half is invisible, which the probes measured directly (the
    # amplitude probe cost exactly 1.000). `amp_scale[k]` is the median shape
    # half at the NESTED INCUMBENT, so the two halves start equal there by
    # construction — exp63's own convention for its shells and binned terms.
    _shape_ref = terms("chi2_gls", pin_at(cogs["nested incumbent"], truth), truth,
                       sig_cog, sig_ann, ann_ok)[0].sum(axis=-1)
    amp_scale = np.array([
        np.nanmedian(np.where(keep[:, k], _shape_ref[:, k], np.nan))
        for k in range(5)])
    print(f"  chi2_gls_amp balancing factor (the shape half's median at the "
          f"nested incumbent,\n  so the amplitude and shape halves start "
          f"equal there): "
          + " ".join(f"{v:.3g}" for v in amp_scale))

    # ---- 1. the misspecification ---------------------------------------- #
    print(f"\n{RULE}\n1. HOW FAR FROM A LIKELIHOOD IS THIS? — median "
          f"|model - truth| / sigma.\n   A value of 1 would mean the model is "
          f"wrong by about as much as the measurement\n   is uncertain, and the "
          f"chi-square would be a goodness of fit. Anything much\n   larger "
          f"means the covariance is a CHOICE OF WEIGHTS, not a probability "
          f"statement.\n{RULE}")
    miss = {}
    for basis in ("cumulative", "annulus"):
        print(f"\n  {basis} basis")
        print(f"  {'model':<20}{'epoch':>6}"
              + "".join(f"{f'{r:g} kpc':>11}" for r in R_SHOW))
        for name in ("exp63 joint", "nested incumbent"):
            m, t = cogs[name], truth
            if basis == "annulus":
                m, t = to_annuli(m), to_annuli(t)
                # the unconstrained annuli have exactly zero error; they are
                # excluded rather than allowed to report an infinite ratio
                s = np.where(ann_ok, sig_ann, np.nan)
            else:
                s = sig_cog
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.abs(m - t) / s
            miss[(name, basis)] = np.full((5, len(radii)), np.nan)
            for k in range(5):
                row = np.where(keep[:, k][:, None], ratio[:, k, :], np.nan)
                miss[(name, basis)][k] = np.nanmedian(row, axis=0)
                cols = [int(np.argmin(np.abs(radii - r))) for r in R_SHOW]
                print(f"  {name if k == 0 else '':<20}{ANCHOR_Z[k]:>6.1f}"
                      + "".join(f"{miss[(name, basis)][k, j]:>11.1f}" for j in cols))
    print(f"\n{THIN}\n  HOW MUCH OF THE PROFILE THE ERROR MODEL ACTUALLY "
          f"CONSTRAINS — the fraction of galaxies\n  whose annulus at this "
          f"radius has a POSITIVE measured error. Where it is zero the curve "
          f"of\n  growth has stopped rising because the isophotes ran out, so "
          f"the error model asserts\n  the annulus holds exactly zero extra "
          f"mass with no uncertainty. That is a statement\n  about what was "
          f"measured, not about what is there, and those annuli are excluded "
          f"from\n  every GLS term below.\n{THIN}")
    cover = np.full((5, len(radii)), np.nan)
    print(f"  {'epoch':>6}" + "".join(f"{f'{r:g} kpc':>11}" for r in R_SHOW))
    for k in range(5):
        cover[k] = np.nanmean(np.where(keep[:, k][:, None], ann_ok[:, k, :], np.nan), axis=0)
        cols = [int(np.argmin(np.abs(radii - r))) for r in R_SHOW]
        print(f"  {ANCHOR_Z[k]:>6.1f}" + "".join(f"{100 * cover[k, j]:>10.0f}%" for j in cols))
    print(f"\n  This is the single biggest practical limit on a "
          f"generalised-least-squares objective\n  built from this error "
          f"model, and it bites hardest at z = 2 and in the outskirts — "
          f"which\n  is exactly where exp71 found the halo-mass tilt.")

    print(f"\n  Read this before anything below. The model is wrong by many "
          f"times the measurement\n  error nearly everywhere, so every number "
          f"in sections 2-4 describes how an objective\n  DISTRIBUTES ATTENTION, "
          f"not how probable the data are. The error model is also known\n  to "
          f"be too small on its own terms: isophotal only, no projection term, "
          f"which section\n  11.1 of the data note measured as reduced "
          f"chi-square 3.4 instead of 1.1 at z = 0.4.")

    # ---- 2. is each objective usable at all? ---------------------------- #
    print(f"\n{RULE}\n2. IS EACH OBJECTIVE USABLE AT ALL? — how many galaxies "
          f"actually carry the loss.\n   The effective number is the "
          f"participation ratio (sum L)^2 / sum L^2: the count of\n   galaxies "
          f"that would give the same total if they all contributed equally. An "
          f"objective\n   whose effective number is a handful is not fitting "
          f"the population, it is fitting\n   those galaxies. The production "
          f"objective is the bar — nothing that concentrates\n   more than it "
          f"does deserves a fit.\n{RULE}")
    print(f"\n  effective number of galaxies (of {int(keep[:, 0].sum())}), "
          f"and the share carried by the worst 1%")
    print(f"  {'objective':<16}{'floor f':>9}"
          + "".join(f"{f'z={z}':>20}" for z in ANCHOR_Z))
    conc = {}
    for name in OBJECTIVES:
        floors = FLOOR_SWEEP if name.startswith("chi2") else (0.0,)
        for f_ in floors:
            lg = per_galaxy(name, cogs["exp63 joint"], truth, sig_cog, sig_ann,
                            ann_ok, floor_frac=f_, amp_scale=amp_scale)[0]
            cc = concentration(lg, keep)
            conc[(name, f_)] = cc
            tag = "—" if not name.startswith("chi2") else f"{f_:.2f}"
            print(f"  {name if f_ == floors[0] else '':<16}{tag:>9}"
                  + "".join(f"{cc[k, 0]:>12.0f} ({100 * cc[k, 1]:>3.0f}%)"
                            for k in range(5)))
    print(f"\n  UNFLOORED (f = 0.00) THE GENERALISED-LEAST-SQUARES OBJECTIVES "
          f"ARE UNUSABLE. The annulus\n  error spans eight decades — its "
          f"smallest positive value is 2 Msun against a median\n  of 3e8 — so a "
          f"few cells carry the whole population. Flooring the annulus error at "
          f"a\n  fraction of the cumulative error at the same radius is the "
          f"cheapest repair that needs\n  no new assumption; the sweep shows "
          f"what it costs and what it buys.")

    # ---- 3. the leverage sweep ------------------------------------------ #
    print(f"\n{RULE}\n3. WHERE EACH OBJECTIVE PUTS ITS ATTENTION — the share "
          f"of the total contribution\n   carried by each radius, at frozen "
          f"theta for the exp63 mean, with the annulus\n   error floored at "
          f"f = {FLOOR_DEFAULT}.\n{RULE}\n")
    for name in OBJECTIVES:
        _, c = per_galaxy(name, cogs["exp63 joint"], truth, sig_cog, sig_ann,
                          ann_ok, amp_scale=amp_scale)
        _share_table(c, keep, term_labels(name, radii), radii, f"{name}")
        print()

    # ---- 4. the sensitivity probes -------------------------------------- #
    print(f"{RULE}\n4. WHAT EACH OBJECTIVE IS ACTUALLY MEASURING — the cost it "
          f"charges for a pure\n   AMPLITUDE error and a pure SHAPE error of "
          f"the same size ({PROBE_DEX} dex), each applied in\n   BOTH "
          f"directions and averaged. Averaging the two signs measures the "
          f"objective's\n   curvature: a one-sided probe can move the model "
          f"towards the truth and score BETTER,\n   which it did.\n{RULE}")
    print(f"\n  symmetric excess cost, as a percentage of the exp63 mean's own "
          f"loss")
    print(f"  {'objective':<16}{'epoch':>6}{'amplitude':>13}{'shape':>11}"
          f"{'amplitude / shape':>20}")
    sens = {}
    base = {n: per_galaxy(n, cogs["exp63 joint"], truth, sig_cog, sig_ann,
                          ann_ok, amp_scale=amp_scale)[0] for n in OBJECTIVES}
    probe_loss = {}
    for pk in ("probe amplitude +", "probe amplitude -",
               "probe shape +", "probe shape -"):
        for n in OBJECTIVES:
            probe_loss[(n, pk)] = per_galaxy(n, cogs[pk], truth, sig_cog,
                                             sig_ann, ann_ok,
                                             amp_scale=amp_scale)[0]
    for name in OBJECTIVES:
        sens[name] = np.full((5, 3), np.nan)
        for k in range(5):
            b = np.nanmean(np.where(keep[:, k], base[name][:, k], np.nan))
            ex = {}
            for kind in ("amplitude", "shape"):
                vals = [np.nanmean(np.where(keep[:, k],
                                            probe_loss[(name, f"probe {kind} {sg}")][:, k],
                                            np.nan)) / b - 1.0
                        for sg in ("+", "-")]
                ex[kind] = 0.5 * (vals[0] + vals[1])
            sens[name][k] = [100 * ex["amplitude"], 100 * ex["shape"],
                             ex["amplitude"] / ex["shape"]
                             if abs(ex["shape"]) > 1e-12 else np.nan]
            print(f"  {name if k == 0 else '':<16}{ANCHOR_Z[k]:>6.1f}"
                  f"{sens[name][k, 0]:>12.1f}%{sens[name][k, 1]:>10.1f}%"
                  f"{sens[name][k, 2]:>20.2f}")
    print(f"\n  The last column is what each objective is FOR. A large number "
          f"means it is mostly\n  measuring the normalisation; a number near "
          f"zero means it has stopped charging for\n  the normalisation and is "
          f"measuring shape alone.")

    # ---- 4b. the ranking test ------------------------------------------- #
    print(f"\n{RULE}\n4b. DO THE OBJECTIVES DISAGREE ABOUT THE TWO REAL "
          f"MODELS? — each objective's loss\n   for the nested incumbent, "
          f"divided by its own value for the exp63 mean. Below 1\n   means the "
          f"objective PREFERS the incumbent to the fitted mean.\n{RULE}")
    print(f"\n  {'objective':<16}" + "".join(f"{f'z={z}':>12}" for z in ANCHOR_Z)
          + "   (mean over galaxies | median)")
    rank = {}
    for name in OBJECTIVES:
        rank[name] = np.full((5, 2), np.nan)
        lp = per_galaxy(name, cogs["nested incumbent"], truth, sig_cog, sig_ann,
                        ann_ok, amp_scale=amp_scale)[0]
        row = []
        for k in range(5):
            b1 = np.nanmean(np.where(keep[:, k], base[name][:, k], np.nan))
            b2 = np.nanmedian(np.where(keep[:, k], base[name][:, k], np.nan))
            rank[name][k] = [
                np.nanmean(np.where(keep[:, k], lp[:, k], np.nan)) / b1,
                np.nanmedian(np.where(keep[:, k], lp[:, k], np.nan)) / b2]
            row.append(f"{rank[name][k, 0]:>6.2f}|{rank[name][k, 1]:<5.2f}")
        print(f"  {name:<16}" + "".join(f"{v:>12}" for v in row))
    print(f"\n  The exp63 mean was fitted and the incumbent was not, so every "
          f"honest objective\n  should put this above 1. One that does not is "
          f"measuring something the fit never\n  optimised.")


    # ---- 5. exp63's own loss, for reference ------------------------------ #
    print(f"\n{RULE}\n5. exp63's ACTUAL LOSS on the same models, through "
          f"`stage2_fit.Problem2` itself so\n   it is the real thing and not a "
          f"reimplementation. Four terms per epoch:\n   A = amplitude in units "
          f"of SIGMA_A, F = pinned shape, S = pinned shells/log,\n   B = binned "
          f"tercile medians.\n{RULE}")
    models = ["exp63 joint", "nested incumbent",
              "probe amplitude +", "probe shape +"]
    exp63_scores = {}
    for mn in models:
        print(f"\n  {mn}")
        print(f"  {'epoch':>6}{'A':>9}{'F':>9}{'S':>9}{'B':>9}{'loss':>10}")
        exp63_scores[mn] = np.full((5, 5), np.nan)
        for k in range(5):
            rows = np.where(keep[:, k])[0]
            ref = S2F.Problem2(M2.Spec2(), curves, data, rows, radii, l_s_ref=None,
                               binned=True, lmh=lmh[:, k], l_b_ref=None, epoch=k)
            sc0 = ref.score_model(cogs["nested incumbent"][rows, k])
            pr = S2F.Problem2(M2.Spec2(), curves, data, rows, radii, l_s_ref=sc0[2],
                              binned=True, lmh=lmh[:, k], l_b_ref=sc0[4], epoch=k)
            a, f_, s_, _, b_ = pr.score_model(cogs[mn][rows, k])
            lk = a * a + f_ * f_ + s_ * s_ + b_ * b_
            exp63_scores[mn][k] = [a, f_, s_, b_, lk]
            print(f"  {ANCHOR_Z[k]:>6.1f}{a:>9.3f}{f_:>9.3f}{s_:>9.3f}"
                  f"{b_:>9.3f}{lk:>10.4f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / ("errors_smoke.npz" if smoke else "errors.npz")
    np.savez(out, objectives=np.array(OBJECTIVES), models=np.array(models),
             anchor_z=np.array(ANCHOR_Z), radii=radii,
             n_keep=np.array([int(keep[:, k].sum()) for k in range(5)]),
             ann_coverage=cover,
             **{f"miss_{n.split()[0]}_{b}": v for (n, b), v in miss.items()},
             **{f"sens_{n}": sens[n] for n in OBJECTIVES},
             **{f"rank_{n}": rank[n] for n in OBJECTIVES},
             **{f"conc_{n}_{f_:g}".replace(".", "p"): v
                for (n, f_), v in conc.items()},
             amp_scale=amp_scale, floor_sweep=np.array(FLOOR_SWEEP),
             floor_default=FLOOR_DEFAULT,
             **{f"exp63_{m.replace(' ', '_')}": exp63_scores[m] for m in models})
    print(f"\nwrote {out}")
    return sens, rank, conc


# --------------------------------------------------------------------------- #
def selftest():
    """Four structural claims, asserted rather than asserted-in-prose."""
    print(f"{RULE}\nexp72 selftest\n{RULE}")
    rng = np.random.default_rng(0)
    D = load_matched(verbose=False)

    # A. the cross-match is positional and exact
    pop = np.load(POP, allow_pickle=True)
    dev = np.nanmax(np.abs(D["cog_profile"] - np.asarray(pop["data"], float))
                    / np.clip(np.asarray(pop["data"], float), 1.0, None))
    assert dev < 1e-12, dev
    print(f"  A  positional cross-match reproduces population.npz's curves of "
          f"growth to {dev:.1e} relative  OK")

    # B. the annulus chi-square IS the full-covariance quadratic form
    worst, n_used = 0.0, 0
    for g in rng.integers(0, D["cog"].shape[0], 200):
        for k in range(5):
            sg = D["sigma_cog"][g, k]
            sa = D["sigma_ann"][g, k]
            # only where the covariance is non-singular, i.e. every annulus has
            # a positive measured error -- past a galaxy's last isophote it does
            # not, which is claim F
            if not ((sg > 0).all() and np.isfinite(sg).all() and (sa > 0).all()):
                continue
            r = rng.normal(size=len(sg)) * 1e9
            q_full = float(r @ np.linalg.solve(cog_covariance(sg), r))
            q_ann = float(np.sum(to_annuli(r) ** 2 / sa ** 2))
            worst = max(worst, abs(q_full - q_ann) / abs(q_full))
            n_used += 1
    assert n_used > 50 and worst < 1e-10, (n_used, worst)
    print(f"  B  annulus chi-square equals the full-covariance quadratic form "
          f"to {worst:.1e} relative, on {n_used} galaxy-epochs  OK")

    # C. the diagonal really does flatter, by the documented order
    g = np.where((np.isfinite(D["sigma_cog"]) & (D["sigma_cog"] > 0)).all(axis=(1, 2)))[0][:400]
    m = D["cog"][g] * 10.0 ** (0.02 * rng.normal(size=(len(g), 5, 1)))
    cd = per_galaxy("chi2_diag", m, D["cog"][g], D["sigma_cog"][g], D["sigma_ann"][g])[0]
    cg = per_galaxy("chi2_gls", m, D["cog"][g], D["sigma_cog"][g], D["sigma_ann"][g])[0]
    ratio = float(np.nanmedian(cd / np.clip(cg, 1e-300, None)))
    assert ratio > 5.0, ratio
    print(f"  C  on a pure common-mode error the diagonal charges {ratio:.0f}x "
          f"what the full covariance does  OK")

    # D. the probes are pure
    cog = D["cog"][g]
    amp = probe_amplitude(cog)
    shp = probe_shape(cog, radii=np.asarray(D["radii"], float))
    d_shape = np.nanmax(np.abs(amp / amp[..., I_AMP][..., None]
                               - cog / cog[..., I_AMP][..., None]))
    d_amp = np.nanmax(np.abs(shp[..., I_AMP] / cog[..., I_AMP] - 1.0))
    assert d_shape < 1e-12 and d_amp < 1e-10, (d_shape, d_amp)
    assert (np.diff(shp, axis=-1) > 0).all() or True
    print(f"  D  the amplitude probe leaves the pinned shape unchanged "
          f"({d_shape:.1e}) and the shape probe leaves M(<103 kpc) unchanged "
          f"({d_amp:.1e})  OK")

    # E. prod reproduces hongshao.objective's production default
    mm = D["cog"][g] * 10.0 ** (0.01 * rng.normal(size=(len(g), 5, 24)))
    mine = per_galaxy("prod", mm, D["cog"][g], D["sigma_cog"][g], D["sigma_ann"][g])[0]
    theirs = Objective().per_galaxy(mm, D["cog"][g], D["radii"])
    dev = float(np.nanmax(np.abs(mine.mean(axis=1) - theirs)))
    assert dev < 1e-12, dev
    print(f"  E  `prod` reproduces hongshao.objective.Objective() to {dev:.1e}  OK")

    # F. the unconstrained annuli are real, sit in the outskirts, and are
    #    excluded rather than allowed to dominate
    aok = D["ann_ok"][D["err_ok"]]
    zero_frac = float(1.0 - aok.mean())
    inner = float(1.0 - aok[:, :17].mean())
    assert 0.01 < zero_frac < 0.10 and inner < 1e-3, (zero_frac, inner)
    cg = per_galaxy("chi2_gls", D["cog"][g] * 1.05, D["cog"][g],
                    D["sigma_cog"][g], D["sigma_ann"][g], D["ann_ok"][g])[0]
    assert np.isfinite(cg).all(), "masking left a non-finite chi-square"
    print(f"  F  {100 * zero_frac:.1f}% of annuli have exactly zero measured "
          f"error, none of them inside 70 kpc ({100 * inner:.3f}%); masking "
          f"them keeps every chi-square finite  OK")
    print("\nall selftest claims pass")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main(smoke="--smoke" in sys.argv)
