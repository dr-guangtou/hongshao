"""exp53 — can a DEPOSITION-ONLY kernel build the extended stellar halo?

The question this branch is actually for (user, 2026-08-21). exp53's central
finding — that removing transport barely moves the compact-galaxy central
deficit — answers a question nobody was asking: the central region was never
the target, it would only have been a pleasant surprise. **The target is the
OUTSKIRT.** Can a model that only deposits new material reproduce the extended
stellar halo of massive galaxies, and its evolution, as well as the
deposition + transport model does?

That needs a sharper instrument than the summary tables. Three things are
measured here, each with a galaxy bootstrap, each split by stellar mass, and
each with a PAIRED per-galaxy comparison against the incumbent — paired because
both models are fitted to the same galaxies, which removes the
galaxy-to-galaxy variance that dominates an unpaired comparison and is the
single biggest gain in accuracy available.

  A. STRUCTURE   how much mass sits in the halo now, per annulus per epoch:
                 median dlog(model/truth) with a bootstrap CI, plus the
                 per-galaxy SCATTER of that residual (does the model reproduce
                 the DIVERSITY of stellar halos, not just the median one?)
  B. EVOLUTION   how fast the halo grows: population-summed model/truth growth
                 ratio per epoch pair, bootstrapped. This is the "and their
                 evolution" half of the question.
  C. HEAD TO HEAD  per galaxy, |dlog| for the deposition-only model minus
                 |dlog| for the incumbent. Negative = deposition-only is
                 CLOSER for that galaxy. Reported as a median difference with
                 a CI and as the fraction of galaxies it wins, which together
                 answer "similarly well?" directly rather than by comparing
                 two separately-quoted numbers.

Annuli are chosen to stay inside the measured grid (2-148 kpc): the CoG's
outermost radius is 148 kpc, so anything beyond is extrapolation and is
excluded. `M(<500)` is NOT used as an outskirt statistic here — it is the
model's normalization, so the model matches it by construction.

Run: `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp53_deposition_only/outskirt.py [--cells a,b] [--nboot N] [--dev]`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import fit as F                                      # noqa: E402
import kernel as K                                   # noqa: E402
import stage2_multiepoch as s2                       # noqa: E402
from hongshao.profile_emulator import density_from_cog        # noqa: E402

KS = [0, 1, 2, 3, 4]
ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
#: (label, inner, outer) in kpc. All inside the measured grid; 148.2 is the
#: outermost CoG radius, so nothing here is extrapolated.
ANNULI = [("M[30,50]", 30.0, 50.0), ("M[50,100]", 50.0, 100.0),
          ("M[100,148]", 100.0, 148.0), ("M[50,148]", 50.0, 148.0)]
#: adjacent pairs as (earlier, later) plus the long baseline
PAIRS = [(1, 0), (2, 1), (3, 2), (4, 3), (4, 0)]
DEFAULT_CELLS = ("moffat-q0", "sersic-q0", "expo-q0", "gauss-q0")
BASELINE = "moffat-qfree"
NBOOT = 2000
SEED = 20260821


def annulus_mass(cogs, R, lo, hi):
    """(n_gal, n_epoch) enclosed mass between lo and hi, from a CoG."""
    n, ne, _ = cogs.shape
    flat = cogs.reshape(-1, cogs.shape[-1])
    out = np.array([np.interp(hi, R, c) - np.interp(lo, R, c) for c in flat])
    return out.reshape(n, ne)


def _boot_median(x, nboot=NBOOT, seed=SEED):
    """Median with a percentile bootstrap CI over GALAXIES."""
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), \
        float(np.percentile(meds, 97.5))


def _boot_ratio(num, den, nboot=NBOOT, seed=SEED):
    """Population-SUMMED ratio sum(num)/sum(den), bootstrapped over galaxies.

    Summed rather than per-galaxy-averaged because the growth of an individual
    galaxy's outskirt can be near zero or negative, which makes a per-galaxy
    ratio explode; the summed form is what exp52 used and is stable.
    """
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if len(num) < 10 or den.sum() == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(nboot, len(num)))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), \
        float(np.percentile(r, 97.5))


def _dlog(model, truth):
    """log10(model/truth), NaN where either is non-positive."""
    ok = np.isfinite(model) & np.isfinite(truth) & (model > 0) & (truth > 0)
    out = np.full(model.shape, np.nan)
    out[ok] = np.log10(model[ok] / truth[ok])
    return out


def main(cells=None, nboot=NBOOT, dev=False):
    pop = np.load(ROOT / "experiments/exp32_full_population/outputs/"
                  "population.npz")
    rows = pop["dev100"] if dev else None
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    data = np.stack([np.stack([g["data"][k] for k in KS]) for g in gals])
    done = F._load(F._store(dev=dev))
    lookup = dict(F.all_cells())
    names = list(cells or DEFAULT_CELLS)

    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    massive = logms >= edges[2]
    bins = [("all", np.ones(len(gals), bool)),
            ("T3 massive", massive)]
    print(f"exp53 — THE OUTSKIRT TEST  (n={len(gals)}, {massive.sum()} in the "
          f"massive tercile logM* >= {edges[2]:.2f})")
    print(f"  {nboot} galaxy bootstraps; CIs are 95%. All annuli lie inside "
          f"the measured 2-148 kpc grid.\n")

    cogs = {}
    for nm in [BASELINE] + names:
        b = F.best_cell(done, nm)
        if b is None:
            print(f"  [{nm}] not fitted — skipped")
            continue
        cogs[nm] = K.population_cogs(lookup[nm], b[0], gals, KS)
    if BASELINE not in cogs:
        raise SystemExit("baseline not fitted")

    # ---------- A. STRUCTURE ------------------------------------------- #
    print("=" * 100)
    print("A. STRUCTURE — median dlog10(model/truth) [dex] with 95% CI, and "
          "the per-galaxy 16-84 SCATTER")
    print("=" * 100)
    for lab, sel in bins:
        for aname, lo, hi in ANNULI:
            mt = annulus_mass(data, R, lo, hi)
            print(f"\n  {aname}  [{lab}]")
            print(f"    {'model':<16}" + "".join(
                f"{f'z={z}':>22}" for z in ANCHOR_Z))
            for nm in [BASELINE] + names:
                if nm not in cogs:
                    continue
                mm = annulus_mass(cogs[nm], R, lo, hi)
                d = _dlog(mm, mt)
                cells_txt = []
                for j in range(len(KS)):
                    v = d[sel, j]
                    med, lo_ci, hi_ci = _boot_median(v, nboot)
                    sc = np.nanpercentile(v, 84) - np.nanpercentile(v, 16)
                    cells_txt.append(f"{med:+.3f}[{lo_ci:+.3f},{hi_ci:+.3f}]"
                                     f" s{sc:.2f}")
                tag = "  (incumbent)" if nm == BASELINE else ""
                print(f"    {nm + tag:<16}" + "".join(
                    f"{c:>22}" for c in cells_txt))

    # ---------- B. EVOLUTION -------------------------------------------- #
    print("\n" + "=" * 100)
    print("B. EVOLUTION — population-summed model/truth GROWTH ratio with 95% "
          "CI (1.000 = the truth's rate)")
    print("=" * 100)
    for lab, sel in bins:
        for aname, lo, hi in ANNULI:
            if aname == "M[30,50]":
                continue
            mt = annulus_mass(data, R, lo, hi)
            print(f"\n  {aname}  [{lab}]")
            print(f"    {'model':<16}" + "".join(
                f"{f'z{ANCHOR_Z[a]}->{ANCHOR_Z[b]}':>24}" for a, b in PAIRS))
            for nm in [BASELINE] + names:
                if nm not in cogs:
                    continue
                mm = annulus_mass(cogs[nm], R, lo, hi)
                cells_txt = []
                for ka, kb in PAIRS:
                    r, lo_ci, hi_ci = _boot_ratio(
                        (mm[:, kb] - mm[:, ka])[sel],
                        (mt[:, kb] - mt[:, ka])[sel], nboot)
                    cells_txt.append(f"{r:+.3f} [{lo_ci:+.3f},{hi_ci:+.3f}]")
                tag = "  (incumbent)" if nm == BASELINE else ""
                print(f"    {nm + tag:<16}" + "".join(
                    f"{c:>24}" for c in cells_txt))

    # ---------- C. HEAD TO HEAD ----------------------------------------- #
    print("\n" + "=" * 100)
    print("C. HEAD TO HEAD, PAIRED per galaxy — median(|dlog| deposition-only "
          "- |dlog| incumbent) [dex]")
    print("   NEGATIVE = the deposition-only model is CLOSER to the truth. "
          "'wins' = fraction of galaxies it is closer for.")
    print("=" * 100)
    for lab, sel in bins:
        for aname, lo, hi in ANNULI:
            mt = annulus_mass(data, R, lo, hi)
            db = np.abs(_dlog(annulus_mass(cogs[BASELINE], R, lo, hi), mt))
            print(f"\n  {aname}  [{lab}]")
            print(f"    {'model':<16}" + "".join(
                f"{f'z={z}':>24}" for z in ANCHOR_Z))
            for nm in names:
                if nm not in cogs:
                    continue
                dm = np.abs(_dlog(annulus_mass(cogs[nm], R, lo, hi), mt))
                cells_txt = []
                for j in range(len(KS)):
                    diff = (dm[:, j] - db[:, j])[sel]
                    med, lo_ci, hi_ci = _boot_median(diff, nboot)
                    win = np.nanmean(diff < 0)
                    cells_txt.append(f"{med:+.3f}[{lo_ci:+.3f},{hi_ci:+.3f}]"
                                     f" w{100*win:.0f}%")
                print(f"    {nm:<16}" + "".join(f"{c:>24}" for c in cells_txt))

    # ---------- D. the outer SURFACE DENSITY -------------------------------- #
    print("\n" + "=" * 100)
    print("D. OUTER SURFACE DENSITY — median dlogSigma (model - truth) [dex] "
          "with 95% CI")
    print("=" * 100)
    lsd, mid = density_from_cog(
        np.log10(np.clip(data.reshape(-1, len(R)), 1.0, None)), R)
    lsd = lsd.reshape(len(gals), len(KS), -1)
    bands = [("30-60 kpc", (mid >= 30) & (mid < 60)),
             ("60-100 kpc", (mid >= 60) & (mid < 100)),
             ("100-148 kpc", (mid >= 100) & (mid <= 148))]
    for lab, sel in bins:
        print(f"\n  [{lab}]")
        print(f"    {'model':<16}{'band':<14}" + "".join(
            f"{f'z={z}':>20}" for z in ANCHOR_Z))
        for nm in [BASELINE] + names:
            if nm not in cogs:
                continue
            c = cogs[nm]
            lsm = np.full_like(lsd, np.nan)
            flat = c.reshape(-1, len(R))
            ok = np.isfinite(flat).all(1) & (flat > 0).all(1)
            tmp = np.full((len(flat), len(mid)), np.nan)
            tmp[ok] = density_from_cog(np.log10(flat[ok]), R)[0]
            lsm = tmp.reshape(len(gals), len(KS), -1)
            for bname, bm in bands:
                cells_txt = []
                for j in range(len(KS)):
                    v = np.nanmedian((lsm[:, j] - lsd[:, j])[:, bm], axis=1)[sel]
                    med, lo_ci, hi_ci = _boot_median(v, nboot)
                    cells_txt.append(f"{med:+.3f}[{lo_ci:+.3f},{hi_ci:+.3f}]")
                tag = "*" if nm == BASELINE else " "
                print(f"    {nm + tag:<16}{bname:<14}" + "".join(
                    f"{c_:>20}" for c_ in cells_txt))
    print("\n  * = incumbent (deposition + transport)")


if __name__ == "__main__":
    cl = None
    nb = NBOOT
    if "--cells" in sys.argv:
        cl = sys.argv[sys.argv.index("--cells") + 1].split(",")
    if "--nboot" in sys.argv:
        nb = int(sys.argv[sys.argv.index("--nboot") + 1])
    main(cl, nb, "--dev" in sys.argv)
