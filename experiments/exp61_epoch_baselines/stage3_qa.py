"""exp61 Stage 3 — the project's STANDARD QA battery on the exp61 models.

The user's request, verbatim: the standard QA figures — average curves of
growth in halo- and stellar-mass bins, the stellar-mass CDFs, the mass
planes, the mass-size relations — because the conclusion has to be judged on
what the profiles look like at each epoch, not only on the fitting metrics.

`hongshao.qa.evaluate` is that battery, the one every promoted model in this
programme has been shown on. Three models, the same galaxies, the same
figures:

  * `incumbent`      the shared seven-parameter law, all five epochs at once.
                     The REFERENCE, and what hongshao v1 actually is.
  * `per-epoch-best` THE CLASS CEILING. At each epoch, the prediction of the
                     fit made at THAT epoch alone (exp61 Stage 0). This is
                     what the baseline table's "single-epoch" row is, as
                     profiles.
  * `zle1.0-refit`   the z <= 1.0 retreat, shown at all five epochs (its own
                     three in scope, z = 1.5 and 2.0 out of it).

TWO THINGS THE READER MUST KNOW BEFORE READING THESE FIGURES.

1. **`per-epoch-best` is NOT a model anyone could adopt.** It is five
   different parameter sets stitched together by epoch — galaxy i's z=0.4
   profile comes from one theta and its z=2.0 profile from another. It is an
   upper bound on what this model class can do epoch by epoch, and nothing
   more. In particular tier 2c, the cross-epoch growth planes, asks whether
   the SAME galaxy evolves coherently between two epochs; for a stitched
   composite that question has no owner, and its tier-2c numbers must not be
   read as any model's coherence. Every other tier is a per-epoch statistic
   and is exactly what it says.

2. **The standing battery caveat (exp57 P-C)**: the headline tiers cannot see
   a redistribution of a few per cent of the mass inside 10 kpc. Changes of
   that size show in the inner-aperture bias table and in the binned residual
   panels, not in tier 3 or in the scatters.

SAMPLE. The exp54 clean sample, every galaxy with finite positive measured
profiles at all five epochs that all three models can represent — the same
construction exp59 Stage 5 used, so these numbers sit beside the ones already
on record. Note it is the FIT sample; exp58 P-C measured that the full QA
population carries different biases at z=2.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp61_epoch_baselines/stage3_qa.py [--tables-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term"):
    sys.path.insert(0, str(p))

import matplotlib.pyplot as plt                          # noqa: E402

import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402
from hongshao import qa                                  # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
QADIR = HERE / "figures" / "qa"
FIGDIR = HERE / "figures"
STAGE0 = HERE / "outputs" / "stage0_fits.npz"
OUT = HERE / "outputs" / "stage3_qa.npz"
#: display order; the label is what appears in every figure and table
MODELS = ("incumbent", "per-epoch-best", "zle1.0-refit")
PRETTY = {"incumbent": "the shared law (hongshao v1's mean)",
          "per-epoch-best": "per-epoch best (the class ceiling, stitched)",
          "zle1.0-refit": r"the $z \leq 1.0$ refit"}
COLORS = {"incumbent": "#000000", "per-epoch-best": "#D55E00",
          "zle1.0-refit": "#56B4E9"}
#: the apertures the side-by-side bias table reports
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(<50)", "kpc:M(<100)")
RULE = "=" * 96


def build_cogs(recs, data, mask, base, theta):
    """The three products, on the full (n, 5, nR) grid."""
    s0 = np.load(STAGE0, allow_pickle=True)
    pr = S35.StackedProblem(base, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    cogs = {"incumbent": pr.predict(theta),
            "zle1.0-refit": pr.predict(np.asarray(s0["fit_zle1.0_theta"],
                                                  float))}
    stitched = np.full_like(cogs["incumbent"], np.nan)
    for j, z in enumerate(ANCHOR_Z):
        th = np.asarray(s0[f"fit_z={z}_theta"], float)
        stitched[:, j, :] = pr.predict(th)[:, j, :]
    cogs["per-epoch-best"] = stitched
    return cogs


def compare_figure(cogs, data, good, msk, name="exp61_qa_compare"):
    """The one bespoke panel the standard battery does not draw: all three
    models' median CoG error against radius, epoch by epoch, on one pair of
    axes per epoch. The battery gives each model its own binned figure; this
    is where they are laid over each other.

    Two samples per model, because they disagree and the disagreement is a
    result (exp58 P-C): the whole clean population (solid), and the
    mh-complete fit mask every loss and every B gate is defined on (thin
    dashed, for the two models where it matters).
    """
    R = np.asarray(F.R_GRID, float)

    def med(m, j, sel=None):
        cur, tru = cogs[m][good][:, j, :], data[good][:, j, :]
        if sel is not None:
            cur, tru = cur[sel], tru[sel]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmedian(100.0 * (cur / tru - 1.0), axis=0)

    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.8), sharex=True,
                             sharey=True)
    ax = axes.ravel()
    for j, z in enumerate(ANCHOR_Z):
        g = ax[j]
        g.axhline(0.0, color="0.55", lw=1.0, ls=":", zorder=1)
        g.axhspan(-5.0, 5.0, color="0.85", alpha=0.5, zorder=0)
        for m in MODELS:
            g.plot(R, med(m, j), "-", lw=1.8, color=COLORS[m], zorder=3,
                   label=PRETTY[m] if j == 0 else None)
        for m in ("incumbent", "per-epoch-best"):
            g.plot(R, med(m, j, msk[:, j]), "--", lw=1.1, color=COLORS[m],
                   zorder=4, label=(f"{m}, on the mh-complete fit mask"
                                    if j == 0 else None))
        g.set_xscale("log")
        g.set_title(f"z = {z:.1f}", fontsize=10)
        if j >= 2:
            g.set_xlabel("radius [kpc]")
        if j % 3 == 0:
            g.set_ylabel("median (model $-$ measured) / measured "
                         f"[{qa._pct()}]")

    t = ax[5]
    t.axis("off")
    h, lab = ax[0].get_legend_handles_labels()
    t.legend(h, lab, loc="upper center", fontsize=8.5, frameon=False,
             bbox_to_anchor=(0.5, 1.02))
    z2 = 4
    ceil_all, ceil_fit = med("per-epoch-best", z2)[-1], \
        med("per-epoch-best", z2, msk[:, z2])[-1]
    inc_all, inc_fit = med("incumbent", z2)[-1], \
        med("incumbent", z2, msk[:, z2])[-1]
    t.text(0.0, 0.58,
           "Each curve is the median over galaxies of the fractional error "
           "in\nthe enclosed stellar mass, at that radius and that epoch. The "
           "grey\nband is $\\pm 5$ per cent.\n\n"
           "At z $\\leq$ 1.0 all three models agree closely and sit inside "
           "the band\nbeyond about 8 kpc, dipping 7 to 13 per cent below it "
           "in the inner\nfew kpc. That inner defect belongs to every epoch "
           "scope alike, so it\nis not what the exp61 retreat is about.\n\n"
           "At z = 1.5 and 2.0 the shared law is low at every radius. The\n"
           "per-epoch ceiling does NOT simply repair that: on the whole\n"
           f"population it crosses to an EXCESS, {ceil_all:+.0f} per cent at "
           "148 kpc at z=2,\nwhile its dashed twin — the same model on the "
           f"mh-complete mask it\nwas fitted to — is {ceil_fit:+.0f} per "
           "cent there. Both models' two\nsamples separate at z=2, so the "
           "sample matters for everyone, but the\nceiling's separate "
           f"{abs(ceil_all - ceil_fit):.0f} points against the shared law's "
           f"{abs(inc_all - inc_fit):.0f}. That is\nthe exp58 P-C sample "
           "split seen as profiles, and it is why the\nper-epoch gain must "
           "not be read as a population-level fix.\n\n"
           "per-epoch best is FIVE parameter sets stitched by epoch, not an\n"
           "adoptable model — an upper bound on this model class, epoch by\n"
           "epoch, and nothing more. " + PRETTY['zle1.0-refit'].capitalize()
           + " is in scope only at\nz = 0.4, 0.7 and 1.0; its z = 1.5 and "
           "2.0 curves show the cost of\nthe retreat outside its scope.",
           fontsize=7.8, color="0.20", va="top", ha="left",
           transform=t.transAxes)

    fig.suptitle("exp61 — the average curve of growth, epoch by epoch: how "
                 "much stellar mass the model puts inside each radius,\n"
                 "against how much the simulation has there", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_fig(fig, FIGDIR / name)


def main(tables_only=False, compare_only=False):
    print(f"{RULE}\nexp61 STAGE 3 — the STANDARD QA battery on the exp61 "
          f"models\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    R = F.R_GRID
    cogs = build_cogs(recs, data, mask, base, theta)

    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    print(f"  {int(good.sum())} of {len(recs)} galaxies have finite positive "
          f"measured profiles at every\n  epoch and are representable by all "
          f"three models; the same set is used for each report.")
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))
    msk = np.asarray(mask, bool)[good]        # the fit mask, on the QA sample
    print("  of those, the mh-complete fit mask keeps "
          + "/".join(f"{int(msk[:, j].sum())}" for j in range(5))
          + " at z=0.4/0.7/1.0/1.5/2.0 —\n  the sample every fitted loss and "
          "every B gate is defined on.")

    if compare_only:
        for p in compare_figure(cogs, data, good, msk):
            print(f"  wrote {p.relative_to(ROOT)}")
        return

    QADIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for m in MODELS:
        print(f"\n{RULE}\nSTANDARD QA — {m}   ({PRETTY[m]})\n{RULE}")
        if m == "per-epoch-best":
            print("  REMINDER: five parameter sets stitched by epoch. Every "
                  "per-epoch tier is a\n  genuine ceiling; tier 2c "
                  "(cross-epoch growth) has no owner here — do not read "
                  "it\n  as any model's coherence.")
        out[m] = qa.evaluate(
            cogs[m][good], data[good], R, list(ANCHOR_Z),
            name=f"exp61_{m}", figdir=(None if tables_only else QADIR),
            figures=not tables_only, verbose=True,
            bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
            bin_by_ms=logms, ms_label=r"logM$_*$ (total)")

    print(f"\n{RULE}\nSIDE BY SIDE, on the same {int(good.sum())} galaxies"
          f"\n{RULE}")
    for key in TABLE_KEYS:
        print(f"\n  {key} — median relative bias [%], closer to 0 is better")
        print(f"  {'model':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for m in MODELS:
            t, mm = out[m]["truth"][key], out[m]["model"][key]
            print(f"  {m:<18}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%"
                for j in range(5)))

    print(f"\n{RULE}\n  THE SAME NUMBER ON THE TWO SAMPLES — why the gate "
          f"tables and this battery disagree\n{RULE}")
    print("  B1/B3 and every fitted loss are defined on the mh-complete + "
          "sane FIT MASK; this\n  battery reports the whole clean "
          "population, which is what the QA figures show and\n  what the "
          "user reads. exp58 P-C measured that the two carry different "
          "biases at\n  z=2; here is that difference, for the same "
          "quantity and the same three models.")
    for key in ("kpc:M(<10)", "kpc:M(<100)"):
        print(f"\n  {key} — median relative bias [%]")
        print(f"  {'model':<18}{'sample':<16}"
              + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for m in MODELS:
            t, mm = out[m]["truth"][key], out[m]["model"][key]
            print(f"  {m:<18}{'all ' + str(int(good.sum())):<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%"
                for j in range(5)))
            print(f"  {'':<18}{'fit mask only':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[msk[:, j], j], t[msk[:, j], j])):>9.1f}%"
                for j in range(5)))

    print("\n  M*(<100 kpc) scatter [dex] — LOWER is better")
    print(f"  {'model':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    k = "kpc:M(<100)"
    for m in MODELS:
        print(f"  {m:<18}" + "".join(
            f"{qa.dex_scatter(out[m]['model'][k][:, j], out[m]['truth'][k][:, j]):>10.4f}"
            for j in range(5)))

    print("\n  tier 3, profile max|rel| beyond 5 kpc (median) — LOWER is "
          "better")
    print(f"  {'model':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for m in MODELS:
        print(f"  {m:<18}" + "".join(
            f"{np.nanmedian(out[m]['mr_out'][:, j]):>10.4f}" for j in range(5)))

    if not tables_only:
        for p in compare_figure(cogs, data, good, msk):
            print(f"\n  wrote {p.relative_to(ROOT)}")
        print(f"  battery figures -> {QADIR.relative_to(ROOT)}")

    np.savez(OUT, names=np.array(MODELS), n_galaxies=int(good.sum()),
             **{f"mr_out_{m}": out[m]["mr_out"] for m in MODELS},
             **{f"bias_{m}_{k}": np.array(
                 [np.nanmedian(qa.relerr(out[m]["model"][k][:, j],
                                         out[m]["truth"][k][:, j]))
                  for j in range(5)])
                for m in MODELS for k in TABLE_KEYS})
    print(f"  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main("--tables-only" in sys.argv[1:],
         "--compare-only" in sys.argv[1:])
