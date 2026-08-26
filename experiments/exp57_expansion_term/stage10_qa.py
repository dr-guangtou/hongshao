"""exp57 — the project's STANDARD QA battery on the expansion models.

`stage5_figures.py` and `stage9_figures.py` answer exp57's own questions. This
runs `hongshao.qa.evaluate`, the battery **every promoted model in this
programme has been shown on**, so exp57's candidate can be compared with exp53's
and exp54's on identical footing: tier 1+2 mass tables per bin set, tier 2b
observational planes, tier 2c cross-epoch growth planes, tier 2d mass-size
planes, tier 2e mass distributions, tier 3 profile visual QA, and the two binned
profile views.

**Both binnings are produced, and they can disagree in sign.** `qa_bins_*` bins
by HALO mass — a model INPUT, so it is free of the regression-to-the-mean
artefact — and `qa_bins_ms_*` bins by total STELLAR mass, which is the view an
observed sample gives but which converts unexplained amplitude scatter into
apparent bin-wise bias. exp53 measured the disagreement and it is
radius-dependent. Read the halo-mass view for the model's purpose.

WHICH MODELS. Three, so that "best" is not left to my judgement alone:

  * `incumbent`        exp54's adopted model, expansion off — the REFERENCE.
    Nothing below means anything without it.
  * `X3-smallcore`     the best model that PASSES the mechanism gates, and
    exp57's actual candidate: core-only expansion, `Rc` = 2.6 kpc.
  * `X4`               the best RAW loss and shape error of anything tried, and
    REJECTED — its conditioning slope is unidentified (three signs across ten
    starts) and it fails the reach gate. Included because "best performed" can
    reasonably mean this one, and a reader should be able to see what its
    numbers look like on the standard battery rather than take my word for it.

A galaxy the model cannot represent is dropped from ALL THREE consistently, so
the three reports describe the same objects.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage10_qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao import qa                                  # noqa: E402

QADIR = HERE / "figures" / "qa"
FITDIR = HERE / "outputs" / "stage3_fits"
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: (label in the report, fit file or None for the incumbent)
MODELS = (("incumbent", None),
          ("X3-smallcore", "gompertz_log-E2-S2+X3#smallcore"),
          ("X4", "gompertz_log-E2-S2+X4"))
RULE = "=" * 96


def main(tables_only=False):
    print(f"{RULE}\nexp57 — the project's STANDARD QA battery "
          f"(`hongshao.qa.evaluate`)\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    R = F.R_GRID

    cogs = {}
    for name, lab in MODELS:
        if lab is None:
            sp = X.XSpec(base, "")
            th = theta
        else:
            z = np.load(FITDIR / f"{lab}.npz", allow_pickle=True)
            sp = X.XSpec(base, str(z["law"]))
            th = np.asarray(z["theta"], float)
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        cogs[name] = pr.predict(th)
        print(f"  {name:<14} {sp.label:<28} "
              + (", ".join(f"{n}={v:+.4f}" for n, v in
                           zip(sp.x_names, th[base.n_theta:])) or "(no expansion)"))

    # ONE admitted set for all three, so the reports describe the same objects.
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    print(f"\n  {int(good.sum())} of {len(recs)} galaxies are representable by "
          f"all three models AND have a\n  finite positive measured profile at "
          f"every epoch. The same set is used for each\n  report, so the three "
          f"are directly comparable.")

    # total stellar mass for the second binning: the truth's outermost point
    # at the first epoch, which is what an observed sample would select on
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))
    QADIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, _ in MODELS:
        print(f"\n{RULE}\nSTANDARD QA — {name}\n{RULE}")
        out[name] = qa.evaluate(
            cogs[name][good], data[good], R, ANCHOR_Z,
            name=f"exp57_{name}", figdir=(None if tables_only else QADIR),
            figures=not tables_only, verbose=True,
            bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
            bin_by_ms=logms, ms_label=r"logM$_*$ (total)")

    print(f"\n{RULE}\nSIDE BY SIDE, on the same {int(good.sum())} galaxies"
          f"\n{RULE}")
    print(f"  THE METRIC THAT MOVES: median relative bias in the INNER "
          f"apertures. This is where\n  exp57's change lives, and it is the "
          f"only place in the standard battery that sees it.")
    for key in ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(<100)"):
        print(f"\n  {key} — median relative bias, closer to 0 is better")
        print(f"  {'model':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for name, _ in MODELS:
            t, m = out[name]["truth"][key], out[name]["model"][key]
            print(f"  {name:<16}"
                  + "".join(
                      f"{100 * np.nanmedian(qa.relerr(m[:, j], t[:, j])):>9.1f}%"
                      for j in range(5)))

    print(f"\n  THE METRICS THAT DO NOT MOVE, and it is worth knowing which "
          f"they are.")
    print(f"  tier 3, profile max|rel| beyond 5 kpc (median) — LOWER is better")
    print(f"  {'model':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for name, _ in MODELS:
        print(f"  {name:<16}"
              + "".join(f"{np.nanmedian(out[name]['mr_out'][:, j]):>10.4f}"
                        for j in range(5)))
    k = "kpc:M(<100)"
    print(f"\n  M*(<100 kpc) scatter [dex] — LOWER is better")
    print(f"  {'model':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for name, _ in MODELS:
        print(f"  {name:<16}"
              + "".join(f"{qa.dex_scatter(out[name]['model'][k][:, j], out[name]['truth'][k][:, j]):>10.4f}"
                        for j in range(5)))
    print(f"\n  A max over radii is dominated by its worst radius and a "
          f"scatter is an amplitude\n  statistic, so neither can see a "
          f"redistribution of a few per cent of the mass\n  inside 10 kpc. "
          f"That is not a defect of the battery; it is why exp57 has gates of\n"
          f"  its own. But it does mean the standard battery alone would have "
          f"reported this\n  experiment as a null.")

    print(f"\n  figures -> {QADIR.relative_to(ROOT)}")
    np.savez(HERE / "outputs" / "stage10_qa.npz",
             names=np.array([m[0] for m in MODELS]), n_galaxies=int(good.sum()),
             **{f"mr_out_{m[0]}": out[m[0]]["mr_out"] for m in MODELS})


if __name__ == "__main__":
    main("--tables-only" in sys.argv)
