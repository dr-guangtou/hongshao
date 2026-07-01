"""exp29 — corrected multi-epoch evaluation: all radii + integrated mass checks.

Two methodology fixes (user, 2026-07):
  1. NO inner-3-kpc mask. For high-z massive progenitors Re < 3 kpc, so the inner
     region holds >50% of the stellar mass; masking it hid the hardest, most
     important part. Fit AND evaluate over all 24 radii.
  2. Beyond the point-wise profile residual, add INTEGRATED checks: aperture masses
     M*(<R) at fixed physical radii, and the outskirt mass M*(>R_out). The outskirt
     (differential) mass amplifies shape errors the cumulative can hide.

Uses the real de-dipped MAH (peak_history), the honest input. Models: independent
per-epoch ceiling and the loose-zdep quad (current best). CoGs cached to
outputs/integrated_check.npz.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/integrated_check.py [n] [--refit]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run import ANCHOR_Z                                                             # noqa: E402
from real_mah_test import real_mah, fit_independent, fit_loose                       # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                         # noqa: E402

set_style()
FIGDIR = HERE / "figures"
NPZ_IN = HERE / "outputs" / "model_compare.npz"
NPZ_OUT = HERE / "outputs" / "integrated_check.npz"
R = COG_RAD_KPC
ALLR = R > 0.0                                       # no inner mask
APERTURES = [10.0, 30.0, 50.0, 100.0]               # fixed physical apertures (kpc)
R_OUT = 50.0                                        # outskirt = M*(>50 kpc)


def apermass(cog, rap):
    return np.interp(rap, R, cog)                    # CoG is monotonic in R


def compute(n):
    d = np.load(NPZ_IN)
    idx, logms, datas = d["index"][:n], d["logms"][:n], d["data"][:n]
    gids, lms, dd, mind, mloose = [], [], [], [], []
    for i in range(len(idx)):
        mah = real_mah(int(idx[i]))
        if mah is None:
            continue
        data = [datas[i][k] for k in range(5)]
        ci, Pg = fit_independent(mah, data, ALLR)    # fit over ALL radii
        cl = fit_loose(mah, data, ALLR, Pg)
        gids.append(int(idx[i])); lms.append(logms[i]); dd.append(datas[i])
        mind.append(ci); mloose.append(cl)
    out = dict(index=np.array(gids), logms=np.array(lms), R=R,
               data=np.array(dd), independent=np.array(mind), loose=np.array(mloose))
    np.savez(NPZ_OUT, **out)
    print(f"wrote {NPZ_OUT}  (n={len(gids)})")
    return np.load(NPZ_OUT)


def metrics(model, data):
    """Per-epoch: profile max|rel| (all R), aperture dlog at APERTURES, outskirt dlog."""
    prof = np.array([np.abs((model[k] - data[k]) / data[k]).max() for k in range(5)])
    ap = {a: np.array([np.log10(apermass(model[k], a)) - np.log10(apermass(data[k], a))
                       for k in range(5)]) for a in APERTURES}
    out_m = np.array([np.log10(max(model[k][-1] - apermass(model[k], R_OUT), 1.0)) for k in range(5)])
    out_d = np.array([np.log10(max(data[k][-1] - apermass(data[k], R_OUT), 1.0)) for k in range(5)])
    return prof, ap, out_m - out_d


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 60)
    d = compute(n) if (refit or not NPZ_OUT.exists()) else np.load(NPZ_OUT)
    logms, data = d["logms"], d["data"]
    models = {"independent": d["independent"], "loose-quad": d["loose"]}
    ng = len(logms)

    # collect per-galaxy metrics
    M = {nm: dict(prof=[], ap={a: [] for a in APERTURES}, out=[]) for nm in models}
    for nm, cogs in models.items():
        for i in range(ng):
            p, ap, o = metrics(cogs[i], data[i])
            M[nm]["prof"].append(p); M[nm]["out"].append(o)
            for a in APERTURES:
                M[nm]["ap"][a].append(ap[a])
        M[nm]["prof"] = np.array(M[nm]["prof"]); M[nm]["out"] = np.array(M[nm]["out"])
        for a in APERTURES:
            M[nm]["ap"][a] = np.array(M[nm]["ap"][a])

    print(f"\nexp29 — corrected evaluation (real MAH, ALL radii, n={ng})\n")
    print("  (1) profile max|rel|, median per epoch:")
    print(f"    {'model':>12s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z))
    for nm in models:
        print(f"    {nm:>12s} | " + " | ".join(f"{100*np.median(M[nm]['prof'][:,k]):5.1f}%" for k in range(5)))

    print("\n  (2) integrated aperture-mass bias  |dlog M*(<R)|  (loose-quad, median |.| per epoch):")
    print(f"    {'aperture':>12s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z))
    for a in APERTURES:
        v = M["loose-quad"]["ap"][a]
        print(f"    {'<'+str(int(a))+' kpc':>12s} | " +
              " | ".join(f"{np.median(np.abs(v[:,k])):5.3f}" for k in range(5)))
    vo = M["loose-quad"]["out"]
    print(f"    {'>50 kpc':>12s} | " + " | ".join(f"{np.median(np.abs(vo[:,k])):5.3f}" for k in range(5)))

    print("\n  (3) same, SIGNED median dlog (model-data; + = model over-predicts):")
    print(f"    {'aperture':>12s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z))
    for a in APERTURES:
        v = M["loose-quad"]["ap"][a]
        print(f"    {'<'+str(int(a))+' kpc':>12s} | " + " | ".join(f"{np.median(v[:,k]):+5.3f}" for k in range(5)))
    print(f"    {'>50 kpc':>12s} | " + " | ".join(f"{np.median(vo[:,k]):+5.3f}" for k in range(5)))

    print(f"\n[note] all radii included (no inner cut). Profile max|rel| epoch-avg: " +
          "  ".join(f"{nm} {100*np.median([np.median(M[nm]['prof'][:,k]) for k in range(5)]):.1f}%"
                    for nm in models))
    _figure(logms, M)


def _figure(logms, M):
    x = np.arange(5)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))

    a = axes[0]                                      # profile metric, all radii
    for nm, c in (("independent", "0.45"), ("loose-quad", OKABE_ITO[1])):
        a.plot(x, [100 * np.median(M[nm]["prof"][:, k]) for k in range(5)], "o-", c=c, lw=2, label=nm)
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    a.set(xlabel="epoch z", ylabel="median profile max|rel| [%]", ylim=(0, None),
          title="A. Profile residual (all radii)")
    a.legend(fontsize=8)

    b = axes[1]                                      # aperture-mass bias, loose
    cmap = matplotlib.colormaps["viridis"]
    for j, ap in enumerate(APERTURES):
        v = M["loose-quad"]["ap"][ap]
        b.plot(x, [np.median(v[:, k]) for k in range(5)], "o-", c=cmap(j / 3), lw=1.8,
               label=f"M*(<{int(ap)} kpc)")
    vo = M["loose-quad"]["out"]
    b.plot(x, [np.median(vo[:, k]) for k in range(5)], "s--", c=OKABE_ITO[5], lw=2, label="M*(>50 kpc)")
    b.axhline(0, c="0.6", lw=0.8); b.set_xticks(x); b.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    b.set(xlabel="epoch z", ylabel=r"median $\Delta\log M_*$ (model$-$data)",
          title="B. Integrated mass bias (loose-quad)")
    b.legend(fontsize=7)

    c = axes[2]                                      # outskirt bias by mass bin
    order = np.argsort(logms); bins = np.array_split(order, 3); labs = ["low", "mid", "high"]
    for (bi, lab), col in zip(enumerate(labs), (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[5])):
        vo = M["loose-quad"]["out"][bins[bi]]
        c.plot(x, [np.median(vo[:, k]) for k in range(5)], "o-", c=col, lw=2, label=f"{lab}-mass")
    c.axhline(0, c="0.6", lw=0.8); c.set_xticks(x); c.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    c.set(xlabel="epoch z", ylabel=r"median $\Delta\log M_*(>50\,{\rm kpc})$",
          title="C. Outskirt-mass bias vs mass")
    c.legend(fontsize=8)
    fig.suptitle("exp29 — corrected evaluation: real MAH, all radii, + integrated aperture/outskirt "
                 "mass checks", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "exp29_integrated_check")[0])


if __name__ == "__main__":
    sys.exit(main())
