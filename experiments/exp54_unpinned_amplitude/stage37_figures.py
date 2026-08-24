"""exp54 Stage 3.7 — QA figures for the size law's dependence on epoch.

`stage37_precheck` — the measurement that licensed the stage.

  (a) The model's error in the stellar mass inside 2 kpc, epoch by epoch, in
      fixed bins of epoch-matched halo mass. If these curves flattened relative
      to the all-galaxies curve the defect would have been the sample's mass
      composition moving, and the stage would have been aimed at nothing. They
      steepen instead.
  (b) The same numbers read down the columns: the error's dependence on halo
      mass at fixed epoch, which steepens with redshift. That interaction is
      what distinguishes the candidates from each other.

`stage37_size_epoch` — what the enrichment bought.

  (a) The central error against epoch for the incumbent, each candidate and the
      free-curve bound. The vertical extent of each curve IS the quantity under
      test; the aggregate profile-shape error is not.
  (b) The fitted size law itself — the deposit's half-mass radius in units of
      its halo's radius, against redshift — over the shaded distribution of
      where the deposited mass actually is, so a curve drawn where there is no
      mass is recognisable as decoration.
  (c) and (d) The diagnostics that were required not to get worse: the error
      beyond 52 kpc and the outer surface-density slope.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage37_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import model as M                                        # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage37_size_epoch as S37                         # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tex, _zcolors             # noqa: E402

FIGDIR = HERE / "figures"
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
_VC = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#7570B3"]


def precheck_figure(name="stage37_precheck"):
    import matplotlib.pyplot as plt
    z = np.load(S37.PRECHECK)
    err, lmh, mask = z["err"], z["lmh"], z["mask"]
    bins = z["mass_bins"]
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))

    ax = axes[0]
    med = np.array([np.nanmedian(err[:, k]) for k in range(5)])
    ax.plot(Z, med, "o-", color="k", lw=2.6, ms=6,
            label=f"every admitted galaxy (span {med[0] - med[4]:.3f} dex)")
    cols = _zcolors(len(bins) - 1)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        row = []
        for k in range(5):
            g = np.isfinite(err[:, k]) & (lmh[:, k] >= lo) & (lmh[:, k] < hi)
            row.append(np.median(err[g, k]) if g.sum() >= 25 else np.nan)
        row = np.array(row)
        # the heaviest bin empties out at z = 2, so quote the span over the
        # epochs it actually has rather than printing a nan
        ok = np.where(np.isfinite(row))[0]
        span = row[ok[0]] - row[ok[-1]] if len(ok) > 1 else np.nan
        note = "" if ok[-1] == 4 else f", to z={Z[ok[-1]]}"
        ax.plot(Z, row, "s--", color=cols[i], lw=1.5, ms=4,
                label=_tex(f"log10 Mh {lo:.1f}-{hi:.1f}"
                           f"  (span {span:.3f}{note})"))
    coh = mask.all(axis=1)
    ax.plot(Z, [np.nanmedian(err[coh, k]) for k in range(5)], "^-",
            color="#B22222", lw=2.0, ms=6,
            label=f"the same {int(coh.sum())} galaxies at all five epochs")
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("observation redshift")
    ax.set_ylabel(_tex("median log10(model / simulation) for M*(<2 kpc)"))
    ax.set_title("(a) the error steepens with redshift AT FIXED HALO MASS,\n"
                 "so it is the galaxies and not the sample", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    grad = z["mass_grad"] if "mass_grad" in z.files else None
    for k in range(5):
        row, ctr = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            g = np.isfinite(err[:, k]) & (lmh[:, k] >= lo) & (lmh[:, k] < hi)
            row.append(np.median(err[g, k]) if g.sum() >= 25 else np.nan)
            ctr.append(0.5 * (lo + min(hi, 14.2)))
        ax.plot(ctr, row, "o-", color=_zcolors(5)[k], lw=1.8, ms=5,
                label=f"z = {Z[k]}")
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel(_tex("log10 M200c at that epoch"))
    ax.set_ylabel(_tex("median log10(model / simulation) for M*(<2 kpc)"))
    ax.set_title("(b) and it depends on halo mass too — the gradient\n"
                 "steepens with redshift, which is an interaction", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, FIGDIR / name)
    print(f"  wrote {FIGDIR / name}.png")


def _size_curve(label, theta, zz, logmh=13.5):
    """R50/R200c against redshift for one fitted size law."""
    sp = S33.spec_from_label(label)
    size = sp.unpack(theta)[1]
    n = len(zz)
    h = type("H", (), dict(
        logmh=np.full(n, logmh), z=np.asarray(zz), dlogc=np.zeros(n),
        r200c=np.ones(n), c_eff=np.full(n, 5.0), f_form=np.full(n, 0.5)))
    return M.r50_of(sp, size, h)


def main_figure(name="stage37_size_epoch"):
    import matplotlib.pyplot as plt
    if not S37.OUT.exists():
        print("  no stage37 output yet — skipping the main figure")
        return
    z = np.load(S37.OUT, allow_pickle=True)
    labels = [str(v) for v in z["labels"]]
    base = labels[0]
    # the span-minimising solution, if it exists: the SAME free size law as
    # `S6f6`, fitted to reduce the central span rather than the loss. It is the
    # only curve here that moves the span, and the only one that shows what
    # moving it costs.
    sb = HERE / "outputs" / f"stage37_spanbound_{base}+S6f6.npz"
    extra = np.load(sb) if sb.exists() else None
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
    R = F.R_GRID

    ax = axes[0, 0]
    for i, lab in enumerate(labels):
        c = "k" if lab == base else _VC[i % len(_VC)]
        v = z[f"central::{lab}"]
        tag = lab.replace(base, "incumbent").replace(f"{base}+", "")
        ax.plot(Z, v, "o-" if lab == base else "s--", color=c,
                lw=2.6 if lab == base else 1.5, ms=5,
                label=f"{tag}  (span {v[0] - v[4]:.3f} dex)")
    if extra is not None:
        v = extra["central"]
        ax.plot(Z, v, "D-", color="#B22222", lw=2.2, ms=6,
                label=f"the same free law fitted FOR THE SPAN "
                      f"(span {v[0] - v[4]:.3f} dex)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("observation redshift")
    ax.set_ylabel(_tex("median log10(model / simulation) for M*(<2 kpc)"))
    ax.set_title("(a) THE QUANTITY UNDER TEST — the vertical extent of each\n"
                 "curve, not the aggregate profile score", fontsize=10)
    ax.legend(fontsize=7.5)

    ax = axes[0, 1]
    zz = np.geomspace(0.4, 15.0, 400)
    zc = np.load(S35.OUT)
    dz, ws = zc["dep_z"], zc["dep_w_star"]
    tw = ax.twinx()
    o = np.argsort(dz)
    dens = ws[o] / np.gradient(np.log1p(dz[o])) / ws.sum()
    tw.fill_between(1 + dz[o], dens, color="#CCCCCC", alpha=0.55, lw=0, zorder=0)
    tw.set_ylim(0, dens.max() * 3.2)
    tw.set_yticks([])
    tw.set_ylabel("where the stellar mass is deposited", fontsize=7.5,
                  color="#777777")
    for i, lab in enumerate(labels):
        c = "k" if lab == base else _VC[i % len(_VC)]
        ax.plot(1 + zz, _size_curve(lab, z[f"theta::{lab}"], zz), color=c,
                lw=2.6 if lab == base else 1.5, zorder=3,
                ls="-" if lab == base else "--",
                label=lab.replace(base, "incumbent").replace(f"{base}+", ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    # the free curve dives four orders of magnitude around z ~ 9, where the
    # grey histogram shows essentially no mass is deposited. Letting it set the
    # range would hide every curve that matters, so the frame is fixed and the
    # excursion is labelled instead of accommodated.
    ax.set_ylim(4e-3, 0.25)
    ax.annotate("the free curve leaves the frame here,\nin a redshift range "
                "carrying almost no mass", xy=(9.5, 5.5e-3), fontsize=7,
                color="#B22222", ha="center")
    ax.set_xlim(1.4, 16)
    ax.set_xticks([1.4, 2, 3, 5, 8, 16])
    ax.set_xticklabels(["0.4", "1", "2", "4", "7", "15"])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("redshift at which the mass was accreted")
    ax.set_ylabel(_tex("deposit half-mass radius / R200c, at log10 Mh = 13.5"))
    ax.set_title("(b) the fitted size law itself", fontsize=10)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    for i, lab in enumerate(labels):
        c = "k" if lab == base else _VC[i % len(_VC)]
        s = z[f"shells::{lab}"]
        ax.plot(Z, [np.nanmedian(s[k, R > 52]) for k in range(5)],
                "o-" if lab == base else "s--", color=c,
                lw=2.6 if lab == base else 1.4, ms=4,
                label=lab.replace(base, "incumbent").replace(f"{base}+", ""))
    if extra is not None and "shells" in extra.files:
        sh = extra["shells"]
        ax.plot(Z, [np.nanmedian(sh[k, R > 52]) for k in range(5)], "D-",
                color="#B22222", lw=2.2, ms=6,
                label="the free law fitted FOR THE SPAN")
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("observation redshift")
    ax.set_ylabel(_tex("median log10(model / simulation) beyond 52 kpc"))
    ax.set_title("(c) AND WHAT IT COSTS — buying the centre at redshift 2\n"
                 "empties the outskirts there", fontsize=10)
    ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    for i, lab in enumerate(labels):
        c = "k" if lab == base else _VC[i % len(_VC)]
        s = z[f"slope::{lab}"]
        ax.plot(Z, [s[k, 0] - s[k, 1] for k in range(5)],
                "o-" if lab == base else "s--", color=c,
                lw=2.6 if lab == base else 1.4, ms=4,
                label=lab.replace(base, "incumbent").replace(f"{base}+", ""))
    ax.axhline(0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("observation redshift")
    ax.set_ylabel("model minus simulation outer slope")
    ax.set_title("(d) required not to get worse: the outer profile shape\n"
                 "(positive = the model is too shallow)", fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, FIGDIR / name)
    print(f"  wrote {FIGDIR / name}.png")


if __name__ == "__main__":
    if S37.PRECHECK.exists():
        precheck_figure()
    main_figure()
