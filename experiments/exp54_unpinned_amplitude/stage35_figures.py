"""exp54 Stage 3.5 — figures for the richer time dependence.

Two figures.

`stage35_time_law` — what the enrichment learned and what it bought.

  (a) The fitted efficiency curve `log10 eps_surv` against redshift at a fixed
      halo mass, for the incumbent and every enriched variant. The shaded
      histogram behind it is where the deposited halo mass actually is, so a
      wiggle drawn in a redshift range that carries almost no mass can be
      recognised as decoration rather than physics.
  (b) The same curves with the incumbent subtracted -- the correction alone, in
      dex. This is the whole of what the extra parameters do.
  (c) **The ladder that decides whether any of this was reachable.** Shared
      laws at the top, including the free shared curve that bounds all of them;
      then the rungs that give each galaxy its OWN numbers. The blocks must not
      be compared across, and are separated and labelled with their sample for
      that reason -- reading the per-galaxy rungs as a target for a shared law
      is the mistake this stage was launched on.
  (d) The normalized-shape score epoch by epoch, incumbent against variants,
      so a gain concentrated at one epoch cannot hide inside an average.

`stage35_identifiability` — whether the new coefficients are determined, which
is the constraint that actually binds.

  (a) One point per parameter of every variant: the conditional loss rise when
      that parameter alone is displaced, against its bootstrap spread over
      galaxies in units of its own value. The two populations separate
      completely, with no overlap in either coordinate.
  (b) The price. The efficiency law's four original parameters are shared by
      every variant, so their bootstrap spread compares directly across the
      row, and it shows what the new terms cost the terms already there.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage35_figures.py
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
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tex, _zcolors             # noqa: E402

FIGDIR = HERE / "figures"
CEILING = HERE / "outputs" / "stage34_ceiling.npz"
PERM = HERE / "outputs" / "stage34_perm_exact.npz"
OBJ = HERE / "outputs" / "stage34_objective.npz"
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: colour per variant; the incumbent is always black
_VC = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#7570B3",
       "#666666"]


def _load():
    z = np.load(S35.OUT, allow_pickle=True)
    labels = [str(v) for v in z["labels"]]
    out = {}
    for i, lab in enumerate(labels):
        out[lab] = dict(
            theta=z[f"theta::{lab}"], sA=z[f"sA::{lab}"], sF=z[f"sF::{lab}"],
            loss=float(z["loss"][i]), npar=int(z["npar"][i]),
            shape_pct=float(z["shape_pct"][i]), ident=float(z["ident"][i]),
            tilt=float(z["tilt"][i]), over=float(z["over"][i]),
            rank=float(z["mean_rank"][i]))
    return labels, out, z


def _eps_curve(label, theta, zz, logmh=13.5):
    """log10 eps_surv against redshift at one halo mass, one deposit each."""
    sp = S33.spec_from_label(label)
    eff = sp.unpack(theta)[0]
    h = type("H", (), dict(logmh=np.full(len(zz), logmh), z=np.asarray(zz),
                           dlogc=np.zeros(len(zz))))
    return M.log_eps(sp, eff, h)


def _bound_rungs():
    """The Stage 3.4 rungs, as percentage profile-shape errors.

    SPLIT BY SAMPLE, because they are not on one. The `masked` block is this
    stage's own sample -- each epoch's mh-complete galaxies -- scored under the
    NNLS surrogate, so each is what ONE feasible solution achieves and the true
    optimum is at most that. The `cohort` block is the 840 galaxies followed at
    all five epochs, optimised under the EXACT production objective; those are
    the 4.5% and 5.0% the plan quotes, and they belong to a different galaxy
    set than everything else on the chart.
    """
    L = F.L_F_REF
    masked, cohort = {}, {}
    if CEILING.exists():
        z = np.load(CEILING, allow_pickle=True)
        for rung, name in (("bins2", "2 time groups, free per galaxy"),
                           ("bins3", "3 time groups, free per galaxy"),
                           ("bins8", "8 time groups, free per galaxy"),
                           ("permuted", "fitted to ANOTHER galaxy's profile"),
                           ("ceiling", "every deposit free, per galaxy")):
            k = f"gompertz_log-E2-S2|masked|{rung}_sF"
            if k in z.files:
                masked[name] = 100 * float(np.nanmean(z[k])) * L
    if OBJ.exists():
        z = np.load(OBJ)
        cohort["every deposit free (exact objective)"] = \
            100 * float(np.mean(z["sF_shape"])) * L
    if PERM.exists():
        z = np.load(PERM, allow_pickle=True)
        cohort["ANOTHER galaxy's profile (exact objective)"] = \
            100 * float(np.mean(z["sF"])) * L
    return masked, cohort


def time_law_figure(labels, res, name="stage35_time_law"):
    import matplotlib.pyplot as plt
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))
    inc = S35.LABELS[0]

    # (a) and (b): the fitted efficiency law, over the mass that feels it
    ax, ax2 = axes[0, 0], axes[0, 1]
    zz = np.geomspace(0.4, 15.0, 600)
    base = _eps_curve(inc, res[inc]["theta"], zz)
    zc = np.load(S35.OUT)
    dz, wh, ws = zc["dep_z"], zc["dep_w_halo"], zc["dep_w_star"]
    for a, w, nm in ((ax, wh, "accreted halo mass"),
                     (ax2, ws, "the stellar mass the incumbent makes")):
        tw = a.twinx()
        o = np.argsort(dz)
        # per unit ln(1+z), so the height is a density and not a bin artefact
        dens = w[o] / np.gradient(np.log1p(dz[o])) / w.sum()
        tw.fill_between(1 + dz[o], dens, color="#CCCCCC", alpha=0.55, lw=0,
                        zorder=0)
        tw.set_ylim(0, dens.max() * 3.2)
        tw.set_yticks([])
        tw.set_ylabel(f"where {nm} is deposited", fontsize=7.5, color="#777777")
    for i, lab in enumerate(labels):
        c = "k" if lab == inc else _VC[i % len(_VC)]
        y = _eps_curve(lab, res[lab]["theta"], zz)
        tag = lab.split("-")[1]
        kw = dict(color=c, lw=2.4 if lab == inc else 1.5, zorder=3,
                  ls="-" if lab == inc else "--", label=tag)
        ax.plot(1 + zz, y, **kw)
        ax2.plot(1 + zz, y - base, **kw)
    for a in (ax, ax2):
        a.set_xscale("log")
        a.set_xlim(1.4, 16)
        a.set_xticks([1.4, 2, 3, 5, 8, 16])
        a.set_xticklabels(["0.4", "1", "2", "4", "7", "15"])
        a.set_xticks([], minor=True)
        a.set_xlabel("redshift at which the mass was accreted")
        a.legend(fontsize=7.5, ncol=2, loc="upper left")
        a.set_zorder(2)
        a.patch.set_visible(False)
    ax.set_ylabel(_tex("log10 surviving central mass fraction, "
                       "at log10 Mh = 13.5"))
    ax.set_title("(a) the fitted efficiency law", fontsize=10)
    ax2.axhline(0, color="k", lw=0.8, alpha=0.5, zorder=1)
    ax2.set_ylabel("difference from the incumbent [dex]")
    ax2.set_title(f"(b) the correction alone — and only "
                  f"{100 * ws[dz > 4].sum() / ws.sum():.0f}{_pct()} of the "
                  f"stellar mass\nis made above z = 4, "
                  f"{100 * ws[dz > 7].sum() / ws.sum():.0f}{_pct()} above z = 7",
                  fontsize=10)

    # (c) the ladder
    ax = axes[1, 0]
    masked, cohort = _bound_rungs()
    shared = [(f"{lab.split('-')[1]}  ({res[lab]['npar']} params)",
               res[lab]["shape_pct"]) for lab in labels]
    rows = ([("A SHARED LAW — no per-galaxy freedom (this stage)", None)]
            + sorted(shared, key=lambda kv: -kv[1])
            + [("PER-GALAXY FREEDOM — same galaxies, surrogate solve", None)]
            + sorted(masked.items(), key=lambda kv: -kv[1])
            + [("PER-GALAXY FREEDOM — the 840 followed at all 5 epochs", None)]
            + sorted(cohort.items(), key=lambda kv: -kv[1]))
    ys = np.arange(len(rows))[::-1]
    for y, (nm, v) in zip(ys, rows):
        if v is None:
            ax.text(0.0, y, nm, fontsize=8.5, weight="bold", va="center",
                    ha="left", color="#333333",
                    bbox=dict(fc="white", ec="none", pad=1.2))
            continue
        head = any(nm.startswith(t) for t in ("E2", "E6", "E7", "E8"))
        ax.barh(y, v, color="#0072B2" if head else "#BBBBBB",
                edgecolor="k", lw=0.4, height=0.62)
        ax.text(v + 0.15, y, f"{v:.2f}{_pct()}", fontsize=7.5, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([nm if v is not None else "" for nm, v in rows],
                       fontsize=7.5)
    ax.set_xlabel(f"profile-shape error [{_pct()}]")
    ax.set_xlim(0, max(v for _, v in rows if v is not None) * 1.30)
    ax.set_title("(c) what is reachable, and by what kind of freedom",
                 fontsize=10)

    # (d) per-epoch shape score
    ax = axes[1, 1]
    for i, lab in enumerate(labels):
        c = "k" if lab == inc else _VC[i % len(_VC)]
        ax.plot(Z, 100 * np.asarray(res[lab]["sF"]) * F.L_F_REF, "o-",
                color=c, lw=2.2 if lab == inc else 1.4, ms=4,
                ls="-" if lab == inc else "--", label=lab.split("-")[1])
    ax.set_xlabel("observation redshift")
    ax.set_ylabel(f"profile-shape error at this epoch [{_pct()}]")
    ax.set_title("(d) epoch by epoch, so an average cannot hide a trade")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, FIGDIR / name)
    print(f"  wrote {FIGDIR / name}.png")


def identifiability_figure(name="stage35_identifiability"):
    """Whether the new coefficients are determined — the binding constraint.

    (a) plots the two things that decide it against each other, one point per
    parameter of every variant. The horizontal axis is the CONDITIONAL loss
    rise: how much worse the fit gets when that parameter alone is displaced
    and nothing is re-optimised. It is not a profile likelihood and a steep
    rise does not prove a parameter identified, but a FLAT one does prove the
    loss cannot locate it. The vertical axis is the bootstrap spread over
    galaxies divided by the parameter's own value: above 0.5 the resampled data
    cannot pin the parameter to within its own size.

    (b) is the price. The efficiency law's four original parameters are shared
    by every variant, so their bootstrap spread can be compared directly across
    the row, and it shows what adding the new terms costs the terms that were
    already there.
    """
    import matplotlib.pyplot as plt
    files = sorted(S35.IDENTDIR.glob("*.npz")) if S35.IDENTDIR.exists() else []
    if not files:
        print("  no identifiability reports yet — skipping that figure")
        return
    order = {l.split("-")[1]: i for i, l in enumerate(S35.LABELS)}
    reps = sorted((np.load(f, allow_pickle=True) for f in files),
                  key=lambda z: order.get(str(z["label"]).split("-")[1], 99))
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    ax = axes[0]
    for z in reps:
        tag = str(z["label"]).split("-")[1]
        names = [str(v) for v in z["names"]]
        th, sl, B = z["theta"], z["slices"], z["boot"]
        rel = B.std(axis=0) / np.maximum(np.abs(th), 1e-9)
        for i, nm in enumerate(names):
            isnew = nm[:1] in ("c", "s", "v") and nm != "c"
            ax.scatter(max(sl[i], 1e-4), rel[i], s=52 if isnew else 22,
                       color="#D55E00" if isnew else "#AAAAAA",
                       edgecolor="k" if isnew else "none", lw=0.5,
                       zorder=3 if isnew else 2)
            if isnew:
                ax.annotate(f"{tag}:{nm}", (max(sl[i], 1e-4), rel[i]),
                            textcoords="offset points", xytext=(6, 3),
                            fontsize=6.5)
    ax.axhline(0.5, color="#B22222", lw=1.0, ls=":")
    ax.text(0.985, 0.52, "above this line the bootstrap cannot pin a "
            "parameter to within its own size", fontsize=7, color="#B22222",
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("conditional loss rise when this parameter alone is moved")
    ax.set_ylabel(_tex("bootstrap spread over galaxies / |value|"))
    ax.set_title("(a) grey = the parameters the incumbent already had,\n"
                 "orange = the new time coefficients", fontsize=10)

    ax = axes[1]
    shared = ["a0", "a_M", "a_z", "a_Mz"]
    w = 0.8 / len(reps)
    cols = _zcolors(len(reps))
    for j, z in enumerate(reps):
        names = [str(v) for v in z["names"]]
        rel = z["boot"].std(axis=0) / np.maximum(np.abs(z["theta"]), 1e-9)
        v = [100 * rel[names.index(n)] for n in shared]
        ax.bar(np.arange(len(shared)) + j * w - 0.4 + w / 2, v, width=w,
               color="k" if j == 0 else cols[j], edgecolor="k", lw=0.3,
               label=str(z["label"]).split("-")[1])
    ax.set_xticks(range(len(shared)))
    ax.set_xticklabels(shared)
    ax.set_ylabel(_tex("bootstrap spread / |value|") + f" [{_pct()}]")
    ax.set_title("(b) what the enrichment costs the parameters that were\n"
                 "already there (black = the incumbent)", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    save_fig(fig, FIGDIR / name)
    print(f"  wrote {FIGDIR / name}.png")


def _load_bounds(res):
    """Fold the E8 ceiling fits into the same dict as the judged variants."""
    z = np.load(S35.OUT, allow_pickle=True)
    if "bound_labels" not in z.files:
        return []
    out = []
    for i, lab in enumerate([str(v) for v in z["bound_labels"]]):
        res[lab] = dict(theta=z[f"theta::{lab}"], sF=z[f"sF::{lab}"],
                        sA=z[f"sA::{lab}"], npar=int(z["bound_npar"][i]),
                        shape_pct=float(z["bound_shape_pct"][i]),
                        loss=float(z["bound_loss"][i]))
        out.append(lab)
    return out


if __name__ == "__main__":
    labels, res, _ = _load()
    bounds = _load_bounds(res)
    order = [l for l in S35.LABELS if l in res]
    time_law_figure(order, res)
    identifiability_figure()
