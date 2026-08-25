"""exp54 Stage 3.9 — the QA figure for the profile likelihoods.

ONE FIGURE, seven panels plus a summary, and it exists to make a single
comparison impossible to misread.

  (a)-(g) One panel per parameter. Two curves against the same displacement
      from the incumbent:

        * the CONDITIONAL slice, dashed — the loss rise when this parameter
          alone is moved and the other six are HELD FIXED. This is the curve
          every identifiability number in this experiment has been quoted from;
        * the PROFILED curve, solid — the loss rise when the other six are
          RE-OPTIMISED at every value. This is the honest one.

      The vertical gap between them, on a logarithmic axis, IS the result: it
      is how much of each parameter's apparent determination was an artefact of
      holding the others fixed. The horizontal rule marks one percentage point
      of profile-shape error (13.79% to 14.79% on the normalised curve of
      growth), the only currency in which a width is quoted here — the loss is
      not a log-likelihood, so no threshold on it means anything
      distributionally, and nothing on this figure is a confidence interval.
      Where the profiled curve crosses that rule is how far the parameter can
      move before the BEST ACHIEVABLE fit gets that much worse. A panel whose
      solid curve never reaches the rule is a parameter this sample does not
      determine at all, which is a finding and not a failure.

      A panel is marked AT BOUND when its grid runs into the parameter's own
      hard limit — `log_f0 <= 0`, a deposit half-mass radius equal to its
      halo's radius at redshift 0, is the one to watch. Where that happens the
      curve stops being a property of the model and becomes a property of the
      bound.

  (h) The summary: the fraction of the conditional rise that survives
      re-optimisation, at the outermost grid point, one bar per parameter, on a
      logarithmic axis. 1.0 would mean the conditional number was honest.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage39_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import stage33 as S33                                    # noqa: E402
import stage39_profile as S39                            # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tex                       # noqa: E402

FIGDIR = HERE / "figures"
#: what each parameter does, in words, for a reader who has not read the model
MEANING = {
    "a0": "stellar mass kept by a\n$10^{13.5}\\,M_\\odot$ halo at $z=0$",
    "a_M": "how that changes\nwith halo mass",
    "a_z": "how it changes\nwith redshift",
    "a_Mz": "how the mass dependence\nitself changes with redshift",
    "log_f0": "deposit size as a fraction of\nits halo's radius at $z=0$",
    "b": "how that size fraction\nchanges with redshift",
    "c": "the shape of one\ndeposit's profile",
}
C_COND, C_PROF, C_RULE = "#D55E00", "#0072B2", "#666666"


def figure(name="stage39_profile"):
    import matplotlib.pyplot as plt

    gridf, profdir, _ = S39.paths(False)
    z = np.load(gridf, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    theta = np.asarray(z["theta"], float)
    lo, hi = np.array(S33.spec_from_label(S39.BASE).bounds()).T

    have = [n for n in names if (profdir / f"{n}.npz").exists()]
    if not have:
        raise SystemExit("no profiles on disk — run --only for each parameter")

    set_style()
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4))
    flat = axes.ravel()
    surv, labels = [], []

    for ax, nm in zip(flat, names):
        j = names.index(nm)
        if nm not in have:
            ax.set_axis_off()
            continue
        r = np.load(profdir / f"{nm}.npz", allow_pickle=True)
        grid = np.asarray(r["grid"], float)
        base = float(r["base_loss"])
        prof = np.asarray(r["loss"], float) - base
        cond = np.asarray(r["cond_loss"], float) - base
        d = grid - theta[j]

        ax.plot(d, np.maximum(cond, 1e-6), "--", color=C_COND, lw=1.6,
                marker="o", ms=3.2,
                label="conditional (other six fixed)")
        ax.plot(d, np.maximum(prof, 1e-6), "-", color=C_PROF, lw=2.0,
                marker="s", ms=3.4,
                label="profiled (other six re-optimised)")
        ax.axhline(S39.RISE_1PCT, color=C_RULE, lw=1.0, ls=":")
        ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, max(2.0, float(np.nanmax(cond)) * 2.0))

        edge = int(np.nanargmax(prof))
        s = prof[edge] / cond[edge] if cond[edge] > 1e-12 else np.nan
        surv.append(s)
        labels.append(nm)

        at_bound = (abs(grid[0] - lo[j]) < S39.BOUND_TOL
                    or abs(grid[-1] - hi[j]) < S39.BOUND_TOL)
        title = _tex(f"{nm} = {theta[j]:+.4f}")
        if at_bound:
            title += "   AT BOUND"
        ax.set_title(title + "\n" + MEANING.get(nm, ""), fontsize=9)
        ax.set_xlabel(_tex(f"displacement from the incumbent"))
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel(_tex("loss rise above 1.874253"))
        ax.text(0.03, 0.94, _tex(f"{100 * s:.2f}{_pct()} of the conditional "
                                 f"rise survives"),
                transform=ax.transAxes, fontsize=7.5, va="top", color=C_PROF)
        if nm == names[0]:
            ax.legend(loc="lower right", fontsize=7)

    bx = flat[7]
    order = np.argsort(surv)
    bx.barh(np.arange(len(surv)), np.array(surv)[order], color=C_PROF,
            height=0.62)
    bx.set_yticks(np.arange(len(surv)))
    bx.set_yticklabels([_tex(labels[i]) for i in order], fontsize=8)
    bx.set_xscale("log")
    bx.axvline(1.0, color=C_COND, lw=1.2, ls="--")
    bx.set_xlabel(_tex("profiled rise / conditional rise, at the grid edge"))
    bx.set_title("how much of each parameter's apparent\nconstraint is real",
                 fontsize=9)
    bx.text(1.0, -0.9, _tex("1.0 = the conditional number was honest"),
            fontsize=7, color=C_COND, ha="right")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)

    fig.suptitle(_tex(
        f"exp54 Stage 3.9 — conditional slices against genuine profile "
        f"likelihoods, {S39.BASE}, 2397 galaxies. Dashed: one parameter moved, "
        f"the other six held fixed. Solid: the same move with the other six "
        f"re-optimised.\nDotted rule: one percentage point of profile-shape "
        f"error ({S39.SHAPE_INCUMBENT:.2f}{_pct()} to "
        f"{S39.SHAPE_INCUMBENT + 1:.2f}{_pct()}). The loss is not a "
        f"log-likelihood; nothing here is a confidence interval."),
        fontsize=8.5, y=1.005)
    fig.tight_layout()
    paths = save_fig(fig, FIGDIR / name)
    print("wrote " + ", ".join(str(p) for p in paths))
    plt.close(fig)


if __name__ == "__main__":
    figure()
