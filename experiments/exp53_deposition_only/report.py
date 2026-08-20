"""exp53 — the verdict: score every fitted cell and draw the figures.

`fit.py` produces fitted thetas; this scores them. Nothing is ranked by the
loss. Each cell passes the same three scorers:

  exp47 `judge.verdict`   the standing project-wide verdict (tier 2b planes
                          full and compact/extended against size-matched
                          controls, tier 2d sizes, inner mass per
                          sub-population, the stage-1 tilt)
  exp47 `physics_gates`   the two GATES (differential deposition; outskirt
                          overshoot). Gates, not tie-breakers.
  exp53 `score.scorecard` the MEASURED added-light kernel (exp38 stage 0.2
                          operator, cell by cell), exp52's differential slopes
                          and core dlogSigma, the transport share, and the
                          population-summed growth ratios that carry exp53's
                          falsifiable prediction.

The model is fitted on `KS = [0,1,2,3]` (z <= 1.5, the adopted exp40 scope) but
SCORED at all five epochs, so z = 2.0 is an extrapolation test -- the standing
convention since exp40/exp43.

Run: `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp53_deposition_only/report.py {table|figures|all} \
[--stage qprofile|shootout] [--dev]`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))
sys.path.insert(0, str(ROOT / "experiments/exp47_compact_defect"))

import kernel as K                                   # noqa: E402
import fit as R53                                    # noqa: E402
import score as S                                    # noqa: E402
import stage2_multiepoch as s2                       # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style   # noqa: E402

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
KS_ALL = [0, 1, 2, 3, 4]
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"


def _setup(dev):
    pop = np.load(POP_NPZ)
    rows = pop["dev100"] if dev else None
    s2._w_init(rows)
    gals, e = s2._W["gals"], s2._W["e"]
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    logmh = np.array([pop["logmh"][g["row"]] for g in gals])
    data = np.stack([np.stack([g["data"][k] for k in KS_ALL]) for g in gals])
    return gals, e, logms, logmh, data


def evaluate_cell(name, spec, theta, gals, e, logms, logmh, data):
    """Every scorer, for one fitted theta. Scored at ALL FIVE epochs."""
    import judge
    cogs = K.population_cogs(spec, theta, gals, KS_ALL)
    v = judge.verdict(cogs, data, e.R)
    gates = judge.physics_gates(cogs, data, logms, e)
    sc = S.scorecard(spec, theta, gals, e.R, logms, logmh, data, ks=KS_ALL)
    return dict(name=name, spec=spec, theta=theta, cogs=cogs, verdict=v,
                gates=gates, score=sc)


def _row(res, loss, n_theta):
    v, g, sc = res["verdict"], res["gates"], res["score"]
    ks = KS_ALL
    pr_long = (max(ks), min(ks))
    kd = S.sign_disagreement(sc["kern_model"], sc["kern_data"], sc["mid"])
    nm = np.array([sc[("kshape", "model", b, pr)][0]
                   for b in range(3) for pr in sc["pairs"]])
    nd = np.array([sc[("kshape", "data", b, pr)][0]
                   for b in range(3) for pr in sc["pairs"]])
    return dict(
        name=res["name"], loss=loss, npar=n_theta,
        m5_all=v[("inner", 5, 0, "all")],
        m5_compact=v[("inner", 5, 0, "compact")],
        m10_all=v[("inner", 10, 0, "all")],
        grow_m10=sc[("growth", "M(<10)", (2, 0))],
        grow_m5=sc[("growth", "M(<5)", (2, 0))],
        grow_out=sc[("growth", "M[50,100]", (2, 0))],
        kern_dex=sc["kern_rms"], kern_sign=kd,
        n_model=float(np.nanmedian(nm)), n_data=float(np.nanmedian(nd)),
        transport=sc["transport_share"],
        plane_z04=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 0, "all")],
        plane_z20=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 4, "all")],
        plane_cmp=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 0, "compact")][0],
        plane_cmp_ctrl=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 0, "compact")][1],
        r50_z04=v[("size", 50, 0)], r50_z20=v[("size", 50, 4)],
        gate_diff=g["diff_model"][0], gate_diff_data=g["diff_data"][0],
        gate_over=g["over_T1"][1], pr_long=pr_long)


def print_table(rows, title):
    print(f"\n=== exp53 {title} — ranked by the added-light kernel distance, "
          f"NEVER by the loss ===")
    print(f"  {'cell':<18}{'npar':>5}{'loss':>11}"
          f"{'kernel':>8}{'sign':>6}{'n mod/dat':>11}"
          f"{'M(<10) grow':>12}{'M(<5) bias':>11}{'M(<5) cmp':>10}"
          f"{'transp':>8}{'GATEdiff':>10}{'GATEover':>9}")
    print(f"  {'':<18}{'':>5}{'':>11}{'[dex]':>8}{'':>6}{'':>11}"
          f"{'(exp52 .573)':>12}{'[%]':>11}{'[%]':>10}{'[%]':>8}"
          f"{'(.30-.45)':>10}{'(|.|<.06)':>9}")
    for r in sorted(rows, key=lambda x: (np.nan_to_num(x["kern_dex"], nan=9))):
        ts = "-" if not np.isfinite(r["transport"]) else f"{100*r['transport']:.0f}"
        print(f"  {r['name']:<18}{r['npar']:5d}{r['loss']:11.6f}"
              f"{r['kern_dex']:8.3f}{100*r['kern_sign']:5.0f}%"
              f"{r['n_model']:6.1f}/{r['n_data']:.1f}"
              f"{r['grow_m10']:12.3f}{100*r['m5_all']:10.1f}%"
              f"{100*r['m5_compact']:9.1f}%{ts:>8}"
              f"{r['gate_diff']:10.2f}{r['gate_over']:+9.3f}")
    print(f"  data marks: GATE differential {rows[0]['gate_diff_data']:.2f}; "
          f"kernel Sersic n {rows[0]['n_data']:.1f}")
    print("\n  planes (tier 2b M(<30) vs M(50-100), centered E/floor) and sizes")
    print(f"  {'cell':<18}{'z=0.4 all':>11}{'z=2.0 all':>11}"
          f"{'z=0.4 compact (ctrl)':>24}{'dlogR50 z0.4':>14}{'z2.0':>8}")
    for r in sorted(rows, key=lambda x: (np.nan_to_num(x["kern_dex"], nan=9))):
        print(f"  {r['name']:<18}{r['plane_z04']:11.1f}{r['plane_z20']:11.1f}"
              f"{r['plane_cmp']:16.1f} ({r['plane_cmp_ctrl']:.1f})"
              f"{r['r50_z04']:+14.3f}{r['r50_z20']:+8.3f}")


# --------------------------------------------------------------------------- #
def cmd_table(stage, dev=False, verbose=False):
    gals, e, logms, logmh, data = _setup(dev)
    done = R53._load(R53._store(stage, dev))
    cells = (R53.qprofile_cells() if stage == "qprofile"
             else R53.shootout_cells())
    rows, keep = [], {}
    for name, spec in cells:
        b = R53.best_cell(done, name)
        if b is None:
            continue
        theta, loss, sname = b
        res = evaluate_cell(name, spec, theta, gals, e, logms, logmh, data)
        rows.append(_row(res, loss, spec.n_theta))
        keep[name] = res
        if verbose:
            S.print_scorecard(res["score"], name)
    if not rows:
        raise SystemExit(f"no fitted cells in {R53._store(stage, dev)}")
    print_table(rows, stage)
    OUTDIR.mkdir(exist_ok=True)
    out = OUTDIR / f"verdict_{stage}{'_dev' if dev else ''}.json"
    out.write_text(json.dumps(
        [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
          for k, v in r.items() if k != "pr_long"} for r in rows], indent=1))
    print(f"\nwrote {out}")
    return keep, rows


def cmd_figures(stage, dev=False):
    keep, rows = cmd_table(stage, dev)
    gals, e, logms, logmh, data = _setup(dev)
    set_style()
    tag = "_dev" if dev else ""
    if stage == "qprofile":
        fig_qprofile(keep, rows, tag)
    fig_kernels(keep, e, tag, stage)


def fig_qprofile(keep, rows, tag):
    """The graded answer: loss, the central-mass prediction, and the kernel
    distance, all against the profiled transport exponent."""
    pts = []
    for r in rows:
        nm = r["name"]
        if not nm.startswith("moffat-q") or nm.endswith("qfree"):
            continue
        pts.append((float(nm.split("-q")[1]), r))
    pts.sort()
    q = np.array([p[0] for p in pts])
    free = [r for r in rows if r["name"] == "moffat-qfree"]
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.1))
    panels = [
        ("loss", lambda r: r["loss"], "production objective", None),
        ("grow_m10", lambda r: r["grow_m10"],
         r"$M_*(<10\,$kpc$)$ growth, model/truth", 1.0),
        ("m5_compact", lambda r: 100 * r["m5_compact"],
         r"compact-galaxy $M_*(<5)$ bias [%]", 0.0),
        ("kern_dex", lambda r: r["kern_dex"],
         "added-light kernel distance [dex]", 0.0)]
    for ax, (key, fn, ylab, mark) in zip(axes, panels):
        y = np.array([fn(p[1]) for p in pts])
        ax.plot(q, y, "o-", color=OKABE_ITO[5], lw=1.8, ms=5, zorder=3)
        if free:
            ax.axvline(0.0, color="0.85", lw=6, zorder=0)
            ax.plot([], [])
        if mark is not None:
            ax.axhline(mark, color="k", ls=":", lw=1.0, zorder=1)
        ax.set_xlabel("transport exponent $q$ (profiled)")
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=9)
    axes[0].text(0.03, 0.95, "$q=0$: deposition only", transform=axes[0].transAxes,
                 fontsize=8, va="top")
    fig.suptitle("exp53 stage A — the PROFILED transport exponent: every other "
                 "parameter re-optimized at each point (a slice is not a "
                 "profile)", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / f"exp53_qprofile{tag}")[0])


def fig_kernels(keep, e, tag, stage):
    """Model vs MEASURED added light, the massive tercile, all epoch pairs."""
    names = [n for n in keep]
    if not names:
        return
    ref = keep[names[0]]["score"]
    pairs = ref["pairs"]
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.3 * len(pairs), 8.0),
                             sharex=True)
    axes = np.atleast_2d(axes)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(names)))
    for irow, b in enumerate((0, 2)):                # low and high tercile
        for j, pr in enumerate(pairs):
            ax = axes[irow, j]
            ax.plot(ref["mid"], ref["kern_data"][b, j], "ko-", ms=3, lw=1.6,
                    label="MEASURED", zorder=5)
            for c, nm in zip(colors, names):
                sc = keep[nm]["score"]
                ax.plot(sc["mid"], sc["kern_model"][b, j], "-", lw=1.3,
                        color=c, label=nm, alpha=0.9)
            ax.axhline(0.0, color="0.7", lw=0.8)
            ax.set(xscale="log", yscale="symlog")
            if irow == 0:
                ax.set_title(f"z{K.ANCHOR_Z[pr[0]]}" r"$\to$"
                             f"{K.ANCHOR_Z[pr[1]]}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"tercile T{b+1}\n"
                              r"added $\Sigma_*$ [M$_\odot$ kpc$^{-2}$]",
                              fontsize=9)
            if irow == 1:
                ax.set_xlabel("$R$ [kpc]")
    axes[0, -1].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"exp53 {stage} — the added-light kernel: candidates against "
                 "the MEASURED stack (exp38 stage 0.2 operator)", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / f"exp53_kernels_{stage}{tag}")[0])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    dev = "--dev" in sys.argv
    stage = "qprofile"
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    verbose = "--verbose" in sys.argv
    if cmd == "table":
        cmd_table(stage, dev, verbose)
    elif cmd == "figures":
        cmd_figures(stage, dev)
    elif cmd == "all":
        for st in ("qprofile", "shootout"):
            if R53._store(st, dev).exists():
                cmd_figures(st, dev)
    else:
        raise SystemExit(__doc__)
