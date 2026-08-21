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
experiments/exp53_deposition_only/report.py {table|slice|qa|figures|all} \
[--stage qprofile|shootout] [--only cell,cell] [--dev]`

`qa` runs the project's STANDARD battery (`hongshao.qa.evaluate`: tier 1+2
mass tables, 2b planes, 2c growth planes, 2d size planes, profile visual QA)
on the four deposition-only family cells plus the adopted baseline. exp53's
own figures answer exp53's own question and do not replace it.
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
from hongshao.qa import _pct                                   # noqa: E402

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
    spec, theta = res["spec"], res["theta"]
    kd = S.sign_disagreement(sc["kern_model"], sc["kern_data"], sc["mid"])
    nm = np.array([sc[("kshape", "model", b, pr)][0]
                   for b in range(3) for pr in sc["pairs"]])
    nd = np.array([sc[("kshape", "data", b, pr)][0]
                   for b in range(3) for pr in sc["pairs"]])
    par = dict(zip(spec.theta_names, np.asarray(theta, float)))
    return dict(
        name=res["name"], loss=loss, npar=n_theta,
        q=par.get("q", spec.q_fixed), log_R50=par["log_R50"], g=par["g"],
        mu=par["mu"], sig=par["sig"],
        shape=par.get(spec.shape_name) if spec.shape_name else np.nan,
        m5_all=v[("inner", 5, 0, "all")],
        m5_compact=v[("inner", 5, 0, "compact")],
        m5_extended=v[("inner", 5, 0, "extended")],
        m10_all=v[("inner", 10, 0, "all")],
        grow_m5=sc[("growth", "M(<5)", (2, 0))],
        grow_m10=sc[("growth", "M(<10)", (2, 0))],
        grow_m30=sc[("growth", "M(<30)", (2, 0))],
        grow_out=sc[("growth", "M[50,100]", (2, 0))],
        kern_dex=sc["kern_rms"], kern_sign=kd,
        n_model=float(np.nanmedian(nm)), n_data=float(np.nanmedian(nd)),
        r50k_model=float(np.nanmedian(
            [sc[("kshape", "model", b, pr)][1]
             for b in range(3) for pr in sc["pairs"]])),
        r50k_data=float(np.nanmedian(
            [sc[("kshape", "data", b, pr)][1]
             for b in range(3) for pr in sc["pairs"]])),
        transport=sc["transport_share"],
        beyond50=sc["reach"]["beyond"].get(50.0, np.nan),
        beyond500=sc["reach"]["beyond"].get(500.0, np.nan),
        dep_r50=sc["reach"]["median_r50"], at_cap=sc["reach"]["at_cap"],
        plane_z04=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 0, "all")],
        plane_z20=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 4, "all")],
        plane_cmp=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 0, "compact")][0],
        plane_cmp_ctrl=v[("plane", "kpc:M(<30)|kpc:M(50-100)", 0, "compact")][1],
        r50_z04=v[("size", 50, 0)], r50_z20=v[("size", 50, 4)],
        tilt_m5=v[("tilt", 0, "vs_m5")],
        gate_diff=g["diff_model"][0], gate_diff_data=g["diff_data"][0],
        gate_over=g["over_T1"][1])


def print_table(rows, title):
    order = sorted(rows, key=lambda x: np.nan_to_num(x["kern_dex"], nan=9.0))
    print(f"\n=== exp53 {title} — the FITTED parameters ===")
    print("  (`q` is the transport exponent, 0 = deposition only; log_R50 is "
          "the deposit half-mass\n   radius at t_obs in log10 kpc; `g` its "
          "growth exponent in time; mu/sig the efficiency\n   window, which is "
          "FREE by design because it is entangled with transport)")
    print(f"  {'cell':<15}{'npar':>5}{'loss':>12}{'q':>7}{'log_R50':>9}"
          f"{'g':>7}{'mu':>7}{'sig':>7}{'shape':>8}   at bound")
    flat = 0
    for r in order:
        sh = "-" if not np.isfinite(r["shape"]) else f"{r['shape']:.3f}"
        sig = f"{r['sig']:.3f}" + ("*" if r["sig"] >= K.SIG_FLAT else "")
        flat += r["sig"] >= K.SIG_FLAT
        print(f"  {r['name']:<15}{r['npar']:5d}{r['loss']:12.6f}{r['q']:7.2f}"
              f"{r['log_R50']:9.3f}{r['g']:7.3f}{r['mu']:7.3f}{sig:>8}"
              f"{sh:>8}   {r.get('bounds', '-')}")
    if flat:
        print(f"  * sig >= {K.SIG_FLAT:g}: the efficiency window is "
              f"OPERATIONALLY FLAT (varies < 12% over the sample's ln(1+z) "
              f"range).\n    Read it as 'deposition tracks halo growth with no "
              f"epoch preference', not as a measured width.")

    print(f"\n=== exp53 {title} — THE VERDICT, ranked by the added-light "
          f"kernel distance, NEVER by the loss ===")
    print(f"  {'cell':<15}{'kernel':>8}{'sign':>6}{'kern n':>9}"
          f"{'kern R50':>11}{'M(<10)grow':>11}{'M(<5)grow':>10}"
          f"{'M(<5) all':>10}{'M(<5) cmp':>10}{'transp':>8}"
          f"{'GATEdif':>9}{'GATEovr':>9}")
    print(f"  {'':<15}{'[dex]':>8}{'':>6}{'mod/dat':>9}{'mod/dat':>11}"
          f"{'exp52 .573':>11}{'':>10}{'[%]':>10}{'[%]':>10}{'[%]':>8}"
          f"{'.30-.45':>9}{'|.|<.06':>9}")
    for r in order:
        ts = "-" if not np.isfinite(r["transport"]) else \
            f"{100 * r['transport']:.0f}"
        print(f"  {r['name']:<15}{r['kern_dex']:8.3f}"
              f"{100 * r['kern_sign']:5.0f}%"
              f"{r['n_model']:5.1f}/{r['n_data']:.1f}"
              f"{r['r50k_model']:7.0f}/{r['r50k_data']:.0f}"
              f"{r['grow_m10']:11.3f}{r['grow_m5']:10.3f}"
              f"{100 * r['m5_all']:9.1f}%{100 * r['m5_compact']:9.1f}%"
              f"{ts:>8}{r['gate_diff']:9.2f}{r['gate_over']:+9.3f}")
    print(f"  DATA marks: kernel Sersic n {order[0]['n_data']:.1f}, kernel R50 "
          f"{order[0]['r50k_data']:.0f} kpc, GATE differential "
          f"{order[0]['gate_diff_data']:.2f}. A growth ratio of 1.000 means "
          f"the model grows\n              that region at the truth's rate "
          f"(exp52: 0.573 for M(<10), 0.989 for M[50,100]).")

    print(f"\n=== exp53 {title} — where the model PUTS its stars (deposit "
          f"reach at z=0.4) ===")
    print("  (mass-weighted over the model's own deposits; comparable across "
          "families only because\n   the scale coordinate is a half-mass "
          "radius. exp52 measured 58.5% of the ADOPTED kernel's\n   "
          "TRANSPORTED mass beyond 50 kpc and 12.0% beyond 500 kpc, where the "
          "normalization discards it.)")
    print(f"  {'cell':<15}{'median dep R50':>16}{'beyond 50 kpc':>15}"
          f"{'beyond 500 kpc':>16}{'at 300 kpc cap':>16}")
    for r in order:
        print(f"  {r['name']:<15}{r['dep_r50']:13.1f} kpc"
              f"{100 * r['beyond50']:14.1f}%{100 * r['beyond500']:15.1f}%"
              f"{100 * r['at_cap']:15.1f}%")

    print(f"\n=== exp53 {title} — planes, sizes and the compact-galaxy tilt "
          f"===")
    print(f"  {'cell':<15}{'2b z=0.4':>10}{'2b z=2.0':>10}"
          f"{'2b z=0.4 compact (ctrl)':>26}{'dlogR50 z0.4':>14}{'z2.0':>8}"
          f"{'rho(tilt,M5)':>14}")
    for r in order:
        print(f"  {r['name']:<15}{r['plane_z04']:10.1f}{r['plane_z20']:10.1f}"
              f"{r['plane_cmp']:18.1f} ({r['plane_cmp_ctrl']:.1f})"
              f"{r['r50_z04']:+14.3f}{r['r50_z20']:+8.3f}"
              f"{r['tilt_m5']:+14.2f}")


# --------------------------------------------------------------------------- #
def cmd_table(stage, dev=False, verbose=False):
    gals, e, logms, logmh, data = _setup(dev)
    # pass the CATALOGUE total stellar mass rather than letting qa fall back to
    # the CoG's outermost point, so the stellar-mass binning matches the one
    # exp53's own outskirt analysis uses
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
        row = _row(res, loss, spec.n_theta)
        hits = R53._bound_audit(spec, theta)
        row["bounds"] = "; ".join(hits) if hits else "-"
        row["start"] = sname
        row["spread"] = R53.cell_spread(done, name)
        rows.append(row)
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


#: the deposition-only family set plus the adopted baseline, for the standard
#: QA battery. `moffat-qfree` is included so every QA figure has the incumbent
#: to read against; it is NOT a deposition-only model.
QA_CELLS = ("moffat-q0", "sersic-q0", "expo-q0", "gauss-q0",
            "gompertz_log-q0", "loglogistic-q0", "richards-q0",
            "moffat-qfree")
#: cross-family figures (the q profile, the kernel comparisons) live here; the
#: per-family QA sets live in `figures/<family>/`. Fifty-plus QA figures in one
#: flat directory is unreviewable, and the family is the axis a reader compares
#: along.
OVERVIEW_DIR = "_overview"


def family_figdir(family):
    d = FIGDIR / family
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_qa(dev=False, cells=None, stage="main"):
    """Run the project's STANDARD QA battery on exp53 candidates.

    exp53's own figures answer exp53's own question (how does the added-light
    kernel respond to `q`, and which family matches it). They are not a
    substitute for `hongshao.qa.evaluate`, which is what every promoted model
    in this program has been shown on: tier 1+2 mass tables per bin set, the
    2b observational planes, 2c growth planes, 2d size planes, and the profile
    visual QA — with figures, binned by halo mass.

    Defaults to the four DEPOSITION-ONLY family cells plus the adopted
    baseline for reference. Model CoGs are evaluated at all five epochs, so
    z = 2.0 is an extrapolation test (fits used z <= 1.5).
    """
    from hongshao import qa
    gals, e, logms, logmh, data = _setup(dev)
    done = R53._load(R53._store(stage, dev))
    lookup = dict(R53.all_cells())
    names = list(cells or QA_CELLS)
    FIGDIR.mkdir(exist_ok=True)
    out = {}
    for name in names:
        spec = lookup.get(name)
        if spec is None:
            print(f"  [{name}] unknown cell — skipped")
            continue
        b = R53.best_cell(done, name)
        if b is None:
            print(f"  [{name}] not fitted — skipped")
            continue
        theta, loss, sname = b
        cogs = K.population_cogs(spec, theta, gals, KS_ALL)
        fdir = family_figdir(spec.family)
        print(f"\n{'=' * 78}\nexp53 STANDARD QA — {name} "
              f"({spec.n_theta} parameters, loss {loss:.6f}, best start "
              f"'{sname}')\n"
              f"  {'DEPOSITION ONLY (q = 0)' if not spec.q_free else 'q free'}"
              f" | fitted on z <= 1.5, SCORED at all five epochs\n{'=' * 78}")
        print(f"  figures -> {fdir.relative_to(ROOT)}")
        out[name] = qa.evaluate(cogs, data, e.R, K.ANCHOR_Z,
                                name=f"exp53_{name}{'_dev' if dev else ''}",
                                figdir=fdir, figures=True, bin_by=logmh,
                                bin_label="logMh", bin_by_ms=logms,
                                ms_label="logM$_*$ (total)")
    return out


def cmd_slice(dev=False):
    """The SLICE, for contrast with the profile — no re-optimization.

    exp49's sharpest warning is that in this model a frozen-parameter slice and
    a profiled curve can disagree in SIGN, and exp52's mechanism claim
    ("transport drains the centre") is a slice statement: it holds the adopted
    theta and inspects the transport term. Measuring the slice here, with the
    identical scorers, makes the comparison explicit instead of rhetorical.
    """
    gals, e, logms, logmh, data = _setup(dev)
    spec = K.Spec("moffat", q_free=True)
    base = K.from_exp38_theta(K.adopted_theta())
    rows = []
    for qv in (0.0,) + R53.Q_GRID[1:] + (base[2],):
        th = base.copy()
        th[2] = float(qv)
        name = f"slice-q{qv:.3f}" + (" (adopted)" if qv == base[2] else "")
        res = evaluate_cell(name, spec, th, gals, e, logms, logmh, data)
        row = _row(res, np.nan, spec.n_theta)
        row["bounds"] = "(not fitted — a SLICE)"
        rows.append(row)
    print_table(rows, "the SLICE at the adopted theta (NOT re-optimized)")
    OUTDIR.mkdir(exist_ok=True)
    out = OUTDIR / f"verdict_slice{'_dev' if dev else ''}.json"
    out.write_text(json.dumps(
        [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
          for k, v in r.items()} for r in rows], indent=1))
    print(f"\nwrote {out}")
    return rows


def cmd_figures(stage, dev=False):
    keep, rows = cmd_table(stage, dev)
    gals, e, logms, logmh, data = _setup(dev)
    set_style()
    tag = "_dev" if dev else ""
    if stage == "qprofile":
        fig_qprofile(keep, rows, tag)
    fig_kernels(keep, e, tag, stage)


def fig_qprofile(keep, rows, tag):
    """The GRADED answer. Top row: what the fit buys. Bottom row: what the fit
    does to the observables the loss cannot see. `q = 0` is the left edge."""
    pts = sorted((r["q"], r) for r in rows
                 if r["name"].startswith("moffat-q")
                 and not r["name"].endswith("qfree"))
    if len(pts) < 3:
        print("  (too few profiled q points for the figure)")
        return
    q = np.array([p[0] for p in pts])
    free = next((r for r in rows if r["name"] == "moffat-qfree"), None)
    panels = [
        (lambda r: r["loss"], "production objective", None, False),
        (lambda r: r["sig"], "efficiency-window width $\\sigma$", None, False),
        (lambda r: r["log_R50"], r"deposit $\log R_{50}$ at $t_{obs}$", None,
         False),
        (lambda r: 100 * r["transport"],
         f"transport share of $M_*[50,100]$ growth [{_pct()}]", None, False),
        (lambda r: r["grow_m10"], r"$M_*(<10)$ growth, model/truth", 1.0, True),
        (lambda r: r["grow_out"], r"$M_*[50,100]$ growth, model/truth", 1.0,
         True),
        (lambda r: 100 * r["m5_compact"],
         f"compact-galaxy $M_*(<5)$ bias [{_pct()}]", 0.0, True),
        (lambda r: r["kern_dex"], "added-light kernel distance [dex]", None,
         True)]
    fig, axes = plt.subplots(2, 4, figsize=(17.4, 8.0))
    for ax, (fn, ylab, mark, judged) in zip(axes.ravel(), panels):
        y = np.array([fn(p[1]) for p in pts], float)
        ax.plot(q, y, "o-", color=OKABE_ITO[5 if judged else 0], lw=1.9, ms=5,
                zorder=3)
        if free is not None and np.isfinite(fn(free)):
            ax.axhline(fn(free), color=OKABE_ITO[1], ls="--", lw=1.2, zorder=2,
                       label=f"$q$ free ({free['q']:.2f})")
            ax.legend(fontsize=7, loc="best")
        if mark is not None:
            ax.axhline(mark, color="k", ls=":", lw=1.0, zorder=1)
        ax.set_xlabel("transport exponent $q$ (PROFILED)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].text(0.03, 0.06, "$q=0$: deposition only",
                    transform=axes[0, 0].transAxes, fontsize=8)
    fig.suptitle("exp53 stage A — the PROFILED transport exponent (every other "
                 "parameter re-optimized at each point). Top: what the fit "
                 "spends. Bottom: what the loss cannot see.", fontsize=12)
    fig.tight_layout()
    od = FIGDIR / OVERVIEW_DIR
    od.mkdir(parents=True, exist_ok=True)
    print("wrote", save_fig(fig, od / f"exp53_qprofile{tag}")[0])


def fig_kernels(keep, e, tag, stage):
    """Model vs MEASURED added light, in the coordinates the JUDGE uses.

    An earlier version plotted the raw added Sigma on a symlog axis. That is
    the wrong frame: the measured kernel's innermost annulus is strongly
    NEGATIVE (the core loses mass), so the symlog range is set by a single
    point at 2.35 kpc and the 4-120 kpc band the distance metric actually
    reads gets squeezed into the top fifth of the panel. Here the top row
    shows the positive kernel on log-log axes over the fit band, and the
    bottom row shows `dlog(model/data)` -- which IS the quantity
    `_kernel_distance` medians, so a candidate's rank is readable off the
    figure rather than only off the table.
    """
    names = list(keep)
    if not names:
        return
    ref = keep[names[0]]["score"]
    pairs, mid = ref["pairs"], ref["mid"]
    band = (mid >= S.KERN_LO) & (mid <= S.KERN_HI)
    order = sorted(names, key=lambda n: keep[n]["score"]["kern_rms"])
    # 8 candidates: colour carries the family, dash carries the q setting.
    fams = {}
    for n in order:
        fams.setdefault(keep[n]["spec"].family, len(fams))
    style = {n: dict(color=OKABE_ITO[fams[keep[n]["spec"].family] % len(OKABE_ITO)],
                     ls="-" if keep[n]["spec"].q_free else "--")
             for n in order}
    nrow, ncol = 2, len(pairs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol + 2.2, 7.2),
                             sharex=True)
    axes = np.atleast_2d(axes)
    b = 2                                    # the massive tercile
    for j, pr in enumerate(pairs):
        kd = ref["kern_data"][b, j]
        pos = band & np.isfinite(kd) & (kd > 0)
        ax = axes[0, j]
        ax.plot(mid[pos], kd[pos], "ko-", ms=4, lw=2.0, zorder=10,
                label="MEASURED")
        lo = axes[1, j]
        for n in order:
            km = keep[n]["score"]["kern_model"][b, j]
            ax.plot(mid[pos], np.where(km[pos] > 0, km[pos], np.nan), lw=1.5,
                    alpha=0.95, label=n, **style[n])
            lo.plot(mid[pos], np.log10(np.where(km[pos] > 0, km[pos], np.nan)
                                       / kd[pos]), lw=1.5, alpha=0.95,
                    **style[n])
        ax.set(xscale="log", yscale="log")
        ax.set_title(f"z{K.ANCHOR_Z[pr[0]]}" r"$\to$" f"{K.ANCHOR_Z[pr[1]]}",
                     fontsize=10)
        lo.axhline(0.0, color="k", lw=1.2, zorder=5)
        lo.axhspan(-0.1, 0.1, color="0.85", zorder=0, lw=0)
        lo.set(xscale="log", ylim=(-0.6, 0.6))
        lo.set_xlabel("$R$ [kpc]")
        if j == 0:
            ax.set_ylabel(r"added $\Sigma_*$ [M$_\odot$ kpc$^{-2}$]"
                          "\nmassive tercile", fontsize=9)
            lo.set_ylabel(r"$\log_{10}$(model / measured)", fontsize=9)
        ax.tick_params(labelsize=8)
        lo.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = [lb if lb == "MEASURED" else
              f"{lb}  ({keep[lb]['score']['kern_rms']:.3f})" for lb in labels]
    fig.legend(handles, labels, fontsize=8, loc="center right",
               frameon=False, title="ranked by median $|$dlog$|$ [dex]",
               title_fontsize=8)
    fig.suptitle(f"exp53 {stage} — the added-light kernel against the MEASURED "
                 f"stack (exp38 stage 0.2 operator). Grey band: $\\pm$0.1 dex. "
                 f"Solid $q$ free, dashed $q=0$.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    od = FIGDIR / OVERVIEW_DIR
    od.mkdir(parents=True, exist_ok=True)
    print("wrote", save_fig(fig, od / f"exp53_kernels_{stage}{tag}")[0])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "table"
    dev = "--dev" in sys.argv
    stage = "qprofile"
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
    verbose = "--verbose" in sys.argv
    if cmd == "table":
        cmd_table(stage, dev, verbose)
    elif cmd == "slice":
        cmd_slice(dev)
    elif cmd == "qa":
        only = None
        if "--only" in sys.argv:
            only = sys.argv[sys.argv.index("--only") + 1].split(",")
        cmd_qa(dev, only)
    elif cmd == "figures":
        cmd_figures(stage, dev)
    elif cmd == "all":
        cmd_slice(dev)
        for st in ("qprofile", "shootout"):
            cmd_figures(st, dev)
    else:
        raise SystemExit(__doc__)
