"""exp63 — the single-epoch round (2026-08-28): the z=0.4 VISUAL GATE and the
lever sweep by evaluation.

The user's direction after Stage 2: judge the model at z=0.4 ONLY, by the
average curves of growth in halo-mass and stellar-mass terciles (the
`qa_bins` / `qa_bins_ms` panels) and the planes; the targets are the highest
halo-mass tercile (Stage 2: -12 per cent at 2-4 kpc) and the outskirts, where
Stage 2 is not better than the incumbent. Nothing at z > 0.4 is looked at
until z=0.4 is right by this gate.

Two entry points:

  --gate  TAGS      the gate for a list of fit files (`outputs/stage2_fit{tag}.npz`),
                    always with the exact-engine incumbent ("baseline") and
                    Stage 2 ("stage2") as references: the tercile residual
                    table at seven radii, the planes, the loss components, and
                    ONE overlay figure `figures/exp63_single_epoch_gate{name}`
                    (median CoGs by halo-mass tercile; residuals by halo-mass
                    and by stellar-mass tercile; z=0.4 only).
  --qa TAGS         the STANDARD `qa.evaluate` battery (qa_bins, qa_bins_ms,
                    qa_planes, ...) for each fit file at z=0.4 ONLY — the model
                    and the data are passed as single-epoch arrays, so nothing
                    at z > 0.4 is evaluated; figures `figures/qa/qa_*_exp63_z04{tag}`.
  --epoch k         (with --gate or --qa) judge at epoch k (0-4 = z 0.4, 0.7, 1.0,
                    1.5, 2.0) instead of z=0.4: that epoch's data, halo mass and
                    stellar mass, the model's integral truncated at that epoch.
                    The gate's QA sample is the galaxies with finite, positive
                    data at that epoch (the fit used the sane + mh-complete mask).
  --sweep [TAG]     the LEVER SWEEP: every candidate lever moved one at a time
                    from the base solution (Stage 2, or the fit at TAG) with no
                    refit, read by the same table. Standing method: sweep a
                    candidate's gate leverage by evaluation BEFORE spending a
                    fit; slices may disqualify, never promote.

Levers (model2.LEVER_NAMES and the existing parameters): the compact share's
floor `w_min`, the halo-mass exponents `g_c` / `g_e` of the two sizes (the size
level is held fixed by compensating log_f: the lever changes only the trend
with halo mass), the extended kernel's shape `c_e` and family (Sersic index),
the compact index `n_c`, the split's position and width, the compact size.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/single_epoch.py --sweep [--smoke]
     experiments/exp63_analytic_growth/single_epoch.py --gate _frozen,_frozen_gc [--name X]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import fit as F                                          # noqa: E402
import model2 as M2                                      # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2                                  # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 100
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
R = F.R_GRID
#: the radii the table reads: the centre (2, 3.7, 4.9 kpc), 10, 30, 100, 148 kpc
TABLE_R = (2.0, 3.72, 4.92, 10.25, 32.58, 103.45, 148.22)
TABLE_I = tuple(int(np.argmin(np.abs(R - r))) for r in TABLE_R)
OUTSKIRT = R >= 50.0
CENTRE = R <= 5.0
PLANES = (("kpc:M(<30)", "kpc:M(30-50)"), ("kpc:M(<30)", "kpc:M(50-100)"))
COLORS = ("#999999", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


# --------------------------------------------------------------------------- #
# the sample                                                                   #
# --------------------------------------------------------------------------- #
class Sample:
    """Everything the gate needs, at ONE epoch (default z=0.4)."""

    def __init__(self, smoke=False, epoch=0):
        recs, data, mask, lmh, spec_inc, th_inc, _ = S0.build(smoke)
        self.curves = E.build_curves(recs, verbose=False)
        self.data = data                                  # (n, 5, 24)
        self.mask = np.asarray(mask, bool)
        self.epoch = epoch
        self.z = E.ANCHOR_Z[epoch]
        self.lmh = lmh[:, epoch]
        self.th_inc = th_inc
        self.good = np.isfinite(data[:, epoch]).all(1) & (data[:, epoch] > 0).all(1) & np.isfinite(self.lmh)
        self.fit_rows = np.where(self.mask[:, epoch] & self.good)[0]
        self.d0 = data[:, epoch]
        self.logms = np.log10(np.clip(data[:, epoch, -1], 1.0, None))
        self._problem = {}

    def predict(self, spec2, theta):
        """M*(<R) at this epoch, full nodes: (n, 24)."""
        return M2.predict2(spec2, theta, self.curves, R, epochs=(self.epoch,))[:, 0]

    def problem(self, spec2):
        key = spec2                                       # frozen dataclass: hashable, all fields
        if key not in self._problem:
            pr = S2.Problem2(spec2, self.curves, self.data, self.fit_rows, R, l_s_ref=None, epoch=self.epoch)
            ref = S2.Problem2(M2.Spec2(), self.curves, self.data, self.fit_rows, R, l_s_ref=None, epoch=self.epoch)
            pr.l_s_ref = ref.scores(M2.nested_theta(self.th_inc))[2]
            self._problem[key] = pr
        return self._problem[key]

    def loss_components(self, spec2, theta):
        """The production (D4) loss components — the same for every product,
        whatever objective its fit used, so the column is comparable."""
        a, f, s, nb = self.problem(spec2).scores(theta)
        return a, f, s, a * a + f * f + s * s


# --------------------------------------------------------------------------- #
# the gate                                                                     #
# --------------------------------------------------------------------------- #
def tercile_residuals(m0, d0, bin_by):
    """Median (model-data)/data per tercile of `bin_by`, raw and amplitude-
    pinned (model scaled to the data's M(<148 kpc), exactly as the `qa_bins`
    figure pins): edges, raw (3, 24), pin (3, 24)."""
    edges = np.quantile(bin_by, [0, 1 / 3, 2 / 3, 1])
    raw, pin = np.empty((3, len(R))), np.empty((3, len(R)))
    for b in range(3):
        sel = (bin_by >= edges[b]) & (bin_by <= edges[b + 1] + 1e-9)
        mm, dd = m0[sel], d0[sel]
        raw[b] = np.median((mm - dd) / dd, axis=0)
        pp = mm / mm[:, -1][:, None] * dd[:, -1][:, None]
        pin[b] = np.median((pp - dd) / dd, axis=0)
    return edges, raw, pin


def plane_numbers(m0, d0):
    """Slope and scatter of the two kpc planes, model and truth."""
    t, m = qa.measure_all(m0[:, None, :], d0[:, None, :], R)[:2]
    out = {}
    for x, y in PLANES:
        xt, yt = np.log10(t[x][:, 0]), np.log10(t[y][:, 0])
        xm, ym = np.log10(np.clip(m[x][:, 0], 1, None)), np.log10(np.clip(m[y][:, 0], 1, None))
        st, sm = qa.plane_stats(xt, yt), qa.plane_stats(xm, ym)
        out[y] = (sm["slope"], st["slope"], sm["scatter"], st["scatter"])
    return out


def gate_row(sample, m0, good=None):
    """The gate's numbers for one product at z=0.4."""
    g = sample.good if good is None else good
    d0 = sample.d0[g]; mm = m0[g]
    _, raw_h, pin_h = tercile_residuals(mm, d0, sample.lmh[g])
    _, raw_s, pin_s = tercile_residuals(mm, d0, sample.logms[g])
    planes = plane_numbers(mm, d0)
    return dict(raw_h=raw_h, pin_h=pin_h, raw_s=raw_s, pin_s=pin_s, planes=planes,
                worst_centre=float(np.max(np.abs(raw_h[:, CENTRE]))),
                worst_outskirt=float(np.max(np.abs(raw_h[:, OUTSKIRT]))),
                top_centre=float(np.mean(raw_h[2, CENTRE])),
                rms_h=float(np.sqrt(np.mean(raw_h ** 2))))


def print_table(rows, title):
    """rows: list of (name, gate_row_dict, loss tuple or None)."""
    print(f"\n{RULE}\n{title}\n{RULE}")
    hdr = "".join(f"{r:>7.0f}" for r in TABLE_R)
    print(f"  {'product':<20}{'tercile':<8}{'':>2}{hdr}   | worst |raw| centre / outskirt | rms(h)")
    for name, g, loss in rows:
        for b, lab in enumerate(("low", "mid", "top")):
            line = "".join(f"{100 * g['raw_h'][b, i]:>+7.1f}" for i in TABLE_I)
            pline = "".join(f"{100 * g['pin_h'][b, i]:>+7.1f}" for i in TABLE_I)
            tail = (f"   | {100 * g['worst_centre']:5.1f} / {100 * g['worst_outskirt']:5.1f}   | {100 * g['rms_h']:5.2f}"
                    if b == 0 else "")
            print(f"  {name if b == 0 else '':<20}{lab + ' raw':<8}{'':>2}{line}{tail}")
            print(f"  {'':<20}{lab + ' pin':<8}{'':>2}{pline}")
        pl = "; ".join(f"{k.split(':')[1]}|M(<30): slope {v[0]:.2f} (truth {v[1]:.2f}), scatter {v[2]:.3f} ({v[3]:.3f})"
                       for k, v in g["planes"].items())
        ls = (f"loss A {loss[0]:.3f} F {loss[1]:.3f} S {loss[2]:.3f} -> {loss[3]:.4f}" if loss else "")
        print(f"  {'':<20}{'planes':<8}{'':>2}{pl}   {ls}")
        sm = "".join(f"{100 * g['raw_s'][b, TABLE_I[-1]]:>+7.1f}" for b in range(3))
        print(f"  {'':<20}{'M* bins':<8}{'':>2}raw at 148 kpc, low/mid/top M* tercile:{sm}")


def gate_figure(sample, products, name, title):
    """products: ordered dict name -> m0 (n, 24). One figure, z=0.4 only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _pct
    set_style()
    g = sample.good
    d0 = sample.d0[g]
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11), sharex=True, height_ratios=[2, 1.2, 1.2])
    for row, (bin_by, lab) in enumerate(((sample.lmh[g], r"$\log M_h(z=0.4)$"),
                                         (sample.logms[g], r"$\log M_*$(total)"))):
        edges = np.quantile(bin_by, [0, 1 / 3, 2 / 3, 1])
        for b in range(3):
            sel = (bin_by >= edges[b]) & (bin_by <= edges[b + 1] + 1e-9)
            rax = axes[row + 1, b]
            if row == 0:
                ax = axes[0, b]
                ax.plot(R, np.median(np.log10(d0[sel]), axis=0), "-", color="k", lw=2.4, label="TNG (median)")
            for (pname, m0), col in zip(products.items(), COLORS):
                mm = m0[g][sel]; dd = d0[sel]
                raw = np.median((mm - dd) / dd, axis=0)
                pp = mm / mm[:, -1][:, None] * dd[:, -1][:, None]
                pin = np.median((pp - dd) / dd, axis=0)
                ls = "--" if pname == "baseline" else "-"
                if row == 0:
                    axes[0, b].plot(R, np.median(np.log10(np.clip(mm, 1, None)), axis=0), ls, color=col, lw=1.6,
                                    label=pname if b == 0 else None)
                rax.plot(R, 100 * raw, ls, color=col, lw=1.6, label=pname if (b == 0 and row == 0) else None)
                rax.plot(R, 100 * pin, ":", color=col, lw=1.2)
            rax.axhline(0, color="0.5", lw=0.8)
            for y in (-10, 10):
                rax.axhline(y, color="0.8", lw=0.7, ls=":")
            rax.set_xscale("log"); rax.set_ylim(-25, 25)
            rax.set_title(f"{lab} {edges[b]:.2f}-{edges[b + 1]:.2f} (n={sel.sum()})", fontsize=9)
            if row == 0:
                axes[0, b].set_xscale("log")
                axes[0, b].set_title(f"{lab} {edges[b]:.2f}-{edges[b + 1]:.2f} (n={sel.sum()})", fontsize=9)
    axes[0, 0].set_ylabel(r"median $\log_{10} M_*(<R)$ [M$_\odot$]")
    axes[1, 0].set_ylabel(f"median (model$-$data)/data [{_pct()}]\nby halo-mass tercile")
    axes[2, 0].set_ylabel(f"median (model$-$data)/data [{_pct()}]\nby stellar-mass tercile")
    for b in range(3):
        axes[2, b].set_xlabel("R [kpc]")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[1, 0].legend(fontsize=7.5, loc="lower right", title="solid raw, dotted amplitude-pinned", title_fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = FIGDIR / f"exp63_single_epoch_gate{name}"
    save_fig(fig, out)
    return out.with_suffix(".png")


def load_fit(tag):
    fz = np.load(OUTDIR / f"stage2_fit{tag}.npz", allow_pickle=True)
    spec2 = S2.spec_from_fit(fz)
    return spec2, np.asarray(fz["theta_best"], float), fz


def run_gate(tags, name, smoke, epoch=0):
    sample = Sample(smoke, epoch)
    s2 = M2.Spec2()
    products, rows = {}, []
    th_nested = M2.nested_theta(sample.th_inc)
    for pname, (spec2, theta) in [("baseline", (s2, th_nested)),
                                  ("stage2", (s2, load_fit("")[1]))] + \
            [(t.strip("_") or "stage2", load_fit(t)[:2]) for t in tags]:
        m0 = sample.predict(spec2, theta)
        products[pname] = m0
        rows.append((pname, gate_row(sample, m0), sample.loss_components(spec2, theta)))
        print(f"  {pname:<20} " + "  ".join(f"{n}={v:+.3f}" for n, v in zip(spec2.theta_names, theta))
              + (f"  [compact in kpc]" if spec2.compact_in_kpc else "") + f"  [{spec2.extended_family}]")
    print_table(rows, f"THE z={sample.z} GATE — median (model-data)/data [%] by halo-mass tercile, raw and amplitude-pinned"
                f" (QA sample {int(sample.good.sum())} galaxies; fit mask {len(sample.fit_rows)})")
    path = gate_figure(sample, products, name, f"exp63 single-epoch round — the z={sample.z} visual gate "
                       "(each product fitted at ONE epoch; the baseline is the incumbent on the exact engine, fitted at five epochs)")
    print(f"\n  wrote {path.relative_to(ROOT)}")
    return rows


# --------------------------------------------------------------------------- #
# the sweep                                                                    #
# --------------------------------------------------------------------------- #
def compact_pivot_shift(sample, spec2, theta):
    """Stellar-mass-weighted mean of (log M(t') - pivot) over each channel's
    deposits by z=0.4: the log_f shift that holds a channel's typical size fixed
    when its mass exponent g changes is -g * this."""
    lt, w, include, _ = E.nodes(**M2.FULL_NODES)
    dm_c, dm_e, _, _, _ = M2._deposits2(spec2, theta, sample.curves, lt)
    lm = np.array([E.log_mah(lt, hc) for hc in sample.curves]) - M2.LEVER_PIVOT_LOGM
    wc = dm_c * w[None, :] * include[0][None, :]; we = dm_e * w[None, :] * include[0][None, :]
    return float((wc * lm).sum() / wc.sum()), float((we * lm).sum() / we.sum())


def run_sweep(base_tag, smoke):
    sample = Sample(smoke)
    spec_b, th_b, _ = load_fit(base_tag)
    spec = M2.Spec2.with_levers(M2.LEVER_NAMES, extended_family=spec_b.extended_family)
    th0 = M2.with_levers_theta(th_b, spec) if not spec_b.levers else th_b
    mc, me = compact_pivot_shift(sample, spec, th0)
    print(f"  base: {base_tag or 'stage2'}; deposit-weighted <log M(t') - 13> = {mc:+.2f} (compact), {me:+.2f} (extended)")
    p0 = spec.unpack(th0)
    rows = [("base", gate_row(sample, sample.predict(spec, th0)), sample.loss_components(spec, th0))]
    grid = [
        ("w_min", (0.1, 0.2, 0.3, 0.4), None),
        ("g_c", (-1 / 3, -0.15, 0.15, 1 / 3), ("log_f_c", mc)),
        ("g_e", (-1 / 3, -0.15, 0.15, 1 / 3), ("log_f_e", me)),
        ("c_e", (0.5, 0.8, 2.0, 3.0), None),
        ("n_c", (0.5, 1.0, 1.5), None),
        ("m_half", (12.5, 13.3, 13.7), None),
        ("d_split", (0.4, 1.6, 3.0), None),
        ("log_f_c", (-1.6, -1.2, -1.0), None),
        ("log_f_e", (-1.0, -0.7), None),
    ]
    for pname, values, comp in grid:
        for v in values:
            th = th0.copy(); th[spec.index(pname)] = v
            if comp is not None:
                th[spec.index(comp[0])] = p0[comp[0]] - v * comp[1]
            rows.append((f"{pname}={v:+.2f}", gate_row(sample, sample.predict(spec, th)),
                         sample.loss_components(spec, th)))
    # the extended kernel as a Sersic deposit (same half-mass size), a family change
    spec_s = M2.Spec2.with_levers(M2.LEVER_NAMES, extended_family="sersic")
    for n_e in (0.5, 1.0, 2.0, 4.0):
        th = th0.copy(); th[spec_s.index("c_e")] = n_e
        rows.append((f"ext=sersic n={n_e:.1f}", gate_row(sample, sample.predict(spec_s, th)),
                     sample.loss_components(spec_s, th)))
    print_table(rows, f"THE LEVER SWEEP by evaluation (no refit) from {base_tag or 'stage2'} — z=0.4, halo-mass terciles")
    # the compact summary the decision reads
    print(f"\n  {'lever':<20}{'top-tercile centre (2-5 kpc) mean raw':>40}{'worst centre':>14}{'worst outskirt':>16}{'rms(h)':>9}{'loss':>10}")
    for name, g, loss in rows:
        print(f"  {name:<20}{100 * g['top_centre']:>+40.1f}{100 * g['worst_centre']:>14.1f}{100 * g['worst_outskirt']:>16.1f}"
              f"{100 * g['rms_h']:>9.2f}{loss[3]:>10.4f}")
    np.savez(OUTDIR / f"single_epoch_sweep{base_tag}{'_smoke' if smoke else ''}.npz",
             names=np.array([r[0] for r in rows]), raw_h=np.array([r[1]["raw_h"] for r in rows]),
             pin_h=np.array([r[1]["pin_h"] for r in rows]), loss=np.array([r[2] for r in rows]))


def run_qa(tags, smoke, epoch=0):
    """The standard battery at ONE epoch, one product per tag."""
    sample = Sample(smoke, epoch)
    g = sample.good
    QADIR = FIGDIR / "qa"
    QADIR.mkdir(parents=True, exist_ok=True)
    ztag = f"z{sample.z:.1f}".replace(".", "")
    for tag in tags:
        spec2, theta, _ = load_fit(tag)
        m0 = sample.predict(spec2, theta)
        name = f"exp63_{ztag}{tag or '_stage2'}{'_smoke' if smoke else ''}"
        print(f"\n{RULE}\nSTANDARD QA at z={sample.z} ONLY — {name}\n{RULE}")
        qa.evaluate(m0[g][:, None, :], sample.d0[g][:, None, :], R, [sample.z], name=name,
                    figdir=QADIR, figures=True, verbose=True,
                    bin_by=sample.lmh[g], bin_label=rf"logM$_h$(z={sample.z})",
                    bin_by_ms=sample.logms[g], ms_label=r"logM$_*$ (total)")


if __name__ == "__main__":
    args = sys.argv[1:]
    smoke = "--smoke" in args
    print(f"{RULE}\nexp63 SINGLE-EPOCH ROUND — z=0.4 only{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    if "--sweep" in args:
        i = args.index("--sweep") + 1
        base = args[i] if i < len(args) and not args[i].startswith("--") else ""
        run_sweep(base, smoke)
    epoch = int(args[args.index("--epoch") + 1]) if "--epoch" in args else 0
    if "--qa" in args:
        run_qa(args[args.index("--qa") + 1].split(","), smoke, epoch)
    if "--gate" in args:
        tags = [t for t in args[args.index("--gate") + 1].split(",") if t]
        name = args[args.index("--name") + 1] if "--name" in args else ""
        run_gate(tags, name + ("_smoke" if smoke else ""), smoke, epoch)
