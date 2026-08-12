"""exp46 stage 1e figure — do the compact z=2 progenitors have special haloes?

Three panels (user request, 2026-08-12):
  1. the mass-size plane at z=2, compact decile highlighted, against a
     STELLAR-MASS-MATCHED control of the same size;
  2. the DiffMAH halo histories of both groups over all haloes;
  3. the same for the REAL simulation MAH.

The mass-matched control is the point of the figure. Compact and
non-compact galaxies could differ in their haloes simply by differing in
mass, so the comparison group is built by 1:1 nearest-neighbour matching
on log M*(<148 kpc) at z=2, without replacement, drawn from the
non-compact galaxies. Any surviving difference in panels 2-3 is then a
difference at fixed stellar mass.

Run: PYTHONPATH=. uv run python \
experiments/exp46_highz_ridge/compact_figure.py {demo|run}
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
E38 = ROOT / "experiments/exp38_deposit_rethink"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
K_HIGHZ = 4
COMPACT_Q = 0.10

C_ALL = "#9a9a9a"
C_COMPACT = "#D55E00"      # Okabe-Ito vermillion
C_MATCH = "#0072B2"        # Okabe-Ito blue


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mass_matched(logms, compact, seed=0):
    """1:1 nearest-neighbour match on ``logms``, without replacement.

    Greedy from the rarest end: compact galaxies are matched in order of
    how isolated they are in mass, so the hardest ones get first pick of
    the pool and the match quality is not spent on easy cases.
    """
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(~compact & np.isfinite(logms))
    targets = np.flatnonzero(compact & np.isfinite(logms))
    # rarest-first: distance to the 20th nearest pool member
    d = np.abs(logms[targets][:, None] - logms[pool][None, :])
    k = min(20, d.shape[1] - 1)
    order = np.argsort(-np.partition(d, k, axis=1)[:, k])
    taken, out = np.zeros(len(pool), bool), np.full(len(targets), -1)
    for t in order:
        dd = np.where(taken, np.inf, d[t])
        j = int(np.argmin(dd + rng.normal(0, 1e-9, len(dd))))
        taken[j] = True
        out[t] = pool[j]
    matched = np.zeros(len(logms), bool)
    matched[out[out >= 0]] = True
    return matched


def _mah_stack(gals, rows_order, t_ref):
    """(n, len(t_ref)) log10 M_h interpolated onto a common time grid."""
    out = np.full((len(rows_order), len(t_ref)), np.nan)
    by_row = {g["row"]: g for g in gals}
    for i, r in enumerate(rows_order):
        g = by_row.get(int(r))
        if g is None:
            continue
        m = g["mah"]
        out[i] = np.interp(t_ref, m["t_full"], m["logMh_full"],
                           left=np.nan, right=m["logMh_full"][-1])
    return out


def _band(ax, t, stack, color, label, lw=2.4):
    ok = np.isfinite(stack).sum(0) > 5
    med = np.nanmedian(stack[:, ok], axis=0)
    lo = np.nanpercentile(stack[:, ok], 16, axis=0)
    hi = np.nanpercentile(stack[:, ok], 84, axis=0)
    ax.fill_between(t[ok], lo, hi, color=color, alpha=0.22, lw=0)
    ax.plot(t[ok], med, color=color, lw=lw, label=label, zorder=5)
    return med, ok


def cmd_run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao import qa, plotting
    from hongshao.tng_data import COG_RAD_KPC as R
    from compact_anatomy import compactness, cliffs_delta

    plotting.set_style()
    p = np.load(POP_NPZ)
    d = p["data"]
    rows = np.arange(len(d))
    r50, _ = compactness(d, R)
    logms2 = np.log10(np.clip(d[:, K_HIGHZ, -1], 1.0, None))

    cut = np.quantile(r50, COMPACT_Q)
    compact = r50 <= cut
    matched = mass_matched(logms2, compact)
    print(f"  compact n={compact.sum()}  matched n={matched.sum()}")
    print(f"  log M* at z=2: compact {np.median(logms2[compact]):.3f}  "
          f"matched {np.median(logms2[matched]):.3f}  "
          f"rest {np.median(logms2[~compact]):.3f}")
    print(f"  match quality: Cliff's delta(compact, matched) on log M* = "
          f"{cliffs_delta(logms2[compact], logms2[matched]):+.3f} "
          f"(0 = perfectly matched)")
    print(f"  R50 [kpc]: compact {np.median(r50[compact]):.2f}  "
          f"matched {np.median(r50[matched]):.2f}  "
          f"(delta {cliffs_delta(r50[compact], r50[matched]):+.2f})")

    # --- the two MAH configurations ---------------------------------------
    s2 = _load_by_path("exp38_stage2", E38 / "stage2_multiepoch.py")
    s2._w_init(None)
    t_ref = np.linspace(0.5, float(s2._W["gals"][0]["mah"]["t_obs"]), 120)
    mah_dm = _mah_stack(s2._W["gals"], rows, t_ref)

    e35 = _load_by_path("exp35_run", ROOT / "experiments/exp35_total_norm/run.py")
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/"
                             "outputs/um_slope_diffmah.npz")["mh_scale"])
    e35._init("real", mh_scale, None)
    mah_real = _mah_stack(e35._G["gals"], rows, t_ref)
    print(f"  MAH stacks: diffmah {np.isfinite(mah_dm).all(1).sum()} rows, "
          f"real {np.isfinite(mah_real).all(1).sum()} rows")

    # --- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    # manual axes surgery below (residual strips), so lay out FIRST and do
    # not call tight_layout afterwards — it would fight the set_position.
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.115, top=0.88,
                        wspace=0.28)
    groups = [("compact decile", compact, C_COMPACT),
              (r"$M_\star$-matched control", matched, C_MATCH)]

    ax = axes[0]
    ax.scatter(logms2[~compact & ~matched], np.log10(r50[~compact & ~matched]),
               s=5, c=C_ALL, alpha=0.30, lw=0, label="all others")
    for lab, m, c in groups:
        ax.scatter(logms2[m], np.log10(r50[m]), s=13, c=c, alpha=0.85, lw=0,
                   label=f"{lab} (n={m.sum()})")
    ax.axhline(np.log10(2.0), color="k", ls=":", lw=1.2)
    ax.text(0.02, np.log10(2.0) + 0.03, "2 kpc — CoG grid start; "
            "smaller $R_{50}$ is extrapolated",
            transform=ax.get_yaxis_transform(),
            fontsize=7.5, color="k", va="bottom")
    ax.text(0.03, 0.03, "the compact decile lies ENTIRELY below the grid\n"
            "start; an on-grid definition ($M_\\star(<10)/M_\\star(<148)$,\n"
            "only 82/240 in common) reproduces panels 2-3",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=3))
    # NOTE: these labels are already math-mode; do NOT pass them through
    # qa._tex, which escapes "<" into "$<$" and closes the math group
    # (it rendered "<148" as "j148").
    ax.set_xlabel(r"$\log_{10} M_\star(<148\ \mathrm{kpc})$ "
                  r"at $z=2$  [$M_\odot$]")
    ax.set_ylabel(r"$\log_{10} R_{50}$ at $z=2$  [kpc]")
    ax.set_title("1. the mass-size plane at $z=2$", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # Panels 2-3 carry a residual strip, because the whole signal lives in
    # the first ~2 Gyr and is ~0.3 dex against a 3-dex vertical range —
    # invisible on the absolute curves the request asks for.
    for slot, stack, name in ((1, mah_dm, "2. DiffMAH (model input)"),
                              (2, mah_real, "3. real MAH (simulation)")):
        base = axes[slot]
        box = base.get_position()
        base.set_position([box.x0, box.y0 + 0.30 * box.height,
                           box.width, 0.70 * box.height])
        rax = fig.add_axes([box.x0, box.y0, box.width, 0.24 * box.height],
                           sharex=base)

        for i in np.flatnonzero(~compact & ~matched)[::4]:
            if np.isfinite(stack[i]).any():
                base.plot(t_ref, stack[i], color=C_ALL, lw=0.35, alpha=0.10,
                          zorder=1)
        # only draw where most of the group actually has a history: real
        # trees begin at different times, so early bins are thin.
        cov = np.isfinite(stack).mean(0)
        good = cov >= 0.80
        for lab, m, c in groups:
            _band(base, t_ref[good], stack[m][:, good], c, lab)
        fin = np.nanmedian(stack[:, -1][compact]), np.nanmedian(
            stack[:, -1][matched])
        gc = np.nanmedian(stack[compact][:, good], 0) - fin[0]
        gm = np.nanmedian(stack[matched][:, good], 0) - fin[1]
        rax.plot(t_ref[good], gc - gm, color=C_COMPACT, lw=2.0)
        rax.axhline(0.0, color=C_MATCH, lw=1.4, ls="--")
        rax.set_ylim(-0.06, 0.42)
        rax.set_xlabel("cosmic time [Gyr]")
        rax.set_ylabel("head\nstart [dex]", fontsize=7.5)
        rax.tick_params(labelsize=7.5)
        base.tick_params(labelbottom=False)
        base.set_ylabel(r"$\log_{10} M_{\rm h}(t)$  [$M_\odot$]")
        base.set_title(name, fontsize=10)
        base.set_ylim(11.0, 13.9)
        base.set_xlim(t_ref[good][0], t_ref[-1])

        j1, j2 = (int(np.argmin(abs(t_ref - v))) for v in (1.0, 2.0))
        st = []
        for tt, j in ((1.0, j1), (2.0, j2)):
            dv = cliffs_delta(stack[compact, j] - stack[compact, -1],
                              stack[matched, j] - stack[matched, -1])
            st.append(f"$t={tt:.0f}$ Gyr: $\\Delta$ = {dv:+.2f}")
        base.text(0.97, 0.06, "compact vs matched, Cliff's $\\Delta$\n"
                  + "\n".join(st), transform=base.transAxes, ha="right",
                  va="bottom", fontsize=7.5,
                  bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=3))
        base.legend(fontsize=8, loc="upper left", framealpha=0.9)
        print(f"  {name}: " + "  ".join(s.replace("$", "") for s in st)
              + f"  (band drawn where >=80% coverage: t >= "
                f"{t_ref[good][0]:.1f} Gyr)")

    fig.suptitle("exp46 — the compact $z=2$ progenitors against a "
                 "stellar-mass-matched control", fontsize=11)
    FIGDIR.mkdir(exist_ok=True)
    plotting.save_fig(fig, str(FIGDIR / "exp46_compact_mah"))
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "compact_matched.npz", compact=compact,
             matched=matched, r50_z2=r50, logms_z2=logms2, t_ref=t_ref,
             mah_diffmah=mah_dm, mah_real=mah_real)
    print(f"  wrote {FIGDIR / 'exp46_compact_mah'}.png/.pdf")


def demo():
    rng = np.random.default_rng(0)
    m = rng.normal(11.0, 0.3, size=1500)

    # (a) a realistic, OVERLAPPING mass offset — the regime this figure is
    #     in — must be substantially closed.
    pick = rng.random(1500) < 1.0 / (1.0 + np.exp((m - 10.85) / 0.12))
    comp = np.zeros(1500, bool)
    comp[np.flatnonzero(pick)[:150]] = True
    mt = mass_matched(m, comp)
    assert mt.sum() == comp.sum() == 150, (mt.sum(), comp.sum())
    assert not (mt & comp).any(), "control must not overlap the target"
    raw = abs(np.median(m[comp]) - np.median(m[~comp]))
    got = abs(np.median(m[comp]) - np.median(m[mt]))
    assert got < 0.1 * raw, (got, raw)

    # (b) the honest limit: a target sitting in the EXTREME tail cannot be
    #     matched by anything, because the pool has nothing that light. The
    #     matcher must not pretend otherwise — it closes the gap only
    #     partially, which is why cmd_run prints the achieved match quality
    #     instead of assuming success.
    tail = np.zeros(1500, bool)
    tail[np.argsort(m)[:150]] = True
    gap_tail = abs(np.median(m[tail]) - np.median(m[mass_matched(m, tail)]))
    assert gap_tail > 0.1, gap_tail

    # (c) no mass difference: still exact size, still disjoint.
    comp2 = rng.random(1500) < 0.1
    mt2 = mass_matched(m, comp2)
    assert mt2.sum() == comp2.sum() and not (mt2 & comp2).any()
    print(f"demo OK — matcher closes an overlapping gap {raw:.3f} -> "
          f"{got:.3f} dex; leaves an unmatchable extreme tail at "
          f"{gap_tail:.3f} dex rather than faking it")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run()
    else:
        raise SystemExit(__doc__)
