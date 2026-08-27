"""exp63 Stage 0 — Result 0: the incumbent's parameters on the EXACT integral.

Three products of the SAME seven numbers (the shared incumbent theta of
exp54, `gompertz_log-E2-S2`), the same galaxies, the same battery:

  72step           `exp54/model.forward` exactly as exp61 built it (the
                   72-step sum, catalog R200c, dropped gap steps). hongshao v1.
  exact-catalog    the exact integral, catalog R200c interpolated in log t.
                   Differs from `72step` ONLY by the discretisation and the
                   dropped steps: this column is the numerical error of v1.
  exact-analytic   the exact integral with R200c from M(t) and rho_c(z)
                   (decision D1). Differs from `exact-catalog` only by the
                   halo-radius source. THE NEW BASELINE every later exp63
                   comparison is made against.

Truncation as Stage 0's D3 rule decided (read from `outputs/stage0_engine.npz`,
never assumed here).

Two gates before anything is written: `exact-catalog` must reproduce the
gate script's own product bit for bit (same code path, same nodes), and
`72step` must reproduce exp61's incumbent tier-3 median at z=0.4 on the QA
sample (0.2765) to 1e-3 — so this battery sits beside the record.

Every bias is printed on BOTH samples (the full QA population and the
mh-complete fit mask), per the standing rule from exp61.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/stage0_rebaseline.py [--smoke] [--tables-only]
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
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 92
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
QADIR = FIGDIR / "qa"
MODELS = ("72step", "exact-catalog", "exact-analytic")
PRETTY = {"72step": "hongshao v1: the 72-step sum (exp54 forward)",
          "exact-catalog": "exact integral, catalog R200c (v1 minus its numerics)",
          "exact-analytic": "exact integral, analytic R200c (the exp63 baseline, D1)"}
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(<100)", "kpc:M(50-100)")
ANCHOR_Z = E.ANCHOR_Z
#: exp61 Stage 3's incumbent tier-3 median beyond 5 kpc at z=0.4, QA sample
EXP61_MR_OUT_Z04 = 0.2765
I5 = 4                                                   # R_GRID[4] = 6.38 kpc


def inner_slope(cogs, R):
    """d log M / d log R between the first grid radius (2 kpc) and R[4] (6.4 kpc)."""
    return np.log10(cogs[..., I5] / cogs[..., 0]) / np.log10(R[I5] / R[0])


def build_products(recs, data, mask, spec, theta, truncate):
    pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    curves = E.build_curves(recs)
    return {"72step": pr.predict(theta),
            "exact-catalog": E.predict(spec, theta, curves, F.R_GRID, mode="catalog",
                                       truncate=truncate),
            "exact-analytic": E.predict(spec, theta, curves, F.R_GRID, mode="analytic",
                                        truncate=truncate)}


def compare_figure(cogs, data, lmh, good, name, models=MODELS, styles=None,
                   pretty=None, title=None):
    """Median (model - data)/data against radius, per epoch, halo-mass
    terciles, the products in `models`: raw (top) and amplitude-pinned at
    100 kpc (bottom) — the exp61 `qa_bins` residual panel, several products at
    once. Reused by Stage 2 with its own product names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex, _pct
    set_style()
    R = F.R_GRID
    lm = lmh[good][:, 0]
    edges = np.percentile(lm, [0, 100 / 3, 200 / 3, 100])
    ls = styles or {"72step": "-", "exact-catalog": "--", "exact-analytic": ":"}
    pretty = pretty or PRETTY
    cols = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    for c in range(3):
        sel = (lm >= edges[c]) & (lm <= edges[c + 1])
        for m in models:
            mm, dd = cogs[m][good][sel], data[good][sel]
            raw = 100 * np.median((mm - dd) / dd, axis=0)                    # (5, R)
            pin = mm / mm[:, :, F.I100][:, :, None] * dd[:, :, F.I100][:, :, None]
            pinned = 100 * np.median((pin - dd) / dd, axis=0)
            for j, z in enumerate(ANCHOR_Z):
                axes[0, c].plot(R, raw[j], ls[m], color=cols[j], lw=1.6,
                                label=(f"z={z}" if m == models[0] else None))
                axes[1, c].plot(R, pinned[j], ls[m], color=cols[j], lw=1.6)
        for r in range(2):
            axes[r, c].axhline(0, color="k", lw=0.8)
            axes[r, c].set_xscale("log")
            axes[r, c].set_ylim(-30, 30)
        axes[0, c].set_title(f"logMh(z=0.4) {edges[c]:.2f}-{edges[c + 1]:.2f} (n={int(sel.sum())})")
        axes[1, c].set_xlabel("R [kpc]")
    axes[0, 0].set_ylabel(f"median (model$-$data)/data [{_pct()}], raw")
    axes[1, 0].set_ylabel(f"median (model$-$data)/data [{_pct()}], amplitude-pinned at 100 kpc")
    h, l = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], color="k", ls=ls[m]) for m in models]
    l += [pretty[m].split(":")[0] if m == "72step" else m for m in models]
    axes[0, 0].legend(h, l, fontsize=7, loc="lower right", ncol=2)
    fig.suptitle(_tex(title or "exp63 Stage 0 — the SAME seven parameters: the 72-step "
                      "sum, the exact integral (catalog R200c), the exact integral "
                      "(analytic R200c)"), fontsize=11)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    save_fig(fig, FIGDIR / name)
    return FIGDIR / f"{name}.png"


def main(smoke=False, tables_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp63 STAGE 0 — Result 0: the incumbent re-baselined on the exact "
          f"integral{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    g = np.load(OUTDIR / f"stage0_engine{tag}.npz", allow_pickle=True)
    truncate = bool(g["d3_keep_truncation"])
    print(f"  D3 from the gate script: {'KEEP' if truncate else 'DROP'} the 3 R200c truncation "
          f"(median mass beyond it {100 * np.median(g['frac_beyond_3r200']):.2f}"
          f"{'%' if True else ''} untruncated)")
    recs, data, mask, lmh, spec, theta, _ = S0.build(smoke)
    R = F.R_GRID
    cogs = build_products(recs, data, mask, spec, theta, truncate)

    # gate 1: same code path as the gate script -> bit for bit
    curves = E.build_curves(recs, verbose=False)
    again = E.predict(spec, theta, curves, R, mode="catalog", truncate=truncate)
    if not np.array_equal(again, cogs["exact-catalog"]):
        raise SystemExit("REFUSING TO WRITE: predict is not deterministic")

    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    msk = np.asarray(mask, bool)[good]
    print(f"  {int(good.sum())} of {len(recs)} galaxies have finite positive measured "
          f"profiles at every epoch and are representable by all three products;\n"
          f"  the mh-complete fit mask keeps "
          + "/".join(f"{int(msk[:, j].sum())}" for j in range(5)) + " of them at z=0.4/0.7/1.0/1.5/2.0")
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))

    # gate 2: the 72step product sits on exp61's record
    mr_out_72 = qa.profile_maxrel(cogs["72step"][good], data[good], R, rmin=qa.RMIN_KPC)
    v = float(np.nanmedian(mr_out_72[:, 0]))
    print(f"  gate: 72step tier-3 median beyond 5 kpc at z=0.4 = {v:.4f} (exp61: {EXP61_MR_OUT_Z04})")
    if not smoke and abs(v - EXP61_MR_OUT_Z04) > 1e-3:
        raise SystemExit("REFUSING TO WRITE: the 72-step product does not reproduce exp61's record")

    QADIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for m in MODELS:
        print(f"\n{RULE}\nSTANDARD QA — {m}   ({PRETTY[m]})\n{RULE}")
        out[m] = qa.evaluate(cogs[m][good], data[good], R, list(ANCHOR_Z),
                             name=f"exp63_{m}{tag}", figdir=(None if tables_only else QADIR),
                             figures=not tables_only, verbose=True,
                             bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
                             bin_by_ms=logms, ms_label=r"logM$_*$ (total)")

    print(f"\n{RULE}\nSIDE BY SIDE — the same seven numbers, three integrations, "
          f"{int(good.sum())} galaxies\n{RULE}")
    print("  Read the columns left to right: v1 -> v1 without its discretisation error -> "
          "the exp63 baseline (analytic R200c).")
    for key in TABLE_KEYS:
        print(f"\n  {key} — median relative bias [%], closer to 0 is better")
        print(f"  {'product':<18}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for m in MODELS:
            t, mm = out[m]["truth"][key], out[m]["model"][key]
            print(f"  {m:<18}{'all ' + str(int(good.sum())):<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<18}{'fit mask only':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[msk[:, j], j], t[msk[:, j], j])):>9.1f}%"
                for j in range(5)))

    print("\n  M*(<100 kpc) scatter [dex] — LOWER is better")
    print(f"  {'product':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for m in MODELS:
        print(f"  {m:<18}" + "".join(
            f"{qa.dex_scatter(out[m]['model']['kpc:M(<100)'][:, j], out[m]['truth']['kpc:M(<100)'][:, j]):>10.4f}"
            for j in range(5)))
    print("\n  tier 3, profile max|rel| beyond 5 kpc (median) — LOWER is better")
    print(f"  {'product':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for m in MODELS:
        print(f"  {m:<18}" + "".join(f"{np.nanmedian(out[m]['mr_out'][:, j]):>10.4f}" for j in range(5)))

    sd = inner_slope(data[good][:, 0], R)
    print(f"\n  inner CoG slope dlogM/dlogR over 2-6.4 kpc at z=0.4 — DATA median {np.median(sd):.3f}, "
          f"16-84% {np.percentile(sd, 16):.2f}-{np.percentile(sd, 84):.2f}")
    slopes = {}
    for m in MODELS:
        slopes[m] = inner_slope(cogs[m][good][:, 0], R)
        print(f"    {m:<18} median {np.median(slopes[m]):.3f}, 16-84% "
              f"{np.percentile(slopes[m], 16):.2f}-{np.percentile(slopes[m], 84):.2f}")

    fig = compare_figure(cogs, data, lmh, good, f"exp63_stage0_compare{tag}")
    print(f"\n  wrote {fig.relative_to(ROOT)}")
    if not tables_only:
        print(f"  battery figures -> {QADIR.relative_to(ROOT)}")
    np.savez(OUTDIR / f"stage0_rebaseline{tag}.npz", names=np.array(MODELS),
             n_galaxies=int(good.sum()), truncate=truncate,
             inner_slope_data=sd, **{f"inner_slope_{m}": slopes[m] for m in MODELS},
             **{f"mr_out_{m}": out[m]["mr_out"] for m in MODELS},
             **{f"bias_{m}_{k}_{s}": np.array([
                 np.nanmedian(qa.relerr(out[m]["model"][k][sel[:, j], j], out[m]["truth"][k][sel[:, j], j]))
                 for j in range(5)])
                for m in MODELS for k in TABLE_KEYS
                for s, sel in (("all", np.ones_like(msk)), ("fitmask", msk))})
    print(f"  saved {(OUTDIR / f'stage0_rebaseline{tag}.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main("--smoke" in sys.argv[1:], "--tables-only" in sys.argv[1:])
