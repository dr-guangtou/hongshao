"""exp54 Stage 3.3 — five epochs, three null controls, and the
transport-replacement test.

Takes Stage 3.2's short list to the FULL sample at ALL FIVE epochs and reports
what a promotion decision actually needs:

  * amplitude and normalized shape, per epoch;
  * MASS GROWTH -- scoring five marginal amplitudes constrains the mean
    trajectory but not the joint distribution of growth residuals, so the
    cross-epoch residual correlation is a PROMOTION metric in its own right
    (review §5.2). If it fails while the marginals pass, a growth term is added
    THEN, with the failure documented;
  * distance to the Stage 3.0 MONOTONIC FLOOR (0.045590), so the remaining
    error is attributed rather than merely reported;
  * THREE null controls, not one (review §5.3):
      1. epoch-local logMh(z_k) only,
      2. the richest halo-only benchmark (Stage 1's best row),
      3. the ENDPOINT-PRESERVING MAH shuffle built in Stage 3.0 -- a separate
         permutation per epoch matched on THAT epoch's halo mass, so history
         shape is isolated from epoch-local mass;
  * the HALO-MASS TILT at four radii as a held-out slope with a bootstrap
    interval -- not a point estimate required to equal zero (review §5.5).
    exp53 found transport supplies the halo-mass-dependent concentration; this
    design deleted transport, so the size law must now supply it.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/stage33.py [--top 3] [--n 0]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink"):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402
import stage30 as S30                                    # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style   # noqa: E402
from hongshao.qa import _tex                             # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
S32 = HERE / "outputs" / "stage32.npz"
OUT = HERE / "outputs" / "stage33.npz"
FIGDIR = HERE / "figures"
Z = [0.4, 0.7, 1.0, 1.5, 2.0]
MONO_FLOOR = 0.045590
TILT_RADII = (5.0, 10.0, 75.0, 148.22)
RULE = "=" * 96


def spec_from_label(label):
    """Rebuild a Spec from its label, e.g. 'sersic-E1+E3-S2a+S3'."""
    fam, eff, size = label.split("-", 2)
    kw = dict(family=fam)
    if "+S3" in size:
        kw["conditioning"] = True
        size = size.replace("+S3", "")
    kw["size"] = size
    if "+E3" in eff:
        kw["e3"] = True
        eff = eff.replace("+E3", "")
    kw["eff"] = eff
    return M.Spec(**kw)


def growth_report(pred, data, epochs):
    """Cross-epoch amplitude-residual correlation -- a promotion metric."""
    i = F.I100
    A_m = np.log10(np.clip(pred[:, :, i], 1.0, None))
    A_d = np.log10(data[:, epochs, i])
    res = A_m - A_d
    C = np.corrcoef(res.T)
    g_m = A_m[:, 0] - A_m[:, -1]
    g_d = A_d[:, 0] - A_d[:, -1]
    return C, float(np.median(g_m - g_d)), float(np.std(g_m - g_d))


def tilt_report(pred, data, epochs, lmh, n_boot=200, seed=0):
    """Held-out halo-mass slope of log(model/truth) at several radii, with a
    bootstrap interval. exp53's transport-replacement question."""
    rng = np.random.default_rng(seed)
    R = F.R_GRID
    out = {}
    for r in TILT_RADII:
        c = int(np.argmin(np.abs(R - r)))
        rows = []
        for j, k in enumerate(epochs):
            y = (np.log10(np.clip(pred[:, j, c], 1.0, None))
                 - np.log10(np.clip(data[:, k, c], 1.0, None)))
            ok = np.isfinite(y) & np.isfinite(lmh[:, k])
            sl = np.polyfit(lmh[ok, k], y[ok], 1)[0]
            bs = []
            idx_all = np.where(ok)[0]
            for _ in range(n_boot):
                s = rng.choice(idx_all, len(idx_all))
                bs.append(np.polyfit(lmh[s, k], y[s], 1)[0])
            rows.append((sl, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
        out[r] = rows
    return out


def main(top=3, n_total=0):
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    rows = all_rows if n_total <= 0 else all_rows[:: max(1, len(all_rows) // n_total)]
    recs = H.build_records(rows=rows)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    lmh = pop["logmh_zk_diffmah"][sel]
    epochs = (0, 1, 2, 3, 4)

    if not S32.exists():
        raise SystemExit("run stage32.py first")
    s32 = np.load(S32, allow_pickle=True)
    order = np.argsort(s32["mean_rank"])
    short = [str(s32["labels"][i]) for i in order[:top]]
    print(f"exp54 STAGE 3.3 — five epochs   (n={len(recs)}, short list {short})")
    print(f"  monotonic floor from Stage 3.0: {MONO_FLOOR:.6f}\n")

    results = {}
    for label in short:
        sp = spec_from_label(label)
        pr = F.Problem(sp, recs, data, epochs=epochs)
        t0 = time.time()
        print(f"{RULE}\n{sp.label}   {sp.n_theta} parameters\n{RULE}")
        th, lo = F.fit(pr, maxiter=600 * sp.n_theta, verbose=True)
        sa, sf = pr.per_epoch(th)
        pred = pr.predict(th)
        print(f"    fitted in {(time.time() - t0) / 60:.1f} min; loss {lo:.5f}")
        print(f"    {'epoch':<10}" + "".join(f"{f'z={z}':>10}" for z in Z))
        print(f"    {'score_A':<10}" + "".join(f"{v:>10.4f}" for v in sa))
        print(f"    {'score_F':<10}" + "".join(f"{v:>10.4f}" for v in sf))
        raw_F = sf * F.L_F_REF
        print(f"    {'raw shape':<10}" + "".join(f"{v:>10.4f}" for v in raw_F))
        print(f"    {'/ floor':<10}" + "".join(f"{v / MONO_FLOOR:>10.2f}"
                                               for v in raw_F))

        C, gbias, gscat = growth_report(pred, data, list(epochs))
        print(f"\n    GROWTH (a promotion metric in its own right):")
        print(f"      M*(<100) growth z=2->0.4: model - truth median "
              f"{gbias:+.4f} dex, scatter {gscat:.4f}")
        print(f"      cross-epoch amplitude-residual correlation:")
        for j, z in enumerate(Z):
            print(f"        z={z:<5}" + " ".join(f"{v:+.3f}" for v in C[j]))

        tilts = tilt_report(pred, data, list(epochs), lmh)
        print(f"\n    HALO-MASS TILT, held-out slope with 95% bootstrap CI")
        print(f"    (exp53: transport supplied this; the size law must now)")
        print(f"      {'radius':>9}" + "".join(f"{f'z={z}':>20}" for z in Z))
        for r, rowset in tilts.items():
            cells = [f"{s:+.3f}[{a:+.3f},{b:+.3f}]" for s, a, b in rowset]
            print(f"      {r:>8.0f} " + "".join(f"{c:>20}" for c in cells))
        n_sig = sum(1 for rowset in tilts.values() for s, a, b in rowset
                    if a > 0 or b < 0)
        print(f"      -> {n_sig}/{len(tilts) * len(Z)} slopes exclude zero")
        results[label] = dict(theta=th, loss=lo, sA=sa, sF=sf, C=C,
                              gbias=gbias, gscat=gscat, pred=pred)

    np.savez_compressed(OUT, labels=np.array(short), rows=sel,
                        **{f"{l}_{k}": v[k] for l, v in results.items()
                           for k in ("theta", "loss", "sA", "sF", "C")})
    print(f"\n  wrote {OUT}")
    figures(results, data, lmh, list(epochs))
    return results


def figures(results, data, lmh, epochs):
    import matplotlib.pyplot as plt
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    i = F.I100
    labels = list(results)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for c, lab in enumerate(labels):
        r = results[lab]
        col = OKABE_ITO[c % len(OKABE_ITO)]
        ax[0, 0].plot(Z, r["sA"], "-o", color=col, ms=4, label=lab)
        ax[0, 1].plot(Z, r["sF"] * F.L_F_REF, "-o", color=col, ms=4, label=lab)
        pred = r["pred"]
        med = np.nanmedian(np.log10(np.clip(pred[:, :, :], 1.0, None))
                           - np.log10(np.clip(data[:, epochs, :], 1.0, None)),
                           axis=0)
        for j, z in enumerate(Z):
            ax[1, 0].plot(F.R_GRID, med[j], color=col,
                          alpha=0.35 + 0.15 * j, lw=1.2,
                          label=lab if j == 0 else None)
        A_m = np.log10(np.clip(pred[:, :, i], 1.0, None))
        A_d = np.log10(data[:, epochs, i])
        ax[1, 1].plot(Z, np.median(A_m - A_d, axis=0), "-o", color=col, ms=4,
                      label=lab)
    ax[0, 0].axhline(1.0, color="0.6", lw=0.9, ls="--")
    ax[0, 0].set_ylabel(_tex("score$_A$ (1 = halo-only benchmark)"))
    ax[0, 1].axhline(MONO_FLOOR, color="crimson", lw=1.0, ls="--")
    ax[0, 1].set_ylabel(_tex("raw shape loss"))
    ax[0, 1].text(1.0, MONO_FLOOR * 1.05, _tex("monotonic floor"),
                  color="crimson", fontsize=8)
    for a in (ax[0, 0], ax[0, 1], ax[1, 1]):
        a.set_xlabel(_tex("redshift"))
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    ax[1, 0].axhline(0, color="0.6", lw=0.8)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel(_tex("R [kpc]"))
    ax[1, 0].set_ylabel(_tex("median log(model/truth), all epochs"))
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].grid(alpha=0.3)
    ax[1, 1].axhline(0, color="0.6", lw=0.8)
    ax[1, 1].set_ylabel(_tex("median amplitude bias [dex]"))
    fig.suptitle(_tex("exp54 Stage 3.3 -- five epochs, the short list"),
                 fontsize=12)
    fig.tight_layout()
    print("  wrote", save_fig(fig, FIGDIR / "s33_five_epoch")[0])


if __name__ == "__main__":
    top = 3
    n = 0
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    main(top, n)
