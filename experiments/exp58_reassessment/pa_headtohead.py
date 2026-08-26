"""exp58 P-A — the transport-vs-deposition comparison, measured like for like.

WHY THIS EXISTS. The user's verdict on exp57 included "compared to the
deposition+transportation model the overall performance is still systematically
falling behind". That comparison has never been made like for like: every
published number for the old transport kernel (exp38's `1ch-mof`, the adopted
product exp52 decomposed) was measured with the model's amplitude PINNED to the
measured M*(<500 kpc) per galaxy per epoch — an oracle the deposition-only
models are never given, because they PREDICT their amplitude from the halo
history alone. exp54's `rebaseline.py` estimated the translation, but those
numbers predate the row-181 fix and no stored `qa.evaluate` output from the
transport era exists. This script measures it head to head, once, on exp57's
exact galaxy sample.

WHAT IS COMPARED. Six model products, all on the SAME galaxies:

  predicted-amplitude (the fair fight — what each model would deliver in use):
    * `transport`   exp38 `1ch-mof` at the adopted theta
                    (`exp40/outputs/latestart.npz::theta_z15`, reproduced
                    through exp53's kernel), its dimensionless shape scaled to
                    an OUT-OF-FOLD predicted log10 M*(<100 kpc) from halo-only
                    features (official DiffMAH 4 params + concentration excess
                    + fz2) — `rebaseline.py`'s `hybrid-mean`, refit here on the
                    sane rows only. This is the best halo-only amplitude the
                    programme has; handing it to the transport kernel is
                    GENEROUS to the transport kernel.
    * `incumbent`   exp54's adopted deposition-only model, whose amplitude is
                    its own prediction (the absolute deposition law).
    * `X3-smallcore` exp57's gate-passing candidate, ditto.

  pinned-at-100 (the shape-only decomposition — every model handed the
  measured M*(<100 kpc), so any remaining difference is radial shape alone):
    * `transport@100`, `incumbent@100`, `X3-smallcore@100`

Paired metrics (tier 3 max|rel|, aperture bias/scatter) are read off the MEAN
amplitude prediction; a stochastic draw would sqrt(2)-inflate them (exp54
rebaseline, "which column to read"). Distribution tiers are NOT quoted here
for the predicted-amplitude transport product, because a conditional mean is
under-dispersed by construction and those tiers would punish it for the wrong
reason.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp58_reassessment/pa_headtohead.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
EXP57 = ROOT / "experiments/exp57_expansion_term"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp53_deposition_only", EXP54, EXP57):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import fit as F                                          # noqa: E402
import kernel as K                                       # noqa: E402
import scoreboard as SB                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_multiepoch as s2                           # noqa: E402
from hongshao import qa                                  # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
KS = (0, 1, 2, 3, 4)
FITDIR = EXP57 / "outputs" / "stage3_fits"
OUT = HERE / "outputs" / "pa_headtohead.npz"
FIGDIR = HERE / "figures" / "qa"
RULE = "=" * 96
#: predicted-amplitude products first, then the shape-only (@100) block
PRODUCTS = ("transport", "incumbent", "X3-smallcore",
            "transport@100", "incumbent@100", "X3-smallcore@100")


def transport_shapes(recs, R_ext):
    """The old kernel's dimensionless CoG shape, (n, 5, 25) on `R_EXT`,
    NaN where the kernel cannot represent the galaxy at that epoch."""
    e = s2._W["e"]
    gal_of = {g["row"]: g for g in s2._W["gals"]}
    spec = K.Spec("moffat", q_free=True)
    theta = K.from_exp38_theta(K.adopted_theta())
    out = np.full((len(recs), len(KS), len(R_ext)), np.nan)
    for i, h in enumerate(recs):
        g = gal_of.get(h.row)
        if g is None:
            continue
        dM = K.deposit_weights(spec, theta, g)
        if dM is None:
            continue
        mah = g["mah"]
        for k in KS:
            m = K.basis(spec, theta, g["cond"], mah["t"], mah["t_obs"],
                        e.pe.AT[k], R_ext) @ (dM * (mah["snap"]
                                                    <= e.ANCHOR_SNAP[k]))
            if np.isfinite(m).all() and m[-1] > 0:
                out[i, k] = m
    return out


def predicted_amplitude(recs, sane_rows_only=True):
    """Out-of-fold predicted log10 M*(<100 kpc), (n, 5) aligned to `recs`.

    `rebaseline.py`'s `hybrid-mean` provider (official DiffMAH 4 params +
    concentration excess + fz2, 5-fold OLS), with one correction: the
    scoreboard cache already EXCLUDES the galaxies whose measured history is
    physically impossible (the row-181 class), so the regression here is fit on
    sane rows by construction. Galaxies absent from the cache get NaN and drop
    from the common set — reported, not silent.
    """
    d = SB.load()
    y = d["y"]
    idx_of = {int(r): i for i, r in enumerate(d["rows"])}
    mu_cache = np.full_like(y, np.nan)
    for k in range(5):
        feats = ([d["dm4"][:, j] for j in range(4)]
                 + [d["c_exc_k"][:, k], d["fz2"]])
        res, _ = SB.score_linear(y[:, k], feats)
        mu_cache[:, k] = y[:, k] - res
    mu = np.full((len(recs), 5), np.nan)
    missing = []
    for i, h in enumerate(recs):
        j = idx_of.get(int(h.row))
        if j is None:
            missing.append(int(h.row))
        else:
            mu[i] = mu_cache[j]
    return mu, missing


def main(smoke=False):
    print(f"{RULE}\nexp58 P-A — transport vs deposition-only, both amplitude-"
          f"PREDICTED, same galaxies\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    R = F.R_GRID
    e = s2._W["e"]
    # F.R_GRID is e.R rounded to 2-3 significant figures (same physical grid);
    # qa is run on F.R_GRID for consistency with exp57's stage10 tables.
    assert np.allclose(e.R, R, rtol=0, atol=0.01), \
        "exp38 grid and exp54 grid must be the same physical grid"
    # the grid point nearest 100 kpc (103.45): used for the pinned block, where
    # using an exact grid point makes the pin exact and identical across models
    i100 = F.I100
    assert abs(R[i100] - 100.0) < 5.0, "the pin anchor must be near 100 kpc"

    # --- the three underlying model CoGs ---------------------------------- #
    cogs = {}
    for name, lab in (("incumbent", None),
                      ("X3-smallcore", "gompertz_log-E2-S2+X3#smallcore")):
        if lab is None:
            sp, th = X.XSpec(base, ""), theta
        else:
            z = np.load(FITDIR / f"{lab}.npz", allow_pickle=True)
            sp = X.XSpec(base, str(z["law"]))
            th = np.asarray(z["theta"], float)
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=KS)
        cogs[name] = pr.predict(th)
        print(f"  {name:<14} {sp.label:<28} amplitude: its own prediction")

    shp = transport_shapes(recs, e.R_EXT)[:, :, :-1]     # drop the 500 kpc pt
    mu, missing = predicted_amplitude(recs)
    if missing:
        print(f"  transport      amplitude cache lacks rows {missing} "
              f"(REJECTED by the sane-history check) -> NaN, dropped from the "
              f"common set below")
    # the predictor's target `y` is log10 M*(<100) interpolated at EXACTLY
    # 100 kpc (scoreboard.build), so the shape is scaled at exactly 100 kpc too
    lr = np.log10(np.asarray(e.R))
    with np.errstate(invalid="ignore", divide="ignore"):
        shp100 = 10.0 ** np.apply_along_axis(
            lambda s: np.interp(np.log10(100.0), lr, s), 2,
            np.log10(np.clip(shp, 1e-300, None)))
    cogs["transport"] = shp * (10.0 ** mu[:, :, None]
                               / shp100[:, :, None])
    print(f"  {'transport':<14} exp38 1ch-mof (adopted theta)  amplitude: "
          f"out-of-fold halo-only regression at 100 kpc")

    # --- the shape-only block: every model pinned to the truth at 100 ----- #
    for name in ("transport", "incumbent", "X3-smallcore"):
        srcs = shp if name == "transport" else cogs[name]
        pin = data[:, :, i100] / srcs[:, :, i100]
        cogs[name + "@100"] = srcs * pin[:, :, None]

    # --- ONE common set --------------------------------------------------- #
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    n = int(good.sum())
    print(f"\n  common set: {n} of {len(recs)} galaxies representable by ALL "
          f"six products with finite\n  positive measured profiles at every "
          f"epoch. Every number below is on these {n}.")
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))

    # --- evaluate --------------------------------------------------------- #
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in PRODUCTS:
        figs = (name == "transport") and not smoke
        out[name] = qa.evaluate(
            cogs[name][good], data[good], R, list(ANCHOR_Z),
            name=("SMOKE_" if smoke else "exp58_") + name.replace("@", "_at"),
            figdir=(FIGDIR if figs else None), figures=figs, verbose=False,
            bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
            bin_by_ms=logms, ms_label=r"logM$_*$ (total)")

    # --- THE TABLES ------------------------------------------------------- #
    def block(names, title):
        print(f"\n{RULE}\n{title}\n{RULE}")
        print(f"\n  TIER 3 — max|rel| CoG error beyond 5 kpc, median over "
              f"galaxies (lower is better)")
        print(f"  {'model':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for nm in names:
            print(f"  {nm:<18}" + "".join(
                f"{np.nanmedian(out[nm]['mr_out'][:, j]):>10.4f}"
                for j in range(5)))
        for key in ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(<100)"):
            print(f"\n  {key} — median relative bias (closer to 0 is better)")
            print(f"  {'model':<18}"
                  + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
            for nm in names:
                t, m = out[nm]["truth"][key], out[nm]["model"][key]
                print(f"  {nm:<18}" + "".join(
                    f"{100 * np.nanmedian(qa.relerr(m[:, j], t[:, j])):>9.1f}%"
                    for j in range(5)))
        print(f"\n  kpc:M(<100) — dex scatter (the amplitude quality itself)")
        print(f"  {'model':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for nm in names:
            t, m = out[nm]["truth"]["kpc:M(<100)"], out[nm]["model"]["kpc:M(<100)"]
            print(f"  {nm:<18}" + "".join(
                f"{qa.dex_scatter(m[:, j], t[:, j]):>10.3f}" for j in range(5)))

    block(("transport", "incumbent", "X3-smallcore"),
          "AMPLITUDE PREDICTED — the fair fight, what each model delivers in "
          "use.\nDistribution tiers are not quoted here: the transport row's "
          "mean amplitude is\nunder-dispersed by construction and they would "
          "punish it for that, not for shape.")
    block(("transport@100", "incumbent@100", "X3-smallcore@100"),
          "PINNED AT THE MEASURED M*(<100 kpc) — shape only. Any difference "
          "left is the\nradial distribution, the thing the transport term was "
          "for.")

    if not smoke:
        HERE.joinpath("outputs").mkdir(exist_ok=True)
        np.savez_compressed(
            OUT, n_common=n, good=good, rows=np.array([h.row for h in recs]),
            **{f"mr_out_{nm.replace('@', '_at')}": out[nm]["mr_out"]
               for nm in PRODUCTS})
        print(f"\n  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
