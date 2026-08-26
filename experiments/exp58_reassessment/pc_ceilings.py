"""exp58 P-C — how much of the amplitude error could each kind of enrichment
remove? Measured as post-hoc DEBIAS CEILINGS on the incumbent. Nothing is
fitted to the physical model; every product below is the incumbent's own CoGs
times a per-galaxy-epoch scalar, so the SHAPE is identical throughout and any
change is amplitude bookkeeping alone.

The ladder, and what each rung bounds (a bound must match the freedom of the
candidate it bounds — exp54 Stage 3.5's lesson):

  `incumbent`   the reference.
  `shared-z`    the per-epoch MEDIAN bias removed, one number per epoch,
                computed on the FIT sample (the mh-complete + sane mask, which
                is what a refit would see). An amplitude enrichment that
                depends on TIME ONLY — any E4/E6/E7-style term — has exactly
                this freedom per epoch, so this is its exact ceiling for
                median-bias statistics and a close bound for tier 3.
  `cond-oof`    an OUT-OF-FOLD per-epoch linear debias on halo-local features
                the model is allowed to see (log10 Mh at the epoch, formation
                time t_half/t at the epoch, concentration excess at the epoch,
                fz2), trained on the fit sample, predicted for everyone. This
                is the honest reachable ceiling for a CONDITIONED efficiency
                enrichment of the same information content.
  `cond-shuf`   the same machinery with the feature ROWS SHUFFLED across
                galaxies (within epoch) before training — the permuted control
                (a ceiling needs one). Any apparent gain here is machinery,
                not information.
  `oracle-amp`  the measured amplitude copied in per galaxy-epoch (the pin at
                103.45 kpc). Everything left is radial shape. NOT reachable by
                any amplitude law; the floor the ladder converges towards.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp58_reassessment/pc_ceilings.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
EXP57 = ROOT / "experiments/exp57_expansion_term"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, EXP57):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import scoreboard as SB                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao import qa                                  # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
NFOLD = 5
SEED = 0
RULE = "=" * 96
LADDER = ("incumbent", "shared-z", "cond-oof", "cond-shuf", "oracle-amp")


def oof_debias(a_res, feats, train, rng=None):
    """Out-of-fold linear prediction of the residual `a_res` (n,) from
    `feats` (n, p), trained on `train` rows only, predicted for ALL rows.
    NaN features are zeroed after standardisation (absent = population mean).
    `rng` shuffles the feature ROWS among galaxies first (the control)."""
    Fm = np.asarray(feats, float)
    if rng is not None:
        Fm = Fm[rng.permutation(len(Fm))]
    mu = np.nanmean(Fm[train], axis=0)
    sd = np.nanstd(Fm[train], axis=0)
    Z = np.nan_to_num((Fm - mu) / np.where(sd > 0, sd, 1.0))
    D = np.column_stack([np.ones(len(Z)), Z])
    fold = np.arange(len(Z)) % NFOLD
    pred = np.zeros(len(Z))
    for f in range(NFOLD):
        tr = train & (fold != f)
        b, *_ = np.linalg.lstsq(D[tr], a_res[tr], rcond=None)
        pred[fold == f] = D[fold == f] @ b
    return pred


def main(smoke=False):
    print(f"{RULE}\nexp58 P-C — the amplitude-enrichment ceiling ladder "
          f"(no physical model refitted)\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    R = F.R_GRID
    i100 = F.I100
    n = len(recs)
    pred = slow.predict(theta)
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, i100] > 0)
    use = np.asarray(mask, bool) & finite & (data[:, :, i100] > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        a_res = (np.log10(np.clip(pred[:, :, i100], 1.0, None))
                 - np.log10(np.clip(data[:, :, i100], 1.0, None)))
    a_res = np.where(finite & (data[:, :, i100] > 0), a_res, np.nan)

    # halo-local features per epoch
    d = SB.load()
    idx_of = {int(r): i for i, r in enumerate(d["rows"])}
    sel = np.array([idx_of.get(int(h.row), -1) for h in recs])
    okd = sel >= 0
    c_exc = np.full((n, 5), np.nan)
    fz2 = np.full(n, np.nan)
    c_exc[okd] = d["c_exc_k"][sel[okd]]
    fz2[okd] = d["fz2"][sel[okd]]
    fform = np.full((n, 5), np.nan)
    for i, h in enumerate(recs):
        for k in range(5):
            w = np.where(h.epoch_mask[k])[0]
            if len(w):
                fform[i, k] = h.f_form[w[-1]]

    # the debias ladder, delta = log10 multiplier per galaxy-epoch
    rng = np.random.default_rng(SEED)
    delta = {nm: np.zeros((n, 5)) for nm in LADDER}
    for k in range(5):
        tr = use[:, k] & np.isfinite(a_res[:, k])
        med = np.nanmedian(a_res[tr, k])
        delta["shared-z"][:, k] = -med
        feats = np.column_stack([lmh[:, k], fform[:, k], c_exc[:, k], fz2])
        a_fill = np.nan_to_num(a_res[:, k])
        delta["cond-oof"][:, k] = -oof_debias(a_fill, feats, tr)
        delta["cond-shuf"][:, k] = -oof_debias(a_fill, feats, tr, rng=rng)
        delta["oracle-amp"][:, k] = np.where(np.isfinite(a_res[:, k]),
                                             -a_res[:, k], 0.0)
        print(f"  z={ANCHOR_Z[k]:<4} fit-sample median bias {med:+.4f} dex; "
              f"cond-oof correction 16-84 spread "
              f"{np.nanpercentile(delta['cond-oof'][tr, k], 84) - np.nanpercentile(delta['cond-oof'][tr, k], 16):.4f} dex "
              f"(shuffled control {np.nanpercentile(delta['cond-shuf'][tr, k], 84) - np.nanpercentile(delta['cond-shuf'][tr, k], 16):.4f})")

    good = (np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
            & finite.all(axis=1))
    print(f"\n  evaluated on the standard QA footing: {int(good.sum())} of "
          f"{n} galaxies (exp57 stage10's set).")

    out = {}
    for nm in LADDER:
        cog = pred * 10.0 ** delta[nm][:, :, None]
        out[nm] = qa.evaluate(cog[good], data[good], R, list(ANCHOR_Z),
                              name=f"exp58_{nm}", figdir=None, figures=False,
                              verbose=False, bin_by=lmh[good][:, 0],
                              bin_label=r"logM$_h$(z=0.4)")

    print(f"\n  TIER 3 — max|rel| beyond 5 kpc, median over galaxies "
          f"(ALL {int(good.sum())}, the QA view)")
    print(f"  {'variant':<14}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for nm in LADDER:
        print(f"  {nm:<14}" + "".join(
            f"{np.nanmedian(out[nm]['mr_out'][:, j]):>10.4f}"
            for j in range(5)))
    ug = use[good]
    print(f"\n  TIER 3 — the same, FIT-SAMPLE galaxy-epochs only (what a "
          f"refit would be scored on)")
    print(f"  {'variant':<14}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for nm in LADDER:
        mr = out[nm]["mr_out"]
        print(f"  {nm:<14}" + "".join(
            f"{np.nanmedian(mr[ug[:, j], j]):>10.4f}" for j in range(5)))
    print(f"\n  Where the two tables disagree, the fit sample and the QA "
          f"sample are different\n  populations at that epoch (the "
          f"progenitor-selection mask), and a law trained on\n  one is being "
          f"judged on the other.")
    for key in ("kpc:M(<10)", "kpc:M(<100)"):
        print(f"\n  {key} — median relative bias")
        print(f"  {'variant':<14}" + "".join(f"{f'z={z}':>10}"
                                             for z in ANCHOR_Z))
        for nm in LADDER:
            t, m = out[nm]["truth"][key], out[nm]["model"][key]
            print(f"  {nm:<14}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(m[:, j], t[:, j])):>9.1f}%"
                for j in range(5)))
    print(f"\n  kpc:M(<100) — dex scatter")
    print(f"  {'variant':<14}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for nm in LADDER:
        t, m = out[nm]["truth"]["kpc:M(<100)"], out[nm]["model"]["kpc:M(<100)"]
        print(f"  {nm:<14}" + "".join(
            f"{qa.dex_scatter(m[:, j], t[:, j]):>10.3f}" for j in range(5)))

    if not smoke:
        HERE.joinpath("outputs").mkdir(exist_ok=True)
        np.savez_compressed(HERE / "outputs" / "pc_ceilings.npz",
                            **{f"mr_out_{nm}": out[nm]["mr_out"]
                               for nm in LADDER},
                            **{f"delta_{nm}": delta[nm] for nm in LADDER})
        print(f"\n  saved outputs/pc_ceilings.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
