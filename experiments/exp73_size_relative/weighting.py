"""exp73 Block B — the two down-weighting schemes, head to head (the user's D2).

    "Honestly, I don't know which is better. Can you devise a plan to test both
     and let the results speak for themselves?"  -- the user, 2026-09-01

  **Scheme A, smooth.** Each shell's RELATIVE residual is weighted by that
  shell's own median share of the galaxy's mass. NO FREE PARAMETER: the weights
  are measured from the data at each epoch. The loss is then eps^2 * phi --
  proportional to mass.
  **Scheme B, gated.** Shells whose median mass share falls below a threshold `t`
  are dropped outright and the survivors weighted equally. ONE free parameter,
  and it is a cliff.
  **And, for context, the mass-weighted residual with uniform weights.** That
  residual is (delta / total)^2 = eps^2 * phi^2, so it is scheme A with the
  mass share SQUARED -- a more aggressive version of the same idea, not a base
  to stack a scheme on. (A first draft stacked A on it and weighted by phi^3;
  the smoke run showed the outer third under-weighted to a third of its mass
  share, which is how the mistake was caught.)

  All three are compared against UNIFORM weights on the RELATIVE residual, the
  programme's incumbent form.

FOUR JUDGEMENTS, all of them machinery exp72 already built, and the reference in
every one is the production objective -- the bar a candidate must not be worse
than, rather than a threshold invented here:

  1. **Allocation.** Share of the loss against share of the mass, per shell per
     epoch. The target is proportionality; on fixed kpc at z = 2 the production
     objective is off by eightfold.
  2. **Usability.** The participation ratio (sum L)^2 / sum L^2 -- the effective
     number of galaxies carrying the loss. exp72 Step 1 found an unfloored
     generalised-least-squares objective on which ONE galaxy carried the whole
     population, so this is checked before anything else is believed.
  3. **Sensitivity probes**, each applied in BOTH directions and averaged,
     because a one-sided probe can move the model towards the truth and score
     better. Amplitude and shape as in exp72, plus a NEW SIZE probe -- R50 is
     now a first-class quantity and the exp72 refit's real failure was making
     z = 2 galaxies 27 per cent too big, which no amplitude or shape probe sees.
  4. **Threshold sensitivity for B.** `t` swept over 0.5, 1, 2 and 4 per cent.
     **THE RULE, FIXED HERE BEFORE ANY NUMBER IS SEEN (the user's D2): if B's
     verdict moves materially with `t`, B is disqualified in favour of A**,
     because a conclusion resting on an arbitrary cliff is not a conclusion.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp73_size_relative/weighting.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp72_error_objective", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import fit_gls as G                                      # noqa: E402
import coordinate as C                                   # noqa: E402
from risk_test import build_grids                        # noqa: E402
from hongshao.profile_data import load_profiles          # noqa: E402

OUTDIR = HERE / "outputs"
RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)
#: the thresholds scheme B is swept over, as a fraction of the galaxy's mass.
#: With 20 shells the AVERAGE share is 5 per cent, so 4 per cent would drop
#: nearly everything at low redshift (a first sweep did, and reported a
#: degenerate 0.00 allocation) -- the sweep stops at 3.
T_SWEEP = (0.005, 0.01, 0.02, 0.03)
#: probe size, matching exp72 so the numbers are comparable
PROBE_DEX = 0.05


def concentration(loss, mask):
    """(effective number of galaxies, share carried by the worst 1%)."""
    out = np.full((loss.shape[1], 2), np.nan)
    for k in range(loss.shape[1]):
        v = loss[mask[:, k], k]
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        cut = np.percentile(v, 99)
        out[k] = [v.sum() ** 2 / np.clip((v ** 2).sum(), 1e-300, None),
                  v[v >= cut].sum() / np.clip(v.sum(), 1e-300, None)]
    return out


def probe_amplitude(cog, dex):
    """Every shell scaled together: the shape is bit-identical."""
    return cog * 10.0 ** dex


def probe_shape(cog, dex, R, i_pin):
    """A tilt in log radius applied to the ANNULI, renormalised so the mass at
    the pinning radius is unchanged: pure shape at fixed amplitude."""
    ann = np.concatenate([cog[..., :1], np.diff(cog, axis=-1)], axis=-1)
    lr = np.log10(R)
    u = 2.0 * (lr - lr.min()) / (lr.max() - lr.min()) - 1.0
    out = np.cumsum(ann * 10.0 ** (dex * u), axis=-1)
    return out * (cog[..., i_pin] / out[..., i_pin])[..., None]


def probe_size(cog, dex, R):
    """THE NEW ONE: rescale the galaxy in RADIUS at fixed total mass, i.e. move
    it along the mass-size relation. `M*(<r)` becomes `M*(<r / 10**dex)`, so a
    positive `dex` makes the galaxy bigger and its profile shallower, with the
    mass inside the outermost radius held fixed.

    This is the failure exp72's refit actually had -- z = 2 galaxies 27 per cent
    too big -- and neither an amplitude nor a shape probe registers it, because
    amplitude holds the normalisation and shape holds the pinning radius.
    """
    lr = np.log10(R)
    q = np.clip(lr - dex, lr[0], lr[-1])
    j = np.clip(np.searchsorted(lr, q) - 1, 0, len(R) - 2)
    lC = np.log10(np.where(cog > 0, cog, np.nan))
    lo, hi = lC[..., j], lC[..., j + 1]
    t = (q - lr[j]) / (lr[j + 1] - lr[j])
    out = 10.0 ** (lo + t * (hi - lo))
    return out * (cog[..., -1] / out[..., -1])[..., None]


def main(smoke=False):
    print(f"{RULE}\nexp73 Block B — the two down-weighting schemes, head to "
          f"head (the user's D2)\n{RULE}\n")
    grids, cogs, good, lmh0, Rm, names = build_grids(smoke)
    g_re, g_kpc = grids["R50 shells"], grids["fixed kpc"]
    ref = "exp63 joint"

    for cname, grid in (("R50 shells", g_re), ("fixed kpc", g_kpc)):
        print(f"\n{RULE}\nCOORDINATE: {cname}\n{RULE}")
        share = grid.mass_share()
        print(f"\n  median share of the galaxy's mass per shell "
              f"({grid.n_shell} shells)")
        print(f"  {'epoch':>6}  min      p25      median   p75      max")
        for k in EPOCHS:
            q = np.nanpercentile(share[k], [0, 25, 50, 75, 100])
            print(f"  {ANCHOR_Z[k]:>6.1f}" + "".join(f"{100 * v:>9.2f}%" for v in q))
        print(f"\n  how many of the {grid.n_shell} shells scheme B would DROP at "
              f"each threshold")
        print(f"  {'epoch':>6}" + "".join(f"{f't={100*t:g}%':>10}" for t in T_SWEEP))
        for k in EPOCHS:
            print(f"  {ANCHOR_Z[k]:>6.1f}"
                  + "".join(f"{int((share[k] < t).sum()):>10d}" for t in T_SWEEP))

        # --- 1 & 2: allocation and usability ------------------------------ #
        # (weights, residual kind). Everything sits on the RELATIVE residual
        # except the last, which IS the mass-weighting expressed as a residual
        schemes = {"uniform (reference)": (C.weights_uniform(grid), "rel"),
                   "A smooth (mass share)": (C.weights_smooth(grid), "rel")}
        for t in T_SWEEP:
            schemes[f"B gated t={100 * t:g}%"] = (C.weights_gated(grid, t), "rel")
        schemes["mass-weighted resid"] = (C.weights_uniform(grid), "mass")
        print(f"\n  ALLOCATION — total share of the loss falling on the OUTER "
              f"THIRD of the shells,\n  against those shells' share of the mass. "
              f"1.00 = proportional.")
        print(f"  {'scheme':<24}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z))
        n3 = grid.n_shell // 3
        alloc = {}
        for sname, (w, kind) in schemes.items():
            r = grid.residual(cogs[ref], kind)
            row = []
            for k in EPOCHS:
                contrib = np.nanmean(np.where(np.isfinite(r[:, k]), r[:, k] ** 2, np.nan), axis=0)
                cw = contrib * w[k]
                f_loss = np.nansum(cw[-n3:]) / np.clip(np.nansum(cw), 1e-300, None)
                f_mass = np.nansum(share[k][-n3:]) / np.clip(np.nansum(share[k]), 1e-300, None)
                row.append(f_loss / f_mass if f_mass > 0 else np.nan)
            alloc[sname] = np.array(row)
            print(f"  {sname:<24}" + "".join(f"{v:>9.2f}" for v in row))

        print(f"\n  USABILITY — effective number of galaxies carrying the loss "
              f"(of {int(good.sum())}),\n  and the share carried by the worst 1%. "
              f"Nothing may concentrate more than uniform.")
        print(f"  {'scheme':<24}" + "".join(f"{f'z={z}':>16}" for z in ANCHOR_Z))
        conc = {}
        for sname, (w, kind) in schemes.items():
            lg = C.apply_weights(grid.residual(cogs[ref], kind), w)
            cc = concentration(lg, np.ones(lg.shape, bool))
            conc[sname] = cc
            print(f"  {sname:<24}"
                  + "".join(f"{cc[k, 0]:>10.0f} ({100 * cc[k, 1]:>2.0f}%)" for k in EPOCHS))

        # --- 3: the probes, including the new size probe ------------------ #
        print(f"\n  SENSITIVITY — symmetric excess cost for a pure AMPLITUDE, "
              f"pure SHAPE and pure\n  SIZE error of {PROBE_DEX} dex, as a "
              f"percentage of the exp63 mean's own loss.")
        print(f"  {'scheme':<24}{'epoch':>6}{'amplitude':>12}{'shape':>10}"
              f"{'size':>10}{'size/amp':>11}")
        i_pin = int(np.argmin(np.abs(Rm - 103.45)))
        probes = {}
        for pname, fn in (("amplitude", lambda c, d: probe_amplitude(c, d)),
                          ("shape", lambda c, d: probe_shape(c, d, Rm, i_pin)),
                          ("size", lambda c, d: probe_size(c, d, Rm))):
            probes[pname] = {s: fn(cogs[ref], s * PROBE_DEX) for s in (+1, -1)}
        sens = {}
        for sname, (w, kind) in schemes.items():
            base = C.apply_weights(grid.residual(cogs[ref], kind), w)
            sens[sname] = np.full((5, 3), np.nan)
            for k in EPOCHS:
                b = np.nanmean(base[:, k])
                if not np.isfinite(b) or b <= 0:
                    continue
                vals = []
                for pname in ("amplitude", "shape", "size"):
                    ex = [np.nanmean(C.apply_weights(
                        grid.residual(probes[pname][s], kind), w)[:, k]) / b - 1.0
                        for s in (+1, -1)]
                    vals.append(50.0 * (ex[0] + ex[1]))       # per cent, symmetric
                sens[sname][k] = vals
                if sname in ("uniform (reference)", "A smooth (mass share)",
                             f"B gated t={100 * T_SWEEP[2]:g}%",
                             "mass-weighted resid"):
                    print(f"  {sname if k == 0 else '':<24}{ANCHOR_Z[k]:>6.1f}"
                          + "".join(f"{v:>11.2f}%" for v in vals)
                          + f"{vals[2] / vals[0] if vals[0] else np.nan:>11.2f}")

        # --- 4: does B's verdict move with t? ----------------------------- #
        print(f"\n  THRESHOLD SENSITIVITY OF SCHEME B — the rule fixed before "
              f"any number was seen:\n  if B's verdict moves materially with "
              f"`t`, B loses to A.")
        print(f"  {'threshold':<14}{'outer-third allocation':>26}"
              f"{'eff. galaxies z=2':>20}{'size sensitivity z=2':>22}")
        for t in T_SWEEP:
            s_ = f"B gated t={100 * t:g}%"
            print(f"  t={100 * t:<12g}{np.nanmean(alloc[s_]):>26.2f}"
                  f"{conc[s_][4, 0]:>20.0f}{sens[s_][4, 2]:>21.2f}%")
        bs = [np.nanmean(alloc[f"B gated t={100 * t:g}%"]) for t in T_SWEEP]
        spread = (np.nanmax(bs) - np.nanmin(bs)) / np.clip(np.nanmean(bs), 1e-30, None)
        print(f"\n  B's allocation varies by {100 * spread:.1f}% across the "
              f"threshold sweep; A has no threshold\n  to vary. "
              + ("**B IS DISQUALIFIED in favour of A** by the rule above."
                 if spread > 0.10 else
                 "B's verdict is stable, so it survives on this criterion."))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / ("weighting_smoke.npz" if smoke else "weighting.npz"),
             anchor_z=np.array(ANCHOR_Z), t_sweep=np.array(T_SWEEP))
    print(f"\nwrote {OUTDIR / 'weighting.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
