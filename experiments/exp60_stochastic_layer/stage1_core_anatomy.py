"""exp60 Stage 1 — the core component's anatomy: what distribution of
per-galaxy expansion strengths does the data actually demand?

THE QUESTION. The layer's new component draws a per-galaxy core-expansion
strength `A_i` (applied through the gate-passing small-core remap, `X3` with
`Rc` at the Stage 2 geometry of 1.98 kpc, around the UNCHANGED incumbent).
Before any distribution is fitted, measure the empirical target: for each
galaxy, the `A_i` that makes the model's central change (at 4.92 kpc,
z=2 → 0.4) equal its measured one — and the fraction of galaxies for which
NO admissible `A_i` works, which is the component's ceiling, stated before
the component is built.

HOW IT IS CHEAP. Galaxies are independent in the forward model, so evaluating
the SHARED-`A` model on a grid of `A` values gives every galaxy's individual
response curve at once; each galaxy's `A_i` is then read off its own
monotone curve by interpolation. Fifteen evaluations replace 2397 root
solves. (Monotonicity — more expansion, less central growth — is CHECKED per
galaxy, not assumed.)

WHAT `A_i` MEANS, and does not. The measured central change mixes physics
(SMBH-driven expansion) with projection/orientation noise (the user's
reading, 2026-08-26). A draw layer models the observed variation whatever
its origin, so the empirical `A_i` distribution absorbs both — which is
correct for generating realistic populations and one more reason `A_i` must
never be treated as a per-galaxy physical measurement.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp60_stochastic_layer/stage1_core_anatomy.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term"):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage4_target as S4                               # noqa: E402

I5 = 3                        # R_GRID[3] = 4.92 kpc, matching Stage 0
#: candidate core radii for the LAYER's remap [log10 kpc]. The first run of
#: this script used the shared-mechanism's gate-optimal 1.98 kpc and found
#: the response at 4.92 kpc nearly FLAT — a 2 kpc remap moves mass from
#: inside 2 kpc to just outside it, which stays inside the 4.92 kpc aperture
#: the target is defined on. The layer needs a core radius larger than the
#: target aperture's scale; the shared-A trade that forced 2 kpc (gate G5b)
#: does not bind a PER-GALAXY draw, because galaxies that need no change
#: draw A near 0, and exp57 Stage 8 measured the reach gate passing for Rc
#: up to ~40 kpc. THE SELECTION RULE, declared before the numbers: the
#: SMALLEST candidate whose response curves cover at least 90% of the
#: measured central changes within the A bounds. The drawn population's
#: reach is re-gated at Stage 3 regardless (gate S4).
RC_CANDIDATES = (0.2967, 0.70, 1.00, 1.30)          # 1.98, 5.0, 10.0, 20 kpc
#: A grid, dense at small |A| where most galaxies live, out to the X-law
#: bounds. The bounds themselves are included so "unreachable" is measured
#: at the law's actual limits.
A_GRID = np.array([-0.9, -0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 1.0,
                   1.5, 2.0, 3.0, 4.5, 6.5, 9.0, 13.0, 20.0])
COVER_MIN = 0.90
OUT = HERE / "outputs" / "stage1_core_anatomy.npz"
RULE = "=" * 96


def main(smoke=False):
    print(f"{RULE}\nexp60 STAGE 1 — the per-galaxy core-expansion anatomy"
          f"\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    n = len(recs)
    fell, ok = S4.decline_flag(data)
    sp = X.XSpec(base, "X3")
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))

    with np.errstate(invalid="ignore", divide="ignore"):
        dc_meas = np.log10(data[:, 0, I5] / data[:, 4, I5])

    def build_curves(log_rc):
        """Every galaxy's central-change response to A, from shared-A
        evaluations (galaxies are independent in the forward model)."""
        c = np.full((len(A_GRID), n), np.nan)
        for j, a in enumerate(A_GRID):
            th = sp.nest(theta).copy()
            th[-2], th[-1] = a, log_rc
            p = pr.predict(th)
            with np.errstate(invalid="ignore", divide="ignore"):
                c[j] = np.log10(p[:, 0, I5] / p[:, 4, I5])
        return c

    # --- the Rc selection, by the declared coverage rule ------------------ #
    print(f"  Rc selection (rule: smallest with >= {100 * COVER_MIN:.0f}% "
          f"coverage of the measured changes):")
    chosen, curves = None, None
    for log_rc in RC_CANDIDATES:
        c = build_curves(log_rc)
        g = ok & np.isfinite(dc_meas) & np.isfinite(c).all(axis=0)
        cover = float(((dc_meas <= c[0]) & (dc_meas >= c[-1]))[g].mean())
        span = np.median(c[0][g] - c[-1][g])
        print(f"    Rc = {10 ** log_rc:5.2f} kpc: median response span "
              f"{span:.3f} dex, coverage {100 * cover:5.1f}%")
        if chosen is None and cover >= COVER_MIN:
            chosen, curves = log_rc, c
    if chosen is None:
        chosen = RC_CANDIDATES[-1]
        curves = build_curves(chosen)
        print(f"    no candidate reaches {100 * COVER_MIN:.0f}% — using the "
              f"largest ({10 ** chosen:.1f} kpc) and reporting the shortfall")
    print(f"  CHOSEN: Rc = {10 ** chosen:.2f} kpc\n")

    good = ok & np.isfinite(dc_meas) & np.isfinite(curves).all(axis=0)
    # The response is a DIFFERENCE of two curves that both decrease with A
    # (each epoch's central mass falls as the core expands), so per-galaxy
    # wiggles of a few thousandths of a dex are expected and harmless to a
    # 1-D inversion. A galaxy is excluded only when its worst uphill step
    # exceeds MONO_TOL — and the worst violation over the sample is printed,
    # so the tolerance is auditable rather than silent.
    MONO_TOL = 0.005
    viol = np.max(np.diff(curves, axis=0), axis=0)
    mono = viol <= MONO_TOL
    print(f"  admitted {int(good.sum())} of {n}; monotone within "
          f"{MONO_TOL} dex for {int((mono & good).sum())} "
          f"(worst uphill step in the sample {np.nanmax(viol[good]):.4f} "
          f"dex; {int((good & ~mono).sum())} excluded)")
    use = good & mono

    # invert each curve at the measured value (curves decrease with A)
    a_i = np.full(n, np.nan)
    lo_hit = use & (dc_meas > curves[0])      # needs contraction beyond -0.9
    hi_hit = use & (dc_meas < curves[-1])     # needs expansion beyond 20
    inv = use & ~lo_hit & ~hi_hit
    idx = np.where(inv)[0]
    for i in idx:
        a_i[i] = np.interp(-dc_meas[i], -curves[:, i], A_GRID)
    a_i[lo_hit], a_i[hi_hit] = A_GRID[0], A_GRID[-1]

    print(f"\n  REACHABILITY — the component's own ceiling, before it is "
          f"built:")
    print(f"    matched exactly     {int(inv.sum()):5d}  "
          f"({100 * inv.sum() / use.sum():.1f}% of admitted)")
    print(f"    need A < -0.9       {int(lo_hit.sum()):5d}  "
          f"({100 * lo_hit.sum() / use.sum():.1f}%)  — centres grew faster "
          f"than the un-expanded model")
    print(f"    need A > 20         {int(hi_hit.sum()):5d}  "
          f"({100 * hi_hit.sum() / use.sum():.1f}%)")

    q = np.nanpercentile(a_i[inv], [5, 25, 50, 75, 95])
    print(f"\n  THE EMPIRICAL A DISTRIBUTION (matched galaxies):")
    print(f"    percentiles 5/25/50/75/95: "
          + "  ".join(f"{v:+.3f}" for v in q))
    print(f"    fraction with A_i > 0.5 (a material expansion): "
          f"{100 * np.mean(a_i[inv] > 0.5):.1f}%")
    def safe_med(m):
        return np.nanmedian(a_i[m]) if m.any() else np.nan
    print(f"    declining galaxies' median A_i "
          f"{safe_med(inv & fell):+.3f} against non-declining "
          f"{safe_med(inv & ~fell):+.3f}")

    # the conditioning the mixture tilt will use
    fform = np.array([h.f_form[-1] for h in recs])
    qs = np.nanpercentile(fform[use], [25, 50, 75])
    edges = [-np.inf, *qs, np.inf]
    print(f"\n  PER f_form(z=0.4) QUARTILE (earliest-assembled first) — the "
          f"tilt target's shape:")
    print(f"    {'quartile':>10} {'median A_i':>12} {'A_i > 0.5':>11} "
          f"{'need A<-0.9':>13}")
    for qk, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = use & (fform > lo) & (fform <= hi)
        print(f"    {f'Q{qk + 1}':>10} "
              f"{np.nanmedian(a_i[m & inv]):>+12.3f} "
              f"{100 * np.mean(a_i[m & inv] > 0.5):>10.1f}% "
              f"{100 * (lo_hit & m).sum() / max(m.sum(), 1):>12.1f}%")
    r_ff = np.corrcoef(fform[inv], a_i[inv])[0, 1]
    print(f"    correlation of A_i with f_form: r = {r_ff:+.3f} (the shared-"
          f"law ceiling was r = -0.317 on the decline FLAG)")

    if not smoke:
        OUT.parent.mkdir(exist_ok=True)
        np.savez(OUT, a_grid=A_GRID, log_rc=chosen, a_i=a_i,
                 matched=inv, lo_hit=lo_hit, hi_hit=hi_hit, use=use,
                 dc_meas=dc_meas, fell=fell, ok=ok, fform=fform)
        print(f"\n  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
