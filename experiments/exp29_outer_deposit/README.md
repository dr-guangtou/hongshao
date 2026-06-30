# exp29 — dip-free MAH + the deposition kernel, toward a multi-epoch emulator

> **STATUS / CORRECTION (supersedes the "centred Gaussian wins" headline below).**
> The slope-`b` result in `run.py` matched only the *tilt* of the inter-epoch
> growth. Fitting the **full** profile / CoG at multiple epochs told a richer story:
> - The centred Gaussian reproduces a *single-epoch* CoG to 0.005 dex but **not the
>   surface-density profile** (it cliffs at large R; `profile_fit.py`), and a frozen
>   single-epoch fit **mispredicts higher-z CoGs** (0.34 dex at z=2, dominated by a
>   mass-growth error; `cog_extrapolate.py`). Free Sérsic was tried and rejected
>   (bad math, steepens the core).
> - **Multi-epoch IS tractable** once the deposit *fraction* is freed: a convex NNLS
>   fit reaches **0.032 dex** jointly across 5 CoGs and **generalizes** (held-out
>   epoch 0.038 dex; `capacity_test.py`), and a **compact ~5–6-param emulator**
>   reaches ~0.040 dex (`emulator_param.py`). The blocker was never the kernel or a
>   rigid fraction — one epoch just can't constrain the fraction.
> - **The deposit *shape* is not the high-z limit** (`single_epoch_all.py`). Fitting
>   the centred Gaussian to *each epoch's CoG independently* reaches **≤1.4% max-rel
>   at every epoch and mass tertile**, with high-mass z=2 (0.9%) as good as z=0.4
>   (0.7%) — the BCG's z=2 is its *best* fit (0.3%). So the multi-epoch tension
>   (z=0.4 degrades to 19% jointly) is a **consistency** problem, not a shape one →
>   build the puff-up model (`PUFF_MODEL_PLAN.md`, un-parked).
>
> See **`PLAN.md`** for the full multi-epoch emulator plan + decision log. The
> sections below are the original (primitive / dip-free-MAH) investigation, kept for
> the record; the slope headline is superseded.

---

## (original) Question
The 2026-06-27 handover asked to return to the exp25/26 deposition kernel with two
changes: (1) feed it a **dip-free MAH** (the old main-branch `peak_history` has
merger-tree drop-outs), and (2) replace the **centred-Gaussian** deposit with a
**multiplicative power-law / outer-weighted (shell)** primitive, because exp26
found the *added* stellar mass is inside-out and multiplicative
(`Σ_low/Σ_high ∝ R^b`, long-baseline stack **b ≈ +0.95**).

This experiment implements both and tests the primitive against the right data —
the **multi-epoch** profiles, not the single z=0.4 curve of growth.

## The two changes
1. **Dip-free MAH** (`dipfree_mah`). The smooth official **DiffMAH fit** curve
   (`exp27/outputs/official_mah.npz → diffmah_log_mah_fit`) replaces
   `peak_history`. Same physical quantity (main-branch SubhaloMass Mpeak, h-free
   Msun) but **monotonic by construction and available for all galaxies** — the
   dip-free fallback the handover recommended over `max_prog_mpeak` (only 11 full
   trees exist). Verified monotonic; the BCG goes logMₕ 9.8 → 15.1 over 99 snaps.
2. **Generalized deposit** (`deposit.py`). One extra shape parameter `p`:
   ```
   Σ_i(R) = A_i · R^p · exp(−R²/2σ_i²)               (surface density)
   M_i(<R) = dM*_i · P(p/2+1, R²/2σ_i²)              (closed-form CoG, gammainc)
   ```
   `p = 0` is the exp25 centred Gaussian exactly (`P(1,x)=1−e^{−x}`); `p > 0` peaks
   off-centre at `R = σ·√p` (a shell); `A_i` is fixed by mass, so — like the
   Gaussian — each epoch still contributes a single width, no free amplitude.

## Why the multi-epoch test, and why the differential slope
A single cumulative profile barely constrains the deposit *shape* (exp25 fit the
z=0.4 CoG to 0.008 dex with the "wrong" primitive). The discriminating data are
the **5 measured epochs** (z=0.4/0.7/1.0/1.5/2.0, exp26 cache). We integrate the
deposits only to each `t(z_k)` ("stop at each redshift"), **stack over ~2400
galaxies**, and compare the inside-out slope `b` of `Δlog Σ = log Σ_lo − log Σ_hi`
for the 4 adjacent pairs and the long baseline. Crucially the model `b` is fit
**only over the radii where the data are valid** — without that, the model's
empty large-R tail at high z (Σ_hi → 0) creates a spurious huge `b`.

The temporal budget (the two-epoch efficiency `ε(z)` that sets how much mass forms
when) is **fixed at the exp25 population median** — this is a *spatial*-primitive
test. The overall size `σ₀` is fixed (60 kpc; `b` is set by the *shape* knobs
`g`, `p`, not the absolute size). Only the width-growth exponent `g` is fit, to
the long-baseline `b` with `p=0`; then `p` is scanned.

## Headline result — the centred Gaussian reproduces inside-out growth; p>0 is worse
`σ(t) = σ₀ (t/t_obs)^g`, best-fit **g = 0.55** (n = 2399):

| epoch pair | data `b` | model p=0 | model p=2 | model p=4 |
|---|---|---|---|---|
| z 0.4–0.7 | +0.13 | **+0.11** | +0.06 | +0.02 |
| z 0.7–1.0 | +0.14 | **+0.13** | +0.08 | +0.04 |
| z 1.0–1.5 | +0.27 | **+0.28** | +0.18 | +0.10 |
| z 1.5–2.0 | +0.33 | **+0.31** | +0.22 | +0.14 |
| **long (z2→0.4)** | **+0.82** | **+0.82** | +0.53 | +0.31 |

- The **centred Gaussian (p=0)** matches the *entire* `b(z)` trend (mean |Δb| =
  **0.012**), not just the long baseline it was fit to.
- **Outer-weighting monotonically makes it worse.** p=2,4 undershoot every pair;
  the long-baseline `b` falls 0.82 → 0.53 → 0.31. The data want `p = 0`.

## Why — the exp26 "not a Gaussian" result is about a *single* deposit
The forward model never deposits one Gaussian. It sums Gaussians whose **width
grows with cosmic time** (`g > 0`): early epochs land narrow/central, late epochs
land wide. The late wide Gaussians dominate the differential at large R, so the
**cumulative added mass is outer-weighted and multiplicative (b ≈ 0.8) even though
every primitive is centred** (figure D). exp26's "the added mass peaks at large R,
not at R=0" is true of a *single* deposit but the model's inside-out mechanism is
`σ(t)`, **not** the deposit's own off-centredness — which is redundant with `g`
and, pushed positive, double-counts the effect and overshoots into undershoot.

## What this refines in exp25
exp25 fit `g` to the z=0.4 CoG alone and got `g ≈ 1.67` (per-galaxy) / `2.41`
(population) — values the single-epoch CoG **cannot actually constrain** (the CoG
is forgiving). The multi-epoch differential **pins `g ≈ 0.55`**, and at that value
the centred Gaussian reproduces the observed `b(z)`. So exp29's contribution is a
*data-constrained* time→radius rule, and a clean answer to the handover's
primitive question: **keep the centred Gaussian.**

## Caveats
- **Efficiency and σ₀ fixed.** `b(z)` is a shape (log-ratio slope), insensitive to
  the overall normalization, so fixing `ε(z)` (median) and `σ₀` (60 kpc) is
  appropriate for the *shape* test — but a full population fit should free them.
  This experiment does **not** refit the absolute profile or `M*_tot`.
- **Ellipticity overshoot** in the isophote `Σ` cancels in `Δlog Σ` (≈constant per
  galaxy across epochs), so `b` is clean of it; absolute Σ normalization is not.
- **`σ(t)` is still a strict time→radius rule.** It reproduces the *stacked* `b(z)`,
  but it cannot encode an *event* (a late major merger that rebuilds the core);
  that remains exp25's flagged structural limitation, untouched by `p`.
- **z ≤ 2, 5 epochs, stacked.** Per-galaxy differentials are too noisy (exp26);
  the result is a population-median statement.

## Decision / next
- **Primitive question answered: the centred Gaussian is correct.** Drop the
  outer-weighted/shell idea — `σ(t)` is the inside-out mechanism and `p` is
  degenerate with `g` and counterproductive when positive.
- **Dip-free MAH is the new default input** for the kernel (DiffMAH fit curve, all
  galaxies, monotonic).
- **Next:** (a) redo the exp25 Phase-3 shared-kernel population *CoG* fit on this
  dip-free MAH with `g` anchored near 0.55; (b) make the width depend on the
  accretion **event** (merger mass ratio / smooth-vs-clumpy), the only way to
  capture late-merger core rebuilding that a pure `σ(t)` cannot.

## Files
- `deposit.py` — the generalized-gamma deposition primitive (pure math, reusable;
  `demo()` self-check: p=0 = Gaussian, mass-conserving, Σ peaks at σ√p).
- `run.py` — dip-free MAH loader, stacked multi-epoch differential fit, figures.
- `single_epoch_all.py` — independent single-epoch fits to every snapshot
  (z=0.4..2.0): is the centred-Gaussian *shape* a high-z limit? (No.) Honest linear
  max/90th-pct relative metric, aperture pinned at 148 kpc; caches best-fit params to
  `outputs/single_epoch_params.npz`; `demo` self-check.
- `puff_fit.py` — the **puff-up model**: one consistent history (mass frozen), only
  widths migrate post-deposition, `σ_i(t_k) = σ₀(t_i/t_obs)^g · (t_k/t_i)^q` (ratio)
  or `√(σ_{i,0}² + κ(t_k−t_i))` (diffusion), fit jointly (6 params). n=60: epoch-avg
  max|rel| **no-puff 9.1% → ratio 7.1% → diff 7.7%**, vs loose-zdep ~4.5%, ceiling
  0.7%. **Puffing helps but does NOT beat the looser z-dependent fit** (diffusion
  nearly inert, κ→0) → with mass frozen, width migration is a weaker lever than
  epoch-dependent mass distribution. `demo` self-check.
- `model_compare.py` — radial comparison of the four models (independent / loose-quad /
  puff-ratio / puff-diff): per-galaxy CoG + residual panels and **median
  relative-residual profiles vs R, per epoch x 3 mass bins**. Shows *where* each model
  fails: loose-quad overshoots the inner ~3-6 kpc at high z; the puff laws misplace mass
  at intermediate radii (10-50 kpc), worst for BCGs; puff-diff distorts the high-z BCG
  CoG outright. Only the independent ceiling is flat ~0 everywhere. `demo` self-check.
- `loose_zdep.py` — joint multi-epoch fit with each kernel param a **polynomial in
  the observation epoch z** (constant/linear/quad), vs the independent ceiling. n=60:
  epoch-avg max|rel| **fixed 10.2% → linear 4.8% → quad 4.5%**, ceiling **0.7%**.
  z-dependence ~halves the error but **plateaus ~6× above the ceiling** (quad≈linear,
  middle epochs hardest) — the degenerate per-epoch params don't lie on a low-order
  z-curve, so parsimonious z-trends can't reach single-epoch quality. Motivates a
  *structured* DOF (puff-up) over generic z-floating. `demo` self-check.
- `param_trends.py` — patterns in those best-fit params. **`g≈1.7` is epoch-stable**
  (a shared spatial kernel); the **efficiency rotates** (`b_early` 3.2→5.8 to high z);
  and the robust puff-up calibration — the **`R50` of the pre-z=2 mass grows ~1.8×
  (≈2.7× for BCGs)** from z=2 to z=0.4 (anchored: at z=2 it equals the data `R50`),
  realized in single-epoch fits via the efficiency, a freedom the joint fit lacks.
  Sets the magnitude for the puff-up law. `demo` self-check.
- `figures/exp29_centred_gaussian_wins.*` — (A) dip-free MAH, (B) stacked Δlog Σ
  data vs Gaussian model, (C) `b(z)` data vs model p=0/2/4, (D) why σ(t) does it.
- `outputs/manifest.json` — best `g`, the `b(z)` tables.

Run: `PYTHONPATH=. uv run python experiments/exp29_outer_deposit/run.py`
