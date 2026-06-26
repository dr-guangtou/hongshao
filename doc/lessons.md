# Lessons

Mistakes, gotchas, and decisions worth remembering. Review at session start.

## Data handling (TNG300 drop)

- **Measure before encoding a data cut.** "Exclude halos whose mass decreases"
  sounded simple, but 98.6% of halos have *some* per-step decrease (median worst
  step ~4%) — normal fluctuation. The meaningful cut is net post-peak decline
  (z=0.4 mass >5% below historical peak → 695 halos). Always probe the
  distribution before choosing a threshold.
- **Verify snapshot indexing empirically, don't assume.** The MAH pickle is
  0-based TNG-native (snaps 1–71; snap 72 = z=0.4 is the unstored observation
  epoch). Confirmed by matching the cosmic-time file `t[snap]` to astropy ages
  at the anchors to <1 Myr. `t` indexes directly with no offset.
- **Don't trust file extensions.** `map_tng100_hist_stellar.hdf5` was 852 MB of
  all-zero bytes, not HDF5. Check magic bytes / load before relying on a file.
- **CoG / aperture arrays can be object-dtype with `None` bins** (~1/484). Coerce
  to NaN and mask; don't assume clean float arrays.
- **A 100% id "match" can still be wrong — check the *id system*, not just overlap.**
  `catgrp_id` (TNG FoF GroupID, 0…4831) matched 3388/3388 into the DiffMAH
  `halo_id` (global SubhaloID, 0…288404) simply because every small integer
  exists in both — but it read *satellites* (`halo_id=1` is a satellite of group
  0, not GroupID 1's central). Always validate a cross-match by a physical
  quantity (here: the matched `logmp_fit` had r≈0 and was ~2 dex off the known
  halo mass). GroupID ≠ SubhaloID; mapping between them needs `GroupFirstSub`.
- **The aperture table's `xy` projection reproduces the old npy aperture masses
  exactly** — that's the verification that `gal_num` == our `index`. Match a new
  table to an old one on a shared quantity before trusting the key.

## Analysis / interpretation

- **A "physical" constraint that scales with the target can fake a trend (exp25
  population fit).** Capping the deposition efficiency at the baryon fraction
  `f_b=0.157` seemed the obviously-physical bound. But via the SHMR, low-mass
  halos are ε-richer, so an `f_b` cap bites them *differentially* — it inverts
  `b_early<b_late`, destroys the two-epoch structure, and could manufacture a
  `z_c(Mₕ)` slope out of a mass-dependent distortion. The correct bound was the
  mass-blind hard limit `ε≤1` (can't form more stars than accreted mass).
  **Before enforcing a constraint, check whether it couples to the very axis you
  are testing the trend along; if so it can create or hide the effect.** I caught
  this only by looking at `b_early` vs `b_late` (not just RMS) on a *mass-stratified*
  validation subsample — the first-N rows were the most massive (the table is
  mass-ordered) and hid it.
- **With n in the thousands, p-values certify trends that don't exist (exp25).**
  `z_c` vs `logMₕ` had p=9e-6 — and r=−0.09, r²<0.01, the *opposite* sign to the
  prediction. n=2540 makes a physically-negligible slope "highly significant."
  Report the effect size (r, slope-over-range vs scatter), not the p-value, and
  cross-check on an independent estimator before claiming a relation. Here the
  honest verdict was a *null*: the quenching redshift carries no halo-mass signal.
- **A degenerate per-object parameter becomes identifiable only as a population
  constraint (exp25).** Per galaxy, `z_c`/`b_early` trade off and `b_late` rails at
  its bounds (identifiable in 1790/2540). Freezing the ~universal shape params at
  their population medians and re-fitting only the identifiable `(σ_0, z_c)` (the
  "reduced model") is the honest population statement, and it agreed with the
  full-fit identifiable subset. Don't read a population trend off a parameter that
  is non-identifiable on a single object — constrain it collectively first.
- **Per-galaxy fits are not a population fit, and the two can disagree on the
  interpretation (exp25 phase 3).** Fitting every galaxy independently (5 params ×
  2540) is exploration; a true population model shares parameters. The shared-kernel
  fit (minimize mean RMS over the whole sample at once) *inverted* the per-galaxy
  "steep early growth" (`b_early>b_late`) to steep-late (`b_early≈0.2<b_late≈4`),
  robustly (3/4 diverse starts to the same lower optimum). Cause: `(g, b_early,
  b_late)` are degenerate — "concentrated early" can be built by small early widths
  OR by slow width-growth + late efficiency. **Lesson:** the median of per-galaxy
  best-fits ≠ the joint population optimum, and a parameter's *physical
  interpretation* can flip between them even when the reconstruction is equally
  good. Report what is robust (reconstruction accuracy, the null mass-trend), and
  flag the degenerate parts as phenomenology, not inferred history. Test the
  population-level version of a trend directly (does adding a `z_c(M_h)` slope to
  the shared fit lower the loss? here: ΔRMS −0.0003, no) — it is stronger than a
  correlation over per-galaxy point estimates.
- **Don't conflate "a parameter is unpredictable" with "the signal is weak."**
  In exp08 the radial-DiffMAH shape params predict poorly from the MAH (R²≤0.22),
  and I wrote that "the MAH's influence on shape is weak." Wrong: a direct
  decomposition shows the MAH explains ~24% of the at-fixed-M0 variance in
  *concentration* (`log M(<10)/M(<100)`, R²+0.17, r≈0.45 — matching exp02/06).
  The shape signal is real and moderate; the radial-DiffMAH params are a
  degenerate, nonlinear *coordinate system* (β_in↔R_c=−0.54) that buries it.
  Lesson: when a parametric target predicts poorly, test the predictability of
  the *observable* it encodes (aperture masses, ratios, PCA modes) before
  concluding the signal is absent. Fixing one param (Δ, DiffMAH-style) doesn't
  fix a multi-axis degeneracy.

## Symbolic regression (PySR, exp12)

- **Don't compare a sparse SR pick against the full linear model — it's an
  unfair test.** PySR's `model_selection="best"` minimizes a complexity-vs-loss
  score, so its "best" equation often *drops* genuinely useful linear features
  (e.g. it kept only `logmp` + `late` for an aperture, dropping `logtc`/`early`),
  making "symbolic vs linear" measure PySR's pruning, not the value of a
  nonlinear term. Fix: always keep the full linear core and test the
  *incremental* value of the discovered nonlinear term (`linear + correction`).
- **To find what a linear model misses, run SR on its residuals, not the raw
  target.** OLS residuals are orthogonal to the linear features, so SR can only
  surface *nonlinear* structure (or nothing). On the raw target, SR wastes its
  complexity budget rediscovering the dominant linear term (`logmp`) and the
  small nonlinearity never surfaces.
- **Restricting SR to `{+,-,*,square}` keeps every equation polynomial**, so it
  reduces to *sparse selection over polynomial cross-terms*: each equation
  expands to a few monomials that plug straight into the existing
  linear-Gaussian emulator (reusing all of exp07/exp11 — CRPS, calibration,
  covariance). Far cleaner than refitting arbitrary nonlinear constants per fold.
- **Validate the SR finding with a residual-vs-feature figure, not just CRPS.**
  The +2% CRPS gain was easy to dismiss as noise; the convex U-shape of the
  linear residual vs `late` made the `late²` term obviously real (AGENTS.md
  visualize-don't-trust-metrics mandate, again decisive).
- Small-subsample SR can mislead on *which* term wins (n=397 picked `logtc·late`
  everywhere; full n=2539 picked `late²` for the outskirts). Validate the
  pipeline small but read the *physics* off the full sample.
- PySR's first import triggers a one-time Julia backend install + precompile
  (~3–5 min); afterwards a full run (8 deterministic serial fits, n=2539) is
  ~100 s.

## Scatter / calibration (exp14)

- **Marginal calibration can be fine while conditional calibration is badly
  off.** The homoscedastic emulator's overall interval coverage matched nominal
  (so marginal CRPS barely moved), yet split by predicted noisiness it covered
  ~80% of clean halos and ~60% of noisy ones inside the 68% interval. Always
  check coverage *within noisiness/feature bins*, not just the aggregate — a
  conditional density model lives or dies on per-object honesty.
- **Heteroscedasticity ≠ sharpness.** Modeling feature-dependent variance gave
  almost no marginal-CRPS gain (+1.1%) but +0.24 nats joint log-score and a ~10×
  smaller conditional-coverage gap. Judge a scatter model by the joint score and
  conditional calibration, not marginal CRPS.
- **Fit the log-variance by Gaussian MLE, not by regressing log(resid^2).** The
  latter is biased (E[log r^2] = log sigma^2 - 1.27 for a Gaussian). Minimizing
  0.5*sum[s + r^2 e^{-s}] with s = log-variance (closed-form gradient, L-BFGS) is
  exact, ~instant, and easy to regularize (ridge the slopes, not the intercept).
- **Diagnose tails with PIT before adding a fancy likelihood.** Once the variance
  was made halo-dependent the PIT was flat (no U-shape) — the Gaussian was
  adequate and a Student-t/flow would have been wasted complexity. The defect was
  heteroscedasticity, not non-Gaussian tails.
- `late` (recent-accretion index) is the single axis that governs the outskirts:
  it carries the mean nonlinearity (exp12 late^2) *and* the excess scatter
  (exp14). Recent accretion both boosts and destabilizes the outer envelope.
- **A predictor unbiased in X looks biased when you bin residuals by the noisy
  truth Y** (exp15). E[pred - Y | Y] has slope exactly -(1-R^2) — regression to
  the mean — present for the mean OR sampled predictions, because the truth on
  the x-axis contains the noise being conditioned on. The +0.3 dex low-end "bias"
  that looked alarming was this artifact (measured slope matched -(1-R^2) to
  three decimals). Diagnose with a *reliability diagram* (bin by PREDICTED, plot
  mean true): slope 1.0 -> the mean is fine. The only cure is more explained
  variance (raise R^2), not a better functional form; when R^2 is at its ceiling
  the shrinkage is irreducible. For generative use, SAMPLE from the predictive —
  the mean-only point estimate is under-dispersed and misses the tails.

## Secondary properties (exp16)

- **"It correlates with formation time" does NOT imply "it's redundant with the
  MAH."** We expected `c_200c` to add nothing once the DiffMAH params were in the
  model (it correlates r=0.58 with z50). Wrong: it adds +5% CRPS on DiffMAH and
  +2.7% even on the full MAH-PCA(4), and is only ~25% MAH-determined. A feature's
  correlation with the MAH bounds neither its independent information nor its
  incremental predictive value — measure the incremental value directly (with a
  shuffle control), don't infer redundancy from a marginal correlation.
- **Test a secondary feature on top of the *richest* representation you have**, not
  just the portable one. `c_200c` on top of DiffMAH(4) conflates two things
  (independent info + DiffMAH smoothing loss); adding it on top of MAH-PCA(4)
  isolates the genuinely-independent part (+2.7%). This split is exactly what
  separated the useful property from the redundant one (exp18): `acc_rate` helps
  on DiffMAH (+0.9%) but vanishes on MAH-PCA (it's MAH detail); `c_200c` survives.
- **"MAH-independent" ≠ "useful" (exp18).** Two orthogonal questions: (1) is the
  property determined by the MAH? [R²(P|MAH)] and (2) does it carry leftover info
  about the target? [partial corr]. 3D halo shape is MAH-independent (R²≈0) yet
  irrelevant to stellar mass (partial corr ≈0.07); `acc_rate` is the opposite
  (MAH-determined, R²=0.35). Only `c_200c` scores on both. Always plot both axes.
- **GBM (trees) is not always the best "ceiling" (exp17).** For a smooth,
  low-dimensional (~5 features), modest-sample relation, an explicit poly-2 beat
  the gradient-boosted-trees ceiling (0.0808 vs 0.0837), and a *more* flexible GBM
  only overfit. Use an analytic poly-2 alongside GBM when probing the nonlinear
  limit; "flexible model" doesn't guarantee "best approximator" at this scale.

## Workflow

- Background `uv run` commands buffer stdout through pipes; redirect to a file
  and read that, or block on the process, rather than polling a pipe.
