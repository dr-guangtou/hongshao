# Lessons

Mistakes, gotchas, and decisions worth remembering. Review at session start.

## Data handling (TNG300 drop)

- **DiffMAH was fit to SubhaloMass, not M200c (exp28 refines exp27).** Per-halo,
  the catalog `log_mah_sim` equals the tree main-branch **SubhaloMass** Mpeak to
  <0.005 dex; vs M200c it differs ~0.05–0.07 dex. exp27's "≡ M200c" held only at the
  sample median (where SubhaloMass≈M200c). Put tree-derived MAHs on SubhaloMass to
  compare with DiffMAH. Also: the DiffMAH *fit* carries a real ~0.1–0.17 dex residual
  vs its own data at z=0.4 (z=0-anchored rolling power law overshoots near z=0.4).
- **DiffMAH catalog bridge = position, not group ID (exp27).** `diffmah_tng.h5`
  is keyed by a z=0 row index (`halo_id`), but its `x/y/z` arrays are the
  main-progenitor-branch positions with **array column == TNG snapshot number**
  and units **cMpc/h**. So matching our snap-72 galaxies is exact: their
  `SubhaloPos` (ckpc/h ÷1000) equals `diffmah[:, x/y/z, 72]` to float precision —
  built from the same SUBFIND catalog. The earlier `catgrp_id↔halo_id` guess in
  todo was wrong; KDTree on the snap-72 column is the robust bridge. Only ~93%
  match: a z=0.4 subhalo with no surviving z=0 main-branch descendant (merged)
  simply has no DiffMAH row, and that shows as a clean *second* distance mode
  (16–122 ckpc/h), so the 1-ckpc/h tolerance is unambiguous — flag, don't force.
- **DiffMAH catalog masses are Msun/h, not Msun (exp27).** `diffmah_tng.h5`
  `log_mah_*`/`logmp_*` are log10(M / [Msun/h]); our M200c is Msun. The raw gap is
  a flat −0.169 dex (= log10 h) that masquerades as a physics offset — it inflated
  the own-vs-official curve RMS from 0.078 to 0.221 dex. Add +log10(1/h)=+0.169
  before comparing; then DiffMAH `log_mah_sim` ≡ M200c to −0.008 dex (same mass
  def). Always reconcile little-h before believing a ~0.17 dex mass discrepancy.
- **Compare DiffMAH fits by reconstructed CURVE, not raw params (exp27).** The
  four params are degenerate: own-vs-official `early` scatters 1.1, `logtc` 0.5
  (ours even rails its bounds because the z=0.4-limited fit can't see the
  transition), yet the reconstructed M(t) agree to 0.078 dex. So normalization /
  M(t) are interchangeable across flavours, but never feed mixed-flavour raw
  `(logtc, early, late)` into one model — pick one flavour per model.
- **Use stdlib `urllib`, not `requests`, in the `uv` venv (exp27).** The project
  venv has no `requests`; `urllib.request` + `ThreadPoolExecutor` pulls 3388 TNG
  MPB files at 1.9 gal/s (10 workers) with retry/backoff — no new dependency. The
  TNG API extracts trees server-side (~5 s/gal serial), so concurrency, not
  bandwidth, is the win. Cache + validate-on-open makes it resumable for free.

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
- **Difference in log/fractional space, not linear, when the profile is steep
  (exp26).** Differencing two measured surface-density profiles `ΔΣ=Σ_low−Σ_high`
  at fixed radius is dominated by the steep, marginally-resolved core: a tiny
  size shift makes a huge linear `ΔΣ` there, so the "added mass" looked mostly
  negative and noise-classified ~40–80% of galaxies as "core-drop". Switching to
  `Δlog Σ` (fractional growth) gave a clean, robust signal: growth rises with R
  (inside-out), `Σ_low/Σ_high ∝ R^b`, and only ~3% truly drop their core. When an
  observable spans orders of magnitude, the fractional change is the robust thing
  to characterise; the linear difference inherits the dynamic range as noise.
- **Small Δt differentials are noise; stack or use the long baseline (exp26).**
  Per-galaxy `ΔΣ`/`Δlog Σ` between *adjacent* snapshots (small Δt, similar Σ) is
  noise-dominated — per-galaxy slope fits scattered around zero. The population
  *stacked* (per-radius median) profile and the long baseline (z=2→0.4) gave the
  real trend. Fit the stacked profile, not the median of per-object fits, when the
  per-object signal is below the noise.
- **Check the sign convention with a known-sign case (exp26).** A swapped
  `(earlier, later)` tuple in the pair list silently flipped every adjacent-pair
  differential (growth read as decline); only the long-baseline call, written in
  the right order, looked correct — which is what exposed the bug. A galaxy that
  grows must have `Σ(later) > Σ(earlier)`: assert the obvious-sign case before
  trusting the subtle ones.
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

- **A cumulative profile barely constrains the deposit *shape* (exp29).** exp25 fit
  the z=0.4 curve of growth to 0.008 dex with a centred Gaussian; the multi-epoch
  *differential* (`Δlog Σ` between epochs) is the discriminating test. Fit the thing
  that actually depends on the unknown (here the per-epoch added mass), not an
  integral that washes it out — and don't read a good single-epoch fit as
  validating the primitive.
- **Don't confuse a cumulative/differential statement with a primitive one
  (exp29).** exp26 "the added mass peaks at large R, not a centred Gaussian" is
  about the *summed* growth; the forward model that reproduces it deposits **centred
  Gaussians whose width σ(t) grows with cosmic time** — late wide clumps make the
  cumulative added mass outer-weighted (b≈0.8) with every primitive centred.
  Outer-weighting `p>0` is then redundant with `g` and overshoots into undershoot.
- **When comparing model vs data slopes, fit both over the *same valid radii*
  (exp29).** A model with a clipped floor (Σ→0 at large R, high z) produces a
  spurious huge `b` where the data have no signal; restricting the model's slope fit
  to the data's flagged radii changed model long-baseline `b` from 6.6 to 0.8.
- **Isolate "is the *shape* the limit?" with an independent single-epoch fit per
  epoch (exp29).** The centred-Gaussian deposit fits *every* epoch's CoG to <1%
  max-rel **including z=2 for the most massive galaxies** (high-mass tertile z=2 =
  0.9% vs z=0.4 = 0.7%; the BCG's z=2 is its *best* fit, 0.3%). So the multi-epoch
  failure (z=0.4 degrades to 19% when fit jointly) is **not** a shape limit — the
  Gaussian sum can describe compact high-z massive profiles — it is a *consistency*
  limit (fixed-width additive deposits can't redistribute early-compact mass). The
  decisive test was fitting each epoch alone, not reasoning from the joint fit.
- **Pin the normalization to the measured aperture, not the deposited total
  (exp29).** Centred Gaussians leak ~8% mass beyond 148 kpc, so normalizing the
  *total* deposit mass to M*(z_k) leaves the model CoG ~0.035 dex low at the outer
  radius and contaminates the shape params. Rescaling the model to match the data at
  the 148-kpc point (`model *= data[-1]/model[-1]`) removes the leak cleanly so the
  shape test is honest.
- **Mean log-RMS hides structured error; report linear max/90th-pct relative
  residual over R>3 kpc (exp29).** Single-epoch log-RMS is ~0.001–0.002 dex at every
  epoch, but the honest linear metric (max|rel|, 90th|rel|, inner ≤3 kpc softening
  excluded) is what makes "z=2 is as easy as z=0.4" a defensible claim rather than an
  averaged-away one.
- **Read parameter trends through degeneracy-robust *derived* quantities, anchored
  where they're constrained (exp29).** The fitted width law `σ(t)=σ₀(t/t_obs)^g` is
  anchored at `t_obs=t(z=0.4)`, so `σ₀` inflates for high-z fits (extrapolation) and
  a single-deposit width `σ(t_ref)` even gave the *wrong-signed* puff conclusion. The
  mass-weighted half-mass radius of the early-formed (pre-z=2) mass is robust — and
  it's anchored at z=2 where that mass is 100% of the galaxy, so `R50_early(z=2)` =
  the data `R50`. That flipped the verdict: early mass must extend ~1.8× (≈2.7× for
  BCGs) by z=0.4. Don't read raw degenerate params; build a derived quantity that an
  independent anchor pins.
- **A single-epoch fit can fake a missing DOF through a *different* knob; the joint
  fit can't (exp29).** The independent fits made early mass more extended at z=0.4 by
  re-spreading it via the per-epoch efficiency `f(t_i)`, not by widening deposits —
  freedom a joint fit lacks (`f` is one fixed function). So "each epoch fits great"
  does not imply "one model fits all epochs"; the param *trend* across epochs is what
  reveals the DOF the joint model is missing.
- **Smooth low-order z-dependence of degenerate params plateaus far above the
  per-epoch ceiling (exp29).** Making the 5 kernel params a polynomial in observation
  z and fitting jointly cut multi-epoch max|rel| from ~10% (fixed) to ~4.5%
  (linear/quad) — a reasonable fit — but quad barely beat linear and both stayed ~6×
  above the 0.7% single-epoch ceiling, because the per-epoch best-fit params are
  degenerate/scattered and don't lie on a low-order curve (you'd need ~quartic =
  per-epoch freedom to reach it). Parsimonious z-trends ≠ single-epoch quality;
  closing the gap needs *structured* freedom (e.g. puff-up: fix mass+g, vary width),
  not more polynomial order. Test the structured model against this ~4.5% benchmark.
- **Core-retaining redistribution breaks the additive consistency floor (exp30).** A
  two-component deposit (retained core at the deposition width + migrated envelope,
  mass-conserving, observation-time dependence only via elapsed time) halves the joint
  multi-epoch error (additive 18.5% → transport 9.1% max|rel|, flat across epochs) and
  edges past the inconsistent loose-zdep fit with a genuine consistent history. The CoG
  stays linear in the masses, so the convex NNLS inner solve survives — the transport
  freedom costs only 2-3 outer params. Two form lessons: a GLOBAL migration clock
  under-migrates by z=2 (front-loaded merging wants tau ∝ t_i), and the migrated width
  must be set at the OBSERVATION epoch, not the (tiny) birth width — else the high-z
  envelope is unreachable.
- **Parametric masses fixed the generalization failure exactly as diagnosed (exp30).**
  Replacing the ~70 free NNLS masses with the 3-param two-epoch efficiency inside the
  dyntrans transport structure (7 params total) cost only +2.2 points in-sample
  (7.5%→9.7%) but transformed prediction: LOEO held-avg 53.7%→24.0% (gap +46→+14),
  beating every previous model at every held-out epoch INCLUDING the previously
  impossible z=0.4 forward extrapolation. The information the free masses "used" was
  mostly epoch-specific noise; 3 efficiency numbers carry the real signal. Corollary:
  when a diagnosis (free masses = liability) is this specific, the targeted fix's
  success/failure is itself a test of the diagnosis — and it passed. Note the fitted
  clock reproduced alpha≈1 (self-similar) for the third independent time.
- **The in-sample winner inverted out-of-sample: free-mass flexibility is a
  generalization liability (exp30).** LOEO (fit 4 epochs, predict the 5th): dyntrans,
  best in-sample (7.5%), is the WORST predictor (53.7% held-out, gap +46); rigid
  additive has the smallest gap (+11); loose-quad — 15 z-drifting params but PARAMETRIC
  masses — sits between (35.3%, +26). The discriminator is the mass parameterization:
  free NNLS masses absorb epoch-specific information and cannot predict an unseen
  epoch, while the transport structure itself predicts totals fine (|dlog M*| 0.06-0.16
  dex) — the shape overfits. The representational gate and the predictive emulator are
  different regimes: never promote an in-sample winner without a held-out test, and
  parametric masses (phase 3) are REQUIRED for prediction, not a refinement.
- **Run the cheap correlation pre-test before building the model extension — and
  believe it (exp30).** Before building event-triggered kicks, the planned go/no-go was:
  does the fitted per-galaxy migration speed (dyntrans alpha) correlate with MAH
  burstiness? It came back NULL (Spearman rho ~ 0). The full event model then confirmed
  it: kicks at the real-MAH bursts underperform the smooth self-similar clock at every
  threshold (10.3-12.1% vs 7.5%), monotonically worse with fewer events, no scatter
  reduction. Physical reading: halo-MAH-step timing is a poor proxy for STELLAR
  redistribution timing (dynamical-friction delays ~Gyr; relaxation continues between
  events). The pre-test cost seconds and predicted the outcome of a multi-minute build.
  FOLLOW-UP (lagged kicks, user hypothesis): letting events fire at t'=(1+beta)t_j
  genuinely helps (10.3->9.5%) with a coherent physical delay (median beta=0.37, IQR
  0.30-0.72 ~ dynamical friction) -- yet still trails the smooth clock (7.5%). So the
  smooth self-similar clock IS the delay-averaged merger clock: the delay physics is
  real, but per-event discreteness carries no in-sample signal beyond it. A negative
  model result can still validate the underlying physics through its fitted parameters.
- **When two model variants win in different regimes, complete the factorial before
  combining (exp30).** Transport (global clock + multi-scale width) won the population;
  envelope (dynamical clock + shared width) won BCGs/high-z. The intuitive "combine the
  winners" model (two-param clock + shared width) merely collapsed onto envelope (11.7%)
  because it inherited the WRONG shared ingredient — the binding difference was the
  width form, not the clock. Completing the 2x2 exposed the untested cell (dynamical
  clock + multi-scale width) as the true winner: dyntrans 7.5% vs transport 9.1% /
  envelope 11.3%, best at every epoch, 4 params, fitted alpha ~ 1 (migration timescale
  = cosmic time at deposition). Attribute wins to INGREDIENTS via a factorial, not to
  whole variants.
- **scipy nnls does not raise on non-finite design matrices; guard the basis (exp30).**
  An optimizer exploring extreme width params produced sig0-underflow x ratio-overflow
  = 0*inf = NaN in the basis; nnls silently returned garbage, one galaxy's NaN poisoned
  the population medians AND the nan-unsafe best-model selection picked the broken mode.
  Clip basis widths to finite ranges, check np.isfinite(A) before nnls (return inf loss),
  and make model-selection comparisons NaN-safe.
- **A "floor" in one metric is not a floor in another (exp29).** The free-mass NNLS
  minimizes L2 (relative SSE), so it is the best model in log-RMS but the WORST in
  max|rel| (18.5% honest, vs loose 9.9%) — free masses buy L2 by concentrating error into
  a worst-radius spike. Always state which metric a "floor"/"ceiling" refers to; report
  both the fitted objective and the reported metric.
- **Do NOT inner-mask the multi-epoch fit; the inner region holds most of the high-z
  mass (exp29).** Masking R<3 kpc was fine for a z=0.4-only emulator but wrong for the
  multi-epoch fit: high-z massive progenitors have Re<3 kpc, so the mask hides >50% of
  their stellar mass and buys a better metric by dropping the hardest region. Corrected
  (real MAH, ALL radii): loose-quad epoch-avg max|rel| is ~10% (ceiling ~2%), not the
  inner-masked 4.5%. Fit and evaluate over all radii for multi-epoch.
- **Standard mass QA: two bin sets (kpc + R_half) tell complementary truths (exp29,
  `mass_qa.py`).** In fixed *kpc* apertures the loose model reproduces M*(<10..<100 kpc)
  to ~1% but under-predicts the far outskirt M*(>50 kpc) by ~50% and M*(>100 kpc) by
  ~88% at z=2 -- because 50-100 kpc is 15-30 R_half out for compact high-z galaxies
  (R_half: 12.7 kpc at z=0.4 -> 3.3 kpc at z=2), i.e. the negligible tail. In *R_half*
  units the SAME model reproduces M*(<4Re) and M*(>2Re, >4Re) to a few % at EVERY epoch.
  So the model gets the profile *shape* (mass relative to size) right across cosmic time;
  the kpc-outskirt "failure" is about absolute radius in the far tail, not shape. Always
  report BOTH bin sets, and both the truth-vs-model value plot (makes the smallness
  explicit) and truth-vs-(Y-X)/X. `mass_qa.evaluate()` is the standard step, run in
  parallel with the profile max|rel| metric after every fit.
- **Pair the point-wise profile residual with INTEGRATED aperture + outskirt mass
  checks (exp29).** Cumulative aperture masses M*(<10..<100 kpc) are reproduced to
  ~0.01 dex even where the profile max|rel| is ~10% -- the cumulative is forgiving. The
  differential OUTSKIRT mass M*(>50 kpc) amplifies shape errors: the loose model
  under-predicts it by up to 0.31 dex (~2x) at z=2, worst for massive galaxies -- the
  sum-of-centred-Gaussians can't build the extended mass around compact high-z
  progenitors. Report M*(<R) at fixed apertures AND M*(>R_out); the outskirt is the
  sensitive diagnostic (small denominator + real shape deficiency).
- **The smooth DiffMAH fit curve flatters the deposition model; use the real MAH for
  honest numbers (exp29).** `dipfree_mah` fed the kernel the *smooth* DiffMAH fit, which
  erases merger-driven bursts (real single-step growth is 7-18% of total vs 2-3% for the
  fit) and provides ~99 evenly-spaced deposits. Swapping in the real de-dipped
  main-branch MAH (`peak_history`, running-max: keeps bursts, removes dips, ~60 gappy
  deposits) raised the best model's multi-epoch max|rel| from 4.4% to 6.1% (R>3), and
  removing the inner-3-kpc cut raised it further to 8.9% (real, all R); the per-epoch
  ceiling stayed low (~2%), so the *shape* is still representable — the *consistency*
  gap is what widens. The bursty/gappy real MAH is a less flexible basis for a smooth
  power-law width law, BUT it is the only input that carries merger *events* — so it is
  the prerequisite for an event-driven width/puff model (the parked "width set by the
  accretion event" idea). Report final numbers on the real MAH + all radii.
- **Free per-deposit mass does NOT relieve the multi-epoch tension; the consistency
  constraint itself is the binding limit (exp29).** Convex free-mass NNLS (each deposit
  a free non-negative mass) fits each epoch ALONE to 0.2% max|rel|, but one shared mass
  vector across epochs (a single consistent additive history) caps the JOINT fit at 12%
  (~60×). Parametric joint models beat free-mass-joint only by relaxing consistency
  (loose-zdep 4.5%) or adding width freedom (puff 7%). So the limit is the single
  consistent additive Gaussian-sum history, not the mass parameterization — reaching the
  per-epoch ceiling needs a non-additive primitive (mass that moves, not just adds).
  Isolate a constraint's cost with an alone-vs-joint comparison in the SAME method/metric
  (here both free-mass NNLS), not a cross-model table (objective/width-basis differences
  muddy it).
- **The physically-appealing DOF is not always the most effective one — let
  performance decide (exp29).** The puff-up model (one consistent history, mass
  frozen, only widths migrate) was the "principled" fix, but it underperformed: n=60
  epoch-avg max|rel| no-puff 9.1% → ratio-law puff 7.1% → diffusion-law 7.7%, vs the
  looser z-dependent-parameter fit at ~4.5%. Width migration with frozen mass is a
  *weaker* lever than letting the deposit mass-distribution (efficiency) vary with
  epoch — consistent with the param-trends finding that single-epoch fits
  de-concentrate early mass via the efficiency, not the width. The diffusion law
  (σ²+=κΔt) was nearly inert (κ→0). Don't pre-commit to the elegant constraint;
  benchmark it against the looser model and keep whatever fits.
- **Degenerate per-galaxy fits poison every population statistic built FROM the
  fitted parameters (exp30 phase 4).** The 45 per-galaxy 7-param fits are individually
  good (10%) but degenerate (b_early spans 3–44, z_c 1.5–48), so the median-θ
  predictor gives 55–82% error and θ–halo correlations are washed out. Population
  models must be refit jointly THROUGH the data (universal θ: 33.6%/30.6%), and
  conditioning slopes fit against the data, not against fitted θ. Use fitted-θ
  correlations only as a cheap structure-selection hint — and pair a liberal
  selection gate (p<0.05) with strict held-out (LOGO) promotion, which correctly
  rejected the pair that regressed on the diffmah config.
- **In-sample ≈ held-out for a low-dimensional shared model means the residual is
  CAPACITY, not overfitting — regularization cannot help (exp30 phase 4).** Universal
  θ: in-sample 32.3% vs LOGO 33.6%. Confirmed: the population-informed f(z) box fixed
  the z_c railing (identifiability) yet left LOGO unchanged/worse. When the
  generalization gap is ~0, only new per-object information can close the distance
  to the per-object floor, not priors/bounds on the shared parameters.
- **The smooth-DiffMAH-input penalty is a per-galaxy effect, not a population one
  (exp30 phase 4).** Per-galaxy fits pay ~2% for the smooth curve (exp29), but with
  ONE shared θ the DiffMAH input is BETTER (30.6% vs 33.6% LOGO): its smooth,
  evenly-spaced ~99-deposit basis suits a global parameter set, while the bursty
  gappy real MAH demands per-galaxy adaptation. Validate input equivalence at the
  level where the model will be used.
- **Summary features must be defined relative to the target epoch; z=0.4-anchored
  summaries made the MAH look useless at high z (exp31 -> exp32).** With t50/fz2
  anchored to z=0.4, the MAH's per-quantity gain decayed to ~0 by z=1 — but with
  epoch-matched features (t50(z_k)/t(z_k), Mh(t_k/2)/Mh(t_k)) the history improves
  every tier at every epoch and is the best regression overall (n=2397). Before
  concluding an input carries no information, check the features are aligned to
  the prediction epoch. Corollary (sharpened from the exp31 lesson): the better a
  regression per galaxy, the WORSE its population distribution — direct-epoch has
  the strongest regression-to-the-mean (plane fidelity 0.54 vs transport 0.19).
- **The aperture-horizon degeneracy: efficiency and deposition width multiply
  into one observable inside the aperture, and per-epoch pinning lets the fit
  delete mass geometrically (exp33/exp32).** The population transport fits
  (multi-epoch AND z=0.4-only) put 15%+ of all formed stars at widths of
  550-19,000 kpc — observationally identical to zero efficiency (measured:
  z=1-2 deposits 14% of the budget, 4% visible), physically absurd, enabled by
  renormalizing each epoch to the 150-kpc mass. Performance numbers are
  unaffected (the basins are observationally equivalent); parameter VALUES of
  all population transport fits must not be read physically. Fixes: prior-side
  (bound widths to the per-galaxy basin, alpha=1, lognormal f(z) peak — 7->5
  params) and data-side (normalize to an asymptotic total from CoG
  extrapolation, making the aperture fraction a fitted datum). A model whose
  selling point is mass conservation must not be normalized per epoch inside
  an aperture.
- **The frozen single-epoch feature set [DiffMAH(4)+c200c] is at its information
  limit; candidates from the recent experiments do not earn slots (exp33 iii).**
  Burstiness carries a REAL, shuffle-controlled but small signal (+1.3% CRPS —
  the first detection of merger content in mass prediction; exp30's
  residual-correlation gate saw none), and everything new combined gives +2.1%.
  Do not adopt burst: it needs the raw MAH and would break the portable,
  differentiable DiffMAH-input configuration for ~1%. Revisit only if a future
  model class shows merger-linked residuals.
- **The smooth DiffMAH fit is the better MAH ENCODING for statistical emulators;
  the "smooth curve flatters" lesson is specific to deposition-kernel models
  (exp33 iii vs exp29).** Replacing the DiffMAH shape params with raw-MAH
  summaries [logmh, t50, fz2] COSTS 8% CRPS (exp10 confirmed from the other
  direction). Feature parameterization quality matters more than feature
  "rawness"; test encodings per model class, don't transfer lessons across.
- **A conditional-mean predictor evaluated in TRUTH-target bins shows
  regression-to-the-mean as fake bin-wise bias; bin QA by a model INPUT
  (exp33).** Mode-3 CoG looked systematically off in truth-M* terciles
  (+0.04/-0.035 dex) yet is unbiased binned by halo mass, and its
  amplitude-pinned shape is good to <=0.01 dex — the apparent failure was the
  0.099 dex SHMR scatter (the information limit), amplified by truth-binning.
  qa's bins figure now takes bin_by (use halo mass) and shows raw AND
  amplitude-pinned residuals.
- **Compare 2-D population distributions with the energy distance (+ split-half
  floor), decomposed into location vs shape — a single relation statistic
  misleads (exp32).** |Δscatter| alone crowned transport-real the best plane
  model (0.169) while the full energy distance exposed a 16.6x-floor LOCATION
  bias it had hidden; centering (shape-only) showed all models are comparably
  far (3.5–4.8x floor) from the real 2-D population, because the energy metric
  is dominated by the ridge/marginals while Δscatter isolates the transverse
  diversity. Report BOTH: Δscatter answers "is the diversity around the relation
  right", energy/floor answers "is the whole 2-D population right", and the
  full-vs-centered split separates fixable amplitude bias from missing
  diversity. (2-D K-S rejected: not distribution-free in 2-D, least sensitive
  to exactly the spread differences at issue.) `hongshao/qa.py` tier 2b now
  reports all three.
- **A railed bound is not evidence of missing physics until a bounds-stress
  test says HOW the freedom would be spent (exp35).** log_s0/g railed at the
  loose physical box in every total-normalized fit; loosening the box (3.0->3.5,
  4->6) improved the loss 4.3% robustly — but the optimizer spent ALL of it
  re-deleting mass past the 500-kpc normalization horizon (z=1-2 visibility
  0.95->0.62, z<1 91% invisible even at 500) while the measured physics
  observables (massive-end aperture fraction, differential-deposition marks)
  did not move. A finite per-epoch normalization radius, wherever placed,
  leaves an invisible channel beyond it; the fix for a load-bearing rail is a
  structural DOF that decouples the constraint (here: two-channel deposit),
  not a looser bound. Run the one-fit stress test before reading a rail
  physically in either direction.
- **A measurable bias in a graduated utility was asserted into the record as
  physics (integrate_density; found 2026-07-13, affected exp33).** The
  `density_from_cog` -> `integrate_density` round-trip is an exact discrete
  identity when the shell areas use the same grid edges — but `integrate_density`
  derived areas from the annulus MID radii, biasing the reconstructed CoG up to
  ~0.24 dex at the steep inner radii. exp33's repr_fix demo MEASURED the bias
  (~0.11 dex inner), then wrote a tolerance assertion codifying it as expected
  behavior ("architecture D's structural handicap"), and the density
  representation was scored 21.1%/51.4% when its true score is 15.9%/28.7% — a
  tie with the baseline (verdict unchanged: ties, doesn't beat; but the "density
  cannot survive the transform" claim was false). Three lessons: (1) when a
  transform has an exact identity, assert EXACTNESS (1e-9), not "small enough" —
  a tolerance assertion can enshrine a bug as documented behavior; (2) a
  function only graduates when the library self-check pins its contract —
  `integrate_density` was never exercised by `profile_emulator`'s `__main__`,
  so the bug rode along silently; (3) before attributing a systematic residual
  to a method's "structural handicap", check the identity the method is supposed
  to satisfy on synthetic input. (The transport/deposition models — exp29/30/32/
  35 — are unaffected: they build model CoGs analytically from the Gaussian
  deposit basis and never call `integrate_density`.)
- **Per-galaxy dex scatter cannot see regression-to-the-mean; the observational
  planes can (exp31).** Per-quantity LOGO regression TIES the physical emulator on
  aperture dex scatter (~0.1) while predicting population distributions that are
  badly too tight and too shallow (M(<30) vs M(50-100) plane: truth slope 1.88 /
  scatter 0.206 -> regression ~1.05/0.05; transport 1.15/0.108). Any emulator meant
  to be compared with observations must be scored on DISTRIBUTION fidelity
  (relation slope/scatter in the observational planes), not only per-object error —
  `hongshao/qa.py` tier 2b is the standard for this.

## Workflow

- Multi-hour compute jobs launched as harness background tasks were repeatedly
  killed mid-run (whole process group, silent, no OOM); plain `nohup uv run ... >
  log` survived for ~1 h jobs, and the robust pattern for longer chains is
  launch-and-`disown` from a foreground shell (orphan the process), with a
  persistent log tail for milestones and `caffeinate -im` against idle sleep.
- Background `uv run` commands buffer stdout through pipes; redirect to a file
  and read that, or block on the process, rather than polling a pipe.
- `np.trapz` is gone in NumPy 2 — use `np.trapezoid`.
- **scipy's Nelder-Mead silently freezes parameters started at (or near)
  zero, and the result looks exactly like a measurement that the parameter
  does nothing. It has produced a false null here twice. Use
  `hongshao.fitting.initial_simplex` for any new fit.** scipy builds its
  starting simplex by perturbing each parameter RELATIVELY, by 5% of its own
  value (`nonzdelt = 0.05`), with one special case: a parameter that is
  exactly 0.0 gets a fixed `zdelt = 0.00025` (checked against scipy 1.13.1).
  Nelder-Mead only explores the space its starting simplex spans, and along a
  direction where the loss barely changes it contracts rather than grows — so
  a parameter handed a span of 0.00025 in a surface where it acts at order
  one comes back where it started. In exp47 three shape slopes were started
  at exactly 0 (deliberately, so the new component nested the previous model
  as a special case) and came back at ~0.01: a property of the start point,
  not of the data. **The bug is broader than "exactly zero"** — because the
  step is purely relative, any parameter starting small in absolute terms but
  acting at order unity gets a meaningless span with no special case needed,
  which is how exp48's six conditioning slopes (sitting *near* zero) hit the
  same wall. The fix is a relative step with an absolute FLOOR,
  `max(0.05|p0|, floor)`, with the floor settable per parameter because
  parameters have different natural scales (exp47 needed 0.3, 0.6 and 0.15
  for three different groups). **Two traps in diagnosing it:** (1) the failure
  is not a crash or a hit iteration limit — scipy reports `success=True` and
  converges on the wrong answer, identically at every budget from 400 to 4000
  iterations, so a longer run does not rescue it; (2) it is not reproducible
  in a simple two-parameter quadratic toy, where Nelder-Mead escapes anyway —
  it needs several well-scaled parameters alongside the weak ones, which is
  why `hongshao/fitting.py`'s self-check uses five. Corollary for the record:
  supplying an explicit simplex CHANGES what a fit returns, so never switch
  an already-reported fit to it without re-running.
- **Identifiability results are CONDITIONAL on the model being right, so audit
  the mechanism before spending a week measuring parameter degeneracies.**
  exp49 (2026-08-20) ran a careful identifiability program on the adopted
  kernel -- residual-Jacobian SVD, widened multi-start fits, a properly profiled
  `g` curve -- and answered its question: `g` = 4.352 +/- 0.003, one coordinate
  of a degeneracy in which all six base parameters correlate at |rho| >= 0.83
  (effective rank 7 of 12), worth 0.064% of the loss. Then exp52 found the
  kernel's deposition mechanism is misspecified, which puts the remaining stages
  on hold because **degeneracy structure measured in a misspecified model need
  not survive fixing the model.** The mechanism audit cost an afternoon; the
  identifiability program was a week of compute. Do the cheap structural check
  first.
- **A model can reproduce an observable well while getting the MECHANISM
  structurally wrong, and nothing in the objective will reveal it — because the
  objective only ever sees the summed profile.** exp52 (2026-08-20): the
  adopted kernel fits the curves of growth to ~10-15% at five epochs, yet
  **96.5% of its M\*[50,100 kpc] growth between z=0.7 and z=0.4 is
  redistribution of stars it already had**, not new deposition, and its own
  deposition supplies only ~11% of the z=2 -> z=0.4 mass growth (the other 89%
  is manufactured by the M(<500 kpc) normalization pinning the model to the
  data). The established picture for massive galaxies is the opposite: late
  growth is ex-situ, driven by continued minor mergers. **Rule: for any model
  with internal machinery, periodically audit HOW it produces the observable,
  not just how well.** Concretely, decompose the change in the observable into
  the model's own mechanisms and check each against the physics — here that
  meant splitting aperture growth into transport / new deposition /
  renormalization, which took an afternoon and found a defect two experiments
  of fit-quality work had only seen the symptoms of (the compact-galaxy central
  deficit: with deposition off, the only way to fill an outskirt is to empty the
  centre).
- **Corollary: watch for observables the model does not actually predict.**
  This kernel never predicts total stellar mass growth — M(<500 kpc) is pinned
  to the measurement — so agreement there is guaranteed by construction and is
  not evidence of anything. Know which of your matches are earned and which are
  imposed.
- **A SLICE IS NOT A PROFILE, and the slice is always the cheap thing that
  looks like an answer. Vary one parameter with the others FROZEN and you
  measure that parameter's mechanical role; vary it with the others
  RE-OPTIMIZED and you measure what the fit would actually do. In a degenerate
  model these differ in magnitude and can differ in SIGN.** This cost two wrong
  claims in one day (2026-08-16, exp49 stage 2), both stated confidently from
  slices: (1) "there is a genuine minimum at `g` ~ 4.07" — the profiled answer
  is 4.35, and the slice's own value was worse than the fully optimized
  bounded fit, which should have been the tell; (2) "`g` is the lever on the
  compact-galaxy defect" — in a frozen slice the mass fraction inside 5 kpc
  roughly doubles per unit of `g`, but under profiling the compact-galaxy
  central-mass deficit gets monotonically WORSE with `g` (-35.7% at g=3.0 to
  -40.5% at g=5.5), because `log_rc` and the rest compensate and over-cancel.
  **The sign flipped.** Rule: never quote a one-parameter response as a
  property of the model unless the other parameters were re-optimized, and say
  which one you did. Corollary measured the same day: the loss optimum
  (g = 4.25-4.35) and the observable of interest move in OPPOSITE directions,
  so this is the "rank by judge, never by loss" lesson with a number on it.
- **Nelder-Mead was the wrong optimizer for this project's losses, and a
  parameter "pinned at its bound" is often just a stalled optimizer.**
  Benchmarked on the real 1ch-mof loss (100 galaxies, z<=1.5, displaced start,
  4000-evaluation budget): L-BFGS-B on finite-difference gradients reached
  0.152764 in 1170 evaluations / 38 s, while the plain Nelder-Mead in use
  exhausted its budget at 0.159980 — 4.7% worse — in 131 s. BFGS, trust-constr
  and `Nelder-Mead(adaptive=True)` all agreed on 0.152764 to six decimals, so
  that is a real minimum and not one method's quirk. **The `adaptive=True`
  flag alone closes most of the gap and costs one keyword.** Gradient methods
  apply because the loss is smooth, which was measured rather than assumed: at
  the solution NONE of the kernel's `clip`/`max` guards are active (0/9900
  deposit events, 0/39600 event-epochs, 0/100 galaxies, box penalty 2e-8), and
  central differences are stable from a step of 1e-2 down to 1e-7. **The
  corollary that matters most:** every method that reached the minimum put `g`
  at ~3.54 and `log_rc` at ~2.47, both interior; every method that failed to
  reach it pinned one or the other against its bound (3.9999, 2.9999). Before
  interpreting a railed parameter as physics, re-fit with a converging
  optimizer. Use `hongshao.fitting.minimize_loss`. Caveat, not yet closed: the
  full-sample refit still ends with `g` at its bound, but it started from the
  already-railed adopted theta and used the one-sided box PENALTY, whose
  derivative vanishes exactly at the bound — so a gradient method can report
  convergence while stuck at that kink. The clean test is real bounds plus
  displaced starts on the full sample.
- **A "this galaxy failed" marker that gets dropped rather than penalized
  makes a worsening fit look better.** `hongshao.objective.reduce_galaxies`
  drops non-finite per-galaxy losses before averaging, which is right for
  reporting and wrong for driving a fit: measured on the full sample, five
  failed galaxies out of 2397 moved the population loss from 0.149704 to
  0.149470 — DOWN. So the scoring module returns NaN and counts it, and
  anything driving an optimizer substitutes its own large sentinel (exp38
  uses 4.0, ~10x a good loss) BEFORE reducing. Keep the sentinel in the fit
  wrapper, not in the scoring: a reported population mean silently containing
  a few 4.0s cannot be told apart from a model that fits badly everywhere.
- **Verify a refactored numerical routine against the original with a
  RELATIVE tolerance, chosen after looking at the range of values it
  produces.** Checking `hongshao.objective` against exp48's reference across
  182 configurations, a fixed 1e-10 absolute threshold flagged a failure that
  was pure floating-point summation order (`einsum` in one, a plain `sum` in
  the other). The losses span 0.05 to ~7e5 across those configurations —
  comparing surface densities with a fractional residual diverges by design,
  which is itself one of exp48's results — so one absolute number cannot be
  right at both ends. All 182 agree to 5.9e-16 relative.

## Figures / QA presentation

- **This machine's matplotlibrc has `text.usetex: True`; raw `<`, `>`, `|`, `%`
  in matplotlib text render as ¡, ¿, em-dash, or eat the rest of the label (%
  is a LaTeX comment) — silently, no error (found 2026-07-13 in every qa.py
  figure).** Route figure text through `hongshao.qa._tex()` (math-wraps <>|)
  and `_pct()` (usetex-aware percent). Also: an axes with a log scale and NO
  plotted data crashes `tight_layout` ("Data cannot be log-scaled") — give
  all-NaN panels explicit finite limits. And the cividis ramp makes adjacent
  epochs indistinguishable — `qa._zcolors` now uses a distinct ordered
  Okabe-Ito subset (user request: visibly distinct epoch colors).

- **The fit-space choice trades inner for outer accuracy in profile emulators
  (exp37, 2026-07-14).** Head-to-head on the same folds and cores, the
  log-density-space product's MEAN prediction is +6-7% biased inside 10 kpc
  where the log-CoG-space product is +1.5-2% (and per-galaxy worst-radius
  errors ~3 points worse), while the density space wins the shape/outskirt
  metrics. Three structural reasons, all measured: (1) differencing amplifies
  where the profile is steep — in CoG space the inner cumulative is a direct,
  smooth target; in density space it is a sum of a few steep shells (small
  denominator, compounded errors) — the exp26 lesson operating in reverse;
  (2) pinning the total to the anchor EXPORTS shape error to the centre: any
  outskirt deficit is compensated by a global rescale that biases the small
  inner cumulative (K=3->8: outskirt bias +9.9%->+0.7% dragged the inner bias
  +7.2%->+4.2% in lockstep); (3) mode-budget competition: pooled density
  shapes are dominated by decades of outskirt dynamic range, so shared PCA
  modes underserve the inner region — the cumulative compresses that range,
  which is why K=3 sufficed in CoG space. Fix directions: segment-pinned
  reconstruction (kpc-annulus masses as explicit coefficients, shape only
  distributing WITHIN segments) or a hybrid (CoG-space mean + density-space
  draws). Corollary: present a model's QA in the space it was FIT, alongside
  the integrated deliverable.

- **A monotone model curve cannot have all its parts AND its total
  median-unbiased at once; choose where the inconsistency lands (exp37 block
  product, 2026-07-14).** Summing median-predicted lognormal components
  undershoots the median-predicted total (measured: +2.8% at z=0.4 -> +6.6%
  at z=2 of rescale); a uniform rescale smears that deficit onto the
  well-determined inner blocks (+5.6% inner bias), while trusting the raw sum
  biases the total (-3.4 -> -6.9%). The fix is to allocate the deficit in
  proportion to each block's EXPECTED median-vs-mean gap,
  B_j (e^(sigma_ln^2/2) - 1) from the fitted heteroscedastic sigma — mass
  lands where the uncertainty lives, tight blocks stay at their direct
  predictions. Corollary for generative layers: drawing the ANNULUS masses
  (block representation) reproduces near-empty high-z annuli that per-radius
  log-CoG or log-density draws cannot, while keeping every draw monotone.

- **A component added to fix one region can re-balance the REST of the
  model into breaking an out-of-model test — measure every judged test at
  every operating point, and test the mechanism before trusting it
  (exp38 stage 3).** Adding a compact core channel fixed the inner-mass
  deficit (M(<5) -11.7% -> -2.8%) and improved held-out shape at every
  epoch, but broke the differential-deposition pass (0.39/0.12 ->
  0.53/0.19 vs data 0.37/0.11) and re-opened the outskirt overshoot. The
  obvious mechanism — late deposits feeding the core — was tested by
  switching core formation off after a fitted cosmic time: NOTHING
  changed (a clean null), so the pressure was the outer kernel
  re-balancing itself once any core absorbed the inner mass. Corollary:
  a fix's side effects live in the parameters it FREES, not only in the
  component it adds; and a mechanistic story about a trade is a
  hypothesis to test (one nested parameter), not a conclusion. The
  kernel-spine + residual-dressing model reached the statistical wall at
  every epoch (15.6 -> 10.8% pinned shape), but the SAME dressing on a flat
  spine (train-median log CoG, no kernel, no per-galaxy physics) tied it to
  within +-0.4 points, alternating sign. The kernel's residual structure is
  feature-reachable — phase 0 had already measured the shared wall (~50%
  residual variance) — so the hybrid's accuracy is the dressing's, not the
  spine's. The guard was pre-registered and cheap (one extra OOF pass);
  without it the hybrid's wall-level numbers would have been credited to
  the physics. Corollary: a spine normalized to a measured datum (M(<500))
  leaks real totals into the draws — its growth-plane score (0.3x floor)
  is not comparable to a feature-only emulator's (1.0-1.3x).
- **A joint multi-epoch constraint can be the lever that fixes a
  single-epoch residual (exp36 multi round).** The low-mass outskirt
  overshoot (+0.12/+0.13 dex) did NOT fall when the split amplitude was
  conditioned on c200c/fz2 at z=0.4 alone (the forgiving CoG loss cannot
  see an outskirt density overshoot; loss moved 0.1544 -> 0.1543), but the
  SAME parameters under the joint 5-epoch fit cut it to +0.03/+0.07 — the
  cross-epoch consistency requirement is what forces the split to spend its
  freedom on the outskirts. Corollary of the exp29 "fit the thing that
  depends on the unknown" lesson: if a residual is invisible to the loss,
  add the CONSTRAINT that prices it, not just the parameter that could fix
  it.
- **A nested model that loses to its own special case is an optimizer
  failure, not a result (exp36).** Warm-starting the 13-param two-channel
  slope fit from exp35's railed single-width theta left it stuck at exp35's
  basin (0.1756) — WORSE than its nested 8-param global fit (0.1590), which
  is impossible at a true optimum. Warm-start nested models from the best
  fitted SUB-model's basin (+ zeros for the new freedom), and use the
  nesting relation as the convergence check: fitted(superset) <=
  fitted(subset) or refit.
- **A frozen-base fit is a DIAGNOSTIC, not a model; and a mechanism whose
  numbers "match" is still just a hypothesis (exp39).** The exp38 core
  channel's physics break was blamed on its inherited power-law tail
  (8.3% of core mass beyond 30 kpc — a plausible story whose arithmetic
  matched the overshoot). The six-form shootout falsified it: a
  zero-leakage Gaussian core breaks the differential test EXACTLY like
  the leaky Moffat (0.54/0.19 vs 0.53/0.18; all six forms within noise
  on loss AND on the failure). The breaker was the freed outer kernel
  re-balancing, isolated by refitting with the kernel PINNED at its
  adopted theta: the core alone then halves the inner deficit with the
  physics untouched by construction. But that frozen fit did NOT
  graduate (user): pinning is an ad-hoc restriction with no physical
  mechanism — the same structure at its own joint optimum breaks the
  physics, so the "improvement" exists only under the freeze. Use the
  pair of fits as a measurement instrument — freed-base measures the
  trade, frozen-base measures the component's standalone contribution —
  and let neither substitute for a model that earns its behavior at a
  true optimum. COMPLETION (retention x inner-aware cell): the failure
  is OBJECTIVE-driven, not component-driven — {6 core forms} + {2
  retention modes} under the inner-aware objective ALL fail the
  differential test in the same 0.48-0.56 band (data 0.37/0.11),
  including mass-conserving retention inside the kernel's own transport
  that targets the measured clock-drain cause and delivers the
  program's best held-out accuracy (14.8% avg) plus the only z=2 inner
  gain. When every mechanism handed to an objective fails an out-of-
  model test the same way, stop building mechanisms: the tension is
  structural between the model family and the data, and the next move
  is a different family or constraint, not another component. (The
  physically-dressed variant — retention conditioned on deposit
  boundness rc0/R200 — was a measured null, b = 0.10: physical
  plausibility does not create signal.)
- **A railed value's meaning is scope-dependent — stress + physics
  every time, never pattern-match the number (exp40).** g = 4.00 at
  the box marked the PATHOLOGICAL second basin in the 5-epoch fit
  (better loss, worse physics; the standing watch item) — yet the SAME
  railed value in the 4-epoch late-start fit is benign: the stress
  test settles interior (g = 4.37) for a 0.1% gain with the physics
  unchanged-to-better. Had the watch item been applied as a
  pattern-match ("g=4 -> bad basin, discard"), the branch's best
  result would have been thrown away. Corollary (the fit-scope
  finding itself): WHICH epochs a joint fit constrains is a physical
  modeling choice, not bookkeeping — excluding the dissipative z>=1.5
  era removed half the core deficit at BETTER physics, and the
  late-anchored transport extrapolated up to z=2 better than the
  z=2-anchored fit fit it (0.21-0.23 vs 0.18, data 0.23). A model
  should be fit where its assumptions hold and judged by
  extrapolation into the regime where they don't.
- **A cross-generation "input X is worse" verdict can be a property of the
  OLD model form — re-run the input test when the form changes (exp42).**
  "The real MAH loses 3-5 held-out points to the smooth DiffMAH curve in
  population-shared fits" held across three Gaussian-deposit generations
  (exp30/32) and was carried as settled. Re-run on the adopted Moffat
  kernel, it REVERSED: the real input ties at z<=1.0 and wins at z>=1.5
  (13.6 vs 15.4 at the z=2 extrapolation), from a fully interior optimum —
  the heavy tail supplies the shape freedom the bursty basis used to
  demand per-galaxy. Bonus: the g=4.0 rail (stress-tested benign but
  standing) simply vanishes on the real input — a rail can also be a
  property of the INPUT basis, not the model or the box. The verdict that
  survives is conditional, not absolute: "DiffMAH is the better input FOR
  a shared theta on a wingless deposit." Record which ingredient a
  comparison was conditioned on, and re-test when that ingredient changes.
- **A suite of single-epoch metrics is structurally blind to cross-epoch
  incoherence — no amount of them adds up to one (exp44 stage 3; now
  `qa` tier 2c).** Every tier in the standard QA set scored one epoch at
  a time, so a model whose per-epoch marginals are all correct could
  evolve individual galaxies arbitrarily and score flawlessly. Measured:
  the adopted stochastic layer held drawn galaxies at rank correlation
  +0.97 between their z=0.4 and z=2.0 sizes where the truth sits at
  +0.33 — a large defect that survived adoption unmeasured, and that the
  planes CANNOT see (swapping the layer for a correctly-decorrelated one
  moved every plane by <= 0.1, necessarily, since the two share their
  single-epoch marginals by construction). Corollary for building the
  metric: the interpretable rank-coherence number is the sharp
  diagnostic; the 2-D energy distance is blunt here because it is
  dominated by the marginals the two models share. When a model predicts
  a SEQUENCE, at least one statistic must span the sequence.
- **When two independent model families measure the SAME latent constant,
  it is a property of the data — use the disagreement-that-didn't-happen
  as the control (exp44 stage 2).** The kernel's per-galaxy individuality
  decorrelates across epochs as AR(1) with rho = 0.56-0.66; the exp37
  statistical emulator's residual latent measures rho = 0.62 with the
  identical estimator. The obvious sceptical reading of the kernel number
  — "that is just its one-consistent-history constraint showing" — is
  falsified by the agreement, because the emulator has NO consistency
  constraint (independent per-epoch cores) and would have to show HIGHER
  persistence under that hypothesis. Cross-family agreement on a fitted
  constant is stronger evidence than either fit alone; look for a second
  family that lacks the structure you suspect, and quote it.
- **Separate "the signal decays" from "my estimate of it is noisy" before
  believing a decay (exp44 stage 2).** A per-galaxy quantity fitted
  independently at each epoch decorrelated to +0.09 across the full
  baseline — which could equally have meant a persistent quantity
  measured badly. Fitting each epoch on two DISJOINT halves of the
  radial grid gives a split-half reliability (Spearman-Brown corrected
  to full length); dividing the cross-epoch correlations by
  sqrt(rel_a rel_b) disattenuates them. Here reliability was 0.96-1.00,
  so the decay was real and the correction was a no-op — but that null
  correction is exactly what made the decay claim safe to publish, and
  it cost one extra pass. (Caveat to state: adjacent radii on a smooth
  cumulative profile are redundant, so a radial split-half BOUNDS
  estimation noise rather than fully characterizing it.)
- **A flexible model can EXTRAPOLATE a metric while being physically
  wrong — judge extrapolation jointly on the target metric and the
  out-of-model physics tests (exp43).** The 2ch-exp kernel fitted to
  z=0.4 alone posted the best extrapolated z=2 shape number in the whole
  ladder (12.7% pinned max|rel|, beating every fitted model) — while its
  inner masses SIGN-FLIPPED (+3 to +8%) and the differential test
  nearest its own fitted epoch broke at 0.54/0.20 vs data 0.37/0.11: the
  split + a degenerate q=1.41 mimicked the shape trend for the wrong
  reasons. The residual-vs-radius view exposed it instantly (a +30-40%
  core overshoot canceling outer deficits under a worst-radius metric).
  Meanwhile the HONEST failure (1ch at z04: q unidentifiable -> 0,
  extrapolation degrading monotonically to 44%) is the diagnosable one.
  Rank extrapolation candidates by metric + physics + residual anatomy
  together; a single scalar invites accidental winners.
- **Score a generative model with population metrics, never paired
  ones — truth-anchored apertures punish exactly the diversity a draw
  is supposed to add (exp41; resolves the exp33 Re-coordinate
  puzzle).** qa's Re-plane convention applies the TRUTH half-mass
  radius to the model — right for a mean prediction, structurally
  wrong for a draw that is (correctly) independent of the specific
  truth realization: the 1-D layer read as 2.7x the plane floor
  paired, but 1.0x — AT the floor — when both populations use their
  OWN sizes. exp33's "generative draws fail only in Re coordinates"
  was the same artifact, unexplained for two weeks. Two corollaries:
  (1) before iterating on a model to fix a failed metric, check
  whether the metric is even well-posed for that model class — the
  Re-preserving refinement was built, run, and rejected before the
  rescoring exposed that its target problem did not exist (though its
  failure is what raised the question: pinning the size destroyed the
  kpc gains, revealing the layer's diversity is mostly SIZE diversity,
  the exp38 self-similarity from the model side); (2) a paired metric
  is still the right tool for the MEAN prediction — keep both, label
  which is which.
- **`np.interp` CLAMPS outside its grid, so `interp(0.0, R, cog)` is not zero
  — it is the mass already inside the innermost aperture (exp52, found in
  exp53).** exp52's growth decomposition built the truth term as
  `interp(r_out) - interp(r_in)` and the model term analytically, where
  `enclosed(0.0, ...)` really is 0. For every `r_in = 0` row that silently
  divided the model's APERTURE growth by the truth's ANNULUS growth. It was
  invisible because it is a 27% effect that only touches the aperture rows:
  the annulus rows (`r_in > 0`) reproduced perfectly, which reads as
  confirmation rather than as a clue. The headline moved from "the model grows
  M(<10) at 57.3% of the truth's rate" to 72.6% — the finding survived, its
  size did not. **Two habits this argues for.** (1) When model and data go
  through DIFFERENT code paths to the same quantity, the two paths are a place
  to put an assertion, not a place to trust symmetry — here, asserting
  `truth_enclosed(0) == 0` would have caught it instantly. (2) When one row of
  a table reproduces an independent number exactly and a neighbouring row does
  not, the discrepancy is a lead about what differs BETWEEN the rows, not
  noise to be explained away in the row that disagrees; chasing "why does
  M(<10) disagree" through the model (deposit ceiling, analytic-vs-interpolated
  radial operator — both tested, both innocent) took three probes, while
  asking "what do M(<10) and M[50,100] not share" would have pointed at `r_in`
  first.
- **A boundary test can PASS with everything frozen and FAIL once refitted —
  and the gap between the two is the whole result (exp53).** exp52 predicted
  that removing the kernel's transport term would improve central masses.
  Switching transport off at the adopted theta and changing nothing else, it
  does, enormously: the compact-galaxy `M*(<5)` bias runs -43.6% -> **+13.6%**.
  Re-fitting the other parameters with transport off, it does not: -43.4% ->
  **-39.4%**, four points out of forty-three. The two answers differ by 57
  points and in SIGN. The model does not respond to losing transport by
  depositing more centrally; it flattens its efficiency window (sig 0.22 ->
  2.68) and shrinks its deposits (log R50 3.27 -> 1.73), buying the outskirt
  fit back a different way and leaving the centre alone. **A mechanism that is
  real in a model's internal bookkeeping is not thereby the CAUSE of that
  model's error** — freezing everything measures the former, only refitting
  measures the latter. Corollary worth the extra cost: run BOTH and report them
  side by side through the identical scorers. The disagreement is not an
  embarrassment to be resolved; reporting only the refit would have hidden that
  exp52's bookkeeping claim is perfectly correct, and reporting only the frozen
  version would have "confirmed" a prediction that is false.
- **When several model families all rail on the same parameter, widen the box
  once and see whether they rail AGAIN — "unidentified" and "boxed" look
  identical from a single fit (exp53).** Three of four deposit families hit
  `log_R50` = 3.5. Widening to 5.0, the Moffat escaped to a finite interior
  3.626 while sersic/expo/gauss all re-railed at 5.0 — and their losses
  converged to within 1e-4 of each other, because "put every deposit at the
  ceiling" is the same model whatever profile shape is hung on it. Two things
  follow. (1) The right verdict was not "the box chose the answer" but
  "nothing chooses the answer": the objective does not constrain a light-tailed
  deposit's scale from above at all. A loss-gain threshold alone mislabels
  this — one cell gained only 0.37% and was flagged benign while still sitting
  on the new bound, so the check must test whether the parameter ESCAPED, not
  just whether the loss moved. (2) It supplies a mechanism for the standing
  exp48 verdict that "the profile does not matter": the profile stops
  mattering precisely because the scale runs away, so that verdict is
  conditional on an unpinned scale rather than a statement about profiles.
- **Check whether a "fixable defect" in the record has already been fixed
  before building the fix (exp53).** The 2026-08-20 handover carried an
  action item — Sersic was eliminated in exp38 by an n-scale degeneracy
  "needing an R50 reparameterization", a fixable numerical artifact, so
  exp53 should supply the reparameterization and re-run it. exp53 did, and
  reported it as a vindication. **exp48 step C had already built
  `sersic_r50` and already re-fitted it inside the full multi-epoch
  kernel**, where it came LAST on the judge. The work was not wasted — it
  reproduced exp48's verdict under a different objective, which strengthens
  it — but it was mis-labelled as new for a whole session, and the real
  (narrow) contribution was buried. Two habits: (1) an inherited "nobody
  ever fixed X" is a claim about the record, so grep the record for X before
  acting on it, especially when the handover cites the ORIGINAL rejection
  rather than the most recent test; (2) when an experiment reproduces a
  prior verdict under different conditions, say so in those words — a
  confirmation across objectives is a real and reportable result, and
  dressing it as a discovery costs the credibility that the confirmation
  actually earns.
- **State the QUANTITY and the BINNING VARIABLE whenever you quote a trend with
  mass — the apparent sign depends on both (exp53).** exp53 reported that its
  deposition-only models over-fill low-mass galaxies' outskirts and under-fill
  massive ones, measured as the `M[50,148]` ANNULUS in STELLAR-mass terciles.
  The user read the standard QA figures — which bin by HALO mass and plot
  CUMULATIVE apertures — and saw the opposite: `M(<148)` under-estimated at low
  halo mass, over-estimated at high. Both were correct. The annulus and the
  aperture disagree because the inner and outer errors have opposite signs, and
  the two together say what neither says alone: the models have a RADIAL
  DISTRIBUTION error that flips sign with halo mass, not a mass-budget error.
  Two habits follow. (1) A "trend with mass" is under-specified until the
  quantity and the binning variable are both named; a reader comparing against
  the project's standard figures will use THEIR conventions, not yours.
  (2) When a bespoke analysis and the standard QA figures appear to disagree,
  that is a lead worth chasing rather than a discrepancy to reconcile away —
  here the disagreement located the actual defect, which neither view had
  identified on its own.
- **Regression to the mean in binned profile QA is RADIUS-DEPENDENT, and a
  blanket warning about it is wrong in both directions (exp53).** Binning by
  the TRUTH stellar mass and comparing against a conditional-mean model tilts
  the residual by `(slope - 1)` dex per dex of bin separation, where
  `slope = r * sigma_model/sigma_truth`. Measured on the exp53 kernel at
  z = 0.4: the slope is **0.79 at `M(<5 kpc)`** — a severe artifact, and the
  stellar-mass and halo-mass binnings there disagree in SIGN (+0.022 against
  -0.016) — but **1.03 at `M(<148)`**, where it is negligible because the
  `M(<500)` normalization pins the total to the truth per galaxy and leaves
  almost no scatter to dilute (`r` = 0.997). So the caution belongs on the
  INNER profile only; applying it to the outer profile would have wrongly
  discounted a real 9-percentage-point halo-mass tilt that survives in both
  binnings. **The general habit**: before invoking regression to the mean as a
  reason to distrust a binned trend, compute the model-on-truth slope for THAT
  quantity — it is two lines, it bounds the artifact exactly, and a
  normalization that pins the quantity per object can defeat the mechanism
  entirely.
- **A per-object normalization can hide a factor-2 error in the thing the model
  is supposed to predict — audit what it supplies (exp53, user's question).**
  Every kernel in this program pins `M(<500 kpc)` per galaxy per epoch to the
  truth, and every judged metric operates on the POST-normalization profiles.
  So nothing scored the model's own mass ASSEMBLY. Measured: the deposit
  weights sum to 1 by construction, so 100% of the stellar mass and 100% of the
  mass growth come from the pinned value, and the incumbent's own accumulation
  grows x1.25 from z=2 to z=0.4 where the truth grows **x2.65** — the pinning
  silently supplies a factor of **2.1**, more than half the growth. Every
  deposition-only variant is within 0.03-0.11 dex of the truth's assembly, a
  3-11x smaller correction, which is the strongest argument for them in the
  whole experiment and was invisible to all thirteen judged metrics.
  **The habit**: for any normalization, pinning, or per-object rescaling, write
  down explicitly what it SUPPLIES and then score that quantity separately. It
  is usually free — the un-normalized prediction is already computed — and it
  is exactly where a model can be most wrong while looking fine, because the
  normalization is applied before anything measures it.
- **A benchmark and the thing it benchmarks must be scored on THE SAME
  GALAXIES — one corrupt object overturned an experiment's headline
  (exp54, 2026-08-23).** `score_A` is `rms[(model - truth)/sigma_A(z)]`, where
  `sigma_A(z)` is Stage 1's best halo-only regression scatter. Stage 1 measured
  it on all 2397 galaxies; Stages 3.1/3.2/3.3 measured the model on stratified
  subsamples of 300/240/1199. **Row 181 fell in the first set and none of the
  others** (it is an odd row, and those subsamples take every second galaxy).
  Its measured `M*(<100 kpc)` collapses from 10^11.712 at z=0.4 to a flat
  ~10^7.96 at all four higher-redshift epochs — a broken cross-match, 2.5 dex
  below the sample's 0.1st percentile. That single object inflated `sigma_A` by
  **18/15/12/8% at z = 0.7/1.0/1.5/2.0**, and the model was never charged for
  it. Correcting it moved `score_A` from **1.128/0.907/0.899/0.929/0.970** to
  **1.131/1.100/1.059/1.058/1.050**: the adopted model does not beat the
  halo-only regression at any epoch, where the published claim was that it beat
  it at four of five. **Three habits.** (1) When a metric is a RATIO of two
  numbers computed by different scripts, assert that both were computed over
  the same object set — a one-line `set` comparison, cheap and decisive.
  (2) Reject physically impossible measurements AT SOURCE, in the feature
  builder, loudly, so every consumer inherits the rejection; `MAX_BACKWARD_DEX`
  is now a named constant next to the target it guards. (3) A benchmark
  degraded by outliers flatters whatever it is compared against, so an
  unexpectedly GOOD result is as much a reason to audit the denominator as a
  bad one is to audit the numerator.
- **Classify which code a fix invalidates from what the code READS, not from a
  list you write by hand (exp54, 2026-08-23).** After fixing the concentration
  excess I judged that 21 of 45 factorial cells consumed it — the E3 and S3
  cells — and re-ran only those. `model.r50_of` contains
  `scale = halo.r200c if spec.size == "S2" else halo.r200c / halo.c_eff`, and
  the `else` branch belongs to **S2a**, which I had reasoned about as if it
  were absent from the factorial. The true count was 33 of 45. The error was
  caught within three cells because the re-run deliberately included CONTROL
  cells asserted to be unaffected and required them to reproduce the cached
  loss exactly: `moffat-E5-S2a` came back at 1.7476 against a cached 1.7385.
  **The habits**: (1) express "which cells does this touch" as a predicate over
  the model spec, in code, next to the thing it describes; (2) any partial
  re-run must include controls that are asserted NOT to move and must refuse to
  merge if they do — a seeded sample and a seeded multi-start make the
  assertion exact, and an exact assertion is what makes it useful.
- **Never wrap an import in a bare `except` (exp54, 2026-08-23).** A
  `try: import scoreboard ... except Exception: d = None` fallback silently
  turned off the entire baseline-refit section of a diagnostic and printed a
  misleading "cache absent" message instead. The real error was
  `module 'scoreboard' has no attribute 'CACHE'`: three experiments in this
  repository ship a module named `scoreboard`, and which one wins depends on
  import order. **Fix both halves**: let the exception surface, and import a
  same-named sibling module BY FILE PATH via `importlib.util.spec_from_file_location`
  rather than by name. Check for name collisions with
  `ls experiments/*/<name>.py` before writing `import <name>`.
- **`setsid` does not exist on macOS, and a `ps` snapshot taken one second
  after launch does not prove a job started (exp54, 2026-08-23).**
  `nohup setsid wrapper.sh &` returned cleanly, `ps | grep -c` reported one
  match, and nothing was running — the match was the dying launcher. Two
  separate long jobs were believed to be in flight for half an hour while the
  log sat unchanged. **The habits**: (1) confirm a background job by watching
  its LOG ADVANCE past a line it could not already have written, never by a
  process count; (2) truncate or delete the log before relaunching, so a stale
  completion marker from the previous run cannot satisfy the wait condition —
  that is exactly how the second job's `until grep EXIT=` returned instantly
  against 35-minute-old output; (3) any run measured in hours should
  checkpoint per unit of work, because the cheapest defence against a job dying
  for reasons outside the script is not needing it to survive.
- **An UNBOUNDED metric can be dominated by one object; a BOUNDED one cannot —
  so check them separately (exp54, 2026-08-23).** The same corrupt galaxy that
  inflated the amplitude benchmark `SIGMA_A` by 8-18% moved the SHAPE loss by
  **0.043%**. The asymmetry is structural, not luck: the amplitude residual is
  `log10(model/truth)`, unbounded, and that galaxy's was 3.76 dex, so its
  squared contribution swamped 2396 others; the shape residual is `(m-d)/d` on
  a curve of growth NORMALIZED to 1 at 100 kpc, so it lives near unity and no
  single galaxy can move a population mean far. **The habit**: when a data
  defect is found, do not assume it contaminates every metric equally — write
  down which metrics are unbounded in the direction the defect points, audit
  those, and MEASURE the leverage on the rest rather than re-deriving them all.
  It bounds the damage quickly, and here it kept the entire shape half of the
  experiment intact.
- **Bound the framework before enriching it — and price the bound's own freedom
  (exp54, 2026-08-23).** Three candidate limitations (the efficiency law, the
  deposit basis, the size law) looked identical in the loss. Replacing the
  model's seven-parameter weight law with per-galaxy non-negative least squares
  over the same deposits — convex, 5 ms per galaxy, no optimiser — separated
  them in an afternoon and retired two queued extensions, one of which was
  already agreed as the next thing to build. **But a ceiling computed with
  per-object freedom will always look reachable.** What makes it interpretable
  is the control: fit the same machinery to a DIFFERENT object. Here the
  71-weight solve fitted to the *wrong* galaxy reached `score_F` 0.672 against
  the true ceiling's 0.559 and the model's 0.872 — so most of the apparent
  headroom was flexibility, not information, and quoting the ceiling alone
  would have licensed exactly the extensions it was built to forestall.
  **The habit**: every ceiling ships with (1) a permuted-target control, (2) a
  count of how much of its nominal freedom it actually uses (here a median of 6
  active weights out of 71), and (3) a rung of the same bound at a freedom
  level a global law could plausibly reach.
- **Test a proposed parameter at the BOUND before fitting it (exp54,
  2026-08-23).** A redshift-dependent deposit shape `c = c0 + c_z ln(1+z)` was
  the agreed next extension. Scanning it over a grid of GLOBAL basis
  parameters, with the per-galaxy freedom held fixed so a gain could not come
  from degrees of freedom, put its optimum at `c_z` = 0 at two different
  freedom levels — minutes of compute against a day of implementing a
  per-deposit shape in shared code. **A term that cannot lower the bound cannot
  lower the fitted loss by the mechanism it was proposed for.** Scan it at more
  than one freedom level: the widest bound can be insensitive to a basis change
  that the actual model would still feel.
- **A metric can be blind in a region and still look converged there (exp54,
  2026-08-23).** The production loss is a fractional residual on the CUMULATIVE
  profile. At z = 2 only 6.3% of the stellar mass lies outside 52 kpc, so a
  sub-per-cent error the loss cannot see is a thirty-per-cent error in that
  annulus — and a per-epoch-free fit that matched the curve of growth to ~1% at
  every radius was still 0.2 dex/dex too shallow in the outer density slope.
  **Before naming a defect, ask whether the objective could have detected it at
  all**; if not, the fix is the objective, not the model.
- **The completeness trap recurs in every new view (exp54, 2026-08-23).** "The
  truth's outer density slope steepens from −2.78 to −3.38 by z = 2 while the
  model barely moves" was measured on the progenitor-selected sample. On the
  mh-complete sample it steepens only to −2.95, and the model's error is +0.21
  dex/dex rather than +0.48 — **more than half of the headline was the changing
  sample.** The identical correction had already been made for the inner
  profile a day earlier. **The habit**: once a completeness mask exists, apply
  it the FIRST time a new diagnostic is plotted, not after it has produced a
  headline.
- **A bound must be optimised on the same objects and the same functional it is
  reported on — three ways to get that wrong appeared in one afternoon (exp54,
  2026-08-24).** (1) The ceiling's solve used all five epochs while its score
  used a per-epoch completeness mask, so the reported object was not the
  optimum of the problem it was compared against. (2) The increment test's
  non-negative solve was weighted by the increment and its cost reported as a
  fraction of the later total, which reversed the ordering of the two failure
  modes it existed to separate. (3) Non-negative least squares minimises a
  fractional residual on the ABSOLUTE curve of growth, and its result was
  quoted as if it were the production shape loss — worth a factor of two
  (0.538 against the exact 0.282). **The habit**: write down the reported
  quantity first, then optimise exactly that; and when the reported loss is a
  mean over objects of a per-object term, minimise it per object and the
  population optimum is exact rather than approximate.
- **Check whether a constraint actually binds the quantity you are reporting
  (exp54, 2026-08-24).** "Deposition-only implies enclosed mass never falls" is
  a true theorem, and it puts almost NO floor under a shape loss that
  renormalises each epoch at 100 kpc — because any set of shapes can be made
  monotone by inflating the later epochs. The basis-free monotone bound is
  0.009 in `score_F`, not the 0.29 an isotonic projection scored; that 0.29 was
  the projection choosing to match the amplitude. **A reference is not a bound
  until it is the argmin of the reported loss.**
- **A control can be an identity (exp54, 2026-08-24).** Giving each galaxy one
  free amplitude per epoch interval CANNOT change its z=2 normalised profile,
  because every pre-z=2 deposit sits in one interval — so that rung had to
  equal the fitted model at z=2, and the equality was read as "no headroom at
  z=2". **Before quoting a control, ask what it is able to change**; if the
  answer is "nothing, at the epoch you care about", it is not a control.
- **When a defect has two candidate mechanisms, split the sample on the
  mechanism before modelling it (exp54, 2026-08-24).** The z=2 central deficit
  looked like a uniform 25 per cent failure. Preregistering "did the central
  mass later FALL?" split it into -37 per cent for the 53 per cent that
  declined — which no static deposition model can follow — and -11.5 per cent
  for the rest. A compact deposit channel had been queued to fix the population
  number; it addresses only the smaller half. **One binary split retired a
  planned experiment more decisively than any fit would have.**
- **Never let a smoke path write production filenames (exp54, 2026-08-24).**
  A `--smoke --figures` run made to test the plotting code silently overwrote
  the full-sample ladder figure with a 60-galaxy version, and it took an
  outside reviewer to notice the sample sizes in the panel titles. Smoke
  outputs get a suffix.

- **A bound built from per-object freedom is not a target for a law that has
  none (exp54, 2026-08-24).** Stage 3.4's temporal ladder — free deposited mass
  in K time groups, 14.2% at K=1 falling to 9.7% at K=8 — was read as "most of
  the reachable improvement sits at a time resolution a shared law can
  express", and a whole stage was launched on it. Every rung of that ladder is
  solved PER GALAXY. A shared law has no per-galaxy freedom at all, so the
  ladder never bounded it. Measured properly, by giving the shared law a FREE
  curve in redshift, the budget is **0.2 percentage points, not 3.3** — and two
  parameters already collect all of it. **The tell was in the output the whole
  time**: the K=1 rung was numerically IDENTICAL to the fitted model, because
  one free amplitude per galaxy cancels out of a profile renormalised at
  100 kpc. A rung that equals the thing it is supposed to bound is announcing
  that it measures a different kind of freedom. **The habit**: before treating
  a number as a target, ask how many free parameters PER OBJECT produced it,
  and build the matching bound at the freedom the candidate actually has.
- **Orthogonalise a new basis against what the model already has, or the
  identifiability report will be about the wrong thing (exp54, 2026-08-24).**
  `E4`, a free piecewise-linear curve in `ln(1+z)` whose knot VALUES were the
  parameters, was the worst of 45 variants with undetermined parameters — and
  the reason was structural, not statistical: `a0 + interp(x, knots, k)` is
  exactly degenerate, since adding a constant to `a0` and subtracting it from
  every knot gives an identical model. Rebuilding the same freedom in a basis
  made orthogonal to a constant and to `ln(1+z)` removed that degeneracy
  entirely (all 8-13 parameters "effective" at the 1e-3 threshold). It did not
  make the terms worth having — the honest verdict was still no — but it meant
  the report was measuring whether the DATA determine the new coefficients
  rather than re-discovering an algebraic identity.
- **Say what the extra freedom is being spent ON, not just what it bought
  (exp54, 2026-08-24).** "13.80% to 13.62%" is a number; "the correction is
  within 0.01 dex of the incumbent everywhere below z = 4 and reaches +1.4 dex
  by z = 15, where 1% of the stellar mass is made" is the explanation, and it
  predicts the poor identifiability rather than merely accompanying it.
  Plotting the fitted law against the mass-weighted distribution of the
  variable it depends on turned a null result into a mechanism.
- **Check a fast path against the slow one on the QUANTITY, not on the score
  (exp54, 2026-08-24).** Replacing a per-galaxy Python loop with one stacked
  array evaluation was worth ~3x and made eight fits plus six bootstrap reports
  feasible in an afternoon. It was gated on comparing the whole predicted
  profile array against the old path galaxy by galaxy at several thetas
  (1.6e-15), not on the two aggregate scores agreeing — a cancellation inside a
  mean over 2397 galaxies and 24 radii would have passed the weaker check.
- **A rejection rule written for the diagnostics does not apply itself to the
  fits (exp54, 2026-08-24).** `MAX_BACKWARD_DEX = 3.0` had existed in
  `scoreboard.py` and `selection.py` since the day row 181 was identified as a
  broken cross-match, and `fit.py` carried a comment explaining that the
  benchmark had been recomputed without it. The model-fitting stages built
  their samples straight from `population.npz` and never called it. From Stage
  3.3's per-epoch run onward — the moment the sample went from a 1199-galaxy
  subsample to all 2397 — that one galaxy was inside every fit, **carrying 18%
  of the total loss and inflating the amplitude score by 24%**. `fit.py`'s
  comment that it was "in NONE of the model-fitting subsamples" was true when
  written and silently expired. **The habit**: when a data-quality rule is
  written, apply it at the point the data is loaded, not at the point the
  figure is drawn; and when a sample definition changes, re-check every comment
  that asserts what is in it.
- **The corrupt galaxy hid where the sample was smallest (exp54,
  2026-08-24).** Section 5.3's headline — the model is ~36% worse than a plain
  regression at total stellar mass on completeness-controlled samples, "the
  strongest argument against the model" — was one galaxy. The completeness cut
  shrinks the sample from 2397 to 840, so a single fixed contribution to a mean
  square grows as a FRACTION as the sample shrinks. That is exactly the
  signature the note read as physics ("the regression barely notices the
  restriction; the model degrades sharply"). Corrected, the ratio is 1.03-1.06
  at z = 0.7-1.5 and **0.94 at z = 2**, where the model beats the benchmark.
  **The habit**: when a defect grows as a sample shrinks, suspect a fixed
  contaminant before a physical trend, and check by removing one object.
- **A readable profile representation needs an explicit inner anchor when its
  other coordinates are enclosed-mass radii (exp50).** Total mass plus
  R20/R50/R80 reconstructed the profile outside 5 kpc but left the measured
  2-kpc mass underconstrained. Adding the 2-kpc mass fraction reduced the median
  representation error from 0.027 dex to 0.004 dex. Quantiles describe where
  most of the mass lies; they do not automatically preserve a central aperture
  below the first quantile.
- **Separate coordinate fidelity from halo-to-coordinate predictability
  (exp50).** Adding R90 improved the median profile representation error from
  0.00437 to 0.00316 dex, but did not improve held-out halo prediction: its
  linear profile CRPS was 0.06601 dex versus 0.06577 dex without R90. A more
  accurate compression can add a target that the available halo variables do
  not predict. Choose coordinates on end-to-end held-out performance, not on
  reconstruction error alone.
- **Use symbolic regression as a stability-filtered correction search when the
  linear core is already strong (exp50).** Keeping the full linear relation and
  searching only its residuals produced three nonlinear terms that recurred in
  at least four of five training folds. They improved held-out profile CRPS by
  2.38% relative to the linear map, recovering about 62% of the dense degree-2
  improvement without presenting fold-specific expressions as physical laws.
- **Present-day concentration does not exhaust the information in concentration
  history (exp50).** On the common 2,366-galaxy sample, adding five fitted
  concentration epochs improved held-out profile CRPS by 2.89% relative to
  DiffMAH plus z=0.4 concentration; mass-conditioned shuffled histories made
  the score 0.12% worse. The effect is small but survives every held-out fold,
  so histories of secondary halo properties deserve direct null-controlled
  tests rather than being dismissed as redundant with MAH.
- **An inner-aperture anchor does not guarantee that total-mass quantile radii
  are resolved (exp51).** A full-sample preflight found a valid z=2 galaxy with
  81.1% of its measured stellar mass inside 2 kpc, so its total-mass R20 and R50
  both lie below the measured radial grid. Describing quantiles of the mass
  outside 2 kpc, alongside the separately modeled 2-kpc fraction, retained the
  object and kept median full-profile reconstruction error near 0.004 dex at
  every epoch. Never drop a physically informative extreme or assign a grid
  boundary as its radius merely to preserve a convenient coordinate system.
- **For high-redshift painting, fit the assembly clock available at that epoch
  instead of borrowing the final descendant's clock (exp51).** DiffMAH fitted
  only to halo history available by z=1.5 and z=2, plus concentration at the
  same epoch, lowered profile CRPS by 8.28% and 8.17% relative to complete
  descendant DiffMAH with concurrent concentration. Future information was not
  the source of the high-redshift success; expressing the same assembly history
  in the prediction epoch's local time frame was more useful.
- **The time label on a secondary halo property is part of the feature
  definition (exp51).** With descendant DiffMAH held fixed, concentration at
  the prediction epoch lowered profile CRPS by 2.23--4.66% relative to z=0.4
  concentration at z>0.4, while mass-conditioned concentration shuffles removed
  the improvement. A physically named feature is not automatically portable
  across redshift when its measurement epoch changes.
- **Smooth coefficient evolution must be checked by removing an entire epoch,
  not inferred from a smooth-looking plot (exp51).** Interpolating the
  descendant-conditioned readable-map coefficients from the other four
  snapshots worsened held-epoch profile CRPS by only 0.43--1.23%, whereas
  copying the nearest snapshot was substantially worse. This establishes
  continuous-redshift closure for that feature definition, but not for the
  epoch-local model whose inputs also evolve and require common scaling first.
- **A symbolic expression is useful only if its end-to-end predictive gain is
  stable and substantial (exp51).** At z=2, stable residual terms improved the
  epoch-local linear profile CRPS by 1.00%, while dense degree-2 terms improved
  it by 5.01%; the fixed z=0.4 terms transferred only a 0.52% gain. Recurrence
  across folds can reject unstable algebra, but it cannot turn a weak correction
  into a redshift-independent physical law.
- **Verify the active representation at the figure boundary when experiments
  reuse modules (exp51 mistake).** The first z=2 symbolic run imported exp50's
  decoder through Python's cached module name even though exp51 coordinates had
  been compressed. The per-target labels in the direct prediction figure
  exposed the mismatch, and the full symbolic run was discarded and repeated
  with an explicit exp51 module reference. When neighboring experiments contain
  same-named modules, avoid bare cached imports and assert target labels and
  dimensions immediately before reconstruction.
- **Write transformed coordinates together with their inverse map in technical
  reports (Exp50--Exp51 report).** The radial-gap variables are logarithms of
  differences between logarithmic radii, not logarithms of physical-radius
  differences. Writing only the forward coordinate definition makes that
  distinction easy to miss. The report now states both transformations and the
  monotone decoder, and its audit checklist was checked against the actual
  implementation before compilation.
- **Render figure-rich PDFs onto an opaque background during visual review
  (Exp50--Exp51 report).** Rendering with an alpha-capable PNG device made the
  transparent margins of imported vector figures appear as black blocks even
  though the PDF was correct. Re-rendering all pages as opaque RGB separated a
  renderer artifact from a document-layout problem and avoided an unnecessary
  change to the publication figures.
- **Inspect the vector and raster exports of density plots independently
  (Exp50--Exp51 report correction).** The PNG exports of the Exp50 hexbin
  diagnostics were complete, while the PDF exports silently omitted four of
  five data clouds in one figure and all six in another. The plotting code now
  rasterizes only the dense hexbin collections, retains vector axes and labels,
  and can rebuild these diagnostics from the saved held-out predictions. The
  report uses the 300-dpi PNG versions for viewer-independent rendering.
- **A low cumulative-profile residual does not establish an identifiable
  analytic coordinate system (exp55).** A five-parameter radial-sigmoid CoG
  reached 0.00556 dex median cumulative-profile RMS, but 89.6% of galaxies put
  at least one parameter within 1% of a bound and the median scaled-Jacobian
  singular-value ratio was only `10^-3.42`. Multiple starts, re-optimized
  profile scans, synthetic recovery, boundary incidence, and radial jackknifes
  must be treated as selection criteria rather than after-the-fact diagnostics.
- **Restricted mixtures can be usable without claiming that their components
  are physical populations (exp55).** Fixing the component shape indices
  globally, defining the component fraction within the measured outer
  aperture, and enforcing ordered radii produced a four-parameter
  Sérsic-plus-Moffat representation with exact synthetic recovery and typical
  radial-jackknife changes below 0.008 dex. This does not solve unrestricted
  multi-Sérsic degeneracy and does not identify the components as in-situ and
  ex-situ stars; it creates stable phenomenological coordinates that can be
  tested against halo assembly.
- **Test analytic CoGs in differential as well as cumulative form (exp55).**
  The best restricted two-Sérsic model had a respectable 0.00727 dex median CoG
  RMS but a 0.145 dex annular-density RMS because both components have
  exponential outer tails. Giving the extended component a Moffat power-law
  tail reduced the corresponding errors to 0.00437 and 0.0660 dex. Cumulative
  residuals alone can hide an incorrect density slope.
- **A numerical integral is not an analytic profile merely because its
  integrand has a formula (exp55 correction).** The first radial-sigmoid
  implementation evaluated the cumulative profile on an internal integration
  grid. Visual and implementation review exposed that it did not meet the
  portability requirement, so it was replaced with the exact softplus
  antiderivative before the family comparison was accepted.
- **A visually excellent cumulative profile can retain a meaningful outer-
  density error (exp55).** Near 140 kpc, a typical annulus contains only 1.09%
  of the mass enclosed by its outer edge. A 0.00531 dex median absolute CoG
  error there becomes a 0.132 dex median absolute density error when adjacent
  cumulative masses are subtracted. Always judge an analytic CoG in cumulative
  apertures, differential annuli, and surface density; none of the three is a
  substitute for the others.
- **An analytic profile can be identifiable from a known CoG while some of its
  coordinates remain almost unpredictable from halos (exp55 halo map).** The
  fixed Sersic-plus-cored-power-law fit has stable per-galaxy parameters, yet
  held-out DiffMAH plus concentration explains 86.5% of the variance in stellar
  mass within 148.2 kpc and only 6.0% and 6.8% in compact fraction and compact
  radius. The reconstructed profile still ties the Exp50 readable map. Judge
  the halo map through reconstructed observables and residual modes, and do not
  equate profile-fit identifiability with deterministic halo predictability.
- **A generated galaxy must be evaluated at its own generated size (exp55
  correction).** The first stochastic QA used the standard paired diagnostic,
  which deliberately evaluates model and TNG at the TNG galaxy's true `Re`, and
  reported 0.213 dex scatter in the `<2Re` versus `2--4Re` plane. That is useful
  for conditional-mean profile error but invalid for a generated population.
  Measuring each draw at its own `Re` gives 0.188 dex for the raw Gaussian and
  0.084 dex after nested residual calibration, versus 0.072 dex in TNG. Keep
  paired fixed-scale diagnostics and self-consistent population diagnostics as
  explicitly separate products.
- **More algebraic flexibility can improve a scalar score while worsening the
  coherent radial error (exp55 refinement).** A complete degree-2 halo map
  lowered held-out profile CRPS by 0.45% relative to the sparse degree-2 map,
  but increased the largest halo-mass-bin median shape residual from 5.45% to
  6.49%. Candidate mean models must pass both a held-out distribution score and
  direct mass-binned radial-residual checks; a small average improvement is not
  evidence that the visible systematic was repaired.
- **A coordinate rotation organizes residual physics but does not create new
  predictive information (exp55 refinement).** Rotating the four analytic
  profile coordinates by their effects on the CoG revealed distinct broad-
  amplitude, central-transfer, intermediate-versus-outer, and weak-inner modes.
  Because this transform is invertible, fitting the same linear design in the
  rotated coordinates cannot be credited as a better conditional mean. Its
  value is to expose which radial combinations are halo-predictable and which
  should remain stochastic.
- **Calibrate an empirical residual generator on inner held-out galaxies, not
  on the final evaluation fold (exp55 refinement).** Nearest-neighbour
  resampling recovered the population-plane geometry but covered only 62.7% of
  profile points with a nominal 68% interval. Rebuilding the residual library
  from cross-fitted predictions did not fix this, whereas one inflation factor
  selected independently within each outer training fold gave 68.4% coverage
  and lower profile CRPS. The selected factors were 1.16 in four folds and 1.08
  in one, supporting a simple calibration rather than an unrestricted residual
  density model.
- **Correct stochastic scatter does not repair an incorrect mean population
  relation (exp55 refinement).** The calibrated generator reproduces the TNG
  R90 scatter to within 0.005 dex, yet its R90 distribution remains 3.63 times
  the TNG split-half energy-distance floor because the predicted mass--R90
  relation is too shallow and the median radius is high by 0.024 dex. Treat
  residual width and conditional-mean structure as separate failure modes.
- **A small integrated-profile error can bias an outer quantile relation
  systematically (exp55 outer-envelope test).** The individually fitted
  `gamma=1.25` analytic family has only 0.00437 dex median CoG RMS, yet changes
  the TNG mass--R90 slope from 0.291 to 0.251 before any halo mapping. Diagnose
  population sizes at the representation, conditional-mean, and stochastic
  layers separately; aggregate CoG RMS does not bound errors in derived
  population relations.
- **Fix the stochastic ensemble size before calibrating empirical quantiles
  (exp55 outer-envelope test).** The inner-selected residual inflation changed
  between 16 and 32 draws because sample 16th and 84th percentiles are discrete.
  Production comparisons now use 32 draws consistently and report complete-
  population variation across draws for geometry metrics.
- **A stochastic generator can improve one population plane while worsening
  another (exp55 outer-envelope test).** Neighbour-residual draws improve
  profile CRPS and the self-consistent size-scaled plane relative to raw
  Gaussian draws, but worsen the R90 energy distance. Retain a standard set of
  fixed-kpc, size-scaled, and outer-quantile diagnostics for every generator;
  no single distribution score is a sufficient selection criterion.
- **Use mass-conditioned nulls before blaming compressed halo histories
  (exp55 outer-envelope test).** Actual-minus-DiffMAH anchor residuals correlate
  only -0.030 to +0.045 with R90 errors, and neither they nor the DiffMAH fit
  RMS improve the held-out profile coherently relative to shuffled controls.
  For this failure, the evidence points to the analytic outer profile rather
  than lost MAH diversity.
- **A global profile hyperparameter may have negligible CoG cost but large
  population consequences (exp55 outer-envelope test).** Changing the fixed
  outer Moffat slope from 1.25 to 1.4 worsens median CoG RMS by only 0.000030
  dex while improving density RMS, boundary incidence, and the generated R90
  relation. Select such coordinates with fold-clean multi-metric gates, not
  only the original fitting loss.
- **A successful hard boundary is evidence for smooth dependence, not for a
  physical discontinuity (exp55 mass-dependent outer slope).** A fold-local
  two-regime slope rule brings the generated R90 distribution to the TNG
  sampling floor and reduces coherent mass-bin residuals, but it leaves visible
  structure near its arbitrary 2/3-mass split. Use it as a diagnostic ceiling
  and next test one smooth, low-complexity mass dependence; do not free the
  slope independently per galaxy.
- **Visual review must check panel-to-bin routing and numerical axis offsets
  (exp55 overnight QA).** A first comparison figure overplotted low- and
  middle-mass residuals because integer panel routing mapped both to one axis,
  and Matplotlib hid the absolute halo-mass threshold behind an offset. Direct
  image inspection caught both; the final figure shows only the lowest and
  highest bins and disables the misleading offset.
- **Every named mass--size relation must state which mass defines the x-axis
  (exp55 smooth-slope correction).** An initial Stage 10 selector minimized the
  R90-slope error against halo mass even though the population failure and
  standard mass--size QA use stellar mass. The wrong selector chose a 0.05-dex
  logistic width in every fold; the corrected stellar-mass--R90 selector chose
  0.20 dex in every fold and materially changed the generated population.
  Encode the x-variable in function names or call the same audited size-relation
  helper used by final QA.
- **Do not reinterpret a candidate selected at the edge of a small grid as a
  measured optimum (exp55 smooth outer slope).** All five folds choose the
  initially widest tested logistic transition, 0.20 dex. A separately declared
  extension to 0.30, 0.50, and 0.80 dex then selects 0.20 dex again in every
  fold, making it an interior grid choice. The extension closes the numerical
  direction without turning a coarse grid result into a high-precision physical
  transition-scale measurement.
- **Full draw-to-draw geometry is a final robustness product, not an inner-loop
  metric (exp55 smooth outer slope).** Recomputing all two-dimensional energy
  distances for 32 complete populations and two models took about ten minutes,
  longer than fitting all smooth profile cells from cache and refitting the
  halo map. Cache fixed truth sampling floors and vectorize repeated distances
  before expanding such comparisons.
- **Derive diagnostic-figure styling from the declared candidate grid (exp55
  Stage 11 correction).** Extending the smooth-width grid from three to six
  values completed and saved all numerical results, but the first figure build
  failed because its color dictionary still contained three hard-coded entries.
  The figure now assigns a sequential `cividis` color from the active grid and
  regenerates from saved outputs, so future grid extensions cannot repeat this
  failure.
- **A conditional slice is not evidence that a parameter is determined, and the
  gap is not small (exp54 Stage 3.9).** Displacing one parameter with the other
  six held fixed overstated every one of seven parameters' widths, by 2.3x to
  17.6x, and only 0.32% to 19.7% of the conditional loss rise survived
  re-optimising the rest. The size of the overstatement is predictable from the
  model's structure rather than from the fit: the four parameters that are the
  constant, two slopes and cross term of one bilinear surface absorb each other
  12-18x, while the one parameter that changes an individual deposit's radial
  SHAPE is absorbed only 2.3x. Before quoting a parameter's precision, ask what
  else in the model could undo a change to it.
- **A finite-difference Hessian cannot deliver the small curvatures of a
  correlated model, and its failure looks like a bug (exp54 Stage 3.9).**
  Across step sizes spanning a factor of ten, the five largest eigenvalues were
  stable to better than 0.1% while the two smallest moved 25x and one went
  negative. It was cancellation — extracting a curvature of order 0.1 from a
  matrix whose diagonal is of order 100 — not non-convergence, which was ruled
  out separately by re-optimising all seven parameters from the incumbent and
  recovering its loss exactly. Rule out non-convergence before trusting a
  negative eigenvalue, and prefer profiling directly to predicting the profile.
- **Size a scan from a measured property of the object, not from a round number
  (exp54 Stage 3.9).** Each profile grid was laid at multiples of the
  displacement at which the CONDITIONAL rise reaches a fixed value, measured by
  bisection in each direction separately. That made every parameter's grid span
  the same range of conditional damage regardless of its units, made the
  asymmetry of the loss visible (one parameter needed +5.02 to cost what -0.40
  cost), and made "how much survives" comparable across parameters.
- **Declare the rule for extending a scan BEFORE seeing which side needs it
  (exp54 Stage 3.9).** Three parameters failed to reach the reporting threshold
  on one side. The extension used multipliers fixed in a module constant and
  applied only to sides that fell short, so extending could not become a licence
  to keep going until a number came out a preferred way.
- **Check whether a spread you are about to quote is the optimiser's budget
  (exp54 Stage 3.9).** The galaxy bootstrap refits each replicate from the
  incumbent with a limited iteration cap, and a flat valley is exactly where a
  limited optimiser would fake a small spread. Four replicates refitted with an
  eight-fold budget returned bit-identical parameters in the same wall clock, so
  the spread is a measurement. The check cost twenty minutes and protects every
  number quoted next to it.
- **Bounding a dimensionless FACTOR does not bound a physical REACH (exp57).**
  The expansion law `X1` was designed so no deposit could ever grow by more than
  `1+A`, on the reasoning that the first transport model failed because its
  factor was unbounded. Fitted, the factor was 2.47 — and 85% of the mass it
  moved still landed beyond 50 kpc, worse than the model it was designed not to
  repeat. Homologous scaling multiplies every radius by the same number, and a
  deposit with a power-law tail has most of its outer mass at large radius, so
  the tail moves furthest in absolute terms. **If a gate is stated in kpc, the
  mechanism has to be bounded in kpc.** A non-homologous remap that expands the
  core and leaves everything beyond a fitted `Rc` untouched cut the same number
  from 85% to 2.2%.
- **The loss can rank two basins of the SAME model backwards (exp57).** A
  multi-start search over the core-only expansion found a core radius of 2.6 kpc
  and a near-homologous one at 92 kpc. The loss preferred the near-homologous
  basin by 0.12%, and that is the one the mechanism gates reject. Multi-start is
  not only insurance against a bad optimum; it is how a second, physically
  different solution becomes visible at all. Gate every basin, not only the
  winner.
- **An assertion does not make duplication safe (exp57).** A diagnostic
  duplicated the model's setup so it could keep deposits separate, and its
  docstring argued the duplication could not drift "because `check_terms`
  asserts that summing these reproduces `predict`". It drifted the very next
  time the model changed — a new radial law was added to the model and not to
  the copy — and the assertion caught it at a relative 6.3e-1. The assertion
  worked; the reasoning that justified the duplication did not. Remove the
  duplication.
- **Write down which POPULATION a target describes, in the constant's name
  (exp57).** exp54's "-0.066 dex" is the median over the DECLINING SUBSET; the
  median over all galaxies is +0.037, the opposite sign. Comparing a model
  median over everyone against it would have selected an expansion strength
  twice too large. The structural guard that worked was returning a named
  record instead of a positional tuple, so a caller cannot pick the wrong field
  silently. This is the fourth instance of this class of error in this project.
- **A gate written as a guard can turn out to be a target, and the null is what
  tells you (exp57).** G5(c) was written to stop expansion from lowering central
  density everywhere, on the assumption that the model already matched the
  measured core growth. Run at the null it was already 0.046 dex out, in the
  direction expansion helps. Running every gate on the un-enriched model first
  costs nothing and is what distinguishes "this enrichment broke something" from
  "this was already broken".
- **Separate what the new term did from what refitting around it did (exp57).**
  Switching the expansion off at each fit's OWN fitted base parameters, rather
  than at the incumbent's, splits a change into the mechanism's contribution and
  the compensation the other parameters supplied. Here it showed the expansion
  doing all the work in every case, and explained one gate failure exactly: the
  term moved the core gap -0.025 while the refit moved it +0.046.
