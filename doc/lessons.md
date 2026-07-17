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
  true optimum.
