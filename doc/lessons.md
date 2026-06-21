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

## Analysis / interpretation

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

## Workflow

- Background `uv run` commands buffer stdout through pipes; redirect to a file
  and read that, or block on the process, rather than polling a pipe.
