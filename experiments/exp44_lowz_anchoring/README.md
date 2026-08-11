# exp44 — "calibrate low, predict high": per-object low-z anchoring

**The question (option A of the 2026-07-21 menu):** the kernel is today
either fully population-shared — which hits the measured
population-sharing wall (M(<5 kpc) ~ -8% in every shared-theta fit,
even single-epoch) — or dressed with a MEAN-ZERO stochastic layer,
which fixes population distributions but by construction cannot improve
any individual galaxy. There is a third mode nobody has tested, and it
is exactly the observational situation: **fit ONE per-galaxy number
against the OBSERVED low-z profile, freeze everything else, and predict
the higher-z epochs.**

Grounding:
- exp41 stage 0: freeing ONE base component per galaxy buys 18.5-22.3%
  median loss improvement, the six single-component deltas rank-correlate
  at |rho| >= 0.92 (ONE latent axis through degenerate coordinates), and
  that axis is FEATURE-ORTHOGONAL (|Spearman rho| <= 0.20 vs logmh,
  c200c, fz2, logms) — genuinely per-object information, unreachable by
  more halo conditioning.
- exp43: the transport clock locks in from TWO epochs, and the 1ch-mof
  z<=0.7 fit predicts the unfitted high-z differential-deposition pairs
  at data precision — the POPULATION physics extrapolates.

Untested: does the PER-GALAXY part transfer upward in redshift too? If
yes, low-z anchoring is a real new capability and the only route on the
menu that legitimately crosses the sharing wall (the prediction epochs
are held out in the epoch direction). If no, the individuality is
epoch-local — a clean null, and a publishable statement about what the
kernel's per-galaxy freedom actually means.

## Design

Base model: the adopted 1ch-mof at its official theta
(exp40 `latestart.npz[theta_z15]`), UNCHANGED. Per galaxy, exactly ONE
coordinate is refit (bounded scalar, inside the physical box — the
exp41 stage-0 protocol), against ANCHOR epochs only.

| arm | anchor epochs | anchoring observable | axis |
|---|---|---|---|
| `sig_z04_prof` | z=0.4 | full pinned profile (the model's own loss) | sig |
| `sig_z07_prof` | z=0.4, 0.7 | full pinned profile | sig |
| `sig_z04_aper` | z=0.4 | kpc aperture masses ONLY | sig |
| `sig_z07_aper` | z=0.4, 0.7 | kpc aperture masses ONLY | sig |
| `rc_z07_prof` | z=0.4, 0.7 | full pinned profile | log_rc |

The `aper` arms are the observationally honest ones: M(<10), M(<30),
M(<50), M(<100) kpc are what a survey actually delivers; the inner
profile shape is not trusted (the exp40 objective note). `sig` is the
axis the adopted stochastic layer uses (continuity); `rc_z07_prof`
tests whether the choice of (near-degenerate) anchoring coordinate
matters at all.

**Controls — the experiment is nothing without these:**
1. `pop` — the population theta, no anchoring. The baseline every arm
   must beat AT THE PREDICTION EPOCHS to mean anything.
2. `oracle` — the same single coordinate fitted against ALL FIVE
   epochs. The ceiling of what one per-galaxy number can do; the gap
   between an anchored arm and the oracle is what low-z information
   fails to capture.
3. `shuffle` — each galaxy is given a deviation fitted from a
   DIFFERENT, randomly chosen galaxy (a permutation of the fitted
   values, so the marginal distribution is identical). If anchored
   does not beat shuffled at the PREDICTION epochs, the "improvement"
   is just added spread, not transferred information. This is the
   experiment's falsification gate.

**Readout.** Per epoch: median pinned shape max|rel| (R>5 kpc), M(<5)
and M(<10) bias. Anchor epochs are IN-SAMPLE by construction (their
improvement is guaranteed and carries no information); everything is
decided at the prediction epochs, marked `*`. Plus the physics tests
(differential deposition, outskirt terciles) on the anchored
population — per-object freedom must not break what objectives broke
in exp39.

## Run

```bash
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
PYTHONPATH=. uv run python experiments/exp44_lowz_anchoring/anchor.py \
    {demo|fit|report|physics|figures} [--dev]
```

## Results (2026-07-21, full n=2397; `outputs/report.npz`,
## figure `figures/exp44_transfer`)

**Verdict: per-galaxy information DOES transfer upward in redshift, but
its reach is about ONE epoch. A single per-galaxy number is worth a
great deal where it is measured (it crosses the population-sharing wall
outright), recovers ~1/3 of that value at the adjacent unanchored
epoch, and is worth essentially nothing beyond it. The individuality is
epoch-local.**

Metric conventions: pinned shape = median per-galaxy worst-radius
absolute relative deviation of the 148-kpc-pinned model curve over
R > 5 kpc (lower better); M(<5) bias = median relative error on
M*(<5 kpc) (0 = unbiased; the population theta sits at -8 to -12%).
`*` = the epoch was NOT anchored.

### 1. One per-galaxy number is worth a lot — the ORACLE ceiling

Fitting the same single coordinate against ALL five epochs (`oracle`)
takes the kernel from the population theta's 18.2/17.4/16.5/16.3/15.4
shape to **13.1/9.8/10.3/12.3/12.5** — gains of 5.1/7.6/6.2/4.1/3.0
points — and halves the inner deficit (M(<5) -9.8 -> -6.4 at z=0.4,
-7.6 -> -2.6 at z=1.0). This confirms the sharing wall is genuinely a
PER-OBJECT information limit, and that ONE coordinate captures most of
what per-object freedom can buy (consistent with exp41's finding that
the six coordinates are one degenerate axis).

At its own anchor epoch, low-z anchoring realizes this immediately:
z=0.4-anchored gives z=0.4 shape **8.4%** vs the population's 18.2%,
and M(<5) -3.4% vs -9.8%. (In-sample by construction — quoted to show
the magnitude available, not as evidence.)

### 2. The transfer decays over ~one epoch

Fraction of the oracle's gain recovered at unanchored epochs, and the
decisive control — the margin over the SHUFFLE null (anchored minus
shuffled shape, positive = the deviation is galaxy-SPECIFIC, not just
added spread):

| arm | z=0.7* | z=1.0* | z=1.5* | z=2.0* | shuffle margin (prediction epochs) |
|---|---|---|---|---|---|
| `sig_z04_prof` | **37%** | -4% | 1% | 34% | +6.1 / +3.2 / +1.7 / +0.4 |
| `sig_z07_prof` | — | **34%** | 13% | 29% | +6.0 / +2.1 / +0.3 |
| `sig_z07_aper` | — | **22%** | 11% | 5% | +4.5 / +2.1 / +0.4 |

The shuffle margin is the honest decay curve: **+6 points at the
adjacent unanchored epoch, +2 to +3 two epochs out, +0.3 to +0.4 at the
far end.** (The apparent 29-34% at z=2.0 is not solid: the oracle's own
gain there is only 3.0 points and the shuffle margin has collapsed to
+0.4 — the base theta is itself extrapolating at z=2.0.)

Two secondary reads:
- **The observationally honest arm works, at ~2/3 strength.** Anchoring
  on kpc APERTURE MASSES ONLY — what a survey actually delivers —
  recovers 22% of the oracle gain at the adjacent epoch versus 34% for
  the full profile. Real applications lose some, not most.
- **The anchoring coordinate barely matters** (`rc_z07_prof` ties or
  slightly beats `sig_z07_prof`: +2.7 vs +2.1 points at z=1.0),
  exactly as exp41's one-degenerate-axis result predicts.

### 3. The inner-mass gain at distant epochs is a MEAN SHIFT, not transfer

This is what the shuffle control was for. Anchored vs shuffled
M(<5) bias at unanchored epochs (`sig_z04_prof`, population -7.9 /
-7.6 / -10.7 / -12.0):

| epoch | anchored | shuffled | verdict |
|---|---|---|---|
| z=0.7* | **-1.6** | -3.0 | genuine per-object transfer |
| z=1.0* | -2.0 | -2.8 | mostly mean shift |
| z=1.5* | -5.9 | -5.7 | **entirely mean shift** |
| z=2.0* | -7.9 | -7.4 | **entirely mean shift** |

The fitted deviation pool is asymmetric (median delta sig -0.022), so
applying ANY draw from it shifts the population's median inner mass
upward. Beyond the adjacent epoch the "halved inner deficit" is that
recalibration, which a plain theta refit would achieve just as well —
not transferred per-object information. **The sharing wall is NOT
crossed at distant epochs.**

### 4. The physics survives per-object freedom

On the anchored population (`sig_z07_aper`): differential deposition
0.43/0.14 at z0.7->0.4 (data 0.37/0.11; population theta 0.40/0.13) —
mildly strained, at the edge of the adopted band but not broken; the
earlier pairs track well (0.34/0.11, 0.30/0.09, 0.22/0.06). The
outskirt residuals IMPROVE markedly: T1 -0.006/+0.004 versus the
population theta's +0.025/+0.028. Unlike every inner-aware OBJECTIVE
tried in exp39, per-object freedom buys inner accuracy without breaking
the outskirts — the tension there was objective-driven, and this is a
different lever.

## Stage 2 — the decorrelation measurement (2026-07-21, full n=2397;
## `outputs/decorr_report.npz`, figure `figures/exp44_decorrelation`)

Stage 1's one-epoch reach had two possible causes, and stage 1 could not
separate them: **(i)** the individuality genuinely EVOLVES with epoch, or
**(ii)** it is persistent but a single-epoch ESTIMATE of it is noisy.
The distinction decides everything: under (ii) better anchoring buys
reach, under (i) nothing does. Method: fit the same coordinate
INDEPENDENTLY at each epoch (per galaxy), take the cross-epoch
correlation matrix, and remove the noise floor by also fitting on two
disjoint halves of the radial grid (odd/even radii) — the within-epoch
split-half correlation, Spearman-Brown corrected to full length, is the
RELIABILITY of a single-epoch estimate, and dividing the cross-epoch
correlations by sqrt(rel_a rel_b) disattenuates them.

**Result: (ii) is ruled out; the individuality is genuinely
epoch-local.**

Single-epoch estimate reliability (1 = noiseless): **1.00 / 1.00 / 1.00
/ 0.99 / 0.96** across z = 0.4 to 2.0. Disattenuation therefore changes
nothing (AR(1) rho 0.558 -> 0.561). Cross-epoch Spearman correlation of
the per-galaxy sig deviation:

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| **z=0.4** | 1.00 | 0.63 | 0.42 | 0.24 | **0.09** |
| **z=0.7** | 0.63 | 1.00 | 0.69 | 0.41 | 0.12 |
| **z=1.0** | 0.42 | 0.69 | 1.00 | 0.60 | 0.20 |
| **z=1.5** | 0.24 | 0.41 | 0.60 | 1.00 | 0.40 |
| **z=2.0** | 0.09 | 0.12 | 0.20 | 0.40 | 1.00 |

The per-galaxy correction needed at z=0.4 is **essentially unrelated**
(rho = +0.09) to the one needed at z=2.0. The matrix is well described
by an AR(1) decay in epoch index: **rho = 0.56** (sig axis) and
**rho = 0.66** (log_rc axis), Pearson and Spearman agreeing to ~0.03.
The population-median deviation drifts only mildly with epoch
(-0.022/-0.020/-0.018/-0.042/+0.000 in sig), so this is individuality
decorrelating, not a systematic mean-model drift.

### The keystone: both model families measure the SAME timescale

exp37 measured the statistical emulator's cross-epoch residual latent at
**rho = 0.62**, using the identical estimator (`multi_epoch.fit_rho`).
The kernel now returns **0.56-0.66**. These are two completely different
model families — a 12-parameter physical transport kernel and a
heteroscedastic conditional-Gaussian emulator over PCA shape modes —
and they agree.

That agreement is what licenses the physical reading. A sceptic would
say the kernel's decorrelation is an artifact of its ONE-CONSISTENT-
HISTORY constraint: if no single theta can serve a galaxy at all epochs,
the per-epoch best fits must differ. But the statistical emulator has NO
consistency constraint whatsoever (independent per-epoch cores,
coefficients merely interpolated in z) — under that hypothesis it should
show markedly HIGHER persistence. It does not. **rho ~ 0.6 is a property
of the galaxies, not of either model's structure.**

### Three consequences

1. **Stage 1's reach limit is fundamental, not instrumental.** With
   rho = 0.56 the correlation is 0.58 at one epoch of separation, 0.34
   at two, 0.18 at three. Anchoring can only transfer what remains
   correlated, and the estimate is already noiseless — richer low-z
   data, more radii, better observables will NOT extend the reach.
2. **The adopted exp41 layer over-persists the individuality.** It
   applies ONE deviation per galaxy, held fixed at every epoch — which
   is rho = 1, the opposite extreme from independence, against a
   measured rho ~ 0.56. A drawn galaxy therefore carries a low-z-
   appropriate deviation into epochs where it should have decorrelated.
   That is a NEW, specific, testable contribution to the layer's known
   high-z symptom (kpc planes stalling at 2.0-2.8x the split-half floor
   for z >= 1.5), independent of exp41's own "the mean model's ridge
   error dominates" diagnosis. Both can be true; only this one is
   cheaply fixable.
3. **The fix is an AR(1)-in-epoch deviation**, with the measured
   rho ~ 0.56 and the marginal distribution unchanged (the measured
   heavy-tailed empirical pool) — precisely the construction exp37
   already uses statistically, and a small change to
   `stage2_draws.drawn_cogs`. **Design caveat to settle first:** letting
   theta vary by epoch means the drawn object is no longer ONE
   consistent history, which is the kernel's structural selling point.
   It is defensible for a generative population layer (the mean model
   stays a consistent history), but mass conservation across epochs and
   the differential-deposition test must be re-checked on the drawn
   population, not assumed.

## Reading and the natural follow-up

For forward modeling: **anchoring on observed low-z data buys about one
epoch of reach.** With z=0.4 data in hand, the z=0.7 prediction improves
substantially (37% of the achievable per-object gain, +6 points over a
wrong-deviation null); the z >= 1.5 predictions are no better than the
plain population kernel. Anchoring is worth doing — it is nearly free —
but it is not a route past the sharing wall at high z.

The decay pattern itself is the finding: the kernel's per-galaxy axis
DECORRELATES with epoch separation, which is precisely the structure
exp37 measured statistically for the emulator (an AR(1)-in-epoch latent
with rho = 0.62). The natural follow-up is to measure that decorrelation
directly for the kernel — fit the axis at each epoch INDEPENDENTLY and
measure the cross-epoch correlation matrix — which would (a) quantify
the reach exactly rather than through anchored-vs-oracle proxies, and
(b) hand the kernel an epoch-coupled individuality model instead of the
current epoch-independent one (the exp41 layer's known limitation, and
the reason its high-z planes stall at 2.0-2.8x the floor).
