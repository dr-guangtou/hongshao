# exp87 — the inverse problem: what the stellar curve of growth knows about its halo (2026-10-02)

Branch `exp87-inverse-halo` (from master `ba7d8ab`). Plan
`doc/plans/2026-10-02-exp87-inverse-halo.md` (approved by the user,
2026-10-02). An INDEPENDENT side study: it touches no Paper-1 file or gate.
Run in autopilot through seven pre-declared stages; every decision taken on
the way is in `outputs/decisions.jsonl`. Figures in `figures/qa/`.

## In plain language

**The question.** Every other experiment in this repository predicts a
galaxy's stellar profile from its halo. This one turns the arrow around:
if all you have is the stellar curve of growth (the stellar mass inside 24
radii from 2 to 148 kpc, so the total stellar mass is included), how well
can you say (1) how massive the halo is, (2) how concentrated it is, and
(3) how it assembled?

**The statistical trap, and how it was handled.** The sample was built by
keeping every TNG300 central whose halo is heavier than 10^13 solar masses
at z = 0.4. That is a cut on the very quantity being predicted. Near the
cut, a galaxy whose stars suggest a 10^12.9 halo can only be in the sample
if its halo is in fact above 10^13, so an ordinary regression bends
upward there, reports a scatter that is too small (0.18 instead of 0.22
dex), and puts 9 per cent of its predicted probability on halo masses the
sample cannot contain. The fix is to build the cut into the likelihood: a
normal distribution with a hidden ("latent") mean and width, cut off at
10^13 and renormalised. On real data that alone improves the score by 11
to 17 per cent, more than any choice of method (figure
`exp87_truncation`). Every method here is scored by the same rule: how
close its whole predicted distribution lies to the truth (the CRPS, in
dex; lower is better), out of sample, on galaxies it never saw. A
synthetic universe with a known answer was used first to prove the
scoring recovers that answer; one galaxy in five was locked away and
scored only once, at the very end.

**What the curve of growth knows about halo mass.** A lot, and almost all
of it is in the outermost light.

- Knowing nothing but the sample itself: 0.170 dex.
- Total stellar mass inside 148 kpc: 0.0875 (a typical error of 0.16 dex).
- The mass in the outermost shell alone, between 132 and 148 kpc: 0.0784.
  One number from the faint outskirts beats the total stellar mass.
- Total mass plus that one shell: 0.0763. All 24 points: 0.0744 (a
  typical error of 0.137 dex). The profile inside 30 kpc on its own:
  0.094, worse than total mass.

So the halo-mass information the profile adds to total stellar mass (15
per cent) is the outer envelope's density, and two numbers carry nearly
all of it (figure `exp87_radius`):

    log Mh = 3.16 + 0.38 log M*(<148 kpc) + 0.62 log M*(132-148 kpc),   latent scatter 0.18 dex

**Does a cleverer method do better?** No. Written in shell masses (the
mass between consecutive radii) the relation is a straight line: adding
curvature, gradient boosting, a Gaussian process, a neural network or a
symbolic-regression search gains nothing above the 1 per cent level set
by the synthetic test, at any epoch, in the populations whose selection is
understood. The symbolic search did find, by itself and in every fold,
that the useful variable is the outer mass fraction, and compact formulas
that match the 24-point model within 1 per cent, but no single formula
recurred across folds, so none is adopted. The apparent "nonlinearity" at
high redshift was a matter of coordinates: the logarithm of a shell mass
is a curved function of the logarithms of cumulative masses.

**Concentration and assembly history.** The curve of growth alone says
little: 8 per cent better than knowing nothing for the concentration, 10
per cent for the time the halo reached half its mass, and the true halo
mass alone says nothing at all about either. But the two TOGETHER say a
lot: profile plus the true halo mass improves concentration by 21 per
cent and formation time by 24 per cent. The assembly information lives in
how far a galaxy sits from the stellar-to-halo mass relation, which the
profile cannot know without the mass. The same holds for the full
accretion history (figure `exp87_assembly`): from the profile, the
halo's growth curve is recovered 6 per cent better than from nothing at
z = 0.4; with the true mass added, 21 per cent better; and the galaxy's
own stellar history across epochs does as well as profile plus true mass.

**The stars remember the halo as it was.** (Added after the first
report, at the user's question.) The z = 0.4 profile predicts the halo's
mass 2.5 Gyr EARLIER better than it predicts the halo's mass at z = 0.4
itself, and at every earlier epoch it beats the true z = 0.4 halo mass as
a predictor of the progenitor's mass. At z = 0.7 it even beats the
z = 0.7 profile. Stellar content lags the halo by about 2.5 Gyr (figure
`exp87_cross_epoch`).

**What it means.** For a mass estimate, measure the outskirts: the
stellar density near 140 kpc is the single best halo-mass proxy in the
profile, and no machine learning is needed beyond a two-term formula and
an honest error distribution. For assembly, the profile is not enough by
itself; it becomes informative only once the halo mass is known from
somewhere else (lensing, dynamics), which is a concrete statement about
what a combined probe would buy.

## The statistics (Stage 1 and the extras)

**Estimand.** At z = 0.4 the parent sample IS the population of TNG300
centrals with M200c >= 10^13 (3388 in the box, 3388 in the parent table,
3380 with a finite curve of growth; Stage 0), so ordinary proper scores on
it are unbiased for that population. The cut breaks three other things:
the predictive family (support [13, inf), a mean that bends near the
cut), transfer to differently selected populations, and the comparability
of variance-normalised numbers. Hence: one predictive contract for every
method — a truncated normal with latent (m, s), fitted by the truncated
likelihood (`heads.TruncLinear`; flexible learners through
`heads.CommonHead`) — the truncated CRPS as the primary score
(`scoring.TruncNormal.crps`, closed form, checked against quadrature to
2e-5 dex and against Monte Carlo), no R^2 or correlation in a headline,
calibration read in bins of the PREDICTION, never of the truth.

**The mechanics gate (`outputs/stage1_gate.log`), all six checks passed**
on a synthetic world with a known latent relation, a mass-function prior
and the same cut:

| check | result |
| --- | --- |
| A the oracle beats wrong predictives | ordinary normal +16.8%, ordinary renormalised above the cut +11.9% (truncation counted twice), the fitted truncated-linear model +0.13% |
| B the truncated likelihood recovers the truth | slope vector within 1.6%, scale within 1.4% (the ordinary fit: slopes 0.73, scale 0.84 of the truth) |
| C calibration | PIT mean 0.5000, variance 0.0838 (uniform: 0.0833) |
| D no false nonlinearity | flexible learners gain 0.0% at the 95th percentile on a linear truth: delta = 1% |
| E power | an injected curvature worth 3% of the CRPS detected in 10 of 10 replicates |
| F a selection that depends on the input | biases the fitted relation by +0.083 dex, as it must |

**On real data (`outputs/stage3_extras.log`).**

| reading | stellar mass alone | 24-point profile |
| --- | ---: | ---: |
| ordinary normal, CRPS | 0.1054 | 0.0853 |
| ordinary, renormalised above the cut | 0.1044 | 0.0839 |
| truncated likelihood | 0.0875 | 0.0762 |
| gain of the truncated fit | 17.0% [15.2, 18.9] | 10.8% [9.3, 12.4] |
| scatter: latent vs ordinary fit | 0.221 vs 0.182 dex | 0.171 vs 0.144 dex |
| predictive mass below the cut, ordinary fit | 9.1% | 8.4% |

- The safe-zone check: among the galaxies whose inputs alone put less
  than 1% of the latent distribution below the cut (24% / 33% of the
  sample), trained and scored inside the zone, the ordinary and truncated
  fits agree to 0.00%. A GLOBAL ordinary fit read in that zone is 13–31%
  worse: it is biased everywhere, not only near the cut.
- The prior is explicit. Forward relation M*(<148) = 0.675 log Mh + const
  with scatter 0.155 dex, so a flat prior would give a resolution of
  0.229 dex; the box's mass function falls as 10^(−1.13 log M) just above
  the cut, which puts the population's mean halo mass at fixed stellar
  mass 0.137 dex BELOW the inverted forward relation. The generative
  inverse (forward fit + prior; `generative.py`) reproduces the
  truncated-likelihood model (0.0757 vs 0.0744) and gives the same score
  with the box prior as with the sample's own, as it must on a complete
  sample.
- Projection: the predicted halo mass moves by 0.044 dex between the three
  projection axes, against an error of 0.144 dex; a model trained on one
  axis scores 0.0780 / 0.0792 / 0.0788 on xy / xz / yz.

**The selection report (`outputs/stage2_selection.log`).**

- At z >= 0.7 the sample is the main progenitors, growth-selected at low
  mass. Above the completeness cuts the selection shifts no component of
  the curve of growth by more than 0.030 forward scatters (tolerance 0.2),
  and adding the curve of growth to the mass improves a prediction of the
  halo's future growth by at most 0.016 in R^2: the "complete above the
  cut" rows are a population claim under that tested assumption.
- The curated samples used by the forward experiments are optimistic for
  this problem: forward resolution 0.208 dex on the parent against 0.185
  on the curated sample (the haloes dropped for a declined accretion
  history sit +0.080 dex in stellar mass at fixed halo mass). On the
  ladder the curated sample scores 4% better than the parent.
- Inverse-completeness weighting is not usable over the whole progenitor
  sample (effective size 8–43 of ~1900) and unnecessary above the cuts
  (99%).

## Halo mass (Stage 3; `outputs/stage3_tables.log`, `stage3_decide.log`)

CRPS / RMSE of the predictive mean, dex, out of fold on the development
galaxies. "asis" = the progenitors as they are (plain normal predictive);
"complete" = above the epoch's completeness cut (truncated there).

| population | n | nothing | M*(<148) | outer shell | M* + shell | mass + sizes | 24 cumulative | 24 shells | shells + quadratic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent, z = 0.4 | 2704 | 0.170 / 0.321 | 0.0875 / 0.162 | 0.0784 / 0.145 | 0.0763 / 0.141 | 0.0799 / 0.147 | 0.0762 / 0.141 | **0.0744 / 0.137** | 0.0744 / 0.137 |
| curated, z = 0.4 (input-only cuts) | 1922 | 0.175 | 0.0833 | 0.0754 | 0.0727 | 0.0763 | 0.0726 | 0.0714 / 0.130 | 0.0718 |
| curated, z = 0.4 (the 2356 fitting sample) | 1892 | 0.174 | 0.0834 | — | — | 0.0764 | 0.0729 | 0.0715 / 0.131 | — |
| complete, z = 0.7 | 1441 | 0.164 | 0.0808 | 0.0721 | 0.0682 | 0.0731 | 0.0689 | **0.0661 / 0.124** | 0.0666 |
| complete, z = 1.0 | 1156 | 0.150 | 0.0810 | 0.0707 | 0.0656 | 0.0704 | 0.0679 | **0.0642 / 0.117** | 0.0643 |
| complete, z = 1.5 | 932 | 0.138 | 0.0858 | 0.0691 | 0.0641 | 0.0718 | 0.0684 | **0.0618 / 0.114** | 0.0628 |
| complete, z = 2.0 | 684 | 0.122 | 0.0824 | 0.0655 | 0.0593 | 0.0655 | 0.0630 | **0.0560 / 0.105** | 0.0574 |
| asis, z = 0.7 | 1918 | 0.187 | 0.0959 | 0.0909 | 0.0836 | 0.0912 | 0.0782 | 0.0825 | **0.0737 / 0.134** |
| asis, z = 1.0 | 1914 | 0.191 | 0.0995 | 0.0997 | 0.0875 | 0.0907 | 0.0774 | 0.0813 | **0.0737 / 0.133** |
| asis, z = 1.5 | 1917 | 0.197 | 0.1048 | 0.182 | 0.1022 | 0.0890 | 0.0792 | 0.0834 | **0.0735 / 0.135** |
| asis, z = 2.0 | 1914 | 0.197 | 0.0981 | 0.154 | 0.0962 | 0.0809 | 0.0725 | 0.0767 | **0.0663 / 0.123** |

Read:

1. **The profile adds 15% to stellar mass at z = 0.4** (0.0875 → 0.0744,
   paired interval [13.0, 16.9]%) and 18–32% at higher redshift in the
   complete populations. Sizes (R20, R50, R80) recover about half of it;
   one outer shell recovers almost all of it.
2. **The gain is the outer envelope.** The two outermost cumulative points
   (132 and 148 kpc) alone give 0.0776; everything inside 30 kpc gives
   0.0943. The outer shell's weight in the two-number latent relation
   (0.62) exceeds the total mass's (0.38).
3. **The relation is linear in the log shell masses.** In every complete
   population the nonlinearity test N (best curvature or flexible model
   over the best linear one, both by truncated likelihood) is not
   significant: +0.05% [−0.4, +0.6] on the parent, −0.3% to −1.8% at
   z = 0.7–2. On log CUMULATIVE masses the same test read +1.6% to +5.5%
   at z >= 1: a coordinate effect. Boosting, a random forest, a neural
   network and a Gaussian process on the raw points are all WORSE than
   the linear model (0.085–0.090): they cannot find the fine linear
   combination that isolates the outer shell.
4. **Curvature survives only in the as-is progenitor samples** (9–14%
   over the linear shell model), which is where the sample has a soft,
   growth-selected lower edge and the smallest progenitors have almost no
   stars at 140 kpc. The as-is rows describe "progenitors of z = 0.4
   massive haloes", not a population of haloes at that redshift.
5. **Calibration.** Coverage of the 68 / 90% intervals 0.70 / 0.90,
   truth-on-prediction slope 1.00, errors beyond 0.5 dex in 0.3% of
   galaxies (figure `exp87_calibration`).

## Concentration and formation time (Stage 3, the oracle extension)

Parent, z = 0.4; CRPS / RMSE.

| target | nothing | true Mh alone | M* alone | the profile (best) | profile + true Mh | skill: profile, profile + Mh |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| log c200c [dex] | 0.0718 / 0.133 | 0.0721 / 0.132 | 0.0706 / 0.130 | 0.0660 / 0.122 | 0.0566 / 0.106 | 8%, 21% |
| t50 [Gyr] | 0.816 / 1.43 | 0.811 / 1.41 | 0.802 / 1.40 | 0.731 / 1.29 | 0.622 / 1.11 | 10%, 24% |
| t75 [Gyr] | 0.763 / 1.33 | 0.756 / 1.32 | 0.748 / 1.31 | 0.717 / 1.26 | 0.584 / 1.05 | 6%, 23% |
| t90 [Gyr] | 0.641 / 1.16 | 0.643 / 1.15 | 0.646 / 1.15 | 0.630 / 1.13 | 0.532 / 0.98 | 2%, 17% |

The profile with the true halo mass beats the true halo mass alone by
21.5% [19.4, 23.5] (concentration) and 22.6% [20.5, 24.5] (t50), and beats
the profile alone by 14–16%. Formation time is the one scalar target with
real curvature (shells + quadratic over shells: 4.3%).

## The accretion history (Stage 5; `outputs/stage5_mah.log`)

Target: the pre-epoch history's shape D(tau) = log Mpeak(t − tau) / M(t)
at lookbacks 0.5–6 Gyr (never anything after the epoch; the official
DiffMAH fit sees the future and is not a target). RMS error of the whole
curve, dex:

| predictor | z = 0.4 | 0.7 | 1.0 | 1.5 | 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| knowing nothing | 0.165 | 0.225 | 0.215 | 0.260 | 0.252 |
| the profile's own mass estimate | 0.160 | 0.217 | 0.210 | 0.255 | 0.249 |
| the true mass alone | 0.161 | 0.216 | 0.206 | 0.249 | 0.246 |
| the profile (ridge) | 0.155 | 0.204 | 0.194 | 0.233 | 0.225 |
| the profile + the true mass | 0.130 | 0.189 | 0.177 | 0.211 | 0.208 |
| the stellar history (all earlier epochs) | 0.127 | 0.175 | 0.157 | 0.210 | — |
| via the three DiffMAH shape parameters | 0.160 | 0.211 | 0.203 | 0.240 | 0.231 |
| floor of the DiffMAH form itself | 0.088 | 0.106 | 0.104 | 0.113 | 0.120 |

- The profile beats its own mass estimate at every epoch (leading
  component +4.8 / +8.5 / +9.2 / +11.4 / +12.5%, all significant): it knows
  something about the history that is not the mass.
- With the true mass, +26.5% at z = 0.4 and +15–20% above; the stellar
  history across epochs adds 13–26% over the single-epoch profile.
- Linear is enough: boosting is 4–7% WORSE than ridge. Predicting the
  DiffMAH parameters and reading the curve is 5–9% worse than predicting
  the curve directly. Shell masses change nothing here (0.1552 vs 0.1545).

## Across epochs: the z = 0.4 profile and the halo's EARLIER mass (Stage 7; `outputs/stage7_cross_epoch.log`)

Added at the user's question after the first report (which covered only
same-epoch predictions at high redshift). Target: the main progenitor's
log M200c at z = 0.7 / 1.0 / 1.5 / 2.0 for the galaxies selected at z = 0.4
(no cut on the target: a plain normal predictive; the curated sample;
development folds; the lockbox untouched). CRPS / RMSE, dex; figure
`exp87_cross_epoch`.

| input | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
| --- | ---: | ---: | ---: | ---: |
| nothing | 0.187 / 0.342 | 0.191 / 0.346 | 0.197 / 0.357 | 0.197 / 0.355 |
| M*(<148) at z = 0.4 | 0.0796 / 0.143 | 0.0815 / 0.147 | 0.1127 / 0.203 | 0.1408 / 0.251 |
| the TRUE halo mass at z = 0.4 | 0.0721 / 0.131 | 0.1006 / 0.179 | 0.1292 / 0.231 | 0.1472 / 0.263 |
| the z = 0.4 profile (shells + quadratic) | **0.0644 / 0.117** | **0.0706 / 0.129** | 0.1029 / 0.187 | 0.1272 / 0.229 |
| the profile at that epoch | 0.0737 / 0.134 | 0.0737 / 0.133 | **0.0735 / 0.135** | **0.0663 / 0.123** |
| both profiles | 0.0636 / 0.115 | 0.0616 / 0.111 | 0.0661 / 0.121 | 0.0645 / 0.121 |
| z = 0.4 profile + true Mh(z = 0.4) | 0.0536 / 0.099 | 0.0692 / 0.128 | 0.1027 / 0.187 | 0.1266 / 0.227 |

1. **The z = 0.4 profile predicts the halo mass at z = 0.7 better than
   the z = 0.7 profile does** (0.0644 vs 0.0737, RMSE 0.117 vs 0.134 dex),
   matches it at z = 1.0, and loses to it at z >= 1.5 (0.103 vs 0.074,
   0.127 vs 0.066), where most of the z = 0.4 stars had not yet formed or
   arrived.
2. **At every earlier epoch the z = 0.4 profile beats the TRUE z = 0.4 halo
   mass as a predictor of the progenitor's mass** (by 11% at z = 0.7, 30%
   at z = 1.0, 20% at z = 1.5, 14% at z = 2.0). The stars are a better
   record of where the halo WAS than the halo's own present mass is.
3. **The lag scan** (the halo's mass at t(z = 0.4) − tau from the z = 0.4
   profile): the skill over knowing nothing rises from 0.59 at tau = 0
   (the truncated-likelihood Stage 3 value) and 0.59 / 0.61 / 0.63 at
   0.5 / 1 / 1.5 Gyr to a PEAK of 0.66 at tau = 2.5 Gyr (RMSE 0.115 dex),
   then falls to 0.59 at 4 Gyr and 0.37 at 6 Gyr. Total stellar mass alone
   peaks at the same lookback. The profile overtakes the true z = 0.4 halo
   mass from tau = 2 Gyr on. Caveat: at tau <= 1.5 Gyr the target still has
   a near-hard lower edge and the plain normal is somewhat misspecified, so
   the rise between 0 and 2 Gyr is less certain than the peak itself (the
   2.5 Gyr value exceeds the correctly-truncated tau = 0 value).
4. **What it knows is the growth.** Added to the true z = 0.4 mass, the
   z = 0.4 profile improves the z = 0.7 / 1.0 progenitor mass by 26% / 31%
   (it knows how much the halo grew recently); at z >= 1.5 the true mass
   adds nothing to the profile (0.1027 vs 0.1029).
5. **Two epochs are better than one**: adding the z = 0.4 profile to the
   same-epoch profile gains 14 / 17 / 10 / 3%; both together reach
   0.062–0.066 dex at every epoch.

Reading: the stellar content at z = 0.4 tracks the halo mass of about
2.5 Gyr earlier (z ≈ 0.75), not the present one — the inverse-direction,
model-free counterpart of the forward model's deposition delay (exp81/82:
the data's stars lag the halo's growth; tau_d = 0.15 Hubble times), and of
exp81's finding that the forward residual correlates with recent growth.
It also re-reads the headline: part of the 0.137 dex error on the z = 0.4
halo mass is not noise but recent halo growth that the stars have not yet
recorded.

## Symbolic regression (Stage 4; `outputs/stage4_*.log`, `outputs/sr/`)

PySR, nested in the folds, on the pseudo-latent response (so the search
cannot spend itself on the selection's hockey stick). Entry rule: N at
z = 0.4 not significant → the first operator stage only, residual mode.
Other cells were run with the same rule and are EXPLORATORY.

| cell | stage, mode, inputs | CRPS vs baseline | gain | recurs | accepted |
| --- | --- | --- | --- | --- | --- |
| halo mass, parent (the entry cell) | S1 residual, summaries / pca4 / raw24 | 0.0757 / 0.0763 / 0.0752 vs 0.0762 | +0.7% / −0.2% / +1.2% | no | no |
| halo mass, complete z = 2 | S1 residual, summaries / pca4 / raw24 | 0.0604 / 0.0623 / 0.0600 vs 0.0630 | +4.1% / +1.0% / +4.7% | no | no |
| halo mass, parent, shell-aware | S1, S2 direct on seven shell masses | 0.0751 / 0.0751 vs 0.0744 | −1.0% / −1.1% | predictions agree to 0.999, forms differ | no |
| halo mass, asis z = 2, shell-aware | S1, S2 residual on seven shell masses | 0.0674 / 0.0675 vs 0.0767 | +12.1% / +12.0% | no (0.85–0.89) | no |
| formation time t50, parent | S1 residual (3 inputs); S2 residual, direct | 0.748–0.749; 0.745, 0.751 vs 0.765 | +2.1–2.3%; +2.7%, +1.9% | no | no |
| concentration, parent | S1 residual (3 inputs); S2 residual, direct | 0.0663 / 0.0658 / 0.0659; 0.0659, 0.0671 vs 0.0670 | +1.0% / +1.7% / +1.6%; +1.5%, −0.2% | no | no |

Read: the search finds real signal where there is some (the coordinate
effect at z = 2, the soft edge of the as-is sample, the curvature of t50)
and none where there is none (the parent). What it writes down on
cumulative inputs is, fold after fold, a squared outer mass fraction —
`square(m148 − m61 − 0.04)`, `square(m148 − 0.05 − m80)`,
`square((m132 − m148) + 0.012)` — the polynomial shadow of the shell
coordinate. Richer operators (division, sqrt, log, exp) never beat the
polynomial stage, so the third stage (pow, tanh, max, min) was not run
anywhere. No formula is adopted; the two-term linear relation above is the
compact result, and it comes from the radius reading, not from the search.

## The lockbox (Stage 6; `outputs/lockbox.json`)

676 parent galaxies (473 curated) were set aside at Stage 0 and scored
once, after every other decision, with each headline model refitted on all
the development galaxies. 1 of 76 scores falls outside the 99% interval the
development folds predict for a sample of that size (0.8 expected by
chance), and that one is BETTER than expected (complete z = 1.5,
cumulative + quadratic: 0.0546 against [0.0564, 0.0750]). The pass rule
was fixed before scoring (`outputs/decisions.jsonl`).

| reading (CRPS, dex unless stated) | development | lockbox |
| --- | ---: | ---: |
| halo mass, parent: knowing nothing | 0.1703 | 0.1700 |
| stellar mass alone | 0.0875 | 0.0894 |
| the outer shell alone | 0.0784 | 0.0793 |
| stellar mass + outer shell | 0.0763 | 0.0769 |
| 24 cumulative masses, linear | 0.0762 | 0.0780 |
| 24 shell masses, linear | 0.0744 | 0.0742 (RMSE 0.136) |
| shells + quadratic | 0.0744 | 0.0737 |
| halo mass, complete z = 1.0 / 2.0, 24 shells | 0.0642 / 0.0560 | 0.0585 / 0.0506 |
| concentration: nothing / profile / profile + true Mh | 0.0718 / 0.0662 / 0.0566 | 0.0702 / 0.0670 / 0.0583 |
| t50 [Gyr]: nothing / profile / profile + true Mh | 0.816 / 0.744 / 0.628 | 0.822 / 0.748 / 0.648 |
| history at z = 0.4, curve RMS: nothing / profile / profile + true Mh | 0.165 / 0.155 / 0.130 | 0.163 / 0.152 / 0.128 |

On the lockbox the profile's gain over stellar mass is 17%; coverage of
the 68 / 90% intervals 0.71 / 0.90. Every claim above survives.

## Caveats

- One simulation, one box: galaxies are clustered, so the bootstrap
  intervals are slightly optimistic; nothing here says the same numbers
  hold in another model of galaxy formation.
- The relation below the cut is not observed. The latent relation is the
  population's only as far as the truncated-normal family holds below
  10^13; no claim is made for unselected haloes.
- What the outer shell IS (smooth intracluster light, or satellites left
  in the map by the measurement's satellite treatment) is not decided
  here; either way it is in the measured curve of growth, and either way
  it traces the halo.
- The parent = box identity (the lower edge is exactly M200c(snapshot 72)
  = 10^13.000 and every box central above it is in the table) bears on the
  open provenance question in `doc/paper1_statistical/SAMPLE_PROVENANCE.md`.
  exp87 does not edit that file.
- Concentration is a noisy target for unrelaxed haloes; its skill ceiling
  is unknown.

## Code and artifacts

`config.py` (every constant; its hash is on each scoreboard row),
`data.py` (samples, features, the frozen split, the box prior),
`scoring.py`, `heads.py`, `methods.py`, `generative.py`, `harness.py`,
`synthetic.py` (Stage 1), `stage0_setup.py`, `stage2_selection.py`,
`stage3_ladder.py`, `stage3_extras.py`, `report.py`, `sr.py`,
`stage4_driver.py`, `mah.py`, `lockbox.py`, `figures.py`. Outputs
(gitignored): `outputs/scoreboard.jsonl` (one row per cell),
`outputs/cells/` (per-galaxy scores), `outputs/decisions.jsonl`, the stage
logs, `outputs/formulas.json`.
