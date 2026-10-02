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

**Can a formula do it? (Stage 8, added at the user's request.)** With the
robustness rule dropped on purpose, a longer and richer symbolic search was
asked for the formulas that turn the z = 0.4 stellar mass distribution into
the halo's mass at each of five epochs, with the distribution written two
ways: as the mass in a central aperture and a series of annuli, and as the
few parameters of a fitted curve of growth. Three things came out. (1) The
search finds short formulas that are as good as the best linear models and
no better: nothing in 35 searches beats the 24-shell reference at any
epoch. (2) The annuli beat every parametrised curve at z <= 1 (by 6, 4 and 3
per cent), because a smooth fit irons out the outermost annulus, and a
Sersic fit is barely better than total stellar mass alone; by z >= 1.5 the
two ways are equal. (3) Every good formula is built on the same two
ingredients, the stellar mass inside 5 kpc and the mass in the outermost
annulus, and the balance between them shifts with epoch: the outermost
annulus is the best single predictor of the halo's mass at z = 0.4 and 0.7,
the 52-80 kpc annulus at z = 1.0 and 1.5, and the innermost 5 kpc at z = 2.
The centre of the z = 0.4 galaxy remembers the halo as it was at z = 2; the
outskirts track the halo as it is now (figure `exp87_explore`).

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

## Exploratory formulas across epochs (Stage 8; `outputs/stage8_report.log`)

Added at the user's request after Stage 7 (plan
`doc/plans/2026-10-02-exp87-stage8-exploratory-sr.md`). **Exploratory by
design**: Stage 4's rule that a formula must recur across folds is dropped,
the searches are longer (900 s on four threads each, against Stage 4's 420 s
on one) and richer (`+ - * / pow max min`, `square cube sqrt log exp tanh`,
up to 45 symbols). Nothing here is an accepted formula in Stage 4's sense.

**What goes in.** Always the z = 0.4 curve of growth, written two ways.
Stellar masses are log10(M / 10^10 Msun), radii log10(R / kpc), and the
target is h = log10(M200c / 10^13 Msun).

| set | approach | variables |
| --- | --- | --- |
| `annuli6` | 1, the profile | `c10` (mass inside 10 kpc), `a10_30`, `a30_50`, `a50_100`, `a100_132`, `a132_148` (mass in each annulus, kpc) |
| `annuli9` | 1, the profile | `c5` (inside 4.9 kpc), `a5_10`, `a10_19`, `a19_33`, `a33_52`, `a52_80`, `a80_103`, `a103_132`, `a132_148` |
| `sersic` | 2, a fitted curve | one Sersic profile: `m_tot`, `lr_e` (effective radius), `n_ser` |
| `hill` | 2 | a logistic in log R: `m_inf`, `lr_h` (half-mass radius), `a_hill` (steepness) |
| `double` | 2 | inner Sersic + outer exponential: `m_tot`, `f_out` (outer mass fraction), `lr_in`, `n_in`, `lr_out` |
| `logpoly` | 2 | a cubic in u = log10(R / 20 kpc): `m20` (mass inside 20 kpc), `a1` (slope), `a2`, `a3` |
| `sizes` | 2 (not a fit) | `m148` (mass inside 148 kpc), `lr20`, `lr50`, `lr80` (radii enclosing 20, 50, 80 per cent) |

How well the fitted curves describe a curve of growth (`cogparams.py`,
figure `exp87_cog_families`): the median rms residual over the 24 radii is
0.014 dex for the Sersic and the logistic, 0.007 for the cubic and 0.006
for the two-component fit; 3.5 per cent of the Sersic indices sit at a
bound.

**What comes out, and how it is scored.** Targets as in Stage 7: z = 0.4 on
the complete parent (truncated at 10^13, so the search runs on Stage 4's
pseudo-latent response and the score is the truncated CRPS); z >= 0.7 is the
main progenitor's mass on the curated sample. One search per input set and
epoch, 35 in all. Folds 0-2 run the search, fold 3 ranks the formulas, fold
4 scores them exactly as found (the strict number). The "refit" score keeps
a formula's shape, refits its constants on four folds and scores the fifth,
over all five folds: it is comparable with Stage 7's tables and mildly
optimistic, because the shape was chosen on folds 0-3. "Three best" = the
three lowest ranking-fold errors among formulas that are different functions
(predictions differing by more than a quarter of the residual scatter), per
epoch and approach; "compact" = the shortest formula within 2 per cent of
the best. A formula with a pole inside the data (it divides by something
that crosses zero, ranks well on one fold and fails on another) is not
ranked. The test: its refit error or its fold-4 error exceeds 1.1 times
that of a straight line in total stellar mass. It removes 8 of 380 formulas
on the annuli and 96 of 903 on the fitted curves (the trivial one-to-three
symbol formulas included); fold 4 is therefore used once, as a blow-up
filter; every formula reported below is 3 to 19 per cent under that line.
Development folds only; the lockbox was not touched.

### The result in one table (CRPS, dex; refit / fold 4)

| model | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stellar mass inside 148 kpc, a line | 0.0875 / 0.0852 | 0.0796 / 0.0744 | 0.0815 / 0.0795 | 0.1127 / 0.1105 | 0.1408 / 0.1401 |
| 24 shell masses, linear | 0.0744 / 0.0743 | 0.0702 / 0.0669 | 0.0770 / 0.0732 | 0.1074 / 0.1047 | 0.1298 / 0.1332 |
| 24 shell masses, linear + quadratic | 0.0744 / 0.0746 | **0.0644** / 0.0619 | **0.0706** / 0.0662 | **0.1029** / 0.1013 | **0.1272** / 0.1311 |
| nine annuli, linear | 0.0747 / 0.0745 | 0.0708 / 0.0673 | 0.0775 / 0.0733 | 0.1094 / 0.1065 | 0.1321 / 0.1337 |
| nine annuli, linear + quadratic | 0.0747 / 0.0740 | 0.0650 / 0.0613 | 0.0714 / 0.0661 | 0.1056 / 0.1035 | 0.1302 / 0.1324 |
| **best formula on the annuli** | 0.0745 / 0.0741 | **0.0644** / 0.0624 | 0.0721 / 0.0688 | 0.1051 / 0.1049 | 0.1298 / 0.1342 |
| its compact version | 0.0748 / 0.0741 | 0.0656 / 0.0626 | 0.0734 / 0.0699 | 0.1058 / 0.1047 | 0.1329 / 0.1351 |
| best linear + quadratic on a fitted curve | 0.0797 / 0.0787 | 0.0677 / 0.0647 | 0.0741 / 0.0710 | 0.1045 / 0.1053 | 0.1289 / 0.1335 |
| **best formula on a fitted curve** | 0.0793 / 0.0792 | 0.0671 / 0.0662 | 0.0744 / 0.0723 | 0.1054 / 0.1073 | 0.1304 / 0.1367 |
| its compact version | 0.0796 / 0.0792 | 0.0695 / 0.0661 | 0.0757 / 0.0743 | 0.1073 / 0.1087 | 0.1322 / 0.1367 |

1. **The formulas reach the linear + quadratic model on the same inputs and
   stop there.** On the annuli the best formula is within 1 per cent of the
   nine-annulus linear + quadratic model at every epoch on the refit score,
   and 0 to 4 per cent WORSE than it on the strict fold-4 score. No formula
   beats the 24-shell linear + quadratic reference anywhere; at z = 0.7 one
   ties it (0.0644). At z = 0.4 the relation is a straight line and the
   formulas say so: the best is 0.0745 against 0.0747 linear.
2. **A dozen symbols is enough.** On the annuli the compact formulas (7 to
   19 symbols) are within 2 per cent of the best ones on fold 4. Panels (a)
   to (e) of the figure: the error stops falling at 10 to 20 symbols.
3. **The annuli beat the fitted curves at z <= 1 and tie them above.** Best
   formula, annuli against fitted curve: 0.0745 vs 0.0793 (6.4 per cent) at
   z = 0.4, 0.0644 vs 0.0671 (4.2) at z = 0.7, 0.0721 vs 0.0744 (3.2) at
   z = 1.0, then 0.1051 vs 0.1054 and 0.1298 vs 0.1304. A smooth fit irons
   out the outermost annulus, which is where the low-redshift information
   is (Stage 3). At z = 0.4 the Sersic and logistic fits (0.0859, 0.0870)
   are barely better than total stellar mass (0.0875); the families that
   keep the outer extent do better: mass + three radii 0.0793, the
   two-component fit 0.0791, the cubic 0.0815.
4. **A longer search does not help.** The nine-annulus search repeated at
   3600 s instead of 900 s: at z = 0.4 the best formula scores 0.0746 /
   0.0742 against 0.0745 / 0.0741; at z = 1.0, 0.0715 / 0.0691 against
   0.0721 / 0.0688, with the same error on the ranking fold (0.1367 dex rms
   both times) although the fit to the search folds improved (0.1314 to
   0.1296). Extra search time buys a closer fit to the galaxies searched
   on, not a better prediction. An earlier 60 s against 300 s check read
   the same way.

### The three best formulas per epoch

`^` is a power; `tanh`, `sqrt`, `exp`, `log` (natural), `square`, `cube`,
`max`, `min` as usual. Constants are the ones refitted on all development
galaxies. Scores: refit CRPS / fold-4 CRPS, dex.

**Halo mass at z = 0.4** (24 shells: 0.0744 / 0.0743)

    profile, 1  [annuli9, 12 symbols, 0.0745 / 0.0741]
        h = 0.7142 * (a132_148 + max(sqrt(c5 + max(0.211, a132_148)), a33_52))
    profile, 2  [annuli6, 19 symbols, 0.0747 / 0.0743]
        h = 0.8225 * (0.4928*c10 + 0.4097 + a132_148) - 0.2267 * (tanh(a100_132) - a50_100 + tanh(a30_50))
    profile, 3  [annuli9, 6 symbols, 0.0754 / 0.0751]
        h = tanh(c5) + 0.6285 * a132_148
    profile, compact  [annuli9, 7 symbols, 0.0748 / 0.0741]
        h = 0.7161 * a132_148 + tanh(sqrt(c5))

    fitted, 1  [sizes, 24 symbols, 0.0793 / 0.0792]
        h = cube(log(1.696^lr80 + m148 - sqrt(max(lr20 + min(lr50 - 0.4083, 0.6002), max(0.4621*lr80, lr50))))) - 0.6535
    fitted, 2  [double, 39 symbols, 0.0791 / 0.0797]
        h = max(-0.5492, min(3.377, (1.868 * min(max(0.5855*lr_out, lr_in), 0.971))^lr_out) * m_tot / (lr_out + 0.5351)
                         + min(0.505^lr_out, min(f_out, 0.1502 / min(lr_in, m_tot)) * (f_out + 0.63)) - 1.444)
    fitted, 3  [sizes, 8 symbols, 0.0798 / 0.0790]
        h = (lr80 - 0.6416) * m148 - sqrt(lr50)
    fitted, compact  [sizes, 16 symbols, 0.0796 / 0.0792]
        h = cube(log(m148 + 1.696^lr80 - sqrt(max(0.4501*lr80, lr50)))) - 0.6678

**Halo mass at z = 0.7** (24 shells + quadratic: 0.0644 / 0.0619)

    profile, 1  [annuli9, 32 symbols, 0.0644 / 0.0624]
        h = 0.2567 * (max(0.5375, exp(a132_148)) + c5
                      + max(max(a132_148 - cube(0.4136 / c5), -1.193) + min(a52_80, min(a103_132 + 0.5013, a19_33)),
                            cube((a132_148 - a5_10) / 0.612)))
    profile, 2  [annuli6, 12 symbols, 0.0659 / 0.0632]
        h = 0.1777 * a50_100 + 0.5361 * (c10 / 1.252 + tanh(a132_148))
    profile, 3  [annuli6, 33 symbols, 0.0656 / 0.0636]
        h = min(max(tanh(a132_148), 0.3085 * min(c10 * a50_100, a100_132 + 0.619) - 0.04435), 0.3159 * cube(a10_30))
            + 0.3378 * (max(-0.8332, a132_148) + max(max(c10, 0.6425), a100_132 + 0.5793))
    profile, compact  [annuli9, 13 symbols, 0.0656 / 0.0626]
        h = 0.3822 * (exp(a132_148 + 0.2738) + tanh(c5 + a52_80 - 1.206))

    fitted, 1  [sizes, 45 symbols, 0.0671 / 0.0662]
        h = max((0.3735*m148 - 0.252) * lr80 * max(lr80, 1.455) - 1.026
                    + 0.7237 / exp(max(square(tanh(lr20)), square(lr80 - 0.9992) - 0.3156)),
                tanh(-1.546 / lr50 + cube(lr80 * lr20 / sqrt(m148))) / cube(lr80 / m148))
    fitted, 2  [logpoly, 32 symbols, 0.0686 / 0.0661]
        h = max(m20 * (tanh((a3 + cube(a2) + a2 + a1) / 0.6398) + min(cube(a2), max(cube(a3 / -0.5257), -0.06858)) + 0.7043) - 0.9291,
                square(a1) + a2)
    fitted, 3  [sizes, 12 symbols, 0.0686 / 0.0657]
        h = (m148 * min(0.4586*lr80, lr50))^1.333 - tanh(lr50)
    fitted, compact  [sizes, 10 symbols, 0.0695 / 0.0661]
        h = (0.4569 * m148 * lr80)^1.345 - tanh(lr50)

**Halo mass at z = 1.0** (24 shells + quadratic: 0.0706 / 0.0662)

    profile, 1  [annuli9, 17 symbols, 0.0721 / 0.0688]
        h = 0.3448 * (min(1.428 * min(a33_52, a19_33) - 0.9788, a132_148) + c5 + a52_80 * a10_19)
    profile, 2  [annuli6, 13 symbols, 0.0746 / 0.0707]
        h = 0.484 * (1.33^a132_148 * (min(a50_100, a30_50) + c10) - 0.9326)
    profile, 3  [annuli9, 24 symbols, 0.0720 / 0.0687]
        h = 0.3086 * (c5 + (a80_103 + 0.5075) * a19_33
                      + min(a132_148, 1.385 * (min(a52_80, a19_33 - square(a103_132 - a52_80)) - 0.6138)))
    profile, compact  [annuli9, 14 symbols, 0.0734 / 0.0699]
        h = 0.3325 * (c5 + min(1.805*a33_52 - 1.027, a132_148) + square(a33_52))

    fitted, 1  [sizes, 21 symbols, 0.0744 / 0.0723]
        h = min((lr80 - 2.024) * lr20, -0.1642) + min(m148 / (lr80 - 0.4307), 0.6487*lr80) * (m148 - 1.042)
    fitted, 2  [logpoly, 21 symbols, 0.0753 / 0.0735]
        h = 1.168 * (tanh(cube(a2 - 0.4139) + a1 + min(0.1321, a3)) + 0.7375) * max(0.7905, m20) - 1.105
    fitted, 3  [double, 36 symbols, 0.0756 / 0.0751]
        h = -0.2912 * max(f_out, tanh(lr_in)) / n_in
            + min(cube(0.911 * m_tot / lr_out), 0.2415*f_out + (log(m_tot / 2.227) - 0.6563 / max(min(lr_out, 1.625), 1.349)) * m_tot + m_tot)
    fitted, compact  [sizes, 13 symbols, 0.0757 / 0.0743]
        h = lr80 * (0.6673*m148 - 0.4749) - 0.2374*lr20 - 0.4535

**Halo mass at z = 1.5** (24 shells + quadratic: 0.1029 / 0.1013)

    profile, 1  [annuli9, 22 symbols, 0.1051 / 0.1049]
        h = c5^0.4659 - 1.453 + 0.5702 * exp(min(a19_33 + min(a52_80 - a5_10 / tanh(c5), -0.4754), tanh(a103_132)))
    profile, 2  [annuli6, 33 symbols, 0.1069 / 0.1057]
        h = (min(c10, max(0.5578, 1.292 + (1.082 + c10) * (a100_132 - a50_100))) - 0.1827) * (max(a100_132, tanh(a100_132)) - 1.179)
            + max(log(c10 + 0.5137), (c10 / 2.389) * a10_30)
    profile, 3  [annuli9, 7 symbols, 0.1092 / 0.1064]
        h = sqrt(exp(a80_103) * c5) - 0.9621
    profile, compact  [annuli9, 19 symbols, 0.1058 / 0.1047]
        h = c5^0.5173 - 1.409 + 0.5258 * exp(min(a19_33 - square(a52_80 - a5_10) - 0.4567, a103_132))

    fitted, 1  [sizes, 34 symbols, 0.1054 / 0.1073]
        h = tanh(lr80) * log(max(lr20, -0.4871 * (lr20 + 0.9273) / lr80 + m148))
            * min(1.79, max(2.759 / max(square(lr50), 0.4877) * (lr20 - 0.1552), 0.7777) * max(lr50, m148))
    fitted, 2  [logpoly, 23 symbols, 0.1048 / 0.1059]
        h = 0.7403 * (cube(a2 - 0.5655) - 0.6029 - square(a3 / 0.468) + m20 * (0.3966*m20 + a3 + a1))
    fitted, 3  [sersic, 22 symbols, 0.1055 / 0.1074]
        h = square(m_tot) * (0.345 - square(exp(max(0.5665 / (m_tot * n_ser), 0.06805*lr_e) * lr_e) * 0.761 / m_tot))
    fitted, compact  [sizes, 10 symbols, 0.1073 / 0.1087]
        h = tanh(lr50) + (m148 - 2.101) * (lr20 + 0.6689)

**Halo mass at z = 2.0** (24 shells + quadratic: 0.1272 / 0.1311)

    profile, 1  [annuli9, 17 symbols, 0.1298 / 0.1342]
        h = min(a103_132 * a19_33, c5 - 1.026 - (a5_10 - a10_19)) + 0.2218 * a132_148 * a10_19
    profile, 2 and compact  [annuli9, 7 symbols, 0.1329 / 0.1351]
        h = c5 - 0.7807^(a132_148 * c5)
    profile, 3  [annuli6, 7 symbols, 0.1343 / 0.1359]
        h = c10 / 0.8198^a132_148 - 1.209

    fitted, 1  [logpoly, 12 symbols, 0.1304 / 0.1367]
        h = 1.712^m20 - square(1.673 * a3) - 1.895 + a2
    fitted, 2  [double, 14 symbols, 0.1299 / 0.1364]
        h = 1.635^m_tot - 2.012 - max(cube(f_out), tanh(square(lr_in) / n_in))
    fitted, 3  [sizes, 35 symbols, 0.1299 / 0.1364]
        h = (max(0.4162*lr20, m148 - 1.734) + 0.5897)
            * (min(max(lr50, tanh(m148)) + (lr50 / lr80 - lr20) / tanh(lr50), 1.145^lr80) * m148 - lr20 / lr80 - 1.757)
    fitted, compact  [logpoly, 7 symbols, 0.1322 / 0.1367]
        h = 1.729^m20 + a2 - 1.954

At z = 2 no formula beats the plain linear model on fold 4 (nine annuli
linear 0.1337, the cubic's four coefficients linear 0.1350).

### What the formulas have in common

The forms differ from epoch to epoch and from search to search, as Stage 4
found, but their ingredients do not (figure `exp87_explore`, panels g, h).

- **The centre and the outermost annulus, in every search on the annuli.**
  `c5` is in 100 per cent of the near-best nine-annulus formulas at every
  epoch; `a132_148` in 100 per cent at z <= 1.0, 45 per cent at z = 1.5
  and 61 per cent at z = 2. The straight line in those two numbers alone
  scores 0.0748 at z = 0.4 (all 24 shells: 0.0744) and 0.0732 at z = 0.7,
  and with curvature 0.0678 at z = 0.7.
- **The best single annulus moves inward with look-back time** (one annulus
  alone, a straight line, CRPS):

  | halo mass at | inside 5 kpc | 19-33 kpc | 52-80 kpc | 103-132 kpc | 132-148 kpc |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | z = 0.4 | 0.1242 | 0.0985 | 0.0876 | 0.0814 | **0.0784** |
  | z = 0.7 | 0.1254 | 0.1045 | 0.0886 | 0.0839 | **0.0838** |
  | z = 1.0 | 0.1267 | 0.1037 | **0.0934** | 0.0943 | 0.0960 |
  | z = 1.5 | 0.1362 | 0.1320 | **0.1255** | 0.1265 | 0.1278 |
  | z = 2.0 | **0.1423** | 0.1581 | 0.1524 | 0.1514 | 0.1515 |

  At z = 2 the mass inside 5 kpc at z = 0.4 is the best single predictor of
  the progenitor's halo mass and every annulus outside it is worse by 0.01
  to 0.02 dex; at z = 0.4 the centre is the worst and the outermost annulus
  the best. The inner stars are the old ones: they record the early halo.
  The envelope records the present one. This is Stage 7's lag, resolved in
  radius.
- **Low redshift wants the outer annulus in linear units.** The z = 0.7 and
  z = 1.0 formulas write `exp(a132_148)` or `1.33^a132_148`: a power of the
  outer mass itself (exponent 0.43 and 0.12) inside a logarithmic relation.
  That is the curvature the linear + quadratic model finds at z >= 0.7; at
  z = 0.4 the same variable enters as a plain line.
- **On the fitted curves the recurring term is mass times outer size.**
  `m148 * lr80` (z = 0.4, 0.7, 1.0): total mass multiplied by the log of the
  radius enclosing 80 per cent. At z = 2 the formulas switch to `m20` (the
  mass inside 20 kpc) plus the curve's curvature `a2`.

Reading: symbolic regression, given freedom and time, confirms Stage 3 and
Stage 7 rather than extending them. The relation between the z = 0.4
profile and the halo's mass is linear at z = 0.4 and gently curved before,
it needs two to four numbers from the profile, and which numbers depends
on the epoch asked about. The annulus table is the result to keep; the
formulas are compact ways of writing it, not unique ones. For a profile
described by a fitted curve, keep a parameter that measures the outer
extent (R80, an outer component's scale): a single Sersic fit loses most
of what the profile knows about the present-day halo.

Not established: none of these formulas was required to recur across
folds, the "three best" are ranked on one fold of about 370 to 540
galaxies, and differences below 1 per cent are inside the noise level set
by Stage 1's synthetic gate. One simulation, projected along one axis.

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
`stage4_driver.py`, `mah.py`, `lockbox.py`, `stage7_cross_epoch.py`,
`cogparams.py` and `stage8_explore.py` (Stage 8), `figures.py`. Outputs
(gitignored): `outputs/scoreboard.jsonl` (one row per cell),
`outputs/cells/` (per-galaxy scores), `outputs/decisions.jsonl`, the stage
logs, `outputs/formulas.json`.
