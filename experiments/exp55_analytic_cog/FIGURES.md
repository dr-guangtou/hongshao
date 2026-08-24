# exp55 figure guide

This guide has two parts. The first describes how well the selected analytic
curve of growth (CoG) represents a known galaxy profile. The second describes
the new held-out prediction of that analytic profile from halo assembly. The
fixed function in both parts is a compact Sérsic component with globally fixed
index `n = 1` plus an extended Moffat component with globally fixed
`gamma = 1.25`.

## Part I: profile representation

### What is being evaluated

Each of the 2,545 redshift-0.4 TNG300 CoGs is fitted independently with four
galaxy-level parameters: stellar mass within 148.2 kpc, compact mass fraction
within that aperture, compact Sérsic half-mass scale, and extended Moffat
half-mass scale. The two scales refer to each formally infinite family before
its cumulative function is renormalized at 148.2 kpc; they are not half-mass
radii of the aperture-truncated components.
The figures therefore test whether this analytic function can represent a
known stellar profile. They do not test whether DiffMAH or halo properties can
predict the four parameters.

The predictions used in the standard QA figures are fold-clean with respect to
the two globally shared shape indices: each galaxy is evaluated using the
Sérsic/Moffat index pair selected on the other folds. The per-galaxy profile
parameters still come from that galaxy's own CoG because representation, not
halo prediction, is the question in exp55.

### Main conclusions from the QA battery

- Cumulative aperture masses are reproduced extremely well. The scatter in
  `log10(model/truth)` is 0.005 dex inside 10 and 30 kpc, 0.006 dex inside
  50 kpc, and 0.003 dex inside 100 kpc. Median biases are within 0.1%.
- Differential outer masses are harder. The scatter is 0.024 dex in the
  10--30 kpc annulus, 0.041 dex in 30--50 kpc, and 0.073 dex in 50--100 kpc.
  The mass beyond 100 kpc has a +12.1% median bias and 0.161 dex scatter.
- The standard inner-versus-outer observational plane is nevertheless close:
  for `M*(<30 kpc)` versus `M*(50--100 kpc)`, the TNG slope/scatter
  `1.54/0.207 dex` becomes `1.48/0.180 dex`. The representation slightly
  compresses the outer diversity.
- The mass--R50 relation is nearly exact: its slope is 0.49 in both TNG and the
  analytic representation, with a +0.004 dex median R50 offset. R80 and R90
  are biased high by +0.008 and +0.017 dex, respectively, showing that the
  remaining discrepancy is concentrated toward the outer profile.
- The median maximum fractional CoG error per galaxy is 2.2% over all radii
  and 1.7% after excluding radii at or below 5 kpc.

These results support the analytic function as a compact CoG representation,
but they do not justify ignoring its outer-density limitation.

### Why a good CoG can have a visibly worse density residual

The CoG is cumulative, while the plotted surface density is derived from the
mass in each finite annulus:

```text
Sigma_j = [M*(<R_out,j) - M*(<R_in,j)]
          / [pi (R_out,j^2 - R_in,j^2)].
```

At large radius, the annular mass is a small difference between two much
larger enclosed masses. Consequently, a small change in the cumulative
residual between adjacent radii is amplified when those cumulative masses are
subtracted.

At the annulus centred near 97 kpc, the annulus contains a median of only 1.68%
of the mass enclosed by its outer edge. A 0.00175 dex median absolute CoG error
at that edge corresponds to a 0.0787 dex median absolute density error. At the
annulus centred near 140 kpc, the corresponding values are 1.09%, 0.00531 dex,
and 0.132 dex. Both TNG and model pass through the same finite-difference
operator, so this is a genuine local-slope limitation, not a rendering error.

### Figure-by-figure captions

### `exp55_qa_radial_summary`

Population radial summary. Panel A compares the median normalized TNG and
analytic CoGs. Panel C shows the signed CoG residual for each galaxy, summarized
by its median and 16th--84th percentile range. Panels B and D show the analogous
finite-annulus surface-density profiles and residuals on an `R^(1/4)` axis.
The model's median CoG remains within a few thousandths of a dex, while the
density residual and its galaxy-to-galaxy spread grow beyond roughly 70 kpc.

### `exp55_component_decomposition`

Compact and extended contributions for three galaxies spanning the TNG
2-kpc mass fraction. The top row shows each component's cumulative contribution
to `M*(<R)/M*,148`; the colored components add exactly to the green model total.
The bottom row shows the corresponding annular surface densities. The compact
term dominates the central density and rapidly becomes negligible, whereas the
extended Moffat term controls the outer profile. From diffuse to compact, the
examples have fitted compact fractions 0.20, 0.29, and 0.63; their compact
Sérsic half-mass scale parameters are 3.53, 2.51, and 2.70 kpc, while their
extended Moffat half-mass scale parameters are 32.1, 37.4, and 46.8 kpc.
“Compact” and “extended” are radial labels, not demonstrated in-situ and
ex-situ populations.

### `qa_dens_exp55_analytic_cog`

Exp54-style density QA split into terciles of redshift-0.4 halo mass, a model
input. TNG is solid and the analytic representation is dashed. The lower row is
the median log-density residual in dex. The positive outer residual is strongest
in the lowest halo-mass tercile, weaker in the middle tercile, and absent as a
population-median trend in the highest tercile. The full scatter remains larger
than these median curves show.

### `qa_bins_exp55_analytic_cog`

Median CoGs in the same halo-mass terciles. The upper row overlays TNG and model;
the lower row gives the fractional residual. The solid residual uses the fitted
amplitude, while the dotted curve renormalizes the model to the true outermost
mass and therefore isolates shape. In exp55 the amplitude itself is fitted from
the CoG, so the dotted curve is diagnostic only and is not the model definition.

### `qa_mass_kpc_aper_exp55_analytic_cog`

Per-galaxy predicted versus TNG cumulative stellar masses within 10, 30, 50,
and 100 kpc. The lower row shows fractional residual versus true aperture mass;
the large outlined point is the population median. This figure demonstrates
the excellent cumulative representation but should be read together with the
differential-mass figure.

### `qa_mass_kpc_diff_exp55_analytic_cog`

Per-galaxy annular and outer-envelope masses in physical kpc. Scatter increases
toward the outskirts because these masses are differences between similar
cumulative values. `M*(100--150 kpc)` is intentionally flagged because the CoG
ends at 148.2 kpc; the QA code does not silently treat 148.2 kpc as 150 kpc.
The `M*(>100 kpc)` panel is the clearest standard-QA view of the remaining
outer-profile limitation.

### `qa_mass_Re_aper_exp55_analytic_cog`

Cumulative masses inside 1, 2, and 4 TNG half-mass radii. The same TNG radius is
used for the model in these paired aperture tests, isolating mass-distribution
error from an error in the model's own size. All three apertures are recovered
with roughly 0.005 dex scatter.

### `qa_mass_Re_diff_exp55_analytic_cog`

Annular and envelope masses in TNG half-mass-radius units. Differential scatter
again grows for the outermost quantities, especially beyond 4 half-mass radii.

### `qa_planes_exp55_analytic_cog`

The three standard observational planes at redshift 0.4. TNG galaxies are
filled and the analytic representations are open. Each title reports the TNG
to model change in fitted slope and vertical scatter. The outer 50--100 kpc
plane shows the largest change, with scatter reduced from 0.207 to 0.180 dex.
Because exp55 fits each galaxy's profile, this is a representation check, not a
held-out halo-prediction plane.

### `qa_size_exp55_analytic_cog`

Panel A compares the TNG and model mass--R50 distributions using each profile's
own R50. Panel B gives the median model-minus-TNG size offset for R50, R80, and
R90 across stellar mass. The increasingly positive R80/R90 offsets at low and
intermediate mass are consistent with the outer-slope limitation.

### `qa_cdf_exp55_analytic_cog`

Population cumulative distributions of representative aperture and annular
masses. Solid curves are TNG and dashed curves are the analytic representation;
the lower row is the model-minus-TNG CDF. The 50--100 kpc annulus has the most
visible difference, but its distribution-distance ratios remain close to the
TNG split-half sampling floor.

### `qa_cases_exp55_analytic_cog`

Two best and two worst individual CoG fits, ranked by their largest fractional
error over the measured radii. The worst examples reach 15--20% local error,
even though the median galaxy's maximum error is 2.2%. These tails motivate
retaining a per-object fit-quality and conditioning flag.

### What should happen next

The next experiment should predict the four profile parameters from halo
properties in held-out galaxies. The same standard QA set should then be
repeated on those halo-predicted profiles. The expected errors will be much
larger than the representation errors measured here, and the total stellar mass
will become a genuine halo-model prediction rather than a parameter fitted from
the known galaxy CoG.

## Part II: held-out halo-to-analytic-profile map

### What is being evaluated

The halo map predicts all four galaxy-level analytic coordinates from the four
DiffMAH parameters and redshift-0.4 halo concentration. All predictions are
held out in five folds. The conditional mean measures how much galaxy-to-galaxy
structure is predictable from the halo; 32 correlated parameter draws per halo
represent the remaining conditional distribution. The relevant comparison is
therefore the TNG CoG itself, not the fitted analytic representation.

The map uses 2,539 galaxies rather than the 2,545 representation sample because
six galaxies lack a complete finite halo-feature vector. The standard QA plots
show the selected degree-2 model. The three experiment-specific figures add the
null controls, parameter-level interpretation, representation floor, and direct
Exp50 comparison.

### Main conclusions from the halo-map QA

- The analytic conditional mean and the Exp50 readable-coordinate conditional
  mean have effectively identical cumulative-profile accuracy. Their median
  per-galaxy CoG RMS errors differ by only -0.000058 dex in favor of the
  analytic map, and each wins for approximately half of the galaxies.
- The complete halo input lowers profile CRPS by 25.84% relative to final halo
  mass alone, by 13.20% relative to mass-conditioned shuffled DiffMAH shape,
  and by 4.44% relative to mass-conditioned shuffled concentration. Assembly
  and concentration therefore contribute distinct held-out information.
- Stellar mass within 148.2 kpc is strongly predictable from the halo
  (held-out R2=0.865), while compact fraction and compact radius are weakly
  predictable (R2=0.060 and 0.068). The reconstructed profile is more useful
  than a deterministic reading of each component parameter.
- The conditional mean is too narrow in the fixed-kpc population planes. The
  correlated draws restore and sometimes exceed the missing diversity. When
  each generated galaxy is measured at its own `Re`, the tight size-scaled
  `<2Re` versus `2--4Re` plane has 0.188 dex scatter, compared with 0.072 dex
  in TNG. The earlier 0.213 dex value used the TNG counterpart's true `Re` and
  is not a valid generated-population statistic.
- Pointwise profile intervals are nevertheless close to calibrated: the
  nominal 68% interval contains 68.3% of TNG profile points, and the nominal
  90% interval contains 87.4%. This is useful but demonstrably insufficient as
  the sole calibration test.

### Experiment-specific figures

#### `exp55_halo_map_skill`

Three matched bar charts compare final mass alone, real and shuffled assembly
features, linear and degree-2 means, the selected analytic map, and the Exp50
reference. The panels report mean profile CRPS, median cumulative-profile RMS,
and median density-profile RMS. The ordered control improvements establish the
incremental value of MAH shape and concentration, while the Exp50 bars show
that the analytic target changes interpretability and continuity more than raw
predictive accuracy.

#### `exp55_halo_map_parameters`

The upper row compares each fitted analytic coordinate with its held-out
conditional mean; the lower row shows residual versus truth. The total mass is
tightly predicted, extended radius has moderate information, and compact
fraction plus compact radius show strong regression toward their population
means. These panels explain why a good reconstructed CoG does not imply that
every component parameter is a well-determined halo outcome.

#### `exp55_halo_map_radial`

Median absolute profile and density errors are shown as functions of radius for
final mass alone, shuffled MAH shape, the selected analytic map, the analytic
representation floor, and Exp50. The analytic map tracks Exp50 closely in the
CoG, is slightly worse in density, and remains far above the representation
floor. Halo prediction, not the fixed analytic formula, dominates the present
error budget.

### Standard QA figures for the halo map

#### `qa_mass_kpc_aper_exp55_halo_map`

Held-out predicted versus TNG cumulative stellar masses inside 10, 30, 50, and
100 kpc, with residuals versus true mass. Median biases stay between -0.5% and
+2.1%, while per-galaxy scatter decreases from 0.111 dex at 10 kpc to 0.097 dex
at 100 kpc.

#### `qa_mass_kpc_diff_exp55_halo_map`

Held-out annular and outer-envelope masses. Scatter is 0.153, 0.164, and 0.166
dex in the 10--30, 30--50, and 50--100 kpc annuli. Mass beyond 100 kpc retains
a +15.8% median bias and 0.188 dex scatter, making it the clearest physical-kpc
limitation.

#### `qa_mass_Re_aper_exp55_halo_map`

Cumulative masses inside one, two, and four TNG half-mass radii. The TNG radius
is used for both truth and model in each pair, separating mass-profile error
from error in the model's own predicted size.

#### `qa_mass_Re_diff_exp55_halo_map`

Annular and envelope masses in TNG half-mass-radius units. The figure tests
whether the halo map places stellar mass correctly relative to each galaxy's
true size, including the weakly constrained outer envelope.

#### `qa_bins_exp55_halo_map`

Median TNG and conditional-mean CoGs in three DiffMAH final-mass bins, with
fractional residuals. The solid residual includes the predicted total stellar
mass; the dotted residual removes amplitude to isolate profile shape.

#### `qa_dens_exp55_halo_map`

Median annular surface-density profiles and residuals in the same three halo-
mass bins. The conditional mean overpredicts the outer density most clearly in
the lower-mass bin; this radial structure is less apparent in cumulative CoGs.

#### `qa_planes_exp55_halo_map`

The three standard redshift-0.4 population planes for TNG and the conditional
mean. The two fixed-kpc planes are too narrow: for example, the `<30 kpc`
versus `50--100 kpc` vertical scatter contracts from 0.206 dex in TNG to 0.083
dex in the mean prediction. This is expected for a conditional mean but sets
the diversity that stochastic draws must recover.

#### `qa_size_exp55_halo_map`

The TNG and conditional-mean mass--R50 plane plus median offsets in R50, R80,
and R90. Median offsets are small, from -0.004 dex for R50 to +0.018 dex for
R90, while the predicted mean relations remain substantially narrower than
TNG.

#### `qa_cdf_exp55_halo_map`

Population cumulative distributions for representative aperture and annular
masses. Solid curves are TNG and dashed curves are the conditional mean; the
lower panels show model-minus-TNG distribution differences relative to the
truth's split-half sampling floor.

#### `qa_cases_exp55_halo_map`

Two best and two worst held-out predictions ranked by the maximum fractional
CoG error. The worst cases are failures of the halo-to-profile prediction,
especially its regressed total stellar mass, rather than evidence that the
analytic function cannot fit those known profiles.

### What should happen next

Do not free the two globally fixed component indices or add redshift evolution
yet. First rotate or otherwise reorganize the four-coordinate residuals into
modes that distinguish halo-predictable structure from stochastic structure.
Re-fit the conditional distribution and require its population draws to match
the fixed-kpc and size-scaled planes as well as pointwise CoG coverage. Only
after that calibration test passes should the same fixed analytic family be
carried across the Exp51 epochs.

## Part III: halo-map refinement, Stages 1--4

### What changed

The refinement keeps the selected analytic profile and the original sparse
degree-2 conditional mean unless a broader mean passes predeclared held-out and
radial-residual gates. It first separates analytic representation error from
halo-mapping error, then rotates the four parameter residuals according to
their effect on the CoG, and finally compares stochastic generators. All model
selection and residual calibration occur inside each outer training fold.

The selected stochastic model resamples complete four-coordinate residual
vectors from nearby training halos and multiplies them by one scale chosen from
inner held-out profile coverage. This preserves measured cross-coordinate
structure. It is an empirical conditional distribution around an analytic
mean, not a new analytic equation for the scatter.

### `exp55_refinement_decomposition`

The upper row separates the analytic representation floor from the held-out
halo-map error in three halo-mass bins. The lower row uses the analytic decoder
Jacobian to attribute the conditional mean's shape residual to errors in total
mass, compact fraction, compact scale, and extended-to-compact scale ratio.
The low-mass 5--10 kpc bump is driven mainly by the radius-ratio coordinate and
partly cancelled by compact-fraction and compact-scale errors. In the high-mass
bin, compact fraction and compact scale produce the central excess, while the
radius ratio suppresses mass at intermediate radii. The residual not closed by
the linearized decomposition has 0.00886 dex median RMS, so the attribution is
useful but not exact.

### `exp55_refinement_mean_candidates`

The left panels compare held-out profile CRPS, density-profile RMS, and the
largest mass-bin median shape residual for four predeclared conditional means.
The right panels show their radial residuals directly. The complete degree-2
basis produces the best scalar CRPS, only 0.45% below the sparse mean, but
worsens the largest coherent radial residual from 5.45% to 6.49%. The cubic
mass variant reduces that residual to 4.98% without a sufficiently large CRPS
gain. No candidate passes the combined gate, so the original sparse mean is
retained.

### `exp55_refinement_residual_modes`

The first panel shows that four profile-effect modes contain 62.5%, 24.4%,
10.2%, and 2.9% of residual CoG variance. The second gives held-out skill for
their absolute coordinates; modes 1 and 3 retain most of the halo-predictable
information. The radial curves show the profile deformation associated with a
positive unit change in each mode. The final annotated heatmap shows that mode
residuals have less than 0.005 Pearson correlation with every original halo
input after the existing mean is fitted. The mode transform is diagnostic and
is not counted as a new mean-model improvement.

### `exp55_refinement_stochastic_candidates`

Six stochastic generators are compared using held-out profile CRPS, nominal
68% pointwise coverage, and energy distance in the two fixed-kpc population
planes plus the self-consistent size-scaled plane. Raw Gaussian draws over-
broaden the size-scaled plane. Empirical nearest-neighbour residuals recover
the joint geometry but initially under-cover. The selected nested-calibrated
version gives 68.4% coverage for the nominal 68% interval, improves profile
CRPS from 0.06429 to 0.06368 dex, and lowers the self-consistent size-scaled
energy ratio from 1.73 to 1.38 times the TNG split-half sampling floor. CRPS
and coverage use all 32 draws per galaxy; the population-distance statistics
are averaged over four complete draws because each distance compares all
2,539 generated galaxies with the full TNG sample.

### `exp55_refinement_population_draws`

One complete generated draw is compared directly with TNG in the three
standard population planes and the stellar-mass relations for R50, R80, and
R90. Each generated galaxy's own half-mass radius defines the size-scaled
apertures. The calibrated generator closely restores the two fixed-kpc planes
and reduces the size-scaled scatter from the raw Gaussian value of 0.188 dex to
0.084 dex, compared with 0.072 dex in TNG. R50 and R80 are broadly recovered.
R90 remains the clearest limitation: its scatter is realistic, but its mean
mass dependence is too shallow and its median is high by 0.024 dex.

### Standard QA for the selected refinement

The `standard_qa` subdirectory repeats the complete ten-figure QA battery for
the retained sparse conditional mean and the selected calibrated population
draws. The conditional-mean panels are intentionally unchanged from Part II;
they remain the appropriate paired checks of held-out profile error. The new
stochastic conclusions should be read from the self-consistent population-draw
figure above, because the standard size-scaled paired panels deliberately use
each TNG counterpart's known `Re`.

### Decision and remaining question

Retain the sparse analytic conditional mean and use nested-calibrated empirical
neighbour residuals for redshift-0.4 experimental draws. Do not broaden the
polynomial mean, free the two global profile indices, or add redshift evolution
as a response to these residuals. The next targeted analysis should explain
the opposite central/intermediate residual signatures at low and high halo
mass and repair the mean R90 trend without merely increasing stochastic width.

## Part IV: stochastic robustness and the fixed outer-slope test

### What is being evaluated

These figures ask why the selected Stage 4 generator misses the TNG
stellar-mass--R90 distribution even though its profile coverage and most
population planes are good. They separate four possible sources: Monte Carlo
variation, the residual-resampling prescription, compression of the actual MAH
by DiffMAH, and the globally fixed Moffat outer slope. The tests use the same
five outer held-out folds; all scale calibration and candidate decisions use
only the corresponding training galaxies.

### `exp55_stochastic_robustness`

The upper panels compare profile CRPS, nominal 68% coverage, and population
energy distances for three random seeds, four neighbour counts, and three
definitions of distance between halos. The lower panels show the distribution
of complete-population slopes and energy distances across all 32 saved draws.
The shaded interval is the 16th--84th percentile range; the dotted reference
is the raw Gaussian generator.

The profile metrics are stable, but every neighbour configuration misses R90:
its energy-distance ratio is 3.27--3.77 times the TNG split-half sampling
floor. Across the 32 complete populations the median is 3.64, with a
16th--84th percentile range of 3.47--3.76. This figure establishes that the
failure is a model feature, not an unlucky draw. It also shows why no one
stochastic metric is sufficient: the empirical generator improves profile
CRPS and the self-consistent size-scaled plane while worsening R90 relative to
the raw Gaussian.

### `exp55_outer_failure_source`

The left panels compare the TNG R50, R80, and R90 relations with the individually
fitted analytic profile, the held-out conditional mean, and a complete generated
population. The fixed `gamma=1.25` representation itself changes the R90 slope
from 0.291 in TNG to 0.251 despite only 0.00437 dex median CoG RMS. The right
panels test whether this error follows actual-minus-DiffMAH history residuals or
the DiffMAH fit RMS. The correlations are weak, from about -0.03 to +0.05.

The figure therefore assigns the dominant outer-size failure to the analytic
tail before the halo map. It does not prove DiffMAH is lossless for every
profile feature; it shows that the specific R90 residual lacks the signature
expected if MAH smoothing were its main cause.

### `exp55_outer_feature_audit`

Each candidate extra halo quantity is fitted in the outer folds and compared
with a mass-conditioned shuffle of that same quantity. The upper row uses the
original `gamma=1.25` profile and the lower row uses `gamma=1.4`. One panel
shows the change in median held-out CoG RMS; the other shows the largest
mass-bin radial residual, so a tiny average gain cannot hide a newly worsened
coherent profile shape.

Portable DiffMAH-predicted growth, actual anchor growth, DiffMAH misses and fit
RMS, formation redshift, concurrent accretion rate, and halo axis ratio all fail
the combined gate. Several improve median RMS by a few ten-thousandths of a dex,
but none also improves the radial residual relative to its shuffle. No extra
halo feature is adopted from this audit.

### `exp55_gamma_generator_comparison`

This figure compares the fixed `gamma=1.25` and `gamma=1.4` profile families
at every relevant layer: individually fitted representation, held-out mean
radial residuals, profile CRPS and coverage, standard population planes, and
R50/R80/R90 relations. The steeper `gamma=1.4` tail lowers density RMS from
0.06601 to 0.06093 dex while changing CoG RMS by only +0.000030 dex. Its
generated R90 energy-distance ratio improves from 3.40 to 2.20, and its
self-consistent size-scaled-plane ratio improves from 1.47 to 0.68.

The remaining R90 slope is 0.259, still below TNG's 0.291. The figure supports
`gamma=1.4` as the better global baseline, but it also shows that no single
fixed slope matches all halo masses.

### Standard QA for fixed gamma=1.4

The `outer_envelope/standard_qa_gamma1p4` directory contains the complete ten
standard figures for the held-out `gamma=1.4` conditional mean and calibrated
draws:

- `qa_bins_exp55_gamma1p4_halo_map` and
  `qa_dens_exp55_gamma1p4_halo_map` show the mass-binned cumulative and
  differential profiles. The high-mass bin retains a coherent radial swing.
- `qa_mass_kpc_aper`, `qa_mass_kpc_diff`, `qa_mass_Re_aper`, and
  `qa_mass_Re_diff` give the cumulative and differential mass diagnostics in
  physical and TNG-size-scaled apertures.
- `qa_planes`, `qa_size`, `qa_cdf`, and `qa_cases` show the conditional mean's
  population planes, size relations, marginal distributions, and representative
  individual successes and failures.

These standard panels deliberately distinguish the held-out conditional mean
from a complete generated population. The stochastic population conclusions
come from the experiment-specific figures, where every draw is measured at its
own generated size.

## Part V: mass-dependent outer-slope boundary test

### `exp55_mass_dependent_outer`

Panel A shows the R90 error of the individually fitted representation versus
DiffMAH final peak halo mass for fixed `gamma=1.25`, fixed `gamma=1.4`, and the
predeclared two-regime rule. The rule uses `gamma=1.4` below each training
fold's 2/3 halo-mass quantile and `gamma=1.25` above it. Panels B and E compare
held-out median shape residuals in the lowest and highest halo-mass terciles.
Panel C compares energy-distance ratios in the two fixed-kpc planes, the
self-consistent size-scaled plane, and R90. Panel D shows profile CRPS, and
Panel F verifies that the fold-local mass thresholds are stable at
`log10(M_peak/Msun)=13.452--13.459`.

The two-regime generator reaches an R90 energy-distance ratio of 0.96, where
one is the TNG sampling floor, compared with 2.20 for fixed `gamma=1.4` and
3.40 for fixed `gamma=1.25`. Its profile CRPS is 0.06377 dex, only 0.40% worse
than the best fixed-slope value, while all three standard population-plane
ratios improve. The high-mass radial residual also shrinks substantially.

Panel A is equally important: it shows structure around the hard transition.
The figure therefore demonstrates a halo-mass dependence of the required outer
slope, not a physical discontinuity at the adopted threshold. The next test
must replace this boundary with a smooth low-complexity law.

### Standard QA for the mass-dependent boundary

The `mass_dependent_outer/standard_qa` directory repeats the complete standard
battery for the held-out two-regime model:

- `qa_bins_exp55_mass_dependent_outer` shows average CoGs and separates the raw
  residual, which includes predicted amplitude, from the amplitude-pinned shape
  residual. The largest raw median residual is about 5% at small radius in the
  middle halo-mass bin; after removing amplitude it is about 2.7%.
- `qa_dens_exp55_mass_dependent_outer` shows median surface-density residuals
  mostly within about 0.04 dex across all three halo-mass terciles. The smooth
  radial oscillation remains visible even when the cumulative CoGs overlap.
- `qa_size_exp55_mass_dependent_outer` shows that the conditional mean is much
  narrower than TNG, as a conditional mean must be. Its R50, R80, and R90
  median offsets are small; stochastic recovery is judged in the custom figure.
- `qa_mass_kpc_aper`, `qa_mass_kpc_diff`, `qa_mass_Re_aper`, and
  `qa_mass_Re_diff` report cumulative and differential masses in physical and
  TNG-size-scaled apertures.
- `qa_planes`, `qa_cdf`, and `qa_cases` inspect conditional-mean plane geometry,
  population marginals, and individual held-out profiles.

The complete stochastic populations, rather than the conditional mean shown
in several standard panels, have an R90 slope of 0.302 with a 16th--84th
percentile range of 0.299--0.307, compared with 0.291 in TNG, and R90 scatter
of about 0.099 dex compared with 0.104 dex in TNG. This distinction prevents a
correctly narrow conditional mean from being mistaken for an under-dispersed
generator.

## Part VI: smooth outer-slope feasibility

### `exp55_smooth_outer`

The smooth candidate fixes the outer slope through

```text
gamma = 1.25 + 0.15 / [1 + exp((logMpeak - transition) / width)],
```

with the transition fixed at each outer-training fold's 2/3 halo-mass quantile.
Panels A--C show the training median CoG RMS, density RMS, and absolute error in
the stellar-mass--R90 slope for six predeclared widths from 0.05 to 0.80 dex.
Each point is one outer fold. Width 0.20 dex is selected in every fold and is
an interior grid value after the Stage 11 extension. The slope error decreases
through 0.20 dex and then increases again, closing the original one-sided test.

Panel D divides each smooth-model metric by the corresponding hard-boundary
metric. Values below one favor the smooth law. The smooth model is essentially
tied in profile CRPS, improves the second fixed-kpc plane and R90, but worsens
the self-consistent size-scaled plane by 10.0034%, just beyond its 10% gate.
Panel E shows median R90 versus halo mass for TNG, one hard-boundary population,
and one smooth population. Panel F confirms the unanimous 0.20-dex training
selection across all five outer folds.

The complete result is formally not selected because it also raises the
largest held-out mass-bin shape residual from 2.74% to 3.34%. Nevertheless, it
is a strong feasibility result: profile CRPS is 0.06391 dex, nominal 68%
coverage is 69.5%, the two fixed-kpc energy ratios are both about 0.61, and the
R90 ratio is 0.93. The generated stellar-mass--R90 slope is 0.2895, compared
with 0.2909 in TNG. Across all 32 complete populations the R90 energy ratio has
median 0.92 and a 16th--84th percentile range of 0.85--0.95.

### Standard QA for the smooth candidate

The `smooth_outer/standard_qa` directory contains the full standard battery
even though the candidate missed its formal gate, because the result is
scientifically informative:

- `qa_bins_exp55_smooth_outer` shows a raw inner-mass residual reaching about
  6% in the middle halo-mass tercile, while the amplitude-pinned shape residual
  peaks near 3.3%. This is the coherent failure that prevents selection.
- `qa_dens_exp55_smooth_outer` shows that the cumulative residual corresponds
  to a smooth density oscillation of roughly -0.04 to +0.04 dex around
  5--30 kpc. The high-mass outer density is well controlled.
- `qa_size_exp55_smooth_outer` shows the narrow held-out conditional mean. The
  complete generated mass--size relation must be read from the custom figure
  and the all-draw intervals, not from this mean-only panel.
- `qa_mass_kpc_aper`, `qa_mass_kpc_diff`, `qa_mass_Re_aper`, and
  `qa_mass_Re_diff` report the complete cumulative and differential mass
  diagnostics in physical and TNG-size-scaled apertures.
- `qa_planes`, `qa_cdf`, and `qa_cases` give the paired conditional-mean
  population planes, marginal distributions, and representative individual
  held-out predictions.

The visual conclusion agrees with the gates: the smooth outer law solves the
far-outer population relation, but a smaller intermediate-radius radial
systematic remains. The extended grid shows that 0.20 dex is no longer a tested
boundary. Further work should diagnose the radial systematic; changing the 10%
gate after seeing the answer is not justified.
