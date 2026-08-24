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
  correlated draws restore and sometimes exceed the missing diversity. In the
  tight size-scaled `<2Re` versus `2--4Re` plane, their 0.213 dex scatter is
  nearly three times the TNG scatter of 0.072 dex. The stochastic residual
  model needs another round before it is a satisfactory population generator.
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
