# exp55 figure guide

This guide describes the complete QA set for the selected analytic curve of
growth (CoG): a compact Sérsic component with globally fixed index `n = 1`
plus an extended Moffat component with globally fixed `gamma = 1.25`.

## What is being evaluated

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

## Main conclusions from the QA battery

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

## Why a good CoG can have a visibly worse density residual

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

## Figure-by-figure captions

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

## What should happen next

The next experiment should predict the four profile parameters from halo
properties in held-out galaxies. The same standard QA set should then be
repeated on those halo-predicted profiles. The expected errors will be much
larger than the representation errors measured here, and the total stellar mass
will become a genuine halo-model prediction rather than a parameter fitted from
the known galaxy CoG.
