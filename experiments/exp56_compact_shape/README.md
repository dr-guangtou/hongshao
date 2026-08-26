# exp56 — Compact-shape closure and radial halo–CoG relation

Status: COMPLETE THROUGH STAGE 4 AND THE STAGE 3B DIRECT-COG FOLLOW-UP. Neither
the lower-index Stage 1 test nor the joint-global shape test changes the
representation. Stage 3b confirms that changing the shared fitting target from
analytic coordinates to the decoded CoG does not repair the held-out profile:
it slightly improves the full cumulative curve while worsening the 5--30 kpc,
density, and size safeguards. The coordinate-regression Stage 3 relation
therefore remains the current baseline.

## Question

Does a globally fixed compact Sérsic index below `n=1` remove the coherent
5--30 kpc residual left by Exp55 while preserving a stable four-coordinate
analytic representation and the outer-size relation that the frozen Exp55
outer-slope laws already recover?

This first test changes only the compact component's global shape. It is a
representation test: every galaxy's four profile coordinates are refitted to
its known redshift-0.4 curve of growth (CoG). It does not yet fit the held-out
halo–CoG relation.

## Scope and frozen references

- Epoch: redshift 0.4.
- Observable: CoG-derived stellar mass on the standard 2.0--148.2 kpc grid.
  Direct projected aperture masses remain cross-checks only.
- Profile family: the Exp55 compact Sérsic plus extended Moffat function.
- Galaxy-level coordinates: stellar mass within 148.2 kpc, compact fraction
  within that aperture, compact scale radius, and ordered extended scale
  radius.
- The component labels are phenomenological. They are not identified with
  in-situ or ex-situ stars.
- Fixed outer-slope baseline: `gamma=1.4`.
- Continuous outer-slope candidate:

  ```text
  gamma(log M_peak) = 1.25
      + 0.15 / {1 + exp[(log M_peak - transition) / 0.20 dex]}.
  ```

  Each outer training fold fixes `transition` to its own two-thirds quantile
  in DiffMAH `log M_peak`. The endpoints, transition definition, and
  0.20-dex width are frozen from Exp55.
- The Exp55 hard `gamma=1.4/1.25` halo-mass boundary remains a diagnostic
  ceiling. It is not retuned and is not a candidate physical discontinuity.

The observational motivation for a connection between stellar outskirts,
current halo mass, and recent assembly is a reason to test that connection in
a later held-out stage. It is not imposed on this representation. No
inner-region connection with early MAH is treated as established physics.

## Stage 1 predeclaration — representation-only compact-index test

Compare exactly these globally fixed compact Sérsic indices:

```text
n = 0.50, 0.75, 1.00
```

Run the complete grid under both frozen outer prescriptions: fixed
`gamma=1.4` and the smooth `gamma(log M_peak)` law above. Refit the same four
galaxy coordinates for every galaxy and every cell with the established four
widely separated starts. Do not fit `n` separately for each galaxy. Do not
retune any part of the outer-slope law.

Use the existing five outer folds. All choices use training galaxies only.
For the smooth law, every fold uses the gamma values implied by that fold's
training-mass transition for both its training and validation galaxies.

### Primary evidence and selection rule

For every galaxy, remove only the fitted-versus-measured amplitude difference
at 148.2 kpc and compute the RMS log-CoG residual over measured radii from
5--30 kpc. Also compute the log-density residual over annuli whose midpoints
lie from 5--30 kpc. Plot the signed median residual versus radius so a smaller
scalar cannot hide a coherent oscillation.

Within each outer training fold, the candidate with the lowest median
5--30 kpc amplitude-pinned CoG RMS is eligible only if its paired improvement
over `n=1` is resolved by a galaxy bootstrap: the 84th percentile of the
bootstrap distribution of the candidate-minus-`n=1` median difference must be
below zero. A merely smaller sample median is not a selection. If neither
`n<1` value clears this condition, retain `n=1`.

Selection is made separately under fixed `gamma=1.4` and the frozen smooth
law. A compact index is not adopted across the experiment unless the result is
consistent across folds and does not fail the safeguards below. If `n=0.5` is
selected, report the inference as one-sided and stop; do not extend the grid
automatically.

### Required safeguards

Report every item below for all six representation cells, not only the
primary winner:

- median full-range CoG RMS and signed amplitude-pinned CoG residuals;
- median full-range density RMS and signed 5--30 kpc density residuals;
- stellar masses inside 2, 5, and 10 kpc;
- R50, R80, and R90 offsets, plus the stellar-mass--R90 slope;
- the fraction of galaxies with any optimizer coordinate within 1% of a
  bound;
- the scaled-Jacobian smallest-to-largest singular-value ratio;
- agreement among the established multiple starts;
- exact synthetic recovery on representative fitted profiles;
- radial-jackknife stability after removing alternating measured radii; and
- separate compact and extended cumulative and density contributions.

An `n<1` candidate is rejected if its apparent 5--30 kpc improvement creates a
new bootstrap-resolved degradation in overall CoG or density accuracy, a new
poorly conditioned coordinate direction, materially increased boundary
incidence, failed exact recovery, or unstable radial-jackknife coordinates.
R50/R80/R90 and the component decomposition determine whether the compact
change leaves the extended component responsible for the outer envelope.

### Small-scale gate

Before any full run, execute the complete path on exactly 90 galaxies selected
across halo mass and compact fraction. The timed path includes all six cells,
fold-clean smooth gamma assignment, metric construction, the synthetic and
radial-jackknife checks, saved outputs, and the direct comparison figure. It
must finish in less than 60 seconds. Demo outputs use a separate `demo_`
prefix and must never overwrite full-sample products.

## Later stages, conditional on Stage 1

### Stage 2 — component and degeneracy diagnosis

For retained indices, plot compact and extended CoGs and density profiles
separately. Fix one galaxy-level coordinate on a grid and re-optimize the
other three. Determine whether `n<1` keeps compact-component mass inside about
10 kpc and reduces compact leakage at 5--30 kpc, or merely exchanges compact
fraction and both radii. Reject a candidate that buys the residual through a
new weak coordinate direction.

### Stage 3 — held-out halo–CoG relation

Only the representation selected inside the outer training folds proceeds.
Refit both the conditional mean and stochastic residual distribution from the
four epoch-local DiffMAH coordinates plus concurrent `c_200c`; do not reuse the
`n=1` calibration. Repeat final-halo-mass-only, mass-conditioned shuffled-MAH,
and mass-conditioned shuffled-concentration controls. Judge decoded profiles
with the standard QA battery: cumulative and differential masses in kpc and
size units, CoGs and density profiles binned by halo and stellar mass,
R50/R80/R90, CRPS, coverage, population geometry, and representative best,
typical, and worst galaxies.

### Stage 4 — radial halo-information test

Compare nested held-out halo–CoG relations using final halo mass, a portable
recent-growth summary derived from DiffMAH, a portable early assembled-fraction
summary, the complete DiffMAH history, and complete DiffMAH plus concurrent
concentration. Shuffle recent growth, early growth, and concentration
separately among galaxies of similar final halo mass. Report incremental CRPS
and RMS versus radius and in the annuli 2--5, 5--10, 10--30, 30--50, 50--100,
and beyond 100 kpc. The result is allowed to be null or to disagree with the
observational motivation.

## Stop conditions

- Stop after the declared compact-index grid before considering another `n`.
- Do not free `n` per galaxy.
- Do not retune the outer-slope endpoints, transition, or width in Exp56
  Stage 1.
- Do not add epochs until the redshift-0.4 representation, identifiability,
  stochastic population geometry, and 5--30 kpc residual are stable.

## Known repository-wide test limitation

Repository-wide test collection is blocked by the pre-existing Exp07 import
of the absent gitignored file
`data/processed/tng300_072_z0p4.fits`. Exp56 verification therefore uses its
own numerical assertions and targeted checks; this limitation is not caused
by Exp56.

## Initial 90-galaxy validation result

The complete mass-stratified validation path finished in 31.11 seconds,
against the required limit of 60 seconds, and therefore passes the small-scale
gate. It fitted all three compact indices under fixed `gamma=1.4` and under
all five fold-local realizations of the frozen smooth law, constructed every
declared metric, ran the exact-recovery and alternating-radius checks on 12
galaxies spanning halo mass, and generated the three direct diagnostic
figures. Demo products use the `demo_` prefix; no full-sample product was
created or overwritten.

This sample is not used for the scientific decision, but its direction is
unambiguous. Under fixed `gamma=1.4`, the median per-galaxy amplitude-pinned
5--30 kpc CoG RMS is 0.00501 dex for `n=1`, compared with the worse values
0.00685 dex for `n=0.75` and 0.01026 dex for `n=0.5`. The corresponding
5--30 kpc density RMS is 0.03524 dex for `n=1`, compared with the worse values
0.03828 and 0.04962 dex. Under the frozen smooth outer law, the 5--30 kpc CoG
RMS is 0.00523 dex for `n=1`, compared with the worse values 0.00607 and
0.00840 dex, while the density RMS is 0.03146 dex for `n=1`, compared with the
worse values 0.03645 and 0.04495 dex. Accordingly, training-only selection
retains `n=1` in all five folds for both outer laws in this validation sample.
This means the proposed lower compact index does not show its intended effect
at small scale; the full sample is still required before closing the test.

The size checks point in the same direction. Under the smooth law, the median
model-minus-TNG `log10 R90` offset is -0.00182 dex for `n=1`, compared with the
larger negative offsets -0.01637 dex for `n=0.75` and -0.02705 dex for
`n=0.5`. Lowering `n` therefore makes the fitted outer quantile systematically
too small on this sample rather than repairing the inherited outer relation.

All six cells recover their own synthetic coordinates to a maximum absolute
coordinate error of `1.12e-9`, better than the declared numerical threshold
of `1e-5`. Across the six cells, removing alternating radii changes the median
stellar-mass coordinate by 0.00077--0.00132 dex, compact fraction by
0.00094--0.00379, compact radius by 0.00228--0.00780 dex, and extended radius
by 0.00374--0.00650 dex, all relative to the corresponding complete-grid fit.
No `n=0.75` or `n=1` fit lies within 1% of an optimizer bound; one of 90
`n=0.5` fits does under each outer law. Thus the smoke path is numerically
sound, while already warning that lower `n` spends freedom in less favorable
profile and size directions.

## Full 2,539-galaxy Stage 1 result

The complete declared grid finished in 478.88 seconds after the 90-galaxy
gate passed. The measured 5--30 kpc errors are monotonically worse as the
globally fixed compact Sérsic index is lowered. All values below are medians
over galaxies; each value compares the fitted analytic profile with the known
TNG CoG-derived profile.

| Outer law | `n` | 5--30 kpc CoG RMS (dex) | 5--30 kpc density RMS (dex) | Full CoG RMS (dex) | Full density RMS (dex) |
|---|---:|---:|---:|---:|---:|
| Fixed `gamma=1.4` | 0.50 | 0.01100 | 0.05501 | 0.00807 | 0.08239 |
| Fixed `gamma=1.4` | 0.75 | 0.00732 | 0.04154 | 0.00525 | 0.06905 |
| Fixed `gamma=1.4` | 1.00 | **0.00551** | **0.03198** | **0.00440** | **0.06093** |
| Frozen smooth law | 0.50 | 0.00947 | 0.04812 | 0.00690 | 0.07163 |
| Frozen smooth law | 0.75 | 0.00625 | 0.03591 | 0.00450 | 0.06135 |
| Frozen smooth law | 1.00 | **0.00495** | **0.02847** | **0.00404** | **0.05650** |

The original `n=1` reference is the numerical winner in all five outer
training folds under both outer laws. Therefore no `n<1` candidate reaches
the predeclared bootstrap-eligibility test; the representation retains
`n=1`. The signed residual curves show the same result directly: reducing
`n` amplifies the coherent oscillation rather than flattening it. This means a
more abruptly truncated compact component is not the missing repair for the
5--30 kpc residual.

The aperture-mass and size safeguards agree with the primary result:

| Outer law | `n` | Error at 2/5/10 kpc (dex) | R50/R80/R90 offset (dex) | Fitted stellar-mass--R90 slope |
|---|---:|---:|---:|---:|
| Fixed `gamma=1.4` | 0.50 | -0.01078 / +0.00462 / -0.01063 | -0.00524 / -0.04870 / -0.04158 | 0.28022 |
| Fixed `gamma=1.4` | 0.75 | -0.00422 / +0.00566 / -0.00620 | -0.00120 / -0.03059 / -0.02708 | 0.27375 |
| Fixed `gamma=1.4` | 1.00 | +0.00113 / +0.00313 / -0.00148 | -0.00076 / -0.01538 / -0.01460 | 0.27032 |
| Frozen smooth law | 0.50 | -0.00888 / +0.00269 / -0.00820 | -0.00393 / -0.03232 / -0.02030 | 0.30672 |
| Frozen smooth law | 0.75 | -0.00292 / +0.00411 / -0.00493 | +0.00034 / -0.01617 / -0.00840 | 0.29793 |
| Frozen smooth law | 1.00 | +0.00208 / +0.00223 / -0.00080 | +0.00123 / -0.00348 / +0.00130 | **0.29239** |

The TNG stellar-mass--R90 slope is 0.29095. Thus `n=1` with the frozen smooth
outer law reproduces the representation-level R90 relation most closely,
whereas lowering `n` moves the R80 and R90 medians systematically inward.

Lower `n` slightly improves the local scaled-Jacobian ratio: the median
`log10(s_min/s_max)` changes from -1.538 at `n=1` to -1.399 at `n=0.5` under
fixed `gamma=1.4`, and from -1.585 to -1.444 under the smooth law. This modest
conditioning gain does not compensate for the worse CoG, density, aperture,
and size accuracy. The fraction of fits with a coordinate within 1% of a
bound is only 0.0032--0.0047 across all six cells, and the near-best solutions
from the four starts agree to better than `3e-6` in every median coordinate
range.

All six cells pass exact synthetic recovery. The largest absolute recovered
coordinate error is `1.90e-9`, compared with the declared maximum of `1e-5`.
For the 12 halo-mass-spanning diagnostic galaxies, the median changes after
removing alternating radii are 0.00083--0.00178 dex in stellar mass,
0.00181--0.00302 in compact fraction, 0.00385--0.00755 dex in compact radius,
and 0.00643--0.01097 dex in extended radius. The `n=0.75` cell has one much
larger compact-fraction/extended-radius jackknife excursion of about 0.27 in
both outer-law families, while the maximum corresponding changes for `n=1`
are 0.011 and 0.020. This isolated instability gives an additional reason not
to retain `n=0.75`.

The component decomposition confirms the mechanical interpretation: lowering
`n` confines the compact component more strongly, but the refitted compact
fraction and two radii compensate, leaving a worse total profile from
5--30 kpc. Because no lower index survives Stage 1, the conditional profiled
coordinate scans in Stage 2 are not triggered. Exp56 therefore closes as a
null result for `n<1`; together with Exp55's worse `n=2` and `n=4` fits, the
discrete evidence brackets the useful global compact index at approximately
`n=1` from both sides.

## Feasibility of fitting global `n` and global Moffat `gamma` together

This is feasible as a population-level, representation-only extension. For
each proposed pair `(n, gamma)`, refit the same four coordinates for every
training galaxy, then minimize the aggregate training-profile objective after
those galaxy coordinates have been profiled out. Select one global pair inside
each outer training fold and apply that fixed pair to its held-out galaxies.
Neither coordinate should be fitted separately per galaxy.

The required diagnostic is a two-dimensional profiled objective surface, not
only a best-fit pair. `n`, `gamma`, compact fraction, and the two radii can
partially compensate, so the extension must report the valley orientation,
curvature or Hessian condition, boundary contact, multi-start agreement,
synthetic recovery, radial-jackknife stability, and the same CoG, density, and
R50/R80/R90 safeguards used here. The newly completed result makes this test
scientifically interpretable: `n=1` is bracketed rather than sitting at the
edge of the explored compact-index range.

A fitted single global `gamma` would still be a representation diagnostic,
not a replacement for Exp55's evidence that outer shape varies with halo mass.
The joint test should therefore compare its held-out profile and size accuracy
with both fixed `gamma=1.4` and the frozen smooth `gamma(log M_peak)` law. It
can establish whether a global `(n, gamma)` compensation exists, but it cannot
by itself establish or erase a physical outer connection with halo mass or
recent assembly.

## Stage 1b predeclaration — joint global compact and outer shape

This extension was declared after completing Stage 1 and before writing its
driver. It asks whether one population-wide compact Sérsic index and one
population-wide Moffat index provide a better four-coordinate representation
than the fixed `(n=1, gamma=1.4)` baseline. It does not fit either shape index
separately for each galaxy and does not fit a halo–CoG relation.

The focused grid is

```text
n     = 0.750, 0.875, 1.000, 1.125, 1.250
gamma = 1.175, 1.250, 1.325, 1.400, 1.475
```

The bounds are fixed by the completed surrounding tests: Exp56 shows worse
profiles at `n=0.5` and `n=0.75`, while Exp55 shows worse profiles at `n=2`
and `n=4`; Exp55 also shows poor boundaries at `gamma=1.1` and worse CoG and
density accuracy at `gamma=1.7`. The new grid therefore resolves the already
bracketed basin around `n=1`, contains the established `gamma=1.25` and
`gamma=1.4` representations exactly, and adds one interpolation point between
them plus one flank on either side. No grid point may be added after seeing the
result.

At every one of the 25 global shape pairs, refit the same four galaxy-level
coordinates with the established four starts. In each of the five existing
outer folds, choose the grid pair with the lowest training-galaxy median
5--30 kpc amplitude-pinned CoG RMS. Report the fold winners and the fraction
of 1,000 galaxy bootstraps selecting each full-sample grid point.

### Conditional local refinement

Attempt one continuous refinement only if the full-sample grid winner is
strictly interior and all five training-fold winners are no more than one grid
step from it in both coordinates. Fit a two-dimensional quadratic to the nine
neighboring grid objectives after scaling both axes by their grid spacing.
The quadratic candidate is valid only if its Hessian is positive definite and
its stationary point lies within the rectangle bounded by the midpoints from
the central grid point to its four immediate neighbors. Evaluate that one
candidate by directly refitting all galaxy coordinates. If any condition
fails or the direct objective is not lower than the central grid value, retain
the grid winner and report why; do not launch another refinement.

This quadratic is a numerical interpolation of the population objective, not
a physical likelihood. Report its two Hessian eigenvalues, their ratio, and
the eigenvector of the shallow direction so any `n`--`gamma` compensation is
visible.

### Selection and safeguards

The primary comparison is the paired per-galaxy 5--30 kpc amplitude-pinned
CoG RMS. A refined or grid candidate replaces fixed `(n=1, gamma=1.4)` only if
the 84th percentile of 1,000 bootstrap candidate-minus-reference median
differences is below zero. It must also avoid a bootstrap-resolved degradation
in full-range CoG RMS, full-range or 5--30 kpc density RMS, absolute R50/R80/R90
error, scaled-Jacobian conditioning, or boundary incidence. Report signed
radial CoG and density residuals, stellar masses within 2, 5, and 10 kpc,
stellar-mass--R90 slope, and component contributions directly.

Run exact synthetic recovery, four-start agreement, and the alternating-radius
jackknife for the selected pair and the fixed reference on the same 12
halo-mass-spanning galaxies. Reject a candidate with a new large instability.
Concretely, every exact-recovery coordinate must remain below `1e-5`; each
maximum jackknife-coordinate change must remain below the larger of 0.05 and
twice the fixed reference's corresponding maximum; and each median near-best
four-start coordinate range must remain below the larger of `1e-4` and five
times the reference range.
The frozen smooth `gamma(log M_peak)` law at `n=1` remains a second external
reference: a successful global pair may become the preferred single-slope
baseline, but it does not replace the smooth law unless its CoG, density,
R50/R80/R90, and stellar-mass--R90 results are at least as accurate.

The full two-dimensional surface must be shown with a sequential `cividis`
scale, numerical cell labels, fold winners, and the refined point if valid.
Separate figures must show direct signed radial residuals and representative
measured-versus-model CoGs. These are representation diagnostics, so the
halo-feature shuffle controls and stochastic QA battery remain reserved for a
later held-out halo–CoG relation rather than being misapplied here.

### Stage 1b small-scale gate

Before any full run, execute the complete 25-cell grid, conditional refinement,
metrics, numerical diagnostics, saved outputs, and all figures on exactly the
same mass-stratified 90-galaxy sample. The complete uncached path must finish
in less than 60 seconds. Demo products use a distinct `joint_demo_` prefix and
must not overwrite Stage 1 or full Stage 1b artifacts.

## Stage 1b result — intermediate-radius gain fails global safeguards

The complete uncached 90-galaxy path passed in 39.04 seconds against the
60-second limit. The full 25-cell grid then fitted 2,539 galaxies in 1,236.58
seconds. All five outer training folds choose the same grid cell,
`(n=1.25, gamma=1.4)`, and 99.7% of 1,000 full-sample galaxy bootstraps choose
that cell. The other 0.3% choose the adjacent `(n=1.125, gamma=1.4)` cell.

The winner lies at the declared upper `n` boundary. Therefore the predeclared
continuous quadratic refinement is not attempted, and the direction is
one-sided for the 5--30 kpc primary objective rather than a measurement of an
optimal global Sérsic index. No higher `n` cell is added after seeing this
result. This boundary does not motivate an extension because the candidate
already fails the independent full-profile and conditioning safeguards, while
Exp55 found worse full CoG accuracy at `n=2` and `n=4`.

| Representation | 5--30 kpc CoG RMS (dex) | 5--30 kpc density RMS (dex) | Full CoG RMS (dex) | Full density RMS (dex) | Median `log10(s_min/s_max)` |
|---|---:|---:|---:|---:|---:|
| Joint-grid winner: `n=1.25`, `gamma=1.4` | 0.004953 | 0.028884 | 0.004585 | 0.057610 | -1.601 |
| Fixed reference: `n=1`, `gamma=1.4` | 0.005510 | 0.031980 | **0.004402** | 0.060934 | **-1.538** |
| Frozen smooth law at `n=1` | **0.004947** | **0.028473** | **0.004041** | **0.056498** | -1.585 |

Relative to fixed `(n=1, gamma=1.4)`, the joint-grid winner has a
bootstrap-resolved paired improvement in 5--30 kpc CoG RMS: the 16th, 50th,
and 84th percentiles of the candidate-minus-reference median difference are
-0.000783, -0.000737, and -0.000699 dex. Its 5--30 kpc and full density RMS
also improve. However, its full-range CoG RMS is worse: the corresponding
paired interval is +0.000008, +0.000058, and +0.000114 dex. Its scaled local
Jacobian ratio is also resolved worse by about 0.058 dex in log10 ratio. The
boundary fraction changes from 0.00354 for the fixed reference to 0.00394 for
the candidate; the paired bootstrap interval includes zero, so this small
difference is not resolved. The candidate therefore fails replacement despite
repairing part of the intended radial interval.

The size results make the trade clearer. The candidate's median
model-minus-TNG R50/R80/R90 offsets are -0.00096, -0.00383, and -0.00439 dex,
better than the fixed reference's -0.00076, -0.01538, and -0.01460 dex at the
outer two quantiles. Nevertheless, its stellar-mass--R90 slope is 0.27322,
still far below the TNG value 0.29095. The frozen smooth law gives 0.29239 and
median R50/R80/R90 offsets +0.00123, -0.00348, and +0.00130 dex. Against that
smooth reference, the global pair has resolved worse full CoG accuracy,
conditioning, and absolute R80/R90 errors. It therefore cannot replace the
mass-dependent outer law.

The numerical checks themselves pass. All fits converge; the largest exact
synthetic-recovery coordinate error for the grid winner is `1.28e-11`, below
the declared `1e-5` limit. Across the 12 halo-mass-spanning galaxies, its
largest alternating-radius changes are 0.00512 dex in total stellar mass,
0.01176 in compact fraction, 0.01986 dex in compact radius, and 0.02497 dex in
extended radius, all within the predeclared limits. The four-start agreement
also passes. Thus non-adoption is a measured profile trade, not optimizer
failure.

Mechanically, increasing the global compact index lets the compact component
contribute farther into the intermediate radii and reduces the fixed
baseline's coherent 5--30 kpc oscillation. The cost appears at smaller radii,
in overall cumulative-profile accuracy, and in a weaker coordinate direction;
one global `gamma` also cannot recover the mass-dependent R90 relation. The
joint fit is feasible and informative, but no single global `(n, gamma)` pair
passes the complete representation test. Retain fixed `(n=1, gamma=1.4)` as
the single-slope baseline and retain the frozen smooth law as the outer-shape
candidate; do not extend the global shape grid.

## Stage 3 predeclaration — refit the held-out halo–CoG relation

Stage 3 carries exactly two representations forward, both with the retained
global compact Sérsic index `n=1`:

1. the deployable fixed outer slope `gamma=1.4`; and
2. the frozen smooth `gamma(log M_peak)` law with endpoints 1.4 and 1.25,
   fold-training transition at the two-thirds halo-mass quantile, and width
   0.20 dex.

The hard outer-slope boundary remains a diagnostic ceiling from Exp55 and the
joint `(n=1.25, gamma=1.4)` grid point remains a rejected representation. They
are not refitted as Stage 3 candidates. Neither `n` nor `gamma` is fitted per
galaxy, and no outer-law coefficient is retuned.

For each of the existing five outer folds, use only the training galaxies to
fit a conditional model from the four epoch-local DiffMAH coordinates plus
concurrent `c_200c` to the four fitted analytic-profile coordinates. For the
smooth representation, use the profile coordinates fitted under that same
fold's frozen transition for both its training and validation galaxies. This
keeps the representation and halo relation fold-clean. Refit every conditional
mean and residual distribution from the Exp56 coordinates; do not reuse the
Exp55 calibration or predictions.

The mean model is frozen to Exp55's selected sparse degree-two basis. The
stochastic model resamples complete four-coordinate residual vectors from
nearby training halos in standardized feature space, preserving measured
cross-coordinate dependence. Select one residual-inflation factor inside each
outer training fold using four inner folds and nominal-68% profile coverage.
No held-out galaxy may contribute to its mean fit, residual library, nearest
neighbours, or scale selection.

### Stage 3 controls

Fit the following four held-out conditional means separately for both profile
representations:

- final DiffMAH peak mass only;
- the complete four-parameter DiffMAH history plus `c_200c`;
- complete inputs after shuffling the three non-mass DiffMAH coordinates among
  galaxies of similar final halo mass; and
- complete inputs after shuffling `c_200c` among galaxies of similar final
  halo mass.

The deterministic smooth `gamma(log M_peak)` value is appended to the
conditional inputs for that representation. Because it is fixed by final halo
mass and the training-fold transition, it does not count as extra measured halo
information and is not shuffled independently.

### Stage 3 judgment

Judge the complete-input relations on decoded held-out CoGs and stochastic
draws, not only on the four intermediate coordinates. Report:

- mean profile CRPS, nominal-68% and nominal-90% pointwise coverage;
- median full-range and 5--30 kpc CoG RMS;
- median full-range and 5--30 kpc density RMS;
- signed CoG and density residuals in halo- and stellar-mass bins;
- cumulative and differential masses in fixed-kpc and size-scaled apertures;
- R50, R80, R90, the stellar-mass--R90 slope, and generated size distributions;
- complete-population geometry in the standard fixed-kpc and size-scaled
  planes; and
- direct best, typical, and worst held-out galaxies.

The smooth representation becomes the Stage 4 primary only if all of the
following are true relative to fixed `gamma=1.4`:

- nominal-68% coverage lies within three percentage points of 68%;
- profile CRPS is no more than 1% worse;
- full-range CoG RMS, density RMS, and the largest coherent halo-mass-bin shape
  residual show no bootstrap-resolved degradation;
- absolute R50, R80, and R90 errors show no bootstrap-resolved degradation;
- generated R90 and self-consistent size-scaled population distances improve;
  and
- no new systematic trend appears in the standard QA figures.

If the smooth law fails any safeguard, Stage 4 uses fixed `gamma=1.4` as its
primary representation and retains the smooth result only as a diagnostic.
If neither complete-input relation is stable and calibrated, stop rather than
interpreting radial feature differences from an unreliable halo–CoG relation.

### Stage 3 small-scale gate

Before the full run, execute both representations, all four conditional-mean
feature sets, fold-internal stochastic calibration, saved outputs, the direct
comparison figures, and the standard QA path on the same mass-stratified
90-galaxy sample. Use eight stochastic draws for this validation and require
the complete uncached path to finish in less than 60 seconds. Demo artifacts
use a `stage3_demo_` prefix and cannot overwrite the full Stage 1, Stage 1b, or
Stage 3 products. The full run uses 32 draws only after this gate passes.

### Stage 3 small-scale validation result

The first complete attempt was stopped at 74.8 seconds because the user's
global matplotlib configuration routed every PDF through external LaTeX. The
scientific calculation had completed, but figure export had not. Stage 3 now
uses matplotlib's built-in MathText for the same equations and retains both
PNG and PDF outputs. With that mechanical change, the complete uncached path
passed on exactly 90 mass-stratified galaxies in 31.18 seconds, including both
representations, all four conditional-mean feature sets, fold-internal
stochastic calibration, eight draws, saved predictions, three direct figures,
and both full standard-QA batteries. All fitted means and draws are finite and
each representation contains five held-out folds. The full run is therefore
authorized by the predeclared gate; the demo metrics are validation only and
do not replace the full-sample representation decision.

## Stage 3 result — retain fixed outer slope for the information test

The full held-out calculation completed for 2,539 galaxies with 32 stochastic
draws in 321.10 seconds. The fixed and smooth representations have calibrated
nominal-68% profile coverage of 68.75% and 68.85%, respectively, against the
68% target. Their mean profile CRPS values against the TNG CoGs are 0.06403 and
0.06375 dex; the paired bootstrap interval for smooth minus fixed is
[-0.00047, -0.00026, +0.00001] dex at the 16th, 50th, and 84th percentiles.

The smooth law improves the median density RMS against the CoG-derived TNG
density profile from 0.12788 to 0.12541 dex. It also reduces the largest
coherent halo-mass-bin CoG shape residual from 5.59% to 3.34%, and its
stellar-mass--R90 slope is 0.29205 against 0.29095 in TNG, compared with
0.26569 for fixed `gamma=1.4`. These are meaningful advantages of the smooth
representation. However, the predeclared adoption rule requires every gate to
pass. The smooth law has a bootstrap-resolved 0.00033 dex larger median
absolute R50 error, and its self-consistent size-plane population distance does
not improve. It therefore fails two safeguards despite improving the outer
profile. Stage 4 uses fixed `gamma=1.4` as its primary representation and keeps
the smooth result as a diagnostic. This is a conservative frozen-rule decision,
not evidence that a constant outer slope is uniformly more accurate.

Direct and standard-QA figures show finite, monotonic decoded profiles and no
new representation-specific failure. Both relations nevertheless compress
the population scatter in several aperture and size planes, so Stage 4 must
measure information content from held-out errors and shuffled controls rather
than interpret the conditional mean as a deterministic galaxy prediction.

## Stage 3b predeclaration — fit the shared relation in decoded CoG space

The completed Stage 3 is a two-step parameter bridge: fit four analytic-profile
coordinates to every TNG CoG, then fit each coordinate from halo properties.
Its five-fold evaluation is held out, but the fitted mean minimizes coordinate
errors rather than the decoded CoG errors used for scientific judgment. An
uneven or curved parameter geometry can therefore penalize harmless coordinate
motion while underweighting a smaller coordinate error that matters strongly
for the profile. Stage 3b tests that target mismatch directly.

Keep all of the following frozen relative to Stage 3:

- the same 2,539 galaxies and deterministic five outer folds;
- globally fixed compact Sersic index `n=1`;
- both the fixed `gamma=1.4` representation and the frozen smooth
  `gamma(log M_peak)` law, without retuning its endpoints, transition rule, or
  0.20-dex width;
- the four epoch-local DiffMAH coordinates plus concurrent `c_200c`;
- the selected sparse degree-two halo basis from Exp55; and
- the final-mass-only, mass-conditioned shuffled-MAH, and
  mass-conditioned shuffled-concentration controls.

For each outer fold, initialize the shared halo-to-coordinate coefficients at
the Stage 3 training-only least-squares solution. Then refit those same
coefficients by minimizing the unweighted residual in `log10 M*(<R)` over the
24 measured radii and outer-training galaxies after decoding the predicted
coordinates through the analytic profile. The directly fitted relation has no
coordinate-error term in its scientific objective and no access to the outer
validation CoGs. Refit every control with the same decoded-CoG objective.

The optimizer must pass three mechanical checks before scientific use:

1. its vectorized decoder agrees with the established one-galaxy decoder to a
   maximum absolute difference below `1e-10` dex;
2. its chained coefficient Jacobian agrees with an independent finite-
   difference Jacobian to relative error below `1e-5`; and
3. a synthetic shared relation is recovered to a maximum decoded-profile
   error below `1e-6` dex. Coefficient equality is not required because weak
   parameter directions are the reason to judge recovery in profile space.

Retain analytic stochastic draws by resampling complete four-coordinate
training residuals near each test halo, but rebuild the residual library around
the directly fitted mean. Select the residual-inflation factor through the same
four inner folds and nominal-68% profile-coverage rule as Stage 3; every inner
mean must also use the decoded-CoG objective. This tests the mean-target change
without silently borrowing the old mean's residual calibration.

Judge Stage 3b against the saved Stage 3 coordinate-regression baseline
separately for fixed and smooth outer slopes. The primary quantities are the
decoded held-out full-range and amplitude-pinned 5--30 kpc CoG RMS, profile
CRPS, the largest coherent halo-mass-bin shape residual, and the direct best,
typical, and worst profile curves. Also require no bootstrap-resolved
degradation in density RMS or absolute R50/R80/R90 error, no predicted mean
coordinate outside the established analytic-profile bounds, calibrated 68%
and 90% coverage, stable results when alternating fitting radii are removed,
and no new failure in the complete standard QA battery. Report coordinate
errors only as diagnostics; they cannot veto a better decoded CoG.

Adopt direct decoded-CoG fitting for later halo-information tests only if it
improves the coherent mass-bin shape residual while preserving the other
safeguards. If it lowers an aggregate RMS but leaves or worsens the visible
mass-bin residual, record it as another objective trade rather than a repair.
Do not interpret any radial assembly connection from Stage 4 until this test
establishes an adequate mean relation.

Before a full run, execute both representations, all four conditional means,
nested stochastic calibration, saved outputs, direct comparison figures, and
both standard-QA batteries on the existing mass-stratified 90-galaxy sample
with eight draws. The complete uncached path must finish in less than 60
seconds. Demo artifacts use a `stage3b_demo_` prefix and cannot overwrite any
completed Stage 1--4 product.

## Stage 3b result — target mismatch is real, but the direct loss is not a repair

The complete 90-galaxy path finished in 35.57 seconds, below the required
60-second limit. It included both representations, all four conditional means,
four-fold inner calibration, eight stochastic draws, an alternating-radius
refit, saved outputs, direct profile figures, and both standard-QA batteries.
The vectorized decoder agrees with the established decoder to
`1.78e-15` dex, the chained coefficient Jacobian agrees with finite differences
to relative error `7.17e-10`, and a synthetic shared relation is recovered to
`9.88e-12` dex in the decoded profile. The full five-fold calculation then
finished in 403.37 seconds for all 2,539 galaxies.

The direct optimization lowers its training CoG loss in every outer fold and
keeps every predicted mean coordinate inside the established analytic bounds.
It also demonstrates the suspected internal degeneracy: for fixed
`gamma=1.4`, held-out coordinate `R^2` is 0.864 for total stellar mass but
-0.280, -0.237, and -0.005 for compact fraction, compact radius, and extended
radius. Those negative coordinate scores do not by themselves reject the
model, because Stage 3b was deliberately judged after decoding. They show that
the optimizer can move substantially in weak analytic-coordinate directions
while leaving the cumulative profile nearly unchanged.

For fixed `gamma=1.4`, direct fitting changes the median per-galaxy full-CoG
RMS against TNG from 0.08415 dex for coordinate regression to 0.08396 dex. The
paired direct-minus-coordinate median improvement is 0.00020 dex, with the
16th--84th percentile bootstrap interval entirely below zero. Profile CRPS
also changes from 0.06403 to 0.06360 dex, and nominal 68% coverage is 68.96%
against the 68% target. The largest amplitude-pinned halo-mass-bin residual
falls from 5.59% to 4.44%, but its paired improvement is not resolved: the
16th, 50th, and 84th percentiles are -2.24, -1.19, and +0.38 percentage points.

Those small cumulative-profile gains are purchased by worse local structure.
Relative to coordinate regression, the paired median 5--30 kpc CoG RMS grows
by 0.00109 dex, full density RMS grows by 0.00282 dex, and 5--30 kpc density
RMS grows by 0.00289 dex; every corresponding bootstrap interval is above
zero. The median absolute errors in `log10 R50`, `log10 R80`, and `log10 R90`
grow by 0.00247, 0.00593, and 0.00444 dex, again with all three intervals above
zero. The standard annular-mass, density, size, and best/worst-galaxy figures
show the same trade rather than a hidden numerical failure.

The visually cleaner median QA curves do not contradict those per-galaxy
metrics. Under fixed `gamma=1.4`, 668 of 2,539 held-out galaxies (26.3%) have a
lower full-CoG RMS under direct training but a higher full-density RMS than
under coordinate training. The illustrative galaxy selected nearest the
median of that trade improves from 0.0910 to 0.0875 dex in full-CoG RMS while
worsening from 0.1286 to 0.1403 dex in density RMS. Its amplitude-pinned CoG
and finite-annulus density residuals are shown in
`exp56_stage3b_model_recipes`: population medians can cancel positive and
negative individual residuals, while a density derivative amplifies changes
between adjacent cumulative masses. Thus an average CoG-by-bin panel can look
better even when individual radial structure and sizes become less accurate.

The frozen smooth outer law reaches a similar conclusion. Its median full-CoG
RMS changes from 0.08427 dex for coordinate regression to 0.08410 dex for
direct fitting, but that small difference is not bootstrap-resolved. Its
largest halo-mass-bin residual worsens from 3.34% to 4.01%. Relative to its
coordinate baseline, the median 5--30 kpc CoG RMS grows by 0.00165 dex, full
density RMS by 0.00271 dex, 5--30 kpc density RMS by 0.00399 dex, and the
absolute R50/R80/R90 errors by 0.00222/0.00505/0.00228 dex; all of these
degradations are bootstrap-resolved. Both representations remain radially
stable: alternating fitting radii change their median decoded profile by only
0.00145 dex, below their roughly 0.004 dex representation floors.

**Decision:** do not adopt this direct decoded-CoG mean for subsequent
halo-information tests. The experiment confirms that the analytic coordinates
are only an intermediate bridge and that coordinate-space loss is not uniquely
privileged. It also shows that an unweighted loss on 24 cumulative masses is
not a sufficient end-to-end objective: correlated cumulative points allow a
small improvement in the average CoG while trading away differential density,
intermediate-radius shape, and sizes. The existing Stage 3 coordinate relation
remains the baseline, and the Stage 4 radial-information result remains
formally inconclusive under its already recorded adequacy failure. A future
test would need a separately predeclared profile-aware objective that balances
cumulative CoG, differential or annular structure, the 5--30 kpc region, and
sizes; it must not be tuned retrospectively within Stage 3b.

## Stage 4 predeclaration — radial halo-information content

Stage 4 asks what halo information improves the fixed-`n=1`, fixed-`gamma=1.4`
halo–CoG relation at different radii. It does not assume that recent assembly
must control the outskirts or that early assembly must control the inner CoG.
Those are hypotheses tested by the held-out comparisons, not constraints on
the fitted signs or radial trends.

The two scalar assembly summaries are derived only from each halo's portable
DiffMAH fit. Evaluate the fitted peak-mass history at the vendored TNG cosmic
times 3.28448, 5.87803, and 9.38911 Gyr for `z=2`, `z=1`, and `z=0.4`:

- recent growth is `log10 Mpeak(z=0.4) - log10 Mpeak(z=1)`, in dex;
- early assembled fraction is `Mpeak(z=2) / Mpeak(z=0.4)`; the regression uses
  its base-10 logarithm and reports the fraction for interpretation.

Do not use the raw TNG history summaries as predictors. Compare them with the
DiffMAH-derived values only as a data-integrity diagnostic.

### Stage 4 nested relations and controls

Fit these five outer-five-fold relations to the same fixed-slope analytic
profile coordinates:

1. final DiffMAH peak mass only;
2. final mass plus recent growth;
3. final mass plus recent growth and early assembled fraction;
4. the complete four-coordinate DiffMAH history; and
5. the complete DiffMAH history plus concurrent `c_200c`.

Use the same conditional model family for every nested relation: an intercept,
all included features linearly, and one quadratic term in final halo mass. This
keeps the mass curvature fixed while adding one measured information direction
at a time. Compare the last relation with the Stage 3 selected sparse mean as
an adequacy diagnostic; do not change the Stage 3 relation or profile family.

For the last four relations, construct a matched control by shuffling only the
new information among galaxies in ten final-mass quantile bins: recent growth,
early assembled fraction, all three non-mass DiffMAH coordinates, or
concentration, respectively. Refit every control inside each outer training
fold. A feature is not independently informative unless its real relation
outperforms both the preceding nested relation and its own mass-conditioned
shuffle.

Generate complete four-coordinate residual vectors with the same nearest-
neighbour resampling used in Stage 3. Select residual inflation separately for
each relation using four inner folds and the nominal-68% profile-coverage
target. No validation galaxy can enter a mean fit, residual library, neighbour
search, scale selection, or shuffle fit.

### Stage 4 measurements and judgment

Report held-out point-prediction RMS and stochastic CRPS at every cumulative
CoG radius, at every CoG-derived density radius, and for every differential
annulus. Also report medians over 2--10, 10--30, and 30--148 kpc, R50/R80/R90,
parameter conditioning inherited from the representation, feature
correlations, calibration, and direct best, typical, and worst galaxies.

For each added information set, bootstrap galaxies 1,000 times and report the
16th, 50th, and 84th percentiles of the real-minus-preceding and
real-minus-shuffled median metric differences. Call an improvement resolved
only when the 84th percentile is below zero in both comparisons. A localized
radial claim additionally requires at least two adjacent cumulative radii or
annuli to pass; an isolated bin is reported without interpretation. Report
null and adverse results as directly as improvements, without assigning an
inner or outer physical connection in advance.

Before the full run, execute the complete five-relation and four-shuffle path,
fold-internal stochastic calibration, saved outputs, all radial/annular and
direct figures, and final-relation standard QA on the same 90 mass-stratified
galaxies with eight draws. Require less than 60 seconds, finite monotonic CoGs,
five complete outer folds, and nominal-68% coverage within three percentage
points of 68%. The full calculation uses 32 draws only after this gate passes.
The full Stage 4 relation is adequate for interpretation only if its profile
CRPS and median CoG RMS are each no more than 1% worse than the Stage 3 fixed-
slope complete relation; otherwise report the measurements as inconclusive.

### Stage 4 small-scale validation result

The complete uncached path passed on exactly 90 mass-stratified galaxies in
21.31 seconds. It fitted all five nested relations and four matched shuffles,
calibrated each stochastic residual model within its outer training folds,
generated eight draws, saved all predictions, produced five direct comparison
figures, and completed the final relation's standard-QA battery. All means and
draws are finite and monotonic, all nine relations contain five complete outer
folds, and the final relation's nominal-68% profile coverage is 69.21% against
the required 65--71% interval. Its profile CRPS and median CoG RMS are 8.0% and
11.3% better than the Stage 3 fixed-slope relation on this validation sample,
so both adequacy safeguards pass. The full Stage 4 run is authorized; no radial
information claim is taken from this validation-scale result.

## Stage 4 result — structured signals, but the adequacy gate fails

The corrected full calculation completed for 2,539 galaxies with 32 draws in
250.90 seconds. All nine real and shuffled relations have finite, monotonic
CoGs and five complete held-out folds. The final complete-DiffMAH-plus-
concentration relation has nominal-68% coverage of 68.68% against the 68%
target. Its profile CRPS against the TNG CoGs is 0.06474 dex and its median CoG
RMS is 0.08644 dex. The Stage 3 selected sparse mean reaches 0.06403 dex and
0.08415 dex, respectively. Stage 4 is therefore 1.11% worse in CRPS and 2.73%
worse in median CoG RMS, failing both predeclared 1% adequacy safeguards.

Within the common linear-plus-mass-quadratic family, the radial comparisons are
highly structured:

- Adding DiffMAH-derived recent growth to final halo mass reduces median
  per-galaxy CoG RMS by 0.01484, 0.02211, and 0.02795 dex in the 2--10,
  10--30, and 30--148 kpc regions relative to final mass alone. The 84th
  percentiles remain below zero against both the preceding model and the
  mass-conditioned recent-growth shuffle in all three regions. The gain is
  present throughout the CoG and is largest outside 30 kpc.
- Adding the DiffMAH-derived early assembled fraction after recent growth
  reduces median CoG RMS by 0.00299 dex at 2--10 kpc, but only 0.00020 and
  0.00024 dex at 10--30 and 30--148 kpc. Its CoG-derived density and annular
  improvements are instead larger outside 10 kpc. All paired regional
  intervals and the two-adjacent-bin tests pass against both references, but
  this metric-dependent radial pattern does not establish an inner--early-MAH
  physical connection.
- Replacing the two explicit summaries with the four raw DiffMAH coordinates
  is worse than the summary relation by 0.00629, 0.00475, and 0.00493 dex in
  median regional CoG RMS. It still strongly beats the version whose non-mass
  DiffMAH coordinates are shuffled. Thus the full history contains predictive
  information, but this common low-complexity mean uses the two physical
  summaries more efficiently.
- Adding `c_200c` to the complete DiffMAH coordinates reduces median CoG RMS by
  0.00559, 0.00649, and 0.00841 dex across the same regions. It beats both the
  preceding relation and shuffled concentration in every reported metric over
  adjacent bins, confirming that concentration carries additional predictive
  information within this model family.

The two DiffMAH-derived summaries are faithful but not identical proxies for
the raw histories. Recent growth correlates with its raw-history measurement at
Spearman `rho=0.719` for all 2,539 galaxies; early assembled fraction correlates
at `rho=0.923` for the 2,537 histories that reach `z=2`. The fitted recent and
early summaries are themselves strongly anticorrelated (`rho=-0.874`), so the
nested early-fraction result is an increment conditional on a closely related
recent-growth coordinate.

The standard QA exposes the same limiting issue as Stage 3: population scatter
is compressed. For example, the fixed-kpc `<30` versus `30--50` kpc plane has
0.050 dex model scatter against 0.172 dex in TNG, and the stellar-mass--R50
plane has 0.054 dex model scatter against 0.173 dex in TNG. The test therefore
delivers useful guidance for the next mean architecture, but the frozen
adequacy rule makes its radial attribution formally inconclusive. Record the
signals as hypotheses to retest with the selected Stage 3 mean architecture;
do not encode an outer--recent or inner--early assignment as established
physics.

## Review

- The representation-only test was written here before the driver.
- `run.py` contains separate `demo` and `run` commands; `run` refuses to start
  unless the saved 90-galaxy record passed the sub-minute and numerical gates.
- The full 2,539-galaxy grid ran only after that saved validation gate passed.
- All three full-sample PNG figures were inspected directly; their PDF
  companions were generated by the same plotting calls. Bootstrap intervals
  are shown on the scalar 5--30 kpc comparisons.
- Ruff, Python compilation with warnings treated as errors, and whitespace
  checks pass after the final figure correction.
- The separately predeclared Stage 1b joint-global grid passed its 90-galaxy
  gate, completed on all 2,539 galaxies, and was reanalyzed from cached fits
  after correcting the boundary-incidence summary. The measured full-run time
  remains 1,236.58 seconds; the cached reanalysis took 45.82 seconds.
- No held-out halo–CoG relation, null-control fit, or repository-wide test
  collection was attempted during Stages 1 and 1b. Stages 3 and 4 subsequently
  completed their declared held-out relations and null controls. Repository-
  wide collection remains blocked by the pre-existing missing Exp07 input
  stated above.
- Stage 3 and Stage 4 each passed a complete uncached 90-galaxy path in 31.18
  and 21.31 seconds before their full runs. Every new PNG and PDF was inspected
  through direct figures and contact sheets. Stage 4's corrected full rerun
  differs only in the finite-pair raw-history diagnostic; its fitted relations
  and decision are deterministic and unchanged.
