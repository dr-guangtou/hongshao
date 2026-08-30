# Exp69 — Damped-cosine reparameterization

Status: STOPPED AT THE PREDECLARED OPERATIONAL GATE. The complete gate took
61.39 seconds, above the frozen 60-second limit. Discovery, selection, and
validation were not entered; Exp67 selection and validation fits remain
untouched.

## Narrow question

Exp67 found that the free damped-cosine projected-density family reconstructs
the measured curves of growth (CoGs) more accurately than unrestricted double
Sersic at every epoch, but its damping coordinate is nearly unidentified. This
experiment asks one question: can a controlled change of coordinates, a fixed
damping choice, or a deterministic branch rule remove that degeneracy without
losing the measured CoG and shell-density accuracy?

This is not a new symbolic search. The squared-slope wrapper and the single
cosine atom are frozen. No new transition, packet, chirp, logarithmic-periodic,
or composed expression may be added after this declaration.

## Data and blind population roles

The experiment inherits Exp67's seed-67 galaxy-level split without changing a
row or moving an epoch:

- discovery: 1,200 galaxies and all five epochs, used to compare the declared
  coordinate systems, choose one global fixed damping value, and freeze at most
  two eligible systems for selection;
- selection: 1,000 galaxies and all five epochs, loaded only if a discovery
  candidate passes every discovery gate, used to choose one final coordinate
  system and one canonical branch rule;
- untouched validation: 1,180 galaxies and all five epochs, loaded only after
  one finalist is frozen, used once under all three fitting objectives and for
  the final promote-or-reject decision.

The preserved, gitignored Exp67 discovery fits are read through
`HONGSHAO_EXP67_ROOT`. Their stored indices must equal the recomputed discovery
indices exactly. Code in the operational and discovery stages must reject any
archive whose indices include an Exp67 selection or validation profile. All
stellar masses are the primary one-dimensional CoG-derived, h-free
`log10(M / Msun)` measurements; direct two-dimensional aperture masses are not
fitting targets.

The 25-galaxy operational sample is the exact Exp67 discovery-only operational
sample: 20 finite-halo-mass quantiles and five missing-halo-mass galaxies, with
all five epochs retained. It is operational only and cannot select a model.

## Frozen physical family and references

All candidates retain

```text
u = ln(1 + R/R_c),
Sigma(R) = Sigma_0 exp[-Q(u)],
Q'(u) = (1-exp(-u)) {2.02 + [a + b T(u)]^2},
T(u) = exp(-lambda u) cos(omega u),
a > 0, omega > 0, lambda >= 0.
```

The amplitude is `log Mstar(<148.2 kpc)`. The exact Exp67 free family, with
coordinates

```text
(log Mstar_148, log R_c, log a, b, log omega, log lambda)
```

and the Exp67 physical bounds is the functional and numerical reference. It is
refit with the same starts and objectives as every candidate; the archived
two-start Exp67 solution is retained as a read-only reproduction check. The
nested atom-removed reference fixes `b = 0`; in that limit `omega` and `lambda`
are declared unidentified and are not interpreted.

Unrestricted double Sersic is the paired raw-accuracy reference, cubic-logit is
the paired conditioning and resolved-CoG reference, and PCA(3) is the numerical
compression reference. These three established references are read, never
refit or retuned.

## Predeclared coordinate systems

Let `R_min = 2.0 kpc`, `R_max = 148.2 kpc`,
`u_min = ln(1 + R_min/R_c)`, `u_max = ln(1 + R_max/R_c)`,
`Delta_u = u_max-u_min`, and `u_p = (u_min+u_max)/2`.

1. **Original free coordinates.** The exact Exp67 six-coordinate system above,
   newly fit with the common multi-start protocol. This is the reference, not a
   candidate promoted by its own result.
2. **One global fixed damping.** Four five-coordinate families fix
   `lambda = 0, 0.25, 0.75, 1.5`. These are the undamped nested limit and the
   three damping values already declared before Exp67. Discovery selects at
   most one value globally across every galaxy and epoch; damping may never be
   chosen per profile or per epoch.
3. **Finite-window pivot coordinates.** Optimize

   ```text
   (log Mstar_148, log R_c, log(a^2),
    c_p, log d, log psi),
   c_p = b exp(-lambda u_p),
   d = lambda Delta_u,
   psi = omega Delta_u.
   ```

   Thus the shape coordinates are the asymptotic slope excess `a^2`, signed
   modulation at the middle of the measured window, decay across that window,
   and phase advance across that window. Decoding is
   `a=sqrt(exp(log(a^2)))`, `lambda=d/Delta_u`,
   `omega=psi/Delta_u`, and `b=c_p exp(lambda u_p)`. Bounds are computed once
   from the Cartesian image of the frozen Exp67 physical bounds over the
   declared `R_c` interval; no quantile or result-dependent bound may be used.
   The original bounded family is therefore nested inside this declared window
   box, and results outside the original physical box are reported separately.
4. **Finite-window orthogonalized modulation.** At every parameter vector,
   regress `h(u)=exp(-lambda u) cos(omega u)` on the two columns `1` and
   `g_c(u)=1-exp(-u)` at the 24 measured radii with uniform weights. If the
   fitted core coefficient is `beta_c`, use

   ```text
   T_perp(u) = h(u) + beta_c exp(-u)
   ```

   and interpret `a` as the asymptotic slope coordinate because
   `T_perp(infinity)=0`. This is the finite-window residual after its constant
   part is absorbed into `a`; its measured-window modulation is orthogonal to
   the constant and first-order core-scale directions. It retains six fitted
   coordinates, the squared-slope physical guarantees, and an elementary
   finite exponential--trigonometric expression for `Q(u)`. Because it changes
   the basis function rather than only relabeling coordinates, it is reported
   separately from the exact pivot reparameterization.
5. **Canonical conditioned branch.** For the original and pivot systems, keep
   every converged start whose mean squared primary-objective residual is at
   most 1.01 times the smallest value plus `1e-14`. Select the solution with the
   largest minimum singular value of the residual Jacobian after every native
   coordinate is scaled by its full declared range. Ties within `1e-12` in that
   singular value are broken by lower objective and then lexicographically by
   `(d, psi, abs(c_p), c_p)`. This deterministic rule is evaluated both as a
   branch label and as a candidate output; it does not delete or hide the other
   near-optimal solutions.

No coordinate system can be added or altered after the operational results are
seen. The fixed-damping list can only shrink through the declared discovery
ranking.

## Fitting objectives and common measurements

Discovery uses only the primary deterministic loss. These are fitting losses,
not likelihoods:

1. primary: uniform-radius RMS in `log10 Mstar(<R)` at the 24 measured radii;
2. secondary: RMS of linear enclosed-mass difference divided by measured
   `Mstar(<148.2 kpc)`;
3. local: RMS in logarithmic mean shell surface density, including 0--2 kpc
   and using a 1 Msun floor only for an exactly zero measured shell.

The finalist is refit under all three objectives. Every fit reports full and
5--30 kpc CoG RMS, shell-density RMS, fixed-kpc aperture and annular masses,
the 100--148.2 kpc shell mass, and R20/R50/R80/R90 errors. The primary fit uses
four separated starts in the operational gate and eight in discovery,
selection, and validation.

## Synthetic recovery and mathematical checks

Before measured-profile fitting, every declared system must pass:

- exact aperture normalization and strictly increasing numerical CoGs;
- finite positive central density, zero central derivative, non-increasing
  projected density, asymptotic slope steeper than `R^-2.02`, finite projected
  mass, and nonnegative spherical Abel deprojection on a dense grid;
- analytic `Q(u)` versus doubled-resolution numerical integration to a maximum
  absolute difference below `3e-7`;
- CoGs at 512 and 2,048 integration points differing by less than `4e-4 dex`;
- noiseless recovery for each system at the declared low, middle, and high
  damping cases and both contrast signs, from four displaced starts, with
  full-CoG RMS below `3e-5 dex` and maximum common physical-descriptor error
  below 0.03 of its declared range after the canonical branch rule.

The fixed-damping systems are tested on synthetic profiles generated by that
same fixed system. The atom-removed case tests decoded-profile recovery only;
its absent harmonic coordinates are explicitly excluded from parameter
recovery.

## Degeneracy diagnostics

For every start, store its coordinates, decoded common physical descriptors,
objective, convergence flag, predicted CoG, residual Jacobian, scaled singular
values, and weakest right-singular vector. Coordinate rescaling alone is not
accepted as evidence: both the native-range-scaled Jacobian and the decoded
profile excursion along its weakest direction must improve.

The operational figures and stored tables include:

- overlays of all multi-start CoGs and residuals, with every solution within
  1% of the best loss highlighted, for the common best, typical, and worst
  reference profiles;
- parameter paths and decoded CoGs at 21 equally spaced locations from -0.5 to
  +0.5 of the available normalized distance along the weakest singular
  direction;
- 9 by 9 profiled damping--contrast and damping--core loss scans for one
  synthetic profile and the weakest measured reference profile, reoptimizing
  the remaining coordinates from the neighboring grid solution;
- if discovery is entered, 25 by 25 versions for the three profiles with the
  weakest original-family Jacobians;
- the maximum decoded-CoG RMS separation among every pair of solutions within
  1% of the best loss, never only the selected solution.

## Coordinate-continuity tests

Continuity is measured only on a stage already authorized for fitting. Within
each epoch, amplitude-pin every measured CoG at 148.2 kpc and find each
profile's nearest other profile by full-CoG RMS. For those fixed pairs, report
the median and 95th percentile Euclidean separation in the common descriptors
`(log R_c, a^2, c_p, d, psi)`, each divided by its declared range. Separately,
report the same descriptor separation for each galaxy across its four adjacent
epoch pairs. The neighbor pairs are chosen from measured CoGs once and reused
for every coordinate system.

A continuity improvement requires both 95th-percentile separations to be at
most 0.75 times the newly refit original-family value and neither median to be
larger than the original value. All raw distributions and the profiles with
the ten largest coordinate jumps are retained; the canonical rule cannot pass
by hiding a discontinuous alternative solution.

## Frozen numerical gates

The 25-galaxy, 125-profile operational path must complete all mechanics,
synthetic recovery, four-start fitting, serialization, continuity calculations,
and the five diagnostic figure classes in less than 60 measured seconds. It is
an operational gate only. Failure stops the experiment before discovery and
does not permit fewer candidates, starts, scans, or figures.

A system is discovery-eligible only if all of the following hold on the 6,000
discovery profiles:

- at least 99.9% of fits converge;
- fewer than 5% of fits place any native coordinate within 1% of a finite
  declared bound;
- at every epoch its median scaled minimum-Jacobian strength is at least 0.10
  of cubic-logit's paired value, and at least four epochs reach 0.20;
- at every epoch that strength is at least four times the newly refit original
  free damped-cosine value;
- the maximum near-optimal decoded-CoG separation is below `0.003 dex`;
- at every epoch its median full-CoG and shell-density RMS are no more than 10%
  larger than the newly refit original free damped cosine;
- it passes the predeclared coordinate-continuity improvement above.

Among eligible fixed-damping cases, choose the one with the lowest worst-epoch
full-CoG RMS ratio to the original family; break ties by shell-density RMS and
then smaller `lambda`. At most the best fixed case and the best eligible free
coordinate case proceed to selection.

Selection tightens the finite-bound incidence to below 1%, requires at least
four epochs at 0.25 of cubic-logit's paired Jacobian strength and every epoch at
0.15, limits maximum near-optimal decoded-CoG separation to `0.001 dex`, and
retains the 10% per-epoch CoG and shell-density accuracy allowance and the same
continuity rule. Eligible systems are ranked by worst-epoch full-CoG RMS ratio
to the original, then worst continuity ratio, then coordinate count. Exactly
one finalist is frozen.

Validation accepts the finalist only if every selection numerical gate repeats
under the primary objective and, under each of the two secondary objectives,
every epoch's median full-CoG, 5--30 kpc CoG, shell-density, and R50/R80/R90
error is no more than 10% larger than the original free family fitted under the
same objective. A failed mathematical, accuracy, convergence, boundary,
Jacobian, multi-start, or continuity gate cannot be traded against another
metric.

## Visual gates

All figures use `hongshao.plotting.set_style()`, the Okabe--Ito palette, and
`cividis` for same-sign sequential maps, and are saved as PNG and PDF with
`hongshao.plotting.save_fig()`. Every generated figure is displayed and listed
for user review in the generating turn.

The operational figure classes are: synthetic recovery; accuracy versus
conditioning; all-solution overlays; weakest-direction paths plus the two
profiled two-dimensional scans; and coordinate-continuity distributions with
the largest jumps shown directly. If any direction is discovery-eligible, the
average measured and fitted CoGs in halo- and stellar-mass bins and the three
z=0.4 stellar-mass planes are mandatory before it can be called interesting.
Only a selection-eligible finalist receives the complete standard aperture,
annular, outskirt, density, size, cumulative-residual, and common best/typical/
worst individual-object QA battery. A validation-stage finalist also receives
the five-epoch stellar-growth diagnostics.

## Result

The analytic and synthetic mechanics passed for all seven declared optimizer
systems and six synthetic cases per system. The standalone mechanics path took
11.04 measured seconds. Every synthetic full-CoG recovery RMS was below the
predeclared `3e-5 dex` reference limit, and every common physical-descriptor
recovery error was below 0.03 of its declared coordinate range. Thus the
coordinate maps and exact elementary density integrals work as intended.

The complete operational path then fit all seven optimizer systems to 125
discovery-only profiles from 25 galaxies, using four starts per fit; stored all
solutions; ran the synthetic recovery, weakest-direction, profiled two-
dimensional scan, and continuity diagnostics; serialized every result; and
rendered all five declared PNG/PDF figure pairs. It took 61.39 seconds, which is
1.39 seconds slower than the frozen 60-second operational target. The declared
rule says an over-time gate stops the experiment without reducing the candidate
set, number of starts, profiled scans, or figure set. Therefore the 6,000-
profile discovery stage was not entered, and none of the following operational
measurements may be used for model selection.

On this operational-only sample, the newly refit original free damping had a
median full-CoG RMS of 0.002040 dex relative to the measured CoGs, a median
shell-density RMS of 0.06645 dex relative to the measured annular densities,
98.4% optimizer convergence, 17.6% of fits within 1% of a finite coordinate
bound, and median scaled minimum-Jacobian strength equal to 0.02335 of the
paired cubic-logit value. Its maximum CoG separation among solutions within 1%
of the best loss was 0.001193 dex.

Fixing `lambda=0.75` exposed the clearest accuracy--conditioning trade on the
same 125 profiles. Its median scaled minimum-Jacobian strength was 0.4482 of
the paired cubic-logit value, 19.19 times the original free-damping reference;
its nearest-profile and adjacent-epoch 95th-percentile coordinate separations
were respectively 0.597 and 0.630 times the original-family values, so both
were smaller than the declared 0.75 continuity target. However, its median
full-CoG RMS was 0.002267 dex, 11.12% worse than the original's 0.002040 dex and
outside the 10% discovery accuracy allowance, and only 99.2% of its fits
converged compared with the required 99.9%. This small sample therefore shows
that fixed damping can remove much of the local null direction, but it does not
establish an acceptable population coordinate system.

Fixing `lambda=0.25` reached 100% convergence and median scaled minimum-
Jacobian strength equal to 0.4226 of cubic-logit's paired value, 18.10 times the
original free-family strength. Its median full-CoG RMS was 0.002491 dex, 22.10%
worse than the original measured-CoG error, and its median shell-density RMS
was 0.08831 dex, 32.91% worse than the original measured-annulus error. Fixing
`lambda=1.5` kept the median full-CoG RMS to 0.002182 dex, 6.93% worse than the
original, and the median shell-density RMS to 0.06877 dex, 3.50% worse than the
original, but 10.4% of fits lay within 1% of a finite bound compared with the
5% discovery limit; its nearest-profile and adjacent-epoch 95th-percentile
coordinate separations were 0.852 and 0.810 times the original values, both
larger than the 0.75 continuity target. The undamped `lambda=0` limit was worse
than the original in median full-CoG RMS, shell-density RMS, convergence, and
near-optimal CoG spread. No fixed value cleared all operational analogues of
the frozen discovery gates.

The exact finite-window pivot coordinates preserved raw reconstruction: their
median full-CoG RMS was 0.002046 dex, only 0.28% worse than the original's
0.002040 dex. They did not remove the degeneracy: their median scaled minimum-
Jacobian strength was 0.01483 of cubic-logit's paired value, only 0.635 times
the already weak original coordinate strength, and only 96.8% of fits
converged. Finite-window orthogonalization was worse still, with median scaled
minimum-Jacobian strength equal to 0.00540 of cubic-logit's paired value, 70.4%
convergence, and 39.2% of fits within 1% of a finite bound. The mechanically
motivated free-coordinate changes therefore give no operational evidence that
relabeling damping removes the null direction.

The canonical conditioned-branch rule left the original-family median full-
CoG RMS effectively unchanged at 0.0020404 dex versus 0.0020402 dex for the
minimum-loss reference. Its median scaled minimum-Jacobian strength increased
only from 0.02335 to 0.02547 of cubic-logit's paired value, a factor of 1.091,
while its nearest-profile and adjacent-epoch 95th-percentile coordinate
separations remained 0.941 and 0.971 times the minimum-loss original values.
Those reductions are much smaller than the required factor of 0.75. A branch
tie-breaker alone does not remove the measured local degeneracy in this sample.

### Decision

Stop before discovery because the complete operational path missed its frozen
runtime gate. The operational figures suggest that globally fixing damping is
the only declared mechanism that materially strengthens the coordinates, but
the tested fixed values trade that gain against accuracy, convergence,
boundary incidence, or continuity. The exact pivot coordinates,
orthogonalization, and canonical branch rule do not remove the weak direction.
Because the test stopped at 25 galaxies, the population question remains
unresolved and no coordinate system is promoted or sent to Exp67 selection or
validation.

## Review checklist

- [x] Predeclare the narrow family, exact references, blind population roles,
  coordinate systems, objectives, synthetic recovery, Jacobian scans,
  multi-start overlays, continuity tests, thresholds, and sub-minute gate
  before writing evaluator or driver code.
- [x] Write focused checks and observe them fail because the evaluator and
  driver do not yet exist.
- [x] Run the complete 25-galaxy operational path. **Result:** 61.39 seconds,
  so it failed the frozen 60-second gate and stopped the experiment.
- [x] Enter discovery only if the operational gate passes without alteration.
  **Not triggered:** the operational gate failed.
- [x] Keep Exp67 selection and validation unseen until their declared gates.
- [x] Display and inspect every generated figure in its generating turn.
- [x] Record the final measured result without adding a post-result family.
