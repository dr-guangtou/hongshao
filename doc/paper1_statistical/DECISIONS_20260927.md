# Paper 1: agreed scope and representation strategy

User decisions September 27, 2026. This note supersedes provisional choices
in the September 26 draft. It authorizes planning choices, not an unrequested
fit, data query, merge, or application to a new simulation.

## Agreed scientific scope

The project predicts projected stellar mass distributions from halo MAHs and
other halo properties. Accurate recovery of an individual MAH is neither a
Paper 1 objective nor a realistic goal for the project. Later work will
combine N-body halos and observations to constrain the galaxy–halo connection
statistically. A small reverse-summary diagnostic remains optional supporting
evidence, not a headline, required figure, or completion criterion.

Paper 1 should establish a useful forward conditional relation for massive
central galaxies, using straightforward methods and their 1-D stellar mass
distributions alone. The main demonstration is TNG300 at z=0.4. Independent
fits at other epochs may enter an appendix if scientifically useful; no joint
model of profile evolution or cross-epoch latent distribution is required.
Clarity and one take-home message matter, not an imposed figure/page count.

Use the native outer CoG aperture near 148 kpc as the stellar-mass anchor,
retaining the exact stored radius in calculations and captions. This is the
recoverable finite-aperture total, not an infinite stellar mass. The core
claim concerns information beyond stellar mass; beyond mass plus size is a
separate, important stronger test.

The user approves an epoch-relevant Paper 1 selection, without exclusions
based on unrelated multi-epoch histories. Predeclare genuine measurement and
input quality criteria. Compare the old strict selection as a sensitivity;
do not modify physical experiments' selections or their archived samples.

Xu and Leidig are collaboration precursors, not simply competing work.
Xu supplied these data, and the satellite-particle treatment matches Xu's
work rather than Leidig's. Xu and HongShao focus on radial ranges relevant
to measurements of individual galaxies. Document the source pipeline and
array definitions; do not explain isophotal/aperture differences by satellites
that have already been removed, or assert that clipping removes smooth ICL.

## 1. Use both measured and DiffMAH histories

Yes. This tests both the existence of the forward relation and how compactly
its halo inputs can be described. It is not a contest between an assumed
“truth ceiling” and a supposedly universally portable model.

### Matched primary comparison

- Measured histories: normalize consistently, compress with a training-fold
  PCA, and use a simple regularized halo-to-profile relation.
- DiffMAH histories: fit the same underlying mass definition over the same
  time interval with the same endpoint convention, then use the fitted
  coordinates in an equivalently controlled relation.
- Both retain the identical exact present halo-mass controls, concentration
  treatment, galaxy sample, outer folds, stellar targets, and score definition.
- State parameter counts: three shape parameters with an externally fixed
  normalization is different from fitting an additional normalization. Do not
  let fitted amplitude covertly supply a different current-mass measurement.
- A raw non-monotone history, its running peak, and a smooth fit are different
  objects. Record which operation changes the scientific input. Historical
  official DiffMAH/SubhaloMass and measured M200c curves cannot be treated as
  a pure measured-versus-smoothed comparison without reconciling definitions.

### One bridge comparison, if the primary results differ

Evaluate the fitted DiffMAH curves on the measured history's time grid and
project them through the same training-only MAH basis. This holds the feature
coordinates fixed while changing the curves. Also compare simple summaries
computed from either curve if raw DiffMAH parameters are poorly conditioned.

This distinguishes loss from smoothing/history coverage from difficulty
learning a relation in DiffMAH's parameter coordinates. Keep that diagnostic
on common feature support and state any extrapolation. It does not uniquely
decompose all sources of prediction error.

### Interpretation

- Agreement supports a compact description of the assembly dependence.
- Better measured-history predictions suggest useful structure beyond the
  smooth description, after endpoint, mass-definition, and flexibility checks.
- Better DiffMAH predictions could reflect useful regularization of noisy
  measured histories, not extra physical information.
- Either outcome can support the paper. Do not preselect a winner or demand
  that the two descriptions produce identical raw coefficients.

## 2. Measured profiles, PCA, and analytic CoGs have different roles

The measured CoG is the common evaluation target. Coarse annular masses and
the measured 1-D density provide essential radial diagnostics. The measured
isophotal density is not silently equated to the derivative of an aperture
CoG; compare like measurements, using the appropriate geometry.

| Description | Role | What must be checked |
|---|---|---|
| Direct radial/coarse-annular targets | Reference with minimal functional-form assumptions | Noise, covariance, radial support, and positivity of generated profiles |
| Mass anchor plus PCA shape coordinates | Primary low-dimensional statistical description | Held-out compression, preserved annular structure, and physically valid correlated draws |
| Existing analytic CoG parameters | Bounded alternative with smooth curves and explicit mass conservation | Approximation bias, stable coordinates, halo predictability, and density/size QA |

The functional-form work is therefore relevant, but should serve the central
forward-prediction result. It should not turn Paper 1 into a history of symbolic
searches or a second paper on a new profile formula.

### Recommended analytic candidate: the existing cubic-logit form

Start with the documented Exp62 family, not the free damped-cosine family.
The latter's accuracy did not resolve its weakly identified parameters in
Exp67/69/70. Similar profiles can have very different fitted coordinates;
that is an obstacle to learning a smooth halo-to-coordinate relation.

For cubic-logit, the resolved-range description can be written

\[
F(R)=\operatorname{sigmoid}[a x+b x^2+c x^3],\quad
x=\ln(R/R_s),\quad a=m+b^2/(3c),\quad m,c>0,
\]

\[
M_\star(<R)=M_{\star,148}\,F(R)/F(R_{\rm ap}).
\]

This has one finite-aperture mass coordinate and four shape/scale coordinates
`R_s, m, b, c`. The polynomial derivative is positive, so cumulative mass
increases. Here `R_s` is the underlying family's infinite-aperture half-mass
radius, **not** the measured finite-aperture R50. The latter satisfies
`F(R50) = F(R_ap)/2` and must be derived consistently.

Exp62 supports using this form as an interpolator over the measured radial
range, not as a physical central density law or an infinite-total estimator.
Its inner continuation has a density turnover. Do not extrapolate inward to
claim a resolved galaxy core or outward to replace Mstar,148 by infinite mass.
Its earlier halo-predicted coordinates were not a demonstrated high-accuracy
replacement for direct/PCA predictions. Existing promotion as an interpolator
does not automatically promote it as the paper's forward model.

### Evaluate the whole chain

1. **Representation:** fit/encode the measured stellar profile itself. Compare
   reconstructed CoG, coarse annuli, density, and finite-aperture radii with
   the data, including tails and parameter stability.
2. **Prediction:** infer all profile coordinates, including Mstar,148, from
   halo inputs only; evaluate the reconstructed stellar profile on held-out
   galaxies. Measured stellar mass/size cannot supply prediction normalization.
3. **Population:** draw correlated coordinates and test mass/size planes,
   radial scatter, physical validity, and assembly-conditioned trends.

A family can pass step 1 and fail step 2 or 3. Conversely, a slightly less
accurate per-galaxy interpolator can be a better predictable representation.
Score the common observables, not the R-squared of differently defined fitted
parameters. Report representation and prediction errors separately without
assuming they add in quadrature.

Use a staged comparison: settle measured/DiffMAH inputs with the common PCA
and direct references first; then substitute the single analytic candidate.
Only run the full two-input-by-representation comparison if this candidate
passes the representation and coordinate-stability checks. No broad new
family search or reopening of protected Exp67 samples is implied.

## 3. Size is a control, not a required observational interface

Keep fixed-kpc CoGs and aperture/annular masses as the main observables. R50
is a derived diagnostic and a stringent control for deciding whether the
information is merely a size change. Do not make the observational method
depend on normalizing every profile by a precisely measured size.

The user's concern about size sensitivity is a motivation to test, not a
blanket claim that aperture profiles are immune to the same problems.
Finite depth, outer-mass definition, sky, geometry, and M/L assumptions can
affect both size and profiles. If the paper argues a practical advantage,
compare their response under the same perturbations; distinguish half-light
from half-mass radius. In TNG, the mass+size control still matters even if
observational size is uncertain.

## 4. N-body application: useful, with a precise interpretation

An optional application can populate a supplied N-body halo catalog using
its histories and other permitted halo properties. Freeze the TNG-trained
mapping, reconcile halo definitions/units/epoch/time grid, inspect feature
support and resolution, and generate correlated stellar profiles including
their mass amplitudes. Flag extrapolation; do not conceal it by clipping.

This demonstrates practical halo-to-profile generation. It does not validate
stellar accuracy in the new catalog, which has no stellar truth, nor prove
that the TNG calibration is universal. A hydro/DMO matched check would test
input transfer more directly; new data are not assumed available. The user
can decide which external simulation/catalog is appropriate after the TNG
relation and input contract are established. No lensing/clustering pipeline
or downstream inference machinery is required for Paper 1.

## Immediate next planning step

Freeze an epoch-relevant data dictionary and sample protocol, then specify
the matched measured-MAH/DiffMAH comparison with a common PCA/direct stellar
reference. Add the existing cubic-logit form as the one bounded alternative.
Keep the central conclusion about forward predictive information beyond
stellar mass, with the size-controlled result refining—not replacing—that
conclusion.
