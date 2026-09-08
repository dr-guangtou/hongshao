# Direction review after Exp75 / Exp77

Review only; no experiment ID reserved, science driver written, data copied,
or fit performed. Exp78 belongs to Claude and studies the size-aware objective.

## Recommended complementary question

Does the physical model's rule for the radial extent of stellar deposits
introduce an avoidable dependence on mass outside the measured aperture?
Start with a diagnostic of the adopted Exp74 measured-input baseline, not
a new correction map or a new deposit family.

Evidence motivating this question:

- Exp74 README's final input comparison explicitly leaves the extended
  deposit's halo-radius scaling for re-tuning: measured histories improve
  important population comparisons but leave excessive high-redshift outer
  masses. This is not evidence that measured MAHs should be discarded.
- Exp63 Stage 2b demonstrates an older failure in which compensating deposit
  sizes and delayed arrival improved the fitted epoch but spoiled earlier
  predictions. It motivates tracing the mass budget, not assuming the current
  baseline suffers the same failure or repeating the rejected delay model.
- `doc/open_questions.md` B2 still lists the deposit truncation convention
  as unscanned. Exp63 Stage 0 justified retaining truncation against an
  untruncated reference; it did not establish robustness across truncation
  radii. Keep the adopted truncation as the reference, not silently remove it.
- Exp77's annular-objective result supports checking where stellar mass lands
  explicitly. Its learned correction is closed and must not be revived here.

## Proposed sequence, subject to user approval

1. Freeze and reproduce the adopted baseline on a declared common sample.
   Trace each channel's contribution by deposition epoch, and distinguish
   total deposited mass from mass inside 30, 50, 100 and 148.22 kpc. Mass
   outside the last measured aperture is a model extrapolation, not a target.
2. Measure sensitivities to existing extended-deposit size parameters and
   the truncation convention separately. Hold the true post-truncation
   half-mass coordinate definition fixed. Record invalid parameter domains;
   do not silently clip them. Begin without refitting or adding parameters.
3. Only if the diagnostic identifies a useful direction, predeclare matched
   refits of existing physical parameters. Keep one shared parameter vector
   across epochs. Adopt Exp78's objective only after Claude freezes it; do
   not jointly change the objective and radial prescription without controls.
4. Require a measured sub-minute check before larger execution, synthetic
   mass-conservation and reference-recovery checks, and full standard QA for
   any candidate. Inspect individual and halo-mass-binned CoGs, outer masses,
   density profiles, sizes, mass planes and cross-epoch growth. Report mean
   accuracy and population scatter separately.

This adds zero fitted population parameters in its first phase. A later
extension needs a separately approved physical motivation and a full parameter
count. Do not claim improved predictive performance from conditional scans.

## Alternatives and boundaries

Early-MAH extrapolation/smoothing is a useful separate sensitivity audit, but
it changes the input and is already owed in the Exp74 handover. A simple
universal delivery delay and broad transport are not fresh ideas: consult
Exp63 Stage 2b and Exp52/57 before proposing them again. Merger-tree information
remains a longer-term physical direction requiring a separate data-query plan.

The next experiment number must be rechecked immediately before creation.
