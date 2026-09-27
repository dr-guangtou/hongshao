# Decisions and questions for the user

## September 28 resolutions and remaining source question

- Destination: independent sibling repository `ushmr1`.
- Retain all qualifying centrals; no extra stellar-population selection.
- Work on analysis, figures, and Markdown evidence notes, not a LaTeX draft
  except captions. Adopt the figure rules in the launch plan.
- Seek evidence for the remembered peak-mass parent selection. Existing
  project statements support it; the original supplied format note does not
  certify the query. See [source audit](SAMPLE_PROVENANCE.md).

The only immediate external question is for the data provider: what exact
mass field, threshold, units, peak interval, and central definition generated
the original IDs? No need to revisit the agreed scientific scope. N-body
catalog and observational-error choices remain optional extensions, not
blockers for preparing the TNG analysis.

## Resolved September 27

The user answered the six scope questions below. Their original wording is
retained as the discussion record, not as pending requests. The authoritative
update is [Agreed scope and representation strategy](DECISIONS_20260927.md).

- Forward prediction, TNG300 proof of principle; accurate individual-MAH
  recovery is not a goal. A small reverse test is optional supporting evidence.
- Begin at z=0.4; independent other-epoch results may enter an appendix.
  There is no figure/page limit, but one clear take-home message is required.
- The epoch-relevant sample exception is approved. Exact applicable quality
  criteria still need to be written before fitting, not inferred from older
  physical-model selections.
- Use Mstar at the native outer CoG aperture near 148 kpc. Test beyond stellar
  mass and additionally beyond stellar mass plus size.
- Xu supplied the data; satellite treatment matches Xu's work. Xu and Leidig
  are collaboration precursors, with different measurement scopes.
- The empirical statistical relation is an appropriate Paper 1 result;
  completion of the physical prescription is not required.

Remaining implementation choices are the exact quality/support rules,
matched history source and fit window, the bounded analytic-profile comparison,
and which N-body catalog (if any) to use for a forward-generation demonstration.
These are not permission to start a new data query or fit in this turn.

## Original questions (September 26; superseded where answered above)

The original plan could be discussed without settling every detail at once. The first
three choices determine the scientific scope; the remaining choices refine
implementation and manuscript coordination. No fitting has begun.

## 1. What strength of conclusion should Paper 1 aim for?

**Recommendation:**

If the intended headline is actual observational recovery, we need realistic
profile and halo-mass errors, selection, and probably an external/mock-data
check. Is that essential to this first paper, or a follow-up?

## 2. Is z=0.4 the main scope, and what length/journal do you prefer?

**Recommendation:** one low-z epoch and about five main figures; the exact
page length is an editorial choice, not a requirement. This avoids making
progenitor selection, redshift-dependent measurement quality, and temporal
scatter into separate major problems. Multiple epochs could be an appendix
only if they add a specific scientific point.

Would you prefer a compact regular paper or a letter-style result? Which
audience matters most: galaxy–halo connection, massive-galaxy assembly, or
statistical modeling?

## 3. May this paper have a distinct, low-z sample comparison?

The current repository rule requires the all-epoch sane-history sample for
every fit. Some of those cuts explicitly use stellar masses and SHMR
residuals. That was appropriate protection for growth-model development,
but could affect a measurement of population scatter and assembly information.

**Recommendation:** authorize a paper-specific comparison of a quality-
certified z=0.4 sample and the established strict sample, with criteria fixed
before fitting. Do not automatically discard plausible but unusual histories.
This is an explicit exception request, not a change already made.

If you prefer to retain the rule unchanged, we can scope the paper to that
selected population and report frozen-model sensitivities, but the broader
selection caveat must remain.

## 4. What finite stellar-mass definition should anchor the paper?

**Recommendation for discussion:** `M*(<100 kpc)` for the main observational
comparison, with the native outer anchor near 148 kpc as a sensitivity.
Use the same anchor for normalized shape and R50. Alternatively retain the
native anchor for continuity and report 100-kpc quantities alongside it.

Should the central claim be beyond **stellar mass**, or beyond **stellar
mass and size**? I recommend testing both and accepting the weaker conclusion
if size accounts for the signal.

## 5. How should we position this relative to Xu / Leidig and related work?

The Xu outskirt-mass paper and *Reaching for the Edge II* are particularly
close in motivation and measurement lineage. Are these the intended direct
precursors, and which data products/method descriptions should we cite or
reuse? Can you confirm the satellite-particle treatment of this exact drop,
and whether usable mock profiles or additional simulation samples are
available without a new data-production project?

This also determines whether observational robustness or cross-simulation
checks can be a modest addition rather than a separate undertaking.

## 6. What role should the emulator have in the paper?

**Recommendation:** the emulator is an empirical description and released
research tool for the TNG conditional relation. Its learned complexity and
simulation dependence are explicit. It is not the general physical forward
framework rejected for the learned corrections in Exp75/Exp77.

Do you agree that the first paper can establish the statistical relation
without completing the physical prescription? If so, no work on Exp79 or
Claude's current physical model is needed for this paper's main result.
