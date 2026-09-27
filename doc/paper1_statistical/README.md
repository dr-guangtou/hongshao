# Paper 1 planning: assembly information in massive-galaxy stellar profiles

Planning study requested September 25, 2026. This is not a new fitted
experiment, a production-model change, or the closeout of Exp79.

**September 28 launch update:** start with the
[ushmr1 execution plan](USHMR1_LAUNCH_PLAN.md) and
[sample provenance audit](SAMPLE_PROVENANCE.md). The new sibling repository
is approved for analysis, publication figures, and Markdown evidence notes;
LaTeX manuscript drafting is deferred. These notes update the operational
destination without changing the scientific comparisons below.

**Updated September 27:** [Agreed scope and representation strategy](DECISIONS_20260927.md)
records the user's decisions and the proposed measured/DiffMAH and
PCA/analytic-CoG comparisons. It supersedes the provisional September 26 scope.

## Recommendation

**Begin a focused paper, but consolidate the evidence before freezing its
headline numbers.** The existing record strongly supports an assembly-
dependent stellar-profile relation in TNG300. Evidence for additional
shape information beyond stellar mass and size is smaller and more
conditional; that distinction should refine the forward-prediction claim.

Agreed scope: massive centrals at z=0.4, a compact probabilistic forward
description, a native 148-kpc stellar-mass anchor, and epoch-relevant sample
criteria. No figure/page limit is imposed. A small reverse diagnostic is
optional, never a goal of accurate individual-history recovery. Independent
other-epoch results may enter an appendix; no joint evolution model is needed.

## Reading order and deliverables

1. [Paper outline](PAPER_OUTLINE.md): argument, section structure, suggested figure
   sequence, abstract skeleton, and how null results would change the claim.
2. [Evidence audit](EVIDENCE_AUDIT.md): useful experiments, verified archived
   measurements, code-level caveats, and publication readiness.
3. [Analysis plan](ANALYSIS_PLAN.md): matched models, mass/size controls,
   sample and mass-definition certification, null tests, uncertainty,
   reverse diagnostic, population QA, priorities, and stopping rules.
4. [Questions and resolutions](QUESTIONS.md): the original questions, now
   answered, and the remaining implementation choices. The Paper 1 sample
   exception was approved September 27.
5. [Literature positioning](LITERATURE_POSITIONING.md): closest precedents
   and a defensible novelty statement, with primary-source links.
6. [Artifact inventory](ARTIFACTS.md): private-copy provenance, checksums,
   and notes from inspecting nine existing figures.

The first practical step is to freeze exact epoch-relevant sample criteria
and certify one matched input/fold record. The highest-value scientific addition is
the controlled separation of stellar-mass amplitude, size, and finer
profile shape—not another search for a better fitting function.

## Isolation and provenance

- Codex branch: `codex-paper1-statistical-plan-20260925`.
- Private worktree: `/Users/shuang/Dropbox/work/project/massive/hongshao_codex_paper1_plan_20260925`.
- Frozen starting commit: `79bba59daf22447c0dfe9c564003bca278fa7812`.
- Claude's `/Users/shuang/Dropbox/work/project/massive/hongshao` checkout is
  protected. No branch switches, edits, code execution, or shared writable
  caches there. Read settled historical artifacts only to make verified private
  copies. The September 28 planning merge is user-authorized but awaits
  coordination with Claude, who retains integration ownership. Do not push.
- Exp79 remains preserved in its own worktree, scientifically unfinished.
- Numerical claims are historical measurements unless explicitly labeled as
  newly recomputed from archived arrays. No new physical or statistical fit is
  authorized by this planning exercise.

## Work plan

- [x] Read the scientific motivation and original analysis roadmap.
- [x] Audit the early MAH/aperture/PCA/emulator experiments and their code.
- [x] Audit later shape-information tests, sample revisions, and data geometry.
- [x] Verify representative numerical artifacts and inspect relevant old figures.
- [x] Check primary literature for novelty, precedents, and overclaim risks.
- [x] Write an evidence inventory separating results, interpretations, and gaps.
- [x] Draft a short-paper narrative, outline, and figure sequence.
- [x] Specify prioritized confirmatory analyses, controls, and decision rules.
- [x] List user decisions needed before implementation and complete a review.

## Review (September 26)

The early statistical experiments and the later Exp62 scale-free analysis
provide a coherent paper foundation. They do not yet share one sample,
mass definition, training-only preprocessing chain, and comparison baseline.
Important older interpretations need qualification: mass normalization is
not mass conditioning, PCA is not inherently non-portable, and a prediction
plateau does not prove an intrinsic information ceiling. Related literature
already establishes stellar-outskirt/assembly connections; the contribution
should be the incremental, radius-resolved information accounting.

No new fits or figures were generated. Archived scores were checked as
historical outputs, not reproduced. Only this private planning checkout and
its private artifact snapshot were written. Claude's checkout and Exp79
were not modified, and no merge, push, or cleanup was performed. Remaining
work is user scope decisions and subsequently authorized implementation.

## September 27 review

The user confirmed a TNG300-centered, forward-prediction paper, the native
148-kpc aperture anchor, a low-z-relevant sample, and Xu's measurement lineage.
Inverse recovery is not the project goal. Proposed representation tests use
both measured and DiffMAH histories and compare direct/PCA profiles with one
existing analytic CoG candidate. An N-body application is a possible generation
demonstration, not independent stellar validation. Only planning documents
were updated; no new fits, data queries, figures, or shared-worktree changes.
