# Exp75 / Exp77 final closeout

Date: 2026-09-09. Both experiments are scientifically complete; no further
fits, data exclusions, or production correction-map integration are owed.
The final dispositions in their READMEs supersede earlier follow-up plans.

## What is retained

- Exp75: halo-predictable residuals diagnose shortcomings of the physical
  prescription. The statistical direct model remains a benchmark.
- Exp77: including annular masses in the objective improves the outskirts
  without materially changing cumulative-profile accuracy. This lesson does
  not require retaining the learned correction architecture.
- The correction PDF documents the mechanism and parameter accounting; its
  earlier frozen-integration suggestion is superseded by the user decision.

## What changes next

A compact physical prescription shared across epochs, tested with cumulative
and annular masses together. Do not confuse a frozen TNG300-trained population
mapping with a framework exposing interpretable parameters observations can
constrain. Preserve historical numerical gates rather than rewriting them.

## Preservation checks

The focused 22-test suite initially found one wall-clock-dependent synthetic
test after the overnight cutoff. The test now bypasses the deadline locally;
the science guard and its separate cutoff checks are unchanged. No scientific
driver, model, or saved prediction is altered by this test-only repair.

All non-cache gitignored files from the two experiment worktrees were checked
against `/Users/shuang/Dropbox/work/project/massive/hongshao_master`, copying
missing files only and refusing differing existing destinations. SHA-256 was
verified at source and destination after copying. Counts include duplicate
reference files present in both worktrees and are not unique-file totals.

| Source worktree | Files checked | SHA-256 of sorted path/hash inventory |
| --- | ---: | --- |
| hongshao_exp75_halo_residual_correction | 241 | 34d2861ab5c5a051b4050e5e6b7cf37cbc17a43fc8b5efd0ed912eb96a342e23 |
| hongshao_exp77_annular_correction_loss | 100 | c579e69dcd28ee7d0ed1d406c9218af29cd7b2f3a250ccf1c5fd2a65b6816f5b |

Excluded only disposable virtual environments and tool/Python caches. No
matching experiment processes were running. Exp75 had no uncommitted changes;
Exp77's only outstanding work was documentation from this conversation. All
earlier science commits were already ancestors of master before this closeout.

The final PDF is `output/pdf/exp77_model_and_profile_correction.pdf` on master;
science outputs and figures retain their experiment-relative paths. These
gitignored artifacts are preserved locally, not stored in Git. This closeout
does not authorize removal of older integration worktrees or other agents'
worktrees. Remove only the two named experiment worktrees after the final
documentation commit is merged and their working trees are clean.
