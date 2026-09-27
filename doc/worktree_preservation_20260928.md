# Worktree record preservation — September 28, 2026

## Status

The useful records from the three requested worktrees have been preserved
in a verified sibling archive and a combined integration branch. **This is
not yet a merge into master.** Claude retains integration ownership under
the agreed two-agent rule. No worktree, branch, or source file was deleted;
no push, new scientific analysis, or figure generation was performed.

Integration branch: `codex-records-preservation-20260928`.

Integration worktree:
`/Users/shuang/Dropbox/work/project/massive/hongshao_codex_records_preservation_20260928`.

## Committed records

- Paper planning: `00d4cbc` and `d47302d`, preserved through a merge commit.
  Main entry points are `doc/paper1_statistical/README.md`,
  `USHMR1_LAUNCH_PLAN.md` in that directory, and
  `doc/journal/2026-09-28_handover.md`.
- Direction review: `bc80018`, preserved through merge `b7716c8`.
  `doc/next_direction_review_20260909.md` is a historical proposal, not a
  new instruction to restart an experiment. Both sets of AGENTS.md additions
  were retained when resolving their overlapping insertions.
- Internal-review branch tip: `7217811` was already an ancestor of master.
  Its substantial local-only notes and presentation work were not in that
  commit; they are preserved privately below, not added to the public repo.

## Verified private archive

Root:
`/Users/shuang/Dropbox/work/project/massive/hongshao_records_archive_20260928`.

Each complete filesystem snapshot is under `snapshots/<original_worktree_name>/`.
Only the root `.git` worktree pointer is excluded. Tracked files, untracked
files, ignored files, small caches, hidden files, and symlinks are retained.
The archive's `manifest.json` records SHA-256, bytes, file mode, source HEAD,
branch, status, untracked/ignored lists, and symlink targets.

| Original worktree | Verified regular files | Bytes in regular files | Special considerations |
|---|---:|---:|---|
| `hongshao_codex_paper1_plan_20260925` | 713 | 21,489,918 | Includes all 22 private paper-audit artifacts |
| `hongshao_codex_direction_review_20260909` | 615 | 17,127,402 | No untracked or ignored entries at preservation |
| `hongshao_internal_review_2026_08_16` | 423 | 67,045,777 | Includes reviews, presentation sources, figures, animations, slides, and two external dependency symlinks |

The source was inventoried before and after copying, and the copied
inventory was checked against both. Every regular-file hash and symlink
target matched. Snapshots are plain folders, not active Git worktrees.

The two presentation dependency links point into the existing Codex runtime
cache, not to a worktree being removed. Their targets were not vendored.
Generated presentations and sources are preserved; a future rebuild still
needs compatible dependencies and the original scientific input paths.
The archive certifies preservation, not fresh reproduction of the old science.

Particularly useful archived locations:

- `snapshots/hongshao_codex_paper1_plan_20260925/doc/paper1_statistical/`
- `snapshots/hongshao_codex_paper1_plan_20260925/data/processed/paper1_audit_snapshot_20260925/`
- `snapshots/hongshao_internal_review_2026_08_16/docs/internal/`
- `snapshots/hongshao_internal_review_2026_08_16/output/`
- `snapshots/hongshao_internal_review_2026_08_16/presentation/`

The archive also carries a Git bundle of the preserved branches and a
private README with verification/restore instructions. Keep it private:
an ignored internal review is not automatically suitable for public release.
The archive is a verified local copy; remote/cloud backup completion has not
been independently established.

## Owner-coordinated completion on master

Claude should first verify its own checkout is idle/clean and inspect the
preservation branch. Only the integration owner should run the merge:

```sh
git status --short --branch
git log --oneline master..codex-records-preservation-20260928
git diff --stat master...codex-records-preservation-20260928
git merge --no-ff codex-records-preservation-20260928
```

Do not publish the private archive or push without separate authorization.
If master advanced, resolve changes while retaining both agents' contributions;
do not overwrite the shared checkout to force the merge.

## Deletion gate

The three original worktrees are recoverable from the archive and retained
Git branches. For the user's requested **preserved back to master** completion,
wait until the coordinated merge is done. Before actual deletion, recheck
source status and file inventories against the archive: new work after this
snapshot invalidates deletion approval. Check no process is still using those
directories, and keep the source branches until integration is confirmed.

Use Git's worktree removal, not recursive filesystem deletion. The paper and
internal-review trees contain ignored/untracked files, so a forced removal
must only follow an explicit, fresh archive verification. Do not prune other
worktrees or delete Exp79. The new integration worktree is a separate cleanup
item after master integration; the sibling archive must remain.

For next-session paper work, the archived handover and planning package no
longer depend on keeping the original planning worktree. Copy the curated
records and verified evidence to the future `ushmr1` repository from here.
