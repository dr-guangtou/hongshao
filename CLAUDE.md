# hongshao — repo rules for every agent session

Read `doc/lessons.md` and the latest `doc/journal/*handover*.md` at the start
of a session. The user's global mandates (`~/.claude/CLAUDE.md`) apply.

## THE FITTING SAMPLE (the user, 2026-08-30 — applies to every experiment)

**Every fit uses all the galaxies selected at z=0.4 whose halo history and
stellar-mass history are sane, at every epoch. Halo-mass completeness is NOT a
fitting criterion.** The mh-complete subset (the progenitors above the
per-epoch completeness cut) is an *after-fit* reporting check, scored with the
fitted parameters frozen — never the sample a fit is run on.

- The rule is implemented once, in
  `experiments/exp54_unpinned_amplitude/selection.py::fitting_sample_mask`:
  finite positive CoGs and a finite DiffMAH halo mass at all five epochs; the
  3 dex backward rule (`sane_history_mask`); and no stellar-history outlier at
  any epoch (`stellar_history_flags`: more than 0.5 dex off the population's
  running-median M*(<100 kpc)–Mh relation at that epoch, a > 1 dex jump of
  M*(<100) between adjacent epochs, or a > 0.3 dex drop of M*(<30)). One
  flagged epoch removes the galaxy from every epoch.
- Every fitting script prints the sample it used, with the counts of what
  each criterion removed, in its log.
- History: exp54 designed the mh-complete sample as an after-fit re-score
  (`selection.py`, section 3), but `stage37_size_epoch.build` handed that
  subset to later experiments as the fit mask, and exp57/exp63 fitted on it
  until 2026-08-30. When you see `S0.build(...)`'s `mask` used as a fit mask,
  that is the old, wrong path.

## THE HALO INPUT (the user, 2026-09-04/05 — applies to every experiment)

**The model may know the halo's full assembly history from the simulation,
including its peak and final halo mass. It must never see the galaxy's own
stellar mass at any epoch.** The forward model has to predict the stellar
mass and the stellar-mass–halo-mass relation, not normalise to them.

- The halo history reaches the engine through one interface,
  `experiments/exp74_c19_history_leak/measured.py::build_input(recs, kind)`
  with `kind` in `official` (DiffMAH curve), `pre-epoch` (DiffMAH fitted
  before each epoch) and `measured` (the merger-tree masses interpolated).
  **Keep all three working**; the recommended default for the application
  is `measured` (exp74 verdict).
- Standing QA gate: at fixed halo mass at an epoch the model's stellar mass
  must not depend on the halo's future growth more than TNG300's does
  (`selection.partial_growth` on the residual; exp74 `stage1_eval.py` §4).
  The official DiffMAH curve fails it 4–40× (C19); do not read a high-z
  number fitted on it without saying so.
- **THE BASELINE MEAN (the user, 2026-09-13, after exp82)**: the deposition
  delay (tau_d held at 0.15 Hubble times at accretion, step arrival) plus
  the expansion exponent q_e on exp63's twelve, fitted under the standard
  objective on the measured history — `exp74/rebaseline.py::adopted_mean()`
  (engine `exp80/size_law.py::predict_law`; 14 parameters, 13 fitted;
  file `exp82/outputs/stage1_fit_delay0.15_fix-tau_d0.15_start_tuned.npz`).
  Chosen on a tau_d grid BY THE GATES (offset 14 of 15, one width pass, the
  centre +0.5 / −6.2 / −6.9 per cent; loss 12.70): the loss's own minimum
  (tau_d 0.20, the wide-split family, 12.15) narrows every size
  distribution and is not the baseline. The previous baseline (exp74's
  optimum, 15.56) stays as `adopted_baseline()` for every comparison.
  Known costs to carry: the mh-complete progenitors' 50–100 kpc shell at
  z ≥ 1.5 (−12 / −18 per cent), the future-dependence gate +0.03 dex per
  dex at z = 0.7 and z = 2, the z = 0.4 early-mass residual (exp83).
  **The gates decide, the loss does not.**

- **THE STOCHASTIC LAYER (the user, 2026-09-23, after exp84)**: hongshao's
  v2 layer is exp84's two-component layer — a cross-epoch-correlated total-mass
  draw plus a per-galaxy, PER-EPOCH draw of the extended deposit size with the
  anatomy's fitted cross-epoch correlation (the compact-size draw calibrated
  to zero) — around `adopted_mean()`: `exp74/rebaseline.py::adopted_layer()`
  (artifact `exp84/outputs/hongshao_v2_layer.npz`, draws through
  `exp84/stage3_adopt.draw_cogs`, engine hook `size_law.predict_law(size_dev=)`).
  Validated held out: widths 12/15 (M*) and 15/15 (Mh), offsets 15/15, S5
  exact, persistence matched. Costs carried: the drawn median profile shifts
  0.012 / 0.031 dex (C27); inner sizes at z <= 1 over-dispersed 1.2x; the
  decline statistics unchanged. Score any layer with `qa.evaluate_draws` on
  the DRAWS, both conditionings, offset and width separately. The v1 layer
  (exp60) is superseded.

## FIGURES (the user, 2026-09-23)

**An experiment's figures go in its own `experiments/expNN_*/figures/qa/`,
never in the repo-level `figures/qa/`** (which holds only the standard
battery of the adopted models). Scripts set `FIGDIR = HERE / "figures/qa"`.

## WORKING RULES FOR TWO AGENTS (the user, 2026-09-09 — replaces every earlier integration note)

- **Claude** uses the main `hongshao` directory, starting each new experiment
  on a feature branch from the current `master`.
- **Codex** uses a separate worktree and feature branch for every experiment;
  it never switches branches or edits files in Claude's directory.
- **Handoff**: Codex commits and verifies its work, then provides the branch
  name and the artifact locations. Claude merges it into `master` and
  preserves any gitignored outputs (copy and hash-verify).
- **Cleanup**: Codex removes its worktree only after confirming that its
  commits and artifacts have been preserved.
- **Branch and experiment ids must still be checked before creation**
  (`git worktree list`, `git branch -a`, `ls experiments`), because the
  worktrees share one Git repository. Taken: exp75, exp77, exp79 (Codex;
  exp79 = the deposition reach boundary, in `hongshao_exp79_deposition_reach`),
  exp76, exp78 (Claude; the size-aware objective, done 2026-09-09), exp80
  (Claude; the deposit size law, branch `exp80-deposit-size-law`,
  2026-09-09/10), exp81 (Claude; the evidence-based rethink, 2026-09-12),
  exp82 (Claude; the delay + expansion exponent, branch
  `exp82-delay-expansion`), exp83 (Claude; the early-mass term, closed and
  not adopted 2026-09-22), exp84 (Claude; the layer re-baselined, branch
  `exp84-layer-rebaseline`, 2026-09-22), exp85 (Claude; the compact
  channel's post-deposition expansion, the centre's mechanism, branch
  `exp85-compact-expansion`, 2026-09-23). exp86 is the next free id; check
  before creating.
- `doc/lessons.md`, `doc/todo.md`, `doc/open_questions.md` are append-only
  and conflict when both agents append; resolve by keeping both sides.

This separates the two agents' working files and gives `master` a single
owner, which addresses both sources of collision seen on 2026-09-08.
