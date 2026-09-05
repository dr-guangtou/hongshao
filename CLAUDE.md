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
