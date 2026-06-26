# Milestone Review — 2026-06

A checkpoint after graduating the emulator: every planned Direction (A–I), Phase
(1–5), and core question from `ultimate_shmr_context.md` /
`ultimate_shmr_possible_directions.md`, mapped against the 23 experiments + the
graduated library, with the genuine gaps ranked.

## Where we stand

The library is **feature- and target-complete for this dataset**: one
heteroscedastic core (`hongshao.emulator`) drives four prediction modes
(`hongshao.profile_emulator`: kpc apertures, Re apertures, CoG, density profile),
with a 5-knob deformation layer (`hongshao.forward`) for external inference.
Working model = **linear mean on `DiffMAH(4) + c_200c` + heteroscedastic full
covariance**, used generatively.

## Coverage of the plan

| Plan element | Status | Experiments |
|---|---|---|
| **Dir A** — aperture ↔ epoch mapping | ✅ | exp01 |
| **Dir B** — DiffMAH vs raw MAH | ✅ DiffMAH ≈ raw at the ceiling; portable | exp06, exp10, exp11, exp13 |
| **Dir C** — radial-DiffMAH profile | ✅ | exp03, exp05, exp24 |
| **Dir D** — PCA / low-rank CoG | ✅ | exp02, exp22, exp22b |
| **Dir E** — two-component profile | ◑ a 2nd component isn't justified at z=0.4; 2-Sérsic degenerate | exp07, (2-Sérsic prototype) |
| **Dir F** — profile–MAH correlation map | ✅ | exp01 |
| **Dir G** — null / shuffle controls | ✅ used throughout | exp01, exp13, exp16, exp18 |
| **Dir H** — probabilistic emulator | ✅ **exceeded** (heteroscedastic + generative + 4 modes + deformer) | exp08–19 → graduate |
| **Dir I** — redshift evolution | ✗ gated (no multi-z profiles) | — |
| **Phases 1–4** | ✅ all | exp01–19 |
| **Phase 5** — mock painting on N-body | ✗ gated (portability) | — |
| **Context Q1–Q6** | ✅ all answered | — |
| **Context Q7** — evolve across redshift | ✗ gated | — |

Six of nine directions and all four early phases are comprehensively closed; the
emulator went well past the planned "minimal Gaussian" into a generative,
deformable library.

## Genuine gaps, ranked

### Data-available now — essentially exhausted
The secondary-property axis is closed: exp16–18 tested `c_200c`, `acc_rate`, and
3D shape and found `DiffMAH + c_200c` is the ceiling. The only untested field in
the drop is `v_sigma_3d` — a quick check, low expected value (other shape /
kinematic properties were null).

### Data-gated — ranked by scientific value
1. **★ Merger-tree granularity + true ex-situ fraction.** *The pivotal stone.*
   We keep concluding "the outskirt residual is **intrinsic**" (exp13/21/22), but
   that floor has only ever been tested against main-branch MAH + the secondaries
   in our drop. The planning docs anticipated this exact fork (§14: *"if MAH
   weakly predicts residuals → add number of major mergers, largest progenitor
   mass ratio, ex-situ fraction"*), and the two-phase picture (context §2)
   predicts **outskirts = accreted stars** — so ex-situ fraction / merger
   statistics are the natural untested predictor of the scatter we can't explain.
   This converts "intrinsic" from a *conclusion* back into a *testable
   hypothesis*. Needs TNG merger trees + in-situ/ex-situ particle tagging (not in
   the processed drop). **Parked** — reliable extraction will take time.
2. **Redshift evolution** (Dir I, Q7, Phase 5). The long-term dream; architecture
   designed (context §7, §10) but not calibratable without multi-z profiles.
3. **Mock profile painting on an N-body catalog** (Phase 5 / portability). The
   emulator is ready; gated on an external sim with `c_200c` + galaxies (or the
   parked DiffMAH cross-match).
4. **100–300 kpc / ICL outskirts** (§1.1). Data caps at ~150 kpc; the far-ICL
   regime is untouched.

### Deliberately skipped / subsumed (defensible)
- **Dir E full in-situ/ex-situ decomposition** — partial; the validation against
  *true* ex-situ folds into gap #1.
- **3D deprojection latent layer** (context §8.2) — we correctly followed the
  doc's "2D-first" recommendation.
- **Physics-based deposition-kernel forward model** (Lackner-Ostriker §6,
  El-Badry §7, integral growth §10.2) — we went data-driven. *Being explored as a
  standalone toy in exp25* (a smooth-Gaussian deposition model, MAH → profile),
  independent of the redshift dream.

## One-line synthesis

> Everything answerable with **main-branch MAH + halo structure** is answered,
> and the emulator is built. The single most valuable unturned stone is the
> **ex-situ fraction / merger-tree lever** — the one variable the two-phase
> picture says should drive the outskirt scatter, the one the planning docs
> flagged as the next step, and the one we've never tested because it isn't in
> the data drop yet.
