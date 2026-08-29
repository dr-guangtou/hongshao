# Session Handover — 2026-08-28 (exp63 Stage 5, the single-epoch round)

*(Follows `2026-08-28_handover.md`, which carries the user's direction this session
executed: z=0.4 only, judged by the binned average CoGs.)*

## Goal

Improve the exp63 two-channel mean at z=0.4 ONLY, judged first by the average
curves of growth in halo- and stellar-mass terciles and the planes; targets: the
top halo-mass tercile (Stage 2: −12 % at 2–4 kpc) and the outskirts. First step:
refit with the time exponents frozen; then the levers the user named (halo-mass-
dependent extended size, the split's form, the extended kernel, an outskirt-aware
objective), each judged by the binned CoGs. No z > 0.4 number until z=0.4 is right.

## Completed This Session (branch `exp63-analytic-growth`, commit `575b2bf` + the
## final doc commit; NOT pushed, nothing merged)

- **Tools.** `stage2_fit.py` gained `--freeze name=val,...`, `--levers g_c,g_e,w_min`,
  `--extended-family sersic`, `--objective outer|binned`, `--compact-kpc`, `--tag`;
  `model2.py` gained the levers (nest at zero; self-check (9)), a Sérsic extended
  kernel, and `Spec2(compact_in_kpc=True)`. New `single_epoch.py`: `--gate TAGS`
  (the z=0.4 visual gate: tercile-median residual table at 7 radii raw + pinned at
  148 kpc, planes, D4 loss components, one overlay figure
  `figures/exp63_single_epoch_gate{name}.png`) and `--sweep` (levers by evaluation).
- **Result (README Stage 5a–5c).** (i) Freezing the exponents (a_z +0.505,
  a_Mz +0.318, b_e −0.896, b_c 0) costs the top tercile: rms over terciles×radii
  4.52 vs Stage 2's 3.89. (ii) The production per-galaxy loss is nearly blind to
  the binned offset the gate reads (per-galaxy scatter 0.18 dex vs offset 0.065 dex
  at 2–5 kpc in the top tercile); `w_min` improves the gate by evaluation but is
  railed at 0 by the fit. (iii) `--objective binned` (rms of the tercile-median
  residual, 1 at the null) reads the gate: rms 3.16 for 1 % of the D4 loss; but
  every product kept the same residual: 2 kpc high vs 3.7–4.9 kpc. (iv) **The
  structural lever: the compact size must not scale with the halo mass at the
  deposit.** `--compact-kpc` under the binned objective: every halo-mass tercile
  within ±3 % at every radius (rms 1.50; top tercile 2–4 kpc −2 %); the size levers
  on the R200c form choose g_c = −0.45 (≈ mass-free compact size), g_e = +0.08
  (extended size stays ∝ R200c — no extra halo-mass dependence needed), w_min 0.12:
  rms 1.16, plane slope 1.44 (truth 1.42), D4 loss 2.796 (best of any frozen fit;
  Stage 2 with free exponents 2.764). (v) The extended kernel's family and the
  outskirt term do nothing at the gate; the ±10–20 % at 148 kpc by STELLAR-mass
  tercile is regression to the mean of a scatter-free mean (every mean model,
  incumbent included) — the stochastic layer's job.
- **Degeneracy stated.** The three best products (rms 1.16–1.50) hold 17, 33 and
  44 % of the stars in the compact channel (2.6–4 kpc, n 0.9–1.9, deposited at
  median z≈2–2.6) against the extended kernel's core; one epoch cannot separate
  them; the earlier epochs can — that is what the extrapolation must be asked
  first, after the user has read the z=0.4 figures.
- Records: README Stage 5 (+ closing summary amended), `doc/todo.md`,
  `doc/lessons.md` (four lessons), memory `loss-is-blind-to-the-binned-gate`.

## Added 2026-08-29 (after the user read the z=0.4 QA figures)

- `single_epoch.py --qa TAGS [--epoch k]`: the standard `qa.evaluate` battery at
  ONE epoch (single-epoch arrays); `--gate --epoch k` judges at any epoch, on the
  whole sample AND on the fit mask (D5). `stage2_fit.py --epoch k` fits one epoch
  alone (its data, mask, sigma_A, terciles; integral truncated at its anchor).
- **5d extrapolation** of Options 2 and 1 (README 5d): both beat the five-epoch
  incumbent at M(<10) at every epoch; Option 2's outskirts collapse at z>=1.5
  (the sharp switch + frozen b_e = -0.9), Option 1 keeps outskirts and plane
  slopes but tilts at the centre (C8 span 0.087).
- **5e per-epoch fits** (README 5e, `figures/exp63_single_epoch_gate_e{1..4}.png`,
  `exp63_single_epoch_params_vs_epoch.png`): the kpc form fits EVERY epoch to
  gate rms 1.05-1.50 on its fit mask (incumbent 5-10); parameters move smoothly
  and read as a time law: b_c ~ -0.5 (compact size 2.64 -> 1.75 kpc toward z=2),
  b_e ~ +0.1 (extended size a constant fraction of R200c at deposit — the frozen
  -0.9 is wrong by +1.0), the split flattens at z>=1.5, small drift in a0/a_M.
  Option 2 converges to the same structure at every epoch (g_c -0.32..-0.40).
  Residual the class keeps: the plane slope at z>=1.5 too steep even when fitted
  there (1.40 vs 1.00 at z=2).
- **5f joint fit, run on the user's word** (`--joint`; README 5f;
  `figures/exp63_single_epoch_joint_summary.png`): one theta with b_c, b_e free
  gate rms 3.44/2.66/3.90/4.03/10.42; with all four time exponents free
  3.33/2.04/3.77/3.72/7.59 (incumbent 4.98/6.54/7.67/5.60/10.48; per-epoch
  1.50/1.05/1.48/1.17/1.36). Better than the five-epoch incumbent at every
  epoch by the gate and the battery (M(<10), tier 3, fit-mask outskirts; C8
  whole-population span 0.013 vs 0.058); not at the per-epoch ceiling — the
  exponents carry ~40% of the gap; z=2 fails in AMPLITUDE (-7..-11% at every
  radius on the fit mask). Joint starts rail n_c at 0.5; the four-exponent
  fits are a flat valley (same loss, different a_z/b_e). Fit files
  `stage2_fit_joint_kpc{,_free}.npz` (merged from `_s0.._s2`), batteries run.
- **Owed by the user**: adopt `_joint_kpc_free` as the exp63 five-epoch mean?
  pursue the efficiency law's time dependence / epoch-dependent n_c and split?
- Fit files added: `stage2_fit_frozen_binned_{kpc,levers}_e{1,2,3,4}.npz`; eval
  logs `stage2_eval_frozen_binned_{levers,kpc}.log` (+ their figures).

## In Progress (Not Finished)

- Nothing running. `_frozen_binned_kpc_levers` finished: rms 1.41, plane slope
  1.42 (= truth), D4 2.811, g_c −0.07 / w_min 0.02 (no mass dependence or floor
  needed once the compact size is in kpc) — README 5c table, last row.
- Fit files (gitignored, regenerable ~10–20 min each with 3 starts):
  `outputs/stage2_fit_frozen{,B,_sersic,_outer,_binned,_sersic_binned,_kpc,_binned_kpc,_binned_levers,_sersic_binned_levers,_binned_kpc_levers}.npz`;
  gate logs `outputs/single_epoch_gate_*.log`; sweep `outputs/single_epoch_sweep.log`.

## Problems / Blockers

- Decisions owed by the user: which z=0.4 candidate to carry (`_frozen_binned_levers`
  is the recommendation: best gate rms, best D4 loss among frozen fits, plane slope
  on the truth); whether the binned objective term is acceptable as part of the
  fit (its starts spread 3.29–3.48 on the R200c form — a median is piecewise-smooth;
  a mean-log proxy would be smooth); merge/push; P10; P9.
- `stage2_fit.py --eval-only` (the full extrapolation + battery) has NOT been run
  on any Stage 5 fit, by the user's rule. It works unchanged on the new fit files
  (`spec_from_fit` rebuilds the spec) — one command per tag, ~5 min each.
- The `default` start fails at once for kpc + levers (the default theta is outside
  the region where the model evaluates); the nested and jitter starts are fine.

## Key Decisions

- b_c frozen at 0 (the incumbent has no compact channel; 0 = a fixed fraction of
  R200c at the deposit) — stated as an assumption in README 5.
- The gate's amplitude pin is the last radius (148 kpc), exactly as `qa_bins`.
- The D4 loss column in every gate table is the production loss whatever
  objective the fit used, so products compare.

## Branch State

- `exp63-analytic-growth`, ahead of `master` by 29+ commits, never pushed. Working
  tree: clean after the final doc commit. Background jobs: none after
  `_frozen_binned_kpc_levers` finishes (check `ps aux | grep stage2_fit`).

## Files Modified This Session

Modified: `experiments/exp63_analytic_growth/{stage2_fit,model2}.py`, its
`README.md`, `doc/todo.md`, `doc/lessons.md`. New:
`experiments/exp63_analytic_growth/single_epoch.py`, this file. Memory: one new
file + index line.
