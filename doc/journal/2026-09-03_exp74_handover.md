# Handover — 2026-09-03, exp74 (C19: the model's past knew the halo's future)

Branch `exp74-c19-history-leak` from `master` `6bc2ecc` (exp73 merged the
evening before). Committed and pushed; **NOT merged** — the user's call.
Plan `doc/plans/2026-09-02-exp74-c19-history-leak.md`; everything in
`experiments/exp74_c19_history_leak/README.md`.

## What was done

- `HaloCurve` gained an anchor field `logt0` (default the official z = 0
  anchor; `engine.py` selftest passes unchanged).
- `history.py`: per galaxy per epoch a DiffMAH curve fitted to the catalog's
  running-peak M200c at snapshots ≤ the anchor, t ≥ 1 Gyr, anchored at the
  epoch with the anchor mass fixed. Tiers, never dropped. G1 and G2 pass;
  the pre-epoch curve predicts future growth at fixed mass with R² 0.000.
- `stage0_frozen.py`: at exp63's frozen θ the input swap zeroes the leak and
  costs +11–18 per cent mass at z = 2. 43 per cent of z = 2 deposits were on
  the official curve's extrapolation (its data cut is 2 Gyr).
- `stage1_refit.py` (per-start processes, `--merge`) and `stage1_eval.py`:
  the refit is exp63 with `a_z` 0.348 → 0.286, 12.6 per cent worse on exp63's
  loss, leak zero, z = 2 fitting tilt −0.122 → −0.046, z = 2 mh-complete total
  −9.7 → −2.5 per cent, R50 width 0.27 → 0.40 at z = 2; costs at z = 1–1.5
  (mh-complete +5–9 per cent) and in high-z sizes (+11 per cent at z = 2).

## Decisions owed

1. **Adopt the pre-epoch curves as the standard input.** My recommendation:
   yes — a model reading the future is not a model of growth. Every
   high-z number in the record was measured with the future in the input;
   the incumbent and the v1 stochastic layer inherit the leak and need
   re-baselining under the new input.
2. **Merge** `exp74-c19-history-leak`.
3. Variant B (the measured history, interpolated) — separates the cost of
   honesty from the cost of DiffMAH's functional form. ~half a day.

## Ops

- Five `predict2` calls per evaluation: ~2 s; a start is 40–100 min. Run
  starts as separate processes (0.4 GB each while fitting; 11.5 GB peak at
  build). Far starts (default, jitter0) rail into the FAIL penalty under
  exp63's objective; use a near start.
- **The machine slept for four hours mid-fit.** `caffeinate -i -w <pid>` on
  every long process, from the start.

## Appendix 2026-09-04/05 — variant B, and the reframing

The user reframed C19 (2026-09-04): for hongshao's application the halo's
full history from the simulation, final mass included, is legitimate input;
only the galaxy's own stellar mass is forbidden. So the official curve's
fault is a wrong conditional distribution (at fixed current halo mass the
model's stellar mass depends on the halo's future 4–40× more than TNG300's),
not a cheat. Variant B (`measured.py`): the measured running-peak M200c,
PCHIP-interpolated in log t, power law before 1.18 Gyr, one curve per galaxy
for every epoch; the engine reads a curve's own table functions
(`log_mah_table`, `dm_dlnt_table`). Gates pass (through the data exactly;
future R² within 0.01 of zero at snapshots and midpoints). Refit: a single
basin at 17.01 under exp63's loss (+10.8 per cent; pre-epoch DiffMAH +12.6);
the same model as the pre-epoch to a point or two everywhere; z = 2
mh-complete total −1.1 per cent (exp63 −9.7). **Recommendation changed: adopt
the MEASURED history as the standard input** (cheapest, best of the honest
inputs, needs no fit to build). Keep the future-dependence test as a QA gate.

Decisions owed: adopt; merge; re-baseline the incumbent and the v1 layer on
the new input. The next modelling targets under the honest input: the
z = 1.0–1.5 massive progenitors (+5–9 per cent) and the high-z sizes (+13 per
cent at z = 2).
