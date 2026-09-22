# exp85 — the centre's mechanism: an age-driven expansion of the compact channel (2026-09-23)

Branch `exp85-compact-expansion` (from master `aae792d`). Plan
`doc/plans/2026-09-23-exp85-compact-expansion.md`. The model under change
is THE ADOPTED MEAN (exp82, `rebaseline.adopted_mean()`: the delay
tau_d = 0.15 held + q_e = 0.152 on exp63's twelve; 14 parameters, 13
fitted). Stage 0 is a frozen-theta probe; Stage 1 (a fit) was to run only
on a probe that passes the gate. **No probe passed; Stage 1 was not run.
The adopted mean stands. Figure: `figures/qa/exp85_probe_summary.png`.**

## In plain language

**The goal.** In TNG300, 42 per cent of our galaxies lose stellar mass in
their inner 4.9 kpc between z = 2 and z = 0.4 (by 0.066 dex in the
median), while the rest gain 0.160 dex there. The adopted model gives
both groups nearly the same central growth (+0.039 and +0.102), so it is
too heavy in the centres of the galaxies that declined and too light in
the others. exp83 showed that this "decline" cannot be produced by any
term that changes how much stellar mass a halo deposits, because it is a
matter of WHERE the old stars sit, not how many there are. The obvious
mechanism, and the one the user named in August, is that a galaxy's
compact, early-formed stars puff up with age (black-hole feedback), so an
old centre is more extended at z = 0.4 than it was at z = 2.

**What was tried.** The model's compact deposits were allowed to expand
after they form, by a factor (t_obs / t')^q_c that grows with the age of
the deposit (the same idea as the adopted model's q_e, which expands the
extended deposits with the halo). A halo-driven form (the expansion tied
to the halo's radius growth) was run as a control. At each strength the
compact-size constants were re-tuned so the population's typical central
concentration at z = 0.4 and z = 2 stayed what it is in the adopted
model, so that the probe reads only how the mass is REDISTRIBUTED
between galaxies. Nine strengths, no fit.

**What happened.** Nothing. The gap between the declining and the
non-declining galaxies' central change stays at 0.054–0.066 dex at every
strength, against TNG's 0.227 (figure, panel a). The knob does lower the
central mass, but it lowers it slightly MORE in the galaxies whose
centres did not decline (figure, panel c: the per-galaxy response has
the wrong sign, rank correlation −0.33 with TNG's change).

**Why.** The anatomy of the model's centre (panel b) says why, and it
could have been read before the probe: in the adopted model only 27 per
cent of the 4.9 kpc mass of a declining galaxy comes from the compact
(in-situ) channel, 33 per cent for the others; the rest is the inner
part of the extended (accreted) channel. An expansion of the compact
stars therefore acts in proportion to a share that is SMALLER in the
galaxies that need it. The decliners' compact stars are older (0.92 vs
0.85 dex in log t_obs / t'), as the mechanism assumes, but the age lever
is a tenth the size of the share lever, and the compact deposits are only
2.2 kpc across, so a moderate expansion is barely seen by a 4.9 kpc
aperture at all.

**What it means.** The centre's mechanism is not in the compact channel
of this model. The model puts most of a massive galaxy's central mass in
the accreted channel, so any mechanism that must act on the decliners'
centres would have to act on THEIR early-accreted deposits — or the
model's split of the centre between in-situ and accreted stars is itself
the thing to question (open question C28). Either is a model change
beyond one knob.

## Stage 0, part 1 — the probe (`outputs/stage0_probe.log`)

The knob: `size_law` knobs `q_c` (s_c × (t_obs / t')^q_c, age driven) and
`q_ch` (s_c × (R200c(t_obs) / R200c(t'))^q_ch, halo driven), both nesting
at zero; the compact kernel is evaluated per epoch when either is on;
exp84's `size_dev` composes with it. `selfcheck` and
`selfcheck_compact_expansion` pass (the default law still reproduces
`model2.predict2` bit for bit; q_c = 0.5 moves M(<5) by −0.10 dex at
z = 0.4 and −0.04 at z = 2, the grid-end total by 0.000).

At each probe value (log_f_c, b_c) were re-tuned by a damped Newton so
the fitting sample's median log M(<4.9)/M(<103) at z = 0.4 AND z = 2 is
the adopted mean's, then a0 re-centred on the z = 0.4 median M(<103).
Gates (the plan): G1 the split of the central change between the two
groups at least half way from the adopted 0.063 to the truth's 0.227,
with both groups' mismatches reduced; G2 the concentration part of the
z = 0.4 early-mass residual (5 kpc at fixed Mh and fixed M(<103)) halved
from rho +0.14; G3 the non-decliners' z = 2 centre median within 0.02 of
+0.051.

| probe | re-tune: log_f_c, b_c (adopted +1.020, −0.863); miss | decliners' change (truth −0.066) | non-decl. change (truth +0.160) | split (truth 0.227) | G2 rho z=0.4 (≤ +0.07) | M(<2) % z=0.4 / 2 | size offsets ≤ 0.05 of 15 | loss (Δ) | gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| adopted mean | — | +0.039 | +0.102 | 0.063 | +0.14 | +0.5 / −6.9 | 10 | 12.75 | — |
| q_c 0.1 | +0.484, −0.166; 0.002 | +0.034 | +0.099 | 0.065 | +0.09 | −6.6 / −10.2 | 7 | +0.12 | fail (G1, G2) |
| q_c 0.2 | −0.200, +0.667; 0.009 | +0.028 | +0.095 | 0.066 | +0.04 | −3.8 / −7.8 | 7 | +0.25 | fail (G1) |
| q_c 0.3 | +1.425, −1.735; 0.009 | +0.027 | +0.081 | 0.054 | +0.18 | −0.3 / +4.9 | 11 | +0.21 | fail (G1, G2) |
| q_c 0.5 | −0.500 (rail), +0.942; 0.038 | +0.009 | +0.075 | 0.066 | +0.03 | −11.8 / −8.4 | 5 | +0.78 | fail (G1) |
| q_c 0.8 | −0.500 (rail), +1.186; 0.106 | +0.005 | +0.064 | 0.060 | +0.15 | −26.6 / −23.7 | 0 | +5.03 | fail |
| q_ch 0.1 | +0.597, −0.354; 0.002 | +0.034 | +0.097 | 0.063 | +0.12 | −7.1 / −9.5 | 7 | +0.13 | fail (G1, G2) |
| q_ch 0.2 | −0.500 (rail), +0.947; 0.005 | +0.031 | +0.097 | 0.066 | +0.04 | +2.7 / −3.1 | 7 | +0.53 | fail (G1) |
| q_ch 0.3 | −0.500 (rail), +0.921; 0.019 | +0.024 | +0.088 | 0.064 | +0.05 | −4.3 / −6.4 | 7 | +0.53 | fail (G1) |
| q_ch 0.5 | −0.500 (rail), +1.151; 0.082 | +0.018 | +0.078 | 0.060 | +0.14 | −22.7 / −22.1 | 2 | +3.69 | fail |

Read:

1. **G1 never moves.** The split stays within ±0.01 of the adopted 0.063
   at every strength of either form; both groups' central change drifts
   down together as the re-tune loses its grip (the miss grows past
   0.02 dex from q_c 0.5 / q_ch 0.3, where log_f_c rails at its lower
   bound). The age-driven and the halo-driven forms are indistinguishable
   here — the halo form does not rescue the reading, so the failure is
   not about which clock drives the expansion.
2. **G2's passes are the re-tune, not the mechanism.** Where the z = 0.4
   concentration correlation halves (q_c 0.2 / 0.5, q_ch 0.2 / 0.3), the
   re-tune has shrunk the compact deposits to 0.03–0.06 of the adopted
   size with b_c flipped positive — a different compact channel, at a
   loss cost of +0.25 to +0.78 and with the size offsets losing 3–5 of
   their 15 cells. The two-target re-tune of two nearly collinear
   constants has a flat valley: neighbouring probe values land at
   log_f_c −0.2 and +1.4. Read the re-tune's landing point before
   crediting a sub-gate.
3. **The sign is wrong at frozen constants** (section 1 of the log, before
   any re-tune): q_c 0.1 lowers M(<5) at z = 0.4 by 0.009 dex in the
   decliners and 0.013 in the non-decliners; q_ch 0.1 by 0.012 vs 0.017.

## Stage 0, part 2 — the anatomy of the centre (`outputs/stage0_anatomy.log`)

Per galaxy from the adopted model's deposits, inside 4.9 kpc (medians,
decliners | non-decliners):

| reading | z = 0.4 | z = 2 |
| --- | --- | --- |
| compact share of M(<4.9), M_c / (M_c + M_e) | 0.27 \| 0.33 | 0.30 \| 0.41 |
| log M_c(<4.9) [Msun] | 10.30 \| 10.25 | 10.29 \| 10.24 |
| log M_e(<4.9) [Msun] | 10.71 \| 10.56 | 10.66 \| 10.39 |
| mean log10(t_obs / t') of the compact stars inside 4.9 kpc | 0.92 \| 0.85 | 0.48 \| 0.41 |
| fraction of that compact mass deposited before 2 Gyr | 0.92 \| 0.82 | 0.94 \| 0.86 |
| compact deposit size inside 4.9 kpc, mass-weighted median [kpc] | 2.21 \| 2.41 | 2.19 \| 2.38 |
| the model's response of log M(<4.9) to q_c = 0.3 at frozen constants [dex] | −0.041 \| −0.050 | −0.016 \| −0.022 |

The response's rank correlation is −0.99 with the compact share, +0.73
with the inside age, −0.42 with TNG's central change (the wrong sign);
the differential response z = 0.4 minus z = 2 (what a decline needs) is
−0.024 for the decliners and −0.028 for the rest. The premise of the
mechanism (older compact stars in the decliners) holds; the leverage does
not, because the decliners' centres in this model are 73 per cent
accreted-channel mass and the compact deposits are 2.2 kpc across.

## Verdict

**Not adopted; Stage 1 not run** (the rule: fit only on a passing probe).
The age-driven compact expansion is closed as the centre's mechanism in
this model class. What it leaves: (a) the decline lives in whichever
channel dominates the decliners' centres, and in the adopted model that
is the inner part of the EXTENDED channel — a mechanism would have to
act on the early-accreted deposits of early-formed haloes, which q_e does
uniformly with R200c and not by age; (b) the model's in-situ share of the
centre (27–33 per cent at 4.9 kpc) is a modelling choice the data have
never been asked about directly — if TNG's centres are in-situ dominated,
the channel split is what is wrong, and a compact expansion would then
have the leverage it lacks here. Logged as C28.

## Code

- `experiments/exp80_deposit_size_law/size_law.py`: knobs `q_c`, `q_ch`;
  `compact_epoch_dependent`, `compact_sizes_at_epoch`; `predict_law`
  evaluates the compact kernel per epoch when either is on;
  `selfcheck_compact_expansion`. `stage0_candidates.BOUNDS` carries both.
- `stage0_probe.py` (the probe; `--grid`, `--no-retune`, `--smoke`),
  `stage0_anatomy.py`, `stage0_figures.py`; `queue.sh` / `judge.sh`
  prepared for a Stage 1 that was not run (`--knobs q_e,q_c`).
- Outputs: `outputs/stage0_probe.log` / `.npz`, `outputs/stage0_anatomy.log`
  / `.npz`, `outputs/selfcheck_size_law.log`, the smoke logs; the figure
  in `figures/qa/`.
