# exp83 — the early-mass conditioning of the centre on the adopted mean (2026-09-13)

Plan: `doc/plans/2026-09-13-exp83-early-mass-centre.md` (roadmap S3). The
model under change is THE ADOPTED MEAN (exp82, `rebaseline.adopted_mean()`:
the delay tau_d = 0.15 held + q_e = 0.152 on exp63's twelve; 14 parameters,
13 fitted). Stage 0 is a frozen-theta probe; Stage 1 (a fit) was to run
only on a probe that passes the gate. No probe passed; the user then chose
(2026-09-21) to fit the mass half anyway — Stage 1 below. **Verdict: the
fitted term is weak (a_early −0.08), the model is the adopted mean within
the gates' resolution, and it is NOT recommended for adoption.**

## Stage 0, part 1 — the residual's form (`outputs/stage0_probe.log`, section 1; `outputs/residual_check_adopted.log`)

The adopted mean's residual r = log10(model/truth) on the fitting sample
(2356 galaxies), against the early-mass fraction x = log M(2 Gyr) −
log Mh(z_k) at fixed halo mass:

| cell | partial rho with x | slope of r on x [dex/dex] | the truth's own slope |
| --- | ---: | ---: | ---: |
| z = 0.4, 5 kpc | +0.37 | +0.173 | +0.241 |
| z = 0.4, 103 kpc | +0.33 | +0.114 | +0.091 |
| z = 1.0, 103 kpc | +0.21 | +0.071 | +0.146 |
| z = 1.5, 5 kpc | +0.07 | +0.026 | +0.356 |
| z = 2.0, 5 kpc | −0.15 | −0.116 | +0.542 |
| z = 2.0, 103 kpc | −0.12 | −0.078 | +0.467 |

- **The form is linear.** In quintiles of x the median residual at z = 0.4
  rises monotonically through the low and middle halo-mass terciles
  (5 kpc: −0.15 → +0.07 and −0.14 → +0.06 dex; 103 kpc: −0.09 → +0.04 and
  −0.08 → +0.03); the high tercile is flat with a downturn in the top
  quintile (−0.05 at 5 kpc). No saturation, no tail.
- **Early-formed = the decliners.** The decliner fraction rises with x from
  12 to 76 per cent across the quintiles at z = 0.4; at 4.9 kpc the
  decliners are +0.02 → +0.09 too heavy at z = 0.4 and −0.12 → −0.05 too
  light at z = 2 across the quintiles.
- **The z = 0.4 residual has TWO components.** The residual at 5 kpc at
  fixed halo mass AND fixed residual at 103 kpc (the concentration part,
  what no efficiency term can reach) still correlates with x: rho +0.14,
  slope +0.059 dex per dex (the truth's own log M(<5)/M(<103) on x is
  +0.150). The mass part (103 kpc) is +0.114 dex per dex; the centre carries
  +0.06 on top of it. At z = 2 the concentration part is −0.11 / −0.038.
- **Where the deposits sit.** Weighted by the adopted model's deposited
  stellar mass by z = 0.4, 20 per cent of the stars were deposited before
  2 Gyr (where the deposit variable is zero for every halo by
  construction); the raw per-deposit phi(t') runs from −1.30 (10th
  percentile) to 0; relative to the population's median at each time it
  runs −0.43 to +0.18.

## Stage 0, part 2 — the frozen-theta probe (`outputs/stage0_probe.log`, section 2)

The term: for a deposit at t', phi(t') = min(log M(2 Gyr) − log M(t'), 0)
taken relative to the fitting sample's median at that time (a fixed table,
`size_law.set_early_ref`; the raw variable is a second time law — the smoke
probe with it, re-centred on z = 0.4, dragged z = 2 down by 0.18 dex). Three
placements, each nesting at zero: `a_early` on the efficiency
(× 10^(a_early phi)), `s_early` on the compact share's logit, `c_early` on
the compact size (× 10^(c_early phi)). The adopted theta held; a0 re-centred
per probe so the z = 0.4 median M(<103) is unchanged (exact).

GATE (the user, 2026-09-13): the z = 0.4 early-mass correlation halved at 5
AND 103 kpc; the decliners' central change z = 2 → 0.4 closer to the truth's
−0.066 than the adopted +0.039; the non-decliners' z = 2 centre median
within 0.02 of the adopted +0.051.

| probe | rho(x) z=0.4 5 kpc (gate ≤ +0.18) | 103 kpc (≤ +0.17) | z=2 5 kpc | decliners' change (truth −0.066) | non-decl. z=2 (Δ) | rho(recent) z=2 | leak z=0.7 / 1 | loss (Δ) | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| adopted mean | +0.37 | +0.33 | −0.15 | +0.039 | +0.051 | +0.13 | −0.039 / −0.026 | 12.75 | — |
| a_early −0.10 | +0.33 | +0.23 | −0.16 | +0.037 | (−0.001) | +0.14 | −0.036 / −0.024 | −0.00 | fail (5, 103) |
| a_early −0.15 | +0.32 | +0.17 | −0.17 | +0.036 | (−0.002) | +0.15 | −0.035 / −0.023 | +0.08 | fail (5) |
| a_early −0.20 | +0.30 | +0.11 | −0.18 | +0.035 | (−0.002) | +0.16 | −0.033 / −0.022 | +0.21 | fail (5) |
| **a_early −0.30** | **+0.26** | **−0.01** | −0.19 | +0.034 | (−0.002) | +0.17 | −0.030 / −0.021 | +0.66 | fail (5) |
| a_early −0.40 | +0.23 | −0.12 | −0.21 | +0.033 | (−0.004) | +0.18 | −0.027 / −0.019 | +1.36 | fail (5) |
| a_early +0.10 | +0.39 | +0.43 | −0.13 | +0.039 | (+0.000) | +0.11 | −0.041 / −0.028 | +0.23 | fail |
| s_early −2 | +0.34 | +0.33 | −0.17 | +0.038 | (+0.003) | +0.15 | unchanged | +0.00 | fail |
| s_early +2 | +0.38 | +0.34 | −0.13 | +0.040 | (−0.001) | +0.11 | unchanged | +0.00 | fail |
| c_early +0.3 / +0.6 / +1.0 | +0.36 / +0.35 / +0.34 | +0.33 | −0.16 / −0.16 / −0.17 | +0.038 / +0.037 / +0.037 | (0.000) | +0.14 / +0.14 / +0.15 | unchanged | −0.01 / −0.03 / −0.03 | fail (5, 103) |
| a_early −0.3 + c_early +0.5 / +1.0 | +0.25 / +0.24 | −0.01 | −0.21 / −0.22 | +0.034 / +0.033 | (+0.004) | +0.18 / +0.19 | −0.030 / −0.021 | +0.60 / +0.56 | fail (5) |

**What the probe says.**

1. **The efficiency term removes the MASS half exactly.** At a_early ≈
   −0.28 the 103 kpc correlation goes +0.33 → 0 (a dex-per-dex slope of
   −0.3 on the relative early fraction, i.e. the efficiency FALLS with the
   mass fraction assembled by 2 Gyr — the opposite sign to the roadmap's
   guess, which was written for the z = 2 side before the target moved).
   The future-dependence gate improves with it (z = 0.7: −0.039 → −0.030;
   the truth −0.006), the non-decliners' z = 2 median does not move, the
   decliners' mismatch shrinks by 0.005. Costs at frozen theta: the loss
   +0.66 (a refit would take some back), the z = 2 recent-growth
   correlation +0.13 → +0.17 (the delay had closed it to +0.13), the z = 2
   early correlation −0.15 → −0.19.
2. **No placement reaches the CENTRE.** The 5 kpc correlation falls only
   +0.37 → +0.26 at a_early = −0.3 (it tracks the mass half and stops at
   the concentration part, +0.14); the share and the compact size do
   nothing to it (+0.34 to +0.38 at every value), alone or combined with
   a_early. The reason is structural: the concentration part lives in the
   stars deposited BEFORE 2 Gyr (20 per cent of the z = 0.4 mass, most of
   the centre), where phi is zero for every halo by construction, so a term
   keyed to the mass fraction assembled by 2 Gyr cannot touch it. The
   early-formed haloes' centres are too concentrated at z = 0.4 and too
   diffuse at z = 2 at fixed total — the decline itself, which a
   per-deposit efficiency or share cannot produce (memory:
   `central-defect-is-outside-the-model-class`).
3. **Gate verdict: fail** at every probe value on the 5 kpc half; two of
   the three sub-gates (the decliners' mismatch, the non-decliners' z = 2
   median) pass at a_early ≤ −0.1. **Stage 1 was not run** (the user's
   rule: fit only on a passing probe).

## Stage 1 — the fit of the mass half (the user's decision, 2026-09-21/22)

`queue.sh 2 -0.28`: three starts from the adopted mean's fourteen with
a_early at −0.28 / 0 / −0.56, the standard objective, tau_d held at 0.15,
two fits at a time. **Parameters: 15 in the model (exp63's twelve, tau_d
held, q_e, a_early); 14 fitted.** Logs `outputs/fit_a_early_s{0,1,2}.log`,
merge `outputs/merge.log`, the judge (full battery, figures
`figures/qa/*exp83_*`; the judged model is labelled `exp80` inside the log,
the script's name for "the model under judgement")
`outputs/eval_adopted_far.log`, residual checks
`outputs/residual_check_adopted_{probe,far}.log`.

| start | a_early: start → fitted | q_e | loss: start → fitted | evaluations |
| --- | --- | ---: | --- | ---: |
| adopted_probe | −0.28 → −0.080 | 0.174 | 13.58 → 12.702 | 1156 |
| adopted_zero | 0 → −0.063 | 0.150 | 12.746 → 12.706 | 1426 |
| adopted_far | −0.56 → −0.082 | 0.177 | 16.89 → 12.699 | 3016 (the cap) |

**One basin, and the parameter moves**: three starts from both sides of
the answer agree on a weak negative coefficient (−0.06 to −0.08) within
0.007 in loss, with the adopted mean's structure (d_split 0.29–0.30, n_c
0.71–0.73). Judged BY NAME: `adopted_far` (the lowest), with `adopted_zero`
alongside.

| reading | previous baseline | adopted mean (exp82) | exp83 fit (a_early −0.082) |
| --- | ---: | ---: | ---: |
| loss, full nodes | 15.56 | 12.70 | 12.65 |
| early-mass correlation of the residual at z = 0.4, 5 kpc / 103 kpc (the target; halved = +0.18 / +0.17) | — | +0.37 / +0.33 | **+0.33 / +0.26** |
| the same at z = 2, 5 kpc / 103 kpc | — | −0.15 / −0.12 | −0.17 / −0.14 |
| recent-growth correlation at z = 2, 5 kpc | — | +0.13 | +0.15 |
| decliners' central change z = 2 → 0.4 (truth −0.066) | +0.061 | +0.039 | +0.036 |
| non-decliners' z = 2 centre median | — | +0.051 | +0.052 |
| size gate at fixed stellar mass, offsets / widths of 15 | 12 / 0 | 14 / 1 | 15 / 0 |
| M(<2 kpc) per cent, z = 0.4 / 1.5 / 2 | +5.6 / −9.8 / −12.6 | +0.5 / −6.2 / −6.9 | −0.2 / −5.9 / −6.8 |
| M(<103 kpc) per cent, z = 0.4 / 1 / 2 | −3.1 / +1.2 / −1.4 | −4.2 / +3.8 / +0.2 | −4.0 / +3.9 / +0.4 |
| future-dependence gate, dex per dex, z = 0.7 / 1 / 1.5 / 2 (truth −0.006 / +0.052 / +0.025 / +0.003) | −0.013 / −0.046 / −0.026 / −0.005 | −0.039 / −0.026 / −0.009 / −0.015 | −0.037 / −0.024 / −0.009 / −0.016 |
| mh-complete 50–100 kpc shell, z = 1.5 / 2 | +4.4 / −0.5 | −11.8 / −17.7 | −9.5 / −14.6 |
| fitting-sample 50–100 kpc shell, z = 1.5 / 2 | +17.6 / +29.6 | +5.8 / +7.5 | +8.8 / +11.9 |

**Read.**

1. **The target is not met.** The z = 0.4 early-mass correlation falls
   from +0.33 to +0.26 at 103 kpc and from +0.37 to +0.33 at 5 kpc — a fifth
   and a tenth of the way, against the gate's half. The loss will not pay
   for the coefficient that removes the mass component (−0.28 costs +0.66
   at frozen theta; the fit settles at −0.08 for a gain of 0.05, 0.4 per
   cent of the loss). This is the prediction made before the fit, and an
   eighth loss-vs-gates reading of a mild kind: the loss is nearly flat in
   the parameter while the diagnostic wants 3.5 times more of it.
2. **Everything else is the adopted mean.** The size gate's 15 / 0 against
   14 / 1 is two threshold crossings, not a change: R20 at z = 2 moves
   +0.050 → +0.049 across the 0.05 line, and the R20 width at z = 0.4 is
   0.80 in both models, on the line. The centre, the cumulative profile
   and the future-dependence gate move by less than their resolution; the
   decliner split is unchanged (the model's change +0.036 against the
   truth's −0.066). The one visible trade is in the 50–100 kpc shell at
   z ≥ 1.5: the mh-complete progenitors 3 points better, the fitting sample
   3–4 points worse (the refitted q_e 0.177, not a_early, is the likely
   mover).
3. **Verdict: NOT recommended for adoption.** A fifteenth parameter that
   buys 0.4 per cent of the loss and a fifth of its own target does not
   earn its place; the adopted mean stays `rebaseline.adopted_mean()`. The
   result is still informative: the halo's early-mass fraction carries a
   real, shuffle-free signal in the total stellar mass at z = 0.4 (the
   frozen probe zeroes it cleanly), but the shared five-epoch objective
   prices it below the damage it does at z = 2, where the same variable
   enters with the opposite sign (−0.15 → −0.17). The two signs are the
   decline seen from its two ends; a term with one sign cannot serve both.

## What was prepared before the user's decision (kept for the record)

- The fit on the mass half alone: `queue.sh 2 -0.28` (three starts from
  the adopted fourteen with a_early at −0.28 / 0 / −0.56, two at a time,
  own sessions), then `judge.sh --best NAME [--figures]` — the judge lists
  the adopted mean AND the previous baseline next to the fit. Model 15
  parameters (exp63's twelve, tau_d held, q_e, a_early); 14 fitted. Check
  that a_early moves (it has a gradient: the loss changes by +0.08 at
  −0.15 and +0.66 at −0.3 at frozen theta, so the fit will pull it toward
  a shallower value than the gate's −0.28 — expect a loss-vs-gates
  reading).
- The centre half needs a different mechanism: the pre-2-Gyr compact
  deposits of early-formed haloes must expand between z = 2 and z = 0.4
  (an age-driven expansion of the compact channel, or a per-galaxy
  formation-time size term), which is a model change outside S3's
  one-parameter scope and is logged in `doc/open_questions.md`.

## Code

- `experiments/exp80_deposit_size_law/size_law.py`: knobs `a_early`,
  `s_early`, `c_early`; `early_fraction_raw`, `set_early_ref`, `early_ref`,
  `early_fraction`; `selfcheck` covers them (LAW_DEFAULT still nests bit
  for bit; each knob moves the profile).
- `experiments/exp63_analytic_growth/model2.py`: `compact_share(...,
  logit_extra=None)` (None leaves the share exactly).
- `experiments/exp80_deposit_size_law/stage1_fit.py`: the reference set at
  build time when an early knob is in use; `--start-adopted knob=value`
  (three starts from the adopted mean: probe / 0 / twice the probe);
  `stage1_eval.py`: the adopted mean judged alongside when the fit extends
  it; `stage0_candidates.py`: the knob bounds.
- `stage0_probe.py`, `queue.sh`, `judge.sh` here; outputs
  `outputs/stage0_probe.log` (this run), `outputs/stage0_probe_round1.log`
  (the first grid without c_early; identical numbers), `outputs/stage0_probe.npz`,
  `outputs/residual_check_adopted.log` (exp81's check on the adopted file,
  identical to exp82's).
