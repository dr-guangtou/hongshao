# exp86 — where the delayed mass lands: the delayed deposits sized by the halo at arrival (2026-09-23)

Branch `exp86-arrival-radius` (from master `7215df2`). Plan
`doc/plans/2026-09-23-exp86-arrival-radius.md` (open question C26(b), the
arrival-radius check exp82 listed and never ran; chosen by the user from
the 2026-09-23 review of the record). The model under change is THE
ADOPTED MEAN (exp82, `rebaseline.adopted_mean()`: tau_d = 0.15 held,
q_e = 0.152; 14 parameters, 13 fitted). Stage 0 is a frozen-theta probe;
Stage 1 (a fit) was to run only on a passing probe. **No probe passed;
Stage 1 was not run. The adopted mean stands. C26(b) is answered: the
complete progenitors' light shell is the selection, not the arrival
radius. Figure: `figures/qa/exp86_probe_summary.png`.**

## In plain language

**The goal.** The adopted model holds accreted stars in transit for a
while before they count (the delay), and that delay is what fixed the
model's outskirts at high redshift. Its largest remaining cost is that
for the progenitors we can follow completely at z = 1.5 and z = 2 (the
massive haloes at those epochs), the model puts too little mass in the
50–100 kpc shell (12 and 18 per cent too little), while for the sample as
a whole it puts slightly too much there (6 and 7 per cent too much). One
reading of that was physical: the delayed stars are sized by the halo as
it was when they were accreted, but they settle later, in a halo that has
grown since. Sizing them by the halo at ARRIVAL would push them outward,
and more so in haloes that grew fast during the delay.

**What was tried.** A knob that moves the delayed deposits' size reference
from the halo at accretion to the halo at arrival (nesting at the adopted
model at zero). Five strengths, no fit. At each strength the extended
channel's two size constants were re-tuned on the size gate so the
population's sizes stay right, and the same re-tune at zero strength is
the control, so the knob is credited only with what it buys beyond
re-tuning constants the model already has.

**What happened.** Nothing moved. The complete progenitors' shell stays at
−12 and −19 per cent at every strength, the whole sample's at +5, both
within a point of the control (figure, panel a). The re-tune takes back
the enlargement exactly (b_e goes from −1.36 to −1.54 in the control and
−1.67 at full strength).

**Why.** The anatomy, read before the probe, said so: the halo grows by
0.13 dex in radius during a typical delay for EVERY galaxy, and the
complete progenitors' deposits are sized up only 0.013 dex more than the
rest's (panel b). The gap the knob was meant to close is 0.20 dex in the
shell residual between the two samples (panel c). A lever with 0.013 dex
of differential cannot close a 0.20 dex gap; what separates the two
samples is who is in them. The complete progenitors at z ≥ 1.5 are the
massive early haloes, the rest are lighter, and the shared law sits
between the two groups' shells with opposite signs. That is C18, the
completeness selection, seen once more.

**What it means.** C26(b) is closed: the shell split is the selection. The
delayed mass is not landing at a systematically wrong radius. There is a
real but shallow trend INSIDE each sample (galaxies whose halo grew more
during the delay are more under-supplied in the shell, rank correlation
−0.23 to −0.29, slope −2 to −2.5 dex per dex), which says the model's
outskirts under-respond to the halo's growth RATE. The knob here scales
sizes only by the halo's radius growth during the delay, a 0.04 dex
spread; a term that scales the extended deposit by the growth rate itself
is exp76's split, the second lead of the review, and the natural next
probe.

## Stage 0, part 1 — the anatomy of the shell (`outputs/stage0_probe.log`, section 1)

Medians, fitting sample | halo-mass-complete progenitors:

| reading | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 |
| --- | --- | --- | --- | --- | --- |
| shell (model − data)/data [%] | −2.2 \| −2.2 | +0.4 \| −3.7 | +4.4 \| −4.9 | +5.3 \| −12.2 | +6.9 \| −18.1 |
| extended-channel share of the model's shell mass | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| arrival factor log10 R200c(t_a)/R200c(t') of the shell's deposits | +0.107 | +0.111 \| +0.112 | +0.117 \| +0.121 | +0.123 \| +0.131 | +0.130 \| +0.138 |
| mean delay of the shell's deposits, t_a − t' [Gyr] | 0.97 | 0.83 | 0.73 | 0.58 | 0.48 |

At z = 1.5 / 2 the shell residual's rank correlation with the arrival
factor is −0.33 / −0.36 overall, −0.23 / −0.29 inside the complete sample
and the same inside the rest; the factor's median differs between the two
samples by +0.014 / +0.013 dex while the shell residual's median differs
by 0.18 / 0.20 dex (−0.056 vs +0.121; −0.087 vs +0.115). Inside the
complete sample the residual's slope on the factor is −2.5 / −2.0 dex per
dex across a 16–84 span of 0.044 / 0.054 dex.

## Stage 0, part 2 — the probe (section 2)

| point | re-tune: log_f_e, b_e (adopted −0.490, −1.355); radius term | shell z=1.5 fit \| mhc | shell z=2 fit \| mhc | M(<10) z=2 fit \| mhc | size cells ≤ 0.05 of 15 | leak z=0.7 / 1 | rho(recent) z=2, 5 kpc | loss (Δ vs control) | gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| adopted mean | as fitted; 0.0337 | +5.8 \| −11.8 | +7.3 \| −17.7 | +1.7 \| +3.2 | 10 | −0.039 / −0.026 | +0.13 | 12.75 | — |
| control (w_arr 0, re-tuned) | −0.393, −1.539; 0.0325 | +5.5 \| −12.2 | +5.4 \| −19.6 | +3.3 \| +5.1 | 10 | −0.039 / −0.026 | +0.12 | 12.87 | reference |
| w_arr 0.25 | −0.443, −1.538; 0.0323 | +4.9 \| −12.5 | +4.9 \| −18.9 | +3.1 \| +4.8 | 9 | −0.039 / −0.026 | +0.11 | +0.01 | fail (G1, G2) |
| w_arr 0.50 | −0.456, −1.607; 0.0321 | +5.3 \| −12.2 | +4.6 \| −18.7 | +3.4 \| +5.0 | 9 | −0.039 / −0.026 | +0.10 | +0.09 | fail (G1, G2) |
| w_arr 0.75 | −0.489, −1.641; 0.0318 | +5.1 \| −11.9 | +4.6 \| −18.6 | +3.6 \| +4.9 | 9 | −0.040 / −0.026 | +0.09 | +0.14 | fail (G1, G2) |
| w_arr 1.00 | −0.525, −1.673; 0.0315 | +4.7 \| −11.7 | +4.4 \| −18.5 | +3.6 \| +5.0 | 9 | −0.040 / −0.026 | +0.08 | +0.20 | fail (G1, G2) |

Read:

1. **G1 never moves.** The complete progenitors' shell at z = 1.5 / 2 is
   −11.7 to −12.5 / −18.5 to −18.9 at every strength (control −12.2 /
   −19.6; the gate asked for −6 / −10). The fitting sample's shell drifts
   by less than a point. The re-tune absorbs the enlargement: b_e steepens
   with w_arr so that the population's sizes are held.
2. **G2 loses one cell** (10 → 9) at every strength: R20 and R50 improve
   at z ≥ 1 (R50 at z = 2 0.043 → 0.029) but the z = 0.4 R80 offset crosses
   the 0.05 line (0.042 → 0.055). G3 passes throughout; the z = 2
   recent-growth correlation even eases (+0.13 → +0.08 at full strength),
   the one reading in the knob's favour.
3. **The control matters.** Re-tuning the two constants alone moves the
   shell by 2 points at z = 2 (+7.3 → +5.4) and the M(<10) bias at z ≥ 1.5
   by 2 points; the knob adds nothing beyond it.

## Verdict

**Not adopted; Stage 1 not run.** The arrival-radius reading of C26(b) is
closed negative: the delayed mass does not land at a systematically
wrong radius, and the complete-versus-fitting shell split at z ≥ 1.5 is
the completeness selection (C18) acting on a shared law. Carry the
−12 / −18 per cent as a selection cost, as exp82 did, and stop reading
it as a model defect to be fixed by a size law. What the anatomy leaves:
a shallow within-sample dependence of the shell residual on the halo's
growth during the delay (−2 to −2.5 dex per dex), which a growth-RATE
term on the extended deposit (exp76's g, re-read on the adopted mean and
chosen by the gates) could address; that is the next probe if the
outskirts at z ≥ 1.5 are pursued further.

## Code

- `experiments/exp80_deposit_size_law/size_law.py`: knob `w_arr`;
  `arrival_time`, `arrival_references`; `sizes_at_nodes(arrival=)` blends
  the extended channel's z and R200c references toward arrival;
  `selfcheck_arrival`. `stage0_candidates.BOUNDS["w_arr"] = (0, 1)`.
- `stage0_probe.py` (the anatomy + the probe; `--grid`, `--no-retune`,
  `--smoke`), `stage0_figures.py`; `queue.sh` / `judge.sh` prepared for a
  Stage 1 that was not run (`--knobs q_e,w_arr`).
- Outputs: `outputs/stage0_probe.log` / `.npz`, `outputs/selfcheck_size_law.log`,
  the smoke log; the figure in `figures/qa/`.
