# exp78 — the size-aware objective

**Question.** exp63's objective scores no size. It has now ranked a basin
against the QA gates for the fifth time (C23: the 14.63 basin, a 25 kpc
"compact" channel that every gate rejects, beats the adopted baseline by 6
per cent of loss) and rejected a lever the gates want (exp76's growth-rate
split). Can a size term be added to the loss so that the loss ranks models
the way the gates do — and, if so, what does refitting the baseline under
that loss buy and cost?

**Rules that bind (the user, 2026-09-09).** No per-galaxy or per-epoch
quantity from the simulation enters the model; the truth appears only inside
the loss. Every candidate states its total fitted parameter count and which
parameters an observer could vary. The gates decide, the loss does not.

Branch `exp78-size-aware-objective`. Baseline: THE BASELINE MEAN
(`exp74/rebaseline.py::adopted_baseline()`, exp74's measured-input optimum),
on the measured history input (`exp74/measured.py::build_input(recs,
"measured")`), scored under the adopted references (the nested incumbent on
the measured curves = 1.000 per referenced term; halo-mass terciles by the
measured mass). Fitting sample: `selection.fitting_sample_mask`, 2356 of 2397
galaxies at every epoch; the size term's sample is the same rows with a
usable merged-grid truth (2356 / 2356 / 2356 / 2356 / 2354).

## Stage 0 — the two candidate terms, no fit (`stage0_terms.py`, `size_terms.py`)

### What the two terms measure

**(a) The radius term.** For each galaxy, R20, R50 and R80 are the radii
enclosing 20, 50 and 80 per cent of ITS OWN stellar mass inside 148 kpc — the
model's radii from the model's curve, the truth's from the truth's — on
exp73's merged grid (0.673–148 kpc, `coordinate.extended_cog`: the
density-rebuilt curve inside 2 kpc spliced onto the stored curve of growth
outside it, so the truth's R20 is measured rather than extrapolated for the
compact high-redshift galaxies). The residual is log10(R_model / R_truth) in
dex. At each epoch the MEDIAN residual is taken in each halo-mass tercile
(the binned term's own terciles), giving 3 fractions × 3 terciles = 9
numbers; the term is their rms. Reported raw (dex) and normalised to the null
(1.000 at the nested incumbent), like S and B. It is a median term on
purpose: it is the size gate's OFFSET sub-gate written as a loss, and it does
not see the per-galaxy size scatter that no mean model can reproduce (C16),
which is the stochastic layer's job.

**(b) exp77's transformed annular term.** On the 50–100 and 100–148 kpc
annuli, T = asinh(M_annulus / (0.001 × total)) / ln 10; the per-galaxy mean
over the two annuli of (T_model − T_truth)², then the rms over galaxies at
each epoch. exp77 divided both sides by the TRUTH's total; a second form
dividing each side by its OWN total is carried because the first is an
amplitude term as much as a shape term.

Neither term has a free constant. **Parameter count for any fit under
them: 12 (exp63's two-channel model) + 0 for the term.**

### The blind-spot probes, asserted on the real truth (z = 0.4 and z = 2 rows)

Synthetic "models" built from the truth itself; a term that cannot see a
change reads 0.

| probe (what the model is) | radius term | annular, truth total (exp77) | annular, own total |
| --- | ---: | ---: | ---: |
| the truth itself | 0 | 0 | 0 |
| the truth × 1.3 in amplitude | **0** (asserted) | 0.114 | 0 |
| the same galaxy 1.2× larger | 0.063 / 0.077 dex (z=0.4 / z=2; log10 1.2 = 0.079) | 0.071 / 0.168 | 0.077 / 0.168 |
| a point mass of 10 % of the total added at the centre | 0.120 / 0.088 | **0.000 (blind)** | 0.041 / 0.040 |
| a per-galaxy size scatter of 0.15 dex with zero median | 0.008 / 0.004 | 0.50 / 0.42 | 0.50 / 0.42 |

Reading. The radius term is blind to a pure amplitude rescaling (by
construction: fractions of the galaxy's own mass), moves by about the size
shift it is given, and moves when mass is added inside the first radius — the
three properties the user required. It barely moves under a zero-median
per-galaxy size scatter, which is the intended behaviour of an offset term.
The size-shift response is a little under log10 1.2 at z = 0.4 because
stretching the curve also moves mass past 148 kpc, so the fractional radii
of the stretched curve are not exactly 1.2× the originals. exp77's annular
term, in its own form, is **blind to a central point mass** (the annuli and
the truth's total do not change) and reads a pure amplitude rescaling as a
0.11 dex error; it is dominated by per-galaxy scatter (0.4–0.5 dex for a
scatter that leaves the medians untouched). The own-total variant fixes the
amplitude blindness but keeps the rest.

### The ranking of the models we already have

Every model on its own input, the fitting sample, full quadrature. The
"loss" column is exp63's A²+F²+S²+B² under the adopted references (no size
term); "offset pass" is the size gate's offset sub-gate at fixed stellar
mass on the standard grid (of 15 = 3 sizes × 5 epochs); the last three
columns are the z = 2 R50 and R80 offsets (dex) and the R80 width ratio.
Each term's Z is the rms over the five normalised per-epoch values.

| model | loss | offset pass | R50 z=2 | R80 z=2 | R80 width z=2 | Z radius | Z annular (exp77) | Z annular (own) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| null (nested incumbent, measured) | 20.86 | — | — | — | — | 1.000 | 1.000 | 1.000 |
| exp63 official (official DiffMAH curves) | 14.83 | 14 | +0.020 | −0.025 | 0.16 | 0.609 | 0.906 | 0.924 |
| **exp74 optimum (the baseline)** | 15.56 | 12 | +0.054 | +0.039 | 0.27 | **0.550** | 0.933 | 0.948 |
| 14.63 basin (C23, gate-rejected) | 14.63 | 10 | +0.082 | +0.064 | 0.22 | 0.642 | **0.807** | **0.797** |
| incumbent re-baselined | 16.14 | 12 | +0.086 | +0.117 | 0.24 | 0.946 | 0.938 | 0.947 |
| exp76 split (g = −0.27, from the basin) | 15.00 | 11 | +0.072 | +0.035 | 0.48 | 0.589 | 0.875 | 0.880 |

Per-epoch radius term, normalised (z = 0.4 / 0.7 / 1.0 / 1.5 / 2.0): the
baseline 0.51 / 0.63 / 0.60 / 0.55 / 0.45; the 14.63 basin 0.66 / 0.67 /
0.56 / 0.66 / 0.65; exp76's split 0.64 / 0.69 / 0.54 / 0.55 / 0.52. Raw, the
baseline's tercile-median size offsets are 0.023–0.052 dex against the
null's 0.045–0.116.

**The radius term PASSES the ranking check**: it orders the models exp74
optimum (0.550) < exp76 split (0.589) < exp63 official (0.609) < 14.63 basin
(0.642) < incumbent (0.946). It prefers the adopted baseline over the 14.63
basin (the gates: offset 12 vs 10 of 15; R50 at z = 2 +0.054 vs +0.082 dex)
and exp76's split over the basin it was started from (11 vs 10; R80 at z = 2
+0.035 vs +0.064, width 0.48 vs 0.22). Where the term sees the basin's
failure: R50 in the top halo-mass tercile at z = 1.5–2 (+0.114 / +0.150 dex
against the baseline's +0.049 / +0.098) and R50 at z = 0.4–0.7 in the lower
terciles (−0.04 against −0.01 to −0.03).

**Both annular forms FAIL it, the same way**: they rank the 14.63 basin
BEST (0.807 / 0.797) and the baseline next to the incumbent (0.933 / 0.948).
The basin fills the 50–148 kpc envelope better per galaxy — that is what it
buys with its 25 kpc channel — and an outskirts term rewards exactly that.
This is exp77's own finding from the other side: a better outer envelope is
not a better size (exp77 worsened R50 at z = 1.5 by 0.02 dex).

**The negative that matters for Stage 1.** With weight 1 per epoch, adding
Z² does NOT flip the loss's ordering of the two basins: the baseline goes
15.56 → 17.07 and the 14.63 basin 14.63 → 16.69. The radius term separates
them by 0.38 in the right direction but the four old terms separate them by
0.93 in the wrong one. The size-aware loss is a compromise between the
per-galaxy centre (S, F) and the size, not a veto; whether the fit under it
lands in a gate-clean basin is what Stage 1 measures, and the gates — not the
loss — decide the result.

Decision: **the radius term enters the loss; the annular term does not.**
Log: `outputs/stage0_terms.log`; numbers: `outputs/stage0_terms.npz`.

## Stage 1 — the fit under A² + F² + S² + B² + Z² (`stage1_fit.py`, `stage1_eval.py`)

**What was fitted.** exp63's 12-parameter two-channel model on the measured
history, the adopted references and the fitting sample, with the radius term
added at weight 1 per epoch (Z = 1.000 at the null, like S and B). **12
fitted parameters, plus 0 for the term; an observer could vary all 12.** The
null loss is 25.85 (five epochs × five terms, A and F not referenced). Four
starts as separate processes, each capped at 3000 evaluations, then a
continuation of the capped baseline start to settle its basin:

| start | loss under the size-aware objective | evaluations | note |
| --- | ---: | ---: | --- |
| the baseline (exp74's optimum) | 17.07 → 16.31 | 3004 (cap) | continuation 16.31 → 16.31, stationary |
| the 14.63 basin | 16.70 → **15.82** | 2367 | converged; **the compact size railed at its upper bound** |
| exp63's official-curve theta | 19.34 → 16.39 | 3017 (cap) | |
| the nested incumbent | 104.9 → 16.45 | 3004 (cap) | railed on the same bound |

**Two basins, and what the loss did with the size term.** The minimum
(15.82) is the 14.63 basin's family: from the baseline it moves the compact
channel's size from 10^0.98 = 9.6 kpc to the box edge 10^1.50 = 31.6 kpc
(`log_f_c` railed) with a steeper time exponent (`b_c` −0.88 → −1.83, the
size going as (1+z)^b_c at the deposit), so early deposits stay at 3–4 kpc
while deposits after z ≈ 1 land at 9–17 kpc; the compact share's halo-mass
threshold rises (`m_half` 11.78 → 12.42) and its width (`d_split` 0.27 →
0.95). The baseline start settles in its own basin at 16.31, also with the
compact size railed, but with a steep compact profile (`n_c` 1.38) and a
smaller share. Under the size-aware loss the baseline scores 17.07 and the
14.63 basin 16.69, so the term (0.38 in the right direction) did not
outweigh the four old terms (0.93 the other way), as Stage 0 predicted.

**Where the term's gain came from.** The radius term at the optimum is 0.34
/ 0.43 / 0.34 / 0.54 / 0.49 (z = 0.4 … 2.0), against the baseline's 0.51 /
0.63 / 0.60 / 0.55 / 0.45. All the improvement is at z ≤ 1, where the
baseline already passed the size gate; at z = 1.5 the term is unchanged and
at z = 2 it is WORSE. The high-redshift size excess sits in the top
halo-mass tercile (R50 +0.077 / +0.115 dex at z = 1.5 / 2 against the
baseline's +0.049 / +0.098) and the four old terms defend it: it is the
honest input's outward push of the extended deposits (exp74 addendum), and
shrinking those galaxies costs the per-galaxy shape terms more than a
weight-1 size term repays. A weight-1 term buys the cheap sizes, not the
expensive ones.

### The gates (`outputs/stage1_eval.log`; figures `figures/qa/*exp78_*`)

Size gate, offset | width, of 15 (fixed stellar mass; the same counts at
fixed halo mass), and the numbers that changed:

| model | loss (old terms) | offset | width | R50 z=1.5 / 2 (dex) | R80 z=1.5 / 2 | R20 width z=0.4 | centre z=0.4 M(<2) | 50–100 kpc z=2 | M(<10) z=2 mh-complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **baseline** | 15.56 | **12** | 0 | +0.037 / +0.054 | +0.015 / +0.039 | 0.69 | +5.6 % | +29.6 % | −0.5 % |
| 14.63 basin | 14.63 | 10 | 0 | +0.069 / +0.082 | +0.029 / +0.064 | 0.71 | +9.0 % | +25.4 % | −2.4 % |
| size-aware (15.82) | 14.87 | 11 | 0 | +0.059 / +0.068 | +0.017 / +0.046 | 0.77 | +8.4 % | +9.8 % | −1.8 % |
| size-aware, base basin (16.31) | 15.28 | 11 | 1 | +0.053 / +0.067 | −0.013 / −0.003 | 0.80 | +6.8 % | −3.9 % | +3.1 % |

Positives of the size-aware optimum against the baseline: the outer
envelope at z ≥ 1.5 (50–100 kpc shell +29.6 → +9.8 per cent at z = 2, +17.6
→ +8.3 at z = 1.5, on the fitting sample); the size WIDTHS at every entry
(R20 0.69 → 0.77 at z = 0.4, R50 0.35 → 0.43 at z = 2); the R20 offsets by
0.003 dex; the halo-mass tilt of the 103 kpc residual smaller at every
epoch. Negatives: the offset sub-gate 12 → 11; R50 worse at z = 1.5 and 2
(+0.037 → +0.059, +0.054 → +0.068); R80 worse at z = 2; the z = 0.4 centre
+5.6 → +8.4 per cent (the basin's central defect); the future-dependence
gate unchanged within 0.005 dex per dex (−0.061 → −0.066 at z = 1; the
truth +0.052). The mass–size slope at z = 2 is 0.37 against the truth's 0.11
(the baseline 0.33): the size error still grows with stellar mass at high
redshift, and neither loss sees a slope.

The second basin (the baseline start settled) is the gate-cleaner of the
two size-aware solutions: R80 within 0.02 dex at every epoch, the first
width pass ever on a mean model (R20 at z = 0.4, 0.80), the envelope at z =
2 −3.9 per cent, the centre +6.8; but R50 at z ≥ 1.5 as bad as the other,
M(<10) at z = 1.5 mh-complete +9.3 (baseline +7.8), M(<100) at z = 0.4
−5.1 (baseline −3.1), and it is 0.49 higher in the loss than the optimum.

**The second basin's curved mass planes (the user's reading of
`qa_planes_exp78_size-aware_(base)`).** In the outer-vs-inner mass planes
at z ≥ 1 the second basin bends: the slope of M(50–100) against M(<30) at
z = 2 is 2.51 against the truth's 1.66 (the baseline 1.63, the optimum
1.83), with the low-mass end falling off a cliff. The cause is the fitted
compact/extended switch: its width `d_split` is 0.137 dex (the baseline
0.27, the optimum 0.95) at `m_half` = 12.30, so a deposit made while the
halo was below 10^12.3 goes almost entirely to the compact channel and one
made above it to the extended channel. At z = 2 the lowest quintile of
M(<30) is 89 per cent compact and holds 10^−1.79 of its inner mass at
30–100 kpc, the highest quintile 24 per cent and 10^−0.82 (the baseline:
47 per cent and 10^−1.27; 11 per cent and 10^−0.82). A threshold in halo
mass becomes a knee in the mass plane; the steeper compact profile (`n_c`
1.38) sharpens it. The tercile MEDIANS the loss scores do not see a knee,
which is why this basin reads well on the envelope and badly in the plane —
one more reading the QA planes give and the loss does not.

**Verdict (the gates decide).** Neither size-aware solution beats the
baseline on the offset sub-gate (11 vs 12) or on R50 at z ≥ 1.5, which is
the failure the term was built to fix; both buy the outer envelope and some
width at the cost of the centre. **Not adopted.** The radius term is the
right term (Stage 0) and its weight-1 refit is a compromise the loss
resolves toward the wide "compact" channel. Two consequences for the
record: (1) the compact channel's size bound (1.5 = 31.6 kpc) is now
binding under every size-aware start, so the model class is asking for a
late-time deposit larger than the bound allows, and "compact" no longer
describes it; (2) the high-z R50 excess of the massive progenitors is not
reachable by re-weighting the loss — it needs the extended deposit's r200
scaling under the honest input (owed since the exp74 addendum), i.e. a model
change, not an objective change (the sixth case of the loss and the gates
disagreeing, C20 / C21 / C23 again).

## Stage 2 — the same fit with the growth-rate split free (13 parameters)

From the Stage 1 optimum with g = 0, −1, −2 (`--growth`): **13 fitted
parameters, plus 0 for the term.** g = 0 stays at 0 (351 evaluations,
15.8248 → 15.8248); g = −1 walks back to **g = −0.005** at 15.812 (railed on
the compact size and `n_c` at 0.5); g = −2 stops at g = −0.08 at 16.15
(capped). The size-aware loss rejects the split exactly as exp63's loss did
(exp76: g → 0 from the baseline, −0.27 from −1 and −2 at 2.5 per cent worse
loss). The size term does not rescue exp76's lever; what exp76's g = −0.27
bought (R80 width 0.22 → 0.48 at z = 2) is a width, and the radius term is
an offset term by design. exp76 stays not adopted, not closed; option 2 (g
fixed by physics, with the layer owning the width) is the remaining route.

## What is owed after exp78

- Re-tune the extended deposit's r200 scaling under the measured input: the
  high-z R50 excess is in the top halo-mass tercile and grows with stellar
  mass; it is a model-class limit, not a weighting.
- The compact channel's size bound: widen it (or re-parametrise the channel
  as "late, extended") and re-run the size-aware fit only if the model
  change above does not remove the demand for it.
- If a size term is kept in the loss, its weight is a decision the gates
  must make (a sweep of the weight, gated, not chosen by the loss); a WIDTH
  term belongs to the stochastic layer, not the mean.

## Files

- `size_terms.py` — the two terms, the probes, the vectorised fractional radius.
- `stage0_terms.py` — Stage 0; `outputs/stage0_terms.{log,npz}`.
- `stage1_fit.py` — `SizeAwareProblem`, the Stage 1 / Stage 2 fits
  (`--starts k:k`, `--continue NAME`, `--growth`, `--merge`);
  `outputs/stage1_fit_start_*.npz`, `stage1_fit.npz`, `stage2_fit_growth*.npz`, logs.
- `stage1_eval.py` — the judge; `outputs/stage1_eval.{log,npz}`, `figures/qa/*exp78_*`.
