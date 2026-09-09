# exp80 — the deposit size law under the measured input, read from the data first

Branch `exp80-deposit-size-law` (from master `e2dcd09`, 2026-09-09). Plan:
`doc/plans/2026-09-09-exp80-deposit-size-law.md`. exp79 (Codex, the
truncation boundary C × R200c) is untouched here: nothing in this experiment
changes C.

**The question.** The programme's standing failure is the high-redshift SIZE
offset of the adopted baseline mean (exp74's measured-input optimum): the
half-mass radius R50 is +0.037 / +0.054 dex too large at z = 1.5 / 2, most of
it in the top halo-mass tercile, with a mass–size slope at z = 2 of 0.19–0.33
(depending on the grid) against the truth's 0.11. exp78 showed this is not an
objective problem. This experiment asks the data, epoch by epoch, WHERE they
want the stars deposited (the deposit half-mass radius s), compares that with
where the baseline's own size law puts them, and only then changes the law.

The baseline's size law (exp63's two-channel model, 12 parameters, on the
measured halo history): a compact channel of Sersic n_c = 0.56 deposits at
s_c = 9.6 (1+z')^−0.88 kpc, and an extended channel (gompertz, c_e = 0.80) at
s_e = 0.19 (1+z')^−1.03 R200c(t') — the halo radius AT THE DEPOSIT TIME t'.
The split sends deposits made while the halo was below 10^11.8 to the compact
channel (width 0.27 dex); by z = 0.4 the extended channel carries 85–95 per
cent of the stellar mass.

## Stage 0 A/B — what the data demand at every epoch, against the baseline (`stage0_deconvolve.py`; no fit)

**Method.** exp63 Stage 1's operator, generalised: for each galaxy and epoch,
the deposit-size distribution W(s) (stellar mass per log deposit size, 20 bins
from 0.3 to 600 kpc) is read from the curve of growth by non-negative least
squares under a fixed deposit kernel, on exp73's merged 0.673–148 kpc grid
(the compact mode is resolved below 2 kpc). Three kernels: the baseline's own
extended kernel (gompertz c = 0.80 — exp63's incumbent kernel to three
decimals), its compact kernel (Sersic n = 0.56), and exp63's Sersic n = 1.
**The same operator is applied to the baseline's own predicted curves**, so
data and model are compared in the same, equally biased, coordinates; and the
baseline's TRUE deposit-size distribution is read analytically from its nodes
(`model2._deposits2`, both channels' masses histogrammed on the same size
grid) as the reference for what the operator does to a known W. Sample: the
fitting sample (2356 galaxies, all five epochs; 2354 at z = 2 with a usable
merged curve); the mh-complete subset is reported alongside. Terciles are the
binned term's, by the measured halo mass at each epoch. Per tercile the
"compact size" is the mass-weighted mean log s below the split between the
two modes, the "extended size" above it, with the tercile median, its
bootstrap standard error and the 16–84 per cent spread. The split is read
from the data's stacked W at each epoch (the minimum between its two largest
modes: 30 kpc at z ≤ 1 under the extended kernel) and applied to the model;
at z ≥ 1.5 the data show ONE mode and a 15 kpc fallback is used. Log:
`outputs/stage0_deconvolve.log`; arrays: `outputs/stage0_deconvolve.npz`;
figures `figures/exp80_stage0_w_stack.png` (the stacked W per kernel and
epoch: data solid, baseline deconvolved dashed, baseline's true deposits
dotted) and `figures/exp80_stage0_sizes_vs_z.png`. The z = 0.4 standard-grid
check reproduces exp63 Stage 1's shares at s ≥ 15 kpc (0.34 / 0.45 / 0.58
against 0.35 / 0.45 / 0.57) and its compact mode bin.

### Finding 1 — the deconvolution's two modes are partly the operator's

The baseline's true deposit-size distribution is a CONTINUUM: deposits from
3 kpc (made at z' ≈ 5) to 60 kpc (made at z' ≈ 0.4), one broad hump peaking
at 36 kpc at z = 0.4 with no gap. Deconvolved with its own kernel it reads as
TWO modes at 7.4 and 54 kpc with a gap between them (gap depth 0.17 against
the data's 0.31), because non-negative least squares returns sparse solutions
under a smooth kernel: a continuum and a pair of spikes make the same curve.
exp63 Stage 1's "bimodal in 92 per cent of galaxies" therefore does not by
itself establish two physical scales (its permuted control tested the
representation, not the bimodality). The comparison that is meaningful is
data-deconvolved against model-deconvolved, and the operator's bias is
readable from the model: at z = 0.4 it moves the true extended size 43 → 76
kpc in the low tercile and 54 → 59 kpc in the high one, and the extended
share by +0.04 to +0.09.

### Finding 2 — the table (extended kernel; kpc; data | baseline deconvolved the same way)

| epoch | tercile | compact mode: data | model | diff (dex, s.e.) | extended mode: data | model | diff (dex, s.e.) | ext. share data | model | s_e / R200c(z_k): data | model |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | low | 7.0 | 6.2 | +0.054 (6) | 61.5 | 76.0 | −0.092 (4) | 0.22 | 0.24 | 0.14 | 0.18 |
| 0.4 | mid | 6.8 | 6.5 | +0.019 (3) | 71.2 | 59.8 | +0.076 (5) | 0.36 | 0.34 | 0.14 | 0.12 |
| 0.4 | high | 6.9 | 7.0 | −0.003 (0) | 88.9 | 58.6 | **+0.181 (14)** | 0.53 | 0.51 | 0.13 | 0.08 |
| 0.7 | low | 6.6 | 6.0 | +0.035 (4) | 80 | 119 | −0.170 (6) | 0.08 | 0.11 | 0.22 | 0.34 |
| 0.7 | mid | 6.6 | 6.4 | +0.012 (2) | 72 | 59 | +0.087 (3) | 0.24 | 0.20 | 0.18 | 0.15 |
| 0.7 | high | 6.6 | 6.8 | −0.019 (3) | 92.5 | 47.6 | **+0.289 (14)** | 0.45 | 0.41 | 0.16 | 0.08 |
| 1.0 | low | 5.9 | 5.5 | +0.032 (4) | (share 0.00) | — | — | 0.00 | 0.05 | — | — |
| 1.0 | mid | 6.1 | 6.2 | −0.004 (1) | 82 | 81 | +0.003 (0) | 0.15 | 0.12 | 0.26 | 0.24 |
| 1.0 | high | 6.1 | 7.0 | **−0.063 (9)** | 102 | 42 | **+0.392 (22)** | 0.38 | 0.29 | 0.22 | 0.09 |
| 1.5 | low | 4.7 | 4.5 | +0.018 (3) | (share 0.00) | — | — | 0.00 | 0.00 | — | — |
| 1.5 | mid | 4.7 | 5.2 | −0.041 (6) | (share 0.00) | — | — | 0.00 | 0.03 | — | — |
| 1.5 | high | 4.9 | 6.1 | **−0.097 (15)** | 121 | 54 | +0.347 (14) | 0.25 | 0.11 | 0.39 | 0.17 |
| 2.0 | low | 4.1 | 3.9 | +0.017 (3) | (share 0.00) | — | — | 0.00 | 0.00 | — | — |
| 2.0 | mid | 4.1 | 4.1 | +0.004 (1) | (share 0.00) | — | — | 0.00 | 0.00 | — | — |
| 2.0 | high | 4.2 | 4.9 | **−0.064 (12)** | (share 0.00) | — | — | 0.00 | 0.02 | — | — |

(Tercile edges in log Mh at z = 0.4: 13.00 / 13.18 / 13.46 / 14.96; at z = 2:
11.62 / 12.57 / 12.82 / 14.07. "s.e." is the number of bootstrap standard
errors of the tercile median the difference amounts to — the "tercile scatter
of the deconvolution" of the plan's gate. The 16–84 per cent spreads are in the
log. The mh-complete subset gives the same numbers where it is populated,
e.g. z = 1 high: 102 vs 42 kpc.)

**What the data say, in words.**

1. **The compact mode is a fixed physical size at every halo mass**, 7.0 kpc
   at z = 0.4 falling to 4.1 kpc at z = 2 (slope −0.7 in log(1+z)), flat
   across the terciles to ±0.02 dex at every epoch. The baseline's compact
   mode grows with halo mass — 3.9 / 4.1 / 4.9 kpc at z = 2, 4.5 / 5.2 / 6.1
   at z = 1.5 — because the mass in it was laid down by the extended channel
   at s_e ∝ R200c(t'), and R200c grows with the halo's mass. The top
   tercile is 0.06–0.10 dex too large at z ≥ 1 (9–15 standard errors), the
   low tercile 0.02 dex too small. This IS the high-redshift size failure:
   at z = 2 there is no extended mode (share 0.00 in every tercile, in the
   data and in the model), so the R50 offset and the mass–size slope (0.19
   vs 0.11 on the merged grid; the deposit-size slope 0.048 vs 0.008) are the
   early deposits' halo-mass dependence, which the data do not have.
2. **The extended mode grows with halo mass and does not shrink with
   redshift.** Data: 62 / 71 / 89 kpc across the terciles at z = 0.4, and in
   the top tercile 89 → 93 → 102 → 121 kpc from z = 0.4 to 1.5; as a fraction
   of the halo radius at the epoch, 0.13–0.14 R200c(z_k) at z = 0.4 in every
   tercile, rising to 0.2–0.4 by z = 1–1.5. Baseline: 76 / 60 / 59 kpc — no
   growth with halo mass (it falls) and shrinking toward high z (59 → 48 →
   42 kpc in the top tercile), i.e. 0.18 / 0.12 / 0.08 R200c(z_k). The top
   tercile's extended mode is 0.18 / 0.29 / 0.39 dex too small at z = 0.4 /
   0.7 / 1.0 — the known z = 1–1.5 massive-progenitor and outskirts
   deficits, seen as a size.
3. **The shares agree.** The extended share is within 0.02–0.05 of the data
   in every tercile at z ≤ 0.7 (0.22 / 0.36 / 0.53 vs 0.24 / 0.34 / 0.51 at
   z = 0.4) and at most 0.09 low at z ≥ 1 in the top tercile. The split and
   the efficiency law are not where the size problem is.
4. **The same reading under the other two kernels.** Under the compact
   (n = 0.56) and n = 1 kernels the single high-z mode's size difference is
   −0.03 / −0.05 / −0.06 dex across the terciles at z = 2 (model larger in
   every tercile, most in the top) and the model's deposit-size slope
   against stellar mass exceeds the data's by 0.03–0.04 dex per dex; at
   z = 0.4 the extended mode in the top tercile is +0.04 to +0.11 dex larger
   in the data. The kernel moves the per-tercile numbers, not the pattern.

**What it implies for the law.** The early deposits (which are the whole
galaxy at z ≥ 1.5 and the 7 kpc mode at z = 0.4) should have a fixed physical
size, shrinking with the deposit redshift but NOT scaling with the halo; the
late deposits should scale with the halo — with the halo radius at the
observed epoch rather than at the deposit time, since the data's extended mode
holds at 0.13 R200c(z_k) across terciles and does not shrink in kpc toward
high z. The baseline's one law s_e ∝ R200c(t') does the opposite at both
ends. The plan's candidates map onto this as: (b) g_e = −1/3 makes EVERY
extended deposit mass-independent (fixes the early end, breaks the late
end's mass dependence); (a) R200c at the observed epoch fixes the late end
and worsens the early end (early deposits inherit R200c(z_obs)'s 0.2 dex
tercile spread); (d) early deposits at a fixed kpc size with late ones
halo-scaled is the reading itself (+3 parameters); (c) a second exponent
changes the time dependence only. Which of them recovers the z = 2 R50
offset at frozen amplitude is Stage 0 C's question.

## Stage 0 C — the candidates at frozen amplitude (`stage0_candidates.py`, `size_law.py`; no fit of the model)

**Method.** `size_law.py` generalises the baseline's size law with named
knobs that all nest at the baseline (`predict_law` reproduces
`model2.predict2` bit for bit at the defaults; asserted): the halo-mass
exponent g_e (candidate b, model2's own lever), the EXPANSION exponent q_e
(candidate a: an extended deposit made at t' is multiplied by
(R200c(t_obs) / R200c(t'))^q_e, so at q_e = 1 its size is set by the halo
radius at the observed epoch — the coordinate expands with the halo, the
mass does not move between deposits), a second (1+z') exponent above a
break (c), and the extended channel's early deposits at a fixed physical
size (d). Each candidate is applied to the baseline with the amplitude law,
the split and both kernel shapes FROZEN; its constants are set by hand from
the table and then tuned on the radius term alone (exp78's Z, raw, summed
in quadrature over the epochs) with everything else frozen. The CONTROL is
the baseline's own law re-tuned the same way (log_f_e and b_e free, no
knob). All 2356 galaxies, full quadrature, the standard QA on the standard
grid. Logs `outputs/stage0_cand_*.log`, table `outputs/stage0_candidates_merge.log`.

**The table** (fitting sample; "off" = the size gate's offset sub-gate at
fixed stellar mass, of 15; R50 offsets in dex; loss = A²+F²+S²+B² at the
frozen theta; M(<2) = the z = 0.4 centre, M(<10) at z = 2, both median
per cent):

| candidate | +par | Z sum | loss | off | R50 z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | R80 z=2 | M(<2) z=0.4 | M(<10) z=2 | M(<2) z=2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **baseline** | 0 | 0.0848 | 15.56 | 12 | −0.016 | −0.027 | −0.010 | +0.037 | +0.054 | +0.039 | +5.6 | −2.4 | −12.6 |
| control: baseline law re-tuned | 0 | 0.0697 | 15.86 | 14 | −0.006 | −0.024 | −0.014 | +0.021 | +0.028 | +0.008 | +12.0 | −0.8 | |
| (b) g_e free | 1 | 0.0697 | 15.87 | 14 | −0.006 | −0.024 | −0.014 | +0.021 | +0.029 | +0.009 | +11.9 | −0.8 | |
| (a) q_e = 1 fixed | 0 | 0.1397 | 19.16 | 8 | +0.014 | −0.025 | −0.032 | −0.012 | −0.005 | −0.064 | −39.5 | +2.6 | |
| **(a) q_e free → 0.127** | 1 | **0.0659** | **15.63** | **15** | +0.005 | −0.022 | −0.013 | +0.013 | **+0.020** | −0.006 | **+4.2** | +0.1 | **−5.0** |
| (a)+(b) q_e, g_e free → 0.122, −0.013 | 2 | 0.0661 | 15.65 | 15 | | | | | +0.020 | | | | |
| (d) early deposits fixed kpc | 3 | 0.0742 | 16.39 | 14 | −0.010 | −0.031 | −0.024 | +0.012 | +0.028 | −0.004 | +12.4 | +0.2 | |
| (c) break in b_e | 2 | 0.0705 | 15.97 | 14 | −0.005 | −0.024 | −0.015 | +0.019 | +0.027 | +0.004 | +12.2 | −0.5 | |
| (a)+(b) q_e = 1, g_e free | 1 | 0.1373 | 19.21 | 7 | +0.009 | −0.029 | −0.036 | −0.006 | +0.001 | −0.053 | −40.5 | +2.3 | |
| (a) q_e alone, constants kept | 1 | 0.0848 | 15.56 | 12 | (tunes back to q_e = 0) | | | | | | | | |

**Reading.**

1. **The plan's gate as written — more than 0.02 dex of the z = 2 R50
   offset recovered at frozen amplitude — cannot discriminate**: the control
   recovers 0.026 dex by re-tuning the two constants the model already
   has (log_f_e −0.72 → −0.55, b_e −1.03 → −1.40: later deposits larger,
   earlier ones smaller), and so does every candidate. What the control
   also shows is WHY the baseline is not there: the re-tune costs +0.30 in
   the loss and takes the z = 0.4 centre from +5.6 to +12 per cent. Under
   the current law the sizes and the centre trade against each other; the
   standard objective chose the centre.
2. **The halo-mass exponent (b), the break (c) and the early-fixed-size
   form (d) all tune back to the control**: g_e → −0.004, the break's
   second exponent buys 0.001 in Z for +0.11 in the loss, (d)'s switch
   redshift runs to its bound (switched off). The radius term does not
   want a halo-mass-independent early deposit at fixed z' — because the
   same exponent would remove the halo-mass dependence of the LATE
   deposits, which the data have (Stage 0 A/B, finding 2), and the term
   scores every epoch. My reading of the table (early deposits fixed in
   kpc) is right about the early end and cannot be written as a single
   exponent at fixed deposit time.
3. **The expansion exponent (a) at q_e = 1 is rejected**: it recovers the
   z = 2 R50 entirely, but by emptying the centre (the early deposits move
   out with the halo: M(<2 kpc) at z = 0.4 −40 per cent), the loss rises
   to 19.2 and the offset gate falls to 8 of 15. (The merge script's
   verdict column marks it PASS on the plan's letter; it fails everything
   else.)
4. **A small expansion, q_e ≈ 0.13, with the constants re-tuned, is the
   one change that buys the sizes without the trade-off.** All 15 offset
   entries pass (the first time on any mean model); R50 at z = 2 +0.054 →
   +0.020 and at z = 1.5 +0.037 → +0.013; R80 at z = 2 +0.039 → −0.006;
   the top-tercile R50 at z = 2 +0.098 → +0.053; the widths up slightly
   (R50 at z = 2 0.35 → 0.38); and the CENTRE IMPROVES: M(<2) at z = 0.4
   +5.6 → +4.2, at z = 1.5 −9.8 → −4.3, at z = 2 −12.6 → −5.0 per cent;
   M(<10) at z = 2 −2.4 → +0.1. The loss at the frozen theta is 15.63
   against the baseline's 15.56 (+0.07, against the control's +0.30): the
   binned term at z = 0.4 worsens (0.33 → 0.46: the z = 0.4 outskirts
   −3 → −5 per cent, M(<10) −0.3 → −4.1) and at z = 2 improves (0.71 →
   0.53). The future-dependence gate is unchanged (−0.061 → −0.061 at
   z = 1). Adding g_e on top changes nothing (→ −0.013). q_e alone, with
   the baseline's constants, tunes back to zero: the knob works together
   with a steeper b_e (earlier deposits smaller at deposit, growing
   afterwards), not by itself.

**Why q_e is the physical reading of the table.** The data's compact mode
shrinks from 7.0 kpc at z = 0.4 to 4.1 kpc at z = 2 at fixed halo mass,
and their extended mode holds at 0.13 R200c of the CURRENT halo. A deposit
that grows after deposition as a fraction of the halo's growth in radius
does both: early deposits are small when made (b_e steeper) and grow to
the few-kpc mode by z = 0.4 in every halo alike (a factor (R200c(0.4) /
R200c(5))^0.13 ≈ 1.2 for the median halo, weakly mass dependent), while
late deposits sit near the current halo's radius. It is a change of the
size COORDINATE, not a transport of mass: nothing moves between deposits,
the truncation radius grows by the same factor, and C stays 3 (exp79's
boundary untouched).

**Decision (Stage 0 C): proceed to Stage 1 with q_e as the thirteenth
parameter.** The plan's letter (0.034 dex recovered against the baseline,
above 0.02) is met; the stricter reading — beyond the control — is met
only in the sense that matters: the control's 0.026 dex is bought with a
centre the loss refuses, q_e's 0.034 with a centre better than the
baseline's at a loss within 0.07. Stage 1 asks the standard objective
whether it agrees.

## Stage 1 — the fit with q_e under the STANDARD objective (`stage1_fit.py`, `stage1_eval.py`)

**What was fitted.** The 13-parameter model — exp63's twelve plus the
expansion exponent q_e (bounds 0 to 1.5, nesting at 0 where the model IS
the baseline) — under exp63's A² + F² + S² + B² with the adopted references
on the measured input and the fitting sample; no size term. **13 fitted
parameters; an observer could vary all 13.** Five starts as separate
processes (3000-evaluation cap), then continuations of the three that
capped:

| start | loss | q_e | evaluations | note |
| --- | ---: | ---: | ---: | --- |
| tuned (the baseline with Stage 0 C's law constants, q_e = 0.127) | 15.63 → 14.77 → **14.664** | 0.64 → 0.43 | 3025 + 2983 | continuation; n_c railed at 0.5 |
| baseline (q_e = 0) | 15.56 → 14.73 → 14.668 | 0.55 → 0.44 | 3011 + 3403 | continuation; the same basin |
| far (q_e = 0.5) | 26.36 → 14.83 → 14.667 | 0.33 → 0.33 | 3011 + 3025 | continuation; n_c railed |
| the 14.63 basin (q_e = 0.127) | 14.89 → 14.633 | 0.016 | 2059 | converged: the basin sheds q_e |
| nested incumbent (q_e = 0.127) | 34.1 → 15.97 | 0.20 | 2927 | m_half, d_split railed |

The three continued starts agree to 0.004 in the loss and in every
parameter within 0.1 (log_f_c 1.11–1.22, b_c −1.32 to −1.46, log_f_e −0.83
to −0.84, b_e −0.49 to −0.62, m_half 11.7–12.0, d_split 0.98–1.15, n_c
0.50–0.51, c_e 1.00–1.04, q_e 0.33–0.44): ONE new basin at 14.66, below the
baseline's 15.56 and above the gate-rejected 14.63 basin, which keeps its
own loss with q_e → 0 (as it shed exp76's g). The loss's overall best is
still the 14.63 basin (C23); the model under judgement is the q_e basin
(the lowest continuation, `stage1_fit_start_cont_tuned.npz`), and the judge
also scores Stage 0 C's frozen point (the baseline with log_f_e −0.58, b_e
−1.43, q_e = 0.127; loss 15.63; no fit). Ops: the first two starts aborted
after 43 evaluations with the loss unchanged (a line-search trial at the
box corner, loss 10^24, finite); the loss is now capped at the failure
penalty and they were rerun (see `doc/lessons.md`).

### The gates (`outputs/stage1_eval.log`; figures `figures/qa/*exp80_*`)

| model | params | loss | offset (of 15) | widths R20/R50/R80 z=0.4 | R50 z=1.5 / 2 (dex) | R50 top tercile z=2 | R80 z=2 | slope z=2 (truth 0.11) | M(<2) z=0.4 / 1.5 / 2 | M(<103) z=0.4 | 50–100 kpc z=2 fit \| mh-c | leak z=1 |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| **baseline** | 12 | 15.56 | 12 | 0.69 / 0.66 / 0.60 | +0.037 / +0.054 | +0.098 | +0.039 | 0.33 | +5.6 / −9.8 / −12.6 | −3.1 | +29.6 \| −0.5 | −0.061 |
| 14.63 basin | 12 | 14.63 | 10 | 0.71 / 0.57 / 0.48 | +0.069 / +0.082 | +0.150 | +0.064 | 0.42 | +9.0 / −10.6 / −13.2 | −4.4 | +25.4 \| +13.7 | −0.066 |
| **exp80, the loss's q_e basin** | 13 | 14.66 | 12 | 0.56 / 0.48 / 0.48 | +0.047 / +0.054 | +0.111 | +0.040 | 0.40 | +4.6 / −10.5 / −11.7 | −4.7 | +16.3 \| −8.2 | −0.062 |
| **exp80, Stage 0 C's frozen point** | 13 (3 set by the radius term) | 15.63 | **15** | **0.78 / 0.76 / 0.67** | **+0.013 / +0.020** | **+0.053** | **−0.006** | **0.30** | +4.2 / **−4.3** / **−5.0** | −5.1 | +17.5 \| −9.2 | −0.061 |

(Widths = the model's size scatter at fixed stellar mass over the truth's;
the mean model fails every width entry as expected, C16. The tilt and the
future-dependence gate are within 0.005 dex per dex of the baseline for
every model. mh-complete profile: M(<10) at z = 1.5 +7.8 (baseline), +7.2
(q_e basin), +10.4 (frozen point); at z = 2 −0.5 / +1.5 / +3.0 per cent.)

**What the standard objective did with q_e.** Given the parameter, the loss
takes it to 0.43 and buys 0.9 in the loss — the per-galaxy shape term S at
z ≤ 1 (0.97 → 0.89) and the binned term at z = 0.7–1 — by making the
deposits expand strongly after deposition and re-arranging the rest
(d_split 0.27 → 1.1, b_e −1.03 → −0.6, c_e 0.80 → 1.0, the compact index to
its floor 0.5). None of it reaches the failure the change was chosen for:
R50 at z = 2 stays at +0.054 dex and the top tercile at +0.11, R20 and the
mass–size slope get worse (0.33 → 0.40), the widths NARROW (R50 at z = 0.4
0.66 → 0.48 of the truth's scatter: fitting harder narrows the population
again), the z = 0.4 outskirts lose (−3.1 → −4.7 per cent) and the outer
mass planes overshoot the other way (M(30–50) | M(<30) slope 1.57 against
the truth's 1.43; the baseline 1.33). It buys the z = 0.4 centre (+5.6 →
+4.6) and the z = 2 envelope on the fitting sample (+30 → +16) at the cost
of the mh-complete envelope (−0.5 → −8). Offset gate 12 of 15, the
baseline's. **Not adopted.** The seventh loss-versus-gates case: the loss
spends a size lever on shape, exactly as it spent exp78's size term.

**What the gates say about the frozen point.** The baseline with q_e =
0.127 and the two size constants re-tuned on the radius term — no fit of
the model — is the best gate reading of any mean model in the programme:
all 15 offset entries pass (the first time), R50 at z = 1.5 / 2 within
0.02 dex on the fitting sample (the plan's success line was 0.05, on both
samples — the mh-complete R50 is not separately scored by the gate; its
profile is in the table), the top-tercile z = 2 excess halved, R80 at z = 2
flat, the mass–size slope moved toward the truth (0.33 → 0.30), every
width entry UP (R50 at z = 0.4 0.66 → 0.76, at z = 2 0.35 → 0.38: a deposit
that grows with its halo's growth carries halo-history diversity into the
sizes), the z = 1.5 and z = 2 centres halved (−9.8 → −4.3, −12.6 → −5.0),
the future-dependence gate unchanged. Its costs: the z = 0.4 outskirts −5.1
per cent (baseline −3.1) and M(<10) at z = 0.4 −4.1; the mh-complete
M(<10) at z = 1.5 +10.4 (baseline +7.8); the loss 15.63 (+0.07). The
standard objective will not stay there: started from it, it walks to the
14.66 basin.

**Verdict.** The size-law change is real and the gates want it at q_e ≈
0.13 with the law's constants set by the sizes; the standard objective,
handed the same parameter, prefers q_e ≈ 0.43 with the high-redshift sizes
unfixed and the population narrower. Under the standing rule (the gates
decide, the loss does not) the loss's q_e basin is rejected and the
frozen point is the candidate — but it is not a fit of the standard
objective, so adopting it is the user's call, and the honest next step is
a fit with q_e FIXED at 0.127 under the standard objective (12 free
parameters), to see whether the objective keeps the sizes when it cannot
spend the lever, and a gated sweep of q_e (0.05–0.25) if it does not.

## What is owed after exp80

- The frozen-q_e fit (q_e = 0.127 fixed, 12 free) under the standard
  objective, and if the sizes survive, its adoption as the baseline mean
  by the gates; if not, a gated q_e sweep — the loss cannot choose q_e.
- The mass–size slope at z = 2 (0.30–0.40 vs 0.11) and the widths are the
  layer's, not the mean's; re-baseline the v1 stochastic layer on whichever
  mean is adopted.
- exp63 Stage 1's bimodality claim is qualified (Finding 1); the two-channel
  form is still warranted by the compact/extended SIZE behaviour, not by
  the deconvolution's modes.
- Not changed here: the truncation boundary C (exp79's).

## Files

- `stage0_deconvolve.py` — Stage 0 A/B; `outputs/stage0_deconvolve.{log,npz}`,
  `figures/exp80_stage0_w_stack.png`, `figures/exp80_stage0_sizes_vs_z.png`.
- `size_law.py` — the generalised size law (`predict_law`, `LAW_DEFAULT`,
  self-check `python size_law.py`).
- `stage0_candidates.py` — Stage 0 C (`--candidate NAME`, `--merge`);
  `outputs/stage0_cand_*.{log,npz}`, `outputs/stage0_candidates_merge.log`.
- `stage1_fit.py` — the fit (`--starts k:k`, `--continue NAME`, `--merge`,
  `--knobs q_e[,g_e]`); `outputs/stage1_fit_start_*.npz`, `stage1_fit.npz`, logs.
- `stage1_eval.py` — the judge (`--best NAME`, `--also NAMES`);
  `outputs/stage1_eval.{log,npz}` (final), `stage1_eval_interim.log` (the
  capped starts), `figures/qa/*exp80_*`.
