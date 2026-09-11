# Roadmap — an evidence-based rethink of the baseline mean model (2026-09-12)

Status: WRITTEN 2026-09-12 (branch `exp81-evidence-rethink`, from the exp80
branch). The measurements are in `experiments/exp81_evidence_rethink/`
(README, logs); the controlled tests reuse exp80's fit and judge with two
new options (`--fix`, `--delay`). Numbers marked (pending) are filled in at
the end of the session from the fits that were running while this was
written.

## 0. The question and the answer in one paragraph

The baseline (exp63's two-channel deposition model on the measured halo
history, 12 parameters) reproduces the average curves of growth of massive
galaxies at every epoch to a few per cent, yet every gate table shows the
same systematic residuals: the centre too light at z ≥ 1.5, the sizes too
large at high redshift in the most massive third, the z = 2 outskirts too
heavy per galaxy, and every size distribution too narrow. Six experiments
attacked these through the objective (exp72, 73, 77, 78) or through single
levers (exp76, 80) and the loss rejected each gate-improving change. The
measurements below say why and what to do instead: **the model's error is
almost entirely per-galaxy scatter, and in the centre at z ≤ 1 that scatter
is at the information limit of the halo history; but in the outskirts at
every epoch and at every radius at z ≥ 1.5, a quarter to a third of the
residual variance is still predictable from the halo's PAST assembly — and
the specific information the model ignores is the halo's formation time.
The truth has a strong assembly bias of the stellar-mass–halo-mass relation
(early-formed haloes hold more stars at fixed mass, ρ = +0.4 to +0.6); the
baseline deposits stars from accreted mass instantly, so its residual runs
the opposite way (ρ = +0.5 with the mass gained in the last Gyr).** One
physical parameter — a deposition delay for accreted mass — removes that
correlation at frozen amplitude; the exp80 expansion exponent handles the
sizes and half the central decline; the widths belong to the stochastic
layer. The roadmap orders these by evidence.

## 1. The evidence (exp81, measurements 1–3; exp80; the record)

**E1 — the error budget** (`residual_budget.log` §1). Residual
log10(model/truth) of M*(<R), fitting sample, per epoch and radius: the
systematic share of the mean square is 0–7 per cent in every cell. The
per-galaxy 16–84 half-width is 0.17–0.20 dex at 2 kpc and 0.11–0.17 dex at
33–148 kpc (the latter is the stellar-mass–halo-mass scatter itself, 0.11
dex at z = 0.4 rising to 0.16 at z = 2). The only medians beyond 0.03 dex
are the centre at z ≥ 1.5 (−0.045 / −0.070 dex at 2 / 5 kpc at z = 1.5;
−0.059 / −0.067 at z = 2). Amplitude-pinned (shape) residuals are within
0.03 dex everywhere except the same centre cells.

**E2 — the halo-predictability ceiling** (§2). A gradient-boosted
regressor from the measured history (log M at 12 times, the growth rate and
the mass at the epoch), 5-fold out-of-fold, on the baseline's residual:
R² = 0.03–0.10 at 2–10 kpc for z ≤ 1 (the model is at the ceiling there:
0.17 dex of central scatter is intrinsic to a halo-only input); R² =
0.22–0.27 at 100–148 kpc at z ≤ 1; R² = 0.23–0.35 at every radius ≥ 5 kpc
at z ≥ 1.5 (rms 0.167 → 0.135 dex at z = 2, 103 kpc). Past-only and
full-history features give the same R²: the unused information is in the
past, not a future leak. The truth's own halo-only R² is 0.54 (2 kpc) to
0.84 (148 kpc) at z = 0.4; the baseline reaches 0.51 to 0.78. exp75's
closed correction map measured the same thing globally (8.4 per cent of the
held-out rms).

**E3 — which information** (`residual_features.log`). At fixed halo mass
the truth's log M*(<R) correlates with the mass at 1–2 Gyr (+0.36 to +0.63),
with the mass gained in the last Gyr (−0.26 to −0.65) and with t50 (−0.47 to
−0.64) at every epoch and radius: early-formed haloes hold more stars. The
baseline's residual has none of this in the centre at z ≤ 1 (|ρ| ≤ 0.10)
but +0.27 to +0.58 with recent growth in the outskirts at z ≤ 1 and at every
radius at z ≥ 1.5 (and −0.3 to −0.6 with the early mass): the model
over-predicts haloes that just grew and under-predicts early-formed ones.
Restricted feature sets: at z = 2 the early history alone gives R² 0.25–0.31,
the recent history 0.18–0.29, the mass at the epoch alone 0.00; at z ≤ 1 the
outskirts need the mid-history.

**E3b — the functional form** (`residual_shape.log`). Binned at fixed halo
mass, the residual is monotonic and close to linear in the log of the
recent growth: at z = 2 from −0.09 dex (haloes that gained nothing in their
last Gyr) to +0.22 dex (haloes that gained 0.5–1.2 dex), slope +0.45 dex per
dex at every radius; +0.63 at z = 1 and +0.45 at z = 0.4 in the outskirts.
The truth's slope is −0.55 to −0.84 dex per dex. The model's stars follow
the halo instantly; the data's lag it. Against the early-mass fraction the
residual runs the other way (−0.4 dex per dex at z = 2), the same fact.

**E4 — the deposition delay at frozen theta** (`delay_probe.log`).
exp63's Stage 2b lever (tau_d: a satellite's stars join the central tau_d
Hubble times after accretion; mass in transit is not deposited) applied to
the baseline: at tau_d = 0.15 the residual's recent-growth correlation goes
from +0.27 / +0.50 / +0.52 to −0.01 / +0.05 / +0.01 (z = 0.4 / 1.0 / 1.5,
outskirts) and from +0.55 to +0.24 at z = 2; the z = 2 50–100 kpc shell
from +34 to −6 per cent. Costs at frozen theta are the ones a refit absorbs:
the amplitude (−12 to −24 per cent at 100 kpc: mass in transit), the sizes
(R50 −0.09 dex), the pinned centre (+0.07 dex). exp63 rejected tau_d on the
OFFICIAL curves at z = 0.4 (the leaky input), never under the measured
history. Fit under the standard objective: (pending, §3 S1).

**E5 — the size law** (exp80). Deconvolution reads early deposits at a fixed
physical size and late ones at 0.13 R200c of the current halo; the
expansion exponent q_e = 0.127 with the size constants re-tuned at frozen
amplitude passes all 15 size-offset entries (first ever), halves the z ≥ 1.5
centre errors, raises every width, at a loss +0.07; the standard objective
takes q_e to 0.43 and spends it on shape at z ≤ 1 (C25). Fixed-q_e fits:
(pending, §3 S2).

**E6 — the central decline** (§3 of `residual_budget.log`). 42 per cent of
the galaxies lose central mass (−0.066 dex at 4.9 kpc, z = 2 → 0.4); the
baseline grows them (+0.061), the q_e point halves the mismatch (+0.034)
but moves the non-decliners the wrong way (+0.058 → +0.073 at z = 2): the
expansion acts on every galaxy, the decline is a subpopulation. The z = 2
centre residual is 28 per cent halo-predictable (E2), from the early
history (E3): the compact, early-formed progenitors (memory
`compact-highz-progenitors-formed-early`).

**E7 — the widths.** Every mean model's size scatter at fixed mass is
0.3–0.7 of the truth's (0 of 15 width entries, C16). E1/E2 say the
per-galaxy scatter in the centre is intrinsic to the halo input: no mean
model can supply it. The v1 stochastic layer (exp60) was never re-baselined
on the measured history and lacks an outer size component.

**E8 — the record** (the survey in `experiments/exp81_evidence_rethink/README.md`).
Roads not taken: exp55's early-exponential / late-Moffat deposition mixture
(C3, never executed); the model-class decision on the decline (C8); the
layer's outer size component (C16); what measured quantity the compact
share follows (C22); the truncation scan (B2). Closed by evidence: direct
CoG maps at statistical parity with PCA (exp50/51), symbolic densities
(exp65–70, nulls), the annular loss (exp77), a residual correction map
(exp75, 8.4 per cent, closed by decision).

## 2. What this rules out

- **Another objective.** The residual is per-galaxy scatter with no halo
  information in the centre at z ≤ 1 (E2): re-weighting cannot create it,
  and the seven loss-vs-gates cases were all fights over 0.03-dex medians
  inside a 0.17-dex scatter. Objective work is closed until the mean model
  carries the formation-time dependence (E3).
- **A direct or machine-learned map as the mean.** The ceiling (E2) bounds
  its gain at R² ≤ 0.10 in the centre and ≤ 0.35 at high z / outskirts —
  exactly the part a physical delay reaches (E4) with one parameter, and
  exp50/51 measured parity with PCA. The emulator exists for the
  generative product. Not the mean.
- **Redesigning the deposition class.** E1 says the class reproduces the
  average curve to a few per cent at every radius and epoch; the failures
  are conditional (on formation time, on halo mass at high z, on decline).
  The class is kept; its conditioning is extended.

## 3. The strategies, ranked

### S1. A deposition delay for accreted mass (the in-transit term) — do first
1. **Motivation.** Stars of an accreted satellite join the central a
   dynamical-friction time after the halo accreted it; the model credits
   them instantly. This is the one physical mechanism that makes a halo's
   stellar mass at fixed halo mass depend on WHEN the mass came — the
   assembly bias the truth has and the model lacks (E3).
2. **Evidence.** E3 (ρ = +0.5 residual with recent growth, −0.6 in the
   truth); E4 (tau_d = 0.15 zeroes it at z ≤ 1.5 and fixes the z = 2
   envelope at frozen theta); E2 (the recoverable variance is in the past
   history, largest at z ≥ 1.5 and in the outskirts — where the delay
   acts). Physically standard; one parameter; the compact (in-situ)
   channel is undelayed by construction.
3. **Feasibility.** Implemented (`model2.Spec2.with_delay`,
   `stage1_fit.py --delay`); 13 parameters; ~1 h per fit. Risk: exp63's
   Stage 2b rejected tau_d = 0.3 — but that was a fit at z = 0.4 ONLY on
   the official curves, in which the optimiser used the delay to keep
   late-accreted stars outside the 148 kpc aperture and the unfitted
   epochs then collapsed (M*(50–100) −97 per cent at z = 2). The fits here
   are joint over the five epochs on the measured history, so that
   failure mode cannot occur; the probe's value (0.15 Hubble times, not
   0.3) is what the z = 2 curves allow. The future-dependence gate must
   be re-read (the delay reads only the past, so it should pass).
4. **Plan.** Three fits: delay alone, delay with q_e = 0.127 fixed, delay
   with q_e free, from the baseline; judge on the gates (offset and width,
   both samples, the leak gate, the planes); adopt if the gates improve
   with the leak gate held; then the fitted-tau_d form.

   **Result (2026-09-12, tau_d held at 0.15 — model2's step arrival has no
   gradient in tau_d, so these fits did not move it; the exponential-arrival
   form that does is in `size_law.arrival_weights` and its fits follow).**
   Loss 12.49 (delay alone, 13 parameters), 12.47 (delay + q_e fixed),
   12.72 (delay + q_e free, 14 parameters) against the baseline's 15.56
   and the old loss-best basin's 14.63 — the first change in the
   programme that the LOSS and the GATES both prefer:

   | model | params | loss | offset / width (of 15) | R50 z=1.5 / 2 | R20 z=2 | R80 z=2 | slope z=2 (truth 0.11) | M(<2) z=0.4 / 1.5 / 2 | M(<103) z=1.5 mh-c | 50–100 kpc z=2 fit \| mh-c |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
   | baseline | 12 | 15.56 | 12 / 0 | +0.037 / +0.054 | +0.092 | +0.039 | 0.33 | +5.6 / −9.8 / −12.6 | +8.6 | +29.6 \| −0.5 |
   | delay alone (tau_d 0.15) | 13 | 12.49 | 13 / 0 | +0.045 / +0.046 | +0.073 | −0.007 | 0.33 | +9.5 / −9.3 / −11.6 | +3.7 | −4.6 \| −25.5 |
   | delay + q_e = 0.127 | 13 | 12.47 | 13 / 0 | +0.045 / +0.046 | — | — | 0.32 | +8.6 / −9.0 / −11.9 | +3.9 | −4.1 \| −25.4 |
   | **delay + q_e free (→ 0.125)** | 14 | **12.72** | **14 / 2** | **+0.018 / +0.030** | +0.054 | −0.019 | **0.20** | **+0.9 / −6.7 / −7.3** | +2.9 | +6.8 \| −19.4 |

   With the delay in place the loss no longer abuses q_e: it settles at
   0.125 — the gate-chosen value of exp80 — with the baseline's structure
   kept (d_split 0.22, n_c 0.66; compare the q_e-only fit's 1.07 / 0.50).
   The delay + q_e model passes 14 of 15 offset entries and 2 width entries
   (R20 at z = 0.4 and 0.7: 0.84 and 0.81 of the truth's scatter — the
   first width passes on any mean model), halves the z = 2 mass–size
   slope's excess, and improves the centre at EVERY epoch. Costs: the
   50–100 kpc shell of the mh-complete progenitors at z ≥ 1.5 is 14–19 per
   cent light (mass moved inward; the cumulative M(<103) there is +2.9 vs
   the baseline's +8.6, better), M(<10) at z = 1 +5.5 (baseline +3.7), and
   the future-dependence gate at z = 0.7 and z = 2 moves from −0.013 /
   −0.007 to −0.043 / −0.027 dex per dex (the truth −0.005 / +0.004) while
   improving at z = 1–1.5 (−0.061 / −0.038 → −0.039 / −0.015). The
   formation-time residual check on the fitted model: (pending, measurement 5).

### S2. The post-deposition expansion at the gate-chosen value (exp80's q_e)
1. **Motivation.** The data's early deposits are a fixed physical size and
   the late ones follow the current halo radius; a deposit that grows with
   its halo's radius reproduces both and emulates half the central decline.
2. **Evidence.** E5, E6; 15 of 15 offsets and every width up at q_e =
   0.127; the loss's own q_e (0.43) is rejected by the gates.
3. **Feasibility.** Implemented; the fixed-q_e fits (0.08 / 0.127 / 0.20)
   are running; ~1 h each.
4. **Plan.** Judge the three; if the sizes survive the refit at a fixed
   q_e, adopt that mean by the gates (a gate-selected q_e, not a
   loss-selected one — C25) and combine with S1.
   **Result (2026-09-12): the sizes do NOT survive.** At q_e = 0.08 /
   0.127 / 0.20 fixed, the twelve free parameters go to the 14.63 basin's
   family in every case (compact size 18–19 kpc with its index railed at
   0.5, d_split ≈ 1.0): losses 14.79 / 14.71 / 14.69, offset gate 10 of 15
   (the baseline 12), R50 at z = 2 unchanged, the z = 2 centre −12 per
   cent. Holding q_e alone holds nothing; the size constants must be held
   with it (the S0C point holds log_f_e and b_e at the size-tuned values).
   S2 is therefore only reachable through S5's two-block selection, or
   through S1, which changes what the loss wants (below).

### S3. Formation-history conditioning of the centre at z ≥ 1.5 (the decline's other half)
1. **Motivation.** The z ≥ 1.5 centre is the one place with a real median
   bias (−0.06 dex) AND 28 per cent halo-predictability from the early
   history; q_e acts on everyone, the decline is a subpopulation of
   early-formed, compact progenitors.
2. **Evidence.** E1, E2, E3 (z = 2, 5 kpc: early-history R² 0.25),
   `compact-highz-progenitors-formed-early` (0.3 dex head start by 1 Gyr).
   exp59's disqualification of per-deposit conditioning was on the leaky
   input with the old engine.
3. **Feasibility.** Moderate: a one-parameter form is needed (candidates:
   the compact channel's share or the efficiency rising with the mass
   fraction assembled by 2 Gyr; or the compact size shrinking with it);
   frozen-theta probes first (an afternoon), then a fit.
4. **Plan.** Regress the z = 2 centre residual on the early-mass fraction
   at fixed halo mass to read the functional form; implement as a knob in
   `size_law.py` / `model2.compact_share`; frozen probe on the decliner
   split (E6's table); fit only if the probe moves the decliners without
   moving the rest. After S1 + S2, since both change the centre.

### S4. The stochastic layer, re-baselined, with an outer size component
1. **Motivation.** The widths (0 of 15) are the largest remaining gate
   block, and E1/E2 prove no mean model can supply the 0.17 dex of central
   scatter: it is intrinsic to a halo-only input.
2. **Evidence.** C16; exp60's layer (R50 width 1.14 → 0.73 across z, R20
   over-dispersed, R80 under by half); the memory
   `stochastic-layer-size-gate-verdict` names the missing outer component.
3. **Feasibility.** Half a day to re-run exp60's stages on a predictor that
   takes the measured history; the size gate already scores draws
   (`exp73/size_gate_layer.py`).
4. **Plan.** After the mean is chosen (S1/S2): re-baseline the layer on it;
   add a second size draw for the extended component; gate at R20/R50/R80
   with offset and width separate; make tier 2e read the draws.

### S5. Gate-consistent model selection (a procedure, not a model)
1. **Motivation.** Seven times the loss preferred a point the gates reject;
   the rule "the gates decide" has only ever been applied to reject.
2. **Evidence.** C23, C25, exp78; E1 shows the disputed quantities are
   0.03-dex medians inside 0.17-dex scatter — the loss cannot see them.
3. **Feasibility.** High: the machinery exists (Stage 0 C's frozen tune,
   the judge). Two blocks: the amplitude / split / kernels by the loss, the
   size-law constants and q_e by the radius term at frozen amplitude,
   alternated to a fixed point.
4. **Plan.** Implement as a wrapper around `stage1_fit.py`; run once S1/S2
   have settled the model; adopt the fixed point by the gates.

### S6. Owed checks that do not change the model
The truncation scan (B2, C = 1–5, coordinate with exp79); the compact
share's measured driver (C22); the jumper snapshot check (±4 per cent on the
z = 2 amplitude); variant B2 (measured-history smoothing). None is a
strategy; each is a half-day and bounds a systematic.

## 4. Sequence

1. (now) S1 and S2 fits → judge → adopt the best gate reading as the mean.
2. S3 probe on that mean; fit only on a passing probe.
3. S5 as the way the next mean is selected.
4. S4 the layer on the adopted mean.
5. S6 in the gaps.
