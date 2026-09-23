# exp86 — where the delayed mass lands: sizing the delayed deposits by the halo at ARRIVAL

Status: IN PROGRESS 2026-09-23 (branch `exp86-arrival-radius` from master
`7215df2`). Source: open question C26(b) (`doc/open_questions.md`, exp81/82:
"the mh-complete progenitors' 50–100 kpc shell at z ≥ 1.5 is 14–19 per cent
light while their cumulative M(<103) improves — is the delayed mass arriving
at the wrong radius, or is this the completeness selection (C18)?"); the
arrival-radius check listed in exp82's plan and never run; the 2026-09-23
review of the record (the user chose this direction). The model under
change is THE ADOPTED MEAN (exp82: `rebaseline.adopted_mean()`, tau_d =
0.15 held, q_e = 0.152; 14 parameters, 13 fitted). The layer (exp84) is
untouched.

## Why this region

exp81's ceiling says a halo-conditioned mean model can still gain only in
the outskirts at z ≤ 1 and at every radius at z ≥ 1.5; the z ≤ 1 centre is
at the halo-information limit. The adopted mean's largest carried cost sits
exactly there: the halo-mass-complete progenitors' 50–100 kpc shell at
z = 1.5 / 2 is −12 / −18 per cent (the fitting sample's +6 / +8; the
cumulative M(<103) on the complete sample +3.4 / −1.4).

## The mechanism (one knob, nesting at zero)

The delay holds an accreted deposit in transit for tau_d Hubble times
(step arrival: it counts at epoch k once t' + tau_d / H(z') ≤ t_k), but the
deposit is still SIZED by the halo at accretion,

    s_e = 10^log_f_e (1+z')^b_e R200c(t') × (R200c(t_obs) / R200c(t'))^q_e .

Physically the stars settle when they arrive, in the halo as it is then.
The knob `w_arr` ∈ [0, 1] moves the size reference from accretion to
arrival: with t_a = t' + tau_d / H(z'),

    log R_ref = (1 − w_arr) log R200c(t') + w_arr log R200c(t_a)
    log(1+z)_ref = (1 − w_arr) log(1+z') + w_arr log(1+z_a)

and s_e = 10^log_f_e (1+z)_ref^b_e R_ref × (R200c(t_obs) / R_ref)^q_e; the
extended deposit's truncation radius follows R_ref (the coordinate at
arrival). w_arr = 0 is the adopted model bit for bit; w_arr = 1 is full
arrival sizing. Mass is conserved; the compact channel is untouched. The
factor R200c(t_a) / R200c(t') is larger for haloes that grow more during
the delay, so the change is DIFFERENTIAL by growth history — which is what
separating the complete progenitors' shell (light) from the fitting
sample's (heavy) needs; a uniform enlargement of the extended deposits
cannot do it (the two samples' shell residuals have opposite signs).

## Protocol

0. **Stage 0, no fit** (`stage0_probe.py`, one process, foreground scale).
   Part 1, the anatomy BEFORE the probe (exp85's lesson): on the adopted
   mean, at z = 1.5 and z = 2, the 50–100 kpc shell's model mass by channel
   and by sample (fitting | mh-complete), and the mass-weighted
   log R200c(t_a) / R200c(t') of the extended deposits in the shell, per
   galaxy, split by sample — the premise: the complete progenitors' delayed
   deposits must be sized up MORE than the rest's, or the knob acts on the
   wrong galaxies.
   Part 2, the frozen probe of w_arr ∈ {0.25, 0.5, 0.75, 1.0}, and the
   CONTROL w_arr = 0, each with (log_f_e, b_e) RE-TUNED on the radius term
   (exp80 Stage 0 C's device: the rms over R20 / R50 / R80 × three
   halo-mass terciles × five epochs at fixed halo mass, everything else
   frozen; a candidate is credited only with what it buys beyond the
   control), then a0 re-centred on the z = 0.4 median M(<103).
   GATE for a probe value, all against the CONTROL:
   - G1 (the target): the mh-complete 50–100 kpc shell deficit at z = 1.5
     AND z = 2 at least halved, with the fitting sample's shell bias not
     moved away from zero by more than 3 points.
   - G2: the size-offset cells within 0.05 dex (R20 / R50 / R80 × 5 epochs,
     fixed halo mass) not fewer than the control's.
   - G3: the future-dependence gate at 103 kpc at no epoch farther from the
     truth than the control's by more than 0.01 dex per dex (the adopted
     mean is already 0.03–0.08 off at z = 0.7–1, a carried cost), and the
     recent-growth correlation of the z = 2 residual not above 0.15 in
     magnitude (the delay closed it to 0.13).
   Reported: M(<10) and M(<103) on both samples, the centre M(<2) per
   epoch, the standard loss at the fit nodes, the re-tuned constants.
1. **Stage 1, only on a passing probe**: the fit under the standard
   objective (`stage1_fit.py --knobs q_e,w_arr --delay 0.15 --fix
   tau_d=0.15 --start-adopted w_arr=PROBE`; three starts probe / 0 / twice
   the probe clipped to 1; `queue.sh 2 PROBE`, two at a time), the merge,
   the judge BY NAME (`judge.sh --best NAME --figures`) against
   `adopted_mean()` and `adopted_baseline()`, the residual check, the
   decliner split for the record. Check that w_arr MOVES from every start.
   PARAMETER COUNT: 15 in the model (exp63's twelve, tau_d held, q_e,
   w_arr); 14 fitted.

## Decision rule

The gates decide, the loss does not. Adoption needs G1–G3 on the fitted
model; the size offsets and widths not below the adopted mean's 14 / 1;
the centre at every epoch within the gate's resolution of +0.5 / −6.2 /
−6.9 per cent; the loss reported either way. If adopted, exp84's layer is
re-scored on the new mean (`qa.evaluate_draws`) before it is carried over.
If the anatomy in part 1 shows no differential (the complete progenitors'
delayed deposits are not sized up more than the rest's), the probe is still
run for the record but the expected verdict is stated before it.
