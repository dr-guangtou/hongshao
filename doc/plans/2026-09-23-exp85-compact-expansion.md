# exp85 — the centre's mechanism: an age-driven expansion of the compact channel

Status: IN PROGRESS 2026-09-23 (branch `exp85-compact-expansion` from master
`aae792d`). Roadmap: `doc/plans/2026-09-12-roadmap-evidence-rethink.md` §8
("next: the centre's mechanism"); the open question left by exp83
(`doc/open_questions.md`, "exp83 ... (a) and (c) stay open and are one
question: a mechanism that moves mass outward in early-formed galaxies
between z = 2 and z = 0.4"). The model under change is THE ADOPTED MEAN
(exp82: `rebaseline.adopted_mean()`, the delay tau_d = 0.15 held + q_e =
0.152 on exp63's twelve; 14 parameters, 13 fitted). The adopted layer
(exp84) is not touched; it is re-read on the new mean only if the mean is
adopted.

## The target (exp83 Stage 0, `experiments/exp83_early_mass_centre/outputs/stage0_probe.log`)

The adopted mean's centre at 4.9 kpc splits by the central decline. The
42 per cent of galaxies whose true M*(<4.9 kpc) fell from z = 2 to 0.4
(by −0.066 dex in the median) are modelled +0.058 too heavy at z = 0.4 and
−0.062 too light at z = 2 (the model's change +0.039); the non-decliners
are the mirror image (−0.054 / +0.051; the model's change +0.102 against
the truth's +0.160). The SPLIT of the central change between the two
groups is 0.226 dex in the truth and 0.063 in the model. At z = 0.4 the
residual at 5 kpc at fixed halo mass AND fixed M(<103) — the concentration
part — still correlates with the early-mass fraction (rho +0.14, slope
+0.059 dex per dex; the truth's own +0.150), and it lives in the stars
deposited before 2 Gyr, where every per-deposit term keyed to the 2 Gyr
mass is zero (exp83's finding). The early-formed haloes' centres are too
concentrated at z = 0.4 and too diffuse at z = 2 at fixed total.

## The mechanism (one knob, nesting at zero; a control form alongside)

A compact deposit made at t' is not a fixed structure: it expands after
deposition. Evaluated at the observed epoch t_obs its size is

    s_c(t', t_obs) = s_c(t') × (t_obs / t')^q_c              [`q_c`, the candidate]

the analogue of q_e for the in-situ stars, driven by the deposit's AGE
and nothing else (the user's reading of the decliners, 2026-08-26:
black-hole-feedback expansion decoupled from halo assembly). The oldest
deposits expand the most, and they are the centre of the early-formed
haloes; the same deposit is less expanded at z = 2 (t_obs 3.3 Gyr) than
at z = 0.4 (9.3 Gyr), which is a decline of the central mass at fixed
total. The control form is the exact analogue of q_e,

    s_c(t', t_obs) = s_c(t') × (R200c(t_obs) / R200c(t'))^q_ch    [`q_ch`, the control]

which couples the expansion to the halo's growth after the deposit; it
is probed for the record (if only the halo form worked, the reading
"decoupled from assembly" would be wrong). Both are `size_law` knobs
(`LAW_DEFAULT` carries them at 0); with either on, the compact kernel is
epoch dependent, evaluated per epoch like the extended one under q_e.
Mass is conserved (nothing moves between deposits); only the coordinate
expands, capped at the compact channel's truncation as in exp84.

## Protocol

0. **Stage 0, no fit** (`stage0_probe.py`, one process, foreground scale):
   the frozen-theta probe of q_c ∈ {0.1, 0.2, 0.3, 0.5, 0.8} and q_ch ∈
   {0.1, 0.2, 0.3, 0.5}. At every probe point the compact constants
   (log_f_c, b_c) are RE-TUNED so the fitting sample's median
   concentration log M(<4.9)/M(<103) at z = 0.4 AND at z = 2 is the
   adopted mean's (a 2-D root find; this is what the fit's own shape
   terms would do first, and it makes the probe read the REDISTRIBUTION
   of central mass between galaxies, not a uniform shift), then a0
   re-centred to hold the z = 0.4 median M(<103) (exact). The adopted
   mean is the control (the re-tune at q_c = 0 is the identity).
   GATE for a probe value:
   - G1 (the target): the split of the central change z = 2 → 0.4 between
     the decliners and the non-decliners moves at least half way from the
     adopted 0.063 to the truth's 0.226 (split ≥ 0.145), with BOTH groups'
     mismatches reduced.
   - G2: the concentration part of the z = 0.4 early-mass residual (5 kpc
     at fixed Mh and fixed M(<103)) halved: rho ≤ +0.07.
   - G3: the non-decliners' z = 2 centre median within 0.02 of the adopted
     +0.051 (the mechanism must act on the old deposits, not everywhere).
   Reported alongside: the early-mass correlations at 5 / 103 kpc at
   z = 0.4 and z = 2, the recent-growth correlation at z = 2 (must stay
   closed), the future-dependence gate at 103 kpc (within 0.03 of the
   truth), the centre M(<2) per cent per epoch, the tercile-median R20 /
   R50 / R80 offsets per epoch (the size gate's offset sub-gate at fixed
   halo mass, `size_terms.radius_term`; the re-tune must not break the
   z ≥ 1.5 sizes the delay bought), the standard loss at the fit nodes.
1. **Stage 1, only on a passing probe**: the fit under the standard
   objective (`stage1_fit.py --knobs q_e,q_c --delay 0.15 --fix
   tau_d=0.15 --start-adopted q_c=PROBE`), three starts from the adopted
   mean's fourteen with q_c at the probe value / 0 / twice the probe
   (`queue.sh 2 PROBE`: two fits at a time, 11.5 GB each, own sessions);
   continuations of capped starts; the merge; the judge BY NAME (`judge.sh
   --best NAME [--figures]`) against `adopted_mean()` and
   `adopted_baseline()`, the full battery, the residual check
   (`exp81/residual_check_model.py FILE --delay --knobs q_e,q_c`), the
   decliner split. Check that q_c MOVES from every start.
   PARAMETER COUNT: 15 in the model (exp63's twelve, tau_d held, q_e,
   q_c); 14 fitted.

## Decision rule

The gates decide, the loss does not (the CLAUDE.md rule). Adoption needs:
G1 and G2 on the fitted model; the size offsets and widths not below the
adopted mean's 14 / 1; the centre at every epoch not worse than
+0.5 / −6.2 / −6.9 per cent by more than the gate's resolution; the
future-dependence gate within 0.03 of the truth; the mh-complete 50–100 kpc
shell at z ≥ 1.5 not worse than −12 / −18 per cent. A fit whose loss falls
while a gate fails is another loss-vs-gates case and is reported as such.
If the mean is adopted, exp84's layer is re-scored on it (`qa.evaluate_draws`
on the draws) before the layer is carried over.
