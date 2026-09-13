# exp82 — the deposition delay and the expansion exponent together, adopted by the gates

Status: RUNNING from 2026-09-12 (branch `exp82-delay-expansion` from master
`ed28a45`). Roadmap: `doc/plans/2026-09-12-roadmap-evidence-rethink.md`
(S1 + S2; agreed sequence in its §5). Evidence: exp81 (the residual's
formation-time dependence; the delay at frozen theta; the single-start
controlled fits: delay at tau_d = 0.15 with q_e free → loss 12.72, 14 of 15
offsets, 2 widths).

## The model (14 parameters, 13 fitted per grid point)

exp63's twelve, plus exp63 Stage 2b's deposition delay tau_d (the extended
channel's stars arrive tau_d Hubble times after the halo accreted their
satellite; mass in transit is not deposited; the compact, in-situ channel is
undelayed) and exp80's expansion exponent q_e (a deposit's size grows after
deposition by the fraction q_e of its halo's growth in radius; the
truncation radius grows with it; C = 3 unchanged). The input is the
measured halo history; the fitting sample; the adopted references.

## Why together, and why a grid in tau_d

exp81: q_e fixed alone drifts the twelve into the gate-rejected wide-compact
family (10 of 15); the delay alone fixes the formation-time residual but not
the sizes; the two fitted together give the first model the loss and the
gates both prefer, with q_e settling at the gate-chosen value on its own.
model2's step arrival has no gradient in tau_d, and the smooth exponential
arrival with tau_d free drifts the twelve into the wide-compact family
again (loss 12.19, offsets 13, no widths). So tau_d is HELD on a grid and
chosen by the gates (C26a; S5-lite), the rest fitted under the standard
objective.

## Protocol

1. Round 1: tau_d ∈ {0.10, 0.15} (step arrival) × starts {tuned (Stage 0
   C's size constants, q_e = 0.127), baseline (q_e = 0), far (q_e = 0.5)},
   q_e free, 3000-evaluation cap, six processes under nohup + caffeinate.
2. Round 2: tau_d ∈ {0.20, 0.30}, the same three starts.
3. Continuations of every capped start that is within 0.1 of its grid
   point's best; the merge per grid point; the judge (tables) per grid
   point, the model under judgement named explicitly.
4. The gate-chosen tau_d: the grid point with the best offset + width count
   whose leak gate is within 0.03 dex per dex of the truth's at every
   epoch and whose z ≤ 1 R50 offsets stay within 0.05; ties by the loss.
5. Round 3: the exponential arrival at the chosen tau_d (held), the same
   three starts — the physical form, judged against the step form.
6. The full judge with figures on the chosen model (both samples, the mass
   planes, the size gate with offset and width separate, the leak gate),
   the residual check (`exp81/residual_check_model.py`), the arrival-radius
   check for the mh-complete progenitors' 50–100 kpc shell at z ≥ 1.5
   (C26b), and the decliner split.
7. Adoption by the user: the baseline mean becomes this model
   (`rebaseline.adopted_baseline()` re-pointed; the CLAUDE.md rule updated);
   then exp83 (S3), exp84 (S4).

Success: better than the baseline on the offset count (12) AND the width
count (0) AND the centre at every epoch, with the leak gate within 0.03
dex per dex of the truth; the mh-complete profile within the baseline's at
10 and 100 kpc. State the parameter count (14, 13 fitted) everywhere.

## Cost

Twelve fits of ~1 h in two rounds of six, three more in round 3, judges of
~10 min each; about 4 h wall clock.
