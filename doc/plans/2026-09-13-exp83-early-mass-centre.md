# exp83 — the early-mass conditioning of the centre on the adopted mean (S3)

Status: STAGE 0 DONE 2026-09-13, no passing probe, Stage 1 not run (branch `exp83-early-mass-centre` from
master `0ddca75`). Roadmap: `doc/plans/2026-09-12-roadmap-evidence-rethink.md`
§3 S3 and §5. The model under change is THE ADOPTED MEAN (exp82:
`rebaseline.adopted_mean()`, the delay tau_d = 0.15 held + q_e = 0.152 on
exp63's twelve; 14 parameters, 13 fitted).

## The target (exp82's residual check, reproduced in `outputs/residual_check_adopted.log`)

The adopted mean's residual log10(model/truth) at z = 0.4 correlates with
the early-mass fraction x = log M(2 Gyr) − log Mh(z = 0.4) at fixed halo
mass: partial Spearman +0.37 at 5 kpc and +0.33 at 103 kpc (out-of-fold
R² 0.22 / 0.15) — the haloes that assembled early are now too heavy at low
redshift, at every radius. At z = 2 the same variable enters with the
opposite sign (−0.15) and the centre splits by the decline: the 42 per cent
of galaxies whose true M*(<4.9 kpc) fell from z = 2 to 0.4 (by −0.066 dex)
are modelled +0.058 at z = 0.4 and −0.062 at z = 2 (the model's change
+0.039); the non-decliners are the mirror image (−0.054 / +0.051).

## The term (one variable, three placements, nesting at zero)

For a deposit made at t' the conditioning variable is the fraction of the
halo's mass at t' that was already assembled by 2 Gyr,

    phi(t') = min(log M(2 Gyr) − log M(t'), 0)         [dex, ≤ 0]

(zero for every deposit before 2 Gyr; for the last deposit before an epoch
it is the per-galaxy x the residual is regressed on), taken RELATIVE to the
fitting sample's median at that time, phi(t') − phi_ref(t')
(`size_law.set_early_ref`, the same device as model2's `alpha_ref` for
`g_rel`): phi is monotone in time for every halo, so the raw variable is a
second time law whose population mean belongs to a_z; the relative variable
says how much earlier than the typical halo of that moment this one
assembled. The smoke probe with the raw variable confirmed the confound
(re-centred on z = 0.4, it dragged z = 2 down by 0.18 dex).

- `a_early` (the candidate): the deposit's stellar efficiency is multiplied
  by 10^(a_early (phi − phi_ref)). a_early < 0 makes deposits in haloes that
  grew more than the typical one since 2 Gyr MORE efficient, so the
  early-formed haloes get lighter at z = 0.4.
- `s_early` (the control placement): the same variable added to the logit
  of the compact share. A share cannot change M(<103) at z = 0.4, so it is
  expected to fail the 103 kpc half of the gate; it is probed for the record.
- `c_early` (added after the first grid, the roadmap's third candidate):
  the compact deposit's size multiplied by 10^(c_early (phi − phi_ref)),
  probed alone and combined with a_early = −0.3.

All three are `size_law` knobs (`LAW_DEFAULT` carries them at 0), so the adopted
engine, the fit script and the judge run unchanged with `--knobs q_e,a_early`.

## Protocol

0. **Stage 0, no fit** (`stage0_probe.py`, one process): the adopted
   model's residual against x at z = 0.4 (5, 103 kpc), z = 1 (103), z = 1.5
   and z = 2 (5 / 33 / 103): partial correlations, slopes next to the
   truth's, binned medians in quintiles of x per halo-mass tercile (the
   form), the decliner split per quintile; then the frozen-theta probe of
   a_early ∈ {−0.4, −0.3, −0.2, −0.15, −0.1, +0.1}, s_early ∈ {−2, +2},
   c_early ∈ {0.3, 0.6, 1.0} and a_early = −0.3 with c_early ∈ {0.5, 1.0},
   each re-centred by a0 to hold the z = 0.4 median M(<103) (exact: a0
   scales every deposit).
   GATE for a probe value: the z = 0.4 early-mass correlation halved at 5
   AND 103 kpc; the decliners' central change z = 2 → 0.4 closer to the
   truth's −0.066 than the adopted +0.039; the non-decliners' z = 2 centre
   median moved by at most 0.02 dex. Reported alongside: the future-
   dependence gate at 103 kpc (must stay within 0.03 of the truth), the
   recent-growth correlation (the delay closed it; it must stay closed),
   the centre M(<2) per cent, the standard loss at the fit nodes.
1. **Stage 1, only on a passing probe**: the fit under the standard
   objective (`stage1_fit.py --knobs q_e,a_early --delay 0.15 --fix
   tau_d=0.15 --start-adopted a_early=PROBE`), three starts from the adopted
   mean's fourteen with a_early at the probe value / 0 / twice the probe
   (`queue.sh 2 PROBE`: two fits at a time, 11.5 GB each, own sessions);
   continuations of capped starts; the merge; the judge BY NAME (`judge.sh
   --best NAME [--figures]`) against `adopted_mean()` and
   `adopted_baseline()`, the full battery, the residual check
   (`exp81/residual_check_model.py FILE --delay --knobs q_e,a_early`), the
   decliner split. Check that a_early MOVES (a frozen parameter with a
   gradient of zero is the exp82 tau_d lesson).
   PARAMETER COUNT: 15 in the model (exp63's twelve, tau_d held, q_e,
   a_early); 14 fitted.

## Decision rule

The gates decide, the loss does not (the CLAUDE.md rule). Adoption needs:
the z = 0.4 early-mass correlation halved on the fitted model, the size
offsets and widths not below the adopted mean's 14 / 1, the centre at every
epoch not worse than +0.5 / −6.2 / −6.9 per cent by more than the gate's
resolution, the future-dependence gate within 0.03 of the truth, the
non-decliners' z = 2 centre within 0.02 dex of the adopted mean's. A fit
whose loss falls while a gate fails is another loss-vs-gates case and is
reported as such.
