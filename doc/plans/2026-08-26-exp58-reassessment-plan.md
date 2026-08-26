# The reassessment plan — where the deposition-only model goes next

*Written 2026-08-26, after exp58's measurements (P-A through P-D) and before
any fit. The user asked for a careful review of the deposition-only model —
not another mechanism — and for a fresh angle on (1) the central region inside
10 kpc and (2) the performance at redshift above 1.5. This plan is that
review's output. Every claim in Section 1 has a measurement behind it in
`experiments/exp58_reassessment/README.md`; nothing below was designed after
seeing a fit, because no fit has been run.*

---

## 1. The reassessment in six sentences

1. **The foundation is sound.** Like for like — both models predicting their
   own amplitude — the old deposition+transport kernel leads the incumbent by
   about 3% on the QA headline and is *behind* `X3` small-core in pure radial
   shape at four of five epochs (P-A). "Systematically falling behind" was an
   artifact of comparing against an oracle-pinned amplitude, and is retired.
2. **The z=2 problem is one number per galaxy, not a profile problem.** The
   error is flat in radius beyond 10 kpc (−0.045 dex median on the fit
   sample); with the amplitude divided out the z=2 shape beyond 10 kpc is
   good to 0.023 dex (P-B).
3. **That number is predictable, and the model is forbidden to predict it.**
   The z=2 amplitude residual correlates, at fixed halo mass and on the
   masked fit sample, with the centre-decline flag (r = −0.45), concentration
   excess (−0.38) and formation time (+0.29) — quantities the efficiency law
   `log_eps(m, z)` never sees. The out-of-fold ceiling test confirms the
   information is real (shuffled control flat) and worth roughly a fifth of
   the z=2 amplitude error (P-C).
4. **The z=2 deficit and the central defect are the same phenomenon.** For
   the 42% of galaxies whose measured centre declines, a monotone model's two
   endpoint errors must sum to at least the decline (median 0.062 dex); the
   incumbent's actual sum is 0.194 dex, placed almost entirely at z=2 because
   `SIGMA_A(z=2)` = 0.161 dex makes z=2 errors cost ~0.6% of the loss. The
   model gives decliners and non-decliners the same central growth (+0.083 vs
   +0.104 dex) where the data splits them (−0.062 vs +0.100) (P-B, P-D).
5. **Two complementary fixes, not one.** With a perfect amplitude the z=2
   core is still −9.4% at `M*(<10 kpc)`: the amplitude law can fix the z=2
   outskirts, only the (already-gated) core-only expansion `X3` can fix the
   z=2 core. Conversely no expansion fixes a radius-independent −0.045 dex.
6. **The loss cannot referee any of this.** z=2 is the loss's *best* epoch
   (score_A 0.94, score_F 0.78) while being the user's visibly worst; the
   backwards ranking the programme has now recorded four times is structural
   here, not incidental. Gates must be written before the next fit.

Two standing scope decisions fall out of the same measurements:

- **The population's width is not the mean law's job.** The model's central-
  growth spread is 0.33 times the data's (exp57 Stage 1), the two decline
  groups need opposite-signed corrections (P-D), and the conditioned ceiling
  recovers only a fifth of the amplitude scatter (P-C). The mean model should
  be scored on conditional medians; distribution tiers belong to the
  cross-epoch-correlated draw layer exp54's rebaseline already built.
- **z=2 must be quoted on the mh-complete subsample.** The fit-sample bias
  (−0.046 dex) and full-QA-sample bias (−0.029 dex) genuinely differ, and a
  law trained on the first overshoots the second by construction (P-C rung 4).
  The battery keeps the full-sample view, but gains a complete-sample column
  at z ≥ 1.5.

## 2. Stage 0 — the bias-sighted gate battery (no fit; half a day)

Written and frozen BEFORE any Stage 1+ fit, run at the null (incumbent) first,
thresholds below fixed now:

| gate | quantity | sample | threshold | incumbent today |
|---|---|---|---|---|
| B1 | per-epoch \|median log10 model/measured `M*(<103)`\| | fit sample | ≤ 0.020 dex at every epoch | fails only z=2 (0.046) |
| B2 | decliner minus non-decliner median amplitude residual at z=2 | fit sample | ≤ 0.060 dex | 0.122 |
| B3 | `M*(<10 kpc)` median bias at z=2 | mh-complete | improves on −12.5% AND z ≤ 1.0 epochs stay within ±1 point of today | −12.5% |
| G2 | displaced-mass reach (exp57's gate, unchanged) | — | ≤ 25% / 10% | n/a (expansion only) |
| G5b | central-error span, non-declining galaxies (exp57, unchanged) | — | ≤ 0.060 | 0.045 |

Rules carried forward: never widen a threshold that fails; gate every
multi-start basin; the loss decides nothing a gate disagrees with.

## 3. Stage 1 — condition the EFFICIENCY law (one to three fits, ~30 min each)

The one enrichment the residuals point at that the model has never been
allowed: `log_eps` gains a formation-time term (and separately the existing
but never-adopted concentration term `E3`):

    F1:  log_eps += e_f * (f_form_j − 0.5)          (per DEPOSIT, epoch-local,
                                                     no stellar information)
    E3:  log_eps += a_c * dlogc_j                   (already implemented)

Pre-registered predictions, written before the fits:

- `e_f` **< 0** — deposits into early-formed haloes (small `t_half/t`) convert
  more efficiently, because those galaxies' measured z=2 mass exceeds the
  shared law's by −0.100 dex in the earliest-formed quartile (P-B).
- `a_c` **> 0** — same argument through concentration excess.
- B1's z=2 entry and B2 must both shrink; if a candidate fixes B1 while
  leaving B2 at 0.12, it is a shared shift wearing a conditioning costume and
  is rejected.
- Identifiability: four starts minimum, one launched from the opposite sign;
  a slope that rails, splits signs, or returns the incumbent's loss to six
  decimals is a null result (the X4 and Stage 3.8 standard).

Known risk, stated with its mitigation: exp54 Stage 3.3 watched size-law
conditioning slopes on exactly these variables collapse to zero on complete
samples (`f:fform` −0.334 → +0.005) — they were fitting the selection. Two
differences here: the objective already runs under the per-epoch mh-complete
mask, and the motivating correlations were measured UNDER that mask (P-B),
which the S3 slopes never were. After the fit, rerun the Stage 3.4 leakage
test (partial correlation of the residual with future halo growth at fixed
mass) — it must stay null.

## 4. Stage 2 — the owed `X3` refit with the size law pinned (one fit, ~2 h)

exp57's open question P-F, unchanged: refit `X3` small-core with the size law
frozen at the incumbent's, to test whether the central gain survives without
the +0.075 dex outskirt drift (the attribution test says the drift is entirely
the refit — the expansion itself contributes +0.0006 dex). Gate: shape error
at or below 13.45% with the `M[50,148]` population error within ±0.02 dex of
the incumbent's.

## 5. Stage 3 — the combination (one fit, ~2–3 h)

Best Stage-1 efficiency law + `X3` with `Rc` FROZEN at 2.6 kpc (freezing
removes the degenerate direction that let X4 drift to a 95 kpc
near-homologous core). Pre-registered prediction: G5(c)'s core-density gap
shrinks below the small-core's +0.067, because P-D shows part of that gap was
the amplitude compromise, not core geometry; G5(b) stays passed; B1/B2 pass.

## 6. Stage 4 — conditioned expansion strength, ONLY if Stage 3 leaves a
decliner-specific central residual

`A → A0 + A_f (f_form − 0.6)` with `Rc` frozen. X4's failure is attributed to
the free `Rc`; this is its one honest retry. Pre-registered: `A_f < 0`; ten
starts; drop permanently if the sign splits again.

## 7. What is explicitly NOT pursued, and why

- **Richer shared time laws for the amplitude** (E4/E6/E7 axis): P-C measured
  the exact ceiling of that whole family — under ±0.008 of tier 3 — and its
  z=2 median fix overshoots the QA population. Closed.
- **Homologous expansion in any form**: exp57's verdict stands (bounding a
  factor does not bound a reach).
- **Reviving transport**: P-A retires its apparent lead; exp52's mechanism
  verdict (96.5% of late outskirt growth was redistribution) still stands.
  Its one real remaining shape edge (z=1.0–1.5, ~0.01 of tier 3) is what
  Stages 3–4 aim at through the gated mechanism instead.
- **Size-law conditioning (S3)**: fits the selection, two independent
  demonstrations (exp54 Stage 3.3).
- **Chasing the population width with the mean law**: provably out of reach
  (0.33× spread; opposite-signed group errors); the draw layer owns it.

## 8. Cost

Stage 0 half a day; Stage 1 three fits × ~30 min plus gates; Stage 2 ~2 h;
Stage 3 ~3 h; Stage 4 ~2 h if triggered. The whole plan is under two working
days of compute, all on the standard evaluators (112 ms and 484 ms per
evaluation, measured in exp57 Stage 0).
