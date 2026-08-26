# exp59 — conditioning the efficiency law on the halo's assembly state

Branch `exp59-conditioned-efficiency`. Plan:
`doc/plans/2026-08-26-exp58-reassessment-plan.md` Stages 0–1. Problems:
[`PROBLEMS.md`](PROBLEMS.md).

**Status: Stage 1 CLOSED AS A MEASURED NEGATIVE, before any fit was spent.**
The gate ran first and disqualified every candidate.

---

## Why this experiment exists

exp58 measured that the z=2 amplitude deficit is carried by the galaxies whose
centres later decline (−0.109 dex against +0.013 for the rest, r = −0.45 at
fixed halo mass, on the masked fit sample), and that the efficiency law
`log_eps(m, z, m·z)` is never allowed to see the quantities that mark them
(formation time, concentration excess). The plan's Stage 1 was to let it see
them: `log_eps += e_f (f_form_j − 0.5)` per deposit, and separately the
existing concentration term.

## The machinery (`conditioned.py`)

`CSpec`/`ConditionedProblem` extend exp54's `StackedProblem` with per-deposit
conditioning terms in `log_eps`, through a single `cond_le` hook so no
diagnostic ever copies `predict` (exp57 P6). Nesting is **EXACT** — with every
coefficient at zero the class reproduces the parent's values, NaN pattern and
loss with **no tolerance at all** (adding an exact-zero term is exact in
floating point), which is a stronger warrant than exp57's 8-ulp version. The
self-check also verifies the mechanism: each galaxy's mass change tracks its
deposit-weighted conditioning mean at r = +0.999.

## Stage 0b — the leverage sweep (`stage0_leverage.py`), THE RESULT

**The question, asked before fitting because the loss cannot referee it**
(exp58 P-B: the entire z=2 bias costs ~0.6% of the loss): can ANY coefficient
on a per-deposit conditioning variable close the z=2 decliner-minus-rest
amplitude split (−0.122 dex) to within the B2 gate (±0.060) without pushing
the z=0.4 split (+0.082 dex, opposite sign — exp58 P-D's crossing) beyond a
+0.010 no-harm guard? Qualification rules declared before the numbers were
read; five candidates swept over coefficients −4 to +4 by direct model
evaluation, base parameters held at the incumbent:

| candidate, per deposit | z=2 split closes at | z=0.4 split there | verdict |
|---|---|---|---|
| `F1` `f_form − 0.5` | coef ≈ −3 (reaches +0.016 at −4) | **+0.27** (3.3× worse) | DISQUALIFIED |
| `C1` `dlogc` | coef ≈ +4 (−0.024) | **+0.19** | DISQUALIFIED |
| `R1` `log10(dmh/Mh)` centred | coef ≈ −2 (+0.001) | **+0.20** | DISQUALIFIED |
| `F2` `(f_form−0.5)·ln(1+z)` | coef ≈ −4 (+0.026) | **+0.17** | DISQUALIFIED |
| `H1` `max(0, 0.45 − f_form)` | **no leverage at all** (split fixed at −0.1216 for every coef) | — | DISQUALIFIED |

Rule B (fix only the z=2 MEDIAN bias, splits unharmed) also finds no
qualifying point: `R1` at +2 does zero the adjusted z=2 median (−0.023 →
−0.006) but drives the z=2 split from −0.122 to −0.182.

**Reading it.** Every variable moves the two epochs' splits in LOCKSTEP —
the response vectors of `F1` and `C1` are anti-parallel to within 3%, so even
their two-parameter combination is degenerate along exactly the direction the
crossing needs. And `H1`'s null result is its own finding: deposits made after
a halo stalls carry almost no mass, so the decliners' late over-supply is NOT
sitting in identifiably "stalled" deposits — the model's late mass for
decliners comes from deposits that look ordinary in every tested coordinate.

**The conclusion, stated carefully.** The decliner amplitude crossing
(−0.109 dex at z=2 → +0.042 at z=0.4) is unreachable by ANY single per-deposit
multiplicative conditioning of the efficiency on formation time, concentration
excess, specific accretion rate, their redshift interaction, or a stalled-halo
hinge. This is the amplitude-level twin of exp57's expansion-spread result:
the galaxies' stellar-side memory (early head start, later decline) is only
weakly traced by the halo record — exp57's diagnostic put the ceiling at
r = −0.317 for the decline flag — and a shared halo-conditioned mean law
cannot manufacture the missing information. What was disqualified is the
Stage 1 CANDIDATE FAMILY, not the model: the incumbent stands, unmodified.

**Caveat, stated rather than hidden.** The sweep is a conditional slice (base
parameters held at the incumbent), which exp54 Stage 3.9 showed can overstate
a parameter's determination — but here it is used only to DISQUALIFY:
a differential the mechanism moves in lockstep cannot be unlocked by refitting
base parameters that are blind to the groups (halo mass separates them at
r = +0.025). A passing sweep would still have required the fit and the gates.

## What this changes in the plan

- **Stage 1 is closed.** No conditioned-efficiency fit is run; five ~30-minute
  fits were saved by fifteen model evaluations.
- **Stage 2 (the owed `X3` small-core refit with the size law pinned) is now
  the active stage** — unaffected by this result and still the only gated
  mechanism that moves the central defect.
- The z=2 amplitude deficit's decliner component moves from "model defect to
  fix" to "measured limit of halo-determinism" pending Stage 2's outcome — the
  candidate science statement being that the stellar mass a galaxy holds at
  z=2 in excess of the halo-conditioned expectation is the same memory that
  makes its centre decline, and the halo record alone cannot carry it.

## The figure

**`figures/exp59_leverage.png`** — the whole Stage 0b result in one panel:
each candidate's trajectory in the plane of the two gate quantities (the
decliner-minus-rest amplitude split at z=2 against the same split at z=0.4)
as its coefficient sweeps −4 to +4. The green box is where a candidate must
land (B2 and the B2b guard); every trajectory runs diagonally past it, and
the stalled-halo switch never leaves the incumbent's dot. Regenerate with
`stage0_figure.py` (reads `outputs/stage0_leverage.npz`).

---

## Stage 2 — the owed `X3` refit with the size law pinned. DONE, and it changes the adoption case.

The fit exp57 left owed: `X3` refitted with the size-law parameters frozen at
the incumbent's, so the central gain cannot be paid for with outskirt drift.
Nine starts (the run was killed externally after four — no crash, no
traceback; the per-start checkpoint plus a deterministic `--starts-only 4-8`
reproduction under a separate tag completed it, merged by `stage2_merge.py`
with consistency assertions). Logs and gates in `outputs/stage2_*`.

| | incumbent | `X3` unpinned (exp57) | **`X3` pinned size (this)** |
|---|---|---|---|
| loss | 1.874253 | 1.833344 | **1.846860** |
| profile-shape error | 13.787% | 13.369% | **13.520%** |
| fitted `A`, `Rc` | — | +1.40, 2.61 kpc | **+1.50, 1.98 kpc** |
| G2 reach beyond 50 / 148 kpc (limits 25 / 10%) | — | 2.2% / 0.2% | **0.5–1.5% / ≤0.1%** |
| G5(a) declining central span (0.270 at the null) | 0.270 | 0.210 | **0.239** |
| G5(b) non-declining span (limit 0.060) | 0.045 | 0.039 | **0.018 — best of anything ever tried** |
| G5(c) core-density gap (null +0.0455) | +0.0455 | +0.0668 | +0.0749 — FAIL |
| G6 outskirt `M[50,148]` population error | −0.076 | **+0.075 (the drift)** | **−0.062 — the drift is GONE, and it improves on the incumbent** |

Three findings:

1. **The outskirt drift was entirely the size-law refit, now proven by
   removal**: pinned, the annulus error improves on the incumbent instead of
   drifting 0.15 dex the other way. exp57's attribution test predicted this;
   the refit confirms it.
2. **The basin pathology does not survive pinning.** Six of nine starts land
   in the small core at the best loss; the large-core direction either loses
   (1.8616) or rails at the `log_Rc` bound. For the first time in this
   programme, **the loss and the gates agree** on which solution is right —
   exp57's P-E conflict was a symptom of the size-law freedom.
3. **The price of pinning is 36% of the central gain**: shape error 13.520%
   against the unpinned 13.369%, and the pre-registered Stage 2 criterion
   (≤ 13.45%) is therefore MISSED by 0.07 points — reported as the miss it
   is. The amplitude gates (B1/B2) sit at the null's values, as Stage 0b
   predicted for any core-only rearrangement; B3 fails on z=1.0 (+2.3% →
   +4.5%) while its z=0.4 entry trips the gate's letter (moves more than
   1 point) by IMPROVING in absolute size (−2.2% → +1.6%) — stated, not
   relitigated.

## Stage 4a — the leverage sweep for the formation-conditioned expansion. DISQUALIFIED; the plan is complete.

The plan's Stage 4 trigger (a decliner-specific central residual after the
expansion stage) is met: declining span 0.239 against non-declining 0.018.
`stage4_leverage.py` swept the `X4` conditioning slope `A_f` from −10 to +10
with `A0` and `Rc` held at the Stage 2 optimum, against a rule declared before
the numbers (declining span ≤ 0.20 with non-declining ≤ 0.060). **No grid
point qualifies**: the declining span never leaves 0.23–0.28. The
pre-registered sign does show its predicted effect — at `A_f` = −10 the
decliners' z=2 central error improves −0.205 → −0.131 — but as a SHIFT of
their whole error curve, not the bend the span needs, so their span worsens
while it happens. The deposit-weighted formation-state contrast between the
groups (~0.03) is simply too weak to carry a differential mechanism, at the
centre exactly as at the amplitude (Stage 0b). The Stage 4 fit is not run.

**Every stage of the approved plan is now executed or measurably closed.
Nothing is adopted — that decision is the user's.** The candidate on the
table is `X3` pinned-size small-core: every mechanism gate passed, both exp57
objections (drift, basin conflict) resolved, at the cost of 0.15 points of
shape error against the unpinned version and with G5(c) — the median core
over-growth — still standing, now measured to be beyond the reach of every
shared mechanism this programme has tested.
