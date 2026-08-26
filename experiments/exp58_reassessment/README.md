# exp58 — reassessment of the deposition-only model

Branch `exp58-reassessment` (off `exp57-expansion-term`). **No model is fitted
in this experiment.** It measures three things the programme has argued about
but never measured, and turns them into a plan. Every number is on exp57's
galaxy sample; scripts regenerate everything (`outputs/` and `figures/` are
gitignored, the logs quoted here are `outputs/p{a,b,c}_full.log`).

The user's three-part verdict on exp57, which this experiment tests:

1. *"compared to the deposition+transportation model the overall performance is
   still systematically falling behind"* — **P-A finds this is the amplitude
   oracle, not the model.** Like for like the gap is ~3%, and in pure shape
   the deposition models are the transport kernel's equal or better at four of
   five epochs.
2. *"the overall model still clearly has problems at high redshift, especially
   z=2"* — **confirmed, and P-B says precisely what it is**: one number per
   galaxy (the total amplitude, median −0.046 dex on the fit sample),
   concentrated in the galaxies whose centres later decline (−0.109 dex),
   plus an inner-core shape deficit shared with z=1.5 — and the loss is
   structurally unable to care about it.
3. *"the improvement in the average CoG is very hard to see"* — already
   established as exp57 P-C (the battery's headline tiers cannot see a 1.4%
   rearrangement of mass inside 10 kpc); restated here for completeness.

---

## P-A — the transport comparison, finally like for like (`pa_headtohead.py`)

**What was done.** The old adopted transport kernel — exp38's `1ch-mof` at the
exp40 theta (`exp40/outputs/latestart.npz::theta_z15`), reproduced through
exp53's kernel — was run through `hongshao.qa.evaluate` on exp57's sample for
the first time ever with its amplitude **predicted** instead of pinned to the
measured `M*(<500 kpc)`. Its amplitude provider is the best halo-only one the
programme has (out-of-fold regression on official DiffMAH 4 params +
concentration excess + fz2, at 100 kpc), which is **generous to the transport
kernel**: that regression is nearly bias-free at every epoch, while the
deposition models must predict their amplitude from their own mechanism. A
second block pins ALL three models to the measured `M*(<103 kpc)` so the pure
radial shape can be compared with the amplitude question removed. Common set:
**2396 of 2397** galaxies (row 181, the known broken cross-match, is the one
drop — note it was still inside exp57's stage10 tables; its removal moves the
incumbent's tier 3 from 0.2765 to 0.2763, i.e. nothing).

**Tier 3** (the QA battery's headline: the largest relative error of the
cumulative mass profile beyond 5 kpc, median over galaxies — lower is better):

| amplitude PREDICTED | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| transport (regression amplitude) | **0.2682** | **0.2647** | **0.2731** | **0.2846** | **0.3144** |
| incumbent (its own amplitude) | 0.2763 | 0.2802 | 0.2867 | 0.2963 | 0.3231 |
| `X3` small-core (its own amplitude) | 0.2764 | 0.2822 | 0.2893 | 0.2941 | 0.3178 |

| amplitude PINNED at measured M*(<103 kpc) — pure shape | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| transport | 0.1754 | 0.1692 | **0.1582** | **0.1571** | 0.1492 |
| incumbent | 0.1742 | 0.1717 | 0.1732 | 0.1674 | 0.1492 |
| `X3` small-core | **0.1688** | **0.1665** | 0.1657 | 0.1594 | **0.1458** |

**The verdict, in words.** The published transport numbers (0.195 at z=0.4)
were oracle-amplitude numbers; handed a *realistic* amplitude the same kernel
scores 0.268, against the incumbent's 0.276 — a **3% relative gap, not 42%**,
confirming exp54's re-baselining estimate (+2.8%) head to head, after the
row-181 fix, on identical galaxies. And most of even that gap is the
**amplitude provider, not the shape**: the regression handed to the transport
kernel has smaller scatter (0.104 vs 0.116 dex at z=0.4) and near-zero bias at
every epoch (worst −1.1%), where the deposition law's own amplitude is −6.4%
at z=2. In pure shape the incumbent already matches the transport kernel at
z=0.4/0.7, and `X3` small-core is **the best of the three at four of five
epochs**. The transport kernel's one genuine remaining shape advantage is
z=1.0–1.5 (0.158–0.157 against the incumbent's 0.173–0.167) — worth knowing,
because that is where its `fc(t)` term moves early central mass outward with
time, which is the capability exp57 established the deposition class lacks.

Two honesty caveats, both stated rather than assumed away: the transport theta
was *fitted* under the 500 kpc pin at the z≤1.5 scope, so this measures the
adopted artifact, not the transport class's ceiling; and its amplitude
regression is trained (out-of-fold) on this sample's measured masses, exactly
as exp54's rebaseline defined `hybrid-mean`. Figures:
`figures/qa/qa_*_exp58_transport.*` — the binned-CoG figure shows the same
z=1.5–2 under-prediction pattern the user has been reading in the deposition
model's figures, now visible in the old model once its oracle is removed.

**Consequence.** "Systematically falling behind the old model" should be
retired as a reason to change course. The deposition-only foundation stands.

---

## P-B — what, exactly, is wrong at z=2 (`pb_z2_diagnosis.py`)

Everything below is the incumbent, nothing fitted, on the fit sample (the
mh-complete + sane mask) unless said otherwise.

**1. It is an amplitude error, not an outskirt-shape error.** The median
log10(model/measured) at z=2 is **−0.045 ± 0.007 dex, flat from 23 kpc to
148 kpc** (−0.0451, −0.0436, −0.0461, −0.0452 at 23 / 52 / 103 / 148 kpc).
With the amplitude divided out, the z=2 shape error beyond 10 kpc is at most
0.023 dex. One number per galaxy is wrong, not the run of the profile.

**2. Plus an inner-core shape deficit that peaks at z=1.5–2.** Amplitude-pinned
median at 2 kpc: **−0.084 dex at z=2, −0.074 at z=1.5**, fading to −0.038 at
z=1.0 and reversing to +0.018 at z=0.4. (A persistent −0.04 to −0.08 dex dip
at the 3.7 kpc grid point exists at *every* epoch — a standing shape defect of
the deposit family worth knowing about, separate from the epoch story.)

**3. The amplitude deficit is carried by identifiable galaxies.** Correlations
of the z=2 amplitude residual, measured on the masked fit sample (so this is
NOT the S3 selection artifact of exp54 Stage 3.3, which lived on the full
biased sample):

| feature | r | r at fixed halo mass | Q1 → Q4 median residual [dex] |
|---|---|---|---|
| centre later declines (flag) | −0.440 | **−0.454** | +0.013 (no) → **−0.109** (yes) |
| concentration excess at z=2 | −0.388 | **−0.378** | +0.036 → −0.103 |
| formation time `t_half/t` at z=2 | +0.297 | +0.285 | −0.100 (early) → −0.008 (late) |
| log10 Mh at z=2 | +0.169 | — | −0.065 → −0.017 |
| FUTURE growth (not visible to model) | +0.112 | +0.107 | small |

**Early-formed, concentrated haloes — the ones whose centres later decline —
have more stellar mass at z=2 than the efficiency law grants them.** This is
the programme's own science result (compact high-redshift progenitors formed
early, with a ~0.3 dex early head start at fixed final mass) showing up as a
model residual. The efficiency law has never been allowed to see it: `log_eps`
depends on (mass, redshift, mass×redshift) only; `f_form` enters no efficiency
law, and the concentration term `E3` was tried once and judged **by the loss**.

**4. The loss cannot care about any of this, and that is structural.** The
incumbent's per-epoch scores: score_A = 1.11 / 1.06 / 1.03 / 1.04 / **0.94**
and score_F = 0.93 / 0.91 / 0.90 / 0.84 / **0.78** — z=2 is the loss's *best*
epoch while being the user's visibly worst. Two reasons: the amplitude score
divides by the per-epoch benchmark scatter `SIGMA_A`, which is **0.161 dex at
z=2** (against 0.104 at z=0.4), so the −0.046 dex systematic bias costs
(0.046/0.161)² ≈ 0.08 of one epoch's score_A² — roughly **0.6% of the total
loss**; and an RMS cannot distinguish a bias from scatter it already carries.
The fifth backwards ranking was never going to be prevented by this loss.

**5. The fit and the QA battery see different z=2 populations.** The fit's
mask admits **839 of 2397** galaxies at z=2 (mh-complete + sane); the QA
battery averages all 2397, 65% of which the fit never saw at that epoch.

---

## P-C — the amplitude-enrichment ceiling ladder (`pc_ceilings.py`)

*(measured on the incumbent's own CoGs scaled per galaxy-epoch — shape
identical throughout, so every change is amplitude bookkeeping alone; the
conditioned rung is out-of-fold and has a row-shuffled control)*

The ladder: `incumbent` → `shared-z` (per-epoch median bias removed — the
exact ceiling of any amplitude enrichment that depends on time only) →
`cond-oof` (out-of-fold linear debias on log10 Mh, formation time,
concentration excess and fz2 at each epoch — the honest reachable ceiling of a
halo-conditioned enrichment) → `cond-shuf` (same machinery, feature rows
shuffled — the permuted control) → `oracle-amp` (measured amplitude copied in
— the floor; everything left is shape).

**Tier 3, fit-sample galaxy-epochs** (median max|rel| beyond 5 kpc):

| variant | z=0.4 | z=1.0 | z=2.0 |
|---|---|---|---|
| incumbent | 0.2765 | 0.2949 | 0.3162 |
| shared-z | 0.2796 | 0.2875 | 0.3194 |
| cond-oof | **0.2699** | **0.2717** | **0.2990** |
| cond-shuf (control) | 0.2794 | 0.2889 | 0.3230 |
| oracle-amp (floor) | 0.1742 | 0.1686 | 0.1484 |

Five findings, each of which constrains the plan:

1. **A time-only amplitude enrichment is worthless**: `shared-z` moves tier 3
   by less than ±0.008 everywhere. Do not spend another E6/E7-style stage on
   the amplitude's time law; exp54 Stage 3.5's closure extends to this axis.
2. **Conditioning carries real information**: `cond-oof` improves every epoch
   and its correction has a 0.13 dex spread at z=2 against the shuffled
   control's 0.018, and the control itself gains nothing. The information is
   real and halo-local — this is the measured warrant for putting formation
   time / concentration into the EFFICIENCY law.
3. **But the reachable amplitude ceiling is modest**: 0.316 → 0.299 at z=2,
   about a fifth of the way to the oracle floor (0.148). Most of the
   amplitude scatter is not predictable from these four halo features — the
   remainder is the generative layer's business (a draw), not the mean law's.
4. **Corrections learned on the fit sample OVERSHOOT the full QA population**
   (`M(<100)` bias at z=2: −6.4% → +4.1% under `shared-z`, +13.2% under
   `cond-oof`): the mh-complete fit sample carries a −0.046 dex bias where the
   full sample carries −0.029, so a law tuned on one over-corrects the other.
   Any amplitude gate must therefore be defined on the fit sample AND the QA
   z=2 numbers must be quoted on the mh-complete subsample — exp54's own
   honest-evaluation doctrine, now with a measured reason.
5. **With a PERFECT amplitude the z=2 core is still −9.4% at `M*(<10 kpc)`**
   (`oracle-amp` row of the aperture table). The amplitude law can fix the
   z=2 outskirts; the z=2 core is the expansion term's job. The two fixes are
   complementary, not competing. (Beware the seductive middle rows: `shared-z`
   shows −2.7% at `M(<10)`/z=2 only because its +4% amplitude overshoot
   cancels the −9% shape deficit in the median — two wrongs, not a fix.)

---

## The unified reading

The z=2 amplitude deficit and the central defect are **one phenomenon priced
into two epochs by the loss**. For the 41.8% of galaxies whose measured
central mass declines from z=2 to z=0.4 (median −0.066 dex at 4.92 kpc), a
monotone-in-time model *cannot* match both endpoints: the two errors must sum
to at least the decline. The incumbent resolves that arithmetic where the loss
makes it cheapest — at z=2, whose benchmark sigma is the largest — which is
why the decliner group sits at −0.109 dex there while the non-decliners sit at
+0.013. exp57's core-only expansion (`X3`) is the release valve for exactly
this constraint, and its one measured QA effect is exactly the predicted one:
the z=2 `M*(<10 kpc)` bias moves −12.5% → −8.9%.

The compromise, measured per group (`pd_decliner_split.py`, incumbent, fit
sample; median log10 of model over measured):

| `M*(<4.92 kpc)`, the centre | z=0.4 | z=1.0 | z=2.0 | \|z=2 err\| + \|z=0.4 err\| |
|---|---|---|---|---|
| declining centres (42%) | +0.020 | −0.030 | **−0.174** | 0.194 |
| non-declining (58%) | **−0.066** | −0.024 | −0.027 | 0.092 |

| `M*(<103 kpc)`, the amplitude | z=0.4 | z=1.0 | z=2.0 |
|---|---|---|---|
| declining centres | **+0.042** | +0.022 | **−0.109** |
| non-declining | −0.040 | −0.005 | +0.013 |

And the bluntest row of the whole experiment — the central CHANGE from z=2 to
z=0.4, median per group: measured **−0.062** (decliners) against **+0.100**
(the rest); the model gives **+0.083 and +0.104** — the same growth for both,
because nothing in it can tell the groups apart. Three further things these
tables say:

1. **The two groups' errors have OPPOSITE signs at both epochs**, at the
   centre and at the amplitude. A shared law splits the difference; any
   improvement to the median alone will move one group's error onto the other.
   This is exp57 Stage 1's "the model's spread is 0.33 times the data's"
   restated as two medians.
2. **The incumbent is far above the arithmetic floor.** The measured central
   decline (median −0.062 dex) forces the two endpoint errors to sum to at
   least 0.062 for the declining group; the incumbent's sum is 0.194 — three
   times the floor. Most of the decliners' error is therefore NOT the
   monotonicity constraint itself but the shared amplitude law's inability to
   give them their early head start; a conditioned law could remove up to two
   thirds of it without leaving the model class.
3. **The decliners' amplitude error crosses zero between z=2 and z=0.4**
   (−0.109 → +0.042): the model under-supplies their early mass and
   over-supplies their late mass. That is precisely "this halo formed its
   stars earlier than the shared efficiency law says", and it is the
   population-level mirror of the science result about early-formed compact
   progenitors.

What this does NOT mean: it does not mean the model class is wrong. It means
(a) the amplitude law is missing a conditioning variable it is allowed to
have, (b) the centre needs the already-gated `X3` term, and (c) the loss and
the battery both need a bias-sighted gate before any of it can be fitted
honestly.
