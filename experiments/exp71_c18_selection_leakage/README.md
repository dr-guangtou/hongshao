# exp71 — C18: is the z = 2 halo-mass dependence the z = 0.4 progenitor selection?

Branch `exp71-c18-selection-leakage`, cut from merged master at `4715e9e`.
Open question: `doc/open_questions.md` C18. **No fit is run anywhere in this
experiment**; every parameter is frozen at the exp63 product.

| script | what it does | writes |
|---|---|---|
| `leakage.py` | C18 itself: the phenomenon, the leakage channel with a permutation null, the tilt the selection alone predicts, and what survives | `outputs/leakage.{npz,log}`, `figures/exp71_c18.{png,pdf}` |
| `channel_origin.py` | the follow-up C18 forced: the channel is in the model, not in TNG300 — so where in the model? | `outputs/channel_origin.{npz,log}` |

`outputs/` and `figures/` are gitignored and regenerable (~10 min each; the
forward pass is cached in `outputs/pred_*.npy`). Run with

```
HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \
PYTHONPATH=. uv run python -u experiments/exp71_c18_selection_leakage/leakage.py [--smoke]
```

## The question, stated without jargon

Every galaxy in this project was picked at redshift 0.4 and then followed
backwards. The "z = 2 sample" is therefore not a sample of z = 2 galaxies: it is
the set of **main progenitors of a z = 0.4 selection**. At z = 2 a halo of a
given mass is in the sample only if it goes on to grow enough to clear the
z = 0.4 threshold. At the heavy end that costs nothing — every massive z = 2
halo clears it — but at the light end it keeps only the fast-growing minority.
So "the model is worse for light haloes at z = 2" mixes a statement about the
model with a statement about which light haloes we chose to look at.

exp63 Stage 5i raised it concretely: the adopted exp63 mean is within a few per
cent of the truth on the whole fitting sample at z = 2 and far off on the
mh-complete subset (`M*(50–100 kpc)`: −1.6% against −28.4%). One θ cannot do
both. Either the model needs a second variable at high redshift — a modelling
step — or the difference is the selection speaking, which is a reporting
statement.

**Definitions used throughout.** *Residual* `y = log10[M*_model(<R)] −
log10[M*_truth(<R)]`, per galaxy per epoch; negative means the model puts less
stellar mass there than TNG300 did. *Halo-mass tilt* = the slope of `y` against
`log10 M200c` across the population at one radius and one epoch, in dex of
stellar-mass error per dex of halo mass; zero means the model is equally right
for light and heavy haloes. *Future growth* `G_k = log10 Mh(z=0.4) − log10
Mh(z_k)`, how much the halo grows **after** the epoch being scored — the
variable the z = 0.4 selection cuts on and the one an epoch-local model cannot
see. *mh-complete subset* = the progenitors above the halo mass at which the
sample's completeness reaches 60% of its z = 0.4 plateau (exp54 Stage 3.4,
unchanged and re-read from `exp54/outputs/selection.npz`).

**What is scored.** `exp63/outputs/stage2_fit_joint_kpc_free_sane.npz`, one θ
fitted at all five epochs, **frozen**. Every table also carries the **nested
incumbent** — the same code with the compact channel's share set to exactly
zero — as the control: if both behave the same, what is measured belongs to the
sample and the model class, not to the exp63 fit.

**The sample** is the repo rule (`CLAUDE.md`): all galaxies selected at z = 0.4
with a sane halo and stellar history at every epoch, `2356 of 2397`. The
mh-complete subset (2356 / 1758 / 1416 / 1129 / 831 per epoch) is the after-fit
check and never a fit mask.

---

## 1. The phenomenon: at z = 2 the two samples disagree about the sign

Halo-mass tilt of the exp63 mean, dex per dex, with bootstrap 95% intervals
(full table for both models and seven quantities in `outputs/leakage.log`):

| quantity | epoch | fitting sample | mh-complete |
|---|---|---|---|
| M(<103 kpc) | 0.4 | +0.012 [+0.000, +0.025] | (identical — no selection acts) |
| | 0.7 | +0.000 [−0.014, +0.014] | +0.018 [−0.000, +0.037] |
| | 1.0 | −0.008 [−0.022, +0.007] | +0.018 [−0.005, +0.037] |
| | 1.5 | −0.017 [−0.034, −0.002] | +0.056 [+0.031, +0.082] |
| | **2.0** | **−0.122 [−0.142, −0.101]** | **+0.061 [+0.022, +0.097]** |

Below z = 2 the fitting-sample tilt is at most 0.017 dex per dex at 103 kpc. At
z = 2 it is −0.122 — and on the mh-complete subset it is **+0.061**, the
opposite sign. The same reversal happens at every radius from 10 kpc outwards
(fitting sample −0.105 / −0.108 / −0.122 / −0.128 at 10 / 52 / 103 / 148 kpc
against mh-complete +0.065 / +0.081 / +0.061 / +0.049). That reversal is the
thing C18 asks about.

The medians the tilt is made of, median `log10(model/truth)` in dex — this is
exp63 Stage 5i's table, reproduced from the same frozen θ:

| quantity | sample | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|---|
| M(<103 kpc) | fitting sample | −0.011 | +0.007 | +0.008 | +0.000 | −0.020 |
| | mh-complete | −0.011 | +0.004 | +0.005 | −0.004 | **−0.045** |
| M(52–103 kpc) | fitting sample | +0.007 | +0.013 | +0.006 | −0.005 | −0.004 |
| | mh-complete | +0.007 | −0.021 | −0.054 | −0.104 | **−0.148** |

−0.045 dex is −9.7 per cent and −0.148 dex is −28.8 per cent, which are exp63
Stage 5i's −9.7% and −28.4%. Nothing has moved; this is the same phenomenon
measured in dex so the tilt and the median sit in the same units.

## 2. The channel exists, and it is not an artefact of the estimator

Partial correlation of the residual with future growth at fixed `log10
M200c(z_k)`, and the slope `dy/dG` in dex per dex, at 103 kpc. The null column
reshuffles `G` **inside equal-count halo-mass bins**, so it keeps every shared
dependence on halo mass that stripping a straight line leaves behind and
destroys only which galaxy has which growth (200 draws):

| epoch | exp63 partial ρ | exp63 dy/dG | nested incumbent ρ | dy/dG | null 95% of ρ |
|---|---|---|---|---|---|
| 0.7 | +0.180 | +0.174 | +0.182 | +0.175 | [−0.002, +0.065] |
| 1.0 | +0.198 | +0.144 | +0.204 | +0.148 | [+0.003, +0.069] |
| 1.5 | +0.207 | +0.129 | +0.224 | +0.141 | [+0.027, +0.099] |
| 2.0 | +0.191 | +0.122 | +0.213 | +0.139 | [+0.060, +0.134] |

Every value is outside its null. exp54 measured +0.19 / +0.22 / +0.23 / +0.21
with `dy/dG` +0.18 / +0.16 / +0.15 / +0.15 for the incumbent under a slightly
different mask; those numbers reproduce here. **The exp63 fit and the nested
incumbent are indistinguishable**, so the channel is not something the exp63 fit
introduced. It survives swapping the conditioning variable from the catalog
`M200c` to the DiffMAH mass the model actually reads (+0.238 / +0.143 at z = 2).

## 3. The size: the selection alone predicts the tilt that is observed

`selection.predicted_spurious_tilt` predicts the tilt the selection **alone**
would produce, from two independently measured things — the completeness curve
and `dy/dG` — with no reference at all to the tilt we observe. `pred (bound)`
takes the growth scatter from the mh-complete subsample, so it understates it
wherever the sample is truncated; `pred (corr)` divides the truncation out with
the standard-normal formula, so it overstates it in the far tail. They bracket
what selection can do. At 103 kpc, dex per dex:

| epoch | observed, fitting sample | observed, mh-complete | observed shift | predicted shift | predicted tilt on the fitting sample |
|---|---|---|---|---|---|
| 0.7 | +0.000 | +0.018 | −0.018 | −0.016 … −0.042 | −0.022 … −0.058 |
| 1.0 | −0.008 | +0.018 | −0.025 | −0.026 … −0.075 | −0.034 … −0.086 |
| 1.5 | −0.017 | +0.056 | −0.074 | −0.037 … −0.117 | −0.048 … −0.135 |
| 2.0 | −0.122 | +0.061 | −0.183 | −0.045 … −0.156 | −0.063 … −0.180 |

Read the last column against the second: **the whole of the z = 2 fitting-sample
tilt at 103 kpc, −0.122, lies inside the −0.063 … −0.180 the selection alone
predicts.** Subtracting the prediction leaves +0.058 … −0.059, which brackets
the mh-complete measurement of +0.061. Panel (c) of `figures/exp71_c18.png` is
that statement: the corrected band covers the mh-complete points at every epoch.

Quantity by quantity at z = 2 (predicted spurious tilt on the fitting sample,
bound … corrected, against the observed):

| quantity | observed, fitting sample | observed, mh-complete | predicted from selection alone |
|---|---|---|---|
| M(<2 kpc) | −0.156 | −0.064 | −0.009 … −0.026 |
| M(<4.9 kpc) | −0.146 | −0.001 | −0.029 … −0.083 |
| M(<10 kpc) | −0.105 | +0.065 | −0.044 … −0.126 |
| M(<52 kpc) | −0.108 | +0.081 | −0.061 … −0.176 |
| M(<103 kpc) | −0.122 | +0.061 | −0.063 … −0.180 |
| M(<148 kpc) | −0.128 | +0.049 | −0.063 … −0.180 |
| M(52–103 kpc) | −0.573 | −0.158 | −0.177 … −0.506 |

**Where the selection does NOT explain it: the innermost aperture.** At
`M*(<2 kpc)` the observed z = 2 shift is −0.092 and the selection predicts at
most −0.026. The centre's halo-mass dependence is its own problem — the one
already on record as the central defect (memory
`central-defect-is-outside-the-model-class`) — and C18's answer does not cover
it. From 10 kpc outward the prediction covers the observation.

## 4. The answer to C18

**Yes.** The difference between the whole fitting sample and the mh-complete
subset at z = 2 is what the z = 0.4 progenitor selection alone predicts, at
every radius from 10 kpc outwards, with no free parameter and no fit. The
honest response is to report it, not to fit it: **a second high-redshift
variable is not required by this evidence.** The exception is `M*(<2 kpc)`,
where the selection accounts for at most a quarter of the shift and the
long-standing central defect accounts for the rest.

Two things this does *not* say. It does not say the model is right at z = 2 —
the mh-complete subset still shows the model 0.045 dex (9.7 per cent) low in
the median at 103 kpc and 0.148 dex (28.8 per cent) low in the 52-103 kpc
shell, and those are real errors at fixed halo mass — just not a *tilt* the
selection failed to explain. They are exp63 Stage 5i's numbers, reproduced
here from the same frozen theta. And it does not say the leakage is harmless
physics, which is what part 2 is about.

---

## Part 2 — the channel is the model's, not the simulation's

C18 was answered by section 3 without ever asking where `dy/dG` comes from. It
has to be asked, because the same measurement made on TNG300 itself gives a
different answer.

### The measured stellar mass does not read future growth. The model does.

Partial slope on future growth at fixed `log10 M200c(z_k)`, 103 kpc, dex per
dex (`channel_origin.py` section 1):

| epoch | model | truth | residual (= model − truth) |
|---|---|---|---|
| 0.7 | +0.168 | −0.006 | +0.174 |
| 1.0 | +0.196 | +0.052 | +0.144 |
| 1.5 | +0.154 | +0.025 | +0.129 |
| 2.0 | +0.125 | +0.003 | +0.122 |

TNG300's own `M*(<103 kpc)` at fixed halo mass has essentially **no** dependence
on how much the halo grows afterwards — +0.003 dex per dex at z = 2. The
residual's dependence is the model's dependence, almost exactly. So the leakage
that section 3 used is not a feature of the universe the model failed to
reproduce; it is something the model **invents**.

### Why: the model's "past" carries the halo's final mass

The model's prediction at z_k is a deterministic functional of the DiffMAH curve
**before** z_k. Cross-validated R² of future growth at fixed `M200c(z_k)`, same
target, same predictor count, out-of-fold (`channel_origin.py` section 2):

| epoch | pre-z_k snapshots at z = | measured history | measured, smoothed | DiffMAH curve | DiffMAH `logmp` alone |
|---|---|---|---|---|---|
| 0.7 | 2.0, 1.5, 1.0 | −0.001 | −0.003 | **0.503** | 0.426 |
| 1.0 | 3.0, 2.0, 1.5 | +0.001 | +0.001 | **0.691** | 0.609 |
| 1.5 | 4.0, 3.0, 2.0 | +0.000 | +0.002 | **0.661** | 0.732 |
| 2.0 | 5.0, 4.0, 3.0 | −0.001 | — | **0.524** | 0.794 |

**Future growth at fixed halo mass is completely unpredictable from the measured
pre-z_k history — R² is zero to three decimals at every epoch — and 52 to 69 per
cent predictable from the DiffMAH curve read at exactly the same times.** The
middle column is the control for the obvious objection, that three raw
snapshots carry halo-finder noise a smooth curve does not: it gives the catalog
the same smoothness (a polynomial of the same order in log time fitted to every
usable pre-z_k snapshot, strictly no data from after z_k) and the R² stays at
zero.

The last column is the mechanism in one number. DiffMAH's `logmp` is the halo's
**peak** mass — a property of the whole history, the future included — and it
enters the curve at every time, so a curve read before z_k still carries it. At
z = 2 that single parameter predicts 79 per cent of future growth at fixed halo
mass. The same leak shows directly in the fitted-minus-measured mass at the
epoch itself: the offset correlates with future growth at ρ = +0.50 / +0.56 /
+0.44 / +0.35, with a slope of +0.35 / +0.33 / +0.23 / +0.17 dex per dex
(section 4).

### What follows

- The z = 2 halo-mass tilt on the fitting sample is real, is the selection, and
  is **fixable at its source**: it exists because the model's halo history is a
  four-parameter global fit that knows the answer. Feed the model a history
  built only from data before z_k — the measured MAH, or a DiffMAH refitted on
  pre-z_k points — and `dy/dG` should fall from +0.12 toward the truth's
  +0.003, taking the spurious tilt with it.
- Nothing in this experiment tests that; it would be a fit, and C18 asked for a
  diagnostic. It is written up as a new open question rather than done here.
- The corroboration table (`channel_origin.py` section 3) conditions the
  residual on the DiffMAH past directly. It is reported with its own warning: a
  partial slope is measured on the part of `G` the control does not explain, and
  where the control explains 55–86 per cent of `G` that remainder is thin and the
  slope on it is free to move in either direction. Section 2 is the clean test,
  because it compares two predictor sets on the same target rather than
  conditioning on a near-determining variable.

### Bearing on results already on record

`halo-history-knows-a-little-about-the-core` measured partial |ρ| ≤ 0.23 between
the z = 0.4 inner shape and DiffMAH `early` / `late` / `t50` at fixed halo mass.
Those are parameters of the same global fit. The leak measured here is about
growth *after* the scored epoch, and z = 0.4 is the selection epoch, so that
result is not directly touched — but it has not been checked against a
measured-history version of the same variables, and it should be before it is
leant on. Recorded, not claimed.
