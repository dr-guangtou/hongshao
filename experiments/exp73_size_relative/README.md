# exp73 — score the galaxy where the galaxy is

Branch `exp73-size-relative-objective`, cut from merged master at `5d142f4`.
Plan: `doc/plans/2026-09-01-exp73-size-relative-objective.md` (the user's
decisions D1/D2/D3) and `doc/plans/2026-09-01-exp73-continuous-session.md` (the
block schedule with its gates). Open questions: C20, C21.

**The question.** The programme scores every epoch on the same fixed physical
radial grid with a relative residual. exp72 measured that this is a
redshift-dependent weighting nobody chose: the median half-mass radius runs
11.6 kpc at z = 0.4 down to 3.0 kpc at z = 2, so 148 kpc is 13 R50 at one end
and 49 R50 at the other, and at z = 2 the production objective spends 27.8 per
cent of its attention on shells holding 3.3 per cent of the galaxy's mass. Does
scoring in units of each galaxy's own size — with the user's two repairs, a
mass-weighted residual and a down-weighting of near-empty shells — let one
shared five-epoch law hold both the inner profile and the mass–size relation?

| script | block | what it does | fit? |
|---|---|---|---|
| `coordinate.py` | A1 | the size-relative coordinate, the mass-weighted residual, the coverage gate, the two weighting schemes; `--selftest`, seven claims | no |
| `risk_test.py` | A2 | does the joint-vs-per-epoch gap narrow under the coordinate; do the per-epoch optima lie on a smooth curve in time | no |
| `weighting.py` | B | the user's D2: smooth mass-share weights against a threshold gate, head to head | no |
| `fit_re.py` | C | the exp63 model refitted on R50 shells with the winning objective | **yes, ~1.5 h** |
| `eval_re.py` | C | the judge: transfer matrix in both coordinates, mass–size relation, the size gate, profiles in per cent, exp71's diagnostics | no |
| `size_gate_check.py` | D | first use of the new `hongshao.qa` size gate on three models whose sizes are already understood | no |

`hongshao/qa.py` gained R20 and the tier 2d **size gate** (Block D, the user's
D1). `outputs/` and `figures/` are gitignored and regenerable.

**Sample:** the repo rule, 2356 of 2397, on a merged 30-radius grid from
0.673 to 148.22 kpc (see A1); 2354 survive the truth-finiteness cut on it.

---

## Block A1 — the coordinate, and what the data can support

### The size-relative coordinate cannot reach inside 1 R50 at high z — on the stored grid

The innermost aperture of the stored 24-radius curve of growth is a fixed 2 kpc.
That is 0.17 R50 at the median z = 0.4 galaxy and **0.66 R50 at the median z = 2
galaxy**. Fraction of galaxy-epochs whose `edge × R50` falls inside the measured
range, worst epoch:

| edge | stored CoG, 2–148 kpc | density-rebuilt CoG, 0.673–160 kpc |
|---|---|---|
| 0.25 R50 | 3% | 56% |
| 0.5 R50 | **26%** | **100%** |
| 0.75 R50 | 62% | 100% |
| 1 R50 | 87% | 100% |
| 4 R50 | 100% | 100% |
| 8 R50 | 84% | 87% |

On the stored grid alone nothing below 1 R50 is usable at z = 2, and the failure
is worse than a uniform loss: it drops the **small** galaxies preferentially, a
mass-dependent selection. **The user pointed out that the isophote density
reaches 0.673 kpc for 100 per cent of galaxies at every epoch.** The curve of
growth rebuilt from it extends the usable window to **0.5–6 R50 at full
coverage**. `extended_cog` splices it inside 2 kpc onto `cog_provided` outside,
rescaled to agree exactly at the splice (the two differ by a smooth −0.008 to
+0.009 dex over 2–103 kpc, §10.3 of the data note); outside 2 kpc the target IS
`cog_provided`, bit-identical, per D1.

**The caveat that travels with every inner shell:** `sigma_resolved` is False
inside ~2.4 kpc — at z = 2, 0 per cent resolved below 1.4 kpc, 60 per cent at
2.0 kpc. The density there is the fitted ellipse family evaluated inward over
the same pixels: real, but not independent. Carried as a flag, not a blocker.

### The coordinate does what it is for, where it applies

On a self-similar synthetic family the R50 coordinate cuts the epoch-to-epoch
spread of shell mass fractions from **12.6 to 1.3 points** (10×). The 1.3 that
remains is the fixed 148 kpc aperture, not the coordinate: both the normalising
total and R50 itself are measured inside it, and it encloses 91.3 per cent of
the biggest galaxies against 97.5 per cent of the smallest. Two things the
selftest found by failing: normalising a size-relative shell by `M*(<148 kpc)`
reintroduces the very epoch trend the coordinate removes (now normalised by the
mass inside the outermost size-relative edge), and a size-relative query lands
at an epoch-dependent position between grid points so interpolation must be
log–log. The rule that makes the coordinate honest — **it is built from the
truth's R50, never the model's** — is asserted: a model 3.7× the truth leaves
the shell radii bit-identical.

### A correction to exp72, and it is the user's density profile that exposed it

exp72 reported the galaxy nearly self-similar in R50 units — the 0.5–1 R50
shell holding 17.1–18.8 per cent of the galaxy across five epochs, a 1.7-point
spread. That shell was measured for **26 per cent** of galaxies at z = 2 on the
stored grid, and the survivors are the largest, which most resemble
low-redshift galaxies.

| 0.5–1 R50 shell, % of galaxy | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | spread |
|---|---|---|---|---|---|---|
| exp72, stored grid (26% covered at z=2) | 17.1 | 17.2 | 17.6 | 18.0 | 18.8 | 1.7 |
| **corrected, merged grid (100%)** | 17.1 | 17.5 | 18.4 | 21.6 | **26.4** | **9.3** |

The reading splits in two. **Outside 1 R50 the galaxy really is nearly
self-similar** — 3.0 / 3.4 / 3.7 points for the 1–2, 2–4, 4–8 R50 shells against
up to 12.4 in fixed kpc; those shells were always fully covered and are
unchanged. **Inside 1 R50 it is not**: massive galaxies were genuinely more
centrally concentrated at z = 2 in units of their own half-mass radius. That is
physics for the model to reproduce, not geometry to normalise away. Corrected
inline in exp72's README and in C21.

---

## Block A2 — the risk test: the coordinate, or the shared law?

exp63 fitted the same twelve-parameter model with one θ for five epochs and,
separately, with a θ per epoch (time exponents frozen). By exp63's own gate the
per-epoch fits are 2.3–2.7× better. The risk named in the plan: if that gap does
not narrow under the new coordinate, the shared law is the limit and no
objective will fix it.

### The metric decided more than the coordinate did

A per-galaxy rms — four shells or twenty, relative or mass-weighted — puts the
joint fit and all five per-epoch fits **within 2 per cent of each other**. The
gap exp63 measures is a **tercile-median** statistic, and a per-galaxy rms trades
those median offsets away (`loss-is-blind-to-the-binned-gate`). Had the risk test
used a per-galaxy metric it would have reported "no gap to close" and cancelled
Blocks B–D on an artefact of its own choosing.

Under the binned tercile-median gate, 20 matched shells over 0.5–6 R50 in both
coordinates, the fixed-kpc edges set to the median R50 at z = 0.4 times the same
multiples:

| gap = loss(shared θ) / loss(that epoch's own θ), z≤1.5 | fixed kpc | R50 shells |
|---|---|---|
| binned, relative residual | 1.28 | **1.07** |
| binned, mass-weighted | 1.46 | **1.14** |
| per-galaxy, either | 1.00 | 1.01 |

**Gate A was written ambiguously.** As a ratio, 1.28 → 1.07 is −16.6 per cent
(below the 25 per cent threshold: skip B–D). As the *cost of sharing* (gap − 1),
0.28 → 0.07 is **−75 per cent** (far above: continue). The cost reading is the
meaningful one, and the plan's rule for an ambiguous gate is to take the
conservative branch and say so — B–D were not skipped.

**The caveat that limits it more than the ambiguity:** the five per-epoch fits
were optimised under exp63's *fixed-kpc* objective, so they are not the
per-epoch optimum under an R50 one. It shows — the R50 binned gap is **0.91 /
0.70 / 0.57** at z = 1.0 / 1.5 / 2.0, i.e. the shared θ *beats* the
epoch-specialised models there. Part of the narrowing is a mis-specialised
comparator. Settling Question 1 needs five single-epoch refits under the R50
objective (~75 min at 0.455 s per evaluation; not spent this session — Block C's
transfer matrix answers "is this coordinate better" more directly). **Open.**

### The per-epoch optima below z = 2 lie on a straight line — D3's lever is small

Because `b_c` is frozen at 0 in the per-epoch fits, `log_f_c` *is* the compact
size in log₁₀ kpc at each epoch — a non-parametric measurement of the size–time
law. Fitted on z ≤ 1.5 (z = 2 excluded: its parameter sequence breaks there and
C18 showed why), a straight line in log₁₀(1+z) leaves:

| parameter | z=0.4 → z=1.5 | line residual | as fraction of range |
|---|---|---|---|
| `log_f_e` (extended size) | −0.683 → −0.491 | **0.012 dex (3%)** | 0.06 |
| `log_f_c` (compact size) | 0.467 → 0.341 | **0.023 dex (6%)** | 0.19 |
| `m_half` | 11.45 → 12.45 | 0.135 | 0.14 |

One power of (1+z) already describes the size evolution below z = 2 to a few per
cent. A quadratic column was computed and then removed: three parameters through
four points removes residual by construction.

---

## Block B — the two down-weighting schemes head to head (D2)

Scheme **A**: each shell's relative residual weighted by its median mass share
(no free parameter). Scheme **B**: shells below a threshold `t` dropped (one
parameter). Both on the relative residual, against uniform weights as the
reference; the **mass-weighted residual** carried as a fourth entry because it
*is* a weighting — (Δ/total)² = ε²φ², scheme A with the mass share squared. (A
first draft stacked A on top of it, weighting by φ³; the smoke run showed the
outer third under-weighted to a third of its mass share, which is how that was
caught.) 2354 galaxies, 20 shells.

**Allocation** — share of the loss on the outer third of the shells over those
shells' share of the mass (1.00 = proportional; not the target where the model
is genuinely worse there, but the right diagnostic for where attention goes):

| R50 shells | z=0.4 | z=1.0 | z=2.0 |
|---|---|---|---|
| uniform (reference) | 2.24 | 2.68 | 3.31 |
| A smooth | 1.96 | 2.29 | 2.08 |
| B gated, t=2% | 2.24 | 2.68 | 1.32 |
| B gated, t=3% | 0.51 | 0.00 | 0.00 |
| **mass-weighted residual** | 1.35 | 1.25 | **0.65** |

**Usability** — effective number of galaxies carrying the loss, of 2354:

| | R50, z=2 | fixed kpc, z=2 |
|---|---|---|
| uniform | 1359 | **2 (78% on two galaxies)** |
| A smooth | 1433 | 2 (72%) |
| B gated, t=1% | 1359 | 981 |
| **mass-weighted** | 1610 | **1720** |

**The user's claim 2, confirmed catastrophically:** on fixed kpc with the
relative residual, **two galaxies carry 78 per cent of the whole population's
loss at z = 2**, through near-empty outer shells. The mass-weighted residual
restores it to 1720.

**Sensitivity** (symmetric probes, ±0.05 dex, as per cent of the exp63 mean's
loss; a new **size** probe rescales the galaxy along the mass–size relation at
fixed total mass, the failure exp72's refit actually had):

| R50 shells, z=2 | amplitude | shape | size | size / amp |
|---|---|---|---|---|
| uniform | 2.20% | 0.09% | 1.75% | 0.80 |
| A smooth | 2.38% | 0.05% | 1.32% | 0.56 |
| mass-weighted | 2.38% | 0.05% | 0.65% | **0.27** |

**Verdict.** **B is disqualified** by the rule fixed before any number was seen:
its allocation varies by 136 per cent (R50) and 215 per cent (fixed kpc) across
the threshold sweep, and at t = 3 per cent it drops every shell at z = 2. **A
barely helps** (2.24 → 1.96 at z = 0.4; nothing at all at fixed kpc z = 2). **The
mass-weighted residual wins** on allocation and usability at every epoch, at one
measured cost: about 3× less sensitive to a pure size error relative to
amplitude at z = 2. That cost is exactly what the Block D size gate is built to
detect, which is why Block C fits under the mass-weighted residual and lets the
gate judge it.

---

## Block C — the refit: first a negative caused by a recorded design error; then, repaired, a null result

`fit_re.py`: 20 R50 shells at 0.5–6 R50, mass-weighted residual plus exp63's
binned term, normalised at the nested incumbent (null = 10.000). Four starts.

| start | loss | | start | loss |
|---|---|---|---|---|
| nested | 6.639 | | default | 6.335 |
| **exp63** | **6.106** | | jitter0 | 6.312 |

Not a single basin (exp63 and exp72 had 8 of 8 agree). The best, from exp63's
own θ, is 39 per cent below the null with the five epochs *balanced* — 1.31 /
1.15 / 1.19 / 1.14 / 1.31 — and `n_c` = 0.97, off the bound it railed against in
every previous fit. Under this objective it beats exp63 by 23 per cent.

**And it is a bad model.** The judge (`eval_re.py`) and the basin comparison:

| | z=0.4 | z=1.0 | z=2.0 |
|---|---|---|---|
| R50, model / truth − 1 | **−22%** | **−26%** | **−24%** |
| mass inside 2 kpc, (model − data)/data | **+102%** | +82% | +53% |
| mass inside 103 kpc | +14% | +16% | +17% |
| exp63's four-term loss | 56.2 | 27.6 | 21.3 (exp63 itself: 3.5 / 3.0 / 3.0) |

Under exp63's loss the refit scores **178 against exp63's 15.3** — eleven times
worse. Every one of the four optima shows the same pattern (2 kpc +97 to
+173 per cent; R50 −15 to −36 per cent), so it is the objective, not a start.

### The cause, proven rather than argued

`SizeGrid.shells()` was `np.diff(cum)` over the edges 0.5 … 6 R50: **twenty bins
and no inner aperture.** Mass added inside the first edge shifts every
cumulative value by the same constant and leaves every difference unchanged —
the objective is not insensitive there, it is exactly blind. Measured on the
refit's own prediction:

| perturbation | loss | change |
|---|---|---|
| as fitted | 6.1061 | — |
| **+100 per cent of the mass inside 0.5 R50** | 6.1194 | **+0.013 (0.2 per cent)** |

A third of the galaxy's mass at z = 0.4 (33 / 33 / 32 / 28 / 24 per cent from
z = 0.4 to 2) sits inside 0.5 R50, and doubling it costs the objective 0.2 per
cent. The fit put it there. Excess central mass pulls the half-mass radius
inward, which is the −22 to −26 per cent in R50; the +14 to +17 per cent at
103 kpc is consistent with a matching blindness beyond 6 R50 (8–12 per cent of
the mass), but my probe of that edge was confounded by interpolation and does
not prove it, so it is stated as consistent, not shown.

**This is a repeat of a recorded mistake.** Memory
`density-objective-cannot-see-the-centre` and open question D1 record exactly
this mechanism, verified to 1e-13 in `hongshao/objective.py`, and the fix — the
`shells` convention that includes M*(<R_first) as the first bin. I built
`shells()` as a bare `np.diff` without consulting it. The coordinate now includes
the inner aperture as bin 0 (selftest claim H: mass added inside the first edge
moves the aperture residual and nothing else), and the fit has **not** been
re-run — that costs ~1.5 h and is the user's call.

### What Block C does and does not say

It does **not** say the size-relative coordinate is wrong: Blocks A and B stand,
and exp63's own θ scored 21 per cent better than the incumbent under this
objective *before* any fitting. It says a fit will find any blind spot an
objective has, and that this one had a third of the galaxy in it. The repaired
objective (aperture + 20 bins, optionally an outer envelope) is what a refit
should use.

### The refit under the repaired objective (2026-09-02 afternoon, the user's decision)

Same script, same four starts, same evaluation cap; the only change is that the
shells now include the mass inside 0.5 R50 as bin 0 (selftest claim H). Block C's
blind-fit outputs are kept under `outputs/*_blockC_blind.*`.

| start | loss | evals | note |
|---|---|---|---|
| nested | 8.478 | 1847 | `m_half` railed, as in every previous fit |
| **exp63** | **7.339** | 1951 | converged |
| default | 8.254 | 3004 | stopped at the cap |
| jitter0 | 8.458 | 3134 | stopped at the cap |

Null 10.000; exp63's own θ scores 7.997 under this objective. So the refit is
**8 per cent under exp63 on its own objective** (Block C had claimed 39 per
cent under the null; the difference between 39 and 27 was the blind spot). The
far starts stopped at the evaluation cap 12–15 per cent above the best, so the
basin is not proven global; it is the same basin the blind fit was not in.

**The fit is a small perturbation of exp63.** Every parameter is within about
10 per cent of exp63's value; only four moved by more than 0.05 — `m_half`
11.757 → 11.664, `d_split` 0.236 → 0.141, `log_f_c` 0.946 → 1.040, `b_e`
−1.195 → −1.261. The blind fit had moved five parameters far away (`a_z` 0.35 →
0.05, `a_Mz` +0.12 → −0.08, `d_split` 0.24 → 0.97, `b_c` −0.80 → −1.37, `n_c`
0.53 → 0.97). **An objective that sees sizes directly, and weights every galaxy
by its own scale, lands on exp63's parameters** — including the size-time
exponents `b_c`, `b_e` that Block E was going to enrich.

**And it is not an improvement.** The judge (`eval_re.py`, `outputs/eval_re.log`,
`figures/qa/*R50_refit*`):

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | exp63 joint, same row |
|---|---|---|---|---|---|---|
| exp63's four-term loss | 3.27 | 3.24 | 3.31 | 3.24 | 3.05 | 3.47 / 3.00 / 2.96 / 2.95 / 2.95 (**16.10 vs 15.33, +5.1%**) |
| median R50, model / truth − 1 | −1% | 0% | −1% | +6% | **+11%** | 0 / +1 / −1 / +3 / +5% |
| size gate, OFFSET passes (of 15) | 14 | | | | | 14 (the R20 row at z = 2 fails for both, at +0.06 dex) |
| M(<10 kpc) median bias, fitting sample | +0.7% | +3.7% | +6.5% | +3.6% | +0.2% | −1.7 / +1.2 / +4.2 / +1.7 / −1.7% |
| M(<100 kpc) | −1.0% | +2.7% | +3.6% | +2.0% | −2.5% | −2.5 / +1.5 / +2.1 / +0.2 / −4.6% |
| M(<1 kpc), density-rebuilt grid | **+46%** | +39% | +32% | +16% | +15% | +35 / +29 / +22 / +8 / +7% |
| M(<148 kpc) | −1.1% | +2.5% | +2.9% | +1.7% | −3.0% | −2.6 / +1.3 / +1.6 / −0.1 / −4.7% |

Read across: the R50 objective buys the total mass at z = 0.4 and z = 2 (about
1.5 points each at 148 kpc) and pays for it inside 10 kpc at z = 0.7–1.5, at
1 kpc everywhere (+46 against +35 per cent at z = 0.4), and in R50 at z ≥ 1.5
(+11 against +5 per cent at z = 2). Nothing the objective could not see before
becomes right; what was right becomes a little less so. The z = 2 halo-mass
tilt of M*(<103 kpc) is −0.147 against exp63's −0.122 on the fitting sample
and +0.036 against +0.061 on the mh-complete subset — a class property, and it
did not move by more than the shared-law compromise moves it.

**What Block C says, now that it is honest.** The size-relative, mass-weighted
objective is C20's result a second time: **a change of objective is a dial on
the shared-law compromise, not a new capability.** Its one positive statement
is about exp63, not about itself — exp63's solution is robust to a very
different way of scoring the same data, so it is not an artefact of exp63's
objective. The residuals it leaves (a third too much mass at 1 kpc at z = 0.4,
3–5 per cent too little total at z = 2, the high-z size offsets, and every
width ratio) are the same ones every objective leaves; they belong to the model
class and its stochastic layer (C13, C16), not to the loss. Block E's richer
size-time law is doubly unmotivated: A2 measured its lever as small, and this
fit, which can see sizes directly, returned exp63's exponents.

## Block D — R20, and the fractional-size distributions as a numbered gate (D1)

`hongshao/qa.py`: `SIZE_FRACTIONS` gains 0.2; `size_gate()` scores, per
fractional size and epoch, **offset** (|median Δlog R| ≤ 0.05 dex — 12 per cent
in size, between exp63's 0.02 and the chi² refit's 0.10) and **width** (the
model's size scatter at fixed mass within 20 per cent of the truth's — exp61
measured narrowing to 0.46 where 0.54 was honest). Scored at fixed **stellar**
mass always and at fixed **halo** mass when `halo_mass_epochs` is passed. The
selftest asserts: the identity passes everything; a 1.6× size offset fails
every offset; a model with the right median but sizes 40 per cent too alike at
fixed mass passes offset and fails width. (A first draft built the narrowed
model through `_assign`, which rank-maps onto fixed marginals and erases any
width change by construction — it measured a width ratio of 1.0001.)

### The first use: the gate catches what it was built to catch, on the right sub-gate

Three models whose size behaviour was already known, 2354 galaxies, scored on
the merged grid (where R20 is measured) and on the standard grid (where the
truth's R20 falls inside the innermost 2 kpc aperture for **up to 93 per cent**
of galaxies at high z and is extrapolated):

| OFFSET sub-gate, passes of 15 | standard grid | merged grid | where it fails |
|---|---|---|---|
| exp63 joint | 14 | **15** | only R20 at z=2 on the standard grid (+0.063 dex, extrapolated); measured, +0.004 |
| chi² refit (exp72) | 8 | 8 | R50 at z≥1.5: **+0.087, +0.104 dex** (22%, 27% too big); R80 likewise; R20 from z≥1.0 |
| nested incumbent | 10 | 10 | z≥1.5: R50 +0.077 / +0.072, R80 +0.058 / +0.140 — the incumbent's high-z sizes are 18–38% too big, which exp63's fit repaired and the χ² undid |

**Three things this settles.** The gate discriminates on the sub-gate it was
designed to: the χ² refit's known failure — z = 2 galaxies 27 per cent too big —
appears as an OFFSET failure at exactly the epochs and sizes where it should,
and exp63's fit, which holds R50 to 5 per cent, passes every offset. **The
density-rebuilt inner curve turned R20 from an unreliable row into a passing
one**: on the standard grid exp63's R20 fails at z = 2 by +0.063 dex where it is
extrapolated; measured on the merged grid it passes at +0.004. And at fixed
**halo** mass the offsets are the same to three decimals (the size–mass relation
is not what moves), so the halo-mass conditioning adds nothing on offset and is
kept for width, where it does.

**The WIDTH sub-gate fails 0 of 15 for every model, everywhere** — and the
redshift trend seen on the smoke sample holds on the full one:

| width ratio, model / truth, fixed stellar mass | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| exp63 joint, R20 (merged) | 0.76 | 0.71 | 0.67 | 0.63 | 0.59 |
| exp63 joint, R50 | 0.63 | 0.51 | 0.41 | 0.31 | **0.30** |
| exp63 joint, R80 | 0.52 | 0.41 | 0.30 | 0.21 | **0.16** |
| chi² refit, R50 | 0.40 | 0.33 | 0.28 | 0.24 | 0.24 |
| nested incumbent, R50 | 0.48 | 0.39 | 0.32 | 0.24 | 0.21 |

A conditional-mean model cannot be as diverse at fixed mass as the population it
was fitted to — C16, and the diversity the stochastic layer exists to supply.
**What is new**: the narrowing is a strong function of both redshift and radius.
At fixed stellar mass exp63's galaxies of a given mass are 63 per cent as diverse
in R50 as the truth's at z = 0.4 and **30 per cent** at z = 2; in R80, 16 per cent
at z = 2. At fixed **halo** mass it is narrower still (R50 0.52 → 0.29), which is
what a deterministic model of halo mass should show — at fixed input its only
scatter is what the assembly history's shape supplies. So the gate counts the
two sub-gates separately: offset discriminates between mean models; width is the
expected reading for any mean model and becomes a verdict only when the layer's
draws are scored. exp61 measured the narrowing at one epoch; this is its
redshift dependence, and the z = 2 outskirts are where the mean is furthest from
the population it describes.

## Block E — the richer size-time law (D3): evaluated, not fitted

A2 measured it. Below z = 2 the per-epoch size optima sit within 0.023 dex
(compact) and 0.012 dex (extended) of a straight line in log₁₀(1+z) — 6 and 3
per cent in size. Register entry D2 found the efficiency law's time dependence
had a 0.2-percentage-point budget; the size law's is of the same order. The fit
Block E would gate is not demanded by the evidence, so it was not spent. What
survives of D3 is its first half: keep the shared law, and drop the z = 2
per-epoch number as a target (exp63 Stage 5i point 3, confirmed by exp71).
