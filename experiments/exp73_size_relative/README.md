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

## Block C — the refit *(fit running; this section is filled when it lands)*

`fit_re.py`: 20 R50 shells at 0.5–6 R50 (truth-only), mass-weighted residual,
**plus exp63's binned tercile-median term** translated to the coordinate (A2:
without it the objective cannot see the epoch tension), each normalised at the
nested incumbent so every epoch starts at 2.000 and the null is 10.000. Same
twelve parameters, bounds, starts, sample, engine and five epochs as exp63 and
exp72; the objective is the only change. Four starts: nested, exp63's θ,
default, jitter0.

Before any start ran: the nested start normalises to 2.000 per epoch to
**0.0e+00**, and **exp63's own θ already scores 7.94 against the null's 10.00**
under this objective (2.15 / 1.62 / 1.44 / 1.33 / 1.40 per epoch) — it is already
21 per cent better than the incumbent by this measure, so the two objectives are
far from orthogonal.

*Pending: the fitted θ; the transfer matrix in both coordinates against exp63,
the chi² refit and the null; the mass–size relation; the size gate; the profiles
in per cent on the merged grid; the battery; exp71's diagnostics.*

---

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
