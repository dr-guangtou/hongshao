# exp57 — an empirical expansion term: deposits grow after they are laid down

Branch `exp57-expansion-term`. Plan:
`doc/plans/2026-08-26-exp57-expansion-term.md`. Problems:
[`PROBLEMS.md`](PROBLEMS.md).

**Status: COMPLETE.** Stages 0–9 done and gated.

---

## Why this experiment exists

exp54 established that its own remaining defect is **outside its model class**.
With every deposit non-negative and its profile fixed at the moment of
deposition, `M*(<R,t)` is non-decreasing in `t` at every radius and every
parameter value — while **42% of these galaxies LOSE a median 14% of their
central stellar mass** between redshift 2 and 0.4. Four enrichments (exp54
Stages 3.5 to 3.8) failed against a target the model class forbids.

An expansion term is the smallest change that admits a declining centre: each
deposit's radii grow after it is deposited. Mass is conserved, nothing is
removed, nothing goes negative, no stellar information enters — and old
deposits can spread outward faster than new ones arrive, so the mass inside a
fixed small radius can fall.

## The constraint that shapes the whole design

**The program's first model was deposition + transport, and it failed in a way
that must not be repeated** (user, 2026-08-26). `exp52_growth_mechanism`
diagnosed it precisely: **the right source with roughly ten times too much
reach.** 93% of the mass it moved came from inside 5 kpc — correct, that is
where AGN-driven adiabatic expansion operates — but 58.5% was delivered beyond
50 kpc, 29.4% beyond 148 kpc (outside the measured profile entirely) and 12.0%
beyond 500 kpc, where the normalisation discarded it. By redshift 0.7 to 0.4,
**96.5% of that model's outskirt growth was redistribution of stars it already
had**, with the crossover at z ≈ 1.2 — exactly where minor-merger accretion
should be taking over.

The root cause was the **deposition schedule**, not the transport term: the
fitted efficiency window deposited 1.5% of the stellar mass at z<1 while the
halo assembled 47% of its mass there, so draining the centre was the only way
to grow an outskirt.

**exp54 does not have that condition** (measured 2026-08-26, before any code):
it deposits **29.3%** of its stellar mass at z<1, at median deposit half-mass
radii of **35 to 50 kpc**, and its own deposition supplies **×2.842** over
z=2→0.4 against a measured ×2.323 — with no pinning term at all.

The **mirror-image trap** is present instead: that ×2.842 against ×2.323 is a
22% over-supply, and expansion moves mass out of the 100 kpc aperture where
`score_F` is normalised and therefore blind. Expansion could be fitted as an
amplitude regulator rather than as a mechanism. Gate G3 exists for that.

---

## Stage 0 (2026-08-26) — cost and the nesting gate (`stage0_cost.py`), COMPLETE

### The nesting gate

Every law must reproduce exp54's incumbent exactly when switched off, or no
later comparison means anything. Measured against
`stage35_time_law.StackedProblem` at the incumbent `gompertz_log-E2-S2`
(loss **1.874253**, 2397 galaxies, 7595 galaxy-epochs):

| law | NaN pattern | profiles | loss |
|---|---|---|---|
| `""` | exact | 1.74e-15 rel (7.8 ulp) | 1.874252997912 (Δ = 2.2e-16) |
| `X1` at `A = 0` | exact | 1.74e-15 rel (7.8 ulp) | Δ = 2.2e-16 |
| `X2` at `A = 0, p = 1` | exact | 1.74e-15 rel (7.8 ulp) | Δ = 2.2e-16 |
| `X0` at `alpha = 0` | exact | 1.74e-15 rel (7.8 ulp) | Δ = 2.2e-16 |

Not bit-for-bit, and the specification that demanded it was wrong — see
[P1](PROBLEMS.md). Expansion forces one matrix product per epoch where exp54
formed one over all five, so the same sum is accumulated in a different order.

### The cost of the model class change

Measured, single-threaded, 10 evaluations each, on all 2397 galaxies:

| evaluator | per evaluation | relative |
|---|---|---|
| exp54 `StackedProblem` | **112.3 ms** | 1.00× |
| `ExpandingProblem`, `X1` | **483.6 ms** | **4.31×** |
| `ExpandingProblem`, `X2` | 489.6 ms | 4.36× |
| `ExpandingProblem`, `X0` | 492.4 ms | 4.39× |

The technical note guessed "several times more per evaluation"; it is 4.3×, and
that number now sets the schedule. A Nelder-Mead fit converges in roughly 2300
evaluations, so **18.5 min per start, ~74 min for a four-start fit**, and eight
candidates run in parallel on ten cores in about the same wall clock. A Stage
3.9-style profile likelihood would be **6.8 h per parameter** — affordable for
one parameter, not for seven.

**The saving that keeps it to 4.3× rather than 5×** is exact and worth
recording: `model.solve_u`, the expensive inversion, depends on the two radii
only through their **ratio**, which homologous scaling leaves unchanged. So it
is evaluated once for all five epochs and only the radius lookup repeats.
`expand.cog_scaled` carries the proof, and the same identity is why homologous
expansion conserves each deposit's mass exactly — `F(g·r_trunc) = 1` still
holds. Verified to 0.0 and 3.3e-16 in `expand.py`'s self-check.

---

## Stage 1 (2026-08-26) — the contract change, and the capability check (`stage1_contract.py`), COMPLETE

### Branch 1 — exp54's monotone-in-time property, measured rather than argued

exp54 Section 8.10 established analytically that with the expansion off,
`M*(<R,t)` is non-decreasing in `t` at every radius and every parameter value.
It was never a coded assertion. Checked here over **230,112 galaxy-radius-epoch
pairs**: the most negative relative change is **0.000e+00** and **zero** pairs
fall by more than 1e-12. The property is exact.

### Branch 2 — is the required decline REACHABLE, and at what price?

**The target had to be restated first, and getting it wrong would have
mis-selected the expansion strength by a factor of two — see
[P3](PROBLEMS.md).** Recomputed on this sample, measured `M*(<4.92 kpc)` from
redshift 2 to 0.4:

| | measured |
|---|---|
| median over **ALL** galaxies | **+0.0368 dex** — the centres GROW on the median |
| fraction that **decline** | **41.8%** (Stage 3.4 said 42%) |
| median among the **decliners** | **−0.0661 dex** (Stage 3.4 said −0.066) |
| 10th to 90th percentile | **−0.099 to +0.438 dex** |

**So the target is a SHAPE, not a shift**: centres that grow by +0.037 dex on
the median while 42% of them fall. Scanning the expansion strength with
everything else held at the incumbent:

| `A` | oldest deposit grows | median, ALL | declining fraction | median of decliners | loss |
|---|---|---|---|---|---|
| 0.00 | 1.00× | +0.1354 | 0.0% | — | 1.8743 |
| 0.50 | 1.50× | +0.0670 | 6.5% | −0.0124 | 2.0240 |
| 0.75 | 1.75× | +0.0375 | 22.2% | −0.0175 | 2.2046 |
| 1.00 | 2.00× | +0.0097 | 43.3% | −0.0264 | 2.4338 |
| 1.50 | 2.50× | −0.0402 | 72.9% | −0.0563 | 2.9849 |
| 2.00 | 3.00× | −0.0829 | 89.1% | −0.0888 | 3.5978 |
| **DATA** | | **+0.0368** | **41.8%** | **−0.0661** | |

**THE CAPABILITY IS REAL BUT IT DOES NOT COME AS ONE NUMBER.** Reproducing each
target on its own needs

| target | required `A` |
|---|---|
| the median over all galaxies (+0.0368) | **0.76** |
| the declining fraction (41.8%) | **0.98** |
| the median among decliners (−0.0661) | **1.65** |

a span of **2.2×**. The reason is measurable and is not about expansion at all:
the model's 10th-to-90th spread at `A = 0` is **+0.070 to +0.248 dex against
the data's −0.099 to +0.438 — 0.33× the width**. **Expansion shifts a
distribution; it cannot widen one**, so that ratio is a ceiling on how well any
single global `A` can match the population's central evolution. This is exp54
open question **C4** (the predicted growth distribution is half as wide as the
simulation's) reappearing as the binding constraint on a different mechanism.

**And the capability is expensive.** Held at the incumbent, the loss rises
monotonically with `A`: 1.8743 → 2.2046 at the `A = 0.76` that matches the
median, → 2.4338 at the `A = 0.98` that matches the declining fraction. Whether
re-optimising the other seven parameters pays that back is Stage 3's question,
and the answer must not be read off this scan — exp53 found a frozen slice and
an honest refit **57 percentage points apart** on exactly this kind of number.

---

## Stage 3 (2026-08-26) — the fits, and THE RESULT SO FAR

### What the expansion term buys

| model | loss | vs incumbent | profile-shape error | `score_A` | fitted |
|---|---|---|---|---|---|
| exp54 incumbent | 1.874253 | — | 13.787% | 1.0558 | — |
| **`X1` bounded** | **1.840466** | **−1.80%** | **13.454%** | 1.0570 | `A = +1.4657` |
| `X0` power law (the OLD form) | 1.840977 | −1.78% | 13.478% | 1.0560 | `alpha = +0.3811` |

`X1` reproduced `A = 1.4657` from **four independent starts** — including one
launched from *contraction*, `A = −0.3` — with a start-to-start spread of
**2.9e-11**. In words: **the oldest deposits end up 2.47 times larger than they
were laid down.**

**This is the largest gain any exp54 enrichment has produced.** Stage 3.5's
entire budget for a richer time law was 0.18 percentage points of shape error,
and a completely *free* shared curve reached only 13.62%; Stage 3.8's compact
channel returned the incumbent's loss to six decimal places. One expansion
parameter reaches **13.454%**.

**And the gain is in the SHAPE, not the amplitude.** `score_A` moves 1.0558 to
1.0570 — very slightly *worse*. So this is not gate G3's amplitude-regulator
trap: expansion is not being fitted to dispose of exp54's 22% mass over-supply
past the 100 kpc blind spot.

**Bounding the reach costs nothing in loss.** `X1` (bounded by construction)
edges out `X0` (the old unbounded power law) on both loss and shape error. The
declared control did its job: whatever is wrong with `X0`, its extra freedom is
not buying anything.

### THE FAILURE — and it is the experiment's main finding

**G2 fails, and it fails worse than the model this experiment was designed not
to repeat.**

| epoch pair | displaced mass beyond 50 kpc | beyond 148 kpc | beyond 500 kpc |
|---|---|---|---|
| **preregistered limit** | **25%** | **10%** | — |
| exp52's failing first model | 58.5% | 29.4% | 12.0% |
| `X1`, z=2.0 → 1.5 | 43.5% | 22.7% | 8.4% |
| `X1`, z=1.5 → 1.0 | 57.4% | 31.3% | 13.8% |
| `X1`, z=1.0 → 0.7 | 70.9% | 41.1% | 19.4% |
| **`X1`, z=0.7 → 0.4** | **85.0%** | **52.1%** | 25.4% |
| `X0`, z=0.7 → 0.4 | 72.5% | 42.1% | 18.9% |

**Why, and it is structural rather than a bad parameter value.** `A = 1.4657`
means no deposit ever grows by more than **2.47×** — the factor really is
bounded, exactly as designed. But homologous scaling multiplies *every* radius
by that factor, and this deposit family has a genuine **power-law tail**, so
material already at 50 kpc lands at 123 kpc however modest 2.47 sounds.
**Bounding the factor does not bound the reach in kpc.** Both functional forms
fail, so it belongs to homologous expansion, not to a particular time law.

**Independently confirmed by a number with no threshold of mine in it.** Gate
G1b computes, with the same operator on model and data, the fraction of a
galaxy's added mass that lands beyond 50 kpc:

| epoch pair | data | null | `X1` |
|---|---|---|---|
| z=1.5 → 1.0 | 0.275 | 0.230 | 0.244 |
| z=1.0 → 0.7 | 0.320 | 0.297 | 0.325 |
| **z=0.7 → 0.4** | **0.366** | **0.364** | **0.420** |

The null was essentially perfect at the last epoch pair; `X1` overshoots. The
gate and the data agree.

### What did NOT fail

**G1 passes comfortably.** The expansion share of `M*[50,100]` growth runs
**12.9% to 18.5%**, against exp52's 76.9% → 96.5% ramp. New deposition still
dominates outskirt growth at every epoch, which is what the physical picture
requires. **The first model's *dominance* failure does not recur** — exp54's
deposition schedule, measured at the top of this file, is why.

**G5 is split, and two of its three parts pass well:**

| | null | `X1` | requirement |
|---|---|---|---|
| (a) central-error span, DECLINING galaxies | 0.270 | **0.169** | close materially — **PASS** |
| (a′) the same, all galaxies | 0.145 | **0.049** | — |
| (b) span, NON-declining galaxies | 0.045 | **0.076** | ≤ 0.060 — **FAIL** |
| (c) core density gap vs measured | +0.0455 | **−0.0019** | shrink — **PASS**, essentially exactly |

(c) is worth stating plainly: the model's median core `dlogSigma` over the long
baseline now matches the measurement to **0.002 dex**, from 0.046 out. **The
defect exp54 spent four stages failing to move has moved.** (b) is the price —
a single global expansion strength shifts every galaxy, including the 58% whose
centres never needed shifting, exactly as Stage 1's spread argument predicted.

**G6, the outskirt structure, does not degrade.** `M[50,148]` population-summed
error improves from −0.076 to −0.055 at z=0.4. Expansion does not pay Stage
3.7's price of emptying the outskirts to fill the centre.

### G4 — is expansion just a heavier deposit tail in disguise? Partly.

exp53 measured that the first model's transport **substituted for a heavy
tail**, and exp54 Stage 3.9 measured that the deposit shape `c` is the only one
of the seven parameters the other six cannot absorb. So the check is: refit with
`c` frozen at the incumbent's `+0.7965` and see what happens.

| | loss | vs incumbent | shape error | fitted `A` | fitted `c` |
|---|---|---|---|---|---|
| incumbent | 1.874253 | — | 13.787% | — | +0.7965 |
| `X1`, `c` free | 1.840466 | −1.80% | 13.454% | **+1.4657** | **+0.6964** |
| `X1`, `c` FROZEN | 1.851403 | −1.22% | 13.582% | **+0.7603** | +0.7965 (held) |

**G4 passes on its preregistered criterion.** `c` moves by **−0.100**, well
inside Stage 3.9's one-percentage-point width for `c` of −0.25 / +0.51. The
deposit shape is not being dragged somewhere else to pay for the expansion.

**But the two are partially degenerate, and that must be said.** Freezing `c`
**halves** the fitted expansion strength (1.4657 → 0.7603) and gives up 40% of
the shape-error gain. So roughly half of what expansion buys is reachable by
making the deposit's tail heavier instead — and half is not: even with `c` held
still, expansion buys 1.22% of loss and 0.205 percentage points of shape error,
which is still more than Stage 3.5's *entire* budget for a free shared time law
(0.18 points).

**And the parameters move in a physically coherent direction.** With expansion
on, `log_f0` goes from −0.843 to −1.015: deposits are born **more compact** and
grow afterwards. That is the mechanism's own story, not a numerical accident.

---

## Stage 6 (2026-08-26) — `X3`, the fix the diagnosis points at

If the failure is that homologous scaling cannot bound the reach in **kpc**,
the fix is a remap that does. `X3` replaces "multiply every radius by `g`" with

```
r(R) = R [ 1 − (1 − 1/G) / (1 + (R/Rc)^2) ]        G = 1 + A (1 − t_j/t_k)
```

read as *the radius the material now at `R` came from*. At `R → 0` it gives
`R/G`: the core expands by `G`, with no hole opened at the centre. At `R ≫ Rc`
it gives `R`: **the outskirts are untouched**. That is what adiabatic expansion
is understood to do physically — it flattens the inner stellar profile, it does
not build a stellar halo at 50–150 kpc — and it is exactly what exp52's
diagnosis asked for.

It is **mass-conserving by construction**, being a monotone rearrangement of the
same profile rather than a reweighting, with the normalisation taken at the
halo-set truncation so `F(r_trunc) = 1` holds exactly. `A = 0` reproduces the
current model to 2e-16.

**It does not nest `X1`**, and two drafts of its docstring claimed it did before
the self-check falsified them — see [P5](PROBLEMS.md). `X1` carries a deposit's
truncation outward with its stars; `X3` holds it at the halo's `3 R200c` and
rearranges mass inside it. Both conserve mass exactly within their own
boundary, and they are compared by fitting both.

---

## THE VERDICT (2026-08-26)

### The scoreboard, ordered by the gates and never by the loss

| model | loss | shape error | G2 >50 kpc | G2 >148 | G5(a) declining | G5(b) not | G5(c) core gap | verdict |
|---|---|---|---|---|---|---|---|---|
| incumbent, expansion off | 1.874253 | 13.787% | — | — | 0.270 | 0.045 | +0.0455 | the reference |
| **`X3` core-only, SMALL core** | 1.833344 | **13.369%** | **2.2%** | **0.2%** | 0.210 | **0.039** | +0.0668 | **G2 PASS**, G5 ab**C** |
| `X0` homologous power law | 1.840977 | 13.478% | 72.5% | 42.1% | 0.151 | 0.079 | +0.0121 | G2 FAIL, G5 a**B**c |
| `X1` + `c` frozen (control) | 1.851403 | 13.582% | 81.3% | 44.9% | 0.190 | 0.052 | −0.0196 | G2 FAIL, **G5 all pass** |
| `X1` homologous, bounded | 1.840466 | 13.454% | 85.0% | 52.1% | 0.169 | 0.076 | −0.0019 | G2 FAIL, G5 a**B**c |
| `X2` homologous + time shape | 1.839900 | 13.463% | 76.1% | 45.0% | 0.156 | 0.079 | +0.0037 | G2 FAIL, G5 a**B**c |
| `X3` core-only, large core | **1.831172** | 13.436% | 53.4% | 8.2% | 0.158 | 0.086 | −0.0049 | G2 FAIL, G5 a**B**c |

Limits: G2 ≤ 25% and ≤ 10%; G5(b) ≤ 0.060. Lower case = that part of G5 passes.

### The four things this experiment establishes

**1. An expansion term works, and it is the first thing in this program to move
exp54's central defect.** The profile-shape error goes **13.787% → 13.369%**,
against a Stage 3.5 budget of 0.18 percentage points for a *free* shared time
law and a Stage 3.8 compact channel that returned the incumbent's loss to six
decimal places. The declining galaxies' central-error span closes **0.270 →
0.210**, and for the homologous variants as far as **0.151**.

**2. Bounding the expansion FACTOR does not bound the REACH — and that is a
property of homologous scaling, not of a bad parameter value.** `X1` was
designed so no deposit can ever grow by more than `1+A`; fitted, that is 2.47×,
and it still delivers **85% of the mass it moves beyond 50 kpc** — worse than
the exp52 model this experiment exists not to repeat (58.5%). `X0`, `X2` and
the large-core `X3` all fail the same way. Homologous scaling multiplies every
radius by the same factor, and this deposit family has a genuine **power-law
tail**, so material already at 50 kpc lands at 123 kpc however modest the
factor sounds.

**3. A non-homologous remap fixes it completely.** `X3` with a fitted core
radius of **2.61 kpc** delivers **0.6–2.2%** of displaced mass beyond 50 kpc and
**0.0–0.2%** beyond 148 kpc, against limits of 25% and 10%. Its expansion share
of outskirt growth falls to **0.0%** — all outskirt growth is new deposition,
which is what the physical picture requires — and the outskirt structure
*improves*, population-summed `M[50,148]` going from −0.076 to **−0.007** dex.
It does this by moving only **1.4%** of the stellar mass over the long
baseline: a tiny redistribution placed exactly where the defect is.

**4. The loss ranks these almost exactly backwards.** `X3`'s multi-start search
found **two basins** — a core radius of 2.6 kpc and a near-homologous one at
92 kpc — and the loss prefers the near-homologous one, **1.831172 against
1.833344**. The completed 9-start run reproduces both: starts 0, 4 and 7 land in
the small core, starts 1, 2, 5, 6 and 8 in the large one, and start 3 rails at
the `log_Rc` bound with a worse loss. Start-to-start spread **2.2e-2**, against
1e-11 for the single-basin laws — **a single start would have reported whichever
basin it happened to fall into.** Ranking by loss would have chosen the long-reach answer in every
comparison in this table. This program has now recorded that failure mode four
times (exp38's loss-optimal basin, exp49's opposing gradients, exp52's correct
profiles with the wrong mechanism, and now this).

### The tension that nothing resolved — and a correction to how I first stated it

**First statement, from the six fits alone**: every long-reach model passes
G5(c) and fails G5(b), and the one short-reach model does the reverse, so
"reach and the core-density target are in direct tension".

**That was too coarse, and Stage 8 refines it.** Sweeping the core radius `Rc`
with everything else held fixed, at BOTH fitted anchors, shows the reach gate is
satisfied over a wide range and the real conflict is between the two halves of
G5:

| `Rc` | G2 (reach) | G5(c) core gap | G5(b) non-decliners |
|---|---|---|---|
| ≲ 3 kpc | **pass** (≈2%) | **fails** (+0.07 to +0.10) | **pass** (0.024–0.038) |
| 4–40 kpc | **pass** (3–22%) | **pass** (+0.02 to −0.005) | **fails** (0.078–0.094) |
| ≳ 60 kpc | **fails** (41–99%) | pass | fails |

**So G2 is not the binding constraint once the mechanism is non-homologous** —
it holds for every `Rc` up to about 40 kpc. The conflict is that the core
radius which fixes the median core density (4–40 kpc) is larger than the one
which protects the galaxies that never needed correcting (≲3 kpc). Short reach
protects them and cannot move (c); intermediate reach moves (c) and over-corrects
them.

**Stage 8 is a CONDITIONAL scan and says so in its own output.** The other eight
parameters are held fixed, which is exactly the object exp54 Stage 3.9 showed
over-states a parameter's determination by 2.3× to 17.6×. It is used here only
for what a conditional slice can do honestly — show the shape of a trade along
one axis — never to locate an optimum, and it is drawn at both anchors so
dependence on where it started is visible rather than hidden. The two anchors
disagree on *where* the (b)/(c) crossover sits (≈3 kpc against ≈10 kpc) and
agree on *that* it exists, which is the part being claimed.

The **attribution test** says why, by switching the expansion off at each fit's
own base parameters:

| model | core gap (fitted) | the expansion alone | the refit alone |
|---|---|---|---|
| `X0` | +0.0121 | −0.0921 | +0.0588 |
| `X1` | −0.0019 | −0.1561 | +0.1087 |
| `X3` large core | −0.0049 | −0.1677 | +0.1174 |
| `X3` **small core** | +0.0668 | **−0.0248** | **+0.0461** |

**In every case the expansion does the work and re-optimising the other seven
parameters partially undoes it**, so none of these is a reparameterisation
carrying the term's credit. And it explains the small core's `(c)` failure
exactly: its expansion moves the gap −0.025 while the refit moves it +0.046, and
the compensation wins.

### The headline is robust to how it is measured

G2 is computed by summing signed mass changes over a chosen set of shells, so a
coarse grid could in principle inflate or deflate it. `check_g2_binning.py`
recomputes it from 5 shells to 60:

| model | as reported (9) | coarse (5) | fine (25) | very fine (60) |
|---|---|---|---|---|
| `X1`, beyond 50 kpc | 85.0% | 85.0% | 80.4% | 80.5% |
| `X1`, beyond 148 kpc | 52.1% | 52.1% | 49.2% | 49.3% |
| **`X3` small core, beyond 50 kpc** | **2.2%** | 4.9% | 1.8% | 1.7% |
| **`X3` small core, beyond 148 kpc** | **0.2%** | 0.4% | 0.1% | 0.1% |

`X1` fails the 25% limit on every grid; `X3` passes it on every grid; the factor
between them is 20 to 40 throughout.

### What is NOT claimed

- **Nothing is adopted.** `X3` small-core is the only candidate that passes the
  mechanism gates, but it fails G5(c), the loss prefers a different basin, and
  adoption is the user's call.
- **G5(b)'s failure for the homologous laws is not fixable by a global
  strength.** Stage 1 measured why: the model's spread is 0.33× the data's, and
  expansion shifts a distribution without widening it.
- **G5(b) fails for every model except `X3` small-core** (0.039, better than the
  null's 0.045) and the `c`-frozen control (0.052). Stage 1 measured why a single
  global strength cannot fix it: the model's spread is 0.33× the data's, and
  expansion shifts a distribution without widening it.

### What the evidence points at next

`diagnose_decliners.py`, run **before** any of this was fitted so the choice
could not be made after seeing which helped: the galaxies whose centres decline
differ from the rest **in formation time and not in mass** — median `t_half/t`
0.548 against 0.617 (`r = −0.317` with the flag) while halo mass gives
`r = +0.025`. **The centres that fall are the ones that were built early.** A
formation-time-conditioned expansion strength is therefore the one shared
extension with something real to condition on, and its ceiling is that
`r = −0.317`.

---

## Stage 9 (2026-08-26) — `X4`, the formation-conditioned expansion. REJECTED.

The one extension the pre-registered diagnostic pointed at: let the expansion
strength depend on the halo's formation time, `A -> A0 + A_f (f_form - 0.6)`,
because the galaxies whose centres decline differ from the rest in formation
time (`r = -0.317`) and not in halo mass (`r = +0.025`).

**It has the best loss and the best profile-shape error of anything tried —
1.814236 and 13.259% — and it is rejected, for two independent reasons.**

**1. The conditioning slope is not identified.** Ten starts land in three
different places, with three different signs:

| `A_f` | starts | loss |
|---|---|---|
| **−15.0000**, at its lower bound | 0, 4, 5, 6 | 1.8304 |
| **≈ 0** (−0.026 to +0.007) | 1, 2, 3, 7, 9 | 1.8311–1.8333 |
| **+9.73** | 8 — *the winner* | **1.8142** |

A parameter that rails at a bound in four starts, vanishes in five and takes a
large positive value in one is not a measurement. exp54 Stage 3.5 rejected a
richer time law on exactly this ground.

**2. The winning sign is the OPPOSITE of the prediction.** The diagnostic says
the centres that fall belong to haloes that formed EARLY (smaller `f_form`), so
they should expand more, which needs `A_f < 0`. The best-loss solution has
`A_f = +9.73` — later-forming haloes expanding more. The starts that *did*
follow the prediction railed at the bound and scored worse. **The prediction was
made before the fit and the fit did not confirm it.**

**And it fails the gates anyway**: `Rc` = 95 kpc, so it is near-homologous, and
G2 fails at 48.6% beyond 50 kpc. G5(b) is the worst of any model at 0.101.

So the extra parameter buys 0.9% of loss, is undetermined, contradicts its own
motivation and fails the reach gate. Nothing about the formation-time hypothesis
is *disproved* — the diagnostic's `r = -0.317` is real — but this
parameterisation of it does not work.

---

## The laws

`g` multiplies **both** the deposit's half-mass radius and its truncation
radius, so the deposit keeps its shape and its mass exactly.

| law | form | parameters | reach |
|---|---|---|---|
| `X1` | `g = 1 + A (1 − t_j/t_k)` | `A` | **bounded by `1+A`, by construction** |
| `X2` | `g = 1 + A (1 − t_j/t_k)^p` | `A`, `p` | bounded by `1+A` |
| `X0` | `g = (t_k/t_j)^alpha` | `alpha` | **unbounded — the old transport form** |

`X1` is the primary candidate. `(1 − t_j/t_k)` lies in `[0, 1)` for every
deposit at every epoch, so `A` is directly interpretable — *the oldest deposits
end up `(1+A)` times larger* — and directly gateable. `X0` is included as a
declared control so that "bounding the reach costs nothing" is measured rather
than assumed; at the old kernel's fitted `alpha = 0.91` a deposit made at 0.5
Gyr and viewed at 9.4 Gyr is scaled by **×14.4**, which is the failure mode.

**The nesting point is INTERIOR, deliberately.** `A` is bounded `(−0.9, 20)`
and may go negative, which would mean deposits *contract*. exp54 Stage 3.8's
compact weight `w0` went to zero, which was its bound, and a null sitting on a
bound is weaker evidence than a null sitting at an interior point. Here "the
model does not want expansion" would be a measurement.

---

## The figures

- **`figures/exp57_expansion.png`** — the VERDICT, six panels: the expansion
  factor (which is not the reach), the reach gate G2, the mechanism split G1,
  the target gate G5, Stage 1's capability scan, and G1b, the panel with no
  threshold of mine in it.
- **`figures/exp57_mechanism.png`** — the MECHANISM, four panels: where the
  moved mass comes from and goes to shell by shell (the panel that makes the
  whole result visible — both laws drain the same 0–5 kpc shell, and one puts
  it 5–30 kpc away while the other spreads it past 500 kpc); the `Rc` trade
  curve; exp54's deposition schedule against the old transport kernel's; and
  the attribution test.

## Reproducing

```
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
export OMP_NUM_THREADS=1
PYTHONPATH=. uv run python -u experiments/exp57_expansion_term/expand.py
PYTHONPATH=. uv run python -u experiments/exp57_expansion_term/stage0_cost.py
PYTHONPATH=. uv run python -u experiments/exp57_expansion_term/stage5_figures.py
PYTHONPATH=. uv run python -u experiments/exp57_expansion_term/stage9_figures.py
```
