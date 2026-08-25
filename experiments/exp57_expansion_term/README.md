# exp57 — an empirical expansion term: deposits grow after they are laid down

Branch `exp57-expansion-term`. Plan:
`doc/plans/2026-08-26-exp57-expansion-term.md`. Problems:
[`PROBLEMS.md`](PROBLEMS.md).

**Status: IN PROGRESS.** Stages 0 and 1 complete; Stage 3 fits running.

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

## Reproducing

```
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
export OMP_NUM_THREADS=1
PYTHONPATH=. uv run python -u experiments/exp57_expansion_term/expand.py
PYTHONPATH=. uv run python -u experiments/exp57_expansion_term/stage0_cost.py
```
