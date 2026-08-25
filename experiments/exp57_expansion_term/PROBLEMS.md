# exp57 — problem tracker

Every problem found during exp57, whether fixed, open, or decided against.
**Nothing is deleted**; items are marked resolved with the date and the answer,
so a problem that comes back is recognisable as a repeat. Newest first within
each status.

Severity: **S1** could change a conclusion · **S2** blocks work or wastes time ·
**S3** cosmetic or bookkeeping.

---

## Open

*(none yet)*

---

## Resolved

### P6 — the gate machinery silently used the wrong basis for a new law
**S1. Found and resolved 2026-08-26, Stage 6.** *Caught by the decomposition's
own self-check, which is the only reason it is not in the results.*

`gates.deposit_terms` duplicated `ExpandingProblem.predict`'s setup so it could
keep the deposits separate instead of summing them. When `X3` was added, the
model gained a second radial form (`cog_core_expanded`) and **the gate copy was
not updated**: it kept calling the homologous `cog_scaled`. So every gate would
have described a **different model** than the one fitted.

`check_terms` caught it immediately — summing the separated deposits missed
`predict` by a **relative 6.3e-1** — and refused to run. That assertion was
written for exactly this and it earned its keep on its first real test.

**Resolution: the duplication is removed, not patched.** `ExpandingProblem`
now exposes `setup(theta)` and `basis(s, k, R)`, and both `predict` and
`deposit_terms` call them. `basis` is the single place a law chooses a radial
form, so a new law cannot be added to the model and forgotten in the
diagnostics.

**The irony is worth recording.** The docstring of the duplicated function
already said "`check_terms` asserts that summing these reproduces `predict`, so
the duplication cannot drift". The duplication drifted the very next time the
model changed. The assertion was the thing that worked; the reasoning that
duplication is safe *because* there is an assertion was wrong.

### P5 — "X3 nests X1" was claimed twice and is false
**S2. Found and resolved 2026-08-26, Stage 6 design.** *Caught by the module's
own self-check, not by inspection.*

`remap_radius`'s docstring asserted that as the core radius goes to infinity,
the core-only expansion `X3` reduces exactly to the homologous `X1`. The
self-check failed: **6.2e-3** in the enclosed fraction at `G = 2.5` with a
truncation 400 times the deposit's half-mass radius.

**Cause.** As `Rc -> infinity` the remap does become `r(R) = R/G` at every
radius — but with the truncation radius held **fixed**, while `X1` carries the
truncation outward with the deposit and renormalises at `G * r_trunc`. The two
differ by the mass lying between `r_trunc / G` and `r_trunc`, and that does not
vanish at large truncation because this deposit family has a genuine
**power-law tail**. A first correction guessed the gap would be negligible at
realistic truncation radii; it is not, and the second self-check said so too.

**Resolution.** The claim is withdrawn rather than the test relaxed. `X1` and
`X3` are two different conventions for what a deposit's mass means — `X1` lets
the boundary travel with the stars, `X3` holds it at the halo's `3 R200c` and
rearranges mass inside it — and **both conserve mass exactly within their own
boundary**. Holding it fixed is the more defensible choice here, since the
truncation is set by the halo and stars rearranging internally are no reason
for the halo's boundary to move. The two are therefore compared by **fitting
both**, which is what the experiment does anyway.

**Lesson**: "this reduces to that in a limit" is a claim, and a cheap numerical
check will falsify it if it is wrong. Two drafts asserted it before the test
settled it.

### P4 — the reach is not bounded by bounding the expansion FACTOR
**S1. Found 2026-08-26, Stage 3. Not a code defect — a design error, and it is
the experiment's main result so far.**

`X1` was designed so that the expansion factor is bounded by `1 + A` for every
deposit at every epoch, on the reasoning that exp52's failure was an unbounded
factor (`g = (t_k/t_j)^alpha` reaching x14.4). Fitted, `A = +1.4657`, so no
deposit ever grows by more than 2.47x — and **G2 fails anyway**, by more than
the model exp52 condemned:

| epoch pair | displaced mass beyond 50 kpc | beyond 148 kpc |
|---|---|---|
| limit | 25% | 10% |
| exp52's failing model | 58.5% | 29.4% |
| `X1`, A = 1.47 | 43.5% to **85.0%** | 22.7% to **52.1%** |
| `X0`, the old unbounded form, alpha = 0.38 | 39.3% to 72.5% | 19.7% to 42.1% |

**Cause, and it is structural.** Homologous scaling multiplies EVERY radius by
the same factor, so a deposit's **power-law tail** moves furthest in absolute
terms: material at 50 kpc scaled by 2.47 lands at 123 kpc regardless of how
modest 2.47 sounds. **Bounding the factor does not bound the reach in kpc.**
Both functional forms fail, so this is a property of homologous expansion, not
of a particular time law.

**Independent confirmation with no threshold of mine in it.** G1b compares the
model against the data with the same operator: at z=0.7 -> 0.4 the null lands
0.364 of its added mass beyond 50 kpc against a measured 0.366 — essentially
perfect — and `X1` overshoots to **0.420**. The gate and the data agree.

**Resolution: `X3`,** a non-homologous remap that bounds the reach in kpc
directly, `r(R) = R [1 - (1 - 1/G)/(1 + (R/Rc)^2)]`. It expands the core by `G`
and leaves everything beyond `Rc` where it was, which is what adiabatic
expansion is understood to do physically and exactly what exp52's diagnosis
asked for. See P5 for how it relates to `X1`.

### P3 — the capability target was stated on the wrong population
**S1. Found and resolved 2026-08-26, Stage 1.** *Would have changed a
conclusion.*

`stage1_contract.py` was written to compare the model's **median over all
galaxies** against exp54 Stage 3.4's **−0.066 dex**. That number is the median
over the **DECLINING SUBSET ONLY**. The two are not the same population and
they are not even the same sign: recomputed on this sample, the measured
`M*(<4.92 kpc)` change from redshift 2 to 0.4 is

| population | median |
|---|---|
| ALL galaxies | **+0.0368 dex** — the centres GROW on the median |
| the declining 41.8% | **−0.0661 dex** |

Left uncorrected, the scan would have chased a −0.066 dex median over the whole
population, which the data does not do, and would have selected an expansion
strength roughly twice too large.

**Resolution.** `decline_stats` now returns a **dict**, not a tuple, so a caller
cannot silently compare a median-over-all against a target defined on a subset;
all four population statistics are reported side by side with the population
each describes named in the output; and the constant is renamed
`TRUTH_DECLINER_DEX`.

**This is the fourth instance in this project of a quantity being compared
against a bound defined on something else** (`doc/lessons.md` records three:
Stage 3.7's span, exp52's aperture-vs-annulus, exp53's binning variable). The
standing rule — *a bound must match what it reports, on the same objects and
the same quantity* — now has a fourth data point, and the structural guard used
here (return a named record, never a positional tuple) is the cheapest one
found so far.

**What it turned into, which is the real Stage 1 result.** Stated correctly,
the target is not a shift but a SHAPE: a population whose centres grow by
+0.037 dex on the median while 42% of them fall. A single global expansion
strength moves a distribution without widening it, and the three targets need
A = 0.76, 0.98 and 1.65 respectively — a factor of 2.2 apart.

### P1 — the nesting gate was specified as "bit-for-bit" and that was unachievable
**S2. Found and resolved 2026-08-26, Stage 0.**

`stage0_cost.py` was written to demand that `ExpandingProblem` with the law
switched off reproduce exp54's `StackedProblem` **bit-for-bit**. It refused to
run: the profiles disagreed by **1.7e-15 relative, about 8 units in the last
place**.

**Cause, and it is not a bug.** exp54 forms one matrix product over all five
epochs at once. Expansion makes the deposit basis depend on the viewing epoch,
which forces one product per epoch, so the same sum is accumulated in a
different order. Floating-point addition is not associative.

**Resolution.** The *specification* was wrong, not the code, and it was
corrected in the open rather than the tolerance quietly widened — which is the
failure mode this project has recorded before. The gate now requires:

| | requirement | measured |
|---|---|---|
| NaN pattern | **exact** (a different set of failing galaxies is a real difference) | exact |
| profiles | < 1e-12 relative | 1.74e-15 (7.8 ulp) |
| loss | < 1e-9 absolute | 2.2e-16 |

The tolerances sit three and six orders of magnitude above what is measured, so
a real refactor error moves them by far more. This is the same standard exp54's
own `check_stacking` adopted when it stacked the per-galaxy loop, and for the
same reason.

**Lesson**: when a gate fails, decide whether the code or the specification is
wrong *before* touching either. Writing down "bit-for-bit" felt like rigour and
was simply not a property the refactor could have.

### P2 — the anchor-epoch cosmic times could not be read the way the first draft read them
**S3. Found and resolved 2026-08-26, Stage 0.**

`expand.ExpandingProblem` needs the cosmic time of each of the five anchor
epochs, because the expansion factor depends on how long ago a deposit was laid
down. The first draft looked for a `snap` field on the `Halo` record; there is
none (`halo.Halo` carries `t`, `z`, `epoch_mask`, and no snapshot index).

**Resolution.** Read it from `epoch_mask` instead — `epoch_mask[k]` is
`snap <= ANCHOR_SNAP[k]`, so its last True entry sits exactly at that anchor —
and then **check two things rather than assume them**: that the redshift there
is the anchor's redshift, and that every galaxy in the sample gives the same
cosmic time, since the whole tabulation rests on the snapshot grid being shared.

**A real finding fell out of the check.** `halo.ANCHOR_Z = (0.4, 0.7, 1.0, 1.5,
2.0)` are **rounded labels**; the snapshots actually sit at **0.39993, 0.70011,
0.99730, 1.49551, 2.00203**. The first tolerance (2e-3) was tighter than the
rounding and the check fired. The expansion factor uses the snapshot's own
cosmic times, never the labels, and the tolerance is now the label's rounding
with that stated in a comment.
