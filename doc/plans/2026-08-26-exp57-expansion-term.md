# exp57 — an empirical expansion term, designed so it cannot repeat exp30's failure

*Written 2026-08-26, before any code, at the user's instruction: "deposition +
transportation is the first model we tried… if we were to try 'expansion'
again, let's have a plan to avoid repeating the same issue."*

---

## 1. What went wrong the first time, precisely

The program's first forward model was **deposition + transport** — tech note 2,
the `1ch-mof` kernel (`exp30_transport_kernel`, `exp38_deposit_rethink`).
Deposits landed compact and a fraction migrated outward on a self-similar
clock:

```
f_c,i = exp(-(t_k - t_i)/(alpha t_i))          alpha = 1  (frozen)
r_w,i = r_c,i (t_k/t_i)^q                      q = 0.91
```

**`exp52_growth_mechanism` diagnosed it.** Splitting each epoch pair's growth
into *transport* (material already present, moved), *renormalisation* (the
`M(<500 kpc)` pinning) and *new deposition*:

| epoch pair | transport share of `M*[50,100]` growth |
|---|---|
| z=2.0 → 1.5 | 18.0% |
| z=1.5 → 1.0 | 38.4% |
| z=1.0 → 0.7 | 76.9% |
| **z=0.7 → 0.4** | **96.5%** |

The crossover sat at **z ≈ 1.2 — exactly where the observational picture says
minor-merger accretion should be taking over.** Over z=2 → 0.4 the galaxies
grew **×2.645** while the kernel's own deposition supplied **×1.115**, i.e.
**11% of the log mass growth; the other 89% was manufactured by the `M(<500)`
pinning.**

**The root cause was the deposition schedule, not the transport term.** The
fitted efficiency window was a lognormal peaking at z=3.58 with a 1σ span of
z=2.6–4.8. It deposited **1.5% of the stellar mass at z<1 while the halo
assembled 47% of its mass there**. With the deposition machine effectively off
after z≈1.5, *the only way the kernel could grow an outskirt was to drain the
centre* — which is also why it under-filled compact galaxies' centres.

**`exp52` Result 4 is the sharpest statement, and it is the design brief for
this experiment**: the transport had **the right source and roughly ten times
too much reach**. 93% of the drained mass came from inside 5 kpc — precisely
where AGN-driven adiabatic expansion should operate — but **58.5% of it was
delivered beyond 50 kpc, 29.4% beyond 148 kpc (outside the measured profile
entirely), and 12.0% beyond 500 kpc, where the normalisation then discarded
it.** Adiabatic expansion flattens the *inner* stellar profile; it does not
build the stellar halo at 50–150 kpc.

**`exp53_deposition_only` then ran the boundary condition** (`q = 0`) and found
two things this plan must respect:

1. **Removing transport did NOT fix the central deficit.** Refitted,
   `moffat-q0` gave −39.4% against the incumbent's −43.4%. A mechanism that is
   real in a model's internal bookkeeping is **not thereby the cause** of that
   model's error — the frozen slice and the honest refit disagreed by 57
   percentage points on that number.
2. **Transport SUBSTITUTES for a heavy tail.** Removing it cost the two
   heavy-tailed families 3–4% of the loss and the two light-tailed ones ~11%.
   *An expansion term can be bought by the fit as a replacement for deposit
   profile shape, and it will be if the shape is not held still.*

---

## 2. What is different now — measured, not assumed

Before deciding whether the pathology can recur, I measured exp54's actual
deposition schedule at the incumbent (`gompertz_log-E2-S2`, 2397 galaxies,
clean sample). **exp54 is structurally in a different place.**

| redshift | share of STELLAR mass deposited | share of HALO mass | median deposit `R50` |
|---|---|---|---|
| 0 – 0.5 | 4.6% | 10.9% | 49.7 kpc |
| 0.5 – 1 | 24.7% | 40.2% | 35.6 kpc |
| 1 – 1.5 | 20.4% | 20.6% | 19.8 kpc |
| 1.5 – 2 | 15.1% | 11.1% | 12.1 kpc |
| 2 – 3 | 19.4% | 10.7% | 6.5 kpc |
| 3 – 5 | 12.8% | 5.4% | 2.0 kpc |
| 5 – 20 | 2.9% | 1.2% | 0.25 kpc |

**exp54 deposits 29.3% of its stellar mass at z<1** (against the halo's 51.1%),
where the transport kernel deposited **1.5%**. And it puts that late mass
*where ex-situ accretion should put it*: median deposit half-mass radius
**35–50 kpc at z<1** against 0.25–2 kpc at z>3.

Growth supplied by the model's **own** deposition, with no pinning term
anywhere in exp54:

| epoch pair | model's own deposition | measured `M*(<100 kpc)` |
|---|---|---|
| z=2.0 → 1.5 | ×1.466 | ×1.294 |
| z=1.5 → 1.0 | ×1.404 | ×1.320 |
| z=1.0 → 0.7 | ×1.191 | ×1.170 |
| z=0.7 → 0.4 | ×1.160 | ×1.162 |
| **z=2.0 → 0.4** | **×2.842** | **×2.323** |

Against the transport kernel's ×1.115 own against ×2.645 measured.

**Three structural differences follow, and together they are why this is not a
straight repeat:**

1. **The deposition machine never switches off.** The condition that forced
   transport to become the only outskirt-growth channel does not hold here.
2. **The `M(<500 kpc)` pinning is gone.** exp54 predicts `log10 M*(<100 kpc)`
   and is scored on it (`score_A`). The renormalisation term — 89% of the old
   model's growth — does not exist, so exp52's three-term decomposition becomes
   a clean **two-term** one: expansion versus new deposition.
3. **Late deposits are already extended.** New mass arrives at 35–50 kpc at
   z<1, which is the ex-situ picture the old model could only imitate by
   draining the centre.

---

## 3. The NEW trap, which is the mirror image of the old one

**exp54 over-supplies growth: ×2.842 against a measured ×2.323, a 22% excess
over the z=2 → 0.4 baseline.**

An expansion term moves mass outward, hence *out of the 100 kpc aperture*, and
therefore **reduces `M*(<100 kpc)`**. The fit has an excess to dispose of, and
expansion is a way to dispose of it.

So the danger here is not "expansion becomes the only way to grow the
outskirts". It is the reverse: **expansion gets fitted as an amplitude
regulator — a way to dump a 22% over-supply past 100 kpc — rather than as a
mechanism that lowers central density.** Note that `score_F` is normalised at
100 kpc and is therefore *blind* to everything beyond it, exactly as the old
`M(<500)` normalisation was blind beyond 500 kpc. The deletion channel exp52
closed by moving the pin outward has partially reopened at a smaller radius.

Two further inherited traps, both cheap to guard:

- **Expansion substituting for tail weight** (exp53). Stage 3.9 just measured
  that `c`, the deposit shape, is the *only* one of the seven parameters the
  others cannot absorb (over-stated 2.3× against 7–18× for the rest). Adding an
  expansion exponent is the most likely thing to destroy that, because both
  control how far a deposit's mass reaches.
- **Reading a slice as a cause** (exp53 Result 1 vs Result 3, 57 points apart).

---

## 4. The form to fit, and why not the one first proposed

The deferred proposal (tech note §10.2 item 3b, `doc/todo.md`) is **homologous**
expansion: both the deposit's half-mass radius and its truncation radius scaled
by `g = (t_k/t_j)^alpha`.

**Homologous scaling is precisely the shape that produced the too-long reach**,
and the arithmetic is unforgiving. A deposit made at `t_j = 0.5 Gyr` viewed at
`t_k = 9.4 Gyr` is scaled by `18.8^alpha`. At the old kernel's `q = 0.91` that
is **×14.5**. And because `gompertz_log`'s curve of growth is a Fréchet with a
genuine power-law tail, homologous scaling moves *every* mass fraction's radius
by the same factor — so the tail moves **furthest in absolute terms**. A
deposit whose 5% tail sits at 50 kpc puts it at 725 kpc. That is exp52's
failure mode reproduced from first principles.

**Adopt instead a saturating homologous expansion**, which keeps every virtue of
the original proposal and bounds the reach:

```
g(dt) = 1 + (G_max - 1) * (1 - exp(-dt / tau))     dt = t_k - t_j
```

- **mass is exactly conserved**, nothing is removed, nothing goes negative;
- **no stellar information enters** — Contract B is intact;
- **`G_max = 1` nests the current model bit-for-bit**, so the fitted `G_max` is
  the measurement;
- **`M*(<R)` at fixed small `R` can now decrease**, which is the whole point:
  42% of these galaxies lose a median 14% of their central mass and a
  deposition-only model cannot follow that at any parameter value;
- and **the reach is bounded by construction** — `G_max` is a directly
  interpretable, directly gateable number rather than an exponent whose
  consequence depends on a ratio of cosmic times.

Two parameters. If `tau` proves undetermined, freeze it and fit `G_max` alone;
the one-parameter version is the primary candidate and the two-parameter one is
its enrichment.

**The power-law form stays on the table as a declared control**, fitted with a
hard cap on `g`, so that "saturating beats power-law" is measured rather than
assumed.

---

## 5. Preregistered gates — all measured BEFORE the loss is consulted

Every one of these is a port of an existing exp52/exp53 artifact, so the
comparison numbers already exist and the thresholds are set now, in advance.

**G1 — the mechanism split.** Port `exp52/decompose.py`, now two terms (no
pinning). Report the **expansion share** of `M*[50,100]` and `M*[100,148]`
growth at every epoch pair.
*Fail if the expansion share exceeds the new-deposition share at any epoch pair
with z<1.2.* (Reference failure: 76.9% and 96.5%.)

**G2 — the reach.** Port `exp52/reach.py`. Report where expansion takes mass
from and delivers it to, per shell.
*Fail if more than 25% of the displaced mass is delivered beyond 50 kpc, or
more than 10% beyond 148 kpc.* (Reference failure: 58.5% and 29.4%.) The
**source** is expected to be inside 10 kpc and should be reported, not gated —
exp52 found the source was already right. **Measured companion (see §8)**: the
model's added-light kernel half-mass radius per epoch pair, against the
measured **15-47 kpc**. Too much reach inflates it, visibly, with no threshold
of mine involved.

**G3 — no invisible deletion.** Report the fraction of deposited stellar mass
lying beyond 100 kpc (where `score_F` is blind) and beyond the truncation
radius, per epoch, for `G_max = 1` and for the fit.
*Fail if the fit increases the beyond-100 kpc fraction by more than it improves
`score_A`* — i.e. if expansion is buying amplitude by deletion. The 22%
over-supply measured in §2 is the specific thing this watches.

**G4 — expansion must not be a tail substitute.** Refit twice: `G_max` free
with `c` free, and `G_max` free with `c` **frozen** at the incumbent's +0.7965.
*Report the shift in `c` and the loss difference.* If freeing `c` alongside
`G_max` moves `c` by more than its Stage 3.9 one-percentage-point width
(−0.25/+0.51), the two are trading and the result is a reparameterisation, not
a mechanism. Then re-profile `c` and `G_max` jointly (Stage 3.9 machinery,
`stage39_profile.py`, ~30 min per parameter) and check whether `c`'s 2.3×
absorption factor survives.

**G5 — the target, split on the preregistered flag, TWO-SIDED.** The mechanism
exists to follow a *declining* central mass. Split on Stage 3.4's flag (did the
measured `M*(<4.92 kpc)` fall between z=2 and z=0.4). Current central-error
span: 0.270 dex on the declining 42%, 0.045 dex on the non-declining 58%.
*Success = the declining group's span closes materially, AND (a) the
non-declining 58% do not open beyond 0.06 dex, AND (b) the model's median core
`dlogSigma` over the long baseline stays within ±0.03 dex of the **measured
+0.09** (§8) — because on the median these cores GROW.* A term that lowers
central density everywhere fails (b), and (b) is measured, not asserted. A term
that improves the aggregate by hurting the galaxies that were already right
fails (a).

**G6 — refit, never slice.** Every number above is quoted from a **full refit**
with `G_max` free. exp53's slice-versus-refit disagreement was 57 points; a
frozen-slice number may be shown for context but may not be the verdict.

---

## 6. Staged plan

**Stage 0 — cost, measured not estimated.** Expansion makes each deposit's
profile depend on the *viewing* epoch as well as the deposition epoch, so the
basis becomes `(n_galaxies, n_deposits, n_epochs, n_radii)` instead of
`(n_galaxies, n_deposits, n_radii)`. The tech note says this "costs several
times more per evaluation" — **measure it.** Current: 0.111 s per evaluation.
If it exceeds ~0.5 s, a fit is 5× more expensive and the multi-start budget has
to be planned around that before anything else is written.

**Stage 1 — the contract.** Stage 0's assertion that `M*(<R,t)` is
non-decreasing in `t` becomes **conditional on `G_max = 1`**. Rewrite it as a
two-branch assertion and verify both branches, so the guarantee is still tested
rather than deleted. Verify `G_max = 1` reproduces the incumbent's loss of
**1.874253** bit-for-bit before fitting anything.

**Stage 2 — the gates on the incumbent.** Run G1, G2, G3 at `G_max = 1`. This
is the null, it costs nothing, and it establishes what the gates read for a
model with no expansion at all. Without it the fitted values have no reference.

**Stage 3 — fit.** `G_max` alone, then `(G_max, tau)`, then the power-law
control with a cap. Multi-start mandatory; lift the nested start by name with
an assertion that it reproduces the parent's loss (standing rule).

**Stage 4 — the gates on the fit**, then G4 and G5. Rank by the gates, then by
the seven-criterion judge, **never by the loss**.

---

## 7. What would make this stop

Stated now so it is not renegotiated later:

- **G1 or G2 fails** → the reach is wrong again; report and stop. Do not widen
  the threshold.
- **G4 shows `c` and `G_max` trading** → it is a reparameterisation of deposit
  shape, not a mechanism. Report as such.
- **G5 shows the non-declining 58% getting worse** → the term is buying an
  aggregate at the expense of the galaxies the model already fits.
- **`G_max` fits to 1** → the model does not want expansion; that is a clean
  negative and matches Stage 3.8's compact channel going to zero weight.

## 8. The empirical anchor is our OWN measured added light — NOT the TNG Stellar Assembly catalog

An earlier draft of this plan proposed verifying TNG's in-situ / ex-situ
Stellar Assembly catalog and using it to turn G1's threshold into a
measurement. **That is withdrawn** (user, 2026-08-26), for two reasons, the
second of which is fatal:

1. **It carries no radial information.** The catalog is per-subhalo scalars —
   in-situ and ex-situ stellar mass, merger-delivered mass, mass ratios. G1 and
   G2 are questions about *where* mass sits and *where it goes*. A scalar
   budget cannot answer either.
2. **TNG's "ex-situ" is not the physical quantity the gates need.** In-situ is
   defined by tracing the **main branch only**, so stars formed in any
   progenitor that later merged are labelled ex-situ — **including major
   mergers at z>3, which contribute heavily to ex-situ and land in the
   CENTRE.** So "ex-situ" is emphatically not "delivered to the outskirts by
   minor mergers", and reading it that way would import exactly the confusion
   between *arrived by merging* and *sitting at large radius* that this whole
   plan exists to avoid.

The most one could defensibly extract is **Δ(ex-situ mass) between our five
epochs** as a population budget for merger-delivered stellar mass per interval —
a constraint on the deposition *rate*, not on radial placement, and still
carrying the classification bias above. **Not worth building the join for, and
not a gate.**

### What to use instead, which is better and already exists

**The measured added-light kernel** (`exp38` stage 0.2, applied by
`exp53/score.py`). Between adjacent epochs, stack the median added surface
density per stellar-mass tercile and fit it. This is **measured from our own
profiles, model-independent by construction, and radial** — it is the direct
empirical answer to "where does new stellar mass actually land?":

- centrally peaked at **3–5 kpc**, Sérsic **n = 2.1–3.1**, kernel half-mass
  radius **15–47 kpc**;
- exp53 already applies the *identical operator* to model curves of growth —
  same terciles, same `Sigma = dM/dA`, same grid fit over the same 4–120 kpc
  band — so a model cell and a data cell are directly comparable.

**And exp52's differential scorecard supplies the single most important
constraint on an expansion term**, measured on the data with no model involved:

> median `dlogSigma ≈ 0.00 ± 0.01` at 2–3 kpc for both z<1 pairs, and
> **+0.09 dex at 2.35 kpc over the long baseline — the core GROWS.**

That is not in tension with the 42% of galaxies whose central mass falls: the
median grows while a large minority declines. But it is a hard two-sided
constraint that the first model never had to face, and it replaces the catalog
idea outright.

**G5 is therefore restated, two-sided:**

*The declining 42% must have their central-error span close materially (from
0.270 dex), while (a) the non-declining 58% do not open beyond 0.06 dex (from
0.045), and (b) the model's own median core `dlogSigma` over the long baseline
stays within ±0.03 dex of the measured **+0.09**.*

An expansion term that lowers central density everywhere will fail (b), and (b)
is measured rather than asserted. That is the guard the first model lacked.

**G2 gains a measured companion**: report the model's added-light kernel
half-mass radius per epoch pair against the measured 15–47 kpc. Expansion with
too much reach inflates it, and the inflation is visible without any threshold
of mine.
