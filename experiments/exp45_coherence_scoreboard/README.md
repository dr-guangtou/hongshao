# exp45 — the cross-epoch coherence scoreboard (retro-scoring the model zoo)

exp44 graduated `hongshao.qa` tier 2c (cross-epoch growth planes) and its
first run exposed a large, previously unmeasurable defect: the
DETERMINISTIC 1ch-mof kernel evolves galaxy sizes at rank coherence
+1.00 / +1.00 / +1.00 / +0.96 (z=0.4 against z=0.7 / 1.0 / 1.5 / 2.0)
where the TRUTH sits at +0.82 / +0.68 / +0.50 / +0.33. Real galaxies
substantially forget their relative size across cosmic time; the model
barely does.

No model in the program was ever selected against this, because the
metric did not exist — every previous tier is a single-epoch statistic
and is structurally blind to it. So before building anything, measure:

**Is the defect UNIVERSAL to the model family, or does some structural
choice already do better?**

Pure evaluation — **no new fitting**. Every parameter vector below is
already on disk; the job is to build its model CoGs and run
`qa.growth_planes`. (Precedent for the shape: exp43, which surveyed
recorded thetas in its own directory.)

## The zoo

Structural axes spanned: channel count (1 vs 2), fit scope (z04 / z07 /
z10 / z15 / z20), objective (plain vs inner-aware), MAH input (DiffMAH
vs real), and — as a cross-FAMILY reference — the statistical emulator,
which carries an AR(1)-in-epoch latent BY DESIGN (rho = 0.62) and so is
the sharpest available test of whether that construction buys realistic
size coherence.

| model | source | axis probed |
|---|---|---|
| 1ch-mof z20 / 2ch-exp z20 | exp38 `stage2_multiepoch.npz` | channel count |
| 1ch-mof z04 / z2.0-only, 2ch-exp same | exp38 `stage3_single.npz` | single-epoch fits |
| 1ch-mof / 2ch-exp, inner-aware | exp38 `stage3_capacity.npz` | objective |
| 1ch-mof z15 (ADOPTED) / z10 | exp40 `latestart.npz` | fit scope |
| 1ch-mof z07, 2ch-exp z07/z10/z15 | exp43 `ladder.npz` | fit scope x channels |
| 1ch-mof z15 on the REAL MAH | exp42 `realfit.npz` | MAH input (bursty) |
| + the exp41 stochastic layer (persistent / ar1) | exp44 | scatter layer |
| **statistical emulator (exp37 block draws)** | exp37 `multi_epoch_block.npz` | CROSS-FAMILY |

The exp37 row uses n=2395 (its two masked broken-progenitor galaxies);
the mask is reconstructed from the documented criterion (progenitor
total more than 2 dex below the z=0.4 descendant) and VERIFIED to
return exactly indices 181 and 2372, so the truth alignment is exact.
The real-MAH row re-initializes the loader with `config="real"`.

## Judged readout

Per model: size rank coherence at each separation against the truth's
+0.82 / +0.68 / +0.50 / +0.33, plus the tier-2c energy ratio. The
headline number is the **coherence excess** at the widest baseline
(model minus truth at z=0.4 vs z=2.0) — positive means too loyal.

`Mtot` is reported but is degenerate for every kernel row (each epoch is
pinned to the measured M(<500 kpc), so those totals ARE the data); it is
informative only for the exp37 row, which predicts totals from features.

## Run

```bash
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
PYTHONPATH=. uv run python experiments/exp45_coherence_scoreboard/\
scoreboard.py {demo|run|figures} [--dev]
```

## Results (2026-08-12, full n=2397; `outputs/scoreboard.npz`,
## figure `figures/exp45_scoreboard`)

**Answer: the defect is UNIVERSAL to the transport-kernel family and
immune to every structural choice within it. No channel count and no fit
scope moves it. Only an explicit EPOCH-COUPLED STOCHASTIC term does —
and the statistical emulator, which has one by design, does not have the
defect at all.**

Size rank coherence, z=0.4 against z=0.7 / 1.0 / 1.5 / 2.0
(TRUTH: **+0.82 / +0.68 / +0.50 / +0.33**), with the widest-baseline
excess over truth and the mean tier-2c energy ratio:

| model | coherence curve | excess | E/floor |
|---|---|---|---|
| 11 deterministic kernels (1ch & 2ch, z04 -> z20) | +1.00 / +1.00 / +0.99-1.00 / +0.82-0.98 | **+0.49 to +0.66** | 8.7-12.3 |
| 1ch inner-aware objective | +0.99 / +0.95 / +0.86 / +0.77 | +0.45 | 5.0 |
| 1ch z15 on the REAL MAH | +0.99 / +0.95 / +0.87 / +0.78 | +0.45 | 7.9 |
| + exp41 layer, `persistent` (adopted) | +1.00 / +1.00 / +1.00 / +0.97 | +0.64 | 6.5 |
| + layer, `ar1` (exp44 option) | +0.74 / +0.62 / +0.54 / +0.48 | **+0.15** | 4.7 |
| **exp37 statistical emulator (CROSS-FAMILY)** | **+0.77 / +0.58 / +0.33 / +0.17** | **-0.16** | **2.1** |

1. **Structure does not help.** Eleven deterministic kernel variants
   spanning both channel counts and all five fit scopes are
   indistinguishable: coherence pinned at +1.00 out to two epochs of
   separation, excess +0.49 to +0.66, energy 8.7-12.3x the floor. A
   galaxy's relative size at z=0.4 essentially determines its relative
   size at z=2 in every one of them. Whatever causes this is common to
   the family, not a consequence of the split, the tail, or the scope.
2. **Two partial exceptions, and they are the same size.** The
   inner-aware objective (1ch only) and the real-MAH input both land at
   excess +0.45 with near-identical curves, despite being completely
   different interventions — one changes what the fit rewards, the other
   changes the input's burstiness. Both were REJECTED for other reasons
   (the inner-aware objective breaks the differential-deposition test,
   exp39/exp40; the real MAH strains the low-mass outskirts, exp42), so
   neither is available as-is. Note 2ch inner-aware shows NO improvement
   (+0.66) — the effect is specific to the single-channel form.
3. **An explicit epoch-coupled stochastic term beats every structural
   choice.** exp44's `ar1` mode reaches excess +0.15 and energy 4.7 —
   better than anything the model form achieves — while the adopted
   `persistent` layer (rho = 1) does nothing at all (+0.64), as its
   construction requires.
4. **The statistical emulator simply does not have the defect.**
   +0.77 / +0.58 / +0.33 / +0.17 against the truth's
   +0.82 / +0.68 / +0.50 / +0.33 — it slightly UNDER-persists, and sits
   at **2.1x the floor where every kernel is at 8.8-12.3x**. Neither
   family was ever tuned on this metric, so it is a clean out-of-model
   comparison, and the mechanism is identifiable: exp37 carries an
   AR(1)-in-epoch latent (rho = 0.62) by design, while a kernel galaxy
   is ONE theta plus ONE accretion history, which forces its size track
   to be near-deterministic.

## Reading

The kernel's cross-epoch over-coherence is **structural to the family
and not reachable by re-parameterizing it** — which is exactly the
"stop building mechanisms, the tension is between the family and the
data" pattern the exp39 lesson describes. The fix has to be an explicit
epoch-coupled stochastic term, and exp44 already built one.

That materially changes the standing of exp44's verdict. `ar1` was kept
as a non-default option because it costs the flagship
differential-deposition pair (0.41 -> 0.29 against data 0.37) and
because single-epoch applications are provably rho-blind. Both remain
true — but it is now the ONLY lever in the entire zoo that addresses a
real, measured, otherwise-unfixable defect, at 4.7x floor against the
adopted layer's 6.5x. **The trade is a genuine decision for the user,
not a settled question:** differential-deposition fidelity versus
cross-epoch realism, with no configuration achieving both.

## Follow-up 1 — the lever is the EFFICIENCY WINDOW WIDTH (`followups.py sweep`)

The two low-coherence fits share three features against the pack (broad
`sig`, small `log_rc`, shallow `g`), which covary in a fit. Sweeping ONE
coordinate at a time off the adopted theta separates them — coherence at
the widest baseline (adopted +0.96, truth +0.33):

| coordinate | swept range | widest-baseline coherence |
|---|---|---|
| **`sig`** (window width) | 0.24 -> 1.20 | **+0.96 -> +0.79 -> +0.74 -> +0.69 -> +0.67** |
| `g` (width-law exponent) | 4.00 -> 2.00 | +0.96 -> +0.96 -> +0.96 -> +0.96 -> +0.87 |
| `q` (migration exponent) | 0.20 -> 1.20 | +0.96 at EVERY value — inert |
| `log_rc` (birth radius) | 2.00 -> 3.00 | +0.96-0.97 — inert |

**Only the efficiency-window width moves it.** The migration exponent —
the model's explicit "how fast does mass move outward" knob — is
completely inert, which is counter-intuitive and worth stating: cross-
epoch size coherence is set by WHEN the stars arrive, not by how they
are transported afterwards. The mechanism is physical: a narrow window
delivers every galaxy's stars in the same brief cosmic era, so all
galaxies' sizes are driven by the same shared clock and their tracks
stay locked together; a broad window samples more of each galaxy's own
accretion history, decorrelating them.

**But the lever saturates well short of the target.** Even at
`sig` = 1.20 — five times the adopted value, and beyond any fitted value
in the program — coherence only reaches +0.67 against the truth's +0.33.
So menu option (C), the flexible efficiency window, is now targeted
rather than speculative, and would genuinely help — but it cannot close
this gap alone. That is consistent with the main result: the defect is
structural to a family in which one galaxy is one theta plus one history.

## Follow-up 2 — the emulator does NOT take over the physics
## (`followups.py emulator_physics`)

Differential deposition on exp37's three saved draws (massive tercile,
fraction of inter-epoch growth landing beyond 50 / 100 kpc):

| epoch pair | data | adopted kernel | exp37 emulator (3 draws) |
|---|---|---|---|
| z0.7 -> z0.4 | 0.37/0.11 | **0.40/0.13** | 0.32/0.09, 0.30/0.08, 0.31/0.09 |
| z1.0 -> z0.7 | 0.36/0.10 | 0.31/0.09 | 0.25/0.07, 0.27/0.08, 0.26/0.08 |
| z1.5 -> z1.0 | 0.27/0.07 | 0.27/0.08 | 0.23/0.06, 0.24/0.06, 0.24/0.06 |
| z2.0 -> z1.5 | 0.23/0.06 | 0.21/0.06 | 0.15/0.04, 0.16/0.03, 0.15/0.03 |

The emulator UNDERSHOOTS at every pair and in every draw, and badly at
the highest (0.15 vs 0.23). The kernel is closer to the data at three of
the four pairs and much closer at the extreme one.

**So the two-model division of labour SURVIVES intact, now measured on
both axes**: the emulator owns accuracy and cross-epoch coherence, the
kernel owns the deposition physics, and neither dominates. That was the
program's founding framing, asserted from accuracy alone; it now has a
second, independent leg.

## Follow-up 3 — the MASS-SIZE plane, now scored (graduated as qa tier 2d)

The user's stated success criterion is "the distributions on the
inner-vs-outer stellar mass **or the mass-size** plane". The first was
scored; the second was not in `qa.PLANES` at all, because that set pairs
measured MASSES with each other. `qa.size_planes()` now ships as **tier
2d** (log M*(<R_max) vs log R_half, per epoch, reported by `evaluate`,
figure `qa_size_<name>`), with the same self-consistency exception as
2c — the model's R_half comes from the MODEL CoG, since using the
truth's would make half the plane correct by construction. Its
self-check pins that: a model with the RIGHT masses and a 1.6x size
offset must be flagged, where the truth-shared convention would score it
perfectly.

**Result: the adopted product does NOT pass this plane** (n=2397; the
deterministic kernel and then the adopted kernel + stochastic layer):

| z | truth slope / scatter | + layer, model | median log R_half, truth -> model | E/floor (centered) |
|---|---|---|---|---|
| 0.4 | +0.50 / 0.173 | +0.34 / 0.148 | +1.06 -> +1.07 | **3.1** (2.9) |
| 0.7 | +0.50 / 0.182 | +0.30 / 0.160 | +0.96 -> +0.97 | 4.0 (3.7) |
| 1.0 | +0.46 / 0.192 | +0.28 / 0.166 | +0.85 -> +0.87 | 3.5 (2.7) |
| 1.5 | +0.30 / 0.200 | +0.19 / 0.154 | +0.64 -> **+0.74** | 5.8 (2.2) |
| 2.0 | +0.08 / 0.195 | +0.06 / 0.115 | +0.48 -> **+0.59** | **7.8** (3.6) |

(Without the layer the deterministic kernel's scatter is roughly HALF
the truth's — 0.072 vs 0.173 at z=0.4 — and E/floor 5.1-10.2. The layer
recovers most of the low-z scatter, as exp41's "the layer's diversity IS
size diversity" finding predicts, but little at z=2.)

Four defects, three of them new:

1. **The relation is too FLAT at every epoch** (+0.34 vs +0.50 at
   z=0.4). The residual panel shows why it is a genuine tilt, not an
   offset: sizes are ~+0.07 dex too LARGE at the low-mass end and
   ~-0.04 dex too SMALL at the high-mass end, crossing zero near
   log M* ~ 11.25.
2. **Sizes are systematically ~25-30% too large at z >= 1.5**
   (+0.10 dex median, at ALL masses) — the model does not shrink its
   galaxies fast enough toward high redshift.
3. **The compact tail is missing entirely.** The truth cloud extends to
   log R_half ~ 0.2 at z=0.4 and below 0 at z=2; the model's has a hard
   lower edge well above it. (Part of the truth's low-mass tail is the
   known near-empty-progenitor data population, so the honest statement
   is the mass-binned median in the residual panel, which is robust.)
4. Even at its best epoch the plane sits at **3.1x the split-half
   floor**, against the 1.0-1.1x the inner-vs-outer mass planes reach at
   z=0.4 with the same layer.

**Reading:** by the user's own criterion — get the observational planes
right — the mass-size plane is now the WEAKEST scored deliverable, and
its high-z behavior (defect 2) is the same phenomenon as the "high-z
ridge error" that exp41 identified as the limit on the mass planes. The
model's high-z galaxies are too big and too uniform; that single
statement covers the z>=1.5 plane residual, the size offset, and the
missing compact tail.

## Follow-up 4 — R80/R90 added, and the size offset IS the inner deficit
## (user hypothesis, 2026-08-12: CONFIRMED)

`qa.enclosed_radius(cog, R, frac)` generalizes `half_mass_radius`, and
tier 2d now scores **R50 / R80 / R90** (`SIZE_FRACTIONS`). The point is
diagnostic: R50 is set by where the INNER mass sits, R80/R90 by the outer
profile, so their relative offsets separate an inner-mass defect from a
uniform size error.

**Median dlog R (model - truth), adopted product:**

| z | R50 | R80 | R90 | R50-R90 gap |
|---|---|---|---|---|
| 0.4 | **+0.024** | -0.009 | -0.001 | +0.025 |
| 0.7 | **+0.038** | +0.002 | +0.007 | +0.031 |
| 1.0 | **+0.053** | +0.017 | +0.017 | +0.036 |
| 1.5 | **+0.084** | +0.054 | +0.046 | +0.038 |
| 2.0 | +0.087 | **+0.116** | **+0.112** | **-0.025** |

**Per-galaxy, the link is essentially deterministic:** Spearman
correlation between a galaxy's dlog R50 and its M*(<5 kpc) bias is
**-0.96 / -0.96 / -0.97 / -0.95 / -0.89** across the five epochs. The
galaxies whose centres the model under-fills are exactly the galaxies
whose R50 it over-predicts. (R90 correlates too, at -0.74 to -0.81 — it
inherits part of the inner error through the total normalization.)

Two conclusions:

1. **At z <= 1.5 the mass-size offset is NOT an independent defect — it
   is the known inner-mass deficit re-expressed in size units.** R50 is
   biased while R80/R90 sit near zero, and that reading is CONSERVATIVE:
   the CoG is nearly flat near R90, so R90 amplifies any real outer mass
   error (measured on a synthetic: an inner-only deficit biasing R50 by
   +0.048 dex still drags R90 by +0.042). R90 nonetheless coming out
   unbiased is therefore strong evidence the cause is inner-localized.
   *This corrects the earlier follow-up-3 reading, which listed "sizes
   ~25-30% too large at z >= 1.5" as a new defect; at z=1.5 it is mostly
   the old one wearing different units.*
2. **At z = 2.0 the ordering FLIPS** (R80/R90 +0.116/+0.112 above R50's
   +0.087): there IS a genuine, separate OUTER over-extension there, on
   top of the inner deficit. That is a distinct high-z effect and the
   only part of the size story that is new — worth keeping separate in
   the high-z discussion, since the two have different causes and
   probably different fixes.

Adding R80/R90 therefore paid for itself twice: it confirmed the
inner-deficit mechanism and it split a single reported "high-z size
defect" into two effects that a half-mass radius alone conflates.

## Remaining follow-ups (not started)
- Menu option (C), the flexible efficiency window, is now TARGETED by
  follow-up 1 — with the honest caveat that the lever saturates at
  ~+0.67 coherence and so cannot close the gap alone.
