# exp60 — the stochastic layer on the deposition-only model

Branch `exp60-stochastic-layer`. Plan (targets and gates frozen before any
code): `doc/plans/2026-08-26-exp60-stochastic-layer.md`. Everything below is
committed scripts + gitignored outputs/logs; every stage's numbers are in
`outputs/stage*.log`.

**Status: Stage 3 CHECKPOINT — composed and judged held-out; the adoption
packaging is the user's call.** Three components: a cross-epoch-correlated
amplitude draw, an empirical size/shape-axis draw, and a formation-tilted
core-expansion mixture, all applied through a validated prediction cube (no
evaluator modified, no per-draw model call).

## The stages, one line each

- **Stage 0** — targets frozen (declining 41.8%, decliner median −0.066,
  quartile gradient 64.3% → 24.4%, amplitude sigma 0.117–0.142 with
  0.67–0.80 nearest-epoch correlation).
- **Stage 1** — the layer's remap needs **Rc ≈ 20 kpc** (the 2 kpc
  gate-optimal core cannot move the 4.92 kpc target aperture: coverage 9.8%
  vs 85.3%); empirical per-galaxy `A_i`: median +0.98, decliners +2.24 vs
  +0.23, quartile medians 1.27 → 0.68; ceiling stated pre-build: 18.4%
  (the compaction tail) unreachable within bounds.
- **Stage 2** — the individuality axis is the deposit shape `c` (22.3%
  median per-galaxy gain), one axis with the size parameters (rho
  0.88–0.91), feature-orthogonal, nearly independent of the core axis
  (rho = −0.16) — the components draw independently.
- **Stage 3a** — the (dc × A) cube: 221 evaluations, bilinear in log CoG,
  worst validated midpoint error 0.018 dex (tails), ~0.003 (dense centre).
- **Stage 3b-pre** — the S1/S3 tension in numbers: with the incumbent as
  mean, NO pool location both matches the decline distribution and leaves
  the median profile untouched.
- **Stage 3c** — the judgment (below).

## Stage 3c — what passed, what is structurally stuck

Held out (pools from one half, scored on the other, both swaps, 8
realizations), against the frozen targets:

| | declining (41.8%) | med chg (+0.037) | dec med (−0.066) | width (0.537) | S3 inner/outer | S2 planes z=0.4 |
|---|---|---|---|---|---|---|
| no-layer | 0.0% | +0.135 | — | 0.179 | 0 / 0 | 2.73 / 2.91 |
| S3-strict tilted | 26.7% | +0.157 | −0.122 | 0.627 | 0.058 / **0.009** | **1.69 / 2.17** |
| S1-first tilted (shift +0.51) | **42.3%** | **+0.048** | −0.141 | 0.611 | 0.207 / 0.064 | 2.56 / 2.37 |
| S1-scaled tilted (scale 0.20) | **41.4%** | **+0.042** | −0.120 | **0.525** | 0.218 / 0.066 | 2.93 / 2.44 |

**PASSES.** The amplitude component is exact (S5: sigma ratio 1.00 at every
epoch, nearest-epoch correlation offset +0.01) — the drawn population
carries the measured total-mass scatter with its measured coherence in
redshift. S4 (outskirt annulus) holds for every variant. S1's fraction,
overall median and width are all reachable HELD OUT by calibration
(S1-scaled), and the formation-time tilt visibly reproduces the conditional
gradient's shape (tilted 56.6/45.4/37.9/29.1 against the measured
64.3/42.7/35.9/24.4; untilted is flat at 48/44/39/33).

**STRUCTURALLY STUCK, all three now quantified rather than argued.**

1. **Drawn decliners decline ~2× too deep** (−0.120 to −0.141 against the
   measured −0.066) at ANY pool scale — even a near-deterministic pool
   (scale 0.20). An exchangeable draw cannot put the right depth on the
   right galaxy; per-galaxy matching (Stage 1) got both fraction and depth,
   random assignment gets one or the other. This is the
   assignment-information ceiling in its final, sharpest form.
2. **Hitting the decline distribution moves the mean** (0.21 dex inner /
   0.065 outer median shift): the S1-hitting variants effectively adopt a
   mean central correction — the same mean-model territory the user
   declined with pinned-`X3` — through the layer's back door. Passing S3
   instead (the strict variant) caps the declining fraction at 26.7%. The
   layer cannot settle a mean-model question; it can only surface it.
3. **The planes stop at 1.69× the floor** (S3-strict, z=0.4; target 1.3)
   and degrade for the S1-hitting variants. IMPORTANT CAVEAT on the context
   bar: exp41's 1.0–1.4× was achieved on the OLD kernel WITH its
   M*(<500 kpc) oracle pin — zero amplitude scatter to explain — so it is
   not a like-for-like bar for an unpinned model ([[truth-pin-is-an-oracle]]
   applies to the layer record too). Like for like, this layer takes the
   unpinned planes 2.73–2.91 → 1.69–2.17.

## The candidate packagings (the user's call)

- **A. "Population layer" (S3-strict):** mean untouched beyond 10 kpc,
  planes improved ~40%, amplitude statistics exact, 27% decliners with the
  right conditional shape. Honest partial; documented misses.
- **B. "Decline-realistic layer" (S1-scaled):** the measured decline
  distribution reproduced held-out (fraction, median, width) — at the cost
  of an implicit mean central shift and softer planes; adopting it means
  revisiting the mean-model decision with this as evidence.
- **C. Iterate:** the remaining levers (2-D size axis, dc–A coupling,
  radius-structured amplitude noise) target S2; the decliner-depth and
  S1/S3 walls are information-theoretic and will not move.

## ADOPTED (user, 2026-08-27): Option A — with Option B on record

**Option A (the population layer, S3-strict tilted) is hongshao v1's
stochastic layer.** Frozen artifact: `outputs/hongshao_v1_layer.npz`
(`stage3d_adopt.py` regenerates it and the battery) — full-sample
calibration (pool shift −0.592, per-epoch additive amplitude widths
0.114–0.141 dex), with stage3c's held-out table as its validation record.
The standard battery with the drawn populations overlaid lives in
`figures/qa/qa_*_exp60_v1_layer.*`.

**Option B (the decline-realistic S1-scaled variant) is ON RECORD, not
adopted**: pool scale 0.20 with the shift calibrated to the declining
fraction (~+0.37 on the calibration halves) reproduces the measured decline
fraction / overall median / width held out, at the price of a 0.21 dex inner
(0.065 outer) shift of the drawn median profile — an implicit mean-model
change. Revisit triggers, as agreed: (1) 3-D or multi-axis profiles shrink
the orientation component of the decline targets ([[decliner-physics
interpretation]] — the targets themselves are partly projection geometry);
or (2) the mean-model question is reopened on other grounds.

## Closing verdict (user, 2026-08-27)

*"We might be seeing the limitation of the deposition-only model's capability
to emulate the multi-epoch CoGs of massive galaxies."* The layer's measured
walls (the decliner depth, the S1/S3 mean coupling, the high-redshift plane
gap), together with exp57–exp59's closed mechanism and conditioning axes,
are the quantitative form of that statement. The adopted package — the
incumbent mean plus the Option A layer, validated at z ≤ 1.0 — is what the
model class delivers; going further awaits additional data (3-D or
multi-axis profiles, the real merger tree).
