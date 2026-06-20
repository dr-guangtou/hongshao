# exp06 — The full assembly history via PCA (no DiffMAH); connecting halo and galaxy modes

## Questions

1. How many numbers does the assembly history actually need? (PCA the full
   `Mpeak(t)` curve, not hand-picked summaries, no DiffMAH.)
2. Do the **MAH principal components** connect to the **profile principal
   components** at fixed final halo mass?
3. Does data-driven MAH-PCA predict the profile better than the hand-picked
   summaries used in exp04/05?
4. (User's side question) Is the curve of growth as compressible as exp02 found,
   and how does it compare to the surface-density profile?

## Method

- Each halo's main-branch `log Mpeak(t)` is interpolated onto a common cosmic-time
  grid (2.2–9.0 Gyr, z ≈ 2.9 → 0.46; 99.6% of the sample is covered, n = 2534)
  and **normalized by M0** → `log[Mpeak(t)/M0]`, isolating assembly *shape* at
  fixed final mass. Then covariance PCA. No DiffMAH, no `halo_id` cross-match.
- CoG-shape PCA recomputed (exp02), plus a differential surface-density-like PCA
  (annulus mass / area from the CoG) for the compressibility comparison.
- Connection: partial Spearman of MAH-PC vs CoG-PC, controlling for M0.
- Head-to-head: predict the full CoG (5-fold CV linear) from M0 / hand-picked
  summaries / MAH-PCA, with a shuffle control.

Driver: `run.py`. Figures: `exp06_mah_pca`, `exp06_connection`, `exp06_compressibility`.

## Key results

**1. The assembly history needs ~3–4 numbers.** MAH-PCA variance: PC1 = 73%,
PC2 = 15%, PC3 = 5% (PC1–3 = 93%; ~6 modes for 99%). PC1 is the overall
early-vs-late assembly level (formation time); PC2 is the *timing* of mass
buildup (a mid-history growth deficit/excess); PC3 finer structure.

**2. MAH-PCA matches the hand-picked summaries** for predicting the profile:

| halo representation | full-CoG RMS | improvement vs M0 |
|---|---|---|
| M0 only | 0.152 dex | — |
| hand-picked summaries (7) | 0.1177 dex | +22.6% |
| **MAH-PCA (top 4)** | **0.1173 dex** | **+22.9%** |
| MAH-PCA shuffled | 0.152 dex | −0.0% |

So 4 principled, data-driven numbers do as well as the 7 hand-chosen summaries.
Two readings: (a) the hand-picked summaries weren't leaving much on the table —
reassuring; (b) MAH-PCA is the cleaner, complete, less arbitrary representation
to carry forward, and it uses the whole curve with no parametric assumption.

**3. Halo and galaxy principal components connect** (partial Spearman, fixed M0):

| | CoG-PC1 (concentration) | CoG-PC2 | CoG-PC3 |
|---|---|---|---|
| MAH-PC1 (level/formation) | +0.11 | +0.23 | +0.21 |
| **MAH-PC2 (timing)** | **+0.46** | +0.12 | −0.12 |
| MAH-PC3 | +0.00 | −0.03 | +0.07 |

There is a clear, compact mode-to-mode link: the **second assembly mode (the
timing of mass buildup) drives the first profile mode (concentration)** most
strongly (r = 0.46), with the dominant MAH level mode feeding the secondary
profile modes. This is the assembly→profile connection expressed directly in
principal-component space.

**4. Compressibility (the CoG-vs-Σ question).** Cumulative variance in 3 modes:
**curve of growth 99.7%**, **surface density (Σ-like) 97.4%**, **assembly history
93.0%**. The CoG is the most compressible (it is cumulative and smooth); the
differential Σ needs slightly more components for the same fidelity; the assembly
history is the richest object of the three. So exp02's "~3 modes" holds for the
CoG (even more so than for Σ).

## Interpretation & caveats

- Using the *full* MAH does not unlock much *more* profile-prediction power than
  good summaries (~23% either way). The ceiling on this dataset is set by what's
  missing (secondary halo properties) and irreducible projection noise — not by
  how we compress the MAH. MAH-PCA's value is being principled and complete, not
  a big accuracy jump.
- Σ-like here is differenced from the CoG (a proxy), not the raw isophote profile.
- The grid starts at z ≈ 2.9; the earliest assembly (z > 3) is not represented.

## Decision

Adopt **MAH-PCA** (top ~4 components of the M0-normalized log-MAH) as the
principled halo-side representation going forward — it equals the hand-picked
summaries, uses the entire history, needs no DiffMAH fit or `halo_id` match, and
connects cleanly to the profile modes. This completes Phase 3 (MAH compression).
Next: the probabilistic emulator (exp07), now with MAH-PCA inputs.
