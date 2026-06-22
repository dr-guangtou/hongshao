# exp16 — Secondary-property test: halo concentration `c_200c`

## Question
Do secondary halo properties reduce the residual stellar-mass scatter once we
condition on the assembly history? The roadmap's prior (exp06, literature) was
that concentration is largely *MAH-determined* and should add little. We test it
for `c_200c` (now available from the aperture table). "Test, don't assume."

## Method
Target: the **CoG-derived** aperture/annulus masses (<10, 10–30, 30–50,
50–100 kpc) — the primary observable (per AGENTS.md; the 2-D `*_aper_proj` are
only for cross-checks). Feature sets, 5-fold CV with a per-aperture linear mean +
homoscedastic Gaussian (exp07 suite: CRPS, R², calibration):
M0 only · M0+c200c · DiffMAH(4) · **DiffMAH+c200c** · DiffMAH+shuffled-c200c
(control) · M0+MAH-PCA(4) · **MAH-PCA(4)+c200c**. The last pair is the decisive
test: does `c_200c` help even on top of the *richer* (non-portable) MAH-PCA(4),
or only on top of the smoothed 4-param DiffMAH (exp10: DiffMAH ≈ 88% of the
MAH-PCA signal)? Plus: R²(c_200c | DiffMAH) and the partial correlation of
`c_200c` with each annulus at fixed DiffMAH.

## Key result
**The prior is wrong: `c_200c` carries information genuinely independent of the
MAH, and it improves the portable emulator.** (n=2533)

| features | overall CRPS | R²(50–100) |
|---|---|---|
| M0 only | 0.1128 | 0.679 |
| M0 + c200c | 0.1015 | 0.731 |
| DiffMAH(4) | 0.0882 | 0.810 |
| **DiffMAH + c200c** | **0.0839** (+5.0%) | 0.825 |
| DiffMAH + shuffled c200c | 0.0882 | 0.810 |
| M0 + MAH-PCA(4) | 0.0850 | 0.819 |
| **MAH-PCA(4) + c200c** | **0.0827** (+2.7%) | 0.828 |

- **Real, not overfitting.** Shuffling `c_200c` within M0 bins collapses the gain
  exactly back to DiffMAH (0.0882). The improvement is genuine signal.
- **Independent of the MAH, not just DiffMAH smoothing loss.** `c_200c` helps
  +5.0% on top of the portable DiffMAH(4) *and* +2.7% on top of the richer
  MAH-PCA(4). Since MAH-PCA(4) ≈ the full MAH for these masses (exp13), the
  surviving +2.7% is information the assembly history does not contain.
- **`c_200c` is only weakly MAH-determined**: R²(c_200c | DiffMAH) = 0.25, so
  ~75% of the concentration variance is independent of the (linear) MAH.
- **Partial correlation at fixed DiffMAH is +0.29 to +0.36** (positive at every
  radius, strongest in the core <10 kpc). At fixed assembly history, **more
  concentrated halos host more stellar mass at all radii** — the leftover annulus
  mass slopes clearly with `c_200c` (Fig 2B, not flat).
- Best model overall: **MAH-PCA(4) + c200c = 0.0827**; best *portable* model:
  DiffMAH + c200c = 0.0839.

### Interpretation
Concentration reflects the inner-halo density / early central assembly that the
smooth 4-parameter MAH (and even MAH-PCA) does not fully encode. At fixed MAH, a
denser (more concentrated) halo built a deeper central potential and hosts a more
massive central galaxy — the effect is strongest in the core (partial r +0.36 at
<10 kpc) but persists to the outskirts (+0.29). This is the kind of *independent*
halo information AGENTS.md anticipated might exist; concentration supplies a
modest but real amount of it. Caveat: `c_200c` is measured at z=0.4 (concurrent),
so part of the signal may be unresolved assembly detail (recent mergers /
substructure) rather than a distinct causal axis — but operationally it adds
predictive power beyond both DiffMAH and MAH-PCA(4).

## Decision
- **Add `c_200c` to the feature set.** It is a portable halo property (available
  in any N-body catalog), so `DiffMAH + c_200c` is a free, portable +5% CRPS
  improvement; `MAH-PCA(4) + c200c` is the best in-sample model.
- **Revise the roadmap stance** that MAH-derived secondary properties are
  redundant: concentration is *not* — it overturns the exp06 expectation.
- Next secondary properties to test the same way: accretion rate (`acc_rate`),
  3D halo shape (`c_to_a_3d`, `b_to_a_3d`). And: does `c_200c` also sharpen the
  *scatter* model (exp14) and shrink the regression-to-mean shrinkage (exp15) by
  raising R²?
