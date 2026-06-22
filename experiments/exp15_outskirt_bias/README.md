# exp15 — The low-mass outskirt "bias" is regression to the mean, not a defect

## Question
exp13's truth-vs-predicted figure for `M*[50-100 kpc]` shows the binned-median
prediction sitting **+0.2 to +0.4 dex above the truth at the low-mass end** — a
~2× over-prediction that looks like a serious bias and a blocker for the end
goal. Is it (i) a genuine mis-specification of the mean we can fix, or (ii) the
unavoidable regression-to-the-mean artifact of binning residuals against the
*noisy* truth?

## Method
Out-of-fold (5-fold CV) predictions for `M*[50-100 kpc]` from the four DiffMAH
params: the linear mean, a nonlinear mean (+`late²`+`logmp²`, exp12), and the
heteroscedastic σ(X) (exp14). Two diagnostics that answer *different* questions,
plus a data-floor check and a generative check:
- **Bin by TRUE Y, plot residual** (what exp13 showed). For any predictor with
  intrinsic scatter, `E[pred−Y | Y]` has slope `−(1−R²)`, derived and overlaid —
  because the truth contains the very noise we condition on.
- **Bin by PREDICTED Y, plot mean true** (reliability diagram). `E[Y | pred]=pred`
  ⇔ the mean is well-specified and there is no fixable bias. *This* is the real
  test.
- **Data floor:** is the low tail a measurement/annulus-clipping artifact?
- **Generative check:** does sampling `μ(X)+σ(X)·z` reproduce the true population
  (incl. the low tail), which the under-dispersed point estimate cannot?

## Key result
**The low-mass "bias" is regression to the mean — a property of the plot, not the
model. The mean is unbiased; the model is sound. Used generatively (as it must be
for painting), it reproduces the population including the low tail.**

- **The residual-vs-true slope is exactly `−(1−R²)`.** Measured −0.190 vs theory
  −0.190 (R²=0.810) at full n (and −0.253 vs −0.254 on the easy subsample). The
  downward slope in exp13 is pure regression to the mean — it appears for *any*
  predictor (mean or sampled) because the noisy truth is on the x-axis.
- **The mean is unbiased in prediction space.** Binned by *predicted* value, the
  reliability slope is **+0.999** and `⟨true⟩−pred` stays within ±0.06 dex across
  all deciles. The nonlinear mean (+late²+logmp²) lies on the same 1:1 line — so
  there is **no fixable bias to remove**; OLS already gives `E[Y|X]` correctly.
- **The data is clean** (setup probe): zero clipped/unphysical annuli, only 2/2539
  galaxies below 10⁹ M⊙. The low tail is a real population — lower-mass, **high
  `late` (recent accretion; 1.5 vs 0.15) and high scatter (σ 0.21 vs 0.17)** — the
  same noisy population exp14 flagged. Its extra scatter slightly steepens the
  *local* regression-to-mean slope, which is why the extreme bin reaches +0.4.
- **Sampling restores the low tail.** The mean-only point estimate is
  under-dispersed (std 0.371 vs true 0.412) and recovers only **5%** of galaxies
  below the 10th percentile (true 10%). Drawing from the predictive `N(μ,σ²)`
  gives std 0.414 and **9%** — matching the truth. The emulator already draws
  (correlated, heteroscedastic) scatter, so the painted population is unbiased.

## Decision
- **No model change.** The mean is unbiased in X (reliability slope 1.0); the
  apparent low-end bias is regression to the mean and cannot be removed by a
  better mean without making the predictor *biased in X* (worse for prediction).
- **Use the model generatively, never as a point estimate.** For the Ultimate-SHMR
  end goal (painting profiles onto halos), sample from `N(μ(X), Σ(X))`; panel C
  shows this reproduces the true distribution including the low-mass tail. Report
  the relation as a reliability diagram (bin by predicted), not residual-vs-truth.
- **The only way to shrink the shrinkage is more explained variance**, and exp13
  showed R² is already at its ceiling for the outskirts (the rest is intrinsic
  projection/ICL noise). So −(1−R²) ≈ −0.19 is as flat as the point estimate can
  be with the information available.
- **Graduation unblocked.** The concern is resolved; the emulator (linear mean +
  heteroscedastic full covariance, used generatively) is sound to graduate.
