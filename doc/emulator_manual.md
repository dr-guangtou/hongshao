# HongShao Emulator — User Manual

How to **train** the Ultimate-SHMR emulator on your own simulation, **predict**
different observables with it, and the **caveats** that will bite you if ignored.

This manual assumes you can already produce, for a set of central galaxies:
- their host halos' **mass accretion histories** (`M_peak` vs cosmic time), and
- the galaxies' **stellar-mass curves of growth** (cumulative `log M*(<R)` on a
  radial grid) — needed only for *training*, not for prediction.

The library is three modules:

| module | role |
|---|---|
| `hongshao.emulator` | the core: fit a mean + heteroscedastic covariance over any target vector; `predict` / `sample` |
| `hongshao.profile_emulator` | target builders (apertures, Re, density) + the full-profile `ProfileEmulator` |
| `hongshao.forward` | the 5-knob deformation layer for external inference |

Supporting: `hongshao.diffmah` (fit the portable MAH parameters),
`hongshao.profiles` (parametric curve-of-growth fits), `hongshao.metrics`
(CRPS / log-score / calibration).

---

## Part 1 — Training the emulator on a new simulation

### 1.1 Build the feature matrix `X` (5 portable halo properties)

The emulator's features are **`[logmp, logtc, early, late, c200c]`** — the order
in `hongshao.emulator.FEATURES`. The first four are DiffMAH parameters of the
main-branch MAH; the fifth is the NFW concentration. All are available in any
N-body simulation.

**DiffMAH parameters** — fit each halo's peak-mass history with
`hongshao.diffmah.fit_mah`:

```python
import numpy as np
from hongshao.diffmah import fit_mah

# Per halo: cosmic time (Gyr) and log10 M_peak at each snapshot, plus the
# observation epoch's age t0_gyr (e.g. ~9.66 Gyr at z=0.4).
params = []
for t_gyr, log_mpeak in zip(times, peak_histories):
    f = fit_mah(t_gyr, log_mpeak, t0_gyr, t_min=0.0)   # t_min drops early, poorly-resolved points
    params.append([f["logmp"], f["logtc"], f["early"], f["late"], f["rms"], f["success"]])
params = np.array(params)
# keep only well-fit halos
ok = (params[:, 5] == 1) & (params[:, 4] < 0.1)
```

`fit_mah` anchors `logmp` at `t0_gyr`, so `logmp ≈ log10 M_halo` at the
observation epoch. See `hongshao/diffmah.py` for the model
(`alpha(logt) = early + (late−early)·sigmoid(k·(logt−logtc))`; transition speed
`k` is fixed at 3.5, Hearin et al. 2021).

**Concentration** `c_200c` — measure it directly from each halo (e.g. from an
NFW fit, or `R_200c / r_s`). No helper is needed; it is a single number per halo.

Then assemble `X` in the exact feature order:

```python
X = np.column_stack([logmp, logtc, early, late, c200c])   # (N, 5)
```

### 1.2 Build the target matrix `Y` (what you want to predict)

The target is built from your galaxies' **curves of growth**: a `(N, R)` array
`cog` of `log10 M*(<R)` at radii `radii` (kpc). Pick a mode:

```python
from hongshao.profile_emulator import aperture_targets, re_targets, density_from_cog

# Mode 1 — fixed kpc apertures/annuli. Column 0 is cumulative M(<edges[0]),
# the rest are annuli. This reproduces the default emulator's 4 targets:
Y = aperture_targets(cog, radii, edges_kpc=[10, 30, 50, 100])      # (N, 4)

# Mode 2 — effective-radius bins (Re = half-mass radius within total_kpc).
# Returns the targets, each galaxy's Re, and a mask dropping galaxies whose
# outer bin exceeds the measured CoG — APPLY IT to both Y and X:
Y_re, Re, mask = re_targets(cog, radii, re_edges=[0.5, 1, 2, 4, 6, 9], total_kpc=120.0)
Y_re, X_re = Y_re[mask], X[mask]

# Mode 4 — the 1-D surface-density profile Sigma(R) = dM/dA (for the profile emulator):
log_sigma, mid_radii = density_from_cog(cog, radii)               # (N, R-1), (R-1,)
```

### 1.3 Fit

**Aperture / Re modes (1 & 2)** — fit the core emulator directly:

```python
from hongshao.emulator import fit

emu = fit(X, Y)                       # default: linear mean + heteroscedastic full covariance
emu_poly = fit(X, Y, mean="poly2")    # optional richer mean (7 degree-2 terms); marginally sharper
```

**Profile modes (3 & 4)** — fit a `ProfileEmulator`, which PCA-compresses the
profile shape, predicts `[total mass, PC1..PCK]`, and reconstructs the per-radius
profile:

```python
from hongshao.profile_emulator import fit_profile

logMtot = cog[:, -1]                                   # the amplitude / anchor

# Mode 3 — cumulative curve of growth:
pe_cog = fit_profile(X, cog[:, :-1], anchor=logMtot, radii=radii[:-1], n_modes=3)

# Mode 4 — surface-density profile:
pe_den = fit_profile(X, log_sigma, anchor=logMtot, radii=mid_radii, n_modes=3)
```

`n_modes=3` is the right default — more PCA modes reconstruct the profile better
but do **not** improve prediction (the higher modes are not halo-predictable).

### 1.4 Validate honestly (cross-validation)

Always judge with out-of-fold predictions, never the training fit. The library's
self-checks show the pattern; the helpers `_cv_oof` (emulator) and `_cv_profile`
(profile) implement 5-fold CV. Score with `hongshao.metrics`:

```python
from hongshao.emulator import _cv_oof
from hongshao.metrics import crps_gaussian, interval_coverage

mu, sigma, cov = _cv_oof(X, Y, mean="linear")
crps = crps_gaussian(Y, mu, sigma).mean()              # lower is better, in dex
levels, coverage = interval_coverage(Y.ravel(), mu.ravel(), sigma.ravel())
# coverage should track levels (0.5/0.68/0.9/0.95) if the uncertainties are honest
```

For a quick sanity benchmark, `python -m hongshao.emulator` reproduces the TNG300
reference numbers (CRPS ≈ 0.083, conditional-coverage gap ≈ 0.01).

---

## Part 2 — Predicting observables

A fitted emulator carries everything it needs; `predict` and `sample` are methods.

### 2.1 The predictive distribution

```python
mu, sigma, cov = emu.predict(X_new)     # (N, T), (N, T), (N, T, T)
```

- `mu` — the conditional mean of each target (the SHMR ridge line).
- `sigma` — the per-target predictive standard deviation (heteroscedastic: it
  varies halo-to-halo).
- `cov` — the **full** target covariance, `diag(sigma) · R · diag(sigma)`, with a
  fixed residual correlation `R`. Use this for joint statistics across radii.

### 2.2 Generative sampling — the right way to use the model

The mean is a *conditional expectation*; it is under-dispersed and regresses to
the mean at the extremes. For any population-level statistic (a stellar-mass
function, a lensing-selected stack, mock catalogs), **sample**:

```python
draws = emu.sample(X_new, size=500, rng=0)     # (500, N, T) correlated, heteroscedastic
```

These draws reproduce the real galaxy population's spread and tails; the mean
alone cannot. (Demonstrated in exp15 / exp22.)

### 2.3 Full profiles

```python
mu_prof, sigma_prof = pe_cog.predict(X_new)    # (N, R) per-radius mean + sigma (analytic)
prof_draws = pe_cog.sample(X_new, size=200)    # (200, N, R) correlated profile draws
scores = pe_cog.scores(X_new)                  # predicted [total mass, PC1..PCK]
```

The CoG reconstruction is linear, so `predict` returns an **analytic** per-radius
mean and sigma — no sampling needed for the uncertainty band.

### 2.4 Deriving one observable from another

Predict the surface-density profile, then **integrate it outward** to recover the
curve of growth and any aperture mass — stably (see Gotcha 3):

```python
from hongshao.profile_emulator import integrate_density

log_sigma_pred, _ = pe_den.predict(X_new)
# central mass M(<R_min): predict it separately as its own 1-target emulator
central = fit(X, cog[:, [0]])                  # log M(<radii[0])
m_central, _, _ = central.predict(X_new)
# pass the EDGE grid `radii` (the one density_from_cog was given), NOT the
# annulus mid radii — the returned CoG lives on radii[1:]
log_cog_pred = integrate_density(log_sigma_pred, radii, central_log_mass=m_central[:, 0])
```

### 2.5 Tuning the relation for external inference (the deformation layer)

For fitting the SHMR to external data, don't vary the emulator's coefficients —
freeze the emulator and infer a 5-knob **deformation** of it. `Deform()` (all
defaults) reproduces the calibrated emulator exactly:

```python
from hongshao.forward import forward, sample, Deform, forward_profile

mu0, sig0, cov0 = forward(emu, X_new)                       # == emu.predict (the baseline)
mu, sig, cov   = forward(emu, X_new, Deform(d0=0.05, d_slope=0.1, f_ab=-0.3, s=1.2))
prof_mu, prof_sig = forward_profile(pe_cog, X_new, Deform(d_out=0.15))   # profile modes
```

| knob | effect | observable it maps to |
|---|---|---|
| `d0` | global stellar-mass normalization | SMF amplitude |
| `d_slope` | tilt vs halo mass | lensing M_halo at fixed M* |
| `d_out` | outskirt-vs-core differential | core/outskirt lensing split |
| `f_ab` | assembly-bias amplitude (`-1` turns it off) | clustering |
| `s` | global scatter scale | SMF shape / lensing spread |

`forward`/`sample` work for any number of targets (apertures, Re); `forward_profile`
deforms a `ProfileEmulator`. Requires a **linear-mean** emulator.

---

## Part 3 — Caveats & gotchas

1. **Use it generatively.** The mean is under-dispersed and shows a
   mass-dependent tilt at the extremes — this is *regression to the mean*, not a
   bug, and cannot be "corrected" in the mean. For population statistics, draw
   from `sample()`. The conditional mean is unbiased *in feature space* (bin by
   prediction, not by truth).

2. **`c_200c` is required, and training needs galaxies.** The features transfer
   to any simulation, but a **gravity-only N-body box has no galaxies**, so you
   cannot *train* on it — training needs stellar-mass curves of growth (hydro or
   a painted catalog). Prediction only needs the 5 halo features. Also: this code
   ships **no pre-fit weights** — you fit on your own labeled sample. (A halo
   catalog without a measured `c_200c` cannot use the default model; a
   DiffMAH-only emulator is weaker — concentration is ~+5% CRPS, mostly on the
   normalization.)

3. **Never get an annulus by differencing a *sampled* cumulative profile.**
   `M(50–100) = M(<100) − M(<50)` is a small difference of two large, nearly-equal
   noisy numbers; sampling the cumulative and subtracting over-disperses the
   annulus wildly (and can go negative). Instead: predict annuli **directly**
   (mode 1 `aperture_targets`), or build the cumulative by integrating the density
   **outward from the centre** (`integrate_density`) — never inward from the total
   or by differencing. The mean is only mildly affected (~0.02 dex); the *samples*
   are what blow up.

4. **Model the reliable quantity; get the other by the stable operation.** On a
   noiseless simulation, predict the **density** profile (more halo-predictable,
   especially in the outskirts) and *integrate* to the CoG. On noisy real data,
   prefer the cumulative CoG and avoid differencing. (The same principle is why
   DiffMAH fits the cumulative MAH and differentiates the smooth model for the
   accretion rate.)

5. **Target = curve-of-growth-derived masses**, the observation-relevant
   quantity. Do not switch the training target to direct 2-D aperture masses
   (`*_aper_proj` in the TNG dataset) — those are for cross-checks only.

6. **There is an irreducible floor (~0.13 dex).** It is the intrinsic
   total-mass SHMR scatter plus projection / ICL / triaxiality noise. No feature
   set (richer MAH, secondary halo properties) or parameterization (PCA, RDM,
   2-Sérsic) we tested beats it — they all converge to the same floor. Don't
   expect a re-parameterization to extract more signal.

7. **The predictable shape signal is mostly *concentration*, and it is one mode.**
   `c_200c` predicts the profile's concentration mode and the total mass; it does
   **not** predict individual parametric shape numbers (radial-DiffMAH inner
   slope, transition radius; Sérsic indices), which are weakly halo-connected
   and often degenerate. Predict the profile (PCA reconstruction or apertures),
   not interpretable shape parameters, if you want accuracy.

8. **Use `n_modes=3`** for profile modes — higher modes reconstruct better but are
   not halo-predictable, so they don't help prediction (and add noise).

9. **Standardization is fit on the training `X`.** Predicting for halos far
   outside the training feature distribution extrapolates the linear mean —
   reasonable near the edges, unsupported far beyond. The training sample here is
   `M_peak(z=0.4) > 10^13 M_sun`; do not expect it to hold for low-mass halos.

10. **Inner ~5 kpc is resolution-limited** (TNG softening). Parametric CoG fits
    (`hongshao.profiles.fit_cog`) use a 5-kpc inner cut for this reason; profile
    reconstructions are most trustworthy at `R ≳ 5 kpc`.

11. **The deformation layer needs a linear-mean emulator** (`fit(..., mean="linear")`,
    the default). It raises on a `poly2` baseline.

---

## Reference: the public API

```text
hongshao.emulator
  FEATURES, TARGETS                      # feature/target name lists
  fit(X, Y, mean="linear", ridge=2.0) -> Emulator
  Emulator.predict(X)                 -> (mu, sigma, cov)
  Emulator.sample(X, size=1, rng=None)-> (size, N, T)

hongshao.profile_emulator
  aperture_targets(cog, radii, edges_kpc)            -> Y
  re_targets(cog, radii, re_edges, total_kpc=120.0)  -> (Y, Re, mask)
  density_from_cog(cog, radii)                       -> (log_sigma, mid_radii)
  integrate_density(log_sigma, radii, central_log_mass) -> log_cog  # on radii[1:]
  fit_profile(X, profile, anchor, radii, n_modes=3, mean="linear", ridge=2.0) -> ProfileEmulator
  ProfileEmulator.predict(X)            -> (mu, sigma)        # per radius
  ProfileEmulator.sample(X, size=1, rng=None) -> (size, N, R)
  ProfileEmulator.scores(X)             -> [anchor, PC1..PCK]

hongshao.forward
  Deform(d0=0, d_slope=0, d_out=0, f_ab=0, s=1)      # defaults = identity baseline
  forward(emu, X, theta=Deform(), norm_weight=None, outer_weight=None) -> (mu, sigma, cov)
  sample(emu, X, theta=Deform(), size=1, rng=None, ...) -> (size, N, T)
  forward_profile(pe, X, theta=Deform(), outer_radius_kpc=50.0) -> (mu, sigma)

hongshao.diffmah
  fit_mah(t_gyr, log_mpeak, t0_gyr, t_min=0.0) -> {logmp, logtc, early, late, rms, success}

hongshao.metrics
  crps_gaussian(y, mu, sigma) ; gaussian_logscore(...) ; interval_coverage(...) ; aic_bic(...)
```

Every library module has a runnable `__main__` self-check —
`python -m hongshao.<module>` — that exercises the API and reproduces the
reference numbers.
