# exp40 — two scope tests on the adopted kernel (user plan, 2026-07-18)

Both tests probe whether the core deficit is tied to the FIT SCOPE
rather than the model form: (1) the starting epoch — at z ~ 2 some
galaxies still form stars and mergers can be dissipative (induce star
formation), regimes outside the additive collisionless transport
model; (2) the objective — observationally the inner profile shape is
not trusted, but M*(<5 kpc) and M*(<10 kpc) are essential, so
"encourage" the aperture masses instead of fitting the inner profile.

Model everywhere: the adopted exp38 1ch-mof kernel (12 parameters),
unchanged. Run: `PYTHONPATH=. uv run python
experiments/exp40_epoch_objective/tests.py
{demo|latestart|latephysics|latestress|latecv|reaim|reaimcv} [--dev]`.

## Test 1 — late-start fits (`latestart`): the transport tax lives in
## the dissipative era; the late fits are consistent and physics-clean

Joint plain-loss fits from later starting epochs, full n=2397.
In-sample M(<5 kpc) bias at z=0.4 vs the record's brackets:

| fit scope | M(<5) z=0.4 in-sample | held-out | params [log_rc, g, q, mu, sig, gamma] |
|---|---|---|---|
| z<=2.0 (adopted, 5 epochs) | -11.7% | ~-11% | [2.37, 3.64, 0.91, 1.47, 0.28, 1.38] |
| z<=1.5 (4 epochs) | -9.8% | -9.5% | [2.74, 4.00g, 0.91, 1.52, 0.24, 1.38] |
| z<=1.0 (3 epochs) | -8.3% | -8.2% | [2.69, 4.00g, 0.79, 1.46, 0.23, 1.37] |
| z=0.4 only (the population-sharing wall) | -7.6% | — | q = 0.00 |

1. **The deficit shrinks monotonically toward the sharing wall as the
   dissipative era is excluded** — the z>=1.5 epochs carry most of the
   ~4-point multi-epoch transport tax. What remains at a z<=1.0 start
   (-8.2% held-out) is essentially the population-sharing limit, which
   no shared fit can cross. The user's hypothesis is SUPPORTED:
   forcing the collisionless kernel through the dissipative era is
   what drains the core, beyond the unavoidable sharing wall.
2. **The two late starts are consistent with each other**: parameters
   agree to 0.05-0.12 (the g rail is shared and benign, below), inner
   biases follow one monotone trend, and both pass the physics.
3. **Physics improves, not degrades** (in-sample, same convention):
   differential massive z0.7->0.4 = 0.40/0.13 (z15) and 0.39/0.12
   (z10) vs data 0.37/0.11; the outskirt overshoot HALVES (T1
   +0.025/+0.028 and +0.020/+0.029 vs the adopted fit's
   +0.047/+0.070). Most striking: the late fits predict the
   EXTRAPOLATED z2.0->1.5 differential pair better than the
   z=2-anchored fit that was trained on it (0.21/0.06 vs 0.18/0.05;
   data 0.23/0.06; the g-stress fit hits it exactly, 0.23/0.06).
   Late-epoch-calibrated transport extrapolates up better than
   high-z-anchored transport fits down.
4. **Held-out accuracy is unchanged on the fitted epochs**: z15 CV
   shape 18.2/17.4/16.6/16.4% vs the adopted marks
   18.5/17.6/16.7/16.2; the unfitted z=2 epoch costs only +1.2 shape
   points as pure extrapolation (15.4 vs 14.2). z10 CV:
   17.5/17.2/17.2 fitted; extrapolation degrades harder (z1.5*/z2.0*
   shape 17.9/17.7, M(<5) -14.2/-15.5) — the 3-epoch scope is too
   short a lever arm for the high-z extrapolation; z<=1.5 is the
   better scope if one is adopted.
5. **The shared g = 4.00 rail is BENIGN** (`latestress`, the exp35
   protocol: g box 4 -> 6): the optimum settles interior at g = 4.37
   for a 0.1% loss gain — a box clipping a nearly-flat ridge (log_rc
   drifts toward its own box on the same ridge; the known
   horizon-flat direction) — with the physics unchanged. NOTE: the
   same railed value (g = 4.0) was the PATHOLOGICAL second basin in
   the 5-epoch fit (worse physics) and is harmless here (better
   physics) — a rail's meaning is scope-dependent; stress + physics,
   never pattern-match the value.

## Test 2 — the re-aimed fiducial judged (`reaim`, `reaimcv`): the
## exp39 prediction confirmed with zero added components

The exp38 "capacity" theta (the fiducial 12-parameter kernel refit
under the inner-aware objective: R>5 shape + M(<5)/M(<10) aperture
terms) had never been run through the judged tests:

| metric | adopted kernel | re-aimed fiducial |
|---|---|---|
| held-out shape by epoch [%] | 18.5/17.6/16.7/16.2/14.2 (avg 16.6) | 16.9/17.1/16.0/13.3/11.0 (avg 14.9) |
| held-out M(<5) / M(<10), z=0.4 | ~-11% / -5.4% | -6.4% / -4.1% |
| differential massive z0.7->0.4 (data 0.37/0.11) | 0.39/0.12 PASS | 0.48/0.15 FAIL |
| outskirt terciles T1 [dex] | +0.026/+0.019 | +0.047/+0.091 |
| bounds | NONE | mu = 3.00 railed |

The re-aimed fiducial reaches best-tier held-out accuracy and halves
the inner deficit — and STILL breaks the differential, with no core,
no retention, nothing added. exp39's conclusion is confirmed in its
purest form: the inner-aware objective alone drives the kernel into
the physics-breaking configuration (mu railed, near-flat envelope
q = 0.28, broad sig = 1.17 — a different basin altogether). The
inner-aware objective is NOT adoptable for the fiducial fit.

## Combined reading

The core deficit decomposes cleanly across the two tests: about half
the transport tax is the price of forcing the collisionless kernel
through the dissipative z>=1.5 era (test 1 removes it by scope), the
rest is the population-sharing wall (untouchable by any shared fit);
trying to pay the tax through the OBJECTIVE instead (test 2) buys the
same inner masses but always at the physics' expense. The one genuine
free lunch found: a z<=1.5 fit scope — same held-out accuracy, better
physics, -1.5 points of inner deficit, and honest extrapolation to
z=2.

## ADOPTION (user, 2026-07-18)

**z<=1.5 is the kernel's OFFICIAL fit scope** (theta in
`outputs/latestart.npz[theta_z15]`; the g = 4.0 rail is
stress-verified benign). **The z<=2.0 five-epoch fit remains available
as the comparison option** (exp38
`outputs/stage2_multiepoch.npz[theta_1ch-mof]`). The inner-aware
objective is NOT adopted for fitting (test 2); it remains an
evaluation metric.
