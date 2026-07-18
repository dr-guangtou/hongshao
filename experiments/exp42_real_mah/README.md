# exp42 — the adopted kernel on the real MAH (completing the record)

The adopted 1ch-mof kernel has only ever been fitted in the DiffMAH
configuration (the smooth reconstructed halo history, ~99 evenly-spaced
deposits). The raw-input head-to-head — the de-dipped real main-branch
peak history, which keeps merger bursts (`real_mah`: running-max, same
deposit-dict keys) — was run on the earlier model generations and never
repeated for the Moffat kernel:

- per-galaxy fits (exp29, Gaussian deposits): the smooth curve FLATTERS
  the model by ~2 points (joint multi-epoch max|rel| 4.4% -> 6.1% on the
  real MAH);
- population-shared fits (exp30 phase 4 / exp32 step 2, 5-7-param
  transport): DiffMAH is BETTER by 3-5 points held-out (LOGO 30.6% vs
  33.6%; 10-fold CV epoch-avg 30.4% vs 35.2%) — the smooth regular basis
  suits one shared theta, the bursty gappy real history demands
  per-galaxy adaptation;
- the real MAH's virtue was population DIVERSITY (centered plane energy
  3.5x floor, the era's best) at a badly biased amplitude — a job now
  done by the exp41 stochastic layer.

This experiment closes the gap: the SAME 12-parameter 1ch-mof structure,
the SAME official z<=1.5 fit scope and protocol (exp40 `latestart`), with
config = "real" instead of "diffmah". Model code is exp38
`stage2_multiepoch.py` unchanged; only the loader config differs.

## Judged comparison (the exp40 z15 records are the marks)

| metric | DiffMAH (official, exp40) | real MAH (this exp) |
|---|---|---|
| joint z<=1.5 loss | 0.1538 | ? |
| params [log_rc, g, q, mu, sig, gamma] | [2.74, 4.00g, 0.91, 1.52, 0.24, 1.38] | ? |
| held-out shape R>5 by epoch (z=2.0 = extrapolation) | 18.2/17.4/16.6/16.4 / 15.4* | ? |
| held-out M(<5) z=0.4 | -9.5% | ? |
| differential massive z0.7->0.4 (data 0.37/0.11) | 0.40/0.13 | ? |
| overshoot T1 [30-60 / 60-148 kpc] | +0.025/+0.028 | ? |
| bounds | g=4.00 (stress-verified benign) | ? |

Also measured: the TRANSFER cross — the official DiffMAH-fitted theta
evaluated on the real-MAH input without refitting — which separates "the
input changed" from "the fit adapted".

Expectation from the record: the real input loses a few points of
held-out accuracy at the population level; open questions are whether the
Moffat tail (which absorbs shape freedom the Gaussian lacked) shrinks
that gap, and whether the physics tests (fitted to nothing) move.

## Run

```bash
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
PYTHONPATH=. uv run python experiments/exp42_real_mah/compare.py \
    {demo|fit|physics|cv} [--dev]
```

`fit` = joint z<=1.5 plain-loss fit on the real MAH (warm-started from
the official theta + a weaker-envelope nudge; parent-side penalties);
`physics` = differential + overshoot for {official theta on real input
(transfer), real-fitted theta}; `cv` = the exp40 10-fold CV protocol on
the real input (z=2.0 column = extrapolation). Long runs via
nohup + caffeinate -im + disown with a log tail.

## Results

(pending)
