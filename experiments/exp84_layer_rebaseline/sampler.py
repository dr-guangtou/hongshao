"""exp84 — THE SAMPLER: the layer's draws around the adopted mean.

Per galaxy, a (5, 2) size-deviation vector [dex] — the compact and the
extended deposit size at each of the five epochs — and a (5,) amplitude
deviation, both from correlated Gaussians whose widths and correlations are
FITTED ON THE CALIBRATION HALF (Stage 1's anatomy, Stage 0's targets) and
never touch the scoring half:

  * sizes: dev = scale * sigma_col * eps, eps ~ N(0, C10), C10 the 10x10
    correlation (5 epochs x 2 axes) of the anatomy's joint deltas, sigma_col
    their per-column half 16-84 width. THE DRAWS ARE CENTRED (the anatomy's
    column medians are NOT applied): a non-zero median deviation would be a
    mean-model correction through the layer's back door (exp60's wall 2,
    the S3-strict convention of the adopted v1). The medians are reported.
    `form="rows"` is the nonparametric control: whole (5, 2) records of
    calibration-half galaxies handed to scoring-half galaxies, centred.
    `corr="identity"` (independent epochs) and `corr="one"` (a persistent
    trait) are the persistence controls. `offsets` (10,) [dex], zero by
    default, shift the draw's centre: the PROFILE-CENTRED variant calibrates
    them so the drawn median profile stays on the mean at each axis's
    reference radius (a symmetric log-size draw moves the median profile,
    because the profile's response to a larger and a smaller deposit is not
    symmetric — Jensen; exp60 saw the same 0.058 dex inside 10 kpc).
  * amplitude: eps ~ N(0, amp_corr) * sig_add, added to the log profile at
    every radius; sig_add per epoch is calibrated so the TOTAL drawn
    deviation at 103 kpc meets Stage 0's target (the width the size draws
    already induce is subtracted in quadrature — exp60's device).
"""
from __future__ import annotations

import numpy as np

import layer_common as LC


def nearest_psd(c, eps=1e-6):
    """The nearest correlation-like matrix with eigenvalues >= eps, unit diagonal."""
    w, v = np.linalg.eigh(0.5 * (c + c.T))
    c2 = (v * np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / d[:, None] / d[None, :]


class Sampler:
    N_AMP_REAL = 8

    def __init__(self, delta_c, delta_e, amp_sigma, amp_corr, calib_idx, form="gauss", corr="fitted", scale=1.0):
        both = np.column_stack([delta_c, delta_e])[calib_idx]            # (m, 10)
        self.medians = np.nanmedian(both, axis=0)
        centred = both - self.medians[None, :]
        self.sigma = LC.half_width(centred)                              # (10,)
        c10 = np.full((10, 10), np.nan)
        for a in range(10):
            for b in range(10):
                g = np.isfinite(centred[:, a]) & np.isfinite(centred[:, b])
                c10[a, b] = np.corrcoef(centred[g, a], centred[g, b])[0, 1] if g.sum() > 10 else 0.0
        self.corr_fitted = nearest_psd(c10)
        self.form, self.corr_kind, self.scale = form, corr, float(scale)
        if corr == "fitted":
            self.corr = self.corr_fitted
        elif corr == "identity":
            self.corr = np.eye(10)
        elif corr == "one":
            # a persistent trait per axis, the two axes correlated as the anatomy's same-epoch mean
            r_ce = float(np.nanmean([c10[j, 5 + j] for j in range(5)]))
            self.corr = nearest_psd(np.block([[np.ones((5, 5)), r_ce * np.ones((5, 5))],
                                              [r_ce * np.ones((5, 5)), np.ones((5, 5))]]))
        else:
            raise ValueError(corr)
        self.L = np.linalg.cholesky(self.corr + 1e-9 * np.eye(10))
        self.pool = centred[np.isfinite(centred).all(axis=1)]           # for the nonparametric form
        self.amp_sigma, self.amp_corr = np.asarray(amp_sigma, float), np.asarray(amp_corr, float)
        self.L_amp = np.linalg.cholesky(self.amp_corr + 1e-9 * np.eye(5))
        self.sig_add = None                                              # set by calibrate_amplitude
        self.offsets = np.zeros(10)                                      # set by calibrate_offsets (profile-centred)
        self.scale_axis = np.ones(2)                                     # per-axis scales (the two-scale variant)

    def describe(self):
        return (f"form={self.form} corr={self.corr_kind} scale={self.scale:.3f} x (c {self.scale_axis[0]:.3f}, e {self.scale_axis[1]:.3f}); offsets c "
                + " ".join(f"{v:+.2f}" for v in self.offsets[:5]) + " e " + " ".join(f"{v:+.2f}" for v in self.offsets[5:])
                + "; sigma_c "
                + " ".join(f"{v:.3f}" for v in self.sigma[:5]) + "; sigma_e " + " ".join(f"{v:.3f}" for v in self.sigma[5:])
                + "; anatomy medians c " + " ".join(f"{v:+.2f}" for v in self.medians[:5])
                + " e " + " ".join(f"{v:+.2f}" for v in self.medians[5:]) + " (not applied)")

    def draw_sizes(self, n, rng):
        """dict(c=(n, 5), e=(n, 5)) in dex."""
        if self.form == "rows":
            v = self.pool[rng.integers(0, len(self.pool), n)] * self.scale
        else:
            v = (rng.standard_normal((n, 10)) @ self.L.T) * self.sigma[None, :] * self.scale
        v = v * np.repeat(self.scale_axis, 5)[None, :] + self.offsets[None, :]
        return dict(c=v[:, :5], e=v[:, 5:])

    def draw_amplitude(self, n, rng):
        """(n, 5) dex, the additive amplitude deviation (zero before calibration)."""
        if self.sig_add is None:
            return np.zeros((n, 5))
        return (rng.standard_normal((n, 5)) @ self.L_amp.T) * self.sig_add[None, :]

    def profiles(self, predict_fn, n, rng):
        """One realization: predict_fn(size_dev) -> (n, 5, nr) Msun; returns
        the drawn CoGs (Msun) with the amplitude deviation applied."""
        dev = self.draw_sizes(n, rng)
        m = predict_fn(dev)
        return m * 10.0 ** self.draw_amplitude(n, rng)[:, :, None]

    def calibrate_amplitude(self, predict_fn, mean, n, rng):
        """sig_add per epoch so the TOTAL drawn deviation at 103 kpc meets the
        target; returns the width the size draws induce on their own."""
        devs = []
        for _ in range(self.N_AMP_REAL):
            m = predict_fn(self.draw_sizes(n, rng))
            with np.errstate(invalid="ignore", divide="ignore"):
                devs.append(np.log10(m[:, :, LC.F.I100] / mean[:, :, LC.F.I100]))
        induced = LC.half_width(np.concatenate(devs))
        self.sig_add = np.sqrt(np.clip(self.amp_sigma ** 2 - induced ** 2, 0.0, None))
        return induced

    #: the reference radius (index on R_GRID) at which each axis's offset
    #: centres the drawn median profile: 4.9 kpc for the compact size, 32.6 kpc
    #: for the extended one (where each axis's response is largest)
    CENTRE_INDEX = {"c": 3, "e": 12}

    def calibrate_offsets(self, predict_fn, mean, n, rng, n_real=4, n_iter=8):
        """The profile-centred variant: per axis and epoch, the offset that
        puts the drawn MEDIAN log M(<R_ref) on the mean's, by damped fixed-
        point iteration on the median residual (the response is close to
        linear in the offset near zero); the two axes alternate."""
        lmean = np.log10(np.clip(mean, 1.0, None))
        hist = []
        for it in range(n_iter):
            devs = [self.draw_sizes(n, rng) for _ in range(n_real)]
            lg = np.stack([np.log10(np.clip(predict_fn(dv), 1.0, None)) for dv in devs])
            med = np.nanmedian(lg.reshape(-1, *lg.shape[2:]), axis=0) - np.nanmedian(lmean, axis=0)   # (5, nr)
            for a, (lo, hi) in (("c", (0, 5)), ("e", (5, 10))):
                r = med[:, self.CENTRE_INDEX[a]]                             # (5,) dex, the median excess
                # a larger deposit LOWERS the enclosed mass at its radius: push the offset against the excess
                self.offsets[lo:hi] += 0.6 * r / max(self.RESPONSE[a], 1e-3)
            hist.append(float(np.max(np.abs(med[:, [3, 12]]))))
        return hist

    #: dlog M(<R_ref) per dex of size deviation, from the engine selfcheck
    #: (+0.3 dex: -0.07 to -0.09 at 4.9 kpc for c, -0.03 to -0.12 at 32.6 kpc
    #: for e): the fixed point's slope, only the step size depends on it
    RESPONSE = {"c": 0.27, "e": 0.25}
