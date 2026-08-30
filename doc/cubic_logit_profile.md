# Cubic-logit projected profile reference

## Status and intended use

The **cubic-logit profile** is a five-parameter analytic representation of a
projected curve of growth (CoG). It was found in Exp62 by modeling the logit of
the enclosed stellar-mass fraction as a monotonic cubic in log radius. The
name states the construction and does not assign physical meanings to its
three shape coordinates.

The family is designed to interpolate HongShao's resolved 2--148 kpc CoGs.
Its continuation to zero and infinite radius is mathematically well defined,
but the central continuation is not a physical galaxy core and some fits have
poorly constrained outer totals. Use the aperture-normalized interface for
HongShao predictions unless an infinite total is specifically required.

## Definition and parameters

Let

\[
x=\ln(R/R_{50}), \qquad
F(R)=\frac{M(<R)}{M_{\rm tot}}=\sigma[P(x)],
\]

where \(\sigma(u)=(1+e^{-u})^{-1}\) and

\[
P(x)=a x+b x^2+c x^3, \qquad
a=m+\frac{b^2}{3c}, \qquad m>0,\quad c>0.
\]

The five intrinsic parameters are:

- \(M_{\rm tot}\): infinite-aperture projected mass;
- \(R_{50}>0\): infinite-aperture projected half-mass radius;
- \(m>0\): minimum derivative of the logit warp;
- \(b\): asymmetric quadratic term;
- \(c>0\): cubic curvature.

This parameterization makes monotonicity exact because

\[
P'(x)=m+3c\left(x+\frac{b}{3c}\right)^2>0.
\]

Therefore \(F\) rises strictly from zero to one. Because \(P(0)=0\),
\(F(R_{50})=1/2\) exactly and this half-mass radius is unique.

For a measured mass \(M_{\rm ap}=M(<R_{\rm ap})\), the appropriate CoG is

\[
M(<R)=M_{\rm ap}\frac{F(R)}{F(R_{\rm ap})}.
\]

In particular, HongShao's \(M_{*,148}\) is not silently relabeled as the
infinite total.

## Exact enclosed radii

For any fraction \(0<p<1\), set \(L=\operatorname{logit}(p)\) and substitute
\(x=y-b/(3c)\). The radius equation becomes the depressed cubic

\[
y^3+q_1y+q_0=0,
\]

with

\[
q_1=\frac{m}{c}>0, \qquad
q_0=-\frac{b^3}{27c^3}-\frac{bm}{3c^2}-\frac{L}{c}.
\]

Because \(q_1>0\), this equation has one real root. A numerically stable exact
form is

\[
y=-2\sqrt{\frac{q_1}{3}}\,
\sinh\left[\frac{1}{3}\operatorname{asinh}\left(
\frac{3q_0}{2q_1}\sqrt{\frac{3}{q_1}}
\right)\right],
\]

and

\[
R_p=R_{50}\exp\left(y-\frac{b}{3c}\right).
\]

Thus \(R_{20}\), \(R_{50}\), \(R_{80}\), and any other quantile require no
root finder. The implementation returns zero and infinity for \(p=0\) and
\(p=1\), respectively.

## Projected surface density

Differentiating the CoG gives the circular projected surface density

\[
\Sigma(R)=\frac{M_{\rm tot}}{2\pi R^2}
F(R)[1-F(R)]P'(x).
\]

Its logarithmic slope is

\[
\frac{d\ln\Sigma}{d\ln R}
=-2+\frac{P''(x)}{P'(x)}+[1-2F(R)]P'(x),
\qquad P''(x)=2b+6cx.
\]

The stationary radii of \(\Sigma\) are roots of this transcendental slope
equation and are found numerically. The cubic-logit tails fall faster than any
power of radius, so \(\Sigma\rightarrow0\) both as \(R\rightarrow0\) and as
\(R\rightarrow\infty\), and all radial moments are finite.

The inward limit is the important caveat: every permitted profile has a
formal central density hole and at least one density maximum at nonzero
radius. A nonnegative spherical three-dimensional density also cannot project
to a surface density with \(\Sigma(0)=0\) and positive mass, since
\(\Sigma(0)=2\int_0^\infty\rho(r)\,dr\). The cubic-logit family should
therefore be treated as a projected, finite-resolution CoG model, not as a
globally physical spherical density law or a direct deprojection model.

## Two-dimensional rendering

The circular profile is rendered by evaluating \(\Sigma(R)\) on a pixel grid.
For projected axis ratio \(0<q\leq1\), position angle \(\phi\), and rotated
coordinates \((x',y')\), use

\[
R_{\rm ell}^2=x'^2+(y'/q)^2, \qquad
\Sigma_q(x,y)=\frac{1}{q}\Sigma(R_{\rm ell}).
\]

The factor \(1/q\) preserves the enclosed mass. The library returns the
density at pixel centers; subpixel integration is advisable when either the
central turnover or the PSF is comparable to a pixel.

## Fourier transform and PSF convolution

For a circular profile, the normalized two-dimensional Fourier transform is
the order-zero Hankel transform

\[
T(k)=\frac{2\pi}{M_{\rm tot}}\int_0^\infty
\Sigma(R)J_0(kR)R\,dR
=\int_{-\infty}^{\infty}
J_0(kR_{50}e^x)F(1-F)P'(x)\,dx.
\]

No elementary closed form is known. `cubic_logit_fourier_transform` evaluates
this integral using fixed Gauss--Legendre quadrature in log radius. It obeys

\[
T(0)=1, \qquad
T(k)=1-\frac{k^2\langle R^2\rangle}{4}+O(k^4).
\]

Circular convolution is simple in Fourier space: multiply \(T(k)\) by the
normalized PSF transfer function and transform back. For a circular Gaussian,

\[
\widetilde{G}(k)=\exp(-\sigma^2k^2/2).
\]

For the normalized Moffat convention

\[
P_{\rm M}(R)=\frac{\beta-1}{\pi\alpha^2}
\left[1+(R/\alpha)^2\right]^{-\beta}, \qquad \beta>1,
\]

the transfer function is

\[
\widetilde{P}_{\rm M}(k)=
\frac{2^{2-\beta}}{\Gamma(\beta-1)}
(k\alpha)^{\beta-1}K_{\beta-1}(k\alpha),
\]

with value one at \(k=0\). Neither Gaussian nor Moffat convolution remains a
cubic-logit profile in real space. Use radial Hankel multiplication for a
circular source and PSF; use the rendered two-dimensional image and a 2-D FFT
for an elliptical source or PSF. The latter avoids the known limitations of
treating an intrinsically elliptical source with a circularized seeing
correction.

## Exp62 population audit

The audit used all 16,900 independently fitted profiles: 3,380 galaxies at
each of five epochs, with no refitting. The promoted implementation reproduces
the stored Stage 5 cubic-logit CoGs over 2--148 kpc to a maximum difference of
\(1.78\times10^{-15}\) dex.

- The median global surface-density maximum is 0.69--0.73 kpc across the five
  epochs. Between 99.94% and 100% of fits have that maximum below the 2 kpc
  inner measurement radius, and only 5 of 16,900 profiles are still rising at
  2 kpc. Thus the formal central hole is almost always outside the measured
  radial range.
- The density maximum encloses a median 1.48% of the infinite total at
  \(z=0.4\), rising to 5.17% at \(z=2\). In 2.08% of all fits it encloses less
  than \(10^{-6}\) of the total; these are extreme inward extrapolations, not
  resolved core measurements.
- Only 36 of 16,900 profiles have more than one density maximum, and only 23
  fail to decrease throughout 2--148 kpc. The resolved density behavior is
  therefore usually simple even though the unconstrained inward continuation
  is not physical.
- The median correction from \(M_{*,148}\) to the model's infinite total is
  0.00563 dex, and its 84th percentile is 0.0241 dex. However, the 99th
  percentile is 0.401 dex and the maximum is 0.771 dex. The model total is a
  benign extrapolation for most fits but must not be trusted blindly in the
  extreme tail.
- For the representative profile, increasing the Fourier quadrature order
  from 1,024 to 4,096 changes the transform by at most \(6.23\times10^{-4}\)
  over \(0\leq kR_{50}\leq30\). A finite-grid demonstration with 2 kpc FWHM
  Gaussian and Moffat PSFs shows the expected smoothing and outward shift of
  the measured half-mass radius. The small loss of same-grid flux is boundary
  cropping; convolution on an infinite plane conserves total flux.

The mathematical conclusion is deliberately narrower than the fitting
result: cubic-logit is a precise, identifiable interpolation of the measured
projected CoG, with exact radii and straightforward rendering. It is not a
replacement for Sérsic when a physical central continuation or spherical
deprojection is required.

## Python interface

```python
from hongshao.cubic_logit import (
    cubic_logit_aperture_cog,
    cubic_logit_enclosed_radius,
    cubic_logit_surface_density,
    render_cubic_logit_image,
)

shape = (12.0, 0.8, 0.4, 0.08)  # R50, m, b, c
radii = cubic_logit_enclosed_radius([0.2, 0.5, 0.8], *shape)
cog = cubic_logit_aperture_cog(
    radius,
    2.5e11,
    148.0,
    *shape,
)
density = cubic_logit_surface_density(radius, 2.6e11, *shape)
image = render_cubic_logit_image(x_grid, y_grid, 2.6e11, *shape)
```

The public module also provides the warp and its derivative, infinite-total
enclosed mass, density stationary points and global maximum, radial moments,
the numerical Fourier transform, and Gaussian and Moffat PSF transfer
functions. Optimizer coordinates used in Exp62 store \(\ln m\) and \(\ln c\);
they must be exponentiated before calling the public functions.

## References

- Graham & Driver (2005), [A Concise Reference to (Projected) Sérsic
  \(R^{1/n}\) Quantities](https://arxiv.org/abs/astro-ph/0503176), provides the
  Sérsic reference-guide pattern followed here: cumulative mass, characteristic
  radii, slopes, concentration, seeing, and deprojection.
- Trujillo et al. (2001), [The effects of seeing on Sérsic
  profiles](https://arxiv.org/abs/astro-ph/0109067), defines the normalized
  Moffat PSF convention used here and discusses the Gaussian limit and the
  treatment of ellipticity.
- NIST DLMF, [Bessel and Hankel
  integrals](https://dlmf.nist.gov/10.22#E43), gives the integral identities
  underlying the radial Fourier transforms.
