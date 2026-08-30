# The deposition model as an analytic integral along the halo's growth curve

*Written 2026-08-28, before exp63; corrected the same day. Every number here
comes from `scripts/04_analytic_check.py` (a throwaway probe: 62 galaxies for
the sum-versus-integral check, 73 for the figure, the incumbent's parameters
frozen). Nothing was fitted. **Correction (2026-08-28, later the same day):**
the first version of this note read the measured curves of growth on a
`geomspace(2, 148, 24)` radius grid instead of the measured grid
$(2^{1/4}+0.1k)^4$; every statement about the inner slope was affected and is
now stated on the true grid — the data rise as $R^{1.00}$ over 2–4.9 kpc, not
$R^{1.45}$. The integral, the closed forms and the discretisation numbers were
never affected (they use the model's own grid consistently). exp63 Stage 0
re-measures everything on `fit.R_GRID`.*

**What this note answers.** The deposition-only model of exp54 (`gompertz_log-E2-S2`,
the "incumbent") builds a galaxy by walking the halo's mass history in 72 merger-tree
steps, depositing stars at each step, and adding up. The user's question was whether
that "follow the increments, then sum" recipe can be written as one analytic
integral — and whether doing so buys anything. The answers: **yes, exactly**, because
every ingredient the sum uses is already a closed-form function of cosmic time; the
sum is a first-order Riemann approximation of that integral and carries a **radius-dependent
numerical bias of −0.027 dex at 2 kpc** (a *level* error the fit absorbed into its
parameters — exp63 Stage 0 measured that the incumbent's central *shape* defect is
unchanged on the exact integral; see the README of `experiments/exp63_analytic_growth`);
and the integral form reveals that the model's predicted profile is nothing more than
the **distribution of stellar mass over deposit size**, smoothed by the deposit
kernel — which is why the class's inner slope is a property of its size law's small
end alone, and which gives a way to read the size law off the data instead of
assuming it.

---

## 1. The sum, and every piece of it as a function of time

The incumbent (exp54 `model.forward`) evaluates, at observation time $t$,

$$
M_*(<R,\,t)=\sum_{j:\,t_j\le t}\varepsilon(M_j,z_j)\,\Delta M_j\;
F\!\left(\frac{R}{s_j};\,\rho_j\right),
\tag{1}
$$

one term per merger-tree step $j$: $\Delta M_j$ is the halo mass gained in the step,
$\varepsilon$ the surviving stellar fraction of it
($\log_{10}\varepsilon=a_0+a_M m+a_z x+a_{Mz}\,m\,x$, with $m=\log_{10}M_j-13.5$
and $x=\ln(1+z_j)$), $F$ the unit-mass cumulative profile of one deposit
(the `gompertz_log` family, $F(y)=\exp(-\ln 2\; y^{-c})$, with $y=R/s$), $s_j$ the
deposit's half-mass radius from the size law $s=10^{f_0}(1+z)^{b}R_{200c}$, and
$\rho_j=s_j/(3R_{200c,j})$ the truncation ratio.

The merger-tree steps are **not** the raw TNG history. `exp29/run.py::dipfree_mah`
evaluates the official DiffMAH fit curve (`diffmah_log_mah_fit`) on the 100-snapshot
grid and differences it; so $M_j$ and $\Delta M_j$ come from the four DiffMAH numbers
of each halo — $(\log m_p,\ \log t_c,\ \alpha_{\rm early},\ \alpha_{\rm late})$, with
the transition speed fixed at $k=3.5$:

$$
\log_{10}M(\tau)=\log m_p+\alpha(\tau)\,(\tau-\tau_0),\qquad
\alpha(\tau)=\alpha_e+(\alpha_l-\alpha_e)\,\sigma\!\big(k(\tau-\tau_c)\big),\qquad
\tau=\log_{10}t .
$$

The check: the analytic curve reproduces the records' $\log M_j$ at every step to
**$3\times10^{-6}$ dex** (official parameters anchored at $t_0=13.80$ Gyr; the
$z=0.4$-anchored columns of `diffmah_combined.fits` do *not* — they deviate by up to
0.45 dex at $z>8$, a bookkeeping inconsistency filed in `doc/todo.md`).

The other ingredients are closed-form too:

- **redshift of time**, flat $\Lambda$CDM, exact:
  $a(t)=(\Omega_m/\Omega_\Lambda)^{1/3}\sinh^{2/3}\!\big(\tfrac32 H_0\sqrt{\Omega_\Lambda}\,t\big)$
  (agrees with astropy to $4\times10^{-6}$ in $z$);
- **halo radius**, $R_{200c}=\big[3M/(4\pi\cdot200\,\rho_c(z))\big]^{1/3}$ with
  $\rho_c(z)=\rho_{c,0}\,[\Omega_m(1+z)^3+\Omega_\Lambda]$. The incumbent reads
  $R_{200c}$ from the TNG group catalog instead; replacing it by this formula changes
  the profiles by 0.04 dex (median of the per-galaxy maximum over radius) — the
  catalog's $M_{200c}$ is not the DiffMAH peak mass. This is a modelling *choice*, not
  an error, and it is listed as a decision in the exp63 plan;
- **the growth rate**, $\mathrm{d}M/\mathrm{d}\ln t = M\,[\alpha(\tau)+\alpha'(\tau)(\tau-\tau_0)]$
  with $\alpha'=(\alpha_l-\alpha_e)\,k\,\sigma(1-\sigma)$.

So Eq. (1) is the Riemann sum of

$$
\boxed{\;M_*(<R,\,t)=\int_{t_{\min}}^{t}\varepsilon\big(M(t'),z(t')\big)\,
\frac{\mathrm{d}M}{\mathrm{d}t'}\;
F\!\left(\frac{R}{s(t')};\,\rho(t')\right)\mathrm{d}t'\;}
\tag{2}
$$

a smooth one-dimensional integral in which **every symbol is an explicit function of
$t'$ given four DiffMAH numbers, the cosmology, and the seven model parameters**.
No catalog, no snapshot grid.

### 1.1 The sum converges to the integral, at first order, and is biased at the centre

Re-doing the sum with each snapshot step split into 2, 4 and 8 sub-steps (the
analytic curve evaluated at the sub-times, the catalog $R_{200c}$ interpolated in
$\log t$) moves it toward Eq. (2) as $1/n$: the median per-galaxy maximum deviation is
0.0128, 0.0064, 0.0032 dex. Gauss–Legendre quadrature in $\ln t$ (12 panels × 24
nodes) is the limit. The 72-step sum itself sits **0.026 dex (median), up to 0.083
dex** from that limit — and the error is not uniform in radius:

| radius (grid points of the measured curve) | 2.0 kpc | 4.9 kpc | 10.2 kpc | 27.5 kpc | 52.3 kpc | 117 kpc | 148 kpc |
|---|---|---|---|---|---|---|---|
| median $\log_{10}$(72-step sum / exact integral), $z=0.4$ | −0.027 | −0.020 | −0.015 | −0.010 | −0.009 | −0.008 | −0.007 |

(figure `figures/04_analytic_check.png`, panel a). The sign and shape are what the
step convention dictates: each step's mass is deposited at the size the halo has at
the *end* of the step, so the earliest, coarsest steps in $\ln t$ push stars outward.
Every central bias the programme has reported for the incumbent — "7 to 13 per cent
low inside a few kpc at every epoch scope" (exp61) — contains about −6 per cent of
this numerical origin *as a level*. exp63 Stage 0 re-measured the frozen incumbent on
Eq. (2): its $z=0.4$ centre goes from −2.4 to +3.6 per cent inside 10 kpc, but the
amplitude-pinned dip at 3–5 kpc and excess at 10–30 kpc are unchanged — the
parameters had absorbed the level; the shape is the model's.

Two smaller bookkeeping effects surfaced by the same check: where the group catalog
has no $R_{200c}$ for a snapshot in the *middle* of a history, `forward` drops that
step's stars entirely (12 of 124 galaxy-epochs in the probe; up to 5.5 per cent of a
galaxy's deposited stellar mass); and the first one to five steps ($z\gtrsim10$–15)
are dropped in every galaxy for the same reason. Eq. (2) has neither problem.

## 2. The convolution form: the profile is the deposit-size distribution

Change the integration variable from time to the logarithm of the deposit size,
$v=\ln s(t')$ (monotone: $s$ grows with time whenever $b<1$, which every fit satisfied).
Then

$$
M_*(<R)=\int W(v)\;K(\ln R-v;\,\rho)\,\mathrm{d}v,\qquad
W(v)\equiv\varepsilon\,\frac{\mathrm{d}M}{\mathrm{d}v},\qquad K(u)\equiv F(e^{u}) .
\tag{3}
$$

$W(v)\,\mathrm{d}v$ is the stellar mass deposited with half-mass radius between
$s$ and $s\,e^{\mathrm{d}v}$: **the distribution of the galaxy's stars over deposit
size.** $K$ is a sigmoid in $\ln R$ rising from 0 (deposit much larger than $R$) to 1
(deposit entirely inside $R$). Eq. (3) is a convolution in log radius, so the
curve of growth is the *cumulative* of $W$ smoothed by $K$, and the mass in a shell is
$W$ itself smoothed by $\mathrm{d}K/\mathrm{d}u$:

$$
\frac{\mathrm{d}M_*}{\mathrm{d}\ln R}=\int W(v)\,K'(\ln R-v)\,\mathrm{d}v .
$$

In words: *the deposition-only class predicts that the stellar mass in a shell at
radius $R$ is the stellar mass that was deposited when the halo's deposit-size scale
was $R$.* The galaxy is a frozen record of its halo's size history. Everything the
class can and cannot do follows from that sentence:

- it is **monotone in time** at every radius because $W\ge0$ (the exp54 Stage 3.8
  theorem, in one line);
- it has **one radial coordinate**, $v(t)$, so the centre and the outskirts are the
  same law read at two times (the exp54 Stage 3.7 finding that one scale cannot serve
  both ends);
- its **inner slope** is the slope of $W$ below $R$ when the kernel is cored (the
  `gompertz_log` kernel vanishes faster than any power of $R$ at small $R$, so the
  kernel contributes no central mass of its own). On the measured grid the data rise
  as $R^{1.00}$ between 2 and 4.9 kpc (median; 16–84 per cent range 0.76–1.14); the
  incumbent rises as $R^{0.88}$, with a distribution less than half as wide
  (0.77–0.94) (panel c). The median gap is modest; the missing *diversity* of
  inner slopes is the larger discrepancy. Under a cored kernel the inner slope is
  set by how $W$ rises below 5 kpc, i.e. by how much stellar mass was deposited at
  sizes of 1–3 kpc — which the size law $s=0.14\,(1+z)^{-0.9}R_{200c}$ reaches only
  at $z\gtrsim4$, when a few per cent of the halo exists, and reaches identically
  for every halo of the same mass. A cuspy kernel (a Sérsic deposit with $n\ge2$
  rises as $R^{\lesssim1.2}$ inside its half-mass radius) is the other route;
  the two are distinguishable only with the earlier epochs, because a compact
  early deposit is already present at $z=2$ while a cuspy kernel puts central mass
  into every deposit.

Under the Mellin transform ($\tilde f(\sigma)=\int_0^\infty y^{\sigma-1}f(y)\,\mathrm{d}y$)
the convolution is a product, and the kernels' transforms are elementary (all three
checked numerically to eight digits):

| deposit family | $F(y)$, $y=R/s$ | $\int_0^\infty y^{-\beta-1}F(y)\,\mathrm{d}y$ |
|---|---|---|
| `gompertz_log` | $\exp(-\ln2\,y^{-c})$ | $(\ln 2)^{-\beta/c}\,\Gamma(\beta/c)/c$ |
| Moffat | $1-(1+y^2)^{1-\gamma}$ | $\dfrac{\gamma-1}{\beta}\,B\!\left(1-\tfrac{\beta}{2},\,\gamma-1+\tfrac{\beta}{2}\right)$ |
| Sérsic | $P(2n,\,y^{1/n})$ | $\Gamma(2n-n\beta)\big/\big(\beta\,\Gamma(2n)\big)$ |

## 3. The closed form: a power-law halo history deposits a power-law galaxy

At $z\gtrsim1$ the universe is close to Einstein–de Sitter ($1+z\propto t^{-2/3}$,
$\rho_c\propto(1+z)^3$) and the DiffMAH index is nearly constant. Then every factor
in $W$ is a power of $t$, and $W(v)=A\,e^{\beta v}$ with

$$
\boxed{\;\beta=\frac{3(1+a_M)\,\alpha-2\,a_z\ln10}{\alpha+2(1-b)}\;}
\tag{4}
$$

($a_{Mz}=0$; with the cross term, $\ln W$ acquires a $-\tfrac23\alpha\,a_{Mz}\,\tau^2$
piece and $W$ becomes a **log-normal bump** in deposit size — the "efficiency peak"
that exp54 Stage 3.2 reported not finding is the cross term). For the `gompertz_log`
kernel the integral in Eq. (3) is then exactly

$$
M_*(<R)=\frac{A\,R^{\beta}}{c\,(\ln2)^{\beta/c}}
\left[\Gamma\!\left(\tfrac{\beta}{c},\,\ln2\,(s_{\min}/R)^{c}\right)
-\Gamma\!\left(\tfrac{\beta}{c},\,\ln2\,(s_{\max}/R)^{c}\right)\right],
\tag{5}
$$

with $\Gamma(q,x)$ the upper incomplete gamma function and $s_{\min},s_{\max}$ the
deposit sizes at the first and last time. Panel (b) of the figure overlays Eq. (5) on
brute-force quadrature for four MAH indices: they agree to $10^{-10}$ dex, including
$\beta<0$. For $s_{\min}\ll R\ll s_{\max}$ and $\beta>0$ the bracket is $\Gamma(\beta/c)$
and the galaxy is a pure power law $R^{\beta}$; for $R\gg s_{\max}$ it converges to the
total deposited mass $A\,s_{\max}^{\beta}/\beta$.

**Is this the Sérsic profile?** No — but it is the Sérsic *integral*. The Sérsic
curve of growth is $M(<R)/M_{\rm tot}=P\big(2n,\,b_n(R/R_e)^{1/n}\big)$: the cumulative
of a Gamma$(2n)$ density in the variable $u=b_n(R/R_e)^{1/n}$, integrated over
*radius* up to $R$. Eq. (5) is the cumulative of a Gamma$(\beta/c)$ density in
$X=\ln2\,(s/R)^{c}$, integrated over *deposit size* up to $s_{\max}$ at fixed $R$;
the radius enters only through the size ratio $(s/R)^{c}$. Both come from the same
one-dimensional integral "power × stretched exponential" — the Sérsic profile's
$\int x\,e^{-b x^{1/n}}dx$ and the gompertz kernel's $\int s^{\beta-1}e^{-\ln2 (s/R)^{c}}ds$
— which is why the incomplete gamma function appears in both, with $n\leftrightarrow1/c$
and $2n\leftrightarrow\beta/c$. As functions of radius they differ at both ends
(`figures/04b_sersic_vs_deposition.png`): a Sérsic curve rises as $R^{2}$ at the
centre (finite central density) and reaches its total through a stretched
exponential; the deposition curve rises as $R^{\beta}$ — a power law of any slope,
a cusp when $\beta<2$ — and reaches its total through the kernel's power-law tail,
$1-M/M_{\rm tot}\propto(s_{\max}/R)^{c}$. Over the measured decade and a half the
$\beta\approx1$ deposition curve does track a de Vaucouleurs ($n=4$) profile
closely, which is why de Vaucouleurs-like galaxies come out of this class; the
resemblance holds on a finite range, not asymptotically, and it is inherited from
the kernel's stretched-exponential form (a modelling choice), not derived. The
statement that *is* a prediction of the class is Eq. (4): the inner slope of the
curve of growth is set by the halo's growth index and the efficiency law.

The full DiffMAH curve rolls from $\alpha_e$ to $\alpha_l$ at $t_c$, so the incumbent's
prediction is, structurally, a **rolling power law in radius**: inner slope
$\beta(\alpha_e)$, outer slope $\beta(\alpha_l)$, transition near $s(t_c)$, cut-off at
$s(t_{\rm obs})$ — the very parametrisation ("radial DiffMAH", $\beta_{\rm in}$,
$\beta_{\rm out}$, $R_c$) proposed for the curve of growth in
`doc/ultimate_shmr_possible_directions.md` §4, now with the map from halo parameters to
profile parameters written down. With the incumbent's efficiency numbers Eq. (4) gives
$\beta=-0.39,\,-0.21,\,+0.05,\,+0.24$ for $\alpha=0.5,1,2,3$: in the pure power-law
regime the class, as fitted, has a much shallower slope than the data's 1.00; what
lifts the incumbent to 0.88 over 2–4.9 kpc is the cut-off of the deposit-size
distribution at its small end ($s_{\min}$), not the power law.

## 4. What the integral form buys, and what it does not

**Numerics.** Eq. (2) is exact, needs ~300 kernel evaluations per galaxy instead of 72
matrix rows, is smooth in every parameter (a gradient-based or automatically
differentiated fit becomes possible; the incumbent's Nelder–Mead fits took 7–10
minutes per start), and removes the three bookkeeping artefacts of §1.1.

**Structure.** Eq. (3) says the model has exactly one internal object, $W(v)$. That
gives an *inversion*: with a kernel chosen, a measured curve of growth can be
deconvolved (non-negative, few-parameter) into the deposit-size distribution the data
require, and the cumulative of that distribution can be matched, quantile by quantile,
to the cumulative stellar mass the efficiency law deposits in time — producing the
**size-versus-time law each galaxy demands**, $s^\ast(t)=C_W^{-1}\big(C_t(t)\big)$,
instead of a size law assumed in advance. Whether $s^\ast(t)/R_{200c}(t)$ is one
universal function (the S2 hypothesis) or two (in-situ and ex-situ scales) is then
a measurement made *before* any population fit, which is the standing method.

**Statistics.** Because $M_*(<R)$ is linear in $W$, any random model of $W$ has
moments that are integrals of the same kind: multiplicative noise in $\varepsilon$
along the history gives a covariance between radii as a double integral of Eq. (3);
lumpy ex-situ accretion (a marked point process along the history) gives the variance
by Campbell's theorem. A stochastic layer can therefore live *inside* the process
rather than on top of its output, with its cross-epoch coherence following from the
history instead of being imposed.

**What it does not buy.** A single epoch determines $W$ only up to the kernel's
smoothing and says nothing about *when* each part of $W$ was laid down: the pair
$(\varepsilon(t),s(t))$ is degenerate at one epoch. The earlier epochs break it — the
$z=2$ curve of growth is Eq. (2) truncated at $t(z=2)$, so it fixes how much of $W$
was in place by then. And the class remains monotone: declining centres are outside
it whatever the integral form (exp54 C8), which is the layer's and the data's problem,
not the mean model's.

## 5. Findings to carry, in one line each

1. The incumbent's 72-step sum is a first-order Riemann approximation of Eq. (2) and
   is biased −0.027 dex at 2 kpc → −0.007 dex at 148 kpc (2073 galaxies in exp63
   Stage 0, all epochs). A level, absorbed by the fit; not the shape defect.
2. The DiffMAH-driven deposition model is a convolution in log radius of the
   deposit-size distribution with the deposit kernel; its curve of growth is the
   smoothed cumulative of that distribution.
3. For a power-law history in Einstein–de Sitter the galaxy is a power law in radius
   with the slope of Eq. (4); for the `gompertz_log` kernel the profile is the
   incomplete-gamma expression (5), verified to $10^{-10}$ dex.
4. On the measured grid the data's central curve of growth rises as $R^{1.00}$ over
   2–4.9 kpc (16–84 per cent: 0.76–1.14); the incumbent's as $R^{0.88}$ (0.77–0.94):
   a modest median gap and a distribution less than half as wide.
5. The $z=0.4$-anchored `diffmah_*` columns of `diffmah_combined.fits` disagree with
   the official DiffMAH curve at $z>8$ by up to 0.45 dex; use the official
   $z=0$-anchored parameters.
