#implementation

> Opt-in full and diagonal noise covariances for Gaussian factors, with precision
> parameters, normalised noise-prior densities and a covariance eigenvalue floor.

> Sources: original derivations below; code: `noise.jl`, `gaussian.jl`,
> `test/learnable_noise.jl`, `ext/LenticulumForwardDiffExt.jl`,
> `ext/LenticulumChainRulesCoreExt.jl`.
>
> Theory (CT-ML wiki): [Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Free-Energy).

## Parameters and objectives

Full precision is $P=Q^{-1}=LL^\top$, with the diagonal of $L$ represented by its logs
and its strict lower triangle packed column by column. Thus $\log\det Q=-2\sum_i\log L_{ii}$.
Diagonal mode stores log-precisions, giving $\log\det Q=-\sum_i\log P_{ii}$.
Initial parameters come from the user's SPD covariance. `ps.noise` is nested so LP-002
flattening, AD and Mycelium's optimiser rules can use it without additional APIs.

For independent residuals with known affine mean, let $T=\sum_{k=1}^N r_k r_k^\top$.
The negative log likelihood, up to the constant $Nd\log(2\pi)/2$, is

$$
F(P)=\frac12\operatorname{tr}(PT)-\frac N2\log\det P.
$$

Its stationary point satisfies $T-NP^{-1}=0$, hence $\widehat Q=T/N$; diagonal
learning gives $\widehat Q_{ii}=T_{ii}/N$. These are residual second moments with
denominator $N$, not the unbiased covariance estimator. `test/learnable_noise.jl`
uses residuals balanced against the input and intercept, so their mean gradients
cancel while the known $A,b$ remain unchanged under the batch update.

Without the log determinant, rescaling $P\mapsto cP$ sends fit to zero as $c\to0$.
Gradient descent on pointwise fit therefore inflates variance. At $T=0$, the normalised
likelihood instead sends precision to infinity. The prior or floor guards that case.

## Noise priors and their density coordinates

`WishartPrecisionPrior(ν, V)` means density with respect to Lebesgue measure on SPD
precision matrices, with mean $\nu V$. Its negative log density is

$$
-\log p(P)=\frac12\operatorname{tr}(V^{-1}P)
-\frac{\nu-d-1}{2}\log\det P
+\frac{\nu d}{2}\log2+\frac\nu2\log\det V+\log\Gamma_d(\nu/2).
$$

Combining with $N$ samples gives

$$
P_{\mathrm{MAP}}=(N+\nu-d-1)(T+V^{-1})^{-1},
$$

when the numerator is positive. In particular an exact fit with one observation has
$P_{\mathrm{MAP}}=(\nu-d)V$. A proper Wishart can still have a boundary mode toward
zero precision when $N+\nu-d-1\leq0$; a finite interior MAP requires that coefficient
to be positive. Its exponential tail suppresses precision tending to infinity.

For diagonal noise, `LogNormalVariancePrior(μ, s)` specifies independent densities on
variances $v_i=Q_{ii}$, with $\log v_i\sim\mathcal N(\mu_i,s_i^2)$:

$$
-\log p(v)=\sum_i\left[\log v_i+\frac{(\log v_i-\mu_i)^2}{2s_i^2}
+\log s_i+\frac12\log(2\pi)\right].
$$

At an exact fit with $N$ samples, $v_{i,\mathrm{MAP}}=\exp[\mu_i-(1+N/2)s_i^2]$.
No Jacobian for log-Cholesky coordinates is added: optimisation finds MAP in the
declared density coordinates, not the mode of a transformed parameter density.

Each local free energy charges one noise-prior density. If callers share one noise
parameter across independent observations, they must arrange to count that prior
once per dataset (or weight its contribution accordingly). Repeating a regularised
local energy $N$ times also repeats the prior $N$ times. The MLE batch test has no prior;
the MAP tests use one observation and count the prior once.

## Floor and first derivatives

With no noise prior, `eigenvalue_floor=ε` caps precision eigenvalues at $1/\varepsilon$.
This is an exact spectral floor, not a diagonal-entry bound. Diagonal mode clips
log-precisions directly. A supplied prior disables flooring, including for initial
covariances below the configured floor.

Full mode reconstructs the capped precision using a symmetric eigendecomposition.
For $g(\lambda)=\min(\lambda,1/\varepsilon)$, its derivative in an eigenbasis multiplies
entry $(i,j)$ by the divided difference

$$
K_{ij}=\begin{cases}
\dfrac{g(\lambda_i)-g(\lambda_j)}{\lambda_i-\lambda_j},&\lambda_i\ne\lambda_j,\\
g'(\lambda_i),&\lambda_i=\lambda_j.
\end{cases}
$$

Repeated eigenvalues within either smooth branch therefore cause no division by zero.
The optional ForwardDiff extension applies this Frechet derivative to each dual
direction; the ChainRulesCore extension supplies its adjoint for Zygote. At the kink
the full spectral rule chooses the saturated-side derivative zero. Derivatives at the
kink are not unique. A hard floor can trap a saturated eigenvalue; choose a prior if
smooth recovery from collapse matters. Clipped precision need not have the original
Cholesky entries: these represent the raw precision before the guard. Normalisation
uses the guarded precision, including its changed determinant.

## Limits and verification

`test/learnable_noise.jl` compares normalised densities with Distributions.jl, parameter
gradients with central finite differences, full/diagonal MLE and both exact-fit MAPs
with the formulas above. It exercises ForwardDiff and Zygote, `parameter_step`, and
spectral derivatives with rotated and repeated eigenspaces. Message and belief
calculations use the current guarded covariance. The prior mean remains fixed.
Device execution, higher-order AD and AD backends beyond these two are not established.
Finite trainable entries and numerically representable SPD matrices are assumed;
arbitrarily large optimiser steps can still overflow exponentials.
