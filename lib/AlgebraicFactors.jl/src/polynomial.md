#implementation #derivation

> `PolynomialFactor` is a learnable named-channel polynomial residual with exact
> data Jacobian, Sampson scalarisation and single-branch Gaussian messages.
> Iteration uses frozen-metric Gauss–Newton; optional branches are in [[homotopy]].
> Game quantities remain absent.

> Sources: original implementation of [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §3(a,b), following [[L-ADR-0007 Point-dependent scalarisations are localised before use]]; code: `polynomial.jl`, `veronese.jl`, `../test/polynomial.jl` (residual invariance, degree-one messages, circle, rank, descent and graph oracles).
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category)

The channel dimensions are a NamedTuple; `keys(dims)` sets concatenation order,
just as for `LinearConstraintFactor` ([[constraint]]). Values may be scalars for
one-coordinate channels or real vectors of the declared length. Raw `energy`
merges disjoint NamedTuples in x and y, with nothing for either empty side.
Unlike that template, duplicate channels are rejected, so one value cannot
silently overwrite another. The auxiliary argument a is unused.

Core's `SampsonNorm` chooses one differentiated argument, `:x` or `:y`, and its
finite-difference reference expects an array. Giving a partial Jacobian for a
named split would change the data-noise model. Therefore the Sampson convention
is **all coordinates packed in y, x=nothing, wrt=:y**. Raw residuals accept the
named form too; their scalarisation asks the caller to pack first. The exact
Jacobian override rejects other argument conventions instead of ignoring `wrt`.
This allows comparison with the unmodified core finite-difference reference.
No LenticulumCore or Mycelium change is needed.

```julia
using AlgebraicFactors, LenticulumCore, LuxCore, Random, LinearAlgebra
f = PolynomialFactor((left=1,right=1), ExponentTable(2,2), 1; Σ=I, λ=0)
ps, st = LuxCore.setup(Xoshiro(7),f)
r, st = energy(f,(left=0.4,),nothing,(right=[0.8],),ps,st)
s, st = scalar_energy(f,nothing,nothing,[0.4,0.8],ps,st)
```

The parameter is `(Θ=rand(rng,Grassmann(k,m)),)`, state empty. Initial and fitted
representatives have orthonormal rows; energy also accepts full-rank row-gauge
transforms so that the invariance obligation is meaningful. Arbitrary deficient
parameters are not silently orthonormalised. Pointwise energy space is Euclidean
of dimension k and the energy is the signed vector Θv. Learnability is true;
`supported_polarities` enumerates one unobserved target, with every other
channel observed. Latent inputs are unsupported. `assemble` returns a
`BayesianLens(PolynomialModel(...), SolverInversion(...))`; likelihood and graph
messages share one routine and `invert` pools the site with the prior exactly
once ([[L-ADR-0006 Separate likelihood_message from invert]]). The default does not enumerate branches; [[homotopy]] describes the optional
root mixture. No IFT derivative rule or game/graph free energy is supplied.
The pointwise Sampson energy is not a normalised likelihood.

Writing D=Dv and J=ΘD, scalarisation uses

$$
\tfrac12 r^\top A^{-1}r,\qquad A=J\Sigma J^\top+\lambda\Theta\Theta^\top.
$$

Identity covariance is the default (Σ=I, or nothing); matrices must be finite,
symmetric PSD with the full data dimension. Core validates this and rejects A
unless its least eigenvalue exceeds k eps(T) times its greatest. λ must be finite
and nonnegative. Singular A throws; positive damping may rescue a rank-deficient
J when Θ has full row rank. There is no pseudoinverse.

Under Θ↦BΘ, r↦Br and A↦BABᵀ, so the energy is unchanged for invertible B.
This includes λ>0. Under a polynomial basis change, Θ_mon=Θ_cheb C, r and J
remain identical, so **undamped** Sampson energy is basis independent. But
Θ_mon Θ_monᵀ=Θ_cheb CCᵀΘ_chebᵀ generally differs from Θ_cheb Θ_chebᵀ.
Thus the required coefficient-identity damping is row-gauge compatible, **not
invariant under a nonorthogonal polynomial basis change**. Transporting the
coefficient metric would require an additional metric interface; it is not
silently inferred. Algebraic distance and nullspace fitting also depend on the
chosen coefficient metric/basis for noisy data.

Tests use seed 3002, 20 points, k=2,N=3,d=2, random row gauges with singular values
0.8 and 1.6, identity/anisotropic data covariance, and λ=0,0.3. Row-gauge and
undamped basis agreement use rtol=1e-10. Exact Jacobians match core's reference
with atol=2e-9,rtol=2e-10. A rank-zero J at the origin throws without damping;
λ=0.3 gives finite row-gauge invariant energy. A concrete basis-converted
one-equation example also verifies that the damped energies differ by more
than 1e-3. These are finite numerical checks,
not guarantees for arbitrary condition numbers.


## Single-branch Gaussian messages (§3b)

At the observed means and target expansion point x₀, compute the exact J=ΘDv,
split into observed blocks Jₒ and target Jᵤ. Observed inputs must be Dirac or
proper Gaussian beliefs; they are independent between channels. With their
block-diagonal covariance Cᵢₙ, use

$$
A = J\Sigma_f J^\top + J_o C_{\mathrm{in}}J_o^\top
    + \lambda\Theta\Theta^\top,\qquad
\Lambda = J_u^\top A^{-1}J_u,\qquad
h = \Lambda x_0-J_u^\top A^{-1}r_0.
$$

Σ_f covers **all data coordinates**, including the target, exactly as in the
complete-data Sampson fit. Observed belief uncertainty is additional to that
factor covariance. Nothing is silently excluded or renormalised. The site is
`GaussianBelief(h,Λ)`: it contains neither the cavity's precision nor its
information. Its coefficients can still depend on the cavity through x₀.

This **freezes the Sampson metric at each linearisation point**. It is an
approximation, rather than integration of a nonlinear likelihood or minimisation
of the full varying-metric Sampson energy. In particular it omits derivatives of
A and any log-determinant normalisation. The covariance is checked at each point
with the same k eps(T) λmax threshold as core, then solved by Cholesky. Singular A
throws with a reason; λ>0 can rescue it if Θ has full row rank. No pseudoinverse
or automatic jitter is used. Invertible row gauges preserve the site as well as
the pointwise energy; nonlinear tests include anisotropic Σ_f, Gaussian observed
uncertainty, damping and an undamped Chebyshev basis conversion.

The initial target x₀ is the proper Gaussian cavity mean or Dirac value. With
`nothing`, `TrivialBelief()` or a zero-information Gaussian, it is the **zero
vector**. Other improper Gaussian cavities have no mean and throw. This default
can land at a stationary point with no first-order information: the circle at
y₀=0 returns a flat y site even for |x|<1. Supply a cavity near the intended branch.
An observed missing/flat/sample/categorical belief throws rather than being
silently marginalised.

`iterations=1` returns the site at that initial point. Each further iteration
pools the previous site with the **original** cavity, takes the posterior mode,
and recomputes the site there. With a fixed A this is iterated posterior
linearisation / Gauss–Newton on

$$
\phi_A(u)=\tfrac12 r(z_o,u)^\top A^{-1}r(z_o,u)
 + \tfrac12(u-\mu_\pi)^\top\Lambda_\pi(u-\mu_\pi).
$$

It follows the iterated EKF convention described in [[nonlinear]]. Refreshing A between steps is a sequence of frozen
objectives, **not** one fixed objective with guaranteed descent. There is no
backtracking, LM safeguard, convergence certificate or global branch guarantee.
At finite iteration count the posterior mean need not equal the last expansion
point; the site always uses that last expansion point. Curvature is Gauss–Newton,
not the full Laplace Hessian away from zero residual, even at a stationary point.

Rank is computed from Jᵤ with LinearAlgebra.rank's default SVD threshold at message
time, never declared by `supports_polarity`. Since A is SPD the exact precision
rank equals rank Jᵤ:

| case | returned site |
|---|---|
| k=q, full rank | proper local Gaussian |
| k<q | improper, flat on the first-order fibre ker Jᵤ |
| k>q | weighted least-squares Gaussian, proper iff rank Jᵤ=q |
| any rank deficiency, including rank zero | retain the singular canonical site, without clipping or artificial precision |

A proper Gaussian cavity makes the posterior proper even when the site is not.
Without a unique posterior mode (rank-deficient Jᵤ and no proper cavity), iteration
stops and returns the current improper site. A Dirac cavity remains fixed.
Tiny nonzero singular values are not truncated: the numerical rank only decides
whether iteration without a cavity can proceed. As with `LinearConstraintFactor`,
roundoff may affect a floating-point precision's numerical rank.

## Message checks and measured results

All checks below are in `../test/polynomial.jl`. The degree-one oracle includes
the **actual** top-level `gaussian.jl` and `constraint.jl` in a private test module,
avoiding a dependency on Lenticulum or its unrelated packages. For v₁=[1;z],
Θ=[−c B₁ … Bₙ], map the linear constraint coefficients to Bᵢ and set
Q=JΣ_fJᵀ+λΘΘᵀ. The constraint's propagation of the observed Cᵢ produces the same
A as above. Seed 3004 exercises 144 cases: every target of dims=(1,2,1), k=1,2,3,
λ=0,0.2 and all Dirac/Gaussian input combinations. Maximum absolute information
and precision discrepancies are 2.9976021664879227e−15 and
1.5543122344752192e−15. Lens likelihood, pooled posterior, target-input exclusion
and affine prior-independence are checked separately. Nonlinear one-step sites
are invariant to cavity precision at the same cavity mean (rtol=2e−15).

For r=x²+y²−1, x=0.6, π=N(0.9,0.16), Σ_f=0.0025 I, λ=0 and 60 iterations,
the posterior mean is 0.8023817080100973. Independent Newton on the separately
written scalar objective **with A fixed at the final point** gives
0.8023817080100974; a 120001-point grid gives 0.80238 (grid tolerance 2.5e−6).
A=0.010038164053492012, site precision 256.547472991432, posterior variance
0.0038052116278629623. This is a fixed-point check of the frozen-metric algorithm,
not a claim about the full Sampson MAP. For Σ_f=δI, δ=1e−2,1e−4,1e−6,1e−8,
upper-root errors are respectively 0.00887746822, 9.75583614e−5,
9.76552701e−7 and 9.76562387e−9.

On the circle at x=√(1−y₀²), y₀=0.1,0.001,0.00001,0 and δ=0.0025,
the site precisions are 4, 0.0004, 4e−8 and **zero**. The full residual covariance
is 4δ, including at tangency: Jᵤ loses rank while the full J does not.
No NaN is returned. A one-equation two-dimensional target also checks rank-one
precision and the tangent null vector Jᵤv=0.

Seed 3005 checks 20 constant-metric circle objectives (Σ_f=0, λ=0.01) with
one versus thirty iterations. None increases the objective; the smallest
improvement is 6.314115064044333e−6. This limited result does not establish
descent under arbitrary initialisation or changing metrics.

The Mycelium graph has independent Gaussian priors a=N(0.7,0.03), b=N(−0.4,0.02)
and polynomial ab+c=0, Σ_f=0, λ=0.005, hence Q=0.01. Input edges absorb and
c emits, so no unavailable-input or branch choice is hidden in initialisation.
BP converges in two sweeps, giving c mean 0.28 and variance 0.0246. An independent
joint objective has its zero-residual MAP at (0.7,−0.4,0.28); its full Laplace
Hessian inverse has c variance 0.0246 (absolute tolerance 2e−12). This checks
that graph and objective, not arbitrary nonlinear BP or the exact non-Gaussian
product marginal (which would include Va Vb).

The full common `test/factor_conformance.jl` checker is **not applicable**: this factor
has neither graph-local nor game free energy, and its branch-dependent metric does not
define a normalised joint model. The checker's 2026-10-26 section opt-outs still require
finite graph-local energy; no placeholder is introduced here. The existing lens/site/posterior
subset remains checked on all degree-one polarities with the independent constraint oracle.

## Still absent

Exact mixture entropies, game/free-energy contributions (LP-009), IFT jvp/vjp rules and
missing-data EM. `PolynomialModel` is pure for this local branch approximation;
the optional mixture mode still exposes no joint latent branch-index API.
