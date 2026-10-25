#implementation #derivation

> `PolynomialFactor` is a learnable named-channel polynomial residual with exact
> data Jacobian and Sampson scalarisation. It supplies no messages or inference.

> Sources: original implementation of [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §3(a), following [[L-ADR-0007 Point-dependent scalarisations are localised before use]]; code: `polynomial.jl`, `veronese.jl`, `../test/polynomial.jl` (PolynomialFactor and Sampson invariance).
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
`supported_polarities` inherits the empty tuple. No root solver, messages,
assembled lens, game entropy or graph/free-energy implementation is promised.
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
