#implementation #derivation

> The forward-only `PluckerFactor` transports points and paired particles through
> $p=z\wedge w$, and matches the exact first two moments of independent Gaussian
> inputs. Gaussian output is a moment approximation to a generally non-Gaussian law.
> The coordinate map, Jacobian and sparse quadrics implement L-SPEC-004 §5.

> Sources: original derivation and implementation of [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §5 and the algebra in [[Grassmann Mixing and Plücker Coordinates]]; code: `plucker.jl`, `../test/runtests.jl`; the deterministic-factor interface follows `lib/Adversarial.jl/src/generator.jl`.
>
> Theory (CT-ML wiki): [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)

## Coordinates, derivatives and equations

`plucker_pairs(r)` lists $(1,2),(1,3),\ldots,(1,r),(2,3),\ldots,(r-1,r)$.
`plucker(z,w)` evaluates $p_{ij}=z_iw_j-z_jw_i$ without converting the arithmetic
of its arguments. The low-level map supports integer, rational and complex inputs;
the factor's beliefs are real vector-valued. Inputs use one-based indexing.

The Jacobian has columns in `vcat(z,w)` order, with entries

$$
\frac{\partial p_{ij}}{\partial z_a}=\delta_{ia}w_j-\delta_{ja}w_i,
\qquad
\frac{\partial p_{ij}}{\partial w_a}=z_i\delta_{ja}-z_j\delta_{ia}.
$$

`plucker_relations(r)` returns one record per increasing quadruple $(i,j,k,l)$,
in lexicographic order, including `indices`, three `monomials` (pairs of integer
indices into the coordinate vector) and `coefficients=(1,-1,1)`. It represents
$p_{ij}p_{kl}-p_{ik}p_{jl}+p_{il}p_{jk}$. There are $\binom r4$ such equations;
`plucker_residuals(r,p)` evaluates them. For $r=4$ the monomials are
`((1,6),(2,5),(3,4))`. This table is usable as polynomial test data; it does not
implement §3's monomial features or `fit_nullspace`.

## Exact moments for independent inputs

Write $z=\mu+\epsilon$, $w=\nu+\delta$, with zero-mean independent fluctuations
of covariance $S,T$. Then

$$
p=\mu\wedge\nu+\epsilon\wedge\nu+\mu\wedge\delta+\epsilon\wedge\delta.
$$

The last three terms have zero mean and are pairwise uncorrelated: each cross
expectation contains a single centred fluctuation from one of the independent
inputs. Thus $\mathbb E p=\mu\wedge\nu$. With $J_z,J_w$ the two Jacobian blocks
at $(\mu,\nu)$, the covariance is

$$
C=J_z S J_z^\top+J_w T J_w^\top+Q,
$$

$$
Q_{(ij),(kl)}=S_{ik}T_{jl}-S_{il}T_{jk}-S_{jk}T_{il}+S_{jl}T_{ik}.
$$

The last expression follows by multiplying the two terms of
$\epsilon_i\delta_j-\epsilon_j\delta_i$ and using independence. No within-input
fourth moments are needed: finite second moments and independence suffice.
`plucker_moments(μ,S,ν,T)` implements this formula and returns `(mean,cov)` without
regularisation. A zero input covariance represents a Dirac. Correlated Gaussian
inputs require cross-covariance terms and are outside this interface.

The test oracle uses a different calculation: it expands all four signed terms
of $\mathbb E[p_{ij}p_{kl}]$ using
$\mathbb E[z_i z_k]=S_{ik}+\mu_i\mu_k$ and
$\mathbb E[w_j w_l]=T_{jl}+\nu_j\nu_l$, then subtracts the product of the means.
It shares neither the Jacobian nor the centred-covariance expression with the implementation.

## Beliefs, covariance rank and integration

`pushforward(f,(zbelief,wbelief),ps,st)` returns a belief and unchanged state.
`factor_message` reads the two named inputs and excludes the output's prior and
incoming message. There is exactly one polarity: observed inputs, unobserved output.
No inverse is supplied: for a nonzero decomposable bivector the pairs form an
$SL_2$-orbit; zero has a larger, degenerate fibre. The residual $p-z\wedge w$
exists, but does not select an inverse belief.

- Two Diracs give a Dirac.
- Two sample beliefs describe a **joint empirical belief**: particles are paired by
  index, with equal positive counts and matching normalised weights (`rtol=1e-12`,
  `atol=0`). This is not the independent product of the clouds. The first cloud's
  raw weights survive unchanged; uniformly weighted and unweighted clouds can match.
  Matrix samples are columns; vector samples are vectors of vectors. A Dirac may
  accompany a cloud on either side, preserving that cloud's weights. Empty clouds,
  zero total weights, mismatched counts/weights and sample/Gaussian pairs throw.
- Separate proper Gaussian beliefs are independent. The emitted `GaussianBelief`
  matches the exact moments when the covariance is representable, not the law.
  Dirac/Gaussian is linear and its exact law is Gaussian, possibly degenerate.

A nonlinear algebraic constraint does **not** imply singular covariance:
independent centred isotropic inputs give $C=2I$ in all bivector coordinates,
although their outputs satisfy the quadrics. In contrast, a fixed nonzero $z$
with a proper Gaussian $w$ gives rank $r-1$, since the linear map $w\mapsto z\wedge w$
has kernel $\operatorname{span}(z)$. Zero $z$ gives zero covariance.

LenticulumCore's `Gaussian(μ,Σ)` requires positive-definite covariance. Its support
for singular **precision** describes an improper likelihood, not a distribution
confined to an affine subspace; using a pseudoinverse would lose the support constraint.
Consequently the default factor rejects output covariance with
$\lambda_{\min}\le n\,\operatorname{eps}(T)\lambda_{\max}$, where $n=\binom r2$ and
$T$ is its floating-point element type. There is no automatic jitter.
For these cases `plucker_moments` supplies exact singular moments, or the caller
can explicitly set `PluckerFactor(r; covariance_jitter=δ)` with finite $\delta>0$.
It adds absolute variance $\delta I$ to the output covariance; that covariance is
then an explicit approximation. The same rank threshold applies after adding it.
This keeps LenticulumCore and Mycelium unchanged. Extending the core belief
representation with affine support would be a separate change.

Parameters and state are empty NamedTuples; the factor is not learnable. Like
`GeneratorFactor`, scalar pointwise energy and graph-local free energy return zero.
These are forward-map accounting conventions, not a density or an energy penalising
inconsistent triples. Game-level `free_energy`/`entropy` and an assembled lens are
not implemented; [[LP-009 Game-level free energy is declared but not implemented]]
remains open. The common conformance checker requires those game quantities even
on its direct-message path, so it does not apply in full. Tests check the available
interface, prior exclusion and actual Mycelium messages without claiming full conformance.
No game-level gap has been closed, so LP-009's list is unchanged.

## Checks and limits

`lib/AlgebraicFactors.jl/test/runtests.jl` checks small hand cases, antisymmetry,
$GL_2$ determinant scaling, Lagrange's identity, exterior-power equivariance with
independently computed minors (including rectangular maps), and central Jacobian
differences (`h=1e-5`, absolute tolerance `2e-10`). It checks the relation count,
quadruple order, Klein signs, vanishing for seeded random decomposable points at
$r=4,5,6$ (`3e-14` absolute), and violations at random and explicit nondecomposable points.

Moment tests compare against the independent second-moment expansion at $r=2,3,4,5$,
including zero covariances on either side. A Monte Carlo test uses **200000** independent
Gaussian pairs at $r=4$, `Xoshiro(5104)`, requiring
$\|\hat\mu-\mu_p\|/\sqrt{\operatorname{tr}C}<0.015$ and
$\|\hat C-C\|_F/\|C\|_F<0.025$. The checked run reports mean error `0.0030714051` and covariance error `0.0048753840`.
Singular rejection, rank $r-1$, explicitly regularised covariance, proper-input
requirements, and a Mycelium DAG with two Gaussian priors are checked as well.

The `Klein quadric data rank (not fit_nullspace)` test builds its own 28 degree-at-most-two
monomials on **400** unnormalised Gaussian-pair points (`Xoshiro(5106)`). At relative
singular-value threshold `1e-12` it checks exactly one null direction and alignment
with the known Klein coefficients (absolute dot product within `2e-14` of one).
The checked run reports smallest relative singular value `4.4408921e-16`, next
`0.0848452043`, and their ratio `1.9105442e14`. The test logs these values.
This is a test of the data, not §3's unimplemented fitting API. Unit-normalised
data are not tested here.

The main implementation difficulty is the distinction between singular precision
and singular covariance. Explicit regularisation and separately exposed exact moments
make that limitation visible. The other is joint particles versus independent Gaussian
marginals: matching particle indices preserves supplied dependence, while the Gaussian
path assumes none. Neither convention reconstructs missing cross-channel correlations.

The implementation uses CPU arrays. The mean/map cost is $O(r^2)$; dense covariance
storage and computation are $O(r^4)$. No inverse fibre beliefs, general polynomial
factor, Sampson scalarisation, nullspace fitting, GPU or AD support is claimed.
