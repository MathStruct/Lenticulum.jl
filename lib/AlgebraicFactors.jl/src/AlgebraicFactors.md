#implementation

> `AlgebraicFactors` starts the algebraic family with exact finite toric potentials.
> It supplies discrete messages, energies and entropies, moment fitting and model
> checks, Grassmannian parameter geometry and retracted updates, and forward Plücker
> transport. General polynomial factors and later families remain unimplemented.

> Sources: original implementation of [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §0, §1.1–§1.7, §2.1–§2.2 and §5; code: `AlgebraicFactors.jl`, `toric.jl`, `fit.jl`, `grassmann.jl`, `plucker.jl`, `../test/runtests.jl`.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)

The module depends on `LenticulumCore`, `Mycelium`, `LinearAlgebra`, `LuxCore` and
`Random`. The last two implement the repository's parameter/state interface; they add
no inference algorithm. Newton and binary-feature IPF use only LinearAlgebra; there
is no external optimiser, LP solver, AD backend or manifold package.

`ToricFactor` supplies the finite potentials. `independence` and `hierarchical` return integer design
matrices so callers can inspect the statistics before constructing a factor. [[toric]]
records the state ordering, label rules and free-energy accounting. [[fit]] describes
`fit_mle`, `inmodel` and `binomial_residuals`, their independent closed-form checks,
and the separate moment-convergence and MLE-existence statuses.

The package test environment follows `Adversarial`'s workspace layout. Tests include the
repository's `test/factor_conformance.jl` through its direct-message path, with independent
site and pooled-posterior fixtures for every supported polarity of the test factors. The CI
matrix and API documentation include this package separately from the top-level `Lenticulum`
package.

The main integration difficulty was the difference between an extrinsic site and a
posterior: graph messages exclude the target prior, whereas pooling multiplies it back in.
The core lens contract also asks for a joint latent posterior, which these marginal sites
do not expose; the conformance checker has a separate path for direct-message factors,
retaining its existing lens checks by default. Another was entropy counting: Mycelium already
supplies the variable correction.
Neither operation belongs in the outgoing site itself.

[[grassmann]] describes `Grassmann`, `project_tangent`, `retract`, `ispoint` and
`Retracted`. The latter wraps Mycelium's gradient descent and momentum without
changing its state layout or recurrence. Projection and QR use only
LinearAlgebra; sampling uses Random. The geometry tests compare descent on a
trace objective with a known eigenplane and drive the same iterates through
`OptimiserFactor`. No Mycelium changes or new dependencies are required.

[[plucker]] records the coordinate order, sparse relation tables and forward-only
`PluckerFactor`. Dirac and paired-particle transport are exact. Independent Gaussian
inputs give exact moments and a Gaussian approximation to the output law.
`plucker_moments` exposes singular covariances; messages reject those unless callers
explicitly request covariance jitter, since the core belief type needs positive-definite
covariance. This factor has no game-level quantities or assembled lens, so the full
conformance checker does not apply (LP-009). Its tests check direct messages and a
Mycelium DAG. A separate Klein data rank test is not nullspace fitting.

Not implemented: general MLE existence certification, Markov-basis computation,
Sampson scalarisation, general polynomial factors, homotopy continuation or inverse Plücker fibres.
