#implementation

> `AlgebraicFactors` starts the algebraic family with exact finite toric potentials.
> It supplies discrete messages, energies and entropies, moment fitting and model
> checks. Other algebraic families remain outside this implementation.

> Sources: original implementation of [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §0 and §1.1–§1.7; code: `AlgebraicFactors.jl`, `toric.jl`, `fit.jl`, `../test/runtests.jl`.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)

The module depends on `LenticulumCore`, `Mycelium`, `LinearAlgebra`, `LuxCore` and
`Random`. The last two implement the repository's parameter/state interface; they add
no inference algorithm. Newton and binary-feature IPF use only LinearAlgebra; there
is no external optimiser, LP solver, AD backend or manifold package.

`ToricFactor` is the factor. `independence` and `hierarchical` return integer design
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

Not implemented: general MLE existence certification, Markov-basis computation,
Grassmannian parameters, polynomial factors, homotopy continuation or Plücker factors.
