#definition

> `MixtureBelief(components, weights)` is a weighted sum ``\sum_k w_k\,b_k`` of beliefs of one
> type. It is how a belief holds **several answers at once**: both branches of a relation,
> both poses of a robot arm. Its product with another belief is again a mixture, so it travels
> through message passing; `reduce_mixture` keeps it from growing.

> Sources: code: `open_model.jl` (the type), `messages.jl` (density and products), `beliefs.jl` in the top-level package (Gaussian components: product, `moment_match`, `reduce_mixture`); Runnalls, *Kullback-Leibler Approach to Gaussian Mixture Reduction*, IEEE Transactions on Aerospace and Electronic Systems 2007 (the merge cost)
>
> Bibliography: [[Bibliography#^runnalls2007kl|Runnalls 2007]]
>
> Theory (CT-ML wiki): [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category)

## Why it is needed

Point inference returns one branch of a multivalued relation; [[Implicit Learners]] can now
find all of them (`implicit_roots`), and a Laplace approximation gives each a Gaussian. A
mixture of those Gaussians, weighted, is the belief that holds the whole answer, and it is the
form in which that answer can be passed to a neighbouring factor
([[Open Problems in Implicit Diffusion Learning]] T1, I1).

## How it behaves

- **Density**: ``\log p(x) = \log\sum_k w_k\,p_k(x)``.
- **Pooling with a belief ``b``**: ``\bigl(\sum_k w_k p_k\bigr)\,p_b \propto \sum_k w_k Z_k\,\frac{p_k\,p_b}{Z_k}``
  with ``Z_k = \int p_k\,p_b``: each component is pooled with ``b``, and its weight is scaled
  by how much it overlaps ``b``. A component far from the evidence loses weight. Exact for
  Gaussian components (the overlap is a ratio of log-partition functions) and categorical ones.
  An uninformative Gaussian message has no normalising constant; it is the same for every
  component and cancels, so the product is still exact.
- **Mixture with mixture**: every pair of components, ``K \cdot L`` of them. That growth is the
  price of exactness.
- **Reduction**: `reduce_mixture` drops negligible components and merges pairs into single
  Gaussians with the same weight, mean and covariance, choosing the pair whose merge costs least
  by Runnalls' bound on the Kullback–Leibler divergence. The overall mean and covariance are
  preserved exactly.
- **Projection**: `moment_match` gives the single Gaussian with the mixture's mean and
  covariance, the projection step of expectation propagation.
- **Entropy**: no closed form; `variable_entropy` is not defined for mixtures.

Related: [[Beliefs]], [[Belief Algebra]], [[Gaussian Belief]], [[Categorical Belief]],
[[Inference Signatures]]
