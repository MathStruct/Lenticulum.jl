#implementation

> `beliefs.jl`: Gaussian belief projections (`moment_match`, `reduce_mixture`) and
> `kl_divergence`.

> Sources: code: `beliefs.jl`, `lib/LenticulumCore.jl/src/gaussian_belief.jl`, `lib/Mycelium.jl/src/gaussian.jl`; Minka, *Expectation Propagation for Approximate Bayesian Inference*, UAI 2001; Runnalls, *Kullback–Leibler Approach to Gaussian Mixture Reduction*, IEEE TAES 2007
>
> Bibliography: [[Bibliography#^minka2001ep|Minka 2001]] · [[Bibliography#^runnalls2007kl|Runnalls 2007]]
>
> Theory: [[Gaussian Belief]] · [[Belief Algebra]] · [[Mixture Belief]]

## 1. Where the Gaussian belief pieces live

The Gaussian belief implementation is split across three layers according to what each package
needs:

| Layer | File | Contents |
|---|---|---|
| `LenticulumCore` | `src/gaussian_belief.jl` | `GaussianBelief(η, Λ)`, `Gaussian(μ, Σ)`, `uninformative`, `isuninformative`, `isproper`, `belief_mean`, `belief_cov`, `logpartition` |
| `Mycelium` | `src/gaussian.jl` | `combine`, `belief_logdensity`, `variable_entropy`, `belief_distance`, `can_damp`, `_damp`, `_product` |
| `Lenticulum` | `src/beliefs.jl` | `kl_divergence`, `moment_match`, `reduce_mixture` |

## 2. Projections onto the Gaussian family

- **`kl_divergence(q, p)`**: Computes $\mathrm{KL}(q \,\|\, p)$ for two proper `GaussianBelief`s
  using their Cholesky factors.
- **`moment_match(b)`**: Computes the inclusive KL projection
  $\arg\min_{q \in \mathcal{G}} \mathrm{KL}(b \,\|\, q)$ by matching the first two moments of a
  `SampleBelief` or `MixtureBelief{<:GaussianBelief}`.
- **`reduce_mixture(m; max_components, prune)`**: Prunes negligible components and greedily
  merges pairs minimizing Runnalls' upper bound on the KL divergence introduced by each merge:
  $$\frac{1}{2}\Bigl[(w_i + w_j)\log\det\Sigma_{ij} - w_i\log\det\Sigma_i - w_j\log\det\Sigma_j\Bigr]$$
  while preserving the total mean and covariance of the unpruned mixture.

## 3. What it does not do

- **Proper components only.** `kl_divergence`, `moment_match(::MixtureBelief)`, and
  `reduce_mixture` require every input `GaussianBelief` component to be proper ($\Lambda \succ 0$)
  so its mean, covariance, and log-determinant exist. Improper likelihood messages must be
  combined with a proper prior before projection or mixture reduction.
- **Inclusive KL (`moment_match`) only.** Exclusive M-projection ($\arg\min_q \mathrm{KL}(q \,\|\, p)$)
  is not implemented here.

## 4. Implementation difficulties

- **Sample covariance singularity at small `N`.** When a `SampleBelief` has fewer distinct samples
  than $d + 1$ (or collinear particles), the weighted empirical covariance $\Sigma$ is rank-deficient
  and its precision $\Lambda = \Sigma^{-1}$ does not exist. `moment_match(s::SampleBelief)` checks
  `isposdef(Symmetric(Σ))` and throws an `ArgumentError` rather than fabricating an arbitrary
  regulariser.
- **Weight-prune vs. moment-preservation trade-off in `reduce_mixture`.** Pruning components with
  $w_i < \text{prune} \cdot \max_k w_k$ before greedy merging shifts the mixture moments by at most
  the pruned weight; setting `prune = 0.0` preserves the exact first two moments across all merges.

Related: [[Gaussian Belief]], [[Mixture Belief]], [[Belief Algebra]], [[src/gaussian|gaussian]]
