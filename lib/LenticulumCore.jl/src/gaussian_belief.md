#implementation

> Sources: code: `gaussian_belief.jl`
>
> Theory: [[Gaussian Belief]] · [[Beliefs]] · [[Belief Algebra]]

Implements: the Gaussian belief type in canonical form, $\mathcal N^{-1}(\eta, \Lambda)$ with
$\Lambda = \Sigma^{-1}$, $\eta = \Lambda\mu$, and its accessors.

## 1. What is in the file

| name | what |
|---|---|
| `GaussianBelief(η, Λ)` | the type; `Λ` may be singular (a likelihood message) |
| `Gaussian(μ, Σ)` | from moments; `Σ` must be positive definite |
| `uninformative(n)` | $\eta = 0$, $\Lambda = 0$: the identity for pooling |
| `isproper`, `belief_mean`, `belief_cov`, `logpartition` | require a proper belief, and say so |
| `dimension` | extends the package's `dimension` |

## 2. Why it is here

It used to live in the top-level `Lenticulum` package, where no `lib/` package could reach it.
So the equilibrium and diffusion factors returned points even where they could compute a
Gaussian, and four notes recorded the same wall ([[The Equilibrium Family]] §5). It moved here,
next to the other five belief types, in October 2026. The split is now:

| layer | Gaussian code |
|---|---|
| `LenticulumCore` (`gaussian_belief.jl`) | the type and its accessors |
| `Mycelium` (`gaussian.jl`) | the message rules: `combine`, densities, entropy, distance, damping, `_product` |
| `Lenticulum` (`beliefs.jl`) | projections: `moment_match`, `reduce_mixture`, and `kl_divergence` |

`Lenticulum` re-exports every name, so user code did not change. The cost is one dependency,
the `LinearAlgebra` standard library.

## 3. A fix that came with the move

`VariationalDiffusion` and `ImplicitLayers` read a prior's point through
`Mycelium.belief_mean`, a function that never existed. The call sat in a `try` block, so every
Gaussian prior was silently treated as having no point. It now calls
`LenticulumCore.belief_mean`, and a Gaussian prior's mean is the anchor and warm start, as
documented. This is tested in VariationalDiffusion's "answers as beliefs" test set.

Related: [[open_model]], [[gaussian]], [[messages]]
