#implementation

> Canonical Gaussian beliefs, their moment accessors and properness checks.

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
| `Mycelium` (`gaussian.jl`) | the message rules: `combine`, densities, entropy, distance, damping, `combine_with_overlap` |
| `Lenticulum` (`beliefs.jl`) | projections: `moment_match`, `reduce_mixture`, and `kl_divergence` |

`Lenticulum` re-exports every name, so user code did not change. The cost is one dependency,
the `LinearAlgebra` standard library.

## 3. A fix that came with the move

`VariationalDiffusion` and `ImplicitLayers` read a prior's point through
`Mycelium.belief_mean`, a function that never existed. The call sat in a `try` block, so every
Gaussian prior was silently treated as having no point. It now calls
`LenticulumCore.belief_mean`, and a Gaussian prior's mean is the anchor and warm start, as
documented. This is tested in VariationalDiffusion's "answers as beliefs" test set.

Related: [[open_model]], [[lib/Mycelium.jl/src/gaussian|gaussian]], [[messages]]


## Device diagonal beliefs (2026-10-27)

For device `Diagonal` precision/covariance, properness is an all-positive finite
check, mean is elementwise division, covariance is reciprocal diagonal and logdet
is the sum of logarithms. The diagonal log-partition uses typed constants and an
elementwise quadratic form, preserving Float32 in values and both tested derivatives.
No dense factorisation or host array copy is needed.
Dense device Gaussian precision/covariance gives an ArgumentError rather than
silently entering the legacy CPU `Matrix` path. CPU factorisation behavior is
unchanged. Hardware dense Gaussian algebra is not claimed.

Device mean/solve paths use elementwise operations; the covariance reverse rule
returns structural `diag` cotangents. A dense precision cotangent conflicted with direct diagonal reads in
Zygote when the same belief contributed moments and variance terms. The analytic
rules are $\bar\eta=\bar\mu/\lambda$, $\bar\lambda=-\bar\eta\,\mu$ and
$\bar\lambda=-\operatorname{diag}(\bar\Sigma)/\lambda^2$, elementwise.
Tests in `lib/ImplicitLayers.jl/test/device_arrays.jl` compare Gaussian graph
expectation derivatives with closed forms and CPU references. Singular diagonal
beliefs are still valid likelihood messages, but moment/density requests reject them.


## Batch C graph polarity check (2026-10-28)

A Gaussian-consuming device factor exposed an ambiguity between GPUArrays and
LinearAlgebra/SparseArrays when `isuninformative` reduced over the whole device-backed
`Diagonal` wrapper. The GPUArraysCore extension now checks its stored diagonal
and information vector directly. This is equivalent to checking the full diagonal
matrix, and keeps CPU behavior unchanged. `Mycelium/test/graph_adjoint_devices.jl`
checks zero/precision-only/information-only beliefs, then tree/loopy Gaussian
schedule replay with scalar indexing disabled. Earlier terminal-only device
Gaussian fixtures did not exercise this polarity-resolution path.

## Symmetry is checked (2026-10-29, LP-019 item 3)

`Symmetric(M)` reads one triangle, so `Gaussian(μ, [1 0.2; 10 1])` was accepted as
`[1 0.2; 0.2 1]`, and the same matrix as a precision passed `isproper` although its symmetric
part has eigenvalues −4.1 and 6.1. `Gaussian`, the precision factorisation behind the moment
accessors, and `isproper` now check $\max|M - M^\top| \le \sqrt{\varepsilon_{\text{Float32}}}\,\max|M|$
first: rounding-level asymmetry, as left by $A\Lambda A^\top$, passes. `GaussianBelief(η, Λ)`
itself still accepts any square `Λ`, since it may be a likelihood; an asymmetric one is simply
not proper. Tested in `test/lp019.jl`.

After review (2026-10-30): a tolerance of $\sqrt{\varepsilon_{\text{Float32}}}$ still admitted
`[1 0.9999; 1.0002 1]`, whose upper triangle is definite and whose symmetric part is not. The
tolerance is now $\sqrt\varepsilon$ of the element type, and definiteness is checked on, and
factorisations use, the symmetric part $M/2 + M^\top/2$ (`_symmetric_part`; the form
$(M + M^\top)/2$ overflowed for entries near floatmax).

Second review (2026-10-30): the tolerance is again $\sqrt{\varepsilon_{\text{Float32}}}$ for every
element type. With definiteness on the symmetric part it only refuses gross asymmetry, and a
type-dependent tolerance had changed under ForwardDiff duals and rejected LP-002's
finite-difference adjoint perturbations. The symmetric part is $M + (M^\top - M)/2$, exact for
symmetric $M$ (a subnormal diagonal survives) and without overflow.
