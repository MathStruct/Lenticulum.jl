"""
    Lenticulum

Implicit machine learning on factor graphs: learning **relations**
``R_\\theta \\subseteq X_1\\times\\cdots\\times X_n`` rather than functions.

This package is the user-facing layer, mirroring `Lux.jl`. It sits on

- `LenticulumCore` — what a **factor** is (a parameterized statistical game);
- `Mycelium` — how factors are **wired** and in what **order** they talk.

## What is here so far

The **linear-Gaussian factor** and its beliefs: the first factor in the project that
implements the entire interface with nothing stubbed — a real open model, an *exact* Bayesian
inversion, a genuine vector energy and entropy, bidirectional polarity, and learnable
parameters.

Its purpose is to be a **test oracle**. Every quantity it produces has a closed form, so the
framework's claims can be checked against arithmetic instead of against itself. Two of them
turned out to be wrong; see `The Gaussian Factor.md` §"What this caught".

- `beliefs.jl` — [`GaussianBelief`](@ref) in canonical form, which makes
  `Mycelium.combine` **addition** and thereby unblocks message passing.
- `gaussian.jl` — [`GaussianFactor`](@ref) and [`GaussianPrior`](@ref).
- `constraint.jl` — [`LinearConstraintFactor`](@ref), the **acausal** n-ary sibling of
  `GaussianFactor`: one equation ``0 = \\sum_i A_i x_i - c + \\varepsilon`` over any number of
  channels, none of them distinguished. This is ModelingToolkit's `0 ~ ...` equation as a
  statistical game; see `ModelingToolkit as an Acausal Relation.md`.
"""
module Lenticulum

using LinearAlgebra: LinearAlgebra, Diagonal, I, Symmetric, cholesky, isposdef, logdet, tr
using Random: Random, AbstractRNG
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore
using Mycelium: Mycelium

include("beliefs.jl")
include("gaussian.jl")
include("constraint.jl")

export GaussianBelief, Gaussian, uninformative
export belief_mean, belief_cov, isproper, logpartition, kl_divergence
export GaussianFactor, GaussianPrior, LinearGaussianModel
export LinearConstraintFactor, LinearConstraintModel
export residual, residual_statistics

end # module
