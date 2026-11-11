# ---------------------------------------------------------------------------
# HardConstraintFactor: the noise-free linear relation  Σᵢ Aᵢ xᵢ = c,  exactly.
#
# Its messages are extended Gaussians (LenticulumCore `extended_gaussian.jl`): the other inputs'
# joint pushed forward to r = c − Σ_{i≠t} Aᵢ xᵢ (+ ε with an optional noise), then pulled back
# through Aₜ. Both steps are exact for flat, hard and Gaussian directions, so the factor is what
# `LinearConstraintFactor(…; noise = 1e-8)` approximates. See `hard_constraint.md`.
# ---------------------------------------------------------------------------

"""
    HardConstraintFactor(dims::NamedTuple, m::Int; noise = 0,
                         init_coeff = (rng, m, n) -> randn(rng, m, n) ./ sqrt(n),
                         init_constant = (rng, m) -> zeros(m))

The linear relation ``\\sum_i A_i x_i = c`` among the channels `dims` (names and dimensions), with
`m` scalar equations, imposed **exactly**. Parameters `(A, c)` as for [`LinearConstraintFactor`](@ref),
and the same polarities: any one channel `Unobserved()`, each other `Observed()` or `Latent()`.

Its messages are [`ExtendedGaussian`](@ref)s: hard along what the equation fixes, flat along what
it cannot see. An optional `noise` (a positive semidefinite `m × m` covariance, or a scalar)
softens the relation in the directions where it is positive; with a positive definite noise the
messages are those of `LinearConstraintFactor` with that noise.

Inputs may be `GaussianBelief`s (also improper), `DiracBelief`s, `ExtendedGaussian`s, or carry
no information (`TrivialBelief()`, no message, a latent channel). Mixtures are refused.

Messages only: the factor has no free energy (a relation without noise has no finite one in
closed form) and is not learnable through the free energy; `local_free_energy` throws.

```julia
# Kirchhoff's current law at a node, exactly: i₁ + i₂ + i₃ = 0
f  = HardConstraintFactor((i1 = 1, i2 = 1, i3 = 1), 1)
ps = (A = (i1 = ones(1, 1), i2 = ones(1, 1), i3 = ones(1, 1)), c = [0.0])
```
"""
struct HardConstraintFactor{names,D<:Tuple,W,B} <: LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names,D}
    m::Int
    Q::Matrix{Float64}
    init_coeff::W
    init_constant::B
end

function HardConstraintFactor(dims::NamedTuple, m::Int; noise = 0,
                              init_coeff = (rng, m, n) -> randn(rng, m, n) ./ sqrt(n),
                              init_constant = (rng, m) -> zeros(m))
    isempty(dims) && throw(ArgumentError("a constraint needs at least one channel"))
    all(d -> d isa Int && d > 0, values(dims)) ||
        throw(ArgumentError("channel dimensions must be positive Ints; got $(dims)"))
    m > 0 || throw(ArgumentError("the residual dimension m must be positive; got $m"))
    Q = noise isa Real ? Matrix{Float64}(noise * I, m, m) : Matrix{Float64}(noise)
    size(Q) == (m, m) || throw(DimensionMismatch("noise must be $m × $m; got $(size(Q))"))
    LenticulumCore._nearly_symmetric(Q) || throw(ArgumentError("noise covariance must be symmetric"))
    Q = Matrix(LenticulumCore._symmetric_part(Q))
    isempty(Q) || minimum(LinearAlgebra.eigvals(Symmetric(Q))) >= -LenticulumCore._eig_tol(LinearAlgebra.eigvals(Symmetric(Q))) ||
        throw(ArgumentError("noise covariance must be positive semidefinite"))
    return HardConstraintFactor(dims, m, Q, init_coeff, init_constant)
end

LuxCore.initialparameters(rng::AbstractRNG, f::HardConstraintFactor{names}) where {names} = (
    A = NamedTuple{names}(map(n -> f.init_coeff(rng, f.m, f.dims[n]), names)),
    c = f.init_constant(rng, f.m),
)
LuxCore.initialstates(::AbstractRNG, ::HardConstraintFactor) = NamedTuple()
LuxCore.parameterlength(f::HardConstraintFactor) = f.m * sum(values(f.dims)) + f.m

LenticulumCore.channels(f::HardConstraintFactor{names}) where {names} =
    map((n, d) -> LenticulumCore.Channel(n, d), names, values(f.dims))
LenticulumCore.supported_polarities(f::HardConstraintFactor{names}) where {names} = _one_unobserved_polarities(names)
LenticulumCore.hasnopolarity(::HardConstraintFactor) = false
LenticulumCore.isunidirectional(::HardConstraintFactor{names}) where {names} = length(names) == 1
function LenticulumCore.supports_polarity(f::HardConstraintFactor{names}, p::LenticulumCore.Polarity) where {names}
    Set(keys(p)) == Set(names) || return false
    return length(LenticulumCore.unobserved_channels(p)) == 1
end
LenticulumCore.islearnable(::HardConstraintFactor) = false

# an input as an extended Gaussian of dimension n
_as_extended(b, n) = LenticulumCore.extended(b, n)
_as_extended(b::LenticulumCore.MixtureBelief, n) = throw(ArgumentError(
    "HardConstraintFactor does not take mixture inputs; collapse them first (`moment_match`)"))

"""
    Mycelium.factor_message(f::HardConstraintFactor, target, polarity, inputs, prior, ps, st)

The message to channel `t`: the joint of the other inputs, as extended Gaussians, pushed forward
to ``r = c - \\sum_{i \\ne t} A_i x_i + \\varepsilon`` (``\\varepsilon \\sim \\mathcal N(0, Q)``, `Q` the
noise, zero by default), then the likelihood of ``x_t`` under ``A_t x_t = r``:

```math
\\mu_{f\\to t} \\;=\\; \\operatorname{preimage}\\bigl(\\operatorname{pushforward}(\\textstyle\\bigotimes_{i\\ne t} \\mu_{i\\to f},\\ -[A_i]_{i \\ne t},\\ c,\\ Q),\\ A_t\\bigr).
```

An [`ExtendedGaussian`](@ref): hard where the equation and the inputs fix ``x_t``, flat along
``\\operatorname{null}(A_t)`` and wherever an uninformative input reaches.
"""
function Mycelium.factor_message(f::HardConstraintFactor{names}, target::Symbol, polarity, inputs, prior, ps, st) where {names}
    target in names || throw(ArgumentError("channel :$target is not a channel of this factor ($(names))"))
    others = Tuple(n for n in names if n !== target)
    A = ps.A
    if isempty(others)
        r = LenticulumCore.ExtendedGaussian(float.(ps.c), f.Q)
    else
        input(n) = haskey(inputs, n) ? getfield(inputs, n) : nothing
        joint = LenticulumCore._independent_join([_as_extended(input(n), f.dims[n]) for n in others])
        M = reduce(hcat, [-Matrix(getfield(A, n)) for n in others])
        r = LenticulumCore.pushforward(joint, M, float.(ps.c), f.Q)
    end
    return LenticulumCore.preimage(r, Matrix(getfield(A, target))), st
end

"""
    Mycelium.local_free_energy(f::HardConstraintFactor, msgs, ps, st)

Not defined: a relation imposed exactly has no finite free energy in closed form. Throws.
"""
Mycelium.local_free_energy(::HardConstraintFactor, msgs, ps, st) = throw(ArgumentError(
    "HardConstraintFactor has no free energy: an exact relation has none in closed form. Use \
     LinearConstraintFactor with a noise for a free energy"))
