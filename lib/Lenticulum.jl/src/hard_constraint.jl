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
# The grading of `LinearConstraintFactor`: with a noise the free energy is that factor's; without,
# `fit` is 0 and the finite remainder is `complexity` and `negentropy` (`local_free_energy`).
LenticulumCore.energyspace(::HardConstraintFactor) = LenticulumCore.GradedEnergySpace((
    fit = LenticulumCore.ScalarEnergySpace(),
    complexity = LenticulumCore.ScalarEnergySpace(),
    negentropy = LenticulumCore.ScalarEnergySpace(),
))
LenticulumCore.scalarisation(::HardConstraintFactor) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(),
    complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation(),
))

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

Mycelium.uses_cavity(::HardConstraintFactor) = true

"""
    Mycelium.local_free_energy(f::HardConstraintFactor, msgs, ps, st)

The factor's Bethe term at its incoming messages. Points and the hard directions of incoming
extended Gaussians are substituted out, leaving the relation ``\\tilde G s = \\tilde c`` on the
remaining coordinates ``s`` (orthonormal along each input's support).

- **Without noise** (`noise = 0`), the limit of `LinearConstraintFactor` with noise
  ``\\varepsilon I`` as ``\\varepsilon \\to 0``, whose divergent parts cancel:

  ```math
  F_f \\;=\\; \\tfrac12 \\log\\det\\bigl(\\tilde G \\tilde G^\\top\\bigr) \\;-\\; H_R(b_f),
  ```

  with ``H_R`` the entropy of the factor belief relative to Lebesgue measure on the solution set
  ``R`` of the relation (`complexity` and `negentropy`; `fit` is `0`). It needs ``\\tilde G`` of
  full row rank: redundant equations, or equations on directions an input already fixes exactly,
  have an infinite free energy, and throw. So does a factor belief that is flat along ``R``.
- **With a positive definite noise**, `LinearConstraintFactor`'s free energy.

A noise that is singular but not zero is not implemented. See `hard_constraint.md`, "Free energy".
"""
function Mycelium.local_free_energy(f::HardConstraintFactor{names}, msgs, ps, st) where {names}
    G, c, canon = _reduced_relation(ps.A, ps.c, msgs, f.dims, names)
    if !iszero(f.Q)
        isposdef(Symmetric(f.Q)) || throw(ArgumentError(
            "HardConstraintFactor: the free energy with a singular nonzero noise is not implemented"))
        r̄, Cr, H = _reduced_residual_statistics(G, c, canon, f.Q)
        Qi = inv(_chol(f.Q, "Q"))
        fit = (r̄' * (Qi * r̄) + tr(Qi * Cr)) / 2
        complexity = logdet(_chol(2π .* f.Q, "2πQ")) / 2
        return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -H)), st
    end
    m, k = size(G)
    F = LinearAlgebra.svd(G; full = true)
    tol = maximum(size(G)) * eps(Float64) * maximum(F.S; init = 0.0)
    (k >= m && count(>(tol), F.S) == m) || throw(ArgumentError(
        "HardConstraintFactor: its $m equations have rank $(count(>(tol), F.S)) on the directions its \
         inputs leave free (redundant equations, or equations on directions an input fixes \
         exactly); the free energy is infinite"))
    N = F.V[:, (m + 1):k]                                # R = s0 + range(N), any solution s0
    complexity = sum(log, F.S; init = 0.0)               # ½ log det(G Gᵀ)
    H = 0.0                                              # k = m: R is a point
    if k > m                                             # the entropy needs only the precision on R
        Λs = _blockdiag(map(last, canon))
        Λz = Symmetric(N' * Λs * N)
        C = cholesky(Λz; check = false)
        LinearAlgebra.issuccess(C) && minimum(LinearAlgebra.eigvals(Λz)) > LenticulumCore._eig_tol(LinearAlgebra.eigvals(Λz)) ||
            throw(ArgumentError("HardConstraintFactor: the factor belief is flat along the relation's \
                solution set (an improper posterior); it has no free energy"))
        H = (k - m) * (1 + log(2π)) / 2 - logdet(C) / 2
    end
    return LenticulumCore.GradedEnergy((fit = 0.0, complexity = complexity, negentropy = -H)), st
end
