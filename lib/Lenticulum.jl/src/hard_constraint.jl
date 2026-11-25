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
no information (`TrivialBelief()`, no message, a latent channel). A mixture is collapsed to its
total moments (`moment_match`), as the relation factors collapse one.

Its Bethe term (`local_free_energy`) is the finite limit of the noisy relation's as the noise
vanishes along the exact directions, for a zero, singular or positive definite noise (see its
docstring). Its own `A` and `c` are not learnable (`islearnable` is `false`); gradients of the
other factors go through it.

```julia
# Kirchhoff's current law at a node, exactly: i₁ + i₂ + i₃ = 0
f  = HardConstraintFactor((i1 = 1, i2 = 1, i3 = 1), 1)
ps = (A = (i1 = ones(1, 1), i2 = ones(1, 1), i3 = ones(1, 1)), c = [0.0])
```
"""
struct HardConstraintFactor{names,D<:Tuple,W,B,Q<:AbstractMatrix} <: LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names,D}
    m::Int
    Q::Q
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
    Q = _as_cov(noise, m)
    size(Q) == (m, m) || throw(DimensionMismatch("noise must be $m × $m; got $(size(Q))"))
    LenticulumCore._nearly_symmetric(Q) || throw(ArgumentError("noise covariance must be symmetric"))
    Q = Matrix(LenticulumCore._symmetric_part(Q))
    if !iszero(Q)
        spectrum = _eigvals(Symmetric(LenticulumCore.stop_gradient(Q)))
        minimum(spectrum) >= -LenticulumCore._eig_tol(spectrum) ||
            throw(ArgumentError("noise covariance must be positive semidefinite"))
    end
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
_as_extended(b, n, T) = LenticulumCore.extended(b, n, T)   # T: the type of no input (P30)
# A mixture is collapsed to its total moments, as the relation factors do (`gaussian.md`, "Mixture
# inputs"): the message is exact for the collapsed input, an assumed-density approximation for the
# mixture (P27). Before P27 it was refused.
_as_extended(b::LenticulumCore.MixtureBelief, n, T) = LenticulumCore.extended(moment_match(b), n)

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
    Q = _covariance_at(f.Q, ps, inputs, prior)
    if isempty(others)
        r = LenticulumCore.ExtendedGaussian(_at_eltype(eltype(Q), ps.c), Q)
    else
        input(n) = haskey(inputs, n) ? getfield(inputs, n) : nothing
        joint = LenticulumCore._independent_join([_as_extended(input(n), f.dims[n], eltype(Q)) for n in others])
        M = reduce(hcat, [-_matrix(getfield(A, n)) for n in others])
        r = LenticulumCore.pushforward(joint, M, _at_eltype(eltype(Q), ps.c), Q)
    end
    return LenticulumCore.preimage(r, _matrix(getfield(A, target))), st
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
- **With a singular nonzero noise** ``Q = Q_s D Q_s^\\top`` (P5), exact along its null space
  ``Q_h`` and soft along its range: the soft equations ``Q_s^\\top(\\tilde G s - \\tilde c) \\sim \\mathcal N(0, D)``
  join the inputs' precision, the exact ones ``Q_h^\\top \\tilde G s = Q_h^\\top \\tilde c`` give the term
  above, and `fit` and `complexity` gain the soft residual's
  ``\\tfrac12\\mathbb E\\,[r^\\top D^{-1} r] + \\tfrac12\\log\\det 2\\pi D``. The exact equations need full row rank.

See `hard_constraint.md`, "Free energy".
"""
function Mycelium.local_free_energy(f::HardConstraintFactor{names}, msgs, ps, st) where {names}
    G, c, canon = _reduced_relation(ps.A, ps.c, msgs, f.dims, names)
    soft = nothing
    Q = _covariance_at(f.Q, ps, msgs)
    if !iszero(Q)
        # positive definite by the eigenvalue tolerance the messages use (`_eig_tol`), not by
        # `isposdef`: a rank-deficient B Bᵀ has eigenvalues of 1e-16 that a Cholesky accepts
        Qh, Qs, d = _noise_split(Q)
        if size(Qh, 2) == 0
            r̄, Cr, H = _reduced_residual_statistics(G, c, canon, Q)
            Qi = _spd_inv(Q, "Q")
            fit = (r̄' * (Qi * r̄) + tr(Qi * Cr)) / 2
            complexity = logdet(_chol((2 * eltype(Q)(π)) .* Q, "2πQ")) / 2
            return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = _negentropy(H))), st
        end
        # P5: a singular nonzero noise Q = Qs D Qsᵀ (Qs, Qh orthonormal bases of its range and null
        # space). The relation is exact along Qh, Qhᵀ(G s − c) = 0, and soft along Qs,
        # Qsᵀ(G s − c) ~ N(0, D): the soft part joins the inputs' precision, the exact part is the
        # noise-free term below (`hard_constraint.md`, "Free energy with a singular noise").
        soft =(G = Qs' * G, c = Qs' * c, d = d)
        G, c = Qh' * G, Qh' * c
    end
    m, k = size(G)
    # G (and the messages) may carry derivatives (P4): the SVD is of the values, and G of full row
    # rank moves its null space N and ½ log det(GGᵀ) = Σ log σᵢ by the first-order terms
    # N = N₀ − G⁺ΔN₀ and tr(G⁺Δ), with Δ the tangent of G (LenticulumCore `extended_gaussian.md`,
    # "Derivatives"). Full row rank is an open condition: no tangent changes it.
    U0, S0, V0, _ = LenticulumCore._value_svd(G)
    tol = maximum(size(G)) * _eps_eltype(eltype(S0)) * LenticulumCore._scale(S0)
    (k >= m && count(>(tol), S0) == m) || throw(ArgumentError(
        "HardConstraintFactor: its $m equations have rank $(count(>(tol), S0)) on the directions its \
         inputs leave free (redundant equations, or equations on directions an input fixes \
         exactly); the free energy is infinite"))
    Δ = LenticulumCore._tangent(G)
    Gp = V0[:, 1:m] * (U0[:, 1:m]' ./ S0[1:m])          # G⁺ at the value
    N = V0[:, (m + 1):k] .- Gp * (Δ * V0[:, (m + 1):k])  # R = s0 + range(N), any solution s0
    complexity = LenticulumCore._logsum(S0) + sum(Gp' .* Δ)   # ½ log det(G Gᵀ)
    H = false                                              # k = m: R is a point
    Λz = nothing
    if k > m                                             # the entropy needs only the precision on R
        Λs = _blockdiag(map(last, canon))
        soft === nothing || (Λs = Λs .+ soft.G' * (soft.G ./ soft.d))
        Λz = N' * Λs * N
        Λz = (Λz + Λz') / 2
        λz = _eigvals(Symmetric(LenticulumCore.stop_gradient(Λz)))
        minimum(λz) > LenticulumCore._eig_tol(λz) || throw(ArgumentError(
            "HardConstraintFactor: the factor belief is flat along the relation's \
             solution set (an improper posterior); it has no free energy"))
        H = (k - m) * (1 + LenticulumCore._log2π(eltype(Λz))) / 2 - logdet(_chol(Λz, "the factor belief's precision on R")) / 2
    end
    soft === nothing &&
        return LenticulumCore.GradedEnergy((fit = false, complexity = complexity, negentropy = _negentropy(H))), st
    # The soft residual r = Gs s − cs under the factor belief on R: s = s0 + N z with s0 a solution
    # of G s0 = c (in plain arithmetic, so that it carries the tangent of G and c) and
    # z ~ N(Λz⁻¹ Nᵀ(η' − Λ' s0), Λz⁻¹), η' and Λ' the inputs' information and precision with the
    # soft part added. The mean does not depend on which solution s0 is taken.
    s0 = G' * (_symmetric_solve(G * G', c))
    if Λz === nothing
        s̄, Cs = s0, nothing
    else
        ηs = reduce(vcat, map(first, canon)) .+ soft.G' * (soft.c ./ soft.d)   # Λs has the soft part (above)
        Sz = _symmetric_inv(Λz)
        s̄ = s0 .+ N * (Sz * (N' * (ηs .- Λs * s0)))
        Cs = N * Sz * N'
    end
    r̄ = soft.G * s̄ .- soft.c
    fit = sum(abs2.(r̄) ./ soft.d) / 2
    Cs === nothing || (fit += tr((soft.G' * (soft.G ./ soft.d)) * Cs) / 2)
    complexity += sum(log.((2 * eltype(soft.d)(π)) .* soft.d)) / 2
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = _negentropy(H))), st
end

# A positive semidefinite noise split into its null space Qh (exact directions), its range Qs and
# the positive eigenvalues d on it. The noise is a constant of the factor (no tangent).
function _noise_split(Q::AbstractMatrix)
    λ, V = LinearAlgebra.eigen(Symmetric(Q))
    pos = λ .> LenticulumCore._eig_tol(λ)
    return V[:, .!pos], V[:, pos], λ[pos]
end

# --- Plates (P6) --------------------------------------------------------------

# Inside a reverse pass (Zygote) this is `true` (LenticulumChainRulesCoreExt).
_within_gradient() = false

"""
    Mycelium.factor_messages(fs::AbstractVector{<:HardConstraintFactor}, target, polarity, inputs, priors, ps, st)

The messages of tied replicas (a plate), as `factor_message` gives them one by one. When the
replicas have one noise and dimensions, and each channel's inputs share their covariance and flat
directions across the replicas (points, `TrivialBelief()`, Gaussians of one precision, extended
Gaussians of one `Σ` and `D`; as with data and a shared prior), every decomposition of
`factor_message` depends on replica 1 only: the message of replica ``i`` is

```math
\\mu_i \\;=\\; L\\,\\Pi_r\\Bigl(c - \\textstyle\\sum_{n \\ne t} A_n m_{n,i}\\Bigr), \\qquad \\Sigma_i = \\Sigma_1,\\quad D_i = D_1,
```

with ``m_{n,i}`` the inputs' means, ``\\Pi_r`` the projection of the pushforward onto the complement
of its flat directions and ``L`` the linear map of `preimage` from that mean to the message's
(`hard_constraint.md`, "Plates"). Each replica's hard constraints are checked for a contradiction
as `preimage` checks them. Otherwise (anything else, dual numbers, a Zygote pass) each replica
goes through `factor_message`. `inputs` is a vector of per-replica `NamedTuple`s, or a
`NamedTuple` of `BeliefBlock`s, which gives a `VectorBlock`.
"""
function Mycelium.factor_messages(fs::AbstractVector{<:HardConstraintFactor{names}}, target::Symbol, polarity,
                                  inputs::AbstractVector, priors::AbstractVector, ps, st) where {names}
    target in names || throw(ArgumentError("channel :$target is not a channel of this factor ($(names))"))
    length(fs) == length(inputs) == length(priors) || throw(DimensionMismatch(
        "$(length(fs)) factors, $(length(inputs)) inputs and $(length(priors)) priors"))
    out = _hard_shared_messages(fs, target, ps, length(inputs),
                                n -> [haskey(inp, n) ? getfield(inp, n) : nothing for inp in inputs],
                                _hard_channel_parts)
    out === nothing || return out, st
    return _hard_each(fs, target, polarity, inputs, priors, ps, st)
end

function Mycelium.factor_messages(fs::AbstractVector{<:HardConstraintFactor{names}}, target::Symbol, polarity,
                                  inputs::NamedTuple, priors::AbstractVector, ps, st) where {names}
    target in names || throw(ArgumentError("channel :$target is not a channel of this factor ($(names))"))
    n = length(fs)
    out = _hard_shared_messages(fs, target, ps, n,
                                c -> haskey(inputs, c) ? getfield(inputs, c) : Mycelium.TrivialBlock(n),
                                _hard_block_parts)
    out === nothing || return Mycelium.VectorBlock(out), st
    cols = [NamedTuple{keys(inputs)}(map(b -> Mycelium.replica(b, i), values(inputs))) for i in 1:n]
    beliefs, st = _hard_each(fs, target, polarity, cols, priors, ps, st)
    # every message is an ExtendedGaussian, which `belief_block` would put in a VectorBlock too; built
    # directly, since `belief_block`'s copy into a `Vector{Any}` stops Zygote (LP-044)
    return Mycelium.VectorBlock(Vector{Any}(beliefs)), st
end

# Each replica through `factor_message`, which leaves `st` as it is. A comprehension, not writes
# into a vector: Zygote cannot differentiate those (Mycelium's default loop writes).
function _hard_each(fs, target, polarity, inputs, priors, ps, st)
    length(fs) == length(inputs) == length(priors) || throw(DimensionMismatch(
        "$(length(fs)) factors, $(length(inputs)) inputs and $(length(priors)) priors"))
    beliefs = [first(Mycelium.factor_message(fs[i], target, polarity, inputs[i], priors[i], ps, st))
               for i in eachindex(fs, inputs, priors)]
    return beliefs, st
end

# One channel's inputs over `k` replicas as shared parts (M, Σ, D): M the d × k matrix of the
# normalised means, Σ and D those of every replica. `nothing` when they are not shared, or not of
# plain host floats (dual numbers take the per-replica path).
function _hard_channel_parts(bs::AbstractVector, d::Int, k::Int)
    all(b -> b === nothing || b isa LenticulumCore.TrivialBelief, bs) &&
        return zeros(Bool, d, k), zeros(Bool, d, d), Matrix{Bool}(I, d, d)
    b1 = first(bs)
    if b1 isa LenticulumCore.DiracBelief
        all(b -> b isa LenticulumCore.DiracBelief && b.value isa AbstractVector{<:Union{Float32,Float64}} && length(b.value) == d, bs) ||
            return nothing
        return reduce(hcat, [b.value for b in bs]), zeros(eltype(b1.value), d, d), zeros(eltype(b1.value), d, 0)
    elseif b1 isa GaussianBelief
        all(b -> b isa GaussianBelief && eltype(b.η) <: Union{Float32,Float64} && length(b.η) == d &&
                 b.improper == b1.improper && (b.Λ === b1.Λ || isequal(b.Λ, b1.Λ)), bs) || return nothing
        eltype(b1.Λ) <: Union{Float32,Float64} || return nothing
        return _hard_gaussian_parts(b1, reduce(hcat, [b.η for b in bs]))
    elseif b1 isa LenticulumCore.ExtendedGaussian{<:Union{Float32,Float64}}
        all(b -> b isa LenticulumCore.ExtendedGaussian{<:Union{Float32,Float64}} && length(b.μ) == d &&
                 (b.Σ === b1.Σ || isequal(b.Σ, b1.Σ)) && (b.D === b1.D || isequal(b.D, b1.D)), bs) || return nothing
        return reduce(hcat, [b.μ for b in bs]), b1.Σ, b1.D
    end
    return nothing
end

# The same from a message block.
_hard_block_parts(b::Mycelium.TrivialBlock, d::Int, k::Int) = _hard_channel_parts(Any[nothing], d, k)
function _hard_block_parts(b::Mycelium.PointBlock, d::Int, k::Int)
    (eltype(b.X) <: Union{Float32,Float64} && size(b.X, 1) == d) || return nothing
    return Matrix(b.X), zeros(eltype(b.X), d, d), zeros(eltype(b.X), d, 0)
end
function _hard_block_parts(b::Mycelium.GaussianBlock, d::Int, k::Int)
    (b.Λ isa AbstractMatrix && eltype(b.Λ) <: Union{Float32,Float64} && eltype(b.η) <: Union{Float32,Float64} && size(b.η, 1) == d &&
     allequal(b.improper)) || return nothing
    return _hard_gaussian_parts(Mycelium.replica(b, 1), Matrix(b.η))
end
_hard_block_parts(b::Mycelium.VectorBlock, d::Int, k::Int) = _hard_channel_parts(b.beliefs, d, k)
_hard_block_parts(b, d::Int, k::Int) = nothing

# Gaussians of one precision Λ with the information vectors as the columns of H: `extended` of the
# first gives Σ = Λ⁺ and D = null(Λ), and the means are Σ ηᵢ. An improper ηᵢ must lie in the range
# of Λ, as `extended` checks; otherwise `nothing`, and the per-replica path throws its error.
function _hard_gaussian_parts(b1::GaussianBelief, H::AbstractMatrix)
    e = LenticulumCore.extended(b1)
    D = e.D
    if size(D, 2) > 0
        for i in axes(H, 2)
            η = view(H, :, i)
            LinearAlgebra.norm(D' * η) <= sqrt(eps(eltype(e.Σ))) * LinearAlgebra.norm(η) || return nothing
        end
    end
    return e.Σ * H, e.Σ, D
end

# The messages of every replica from shared parts, or `nothing` when the replicas do not share
# them. `get(n)` is channel n's inputs, `parts(x, d, k)` turns them into (M, Σ, D).
function _hard_shared_messages(fs, target, ps, k, get, parts)
    length(fs) == k || throw(DimensionMismatch("$(length(fs)) factors and $k inputs"))
    k > 1 || return nothing
    _within_gradient() && return nothing
    f = first(fs)
    all(h -> h === f || (h.dims == f.dims && isequal(h.Q, f.Q)), fs) || return nothing
    names = keys(f.dims)
    At = getfield(ps.A, target)
    eltype(At) <: Union{Float32,Float64} && eltype(ps.c) <: Union{Float32,Float64} || return nothing
    others = Tuple(n for n in names if n !== target)
    all(n -> eltype(getfield(ps.A, n)) <: Union{Float32,Float64}, others) || return nothing
    chans = Any[]
    for n in others
        p = parts(get(n), f.dims[n], k)
        p === nothing && return nothing
        push!(chans, p)
    end
    Q = _covariance_at(f.Q, ps, chans)
    return _hard_shared_kernel(Q, Matrix(At), copy(ps.c), others, ps.A, chans, k)
end

# Replica 1 through the path of `factor_message` (its decompositions are every replica's), then
# the linear map from the inputs' means to the message mean, applied to all replicas at once.
function _hard_shared_kernel(Q, At, c, others, A, chans, k)
    LC = LenticulumCore
    if isempty(others)
        msg = LC.preimage(LC.ExtendedGaussian(c, Q), At)
        return Any[msg for _ in 1:k]
    end
    joint1 = LC._independent_join([LC.ExtendedGaussian(M[:, 1], Σ, D) for (M, Σ, D) in chans])
    Mo = reduce(hcat, [-_matrix(getfield(A, n)) for n in others])
    r1 = LC.pushforward(joint1, Mo, c, Q)
    msg1 = LC.preimage(r1, At)
    # the pushforward's mean, every replica: Πr (Mo J + c), Πr onto the complement of its flat directions
    Ur = LC._complement(r1.D)
    J = reduce(vcat, [M for (M, _, _) in chans])
    μr = Ur * (Ur' * (Mo * J .+ c))
    # `preimage(r, At)`: the potential [C x = H'μr] · exp(ηᵀx − xᵀΛp x/2) with C = H'At, η = At'Λr μr
    H = LC.hard_directions(r1)
    _, Λr = LC._canonical_part(r1)
    C = H' * At
    Λp = At' * Λr * At
    nt = size(At, 2)
    if size(C, 1) == 0
        P, N = _zeros_like(Λp, eltype(Λp), nt, 0), Matrix{eltype(Λp)}(I, nt, nt)
    else
        U0, S0, V0, rk = LC._value_svd(C)
        P = V0[:, 1:rk] * (U0[:, 1:rk]' ./ S0[1:rk])
        N = V0[:, (rk + 1):nt]
        # each replica's constraints must have a common solution (`_from_potential`, strict)
        Z = H' * μr
        R = C * (P * Z) .- Z
        for i in 1:k
            zi, ri = view(Z, :, i), view(R, :, i)
            tol = sqrt(eps(eltype(μr))) * max(LinearAlgebra.norm(zi), LinearAlgebra.norm(view(μr, :, i)), one(eltype(μr)))
            LinearAlgebra.norm(ri) <= tol || throw(ArgumentError(
                "contradictory hard constraints: they fix a direction to different values \
                 (residual $(LinearAlgebra.norm(ri)), replica $i)"))
        end
    end
    # x = x0 + N Sz Nᵀ(η − Λp x0), x0 = P H'μr
    L = P * H'
    if size(N, 2) > 0
        Λz = N' * Λp * N
        V, K, _ = LC._psd_split((Λz + Λz') / 2, "the precision of a product")
        G = N * LC._split_pinv(V, K) * N'
        L = L .- G * (Λp * L) .+ G * (At' * Λr)
    end
    Uo = LC._complement(msg1.D)
    X = Uo * (Uo' * (L * μr))                    # normalised: onto the complement of the flat directions
    out = Vector{Any}(undef, k)
    out[1] = msg1
    for i in 2:k
        out[i] = LC.ExtendedGaussian(Val(:normalised), X[:, i], msg1.Σ, msg1.D)
    end
    return out
end

# --- Mean-field nodes --------------------------------------------------------

"""
    Mycelium.mean_field_message(f::HardConstraintFactor, target, polarity, beliefs, ps, st)
    Mycelium.mean_field_energy(f::HardConstraintFactor, marginals, ps, st)

With a positive definite noise the factor is `LinearConstraintFactor` with that noise, and these are
its methods. Otherwise refused, with the reason. The mean-field message ``\\exp \\mathbb E_q[\\log f]`` of an
exact relation is the limit of `LinearConstraintFactor`'s with noise ``\\varepsilon I \\to 0``: hard along
``\\operatorname{range}(A_t^\\top)``, at ``A_t x_t = c - \\sum_{i\\ne t} A_i \\mathbb E_q[x_i]``. Along those
directions the variable's marginal then sits exactly where the others' means put it, whatever its
prior says there, so an assignment that satisfies the relation is a fixed point and the iteration
does not leave it; and the average energy ``\\mathbb E_q[-\\log f]`` is infinite unless every
``A_i x_i`` is deterministic under ``q`` (`hard_constraint.md`, "Mean-field nodes"). A singular nonzero
noise is exact along its null space and refused alike. Keep the node on `BeliefPropagation()`
(exact on trees), or give the relation a positive definite noise.
"""
function Mycelium.mean_field_message(f::HardConstraintFactor, target::Symbol, polarity, beliefs, ps, st)
    return Mycelium.mean_field_message(_noisy_twin(f), target, polarity, beliefs, ps, st)
end
Mycelium.mean_field_energy(f::HardConstraintFactor, marginals, ps, st) =
    Mycelium.mean_field_energy(_noisy_twin(f), marginals, ps, st)

# The LinearConstraintFactor with the same channels, equations and (positive definite) noise; the
# parameters (A, c) are laid out alike.
function _noisy_twin(f::HardConstraintFactor)
    isposdef(Symmetric(f.Q)) || throw(ArgumentError(_HARD_MEAN_FIELD))
    return LinearConstraintFactor(f.dims, f.m; noise = f.Q)
end

const _HARD_MEAN_FIELD = "HardConstraintFactor without a positive definite noise has no mean-field \
    message: under an exact relation a MeanField() node puts each variable where the others' means \
    fix it, ignoring its prior along the directions the relation fixes, so the iteration stays where \
    it starts there, and its average energy is infinite. Keep the node on BeliefPropagation(), or \
    give the relation a positive definite noise"

Adapt.@adapt_structure HardConstraintFactor
