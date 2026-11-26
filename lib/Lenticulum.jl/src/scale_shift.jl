# ---------------------------------------------------------------------------
# ScaleShiftFactor: the normalising change of variables  z = (x − b) ⊘ s,  exactly (backlog U3).
#
# The density δ(x − s ⊙ z − b) of x given z (Lebesgue measure on x), with s = exp(logs) and b per
# coordinate or per group of coordinates. It is HardConstraintFactor's relation x − diag(s) z = b,
# whose exact messages it uses, and whose Bethe term carries the log-Jacobian Σᵢ log sᵢ through its
# ½ log det(G̃G̃ᵀ). See `scale_shift.md`.
# ---------------------------------------------------------------------------

"""
    ScaleShiftFactor(x::Symbol, z::Symbol, dim::Int; groups = 1:dim)

The exact affine change of variables ``z = (x - b) \\oslash s`` between two channels of dimension
`dim`, read as the density ``\\delta(x - s \\odot z - b)`` of ``x`` given ``z``: a standardising layer
(Glow's ActNorm, the affine part of batch and group normalisation) with its log-Jacobian. Parameters
`(b, logs)`, one entry per group, ``s = \\exp(\\text{logs})``; `groups[i]` is the group of coordinate
`i`, the groups are `1:G` and none is empty. `initialparameters` is the identity (`b = 0`,
`logs = 0`); [`scale_shift_parameters`](@ref) sets them from data. Freeze the node for a fixed map.

Polarities, messages and inputs as [`HardConstraintFactor`](@ref), whose relation
``x - \\operatorname{diag}(s)\\,z = b`` it is: any one channel `Unobserved()`, each other `Observed()`
or `Latent()`; [`ExtendedGaussian`](@ref) messages, exact for points, Gaussians, extended Gaussians
and no information; a mixture is collapsed to its moments.

Its Bethe term is `HardConstraintFactor`'s. Without hard directions in the inputs it is
``-H(q_z) = -H(q_x) + \\sum_i \\log s_i``, ``q`` the normalised product of the two inputs: with ``x``
observed and a prior ``p_z`` on ``z``, the graph's free energy is
``-\\log p_z((x-b)\\oslash s) + \\sum_i \\log s_i = -\\log p(x)``, so a learned scale cannot shrink
``x`` to raise the density of ``z`` (LP-022).

```julia
f  = ScaleShiftFactor(:x, :z, 4; groups = [1, 1, 2, 2])   # two groups of two coordinates
ps = scale_shift_parameters(f, X)                         # X: 4 × N data, z standardised per group
```
"""
struct ScaleShiftFactor{H<:HardConstraintFactor} <: LenticulumCore.AbstractLenticulumFactor
    x::Symbol
    z::Symbol
    dim::Int
    groups::Vector{Int}
    ngroups::Int
    hard::H
    ScaleShiftFactor(x, z, dim, groups, ngroups, hard::H) where {H<:HardConstraintFactor} =
        new{H}(x, z, dim, groups, ngroups, hard)
    # every argument is checked here, the dimension before its default groups (review of U3)
    function ScaleShiftFactor(x::Symbol, z::Symbol, dim; groups = dim isa Integer ? (1:dim) : nothing)
        x === z && throw(ArgumentError("ScaleShiftFactor needs two different channels; got :$x twice"))
        (dim isa Integer && !(dim isa Bool) && dim > 0) || throw(ArgumentError("dim must be a positive integer; got $dim"))
        (groups isa AbstractVector && all(g -> g isa Integer && !(g isa Bool), groups)) ||
            throw(ArgumentError("groups must be a vector of integers; got $(repr(groups))"))
        length(groups) == dim || throw(DimensionMismatch("groups has length $(length(groups)), dim is $dim"))
        G = maximum(groups)
        (minimum(groups) >= 1 && Set(groups) == Set(1:G)) ||
            throw(ArgumentError("the groups must be 1:G with none empty; got $(sort(unique(groups)))"))
        hard = HardConstraintFactor(NamedTuple{(x, z)}((Int(dim), Int(dim))), Int(dim))
        return new{typeof(hard)}(x, z, Int(dim), collect(Int, groups), G, hard)
    end
end

LuxCore.initialparameters(::AbstractRNG, f::ScaleShiftFactor) = (b = zeros(f.ngroups), logs = zeros(f.ngroups))
LuxCore.initialstates(::AbstractRNG, ::ScaleShiftFactor) = NamedTuple()
LuxCore.parameterlength(f::ScaleShiftFactor) = 2 * f.ngroups

LenticulumCore.channels(f::ScaleShiftFactor) = (LenticulumCore.Channel(f.x, f.dim), LenticulumCore.Channel(f.z, f.dim))
LenticulumCore.supported_polarities(f::ScaleShiftFactor) = _one_unobserved_polarities((f.x, f.z))
LenticulumCore.hasnopolarity(::ScaleShiftFactor) = false
LenticulumCore.isunidirectional(::ScaleShiftFactor) = false
LenticulumCore.supports_polarity(f::ScaleShiftFactor, p::LenticulumCore.Polarity) =
    LenticulumCore.supports_polarity(f.hard, p)
LenticulumCore.islearnable(::ScaleShiftFactor) = true
Mycelium.exact_messages(::ScaleShiftFactor) = true
Mycelium.repeatable_messages(::ScaleShiftFactor) = true
Mycelium.uses_cavity(::ScaleShiftFactor) = true
coupling_precision(::ScaleShiftFactor, ps) = :hard
LenticulumCore.energyspace(f::ScaleShiftFactor) = LenticulumCore.energyspace(f.hard)
LenticulumCore.scalarisation(f::ScaleShiftFactor) = LenticulumCore.scalarisation(f.hard)

# The relation's parameters for HardConstraintFactor: A_x = I, A_z = −diag(s), c = b, per coordinate,
# in the promoted float type of (b, logs). Written without mutation, so that AD goes through.
function _scale_shift_relation(f::ScaleShiftFactor, ps, inputs::Vararg{Any,N}) where {N}
    s, b = _scale_shift_map(f, ps, inputs...)
    A = NamedTuple{(f.x, f.z)}((_eye(b, eltype(b), f.dim), -_matrix(Diagonal(s))))
    return (A = A, c = b)
end

# s = exp(logs) and b per coordinate, in the working type of the call: the floating promotion of
# (b, logs) and the inputs — integer b/logs do not widen a Float32 call (LP-094 finding 7).
function _scale_shift_map(f::ScaleShiftFactor, ps, inputs::Vararg{Any,N}) where {N}
    length(ps.b) == f.ngroups && length(ps.logs) == f.ngroups || throw(DimensionMismatch(
        "ScaleShiftFactor has $(f.ngroups) group(s); got b of length $(length(ps.b)), logs of length $(length(ps.logs))"))
    T = _compute_eltype(Float64, ps.b, ps.logs, inputs...)
    return _at_eltype(T, exp.(ps.logs[f.groups])), _at_eltype(T, ps.b[f.groups])
end

"""
    Mycelium.factor_message(f::ScaleShiftFactor, target, polarity, inputs, prior, ps, st)

The exact message of ``x - \\operatorname{diag}(s)\\,z = b``: to ``z`` the input of ``x`` mapped by
``(x - b) \\oslash s``, to ``x`` the input of ``z`` mapped by ``s \\odot z + b``, as an
[`ExtendedGaussian`](@ref) (mean, covariance and flat directions mapped coordinatewise).
"""
function Mycelium.factor_message(f::ScaleShiftFactor, target::Symbol, polarity, inputs, prior, ps, st)
    target === f.x || target === f.z ||
        throw(ArgumentError("channel :$target is not a channel of this factor ($((f.x, f.z)))"))
    return _scale_shift_message(f, target, inputs, ps), st
end

# The map applied directly to the input's mean, covariance and flat directions: an invertible
# diagonal map is exact on each, and the constructor renormalises (orthonormal flat directions, the
# mean and covariance on their complement). HardConstraintFactor's pushforward and preimage would
# reach the same message through eigenvalue cuts that took a Float32 variance of 1e-6 next to 1 for
# a hard direction (review of U3, finding 2), at O(d³) instead of O(d²).
function _scale_shift_message(f::ScaleShiftFactor, target::Symbol, inputs, ps)
    s, b = _scale_shift_map(f, ps, inputs)
    m = _split_input(inputs)(target === f.x ? f.z : f.x)
    e = _no_information(m) ? _no_information_like(s, f.dim) : _as_extended(m, f.dim, eltype(s))
    length(e.μ) == f.dim || throw(DimensionMismatch("ScaleShiftFactor has dimension $(f.dim), its input $(length(e.μ))"))
    if target === f.z
        μ, Σ, D = (e.μ .- b) ./ s, e.Σ ./ (s .* s'), e.D ./ s
    else
        μ, Σ, D = s .* e.μ .+ b, (s .* s') .* e.Σ, s .* e.D
    end
    return LenticulumCore.ExtendedGaussian(μ, (Σ + Σ') / 2, D)
end

# No information as an extended Gaussian stored like `s`: `extended(nothing, …)` allocates host
# arrays, and a CUDA broadcast of those against a device `s` fails (JLArrays accepts it; Q8).
function _no_information_like(s, d)
    T = eltype(s)
    return LenticulumCore.ExtendedGaussian(LenticulumCore._zeros_like(s, T, d),
        LenticulumCore._zeros_like(s, T, d, d), LenticulumCore._eye(s, T, d))
end

# The relation in the parameters' float type: HardConstraintFactor takes its element type from its
# (zero) noise, stored as Float64 in `f.hard`, which would widen Float32 messages (P30). A dual
# number is not an AbstractFloat and keeps `f.hard`.
_scale_shift_hard(f::ScaleShiftFactor, ::Type{T}) where {T} =
    T <: AbstractFloat && T !== Float64 ? HardConstraintFactor(NamedTuple{(f.x, f.z)}((f.dim, f.dim)), f.dim; noise = zero(T)) : f.hard

"""
    Mycelium.local_free_energy(f::ScaleShiftFactor, msgs, ps, st)

`HardConstraintFactor`'s Bethe term for ``x - \\operatorname{diag}(s)\\,z = b``:
``\\tfrac12\\log\\det(\\tilde G\\tilde G^\\top) - H_R(b_f)``. With ``x`` a point it is
``\\sum_i \\log s_i`` (the log-Jacobian); without hard directions in the inputs, ``-H(q_z)``. Graded
as `HardConstraintFactor` (`fit` 0). See `scale_shift.md`, "Free energy".
"""
function Mycelium.local_free_energy(f::ScaleShiftFactor, msgs, ps, st)
    rel = _scale_shift_relation(f, ps, msgs)
    return first(Mycelium.local_free_energy(_scale_shift_hard(f, eltype(rel.c)), msgs, rel, NamedTuple())), st
end

# ---------------------------------------------------------------------------
# The parameter gradient. A deterministic factor has no envelope gradient at a fixed belief: its
# belief lives on the relation, which moves with (b, s), and the beliefs of its neighbours are
# moved with it (a point z* = (x − b) ⊘ s). Holding the incoming messages m fixed instead is right
# for −log Z_f(m, θ) = −log ∫ f m_x m_z, the factor's term in the message form of the Bethe free
# energy, which is stationary in the messages at a fixed point:  dF/dθ = −∂_θ log Z_f.
# And −log Z_f = F_f + E_{b_f}[−log m_x − log m_z] (up to constants of the messages), so the term
# differentiated is F_f plus the inputs' energies under the factor belief, minus their value at
# the current θ: the value stays F_f, the derivative is −∂_θ log Z_f. See `scale_shift.md`.
# ---------------------------------------------------------------------------

"""
    ScaleShiftBelief

What `Mycelium.factor_belief` returns for a [`ScaleShiftFactor`](@ref): the incoming messages and
the inputs' energies under the factor belief at the parameters it was taken at. Its
`belief_free_energy` has the value of the factor's Bethe term and the derivative of
``-\\log Z_f`` at fixed messages, the graph free energy's derivative at a fixed point.
"""
struct ScaleShiftBelief{M,T}
    msgs::M
    energy::T
end

# E_{b_f}[−log m_x − log m_z] over the Gaussian directions of the inputs (a point, a hard
# direction, a flat direction or no input adds a constant): ½ tr(ΛC) + ½ δᵀΛδ per input, Λ its
# precision on those directions, (μ, C) the factor belief's marginal on that channel and δ = μ − μₘ
# its offset from the input's mean. Centred on μₘ: the uncentred ½ μᵀΛμ − ηᵀμ lost the change of
# the energy to cancellation at a mean of 1e8 (review of U3, finding 3).
function _scale_shift_input_energy(f::ScaleShiftFactor, msgs, ps)
    T = _compute_eltype(eltype(ps.b), ps, msgs)
    input = _split_input(msgs)
    total = zero(T)
    for n in (f.x, f.z)
        m = input(n)
        (_no_information(m) || m isa LenticulumCore.DiracBelief) && continue
        e = _as_extended(m, f.dim, T)
        _, Λ = LenticulumCore._canonical_part(e)   # the messages are constants of θ
        iszero(Λ) && continue
        q = Mycelium.combine(e, _scale_shift_message(f, n, msgs, ps))
        δ = q.μ .- e.μ
        total += (sum(Λ .* q.Σ) + LinearAlgebra.dot(δ, Λ * δ)) / 2
    end
    return total
end

function Mycelium.factor_belief(f::ScaleShiftFactor, msgs, ps, st)
    energy = _scale_shift_input_energy(f, msgs, ps)
    return ScaleShiftBelief(msgs, oftype(energy, LenticulumCore.stop_gradient(energy)))
end

function Mycelium.belief_free_energy(f::ScaleShiftFactor, b::ScaleShiftBelief, ps, st)
    F, _ = Mycelium.local_free_energy(f, b.msgs, ps, st)
    e = _scale_shift_input_energy(f, b.msgs, ps) - b.energy
    return LenticulumCore.GradedEnergy((fit = F.parts.fit + e, complexity = F.parts.complexity,
                                        negentropy = F.parts.negentropy)), st
end

Mycelium.mean_field_message(f::ScaleShiftFactor, target::Symbol, polarity, beliefs, ps, st) =
    Mycelium.mean_field_message(f.hard, target, polarity, beliefs, _scale_shift_relation(f, ps, beliefs), st)

"""
    scale_shift_parameters(f::ScaleShiftFactor, X::AbstractMatrix) -> (b, logs)

Parameters that standardise the data `X` (`dim × N`, one sample per column) per group: `b` the
mean and `s` the standard deviation (normalised by `N`) of the group's entries over all its
coordinates and samples, so that ``z = (x - b) \\oslash s`` has mean 0 and variance 1 per group
(Glow's data-dependent initialisation of ActNorm). A group whose entries are all equal has no
scale and throws.
"""
function scale_shift_parameters(f::ScaleShiftFactor, X::AbstractMatrix{<:Real})
    size(X, 1) == f.dim || throw(DimensionMismatch("X has $(size(X, 1)) rows, the factor dimension $(f.dim)"))
    size(X, 2) >= 1 || throw(ArgumentError("X has no samples"))
    T = float(eltype(X))
    device = _device_leaf(X)
    b, logs = device === nothing ? (similar(X, T, f.ngroups), similar(X, T, f.ngroups)) :
                                  (Vector{T}(undef, f.ngroups), Vector{T}(undef, f.ngroups))
    for g in 1:f.ngroups
        # in floating point from the start (an Int64 sum overflowed), the deviations scaled by
        # their largest before squaring (1e±200 overflowed or underflowed), and equal entries
        # caught exactly, not through a rounded spread (review of U3, finding 5)
        rows = device === nothing ? f.groups .== g : findall(==(g), f.groups)
        v = T.(vec(X[rows, :]))
        (device === nothing ? allequal(v) : minimum(v) == maximum(v)) && throw(ArgumentError("group $g of the data has no spread; its scale is undefined"))
        m = sum(v) / length(v)
        a = maximum(abs, v .- m)
        σ = a * sqrt(sum(abs2, (v .- m) ./ a) / length(v))
        σ > 0 || throw(ArgumentError("group $g of the data has no spread; its scale is undefined"))
        b[g], logs[g] = m, log(σ)
    end
    return (b = _on_device(device, b), logs = _on_device(device, logs))
end

Adapt.@adapt_structure ScaleShiftBelief
Adapt.adapt_structure(to, f::ScaleShiftFactor) =
    ScaleShiftFactor(f.x, f.z, f.dim, f.groups, f.ngroups, Adapt.adapt(to, f.hard))
