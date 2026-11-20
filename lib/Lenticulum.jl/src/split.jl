# ---------------------------------------------------------------------------
# SplitFactor: a channel and pieces of it,  x_k = z[I_k],  exactly (backlog U2).
#
# The product of Dirac deltas ∏ₖ δ(xₖ − z[Iₖ]). Every message is computed on z: each piece's input
# is pulled back to z through its selection (flat off Iₖ), the pullbacks and z's own input are
# multiplied, and a piece's message is the marginal of that product on its coordinates. All three
# steps are exact for extended Gaussians, so the factor is exact for points, Gaussians and no
# information. Its free energy is HardConstraintFactor's for the relation S z − x = 0. See `split.md`.
# ---------------------------------------------------------------------------

"""
    SplitFactor(whole::Symbol, pieces::NamedTuple; dim = maximum index)

The exact relation ``x_k = z[I_k]`` between the channel `whole` (``z``, of dimension `dim`) and each
piece channel ``x_k``, given as `pieces = (x = I₁, y = I₂, …)` with index sets into `1:dim`: a
concatenation (pieces that partition `1:dim` in order), a split, a permutation, a selection.
Pieces may overlap (a coordinate copied into two pieces), and a coordinate that no piece covers is
left free (the factor says nothing about it). No noise and no parameters.

Polarities as [`HardConstraintFactor`](@ref): any one channel `Unobserved()`, each other
`Observed()` or `Latent()`. Its messages are [`ExtendedGaussian`](@ref)s, the exact sum-product
messages for points, Gaussians (also improper), extended Gaussians and no information; a mixture
is collapsed to its moments first. The message to a piece is the marginal on ``I_k`` of
``m_z(z) \\prod_{j \\ne k} m_j(z[I_j])``; the message to `whole` is ``\\prod_k m_k(z[I_k])``.

Its Bethe term is that of the relation ``S z - x = 0`` as `HardConstraintFactor` computes it (the
limit of the noisy relation). When no input has hard directions it is ``-H(q)``, ``q`` the
normalised product of all inputs on ``z``: [`RelayFactor`](@ref)'s term, which
`SplitFactor(:z, (x = 1:d,))` reproduces for proper Gaussian inputs. Unlike `RelayFactor`, it
collapses a mixture input to its moments.

```julia
# z ∈ ℝ⁵ as a position (1:2) and a velocity (3:5)
f = SplitFactor(:z, (pos = 1:2, vel = 3:5))
```
"""
struct SplitFactor{names,I<:Tuple} <: LenticulumCore.AbstractLenticulumFactor
    whole::Symbol
    pieces::NamedTuple{names,I}
    dim::Int
    # every constructor goes through the checks (review of U2, finding 7)
    function SplitFactor(whole::Symbol, pieces::NamedTuple, dim::Integer)
        isempty(pieces) && throw(ArgumentError("SplitFactor needs at least one piece"))
        whole in keys(pieces) && throw(ArgumentError("the whole :$whole is also the name of a piece"))
        (dim isa Bool || dim <= 0) && throw(ArgumentError("dim must be a positive integer; got $dim"))
        idx = map(pieces) do I
            (I isa AbstractVector{<:Integer} && !(eltype(I) <: Bool)) ||
                throw(ArgumentError("a piece's index set must be a vector of integers; got $(typeof(I))"))
            isempty(I) && throw(ArgumentError("a piece's index set is empty"))
            allunique(I) || throw(ArgumentError("a piece's index set repeats an index: $(I)"))
            all(i -> 1 <= i <= dim, I) || throw(ArgumentError("piece indices must lie in 1:$dim; got $(I)"))
            collect(Int, I)
        end
        return new{keys(idx),typeof(values(idx))}(whole, idx, Int(dim))
    end
end

function SplitFactor(whole::Symbol, pieces::NamedTuple; dim = nothing)
    if dim === nothing
        isempty(pieces) && throw(ArgumentError("SplitFactor needs at least one piece"))
        all(I -> I isa AbstractVector{<:Integer} && !isempty(I), values(pieces)) ||
            throw(ArgumentError("every piece needs a non-empty vector of integer indices"))
        dim = maximum(maximum, values(pieces))
    end
    (dim isa Integer && !(dim isa Bool)) || throw(ArgumentError("dim must be a positive integer; got $dim"))
    dim <= typemax(Int) || throw(ArgumentError("dim = $dim exceeds typemax(Int)"))
    return SplitFactor(whole, pieces, Int(dim))
end

LuxCore.initialparameters(::AbstractRNG, ::SplitFactor) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::SplitFactor) = NamedTuple()
LuxCore.parameterlength(::SplitFactor) = 0

_split_names(f::SplitFactor{names}) where {names} = (f.whole, names...)
LenticulumCore.channels(f::SplitFactor{names}) where {names} =
    (LenticulumCore.Channel(f.whole, f.dim), map((n, I) -> LenticulumCore.Channel(n, length(I)), names, values(f.pieces))...)
LenticulumCore.supported_polarities(f::SplitFactor) = _one_unobserved_polarities(_split_names(f))
LenticulumCore.hasnopolarity(::SplitFactor) = false
LenticulumCore.isunidirectional(::SplitFactor) = false
function LenticulumCore.supports_polarity(f::SplitFactor, p::LenticulumCore.Polarity)
    Set(keys(p)) == Set(_split_names(f)) || return false
    length(LenticulumCore.unobserved_channels(p)) == 1 || return false
    return all(n -> p[n] isa Union{LenticulumCore.Unobserved,LenticulumCore.Observed,LenticulumCore.Latent}, _split_names(f))
end
LenticulumCore.islearnable(::SplitFactor) = false
Mycelium.exact_messages(::SplitFactor) = true
coupling_precision(::SplitFactor, ps) = :hard
# the grading of `HardConstraintFactor`, whose free energy it has (`local_free_energy` below)
LenticulumCore.energyspace(::SplitFactor) = LenticulumCore.energyspace(HardConstraintFactor((x = 1,), 1))
LenticulumCore.scalarisation(::SplitFactor) = LenticulumCore.scalarisation(HardConstraintFactor((x = 1,), 1))

# The selection matrix S with S z = z[idx], built by indexing an identity.
_selection(idx::AbstractVector{Int}, d::Int, ::Type{T}) where {T} = Matrix{T}(LinearAlgebra.I, d, d)[idx, :]

_split_input(inputs) = n -> haskey(inputs, n) ? getfield(inputs, n) : nothing
_no_information(b) = b === nothing || b isa LenticulumCore.TrivialBelief

# The float type of the inputs that take part (`skip` left out): their promotion, Float64 when
# none is informative (P30). The input left out does not set it (review of U2, finding 3).
function _split_eltype(f::SplitFactor, input, skip)
    T = Union{}
    for n in _split_names(f)
        n === skip && continue
        b = input(n)
        _no_information(b) && continue
        T = promote_type(T, eltype(_as_extended(b, 1, Float64).μ))
    end
    return T === Union{} ? Float64 : float(T)
end

# The product on z of the inputs other than `skip` (a channel name, or `nothing` for none: the
# free energy), each piece pulled back through its selection. Pieces with pairwise disjoint index
# sets are joined in covariance form (block-diagonal, permuted into place), which multiplies
# nothing and so cuts no precision: in precision form, a Float32 product of pieces with variances 1
# and 1e6 lost the second to the eigenvalue tolerance (review of U2, finding 3). Overlapping pieces
# are multiplied as preimages (`combine`).
function _split_on_whole(f::SplitFactor{names}, input, skip) where {names}
    T = _split_eltype(f, input, skip)
    used = Tuple(n for n in names if n !== skip && !_no_information(input(n)))
    es = map(used) do n
        e = _as_extended(input(n), length(f.pieces[n]), T)
        length(e.μ) == length(f.pieces[n]) || throw(DimensionMismatch(
            "piece :$n has dimension $(length(f.pieces[n])), its input $(length(e.μ))"))
        e
    end
    covered = reduce(vcat, map(n -> f.pieces[n], used); init = Int[])
    q = if allunique(covered)
        rest = setdiff(1:f.dim, covered)
        blocks = isempty(rest) ? collect(es) : [es..., LenticulumCore.extended(nothing, length(rest), T)]
        isempty(blocks) ? LenticulumCore.extended(nothing, f.dim, T) :
            LenticulumCore.pushforward(LenticulumCore._independent_join(blocks),
                                       Matrix{T}(LinearAlgebra.I, f.dim, f.dim)[:, vcat(covered, rest)])
    else
        foldl(Mycelium.combine, [LenticulumCore.preimage(e, _selection(f.pieces[n], f.dim, T)) for (n, e) in zip(used, es)])
    end
    if skip !== f.whole && !_no_information(input(f.whole))
        ez = _as_extended(input(f.whole), f.dim, T)
        length(ez.μ) == f.dim || throw(DimensionMismatch("the whole :$(f.whole) has dimension $(f.dim), its input $(length(ez.μ))"))
        q = LenticulumCore.isuninformative(q) ? ez : Mycelium.combine(ez, q)
    end
    return q
end

# The marginal of q on idx, with its hard directions judged against q. `marginalise` alone judges
# them on the scale of the marginal: a piece that only coordinates fixed by an observation reach has
# a covariance of pure rounding (1e-31 for a q of scale 1, tree 6 of the oracle test), whose
# eigenvalues a relative tolerance takes for variances. That rounding is second order in the
# precision (a basis of q's free directions off by ε leaves ε² on its hard coordinates), so the cut
# is at the square of q's tolerance over its scale: a true variance of 1e-16 against 1 is kept
# (review of U2, finding 2). On values only; a derivative through the cut is not defined.
function _split_marginal(q::LenticulumCore.ExtendedGaussian, idx)
    m = LenticulumCore.marginalise(q, idx)
    isempty(m.Σ) && return m
    λq = LinearAlgebra.eigvals(Symmetric(LenticulumCore.stop_gradient(q.Σ)))
    scale = LenticulumCore._scale(λq)
    iszero(scale) && return m
    tol = LenticulumCore._eig_tol(λq)^2 / scale
    E = LinearAlgebra.eigen(Symmetric(LenticulumCore.stop_gradient(m.Σ)))
    small = E.values .<= tol
    count(small) > size(LenticulumCore.hard_directions(m), 2) || return m
    V = E.vectors[:, .!small]
    Σ = V * Diagonal(E.values[.!small]) * V'
    return LenticulumCore.ExtendedGaussian(m.μ, (Σ + Σ') / 2, m.D)
end

"""
    Mycelium.factor_message(f::SplitFactor, target, polarity, inputs, prior, ps, st)

To a piece ``k``: the marginal on ``I_k`` of ``m_z(z)\\prod_{j\\ne k} m_j(z[I_j])``. To the whole:
``\\prod_k m_k(z[I_k])``, flat along the coordinates no informative piece covers. An
[`ExtendedGaussian`](@ref) either way.
"""
function Mycelium.factor_message(f::SplitFactor{names}, target::Symbol, polarity, inputs, prior, ps, st) where {names}
    target === f.whole || target in names ||
        throw(ArgumentError("channel :$target is not a channel of this factor ($(_split_names(f)))"))
    q = _split_on_whole(f, _split_input(inputs), target)
    target === f.whole && return q, st
    return _split_marginal(q, f.pieces[target]), st
end

Mycelium.uses_cavity(::SplitFactor) = true

# The relation as a `HardConstraintFactor`: S z − x = 0, S the stacked selections, one block −I per
# piece, in the element type T.
function _split_relation(f::SplitFactor{names}, ::Type{T}) where {names, T}
    ns = map(n -> length(f.pieces[n]), names)
    m = sum(ns)
    rows = cumsum((0, ns...))
    A_whole = reduce(vcat, map(n -> _selection(f.pieces[n], f.dim, T), names))
    A_pieces = ntuple(k -> vcat(zeros(T, rows[k], ns[k]), -Matrix{T}(LinearAlgebra.I, ns[k], ns[k]),
                                 zeros(T, m - rows[k + 1], ns[k])), length(names))
    hard = HardConstraintFactor(NamedTuple{(f.whole, names...)}((f.dim, ns...)), m)
    ps = (A = NamedTuple{(f.whole, names...)}((A_whole, A_pieces...)), c = zeros(T, m))
    return hard, ps
end

"""
    Mycelium.local_free_energy(f::SplitFactor, msgs, ps, st)

The Bethe term of the relation ``S z - x = 0``, the limit of the noisy relation's term as the noise
vanishes. When no input has a hard direction (no point, no extended Gaussian with a hard
direction) it is ``-H(q)``, ``q`` the normalised product of the incoming messages on ``z``
(`RelayFactor`'s term; the half-log-determinants of the relation and of its solution set cancel),
computed from ``q`` in covariance form. Otherwise it is [`HardConstraintFactor`](@ref)'s
``\\tfrac12 \\log\\det(\\tilde G \\tilde G^\\top) - H_R(b_f)``, which keeps the Jacobian that a hard
direction oblique to the coordinates leaves (review of U2, finding 8). Graded as
`HardConstraintFactor`. Throws where that factor does: a factor belief flat along the relation (an
improper posterior), or an equation on a direction that two inputs both fix (an infinite free
energy). See `split.md`, "Free energy".
"""
function Mycelium.local_free_energy(f::SplitFactor, msgs, ps, st)
    input = _split_input(msgs)
    if !any(n -> _has_hard(input(n)), _split_names(f))
        q = _split_on_whole(f, input, nothing)
        size(q.D, 2) == 0 || throw(ArgumentError(
            "SplitFactor: the factor belief is flat along $(size(q.D, 2)) direction(s) of :$(f.whole) \
             (an improper posterior); it has no free energy"))
        H = Mycelium.variable_entropy(q)
        return LenticulumCore.GradedEnergy((fit = false, complexity = zero(H), negentropy = -H)), st
    end
    hard, hps = _split_relation(f, _split_eltype(f, input, nothing))
    F, _ = Mycelium.local_free_energy(hard, msgs, hps, NamedTuple())
    return F, st
end
_has_hard(b) = false
_has_hard(::LenticulumCore.DiracBelief) = true
_has_hard(b::LenticulumCore.ExtendedGaussian) = size(LenticulumCore.hard_directions(b), 2) > 0
