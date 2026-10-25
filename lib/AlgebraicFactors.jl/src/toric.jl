"""
    ToricFactor(names::Tuple, alphabets::Tuple, A::AbstractMatrix{<:Integer};
                normalised = false, labels = nothing)

Finite potential `exp(dot(ps.η, A[:, x]))`, with one channel per name. Joint states
follow `CartesianIndices(alphabets)`: the first variable varies fastest. `A` has one
column per joint state; its rows need not be independent or contain a constant.
Parameters are `(η = zeros(size(A, 1)),)` initially (Float64); state is empty.
The normalised distributions agree exactly when `A' * (η - η′)` is constant.

States are `1:alphabets[i]`, or `labels[i]` when a tuple of unique-label vectors is
provided. Pointwise tuples and Diracs match labels by `isequal`, never by interpreting
an integer label as a position. Unlabelled categorical inputs are positional; labelled
ones must have exactly the channel's support and are reordered by label. Outgoing
categoricals use the factor's label order (`labels = nothing` for default states).

Every polarity with one unobserved channel and all others observed or latent is supported.
Missing, `nothing`, `missing`, `TrivialBelief()` and `Latent()` inputs have unit weight.
Messages exclude both the target input and its prior. Enumeration visits all
`prod(alphabets)` states; scoring also costs `size(A, 1)` per state. No fitting is provided.
"""
struct ToricFactor{N,S,M,L} <: LenticulumCore.AbstractLenticulumFactor
    names::N
    alphabets::S
    A::M
    labels::L
    customlabels::Bool
    normalised::Bool
    function ToricFactor(names::Tuple, alphabets::Tuple, A::AbstractMatrix{<:Integer};
                         normalised::Bool = false, labels = nothing)
        sizes, nstates = _sizes(names, alphabets)
        size(A, 2) == nstates || throw(ArgumentError("A needs $nstates columns"))
        labs = if labels === nothing
            map(n -> collect(1:n), sizes)
        else
            labels isa Tuple && length(labels) == length(names) ||
                throw(ArgumentError("labels must be a tuple with one vector per channel"))
            all(i -> labels[i] isa AbstractVector && length(labels[i]) == sizes[i] &&
                     allunique(labels[i]), eachindex(names)) ||
                throw(ArgumentError("each label vector must have the alphabet's size and unique labels"))
            map(collect, labels)
        end
        mat = copy(A)
        new{typeof(names),typeof(sizes),typeof(mat),typeof(labs)}(
            names, sizes, mat, labs, labels !== nothing, normalised)
    end
end

function _sizes(names, alphabets)
    !isempty(names) && all(n -> n isa Symbol, names) && allunique(names) ||
        throw(ArgumentError("names must be nonempty, unique Symbols"))
    length(names) == length(alphabets) || throw(ArgumentError("one alphabet per name is required"))
    all(n -> n isa Integer && !(n isa Bool) && n > 0, alphabets) ||
        throw(ArgumentError("alphabet sizes must be positive integers"))
    try
        sizes = map(Int, alphabets)
        return sizes, foldl(Base.checked_mul, sizes; init = 1)
    catch err
        err isa Union{InexactError,OverflowError} || rethrow()
        throw(ArgumentError("joint alphabet size does not fit in Int"))
    end
end

"""
    hierarchical(names, alphabets, generators) -> Matrix{Int}

Design matrix with one block of marginal-cell indicator rows per generator, in the
supplied order. Each generator is a tuple of distinct channel names or 1-based indices.
Within a block, cells follow the generator's order, first listed variable fastest.
Columns follow `CartesianIndices(alphabets)`. Empty generators give a constant row;
an empty generator collection gives a zero-row (uniform) model. Redundant blocks are kept.
"""
function hierarchical(names::Tuple, alphabets::Tuple, generators)
    sizes, nstates = _sizes(names, alphabets)
    blocks = map(collect(generators)) do gen
        gen isa Tuple || throw(ArgumentError("each generator must be a tuple"))
        inds = map(gen) do g
            i = g isa Symbol ? findfirst(isequal(g), names) :
                g isa Integer && !(g isa Bool) ? g : nothing
            i !== nothing && 1 <= i <= length(names) ||
                throw(ArgumentError("unknown generator channel $g"))
            Int(i)
        end
        allunique(inds) || throw(ArgumentError("generator channels must be distinct"))
        inds
    end
    rows = [prod(i -> sizes[i], inds; init = 1) for inds in blocks]
    A = zeros(Int, sum(rows; init = 0), nstates)
    offset = 0
    for (inds, nrows) in zip(blocks, rows)
        dims = map(i -> sizes[i], inds)
        for (col, state) in enumerate(CartesianIndices(sizes))
            cell = LinearIndices(dims)[map(i -> state[i], inds)...]
            A[offset + cell, col] = 1
        end
        offset += nrows
    end
    return A
end

"""
    independence(names, alphabets) -> Matrix{Int}

The hierarchical design with singleton generators in channel order. For a 2×3 table
there are five rows: two indicators for the first variable, then three for the second.
"""
independence(names::Tuple, alphabets::Tuple) =
    hierarchical(names, alphabets, map(n -> (n,), names))

LenticulumCore.channels(f::ToricFactor) =
    map((n, s) -> LenticulumCore.Channel(n, s), f.names, f.alphabets)
LenticulumCore.islearnable(::ToricFactor) = true
LuxCore.initialparameters(::AbstractRNG, f::ToricFactor) = (η = zeros(size(f.A, 1)),)
LuxCore.initialstates(::AbstractRNG, ::ToricFactor) = NamedTuple()
LuxCore.parameterlength(f::ToricFactor) = size(f.A, 1)
LuxCore.statelength(::ToricFactor) = 0

function LenticulumCore.supported_polarities(f::ToricFactor)
    n = length(f.names)
    return Tuple(LenticulumCore.Polarity(NamedTuple{f.names}(ntuple(n) do i
        i == t ? LenticulumCore.Unobserved() :
        iszero(mask & (1 << (i < t ? i - 1 : i - 2))) ?
        LenticulumCore.Latent() : LenticulumCore.Observed()
    end)) for t in 1:n for mask in 0:(2^(n - 1) - 1))
end
function LenticulumCore.supports_polarity(f::ToricFactor, p::LenticulumCore.Polarity)
    keys(p) == f.names && length(LenticulumCore.unobserved_channels(p)) == 1 &&
        all(n -> p[n] isa Union{LenticulumCore.Observed,LenticulumCore.Unobserved,
                               LenticulumCore.Latent}, f.names)
end

function _logits(f, ps)
    ps.η isa AbstractVector{<:Real} && length(ps.η) == size(f.A, 1) ||
        throw(ArgumentError("ps.η must be a real vector with one entry per row of A"))
    all(isfinite, ps.η) || throw(ArgumentError("ps.η must be finite"))
    l = float.(transpose(f.A) * ps.η)
    all(isfinite, l) || throw(ArgumentError("logits must be finite"))
    return l
end

function _lognormalise(l)
    m = maximum(l)
    m == -Inf && throw(ArgumentError("evidence has zero weight on every joint state"))
    shifted = l .- m
    return shifted .- log(sum(exp, shifted))
end
_outlabels(f, i) = f.customlabels ? f.labels[i] : nothing
function _labelindex(f, i, x)
    k = findfirst(isequal(x), f.labels[i])
    k === nothing && throw(ArgumentError("unknown label $x on channel :$(f.names[i])"))
    return k
end

"""
    LenticulumCore.energy(f::ToricFactor, x::Tuple, a, y, ps, st)

Pointwise scalar energy `-dot(ps.η, a_x)` (plus log partition when normalised).
`x` contains all channel labels in channel order; `a` and `y` are unused. The shorter
`energy(f, x, ps, st)` is equivalent. Both return `(energy, st)`.
"""
function LenticulumCore.energy(f::ToricFactor, x::Tuple, a, y, ps, st)
    length(x) == length(f.names) || throw(ArgumentError("one label per channel is required"))
    inds = map(i -> _labelindex(f, i, x[i]), eachindex(f.names))
    col = LinearIndices(f.alphabets)[inds...]
    l = _logits(f, ps)
    return (f.normalised ? -_lognormalise(l)[col] : -l[col], st)
end
LenticulumCore.energy(f::ToricFactor, x::Tuple, ps, st) =
    LenticulumCore.energy(f, x, nothing, nothing, ps, st)

_incoming(msgs::NamedTuple, n) = get(msgs, n, nothing)
_incoming(::Nothing, n) = nothing
function _weights(f, i, b, T)
    if b === nothing || b === missing || b isa Union{LenticulumCore.TrivialBelief,LenticulumCore.Latent}
        return zeros(T, f.alphabets[i])
    elseif b isa LenticulumCore.DiracBelief
        k = _labelindex(f, i, b.value)
        return [j == k ? zero(T) : T(-Inf) for j in 1:f.alphabets[i]]
    elseif b isa LenticulumCore.CategoricalBelief
        length(b.logp) == f.alphabets[i] || throw(ArgumentError("categorical alphabet size mismatch"))
        b.labels === nothing && return b.logp
        allunique(b.labels) || throw(ArgumentError("categorical labels must be unique"))
        inds = [_labelindex(f, i, x) for x in b.labels]
        return b.logp[invperm(inds)]
    end
    throw(ArgumentError("expected Dirac, categorical, or latent input on :$(f.names[i]); got $(typeof(b))"))
end

function _joint_logweights(f, msgs, l; skip = 0)
    weights = map(eachindex(f.names)) do i
        i == skip ? zeros(eltype(l), f.alphabets[i]) :
        _weights(f, i, _incoming(msgs, f.names[i]), eltype(l))
    end
    # Remove a common logit before adding evidence: even a huge constant potential
    # must not round away finite log-probabilities supplied by incoming messages.
    m = maximum(l)
    return [(l[k] - m) + sum(i -> weights[i][x[i]], eachindex(f.names))
            for (k, x) in enumerate(CartesianIndices(f.alphabets))]
end

"""
    Mycelium.factor_message(f::ToricFactor, target, polarity, inputs, prior, ps, st)

Exact sum-product site on `target`, integrating every other channel. `prior` and any
target entry in `inputs` are excluded. `normalised` changes only a constant and hence
does not affect the returned categorical probabilities. Returns `(belief, st)`.
"""
function Mycelium.factor_message(f::ToricFactor, target::Symbol, polarity, inputs, prior, ps, st)
    LenticulumCore.supports_polarity(f, polarity) &&
        LenticulumCore.unobserved_channels(polarity) == (target,) ||
        throw(ArgumentError("unsupported target/polarity"))
    t = findfirst(isequal(target), f.names)
    l = _logits(f, ps)
    joint = _joint_logweights(f, inputs, l; skip = t)
    # Shift once before marginalising so a large common logit never overflows.
    joint = _lognormalise(joint)
    out = fill(eltype(joint)(-Inf), f.alphabets[t])
    for (k, x) in enumerate(CartesianIndices(f.alphabets))
        a, b = out[x[t]], joint[k]
        m = max(a, b)
        out[x[t]] = m == -Inf ? m : m + log1p(exp(-abs(a - b)))
    end
    return (LenticulumCore.CategoricalBelief(Val(:log), _lognormalise(out), _outlabels(f, t)), st)
end

function _factor_statistics(f, msgs, ps)
    l = _logits(f, ps)
    logq = _lognormalise(_joint_logweights(f, msgs, l))
    energies = f.normalised ? -_lognormalise(l) : -l
    U = sum(logq[k] == -Inf ? zero(eltype(l)) : exp(logq[k]) * energies[k] for k in eachindex(l))
    H = -sum(v == -Inf ? zero(v) : exp(v) * v for v in logq)
    return U, H
end

"""
    Mycelium.local_free_energy(f::ToricFactor, msgs, ps, st)

`E_q[energy] - H(q)` for the joint factor belief
`q ∝ exp(A'η) * prod(msgs)`. These are variable-to-factor messages, not variable
marginals. No incoming-message energy is charged. Mycelium separately adds
`sum((degree(v)-1) * H(b_v))`; adding that correction here would count it twice.
Returns a scalar and the unchanged state.
"""
function Mycelium.local_free_energy(f::ToricFactor, msgs, ps, st)
    U, H = _factor_statistics(f, msgs, ps)
    return (U - H, st)
end

function _game_messages(f, π, y)
    f.normalised || throw(ArgumentError("game quantities require normalised = true"))
    y === nothing && (y = NamedTuple())
    y isa NamedTuple || throw(ArgumentError("game observations must be a NamedTuple of channel beliefs or labels"))
    all(n -> n in f.names, keys(y)) || throw(ArgumentError("unknown observation channel"))
    obs = map(v -> v isa LenticulumCore.AbstractBelief || v === nothing || v === missing ||
                   v isa LenticulumCore.Latent ? v : LenticulumCore.DiracBelief(v), y)
    if π isa NamedTuple
        all(n -> n in f.names, keys(π)) || throw(ArgumentError("unknown prior channel"))
        isempty(intersect(keys(π), keys(y))) || throw(ArgumentError("prior and observation channels must be disjoint"))
        return merge(π, obs)
    elseif π === nothing || π isa LenticulumCore.TrivialBelief
        return obs
    elseif π isa LenticulumCore.AbstractBelief
        remaining = filter(n -> !(n in keys(y)), f.names)
        length(remaining) == 1 || throw(ArgumentError("a single prior needs exactly one channel omitted from y; use a NamedTuple otherwise"))
        return merge(NamedTuple{remaining}((π,)), obs)
    end
    throw(ArgumentError("prior must be a belief, NamedTuple of beliefs, or nothing"))
end

"""
    LenticulumCore.free_energy(f::ToricFactor, π, y, ps, st)
    LenticulumCore.entropy(f::ToricFactor, π, y, ps, st)

Exact scalar game quantities for `normalised = true`, using the same joint factor
belief as `local_free_energy`. `π` is a NamedTuple of incoming channel beliefs and `y`
a disjoint NamedTuple of observations (beliefs or labels). A single belief `π` is
assigned to the sole channel omitted from `y`. `nothing` supplies no evidence;
unspecified channels are summed with unit weight. Entropy is positive Shannon entropy.
"""
LenticulumCore.free_energy(f::ToricFactor, π, y, ps, st) =
    Mycelium.local_free_energy(f, _game_messages(f, π, y), ps, st)
function LenticulumCore.entropy(f::ToricFactor, π, y, ps, st)
    _, H = _factor_statistics(f, _game_messages(f, π, y), ps)
    return (H, st)
end
