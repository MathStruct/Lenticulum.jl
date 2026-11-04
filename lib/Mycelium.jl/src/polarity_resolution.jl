# ---------------------------------------------------------------------------
# Turning "which message am I computing?" into a `Polarity`.
#
# This is the step that has no counterpart in Lux. A Lux layer knows its direction; a
# Lenticulum factor does not, and the graph must decide it per message:
#
#     target channel                              -> Unobserved()   (what we infer)
#     channels with an available incoming message -> Observed()     (what we clamp)
#     everything else                             -> Latent()       (marginal / scratch)
#
# Then `LenticulumCore.assemble(factor, polarity)` produces the lens and the message is an
# inversion. See `polarity_resolution.md` and `Messages are Inversions.md`.
# ---------------------------------------------------------------------------

"""
    PolarityError(factor, channel, reason)

Raised when a requested message is not legal: the edge direction forbids it, or the factor
does not support the resulting polarity.
"""
struct PolarityError <: Exception
    factor::Symbol
    channel::Symbol
    reason::String
end
Base.showerror(io::IO, e::PolarityError) =
    print(io, "PolarityError: factor :", e.factor, ", channel :", e.channel, " — ", e.reason)

"""
    validate(g::FactorGraph)

Structural checks that must hold before any message can be scheduled. Throws on failure,
returns `g` on success.

Checks:
1. every edge's channel is declared by its factor (`LenticulumCore.channels`);
2. every factor has at least one edge (an isolated factor contributes energy nobody reads);
3. every variable has at least one edge;
4. no variable is written by two `Emitting` edges *and* read by none — a variable that is
   only ever produced is a dangling output, which is legal but almost always a wiring bug,
   so it is reported as a warning rather than an error.
"""
function validate(g::FactorGraph)
    for (i, e) in enumerate(g.edges)
        fn = factornode(g, e.factor)
        decl = hasmethod(LenticulumCore.channels, Tuple{typeof(fn.factor)}) ?
            Tuple(LenticulumCore.channelname(c) for c in LenticulumCore.channels(fn.factor)) :
            nothing
        if decl !== nothing && !(e.channel in decl)
            throw(PolarityError(fn.name, e.channel,
                "not a declared channel of this factor; declared: $(decl)"))
        end
    end
    for f in g.factors
        isempty(g.edges_of_factor[f.id]) &&
            throw(ArgumentError("factor :$(f.name) has no edges"))
    end
    for v in g.variables
        isempty(g.edges_of_variable[v.id]) &&
            throw(ArgumentError("variable :$(v.name) has no edges"))
        n_emit = count(ei -> g.edges[ei].direction isa Emitting, g.edges_of_variable[v.id])
        n_abs = count(ei -> can_absorb(g, ei), g.edges_of_variable[v.id])
        if n_emit >= 2 && n_abs == 0
            @warn "variable :$(v.name) is written by $n_emit Emitting edges and read by none (dangling output)"
        end
    end
    return g
end

"""
    can_emit(g, edge_index) -> Bool
    can_absorb(g, edge_index) -> Bool

Whether the edge's direction permits a factor → variable (resp. variable → factor) message.
"""
can_emit(g::FactorGraph, ei::Int) = emits(g.edges[ei].direction)
can_absorb(g::FactorGraph, ei::Int) = absorbs(g.edges[ei].direction)

"""
    resolve_polarity(g, fid, target_channel, available_channels) -> Polarity

Build the `Polarity` for the message that infers `target_channel` from the channels in
`available_channels`.

`available_channels` is the set of channels for which an incoming (variable → factor)
message exists *and* whose edge absorbs. Every other declared channel of the factor becomes
`Latent()`.

Precisely one channel is `Unobserved()`. That is a deliberate restriction for v0: a message
addresses one variable. Joint messages over several channels at once are the correct
generalisation (and the only way to keep correlations between them — cf. the mean-field
laxness of `Composition of Bayesian Lenses.md` Remark 16), and are noted as future work in
`polarity_resolution.md`.
"""
function resolve_polarity(g::FactorGraph, fid::Int, target::Symbol, available)
    fn = factornode(g, fid)
    decl = Tuple(LenticulumCore.channelname(c) for c in LenticulumCore.channels(fn.factor))
    target in decl || throw(PolarityError(fn.name, target, "not a declared channel"))
    avail = Set{Symbol}(available)
    target in avail && throw(PolarityError(fn.name, target,
        "the target channel cannot also be observed"))
    vals = map(decl) do c
        c === target        ? LenticulumCore.Unobserved() :
        c in avail          ? LenticulumCore.Observed()   :
                              LenticulumCore.Latent()
    end
    return LenticulumCore.Polarity(NamedTuple{decl}(vals))
end

"""
    check_legal(g, fid, polarity)

Verify that a polarity is compatible with the *edge directions* of this graph, and with the
factor's own `supports_polarity`. Throws a [`PolarityError`](@ref) naming the offending
channel.

The two checks are genuinely different:

- **edge direction** is a property of the *wiring*: an `Emitting` edge may not be
  `Observed()`, an `Absorbing` edge may not be `Unobserved()`;
- **`supports_polarity`** is a property of the *factor*: can it be run this way at all?

Together they are the replacement for Lux's acyclicity restriction. A Lux `Chain` is
well-formed iff the wiring is a DAG; a Mycelium graph is well-formed iff every scheduled
message passes both checks.
"""
function check_legal(g::FactorGraph, fid::Int, p::LenticulumCore.Polarity)
    fn = factornode(g, fid)
    for ei in g.edges_of_factor[fid]
        e = g.edges[ei]
        pol = p[e.channel]
        if pol isa LenticulumCore.Observed && !absorbs(e.direction)
            throw(PolarityError(fn.name, e.channel,
                "edge is $(nameof(typeof(e.direction))), so this channel cannot be Observed"))
        end
        if pol isa LenticulumCore.Unobserved && !emits(e.direction)
            throw(PolarityError(fn.name, e.channel,
                "edge is $(nameof(typeof(e.direction))), so this channel cannot be Unobserved"))
        end
    end
    LenticulumCore.supports_polarity(fn.factor, p) || throw(PolarityError(
        fn.name, first(LenticulumCore.unobserved_channels(p)),
        "factor does not support this polarity (supports_polarity returned false)"))
    return nothing
end

"""
    legal_targets(g, fid) -> Vector{Symbol}

The channels of factor `fid` on which it could ever emit, given the edge directions alone.
A factor with exactly one legal target is unidirectional *in this graph*, which is a weaker
and more useful notion than `LenticulumCore.isunidirectional` (a property of the factor).
"""
legal_targets(g::FactorGraph, fid::Int) =
    [g.edges[ei].channel for ei in g.edges_of_factor[fid] if can_emit(g, ei)]

"""
    legal_sources(g, fid) -> Vector{Symbol}

The channels of factor `fid` from which it could ever read.
"""
legal_sources(g::FactorGraph, fid::Int) =
    [g.edges[ei].channel for ei in g.edges_of_factor[fid] if can_absorb(g, ei)]

# --- Parameter trees (review of 2026-11-04, LP-023) -----------------------------------------

"""
    validate_parameters(g, ps) -> nothing

Checks a graph-wide parameter tree before inference or training, which otherwise read a missing
key as an empty tree (`_subtree`) and fail later or not at all:

- every node whose `LuxCore.initialparameters` is not empty has its parameter key in `ps`;
- that entry has the same names (recursively) and array sizes as the initial parameters (it is not
  compared by value; a declared `output` coefficient passed as a parameter is caught here);
- every frozen path of a node exists in its entry.

Throws an `ArgumentError` listing every problem. Keys of `ps` that no node reads are reported in
a warning (a misspelt key, say), not an error.
"""
function validate_parameters(g::FactorGraph, ps)
    problems = String[]
    seen = Set{Symbol}()
    for fn in g.factors
        fn.params in seen && continue
        push!(seen, fn.params)
        p0 = LuxCore.initialparameters(Random.Xoshiro(0), fn.factor)
        has = ps isa NamedTuple && haskey(ps, fn.params)
        if !has
            _isempty_tree(p0) || push!(problems, "no parameters for :$(fn.params) ($(nameof(typeof(fn.factor))) node :$(fn.name))")
            continue
        end
        p = getfield(ps, fn.params)
        _shape_problems!(problems, string(fn.params), p, p0)
        paths = LenticulumCore._freeze_paths(fn.freeze)
        paths === :all && continue
        for path in paths
            _has_path(p, path) || push!(problems, "frozen path $(path) of :$(fn.name) is not in its parameters")
        end
    end
    isempty(problems) || throw(ArgumentError("validate_parameters: " * join(problems, "; ")))
    extra = ps isa NamedTuple ? [k for k in keys(ps) if !(k in seen)] : Symbol[]
    isempty(extra) || @warn "validate_parameters: keys no node reads: $(join(extra, ", "))"
    return nothing
end
_isempty_tree(x::NamedTuple) = all(_isempty_tree, values(x))
_isempty_tree(x::Tuple) = all(_isempty_tree, x)
_isempty_tree(_) = false
function _shape_problems!(out, where, p, p0)
    if p0 isa NamedTuple
        p isa NamedTuple || return push!(out, "$where should be a NamedTuple with keys $(keys(p0)); got $(typeof(p))")
        Set(keys(p)) == Set(keys(p0)) || push!(out, "$where has keys $(keys(p)); expected $(keys(p0))")
        for k in keys(p0)
            haskey(p, k) && _shape_problems!(out, "$where.$k", getfield(p, k), getfield(p0, k))
        end
    elseif p0 isa AbstractArray
        p isa AbstractArray && size(p) == size(p0) || push!(out, "$where has size $(p isa AbstractArray ? size(p) : typeof(p)); expected $(size(p0))")
    elseif p0 isa Number
        p isa Number || push!(out, "$where should be a number; got $(typeof(p))")
    end
    return out
end
_has_path(x, path) = isempty(path) ? true :
    (x isa NamedTuple && path[1] isa Symbol && haskey(x, path[1]) && _has_path(getfield(x, path[1]), Base.tail(path))) ||
    (x isa Tuple && path[1] isa Int && 1 <= path[1] <= length(x) && _has_path(x[path[1]], Base.tail(path)))

