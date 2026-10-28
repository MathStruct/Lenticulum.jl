# Functional replay of a fixed schedule. Reference boundary values are captured outside
# AD, so the same cuts also have a well-defined central-difference reference.

"""
    GradientEstimate(value, standard_error, nsamples)

Monte Carlo parameter gradient and componentwise standard error, both shaped like `ps`.
This is an estimate of an expectation, not an exact derivative or confidence bound.
"""
struct GradientEstimate{V,S}
    value::V
    standard_error::S
    nsamples::Int
end

struct _ReplayStore <: AbstractMessageStore
    to_variable::Vector{Any}
    to_factor::Vector{Any}
    iteration::Int
end
_ReplayStore(g::FactorGraph) = _ReplayStore(Vector{Any}(nothing,nedges(g)),Vector{Any}(nothing,nedges(g)),0)

# A buffered operation reads the store at the start of its sweep. Lux state still
# threads task by task, matching sweep!(::FloodingSchedule).
struct _ReplayTask
    task::MessageTask
    buffered::Bool
    first::Bool
end
_task(t::MessageTask) = t
_task(t::_ReplayTask) = t.task
_buffer(s, buffer, t::MessageTask) = (s, s)
_buffer(s, buffer, t::_ReplayTask) = begin
    saved = t.first ? s : buffer
    (t.buffered ? saved : s, saved)
end

function _realised_tasks(g, schedule, nsweeps, ps, st; damping=0.0)
    nsweeps isa Int && nsweeps > 0 || throw(ArgumentError("nsweeps must be a positive Int"))
    0 <= damping < 1 || throw(ArgumentError("damping must be in [0, 1)"))
    if schedule isa ResidualSchedule
        schedule.maxtasks > 0 || throw(ArgumentError("ResidualSchedule.maxtasks must be positive"))
        store = MessageStore(g)
        chosen = MessageTask[]
        for _ in 1:nsweeps
            _, st = sweep!(store, g, schedule, ps, st; damping, record=chosen)
        end
        return chosen
    end
    schedule isa Union{SequentialSchedule,TreeSchedule,ForwardBackwardSchedule,FloodingSchedule} ||
        throw(ArgumentError("unsupported gradient replay schedule $(typeof(schedule))"))
    ts = tasks(schedule)
    all(t -> 1 <= t.edge <= nedges(g) && t.kind in (:to_factor, :to_variable) && islegal(g,t), ts) ||
        throw(ArgumentError("replay schedule contains an illegal message task"))
    return [_ReplayTask(t, schedule isa FloodingSchedule, i == 1) for _ in 1:nsweeps for (i,t) in enumerate(ts)]
end

function _damped_candidate(s, g, t, ps, st, damping, ad=nothing)
    b, state = _candidate(s, g, t, ps, st, ad)
    old = t.kind === :to_variable ? s.to_variable[t.edge] : nothing
    if old !== nothing && damping > 0
        b = damp(b, old.belief, 1 - damping)
    end
    return b, state
end

function _check_schedule_ad(ad)
    ad === nothing && return
    nameof(typeof(ad)) in (:AutoForwardDiff, :AutoZygote) || throw(ArgumentError(
        "schedule differentiation supports reference differences, AutoForwardDiff and AutoZygote; got $(typeof(ad))"))
end
_gradient_message(ad, args...) = factor_message(args...)

function _candidate(s, g, t, ps, st, ad=nothing)
    e = g.edges[t.edge]
    if t.kind === :to_factor
        can_absorb(g, t.edge) || throw(ArgumentError("schedule sends to an emitting-only factor port"))
        return excluded_marginal(s, g, e.variable, t.edge), st
    end
    can_emit(g, t.edge) || throw(ArgumentError("schedule sends from an absorbing-only factor port"))
    f = factornode(g, e.factor)
    chans = available_channels(s, g, f.id; skip = t.edge)
    pol = resolve_polarity(g, f.id, e.channel, chans)
    if isempty(chans) && !issink(f.factor) && !LenticulumCore.supports_polarity(f.factor, pol)
        return LenticulumCore.TrivialBelief(), st
    end
    check_legal(g, f.id, pol)
    b, state = _gradient_message(ad, f.factor, e.channel, pol, _inputs(s, g, f.id, chans),
        excluded_marginal(s, g, e.variable, t.edge), _subtree(ps, f.name), _subtree(st, f.name))
    return b, _setsubtree(st, f.name, state)
end

function _functional_infer(g, ts, ps, st, report, damping=0.0)
    final_s, final_state, _ = foldl(ts; init=(_ReplayStore(g), st, _ReplayStore(g))) do acc, op
        store, state, buffer = acc
        read, buffer = _buffer(store, buffer, op)
        t = _task(op)
        e = g.edges[t.edge]
        psnames = keys(ps)
        pinned = NamedTuple{psnames}(map(name -> _factor_parameters(factornode(g,name),ps[name]), psnames))
        b, state = _damped_candidate(read, g, t, pinned, state, damping, Val(:chainrules))
        b = e.detach ? LenticulumCore.stop_gradient(b) : b
        return _with_message(store,t,b), LenticulumCore.stop_gradient(state), buffer
    end
    names = Tuple(v.name for v in _variable_nodes(g))
    final_store = MessageStore(copy(final_s.to_variable),copy(final_s.to_factor),report.sweeps)
    return NamedTuple{names}(map(v -> marginal(final_s,g,v.id), _variable_nodes(g))), report, final_state, final_store
end

function _with_message(s, t, belief)
    msg = Message(belief, s.iteration + 1)
    tv = t.kind === :to_variable ? [i == t.edge ? msg : m for (i,m) in enumerate(s.to_variable)] : s.to_variable
    tf = t.kind === :to_factor ? [i == t.edge ? msg : m for (i,m) in enumerate(s.to_factor)] : s.to_factor
    return _ReplayStore(tv, tf, s.iteration + 1)
end

function _graph_scalar(s, g, ps, st)
    terms = map(_factor_nodes(g)) do f
        _local_scalar(_subtree(ps, f.name), f.factor, messages_into(s, g, f.id), _subtree(st, f.name))
    end
    correction = sum(map(_variable_nodes(g)) do v
        counting_number(g, v.id) * variable_free_energy(marginal(s, g, v.id))
    end)
    return sum(terms) + correction
end

_check_pathwise(::Union{LenticulumCore.DiracBelief,LenticulumCore.GaussianBelief,LenticulumCore.TrivialBelief}) = nothing
_check_pathwise(b) = throw(ArgumentError(
    "PathwiseCoupling requires Gaussian or Dirac messages (or an uninformative identity); $(typeof(b)) has no reparametrisation rule"))

function _draw(rng, b::LenticulumCore.GaussianBelief)
    LenticulumCore.isproper(b) || throw(ArgumentError("score sampling requires a proper Gaussian message"))
    return LenticulumCore.belief_mean(b) + LinearAlgebra.cholesky(LinearAlgebra.Symmetric(LenticulumCore.belief_cov(b))).L *
        Random.randn(rng, LenticulumCore.dimension(b))
end
function _draw(rng, b::LenticulumCore.CategoricalBelief)
    all(isfinite,b.logp) || throw(ArgumentError(
        "categorical score gradients require strictly positive probabilities; changing support at a boundary has no supplied derivative rule"))
    p = LenticulumCore.probabilities(b)
    u = Random.rand(rng)
    k = min(searchsortedfirst(cumsum(p), u), length(p))
    return b.labels === nothing ? k : b.labels[k]
end
_draw(rng, b) = throw(ArgumentError("ScoreFunctionCoupling needs a sampleable Gaussian or categorical message with a density; got $(typeof(b))"))

function _score_baseline(e)
    b = e.coupling.baseline
    b === nothing && return 0.0
    b isa Real && isfinite(b) || throw(ArgumentError(
        "score baseline must be a finite, parameter-independent scalar (or nothing)"))
    return b
end

function _reference_tape(g, ts, ps, st, rng, damping=0.0)
    s = _ReplayStore(g)
    buffer = s
    tape = Any[]
    for op in ts
        read, buffer = _buffer(s, buffer, op)
        t = _task(op)
        b, st = _damped_candidate(read, g, t, ps, st, damping)
        e = g.edges[t.edge]
        if !e.detach && e.coupling isa LenticulumCore.PathwiseCoupling
            _check_pathwise(b)
        end
        if !e.detach && e.coupling isa LenticulumCore.ScoreFunctionCoupling
            e.direction isa Absorbing && t.kind === :to_factor || throw(ArgumentError(
                "score coupling currently requires an Absorbing edge: sampling in bidirectional BP changes its density/entropy objective"))
            all(ei -> !emits(g.edges[ei].direction), g.edges_of_factor[e.factor]) || throw(ArgumentError(
                "score sampling currently requires a terminal factor with only Absorbing ports"))
            ports = g.edges_of_factor[e.factor]
            count(ei -> g.edges[ei].variable == e.variable,ports) == 1 || throw(ArgumentError(
                "a sampled terminal factor cannot read the same variable on multiple ports without a joint sampling rule"))
            _score_baseline(e)
            x = _draw(rng, b)
            entry = (belief=LenticulumCore.DiracBelief(x), sample=x, logp=belief_logdensity(b, x))
        else
            entry = (belief=b, sample=nothing, logp=0.0)
        end
        push!(tape, entry)
        s = _with_message(s, t, entry.belief)
    end
    weighted = map(zip(ts,tape)) do (op,entry)
        t = _task(op)
        e = g.edges[t.edge]
        f = g.factors[e.factor]
        loss = !e.detach && e.coupling isa LenticulumCore.ScoreFunctionCoupling ?
            _local_scalar(_subtree(ps,f.name),f.factor,messages_into(s,g,f.id),_subtree(st,f.name)) : 0.0
        merge(entry,(loss=loss,))
    end
    return weighted, s, st
end

# The score samples and coefficients are fixed reference contexts. Only their log
# densities are differentiated. This is the stochastic-computation-graph identity.
function _replay(ps, g, ts, tape, st, ad, damping)
    final_s, final_state, correction, _ = foldl(zip(ts, tape); init=(_ReplayStore(g), st, 0.0, _ReplayStore(g))) do acc, item
        store, state, score, buffer = acc
        op, ref = item
        read, buffer = _buffer(store, buffer, op)
        t = _task(op)
        e = g.edges[t.edge]
        b, state = _damped_candidate(read, g, t, ps, state, damping, ad)
        if e.detach || e.coupling isa DiagonalCoupling
            b = ref.belief
        elseif e.coupling isa LenticulumCore.ScoreFunctionCoupling
            score += (ref.loss - _score_baseline(e)) * (belief_logdensity(b, ref.sample) - ref.logp)
            b = ref.belief
        end
        return _with_message(store, t, b), state, score, buffer
    end
    return _graph_scalar(final_s, g, ps, final_state) + correction
end

function _pinned_parameters(ps, reference_ps, g)
    names = keys(reference_ps)
    pinned = NamedTuple{names}(map(names) do name
        f = factornode(g, name)
        indices = _active_indices(f, reference_ps[name])
        θ = LenticulumCore._flatten_active(ps[name])
        _replace_active(reference_ps[name], indices, _gather_active(θ, indices))
    end)
    return pinned
end

function _coupled_loss(ps, reference_ps, g, ts, tape, st, ad, damping)
    return _replay(_pinned_parameters(ps, reference_ps, g), g, ts, tape, st, ad, damping)
end

function _coupled_gradient(s, g, ps, st; ad=nothing, ε=1e-6, schedule=nothing,
        nsweeps=1, rng=nothing, nsamples=nothing, return_state=false, damping=0.0,
        adjoint=:unrolled, maxsweeps=1000, tol=1e-10, linear_solver=:dense,
        linear_tol=1e-10, linear_maxiter=1000)
    _check_energy_edges(g)
    adjoint in (:unrolled, :implicit) || throw(ArgumentError("adjoint must be :unrolled or :implicit"))
    if adjoint === :implicit
        return _implicit_graph_gradient(g, ps, st; ad, ε, schedule, damping, maxsweeps, tol,
            linear_solver, linear_tol, linear_maxiter, rng, nsamples, return_state)
    end
    schedule === nothing && throw(ArgumentError(
        "non-diagonal graph gradients require schedule=... and an explicit finite nsweeps (default 1); replay starts from empty messages"))
    _check_schedule_ad(ad)
    for f in g.factors
        LenticulumCore._check_factor_ad(ad, f.factor, _subtree(st, f.name))
    end
    stochastic = any(e -> !e.detach && e.coupling isa LenticulumCore.ScoreFunctionCoupling, g.edges)
    if stochastic
        nsweeps == 1 && !(schedule isa Union{FloodingSchedule,ResidualSchedule}) && damping == 0 ||
            throw(ArgumentError("score sampling currently supports one undamped static sweep"))
        rng isa AbstractRNG || throw(ArgumentError("ScoreFunctionCoupling requires an explicit rng"))
        nsamples isa Int && nsamples >= 2 || throw(ArgumentError("score gradient standard errors require nsamples >= 2"))
    else
        nsamples === nothing || throw(ArgumentError("nsamples is only used for score coupling"))
    end
    ts = _realised_tasks(g, schedule, nsweeps, ps, st; damping)
    n = stochastic ? nsamples : 1
    results = map(1:n) do draw
        tape, reference_store, reference_state = _reference_tape(g, ts, ps, st, rng, damping)
        gradient = LenticulumCore._parameter_gradient(ad, _coupled_loss, ps, ps, g, ts, tape, st, ad, damping; ε)
        state = return_state && draw == 1 ? last(factor_free_energies(reference_store,g,ps,reference_state)) : nothing
        return gradient,state
    end
    draws = map(first,results)
    if !stochastic
        return return_state ? only(results) : only(draws)
    end
    vectors = map(LenticulumCore._flatten, draws)
    μ = sum(vectors) / n
    se = sqrt.(sum(v -> abs2.(v - μ), vectors) / (n * (n - 1)))
    estimate = GradientEstimate(LenticulumCore._unflatten(μ, ps), LenticulumCore._unflatten(se, ps), n)
    return return_state ? (estimate,last(first(results))) : estimate
end

"""
    StraightThroughFactor(forward, surrogate; channels=(:in, :out))

Parameterless directed factor: forward values use `forward`, input derivatives use
`surrogate`. Callables must have matching scalar/array output shape. ForwardDiff and
ChainRules reverse AD are supported. Reference differences of the primal see `forward`.
"""
struct StraightThroughFactor{F,S} <: LenticulumCore.AbstractLenticulumFactor
    forward::F
    surrogate::S
    chans::Tuple{Symbol,Symbol}
end
function StraightThroughFactor(f, s; channels=(:in, :out))
    channels isa Tuple{Symbol,Symbol} && channels[1] !== channels[2] || throw(ArgumentError(
        "StraightThroughFactor channels must be two distinct Symbols"))
    return StraightThroughFactor(f, s, channels)
end
LenticulumCore.channels(f::StraightThroughFactor) = Tuple(LenticulumCore.Channel(c, nothing) for c in f.chans)
LenticulumCore.supported_polarities(f::StraightThroughFactor) =
    (_polarity(f.chans, (LenticulumCore.Observed(), LenticulumCore.Unobserved())),)
LenticulumCore.supports_polarity(f::StraightThroughFactor, p::LenticulumCore.Polarity) =
    p == only(LenticulumCore.supported_polarities(f))
LenticulumCore.islearnable(::StraightThroughFactor) = false
local_free_energy(::StraightThroughFactor, beliefs, ps, st) = (0.0, st)
LenticulumCore.free_energy(::StraightThroughFactor, prior, y, ps, st) = (0.0, st)
LenticulumCore.entropy(::StraightThroughFactor, prior, y, ps, st) = (0.0, st)
LenticulumCore.energy(::StraightThroughFactor, x, a, y, ps, st) = (0.0, st)
function LenticulumCore._check_factor_ad(ad, ::StraightThroughFactor)
    ad !== nothing && nameof(typeof(ad)) in (:AutoForwardDiff,:AutoZygote) || throw(ArgumentError(
        "StraightThroughFactor requires AutoForwardDiff or AutoZygote; differences of its forward function do not implement the surrogate Jacobian"))
    return nothing
end
function factor_message(f::StraightThroughFactor, target::Symbol, pol, inputs, prior, ps, st)
    return LenticulumCore.DiracBelief(LenticulumCore.straight_through(f.forward, f.surrogate, point(inputs[f.chans[1]]))), st
end
