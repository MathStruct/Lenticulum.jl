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

# A buffered operation reads the store at the start of its sweep. Lux state still
# threads task by task, matching sweep!(::FloodingSchedule).
struct _ReplayTask
    task::MessageTask
    buffered::Bool
    first::Bool
end
_task(t::MessageTask) = t
_task(t::_ReplayTask) = t.task
_starts_sweep(::MessageTask) = true
_starts_sweep(t::_ReplayTask) = t.first
_buffered(::MessageTask) = false
_buffered(t::_ReplayTask) = t.buffered

function _task_reads(g, t::MessageTask)
    e = g.edges[t.edge]
    tf = t.kind === :to_factor ? Int[] : g.edges_of_factor[e.factor]
    return (tf, g.edges_of_variable[e.variable], [e.variable])
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
        return excluded_marginal(s, g, e.variable, t.edge), nothing
    end
    can_emit(g, t.edge) || throw(ArgumentError("schedule sends from an absorbing-only factor port"))
    f = factornode(g, e.factor)
    chans = available_channels(s, g, f.id; skip = t.edge)
    pol = resolve_polarity(g, f.id, e.channel, chans)
    if isempty(chans) && !issink(f.factor) && !LenticulumCore.supports_polarity(f.factor, pol)
        return LenticulumCore.TrivialBelief(), nothing
    end
    check_legal(g, f.id, pol)
    b, state = _gradient_message(ad, f.factor, e.channel, pol, _inputs(s, g, f.id, chans),
        excluded_marginal(s, g, e.variable, t.edge), ps, st)
    return b, Some(state)
end

# Convert a vector to a named tree in one pass. Reverse mode unpacks its cotangent by position.
_named_tree(names, vals) = NamedTuple{names}(Tuple(vals))

function _parameter_nodes(g, names, @nospecialize(ps))
    groups = _key_groups(g)
    positions = Mycelium._keypositions(ps)
    nodes = similar(g.factors, length(names))
    for (i, name) in enumerate(names)
        ids = get(groups, name, nothing)
        ids === nothing && throw(ArgumentError("no factor reads parameters :$name"))
        haskey(positions, name) || getfield(ps, name) # Preserve the missing-field error.
        nodes[i] = g.factors[first(ids)]
    end
    return nodes
end

_marginal_reads(g, vs) = [(Int[], g.edges_of_variable[v.id], [v.id]) for v in vs]

function _functional_infer(g, ts, ps, st, report, damping=0.0)
    params = if isempty(ts)
        nothing
    else
        names = keys(ps)
        nodes = _parameter_nodes(g, names, ps)
        pinned = _named_tree(names, map(_factor_parameters, nodes, Mycelium._subtrees(ps, nodes)))
        Mycelium._subtrees(pinned, g.factors)
    end
    # I30, step 5: the state travels boxed through `_replay_run` (its rule unboxes the cotangent);
    # the pinning of `ps` stays here, in the per-factor form.
    s, boxed_state, _ = _replay_run(Val(:infer), g, ts, params, LenticulumCore._Erased(st), nothing, damping, Val(:chainrules), report.sweeps)
    vs = _variable_nodes(g)
    marginals = map((v, sl) -> marginal(sl, g, v.id), vs, Mycelium._local_stores(s, _marginal_reads(g, vs)))
    return _named_tree(Tuple(v.name for v in vs), marginals), report, boxed_state.tree, s
end

function _task_step(::Val{:infer}, s, g, t, ps, st, ref, damping, ad)
    b, state = _damped_candidate(s, g, t, ps, st, damping, ad)
    b = g.edges[t.edge].detach ? LenticulumCore.stop_gradient(b) : b
    return b, state === nothing ? nothing : Some(LenticulumCore.stop_gradient(something(state))), nothing
end

function _task_step(::Val{:replay}, s, g, t, ps, st, ref, damping, ad)
    e = g.edges[t.edge]
    b, state = _damped_candidate(s, g, t, ps, st, damping, ad)
    if e.detach || e.coupling isa DiagonalCoupling
        return ref.belief, state, nothing
    elseif e.coupling isa LenticulumCore.ScoreFunctionCoupling
        score = (ref.loss - _score_baseline(e)) * (belief_logdensity(b, ref.sample) - ref.logp)
        return ref.belief, state, score
    end
    return b, state, nothing
end

# I30, step 5: the state travels boxed. The primal unboxes for `_replay_forward` (whose reads of
# `st` are not differentiated: they run inside this function's rule, or under ForwardDiff as the
# plain primal) and boxes the final state again.
function _replay_run(mode, g, ts, params, st::LenticulumCore._Erased, tape, damping, ad, iteration)
    evaluate(i, fid, tv, tf, sv, save, args...) = _task_step(mode, args...)
    (s, final_state, score), _ = _replay_forward(evaluate, mode, g, ts, params, st.tree, tape, damping, ad, iteration)
    return (s, LenticulumCore._Erased(final_state), score)
end
# An unboxed state is boxed for the call and the final state unboxed again (callers outside the
# differentiated region keep receiving the tree).
function _replay_run(mode, g, ts, params, st, tape, damping, ad, iteration)
    s, boxed, score = _replay_run(mode, g, ts, params, LenticulumCore._Erased(st), tape, damping, ad, iteration)
    return (s, boxed.tree, score)
end

# Message version 0 is absent. Task i writes version i. At a sweep's first
# write to a slot, save its old version for the buffered reads in that sweep.
_read_version(live, saved, e, start) = live[e] >= start ? saved[e] : live[e]

function _replay_forward(evaluate, mode, g, ts, params, @nospecialize(st), tape, damping, ad, iteration)
    tv, tf = zeros(Int, nedges(g)), zeros(Int, nedges(g))
    saved_tv, saved_tf = similar(tv), similar(tf)
    messages, states = Vector{Any}(undef, length(ts)), Vector{Any}(undef, length(ts))
    state_names = st === nothing ? Symbol[] : collect(Symbol, keys(st))
    positions = st === nothing ? Dict{Symbol,Int}() : Mycelium._keypositions(st)
    live_state = collect(-1:-1:-length(state_names))
    has_state_tree = st !== nothing
    infer = mode isa Val{:infer}
    pins = Any[]
    score = 0.0
    start = 1
    for (i, op) in enumerate(ts)
        _starts_sweep(op) && (start = i)
        t = _task(op)
        read_tf, read_tv, vids = _task_reads(g, t)
        if _buffered(op)
            vt = [_read_version(tv, saved_tv, e, start) for e in read_tv]
            vf = [_read_version(tf, saved_tf, e, start) for e in read_tf]
        else
            vt, vf = tv[read_tv], tf[read_tf]
        end
        local_store = Mycelium._LocalStore(
            Mycelium._EdgeSlice(read_tv, _version_messages(messages, vt)),
            Mycelium._EdgeSlice(read_tf, _version_messages(messages, vf)), vids, pins)

        fid = g.edges[t.edge].factor
        key = g.factors[fid].params
        slot = get(positions, key, 0)
        sv = slot == 0 ? 0 : live_state[slot]
        state = _version_state(states, st, sv)
        if infer && sv < 0 && i > 1
            state = LenticulumCore.stop_gradient(state)
            sv = 0
        end
        ref = tape === nothing ? nothing : tape[i]
        b, nextstate, term = evaluate(i, fid, vt, vf, sv, has_state_tree,
            local_store, g, t, params[fid], state, ref, damping, ad)
        term === nothing || (score += term)
        messages[i] = Message(b, i)
        live, saved = t.kind === :to_factor ? (tf, saved_tf) : (tv, saved_tv)
        e = t.edge
        live[e] < start && (saved[e] = live[e])
        live[e] = i

        if nextstate !== nothing
            if has_state_tree
                if slot == 0
                    slot = length(state_names) + 1
                    positions[key] = slot
                    push!(state_names, key)
                    push!(live_state, i)
                else
                    live_state[slot] = i
                end
                states[i] = something(nextstate)
            end
            has_state_tree = true
        end
    end
    s = MessageStore(_version_messages(messages, tv), _version_messages(messages, tf), iteration)
    if has_state_tree
        vals = Any[_version_state(states, st, v) for v in live_state]
        if infer && !isempty(ts)
            for i in eachindex(live_state)
                live_state[i] < 0 && (vals[i] = LenticulumCore.stop_gradient(vals[i]))
            end
        end
        final_state = _named_tree(Tuple(state_names), vals)
    else
        final_state = st
    end
    return (s, final_state, score), (tv, tf, live_state)
end

_version_messages(messages, versions) = Any[v == 0 ? nothing : messages[v] for v in versions]
_version_state(states, @nospecialize(st), v) = v > 0 ? states[v] : v < 0 ? getfield(st, -v) : NamedTuple()

# LP-021: a factor whose edges are all cut (diagonal or detached) has no derivative path through
# its messages, so, as in the default driver (LP-020), its term is differentiated at its fixed
# belief. `frozen[f]` is that belief, evaluated at the reference messages outside AD, or
# `nothing` (a factor with a derivative path, or without `factor_belief`: fixed messages).
_all_cut(g, f) = all(ei -> g.edges[ei].detach || g.edges[ei].coupling isa DiagonalCoupling, g.edges_of_factor[f.id])

# LP-021, option 3. Couplings with a deterministic derivative path (or none); score estimators and
# straight-through surrogates are other semantics, kept for the replay.
_deterministic_couplings(g) = all(e -> e.detach || e.coupling isa Union{DiagonalCoupling,LenticulumCore.ExactCoupling,LenticulumCore.PathwiseCoupling}, g.edges) &&
    !any(f -> f.factor isa StraightThroughFactor, g.factors)
# An edge whose message cannot depend on the parameters: a data factor's, or any edge of an observed
# variable (one a `DataFactor` emits into). Cutting it changes nothing.
function _constant_edge(g, ei)
    e = g.edges[ei]
    g.factors[e.factor].factor isa DataFactor && return true
    return any(ej -> g.factors[g.edges[ej].factor].factor isa DataFactor, g.edges_of_variable[e.variable])
end
_cut_edge(e) = e.detach || e.coupling isa DiagonalCoupling
# The mixed case: a cut and a differentiated edge, both carrying parameter-dependent messages.
function _mixed_cuts(g)
    live = [ei for ei in eachindex(g.edges) if !_constant_edge(g, ei)]
    return any(ei -> _cut_edge(g.edges[ei]), live) && any(ei -> !_cut_edge(g.edges[ei]), live)
end

function _frozen_beliefs(s, g, ps, st)
    psi, sti = _indexed(ps), _indexed(st)
    return Any[f.rule isa BeliefPropagation && _all_cut(g, f) ?
               factor_belief(f.factor, messages_into(s, g, f.id), _subtree(psi, f.params), _subtree(sti, f.params)) :
               nothing for f in g.factors]
end

_graph_scalar(s, g, ps, st, frozen = nothing) =
    _graph_scalar_of(s, g, Mycelium._subtrees(ps, _factor_nodes(g)), st, frozen)
# The same with the factors' parameter subtrees given (`subs[i]` for `g.factors[i]`): the replay reads
# `ps` once for both uses, so that Zygote adds the two cotangents per subtree, not over the whole tree (I30).
function _graph_scalar_of(s, g, subs, st, frozen = nothing)
    fs, vs = _factor_nodes(g), _variable_nodes(g)
    locals = Mycelium._local_stores(s, Mycelium._node_reads(g, fs))
    beliefs = frozen === nothing ? fill(nothing, length(fs)) : frozen
    terms = map(fs, subs, Mycelium._states(st, fs, NamedTuple()), locals, beliefs) do f, p, state, sl, b
        b === nothing ? _local_scalar(p, f.factor, messages_into(sl, g, f.id), state) :
                        _belief_scalar(p, f.factor, b, state)
    end
    corrections = map((v, sl) -> _variable_term(sl, g, v.id), vs,
        Mycelium._local_stores(s, Mycelium._variable_reads_of(g, vs)))
    # Keep the two original sums, including Base's reduction order.
    return sum(terms) + sum(corrections)
end

_check_pathwise(::Union{LenticulumCore.DiracBelief,LenticulumCore.GaussianBelief,LenticulumCore.TrivialBelief}) = nothing
_check_pathwise(b) = throw(ArgumentError(
    "PathwiseCoupling requires Gaussian or Dirac messages (or an uninformative identity); $(typeof(b)) has no reparametrisation rule"))

function _draw(rng, b::LenticulumCore.GaussianBelief)
    LenticulumCore.isproper(b) || throw(ArgumentError("score sampling requires a proper Gaussian message"))
    Σ = LenticulumCore.belief_cov(b)
    C = LenticulumCore._symmetric_cholesky(Σ)
    z = Random.randn(rng, LenticulumCore.dimension(b))
    device = LenticulumCore._device_leaf(Σ)
    L = device === nothing ? C.L : LenticulumCore._lower_matrix(C)
    return LenticulumCore.belief_mean(b) + L * LenticulumCore._on_device(device, z)
end
function _draw(rng, b::LenticulumCore.CategoricalBelief)
    all(isfinite,b.logp) || throw(ArgumentError(
        "categorical score gradients require strictly positive probabilities; changing support at a boundary has no supplied derivative rule"))
    p = LenticulumCore.probabilities(b)
    u = Random.rand(rng)
    k = min(searchsortedfirst(cumsum(LenticulumCore._host_values(p)), u), length(p))
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
    tape = Vector{Any}(undef, length(ts))
    function evaluate(i, fid, tv, tf, sv, save, read, g, t, ps, state, ref, damping, ad)
        b, nextstate = _damped_candidate(read, g, t, ps, state, damping, ad)
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
        tape[i] = entry
        return entry.belief, nextstate, nothing
    end
    (s, st, _), _ = _replay_forward(evaluate, Val(:reference), g, ts,
        Mycelium._subtrees(ps, g.factors), st, nothing, damping, nothing, length(ts))
    psi, sti = _indexed(ps), _indexed(st)
    weighted = map(zip(ts,tape)) do (op,entry)
        t = _task(op)
        e = g.edges[t.edge]
        f = g.factors[e.factor]
        loss = !e.detach && e.coupling isa LenticulumCore.ScoreFunctionCoupling ?
            _local_scalar(_subtree(psi, f.params),f.factor,messages_into(s,g,f.id),_subtree(sti, f.params)) : 0.0
        merge(entry,(loss=loss,))
    end
    return weighted, s, st
end

# The score samples and coefficients are fixed reference contexts. Only their log
# densities are differentiated. This is the stochastic-computation-graph identity.
# I30, steps 2, 4, 5: `ps` and `st` arrive boxed (`_Erased`), so no signature of the glue Zygote
# transforms depends on the structure; the reads of the trees go through the rules of
# `Mycelium._subtrees`/`Mycelium._states` on the box.
function _replay(ps::LenticulumCore._Erased, g, ts, tape, st::LenticulumCore._Erased, ad, damping, frozen = nothing)
    subs = Mycelium._subtrees(ps, g.factors)
    final_s, final_state, correction = _replay_run(Val(:replay), g, ts, subs, st, tape, damping, ad, length(ts))
    return _graph_scalar_of(final_s, g, subs, final_state, frozen) + correction
end
# Unboxed trees are boxed for the call (callers outside the differentiated region keep passing and
# receiving trees).
_replay(ps, g, ts, tape, st, ad, damping, frozen = nothing) =
    _replay(LenticulumCore._Erased(ps), g, ts, tape, LenticulumCore._Erased(st), ad, damping, frozen)

# I30, step 3: the active positions of the graph-wide flat vector θ, built once outside AD
# (through the offset tree: each key's subtree occupies a contiguous range of θ). `posmap[i]` is
# the position of coordinate `i` among the active ones, 0 when frozen. The per-key `_active_indices`
# of the key's first node decide, as `_pinned_parameters` did.
function _graph_posmap(g, ps)
    posmap = zeros(Int, LenticulumCore._flatlen(ps))
    start = 1
    for (name, sub) in pairs(ps)
        f = _key_node(g, name)
        for (k, i) in enumerate(_active_indices(f, sub))
            posmap[start + i - 1] = k
        end
        start += LenticulumCore._flatlen(sub)
    end
    return posmap
end

# The pinning as a map of the flat coordinates, applied inside the differentiated boundary:
# `θ -> unflatten(_replace_coordinates(θref, posmap, θ))`. `identity` when nothing is frozen.
function _graph_pin(g, ps)
    posmap = _graph_posmap(g, ps)
    all(!iszero, posmap) && return identity
    θref = LenticulumCore._flatten(ps)
    mask = posmap .!= 0
    return θ -> _pin_coordinates_wide(θref, mask, θ)
end

function _pinned_parameters(ps, reference_ps, g)
    names = keys(reference_ps)
    nodes = _parameter_nodes(g, names, ps)
    values = map(nodes, Mycelium._subtrees(ps, nodes), Mycelium._subtrees(reference_ps, nodes)) do f, p, reference
        indices = _active_indices(f, reference)
        θ = LenticulumCore._flatten_active(p)
        _replace_active(reference, indices, _gather_active(θ, indices))
    end
    return _named_tree(names, values)
end

# I30, step 3: the pinning left the differentiated region — the drivers pin in flat coordinates
# through `_graph_pin`, at the boundary of `_graph_parameter_gradient`.
function _coupled_loss(ps::LenticulumCore._Erased, g, ts, tape, st::LenticulumCore._Erased, ad, damping, frozen = nothing)
    return _replay(ps, g, ts, tape, st, ad, damping, frozen)
end
# The per-key pinning of a reference tree, then the boxed call (the form the I17 tests drive
# directly; the drivers pin in flat coordinates instead).
function _coupled_loss(ps, reference_ps, g, ts, tape, st, ad, damping, frozen = nothing)
    return _coupled_loss(LenticulumCore._Erased(_pinned_parameters(ps, reference_ps, g)), g, ts, tape,
        LenticulumCore._Erased(st), ad, damping, frozen)
end

function _coupled_gradient(s, g, ps, st; ad=nothing, ε=1e-6, schedule=nothing,
        nsweeps=1, rng=nothing, nsamples=nothing, return_state=false, damping=0.0,
        adjoint=:unrolled, maxsweeps=1000, tol=1e-10, linear_solver=:dense,
        linear_tol=1e-10, linear_maxiter=1000)
    _check_energy_edges(g; mean_field = false,
        what = "differentiating through inference (non-diagonal couplings, adjoint = :implicit)")
    adjoint in (:unrolled, :implicit) || throw(ArgumentError("adjoint must be :unrolled or :implicit"))
    if adjoint === :implicit
        return _implicit_graph_gradient(g, ps, st; ad, ε, schedule, damping, maxsweeps, tol,
            linear_solver, linear_tol, linear_maxiter, rng, nsamples, return_state)
    end
    schedule === nothing && throw(ArgumentError(
        "non-diagonal graph gradients require schedule=... and an explicit finite nsweeps (default 1); replay starts from empty messages"))
    _check_schedule_ad(ad)
    sti = _indexed(st)
    for f in g.factors
        LenticulumCore._check_factor_ad(ad, f.factor, _subtree(sti, f.params))
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
    _mixed_cuts(g) && @warn("the graph mixes cut (diagonal) and differentiated edges carrying parameter-dependent \
        messages; the replay's gradient holds the cut messages fixed and is not the free-energy gradient, \
        even at convergence (LP-021). Omit `schedule`, or use `adjoint = :implicit`, for the free-energy gradient.", maxlog = 1)
    ts = _realised_tasks(g, schedule, nsweeps, ps, st; damping)
    n = stochastic ? nsamples : 1
    pin = _graph_pin(g, ps)                       # I30, step 3: outside AD, in flat coordinates
    st_box = LenticulumCore._Erased(st)
    # A loop into `Vector{Any}`: `map` and `only` over results typed by the gradient tree (whose leaves
    # are views) were inferred per structure, 0.65 s of 3 s at 61 factors (I30's profile).
    results = Any[]
    for draw in 1:n
        tape, reference_store, reference_state = _reference_tape(g, ts, ps, st, rng, damping)
        frozen = _frozen_beliefs(reference_store, g, ps, reference_state)
        gradient = LenticulumCore._graph_parameter_gradient(ad, _coupled_loss, ps, g, ts, tape, st_box, ad, damping, frozen; ε, pin)
        # I27: the replayed free energy, which this differentiates, must be finite; at each draw
        # for a score edge, whose replay samples its beliefs. One more forward replay.
        _check_finite(_coupled_loss(LenticulumCore._Erased(ps), g, ts, tape, st_box, ad, damping, frozen))
        state = return_state && draw == 1 ? last(factor_free_energies(reference_store,g,ps,reference_state)) : nothing
        push!(results, (gradient, state))
    end
    draws = Any[first(r) for r in results]
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
