# ---------------------------------------------------------------------------
# Parameter gradients of the graph free energy, and the optimiser step.
#
# At a fixed point of message passing the beliefs are stationary, so the derivative of the Bethe
# free energy in the parameters is the sum of the factors' partial derivatives at fixed beliefs
# (the envelope gradient, LP-020). Non-diagonal couplings and `adjoint = :implicit` differentiate
# through inference instead (`gradient_couplings.jl`, `graph_adjoint.jl`).
#
# See `gradient.md`; the free energy itself is in Mycelium's `free_energy.jl`.
# ---------------------------------------------------------------------------

# The factor belief is evaluated once, outside AD, and held fixed: the envelope gradient of the
# Bethe free energy at a fixed point (LP-020). Factors with no `factor_belief` fall back to fixed
# incoming messages, which agrees with it when those messages are points.
_local_scalar(ps, factor, beliefs, st) = LenticulumCore.scalarise(
    LenticulumCore.scalarisation(factor), first(local_free_energy(factor, beliefs, ps, st)))
_belief_scalar(ps, factor, b, st) = LenticulumCore.scalarise(
    LenticulumCore.scalarisation(factor), first(belief_free_energy(factor, b, ps, st)))
_mean_field_scalar(ps, factor, marginals, st) = LenticulumCore.scalarise(
    LenticulumCore.scalarisation(factor), first(mean_field_energy(factor, marginals, ps, st)))

# Only a point message pins the belief; a missing or trivial one leaves it free to move.
_point_message(::LenticulumCore.DiracBelief) = true
_point_message(_) = false

# What a node's parameter gradient differentiates, as (scalar function, factor, fixed data): the
# scalar is a function of (ps, factor, data, st). Evaluated outside AD.
function _gradient_item(f, msgs, p, st)
    # A mean-field node's term is already at fixed beliefs, the marginals (`msgs` holds them).
    f.rule isa MeanField && return (_mean_field_scalar, f.factor, msgs)
    b = factor_belief(f.factor, msgs, p, st)
    b === nothing || return (_belief_scalar, f.factor, b)
    uses_cavity(f.factor) && !all(_point_message, values(msgs)) && @warn(
        "factor :$(f.name) ($(nameof(typeof(f.factor)))) has incoming messages that are not all points but no \
         `factor_belief`, so its parameter gradient holds the messages fixed, not its belief; \
         this is not the free-energy gradient in general (LP-020)", maxlog = 1)
    return (_local_scalar, f.factor, msgs)
end

# The plate rows of a key whose messages are blocks and whose factor type has batched beliefs, as
# one gradient item each; the other nodes are returned for the per-node path.
_row_items(s, g, key::Symbol, p, stf, nodes) = _row_items(s, g, _energy_rows(s, g), key, p, stf, nodes)
function _row_items(s, g, rows, key, p, stf, nodes)
    items = Any[]
    covered = Set{Int}()
    for row in rows
        g.factors[first(row.fids)].params === key || continue
        msgs = _row_messages(s, row)
        msgs === nothing && continue
        factors = [g.factors[i].factor for i in row.fids]
        B = factor_beliefs(factors, msgs, p, stf)
        B === nothing && continue
        push!(items, (_beliefs_scalar, factors, B))
        union!(covered, row.fids)
    end
    return items, [h for h in nodes if !(h.id in covered)]
end

# Tied nodes: one parameter vector, the sum of their scalars, differentiated in one call.
_tied_scalar(ps, items, st) = sum(map(it -> it[1](ps, it[2], it[3], st), items))

"""
    LenticulumCore.free_energy_gradient(store, g, ps, st; ad = nothing, ε = 1e-6,
        schedule = nothing, nsweeps = 1, damping = 0.0, adjoint = :unrolled,
        maxsweeps = 1000, tol = 1e-10, linear_solver = :dense,
        linear_tol = 1e-10, linear_maxiter = 1000)

Parameter gradient of the graph scalar free energy with each factor's **belief** held fixed:
``\\sum_c \\partial_\\theta F_c(b_c, \\theta)``, with ``b_c`` from [`factor_belief`](@ref) and ``F_c`` from
[`belief_free_energy`](@ref). At a belief-propagation fixed point this is the derivative of the
Bethe free energy (the envelope theorem), so on a tree it is the gradient of ``-\\log p`` (LP-020).
A factor without `factor_belief` is differentiated at fixed incoming messages instead, which
agrees when those messages are points; otherwise that is not the free-energy
gradient, and a warning says so. A [`MeanField`](@ref) node's term is differentiated at the fixed
marginals of its variables, which is the envelope gradient of the merged free energy at a fixed
point of the hybrid message passing; non-diagonal couplings, `adjoint = :implicit` and
sampling refuse such nodes. Bethe variable corrections are
constant with respect to parameters at fixed beliefs. Exact for the default linear
scalarisations, as the envelope argument is about the summed free energy. Nonlearnable subtrees receive
zeros. Frozen coordinates receive zeros too. The result has `ps`'s structure.
**Couplings (LP-021).** At a fixed point the free-energy gradient does not depend on the edges'
couplings, so without `schedule` a graph whose couplings are diagonal, exact or pathwise gets this
gradient at the store's beliefs too. A coupling changes the result only in a replay: with
`schedule`, non-diagonal edges replay finite `nsweeps` (default 1) from empty messages and
differentiate that computation, a cut (diagonal) edge holding its message fixed. When cut and
differentiated edges whose messages depend on the parameters meet in one graph (the mixed case),
that is not the free-energy gradient even when the replay converges (off by 0.2–0.7 in
`test/lp021.jl`), and a warning says so; `adjoint = :implicit` returns the envelope gradient at its
converged fixed point for such a graph instead. Score edges also
require `rng` and `nsamples >= 2`, returning a `GradientEstimate`. No input store or
state is mutated. Replay includes flooding snapshots, realised residual task choices
and outgoing damping. `adjoint=:implicit` instead differentiates a converged synchronous
BP fixed point, found from empty messages using `schedule`, `maxsweeps` and `tol`.
It requires CPU Gaussian/Dirac messages, empty message state and an invertible
`I-DT`; diagonal/detach rows are constant cuts, and a factor whose edges are all cut is held at
its fixed belief, as in the diagonal driver (LP-021). Dense is the reference linear solve;
`:krylov` needs Krylov.jl. Both check their original linear residual. Reference
differences, ForwardDiff and Zygote are supported. See `graph_adjoint.md` and
`gradient_couplings.md` for equations, estimator domains, costs and mixed rules.
"""
function LenticulumCore.free_energy_gradient(s::MessageStore, g::FactorGraph, ps, st;
                                            ad = nothing, ε = 1e-6,
                                            schedule = nothing, nsweeps = 1,
                                            rng = nothing, nsamples = nothing, damping = 0.0,
                                            adjoint = :unrolled, maxsweeps = 1000, tol = 1e-10,
                                            linear_solver = :dense, linear_tol = 1e-10,
                                            linear_maxiter = 1000)
    # Keep the same cavity validation as the reporting API.
    _check_energy_edges(g)
    _check_unnormalised(g)
    adjoint in (:unrolled, :implicit) || throw(ArgumentError("adjoint must be :unrolled or :implicit"))
    if _uses_coupled_driver(g, adjoint, schedule)
        return _coupled_gradient(s, g, ps, st; ad, ε, schedule, nsweeps, rng, nsamples,
            damping, adjoint, maxsweeps, tol, linear_solver, linear_tol, linear_maxiter)
    end
    return _envelope_gradient(s, g, ps, st; ad, ε)
end

# Which driver a gradient request takes; one rule for `free_energy_gradient` and `parameter_step`.
# LP-021, option 3: without a replay (no `schedule`) the couplings do not change the free-energy
# gradient, which at a fixed point is the envelope one; they matter for a replay only.
function _uses_coupled_driver(g, adjoint, schedule)
    coupled = adjoint === :implicit || any(e -> !(e.coupling isa DiagonalCoupling), g.edges)
    return coupled && !(adjoint === :unrolled && schedule === nothing && _deterministic_couplings(g))
end

# The free-energy gradient at the store's beliefs (LP-020): each factor's term differentiated at
# its fixed belief, tied nodes summed.
function _envelope_gradient(s, g, ps, st; ad = nothing, ε = 1e-6)
    keys_ = parameter_keys(g)
    # The groups, the rows and the trees' key positions once, not per key (I5).
    groups, rows, psi, sti = _key_groups(g), _energy_rows(s, g), _indexed(ps), _indexed(st)
    grads = map(keys_) do key
        nodes = [g.factors[i] for i in groups[key]]
        f = first(nodes)
        p = _subtree(psi, key)
        if !LenticulumCore.islearnable(f.factor)
            θ = LenticulumCore._flatten(p)
            return LenticulumCore._unflatten(zero(θ), p)
        end
        stf = _subtree(sti, key)
        LenticulumCore._check_factor_ad(ad, f.factor, stf)
        items, rest = _row_items(s, g, rows, key, p, stf, nodes)
        append!(items, map(h -> _gradient_item(h, h.rule isa MeanField ? marginals_into(s, g, h.id) :
                                                                  messages_into(s, g, h.id), p, stf), rest))
        grad = length(items) == 1 ?
            LenticulumCore._parameter_gradient(ad, only(items)[1], p, only(items)[2], only(items)[3], stf; ε) :
            LenticulumCore._parameter_gradient(ad, _tied_scalar, p, items, stf; ε)
        indices = _active_indices(f, p)
        θ = LenticulumCore._flatten(grad)
        return LenticulumCore._unflatten(_mask_gradient(θ, indices), p)
    end
    return NamedTuple{keys_}(grads)
end

"""
    parameter_step(store, g, ps, st; ad = nothing, rule = GradientDescent(0.01), optimiser_state = nothing)
        -> (ps_new, st_new, optimiser_state_new)

One graph free-energy parameter step with an explicit AD backend (`nothing` selects
reference differences). Inference must populate incoming messages first. The rule's
`get` supplies a look-ahead parameter vector per learnable factor, the gradient is
computed at those parameters with the factor beliefs fixed (see
[`free_energy_gradient`](@ref LenticulumCore.free_energy_gradient)), and `put` applies the update.
Optimiser state is a NamedTuple of vectors keyed by learnable factor name, initialised
to zero if omitted. Pass the returned state on the next step (Momentum/Nesterov).
Lux state advances once, by evaluating local energies at the look-ahead parameters.
Frozen coordinates are excluded from state vectors; fully frozen factors have no entry.
Non-diagonal drivers forward `schedule`, `nsweeps`, `rng`, `nsamples` to the gradient API.
Score steps return `(ps_new, st_new, optimiser_state_new, estimate)` so uncertainty
is available to the caller; deterministic steps retain the three-tuple return.
Coupled steps commit inference and one local-energy state advance from the first
replicate; other replicates do not accumulate Lux state. This does not update the input
message store or exposed OptimiserFactor wires.
"""
function parameter_step(s::MessageStore, g::FactorGraph, ps, st;
                        ad = nothing, rule = GradientDescent(0.01), optimiser_state = nothing,
                        kwargs...)
    # Every tree is read by position (I5): `ps`, the optimiser state and the gradient.
    groups, psi = _key_groups(g), _indexed(ps)
    fs = Tuple(f for f in map(k -> g.factors[first(groups[k])], parameter_keys(g))
               if !isempty(_active_indices(f, _subtree(psi, f.params))))
    names = Tuple(f.params for f in fs)
    indices = map(f -> _active_indices(f, _subtree(psi, f.params)), fs)
    θs = map((f, ii) -> _gather_active(LenticulumCore._flatten(_subtree(psi, f.params)), ii), fs, indices)
    os = optimiser_state === nothing ? NamedTuple{names}(map(zero, θs)) : optimiser_state
    osi = _indexed(os)
    for (f, θ) in zip(fs, θs)
        o = _stored(osi, f.params)
        o !== nothing && length(o) == length(θ) || throw(ArgumentError(
            "optimiser_state for :$(f.params) must contain exactly $(length(θ)) active coordinates; reinitialise it after changing freeze paths"))
    end
    look = map(fs, θs, indices) do f, θ, ii
        _replace_active(_subtree(psi, f.params), ii, rule_get(rule, _stored(osi, f.params), θ))
    end
    at = merge(ps, NamedTuple{names}(look))
    if _uses_coupled_driver(g, get(kwargs, :adjoint, :unrolled), get(kwargs, :schedule, nothing))
        result, stnew = _coupled_gradient(s,g,at,st;ad,return_state=true,kwargs...)
    else
        result = LenticulumCore.free_energy_gradient(s, g, at, st; ad, kwargs...)
        _, stnew = factor_free_energies(s, g, at, st)
    end
    grad = result isa GradientEstimate ? result.value : result
    gi = _indexed(grad)
    updates = map(fs, θs, indices) do f, θ, ii
        rule_put(rule, _stored(osi, f.params), θ, _gather_active(LenticulumCore._flatten(_stored(gi, f.params)), ii))
    end
    newps = map(fs, updates, indices) do f, u, ii
        _replace_active(_subtree(psi, f.params), ii, u[2])
    end
    output = (merge(ps, NamedTuple{names}(newps)), stnew,
              NamedTuple{names}(map(first, updates)))
    return result isa GradientEstimate ? (output..., result) : output
end

function _replace_active(ps, indices, active)
    θ = LenticulumCore._flatten_active(ps)
    return LenticulumCore._unflatten(_replace_coordinates(θ, indices, active), ps)
end

function _replace_coordinates(θ, indices, active)
    positions = Dict(i => k for (k, i) in enumerate(indices))
    return [haskey(positions, i) ? active[positions[i]] : θ[i] for i in eachindex(θ)]
end

# The nodes that warrant the warning, by name: unfrozen, learnable `learns_unnormalised` factors
# with a channel on a variable that a `DataFactor` observes; inside a cluster node, the clustered
# factors whose channels reach such a boundary variable.
function _unnormalised_nodes(g::FactorGraph)
    data = Set{Int}(g.edges[ei].variable for fn in g.factors if fn.factor isa DataFactor for ei in g.edges_of_factor[fn.id])
    isempty(data) && return Symbol[]
    names = Symbol[]
    for fn in g.factors
        (fn.freeze === :all || LenticulumCore.isfrozen(fn.factor) ||
         any(p -> !isempty(p) && first(p) === :A, LenticulumCore._freeze_paths(fn.freeze))) && continue
        vars = Dict(g.edges[ei].channel => g.edges[ei].variable for ei in g.edges_of_factor[fn.id])
        if fn.factor isa ClusterFactor
            c = fn.factor
            observed = Set(ch for (ch, v) in vars if v in data)
            # the cluster node's paths are the clustered nodes' under their keys: (key,) or (key, :A, ...)
            paths = LenticulumCore._freeze_paths(fn.freeze)
            frozenA(k) = any(p -> p[1] === k && (length(p) == 1 || p[2] === :A), paths)
            for (f, w, nm, k) in zip(c.factors, c.wiring, c.names, c.keys)
                obs = Set(p.first for p in w if p.second in observed)
                LenticulumCore.islearnable(f) && !frozenA(k) && learns_unnormalised(f, obs) && push!(names, Symbol(fn.name, "/", nm))
            end
        elseif LenticulumCore.islearnable(fn.factor) && learns_unnormalised(fn.factor, Set(ch for (ch, v) in vars if v in data))
            push!(names, fn.name)
        end
    end
    return names
end
function _check_unnormalised(g::FactorGraph)
    names = _unnormalised_nodes(g)
    isempty(names) && return nothing
    shown = join(first(names, 3), ", ") * (length(names) > 3 ? " and $(length(names) - 3) more" : "")
    @warn "free_energy_gradient: $shown learn a relation on observed data that is not normalised in its data " *
          "channels, so the free energy can fall without fitting (a constraint factor's A = 0 or an unnormalised " *
          "toric potential, unbounded below; a DEQ cell that is the identity in its observed state). Declare an " *
          "`output`, freeze the coefficient, normalise, or train by a loss on the solved state " *
          "(LP-022; vault/Energy-Based/Why Learned Relations Do Not Collapse.md)." maxlog = 1
    return nothing
end
