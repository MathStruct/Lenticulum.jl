# ---------------------------------------------------------------------------
# Free energy on a graph.
#
# AutoBayes Theorem 23 gives the chain rule for a SEQUENCE of factors. A graph is not a
# sequence, and the generalisation is the BETHE free energy:
#
#     U = Σ_f U_f                                    energies just ADD
#     H = Σ_f H_f  -  Σ_v (d_v - 1) H_v              entropies need a COUNTING CORRECTION
#
# That is the SAME energy/entropy asymmetry AutoBayes identifies, appearing again one level
# up. It is the strongest independent confirmation of the paper's central claim that I know
# of, and it is why this file exists rather than a naive `sum(free_energy, factors)`.
#
# See `free_energy.md` and `Bethe Free Energy.md`.
# ---------------------------------------------------------------------------

"""
    counting_number(g, vid) -> Int

The Bethe counting number ``1 - d_v`` of a variable, where ``d_v`` is its degree.

A variable of degree 1 counts `0` (nothing to correct: only one factor mentions it). A
variable of degree 2 counts `-1` (its entropy was counted twice, so subtract one copy). And
so on.
"""
counting_number(g::FactorGraph, vid::Int) = 1 - variable_degree(g, vid)

"""
    counting_numbers(g) -> NamedTuple

Every variable's counting number, keyed by name.
"""
counting_numbers(g::FactorGraph) = NamedTuple{Tuple(v.name for v in g.variables)}(
    Tuple(counting_number(g, v.id) for v in g.variables))

"""
    total_counting_number(g) -> Int

``\\sum_f 1 + \\sum_v (1 - d_v)``.

> **This equals the Euler characteristic** ``|F| + |V| - |E|``, because
> ``\\sum_v d_v = |E|``. So it is `1` exactly when the graph is a connected tree, and
> `1 - L` when it has `L` independent loops.
>
> That single integer measures how badly the Bethe bookkeeping can be wrong: on a tree the
> counting is exact, and the deficit from `1` is the number of loops the approximation has
> to pretend are not there.

Asserted against [`euler_characteristic`](@ref) in the test suite, since it is the cheapest
possible check that the graph and the free-energy accounting agree.
"""
total_counting_number(g::FactorGraph) =
    nfactors(g) + sum(v -> counting_number(g, v.id), g.variables; init = 0)

"""
    variable_entropy(belief) -> Real

The entropy ``H(b_v)`` of a variable's marginal, needed for the counting correction.

Implemented only where it is unambiguous: a `DiracBelief` has entropy `0` (differential
entropy of a point mass is ``-\\infty``, but the *correction term* it contributes is zero
because a clamped variable carries no free bits — this is the convention BP uses for
observed nodes). Anything else throws rather than guess.
"""
variable_entropy(::LenticulumCore.DiracBelief) = 0.0
variable_entropy(::LenticulumCore.TrivialBelief) = 0.0
variable_entropy(b::LenticulumCore.CategoricalBelief) =
    -sum(l == -Inf ? 0.0 : exp(l) * l for l in b.logp)   # Shannon entropy, 0 log 0 = 0
function variable_entropy(m::LenticulumCore.MixtureBelief{<:LenticulumCore.CategoricalBelief})
    # Exact Shannon entropy of the marginalized categorical distribution
    isempty(m.components) && return 0.0
    c1 = first(m.components)
    n = length(c1.logp)
    w = LenticulumCore.mixture_weights(m)
    p = zeros(Float64, n)
    for (wk, ck) in zip(w, m.components)
        _same_support(c1, ck) || throw(ArgumentError("mixture components must share support"))
        p .+= wk .* LenticulumCore.probabilities(ck)
    end
    return -sum(pk <= 0 ? 0.0 : pk * log(pk) for pk in p)
end
variable_entropy(b) = throw(ArgumentError(
    "no `variable_entropy` for $(typeof(b)). The Bethe counting correction needs the \
     entropy of each variable marginal; implement it for your belief representation."))

"""
    factor_free_energies(store, g, ps, st) -> (GradedEnergy, st)

Each factor's **vector** free energy ``\\mathbf{F}^f``, graded by factor name.

This is ``E_G = \\bigoplus_f E_f`` from `Scalar and Multivariate Energy.md`, realised: the
composite energy of a graph *is* the per-factor breakdown, not a number that a logger later
decomposes.

Throws `ArgumentError` if a factor with [`uses_cavity`](@ref) has an `Emitting()` edge:
its local free energy needs variable-to-factor messages. Use `Bidirectional()` on that edge.
Missing messages on edges that can carry them remain `nothing`.
"""
function factor_free_energies(s::AbstractMessageStore, g::FactorGraph, ps, st)
    _check_energy_edges(g)
    names = Tuple(f.name for f in g.factors)
    results = map(_factor_nodes(g)) do f
        beliefs = messages_into(s, g, f.id)
        psf = _factor_parameters(f, _subtree(ps, f.name))
        stf = _subtree(st, f.name)
        local_free_energy(f.factor, beliefs, psf, stf)
    end
    stnew = foldl(zip(names,results); init=st) do acc, item
        _setsubtree(acc, item[1], item[2][2])
    end
    return LenticulumCore.GradedEnergy(NamedTuple{names}(map(first,results))), stnew
end

"""
    messages_into(store, g, fid) -> NamedTuple

The **incoming messages** ``\\mu_{i \\to a}`` of each variable this factor touches, keyed by
channel. `nothing` when no message is stored, including on edges that cannot carry one.
[`factor_free_energies`](@ref) rejects such edge directions for factors with
[`uses_cavity`](@ref); it does not reject a missing message on a permitted edge.

> [!warning] These are messages, not marginals — and the difference is the whole formula
> The Bethe factor belief is ``b_a \\propto f_a \\prod_{i \\in a}\\mu_{i\\to a}``: the
> factor's own potential times the messages *into* it. Using the variable marginals instead
> would multiply in the factor's own outgoing message as well, counting its evidence twice.
>
> An earlier version of this file did exactly that. It was invisible until a factor with a
> non-trivial entropy existed to expose it — see `free_energy.md` §7.
"""
function messages_into(s::AbstractMessageStore, g::FactorGraph, fid::Int)
    chans = Tuple(g.edges[ei].channel for ei in g.edges_of_factor[fid])
    vals = Tuple(
        (m = s.to_factor[ei]; m === nothing ? nothing :
            g.edges[ei].detach ? LenticulumCore.stop_gradient(m.belief) : m.belief)
        for ei in g.edges_of_factor[fid]
    )
    return NamedTuple{chans}(vals)
end

"""
    variable_free_energy(belief) -> Real

``F_v = -H(b_v)``. A variable has no energy of its own — it is a wire, not a node with a
potential — so its free energy is *minus* its entropy.

> [!warning] The sign here is the whole correction, and it was wrong
> The Bethe free energy is ``F_\\beta = \\sum_\\alpha c_\\alpha F_\\alpha`` over factors
> **and** variables, with ``c_f = 1`` and ``c_v = 1 - d_v``. Because ``F_v = -H_v``, the
> variable term is ``\\sum_v (1-d_v)(-H_v) = +\\sum_v (d_v - 1)H_v`` — it **adds back** the
> entropy that the factor terms subtracted ``d_v`` times.
>
> An earlier version applied ``c_v`` to ``+H_v``, flipping the sign. Every test passed, because
> every belief in them was a `DiracBelief` with ``H_v = 0``. See `free_energy.md` §6.
"""
variable_free_energy(b) = -variable_entropy(b)

"""
    variable_corrections(store, g) -> GradedEnergy

The per-variable counting terms ``c_v F_v = (1 - d_v)\\,(-H(b_v))``, graded by variable name.
"""
function variable_corrections(s::MessageStore, g::FactorGraph)
    names = Tuple(v.name for v in g.variables)
    vals = Tuple(counting_number(g, v.id) * variable_free_energy(marginal(s, g, v.id))
                 for v in g.variables)
    return LenticulumCore.GradedEnergy(NamedTuple{names}(vals))
end

"""
    bethe_free_energy(store, g, ps, st) -> (GradedEnergy, st)

The graph free energy, graded into `(factors = …, variables = …)`:

```math
\\mathbf{F}_{\\text{Bethe}} = \\bigoplus_f \\mathbf{F}^f \\ \\oplus\\ \\bigoplus_v (1-d_v)\\,(-\\mathbf{H}^v)
```

**Exact on a tree.** On a loopy graph it is the Bethe approximation, whose stationary points
are exactly the fixed points of loopy belief propagation (Yedidia–Freeman–Weiss) — so running
BP and minimising this quantity are the same activity, which is a genuinely useful thing to
know when the two appear to disagree.

Contrast [`chain_free_energy`](@ref), which follows AutoBayes Theorem 23 literally and is
**schedule-dependent**.

Uses [`factor_free_energies`](@ref), including its cavity edge-direction check.
"""
function bethe_free_energy(s::MessageStore, g::FactorGraph, ps, st)
    F, st = factor_free_energies(s, g, ps, st)
    V = variable_corrections(s, g)
    return LenticulumCore.GradedEnergy((; factors = F, variables = V)), st
end

"""
    chain_free_energy(Fs) -> GradedEnergy

AutoBayes Theorem 23 applied along a linear order:
``\\mathbf{F}^{dc} = (\\mathbb{E}[\\mathbf{F}^c],\\ \\mathbf{F}^d)`` folded right to left.

`Fs` is the ordered vector of per-factor vector free energies. Uses
`LenticulumCore.compose_free_energy`, so the nesting matches what
`Composition of Statistical Games.md` describes.

> [!warning] This is schedule-dependent and Bethe is not
> Theorem 23's recursion refers to "downstream", which on a graph is a property of the
> *message order*, not of the wiring. Two different schedules give two different
> decompositions of the same total. Bethe's form has no order in it at all.
>
> They agree when every belief is a `DiracBelief` (deterministic inference: all
> expectations are evaluations and all variable entropies vanish), which is exactly the
> regime the test suite checks. Off it, prefer Bethe for reporting and the chain form for
> reasoning about a specific message path.
"""
function chain_free_energy(Fs::AbstractVector)
    isempty(Fs) && throw(ArgumentError("no factors"))
    length(Fs) == 1 && return Fs[1]
    acc = Fs[end]
    for i in (length(Fs) - 1):-1:1
        acc = LenticulumCore.compose_free_energy([Fs[i]], acc)
    end
    return acc
end

"""
    LenticulumCore.scalar_free_energy(store, g, ps, st) -> (Real, st)

The scalarised Bethe free energy: what an optimiser minimises.

This **extends** `LenticulumCore.scalar_free_energy` rather than defining a new function of
the same name: "the scalar free energy" of a graph and of a factor are the same concept at
two scales, and having two exported bindings with that name would force every downstream user
to disambiguate. (The test suite caught exactly that.)

Each factor's own `scalarisation` is applied to its own summand, per
`GradedScalarisation` — so a residual factor can use a squared norm while a
likelihood factor uses the identity, in the same graph. See
`Scalar and Multivariate Energy.md` §5 for when that is strict and when it is lax.

Uses [`factor_free_energies`](@ref), including its cavity edge-direction check.
"""
function LenticulumCore.scalar_free_energy(s::MessageStore, g::FactorGraph, ps, st)
    F, st = factor_free_energies(s, g, ps, st)
    total = 0.0
    for f in g.factors
        σ = LenticulumCore.scalarisation(f.factor)
        total += LenticulumCore.scalarise(σ, F[f.name])
    end
    for v in g.variables
        total += counting_number(g, v.id) * variable_free_energy(marginal(s, g, v.id))
    end
    return total, st
end

# Snapshot beliefs outside AD: training is a partial derivative at fixed messages,
# not differentiation through the mutating inference schedule.
_local_scalar(ps, factor, beliefs, st) = LenticulumCore.scalarise(
    LenticulumCore.scalarisation(factor), first(local_free_energy(factor, beliefs, ps, st)))

"""
    LenticulumCore.free_energy_gradient(store, g, ps, st; ad = nothing, ε = 1e-6,
        schedule = nothing, nsweeps = 1, damping = 0.0, adjoint = :unrolled,
        maxsweeps = 1000, tol = 1e-10, linear_solver = :dense,
        linear_tol = 1e-10, linear_maxiter = 1000)

Parameter gradient of the graph scalar free energy at fixed incoming messages.
This is the sum of scalarised local free energies; Bethe variable corrections are
constant with respect to parameters at fixed beliefs. Nonlearnable subtrees receive
zeros. Frozen coordinates receive zeros too. The result has `ps`'s structure.
Non-diagonal edges require `schedule` and finite `nsweeps` (default 1), replaying from
empty messages rather than differentiating this store's history. Score edges also
require `rng` and `nsamples >= 2`, returning a `GradientEstimate`. No input store or
state is mutated. Replay includes flooding snapshots, realised residual task choices
and outgoing damping. `adjoint=:implicit` instead differentiates a converged synchronous
BP fixed point, found from empty messages using `schedule`, `maxsweeps` and `tol`.
It requires CPU Gaussian/Dirac messages, empty message state and an invertible
`I-DT`; diagonal/detach rows are constant cuts. Dense is the reference linear solve;
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
    adjoint in (:unrolled, :implicit) || throw(ArgumentError("adjoint must be :unrolled or :implicit"))
    if adjoint === :implicit || any(e -> !(e.coupling isa DiagonalCoupling), g.edges)
        return _coupled_gradient(s, g, ps, st; ad, ε, schedule, nsweeps, rng, nsamples,
            damping, adjoint, maxsweeps, tol, linear_solver, linear_tol, linear_maxiter)
    end
    names = Tuple(f.name for f in g.factors)
    grads = map(Tuple(g.factors)) do f
        p = _subtree(ps, f.name)
        if !LenticulumCore.islearnable(f.factor)
            θ = LenticulumCore._flatten(p)
            return LenticulumCore._unflatten(zero(θ), p)
        end
        LenticulumCore._check_factor_ad(ad, f.factor, _subtree(st, f.name))
        grad = LenticulumCore._parameter_gradient(ad, _local_scalar, p,
            f.factor, messages_into(s, g, f.id), _subtree(st, f.name); ε)
        indices = _active_indices(f, p)
        θ = LenticulumCore._flatten(grad)
        return LenticulumCore._unflatten(_mask_gradient(θ, indices), p)
    end
    return NamedTuple{names}(grads)
end

"""
    parameter_step(store, g, ps, st; ad, rule = GradientDescent(0.01), optimiser_state = nothing)
        -> (ps_new, st_new, optimiser_state_new)

One graph free-energy parameter step with an explicit AD backend (`nothing` selects
reference differences). Inference must populate incoming messages first. The rule's
`get` supplies a look-ahead parameter vector per learnable factor, the gradient is
computed at those parameters with messages fixed, and `put` applies the update.
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
                        ad, rule = GradientDescent(0.01), optimiser_state = nothing,
                        kwargs...)
    fs = Tuple(f for f in g.factors if !isempty(_active_indices(f, _subtree(ps, f.name))))
    names = Tuple(f.name for f in fs)
    indices = map(f -> _active_indices(f, _subtree(ps, f.name)), fs)
    θs = map((f, ii) -> _gather_active(LenticulumCore._flatten(_subtree(ps, f.name)), ii), fs, indices)
    os = optimiser_state === nothing ? NamedTuple{names}(map(zero, θs)) : optimiser_state
    for (f, θ) in zip(fs, θs)
        haskey(os,f.name) && length(os[f.name]) == length(θ) || throw(ArgumentError(
            "optimiser_state for :$(f.name) must contain exactly $(length(θ)) active coordinates; reinitialise it after changing freeze paths"))
    end
    look = map(fs, θs, indices) do f, θ, ii
        _replace_active(_subtree(ps, f.name), ii, rule_get(rule, os[f.name], θ))
    end
    at = merge(ps, NamedTuple{names}(look))
    coupled = get(kwargs, :adjoint, :unrolled) === :implicit || any(e -> !(e.coupling isa DiagonalCoupling), g.edges)
    if coupled
        result, stnew = _coupled_gradient(s,g,at,st;ad,return_state=true,kwargs...)
    else
        result = LenticulumCore.free_energy_gradient(s, g, at, st; ad, kwargs...)
        _, stnew = factor_free_energies(s, g, at, st)
    end
    grad = result isa GradientEstimate ? result.value : result
    updates = map(fs, θs, indices) do f, θ, ii
        rule_put(rule, os[f.name], θ, _gather_active(LenticulumCore._flatten(grad[f.name]), ii))
    end
    newps = map(fs, updates, indices) do f, u, ii
        _replace_active(_subtree(ps, f.name), ii, u[2])
    end
    output = (merge(ps, NamedTuple{names}(newps)), stnew,
              NamedTuple{names}(map(first, updates)))
    return result isa GradientEstimate ? (output..., result) : output
end

function _replace_active(ps, indices, active)
    θ = LenticulumCore._flatten_active(ps)
    return LenticulumCore._unflatten(_replace_coordinates(θ, indices, active), ps)
end

function _check_energy_edges(g)
    for f in g.factors
        if uses_cavity(f.factor)
            for ei in g.edges_of_factor[f.id]
                edge = g.edges[ei]
                absorbs(edge.direction) || throw(ArgumentError(
                    "factor :$(f.name) ($(typeof(f.factor))) uses cavity messages for its free energy, " *
                    "but channel :$(edge.channel) on variable :$(g.variables[edge.variable].name) " *
                    "has an Emitting() edge; use Bidirectional() (LP-014)"))
            end
        end
    end
    return nothing
end

_gather_active(θ, indices) = θ[indices]
_mask_gradient(θ, indices) = [i in indices ? θ[i] : zero(θ[i]) for i in eachindex(θ)]
function _replace_coordinates(θ, indices, active)
    positions = Dict(i => k for (k, i) in enumerate(indices))
    return [haskey(positions, i) ? active[positions[i]] : θ[i] for i in eachindex(θ)]
end

_index_metadata(θ, indices) = indices
