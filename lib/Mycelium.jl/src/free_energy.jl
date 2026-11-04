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

The counting number ``1 - d_v`` of a variable, where ``d_v`` counts its edges to
[`BeliefPropagation`](@ref) nodes, and to [`ExpectationPropagationRule`](@ref) nodes, which count the
same way. With every node on belief propagation that is the degree, and this is the Bethe
counting number.

A variable of degree 1 counts `0` (nothing to correct: only one factor mentions it). A
variable of degree 2 counts `-1` (its entropy was counted twice, so subtract one copy). And
so on.

Edges to [`MeanField`](@ref) nodes do not count: a mean-field node contributes an average energy
and no entropy, so the variable's own entropy is counted ``1 - |N_{BP}(v)|`` times, as in the
merged free energy of Riegler et al. (2013), eq. (17). A variable with mean-field neighbours only
counts `1`: its full entropy, as in naive mean field.
"""
counting_number(g::FactorGraph, vid::Int) = 1 - bp_degree(g, vid)

"""
    bp_degree(g, vid) -> Int

The number of edges from variable `vid` to [`BeliefPropagation`](@ref) and
[`ExpectationPropagationRule`](@ref) nodes, the nodes the Bethe free energy counts with an entropy.
"""
bp_degree(g::FactorGraph, vid::Int) =
    count(ei -> _cavity_rule(factornode(g, g.edges[ei].factor).rule), g.edges_of_variable[vid])

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

With [`MeanField`](@ref) nodes the counting numbers are those of the merged free energy
([`counting_number`](@ref)), and this sum is no longer the Euler characteristic.

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

A [`BeliefPropagation`](@ref) or [`ExpectationPropagationRule`](@ref) node contributes its [`local_free_energy`](@ref) at its incoming
messages; a [`MeanField`](@ref) node its [`mean_field_energy`](@ref) at the marginals of its
variables, an average energy with no entropy (Riegler et al. 2013, eq. 17).

Throws `ArgumentError` if a factor with [`uses_cavity`](@ref) has an `Emitting()` edge:
its local free energy needs variable-to-factor messages. Use `Bidirectional()` on that edge.
Missing messages on edges that can carry them remain `nothing`.
"""
function factor_free_energies(s::AbstractMessageStore, g::FactorGraph, ps, st)
    results, stnew = _factor_terms(s, g, ps, st)
    names = Tuple(f.name for f in g.factors)
    return LenticulumCore.GradedEnergy(NamedTuple{names}(Tuple(results))), stnew
end

# Each node's free energy, in node order, and the threaded state. A vector, not a NamedTuple:
# `scalar_free_energy` sums it without a type with one field per node (`plates.md`). A plate
# factor row whose messages are held as blocks is evaluated in one `local_free_energies` call.
function _factor_terms(s::AbstractMessageStore, g::FactorGraph, ps, st)
    _check_energy_edges(g)
    _check_tied_state(g, st)
    rows = _energy_rows(s, g)
    isempty(rows) || return _factor_terms_rows(s, g, ps, st, rows)
    results = map(_factor_nodes(g)) do f
        psf = _factor_parameters(f, _subtree(ps, f.params))
        stf = _subtree(st, f.params)
        f.rule isa MeanField && return mean_field_energy(f.factor, marginals_into(s, g, f.id), psf, stf)
        local_free_energy(f.factor, messages_into(s, g, f.id), psf, stf)
    end
    stnew = foldl(zip(g.factors,results); init=st) do acc, item
        _setsubtree(acc, item[1].params, item[2][2])
    end
    return map(first, results), stnew
end

# --- Batched energies over plates (L-ADR-0010, step 2) -------------------------------

"""
    local_free_energies(factors, msgs::NamedTuple, ps, st) -> (energies::AbstractVector, st)

The [`local_free_energy`](@ref) of each of several tied replicas at once: `factors[i]` is replica
`i`'s factor, sharing the parameters `ps`, and `msgs` holds one [`BeliefBlock`](@ref) of incoming
messages per channel (or `nothing` where no message is stored). `energies[i]` must be what
`local_free_energy` returns for replica `i`. The graph free energy calls this for each plate
factor row whose messages are held as blocks. The default evaluates each replica's column; a
factor type overrides it to share the work that depends only on the parameters.
"""
function local_free_energies(factors::AbstractVector, msgs::NamedTuple, ps, st)
    out = Vector{Any}(undef, length(factors))
    for i in eachindex(factors)
        out[i], st = local_free_energy(factors[i], _column(msgs, i), ps, st)
    end
    return out, st
end

"""
    factor_beliefs(factors, msgs::NamedTuple, ps, st) -> beliefs or nothing
    belief_free_energies(factors, beliefs, ps, st) -> (energies::AbstractVector, st)

The batched forms of [`factor_belief`](@ref) and [`belief_free_energy`](@ref), for the parameter
gradient over a plate: `factor_beliefs` evaluates the replicas' factor beliefs once, outside AD,
from message blocks as in [`local_free_energies`](@ref), and `belief_free_energies` returns each
replica's free energy at them, differentiable in `ps`. The defaults go replica by replica (a
vector of `factor_belief`s, `nothing` if a replica has none). A factor type overrides both
together, with a batched belief type of its own. `nothing` from `factor_beliefs` makes the
gradient take the per-replica path, including its LP-020 warning.
"""
function factor_beliefs(factors::AbstractVector, msgs::NamedTuple, ps, st)
    bs = [factor_belief(factors[i], _column(msgs, i), ps, st) for i in eachindex(factors)]
    return any(b -> b === nothing, bs) ? nothing : bs
end
"""
    belief_free_energies(factors, beliefs, ps, st) -> (energies::AbstractVector, st)

Each replica's [`belief_free_energy`](@ref) at the beliefs [`factor_beliefs`](@ref) returned,
differentiable in `ps`; see there.
"""
belief_free_energies(factors::AbstractVector, beliefs::AbstractVector, ps, st) =
    (map((f, b) -> first(belief_free_energy(f, b, ps, st)), factors, beliefs), st)

_column(msgs::NamedTuple, i) = NamedTuple{keys(msgs)}(map(b -> b === nothing ? nothing : replica(b, i), values(msgs)))

# A plate factor row the energies can take in one call: belief-propagation nodes on one parameter
# key, each with its edges in the same block rows in the same channel order, replica i in column
# i, no edge detached.
struct _EnergyRow
    fids::Vector{Int}
    channels::Tuple
    rows::Vector{Int}
end

function _energy_rows(s, g::FactorGraph)
    _has_rows(s, g) || return _EnergyRow[]
    out = _EnergyRow[]
    for pl in g.plates
        F::Matrix{Int} = pl.factors
        for k in axes(F, 1)
            r = _energy_row(g, s.to_factor, F[k, :])
            r === nothing || push!(out, r)
        end
    end
    return out
end

function _energy_row(g::FactorGraph, idx::EdgeMessages, fids::Vector{Int})
    f1 = g.factors[first(fids)]
    _cavity_rule(f1.rule) || return nothing          # local free energies at incoming messages
    E1 = g.edges_of_factor[f1.id]
    rows = Int[idx.row[e] for e in E1]
    all(!=(0), rows) || return nothing
    for (col, fid) in enumerate(fids)
        f = g.factors[fid]
        f.params === f1.params && f.rule === f1.rule || return nothing
        Ei = g.edges_of_factor[fid]
        length(Ei) == length(E1) || return nothing
        for j in eachindex(Ei)
            e = g.edges[Ei[j]]
            idx.row[Ei[j]] == rows[j] && idx.col[Ei[j]] == col && !e.detach &&
                e.channel === g.edges[E1[j]].channel || return nothing
        end
    end
    return _EnergyRow(fids, Tuple(g.edges[e].channel for e in E1), rows)
end

# The row's incoming messages as blocks, or `nothing` when a row of edges is partly filled.
function _row_messages(s, row::_EnergyRow)
    bs = map(r -> _row_block(s.to_factor, r), row.rows)
    any(b -> b === missing, bs) && return nothing
    return NamedTuple{row.channels}(Tuple(bs))
end

# The rows' energies in one call each and the other nodes' one by one. With `scalar = true` the
# result is the sum of the scalarised terms instead of the vector. A row's key is stateless when
# tied (`_check_tied_state`), and its state is set once.
function _factor_terms_rows(s, g, ps, st, rows; scalar::Bool = false)
    results = scalar ? nothing : Vector{Any}(nothing, nfactors(g))
    covered = falses(nfactors(g))
    total = 0.0
    for row in rows
        msgs = _row_messages(s, row)
        msgs === nothing && continue
        f1 = g.factors[first(row.fids)]
        psf = _factor_parameters(f1, _subtree(ps, f1.params))
        factors = identity.([g.factors[i].factor for i in row.fids])
        E, stf = local_free_energies(factors, msgs, psf, _subtree(st, f1.params))
        length(E) == length(row.fids) || throw(ArgumentError(
            "local_free_energies of $(nameof(typeof(f1.factor))) returned $(length(E)) energies for $(length(row.fids)) replicas"))
        if scalar
            total += _row_scalar(factors, E)
        else
            for (i, fid) in enumerate(row.fids)
                results[fid] = E[i]
            end
        end
        covered[row.fids] .= true
        st = _setsubtree(st, f1.params, stf)
    end
    for f in g.factors
        covered[f.id] && continue
        psf = _factor_parameters(f, _subtree(ps, f.params))
        stf = _subtree(st, f.params)
        Ff, stf = f.rule isa MeanField ? mean_field_energy(f.factor, marginals_into(s, g, f.id), psf, stf) :
                  local_free_energy(f.factor, messages_into(s, g, f.id), psf, stf)
        scalar ? (total += LenticulumCore.scalarise(LenticulumCore.scalarisation(f.factor), Ff)) : (results[f.id] = Ff)
        st = _setsubtree(st, f.params, stf)
    end
    return scalar ? total : results, st
end

# The sum of a row's scalarised energies; `factors` concretely typed, so this compiles per type.
_row_scalar(factors::AbstractVector, E::AbstractVector) =
    sum(i -> LenticulumCore.scalarise(LenticulumCore.scalarisation(factors[i]), E[i]), eachindex(factors, E); init = 0.0)

# The variable terms of a plate's variable rows in one block operation each: the marginal block
# (the combine of the rows of to-variable messages) and its entropies. Returns the total and the
# variables it covered; a row with a pinned variable, a partly filled row, or edges outside
# block rows is left to the per-variable path.
function _variable_terms_rows(s, g)
    total, done = 0.0, falses(nvariables(g))
    _has_rows(s, g) || return total, done
    bp = BitVector([_cavity_rule(f.rule) for f in g.factors])
    for pl in g.plates
        V::Matrix{Int} = pl.variables
        for k in axes(V, 1)
            vids = V[k, :]
            t = _variable_row_term(s, g, s.to_variable, bp, vids)
            t === nothing && continue
            total += t
            done[vids] .= true
        end
    end
    return total, done
end

# The edges of every variable of the row in the same block rows, replica i in column i, so that
# they also have one counting number (their factors are in the same factor rows).
function _variable_row_term(s, g::FactorGraph, idx::EdgeMessages, bp::BitVector, vids::Vector{Int})
    isempty(s.pins) || any(v -> _pin(s, v) !== nothing, vids) && return nothing
    E1 = g.edges_of_variable[first(vids)]
    rows = Int[idx.row[e] for e in E1]
    all(!=(0), rows) || return nothing
    for (i, v) in enumerate(vids)
        Ei = g.edges_of_variable[v]
        length(Ei) == length(E1) || return nothing
        for j in eachindex(Ei)
            idx.row[Ei[j]] == rows[j] && idx.col[Ei[j]] == i && bp[g.edges[Ei[j]].factor] == bp[g.edges[E1[j]].factor] ||
                return nothing
        end
    end
    c = counting_number(g, first(vids))
    c == 0 && return 0.0
    acc = TrivialBlock(length(vids))
    for r in rows
        b = _row_block(idx, r)
        b === nothing && continue
        b === missing && return nothing
        acc = combine_blocks(acc, b)
    end
    return c * sum(-, _block_entropies(acc))
end

# The entropy of each column, `variable_entropy` of its belief; one evaluation for a shared precision.
_block_entropies(b::PointBlock) = zeros(nreplicas(b))
_block_entropies(b::TrivialBlock) = zeros(nreplicas(b))
function _block_entropies(b::GaussianBlock)
    _shared(b) && all(==(b.improper[1]), b.improper) && return fill(variable_entropy(replica(b, 1)), nreplicas(b))
    return [variable_entropy(replica(b, i)) for i in 1:nreplicas(b)]
end
_block_entropies(b::CategoricalBlock) =
    [-sum(k -> b.logp[k, i] == -Inf ? 0.0 : exp(b.logp[k, i]) * b.logp[k, i], axes(b.logp, 1)) for i in axes(b.logp, 2)]
_block_entropies(b::BeliefBlock) = [variable_entropy(replica(b, i)) for i in 1:nreplicas(b)]

# The gradient's scalar for a plate row on batched beliefs: the replicas' scalarised free energies.
function _beliefs_scalar(ps, factors, beliefs, st)
    E, _ = belief_free_energies(factors, beliefs, ps, st)
    return sum(map((f, e) -> LenticulumCore.scalarise(LenticulumCore.scalarisation(f), e), factors, E))
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
    marginals_into(store, g, fid) -> NamedTuple

The current **marginals** of the variables a factor touches, keyed by channel: what a
[`MeanField`](@ref) node reads, both for its messages and for its [`mean_field_energy`](@ref).
Contrast [`messages_into`](@ref), the cavity messages a belief-propagation node reads.
"""
function marginals_into(s::AbstractMessageStore, g::FactorGraph, fid::Int)
    chans = Tuple(g.edges[ei].channel for ei in g.edges_of_factor[fid])
    return NamedTuple{chans}(Tuple(marginal(s, g, g.edges[ei].variable) for ei in g.edges_of_factor[fid]))
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

The per-variable counting terms ``c_v F_v = (1 - d_v)\\,(-H(b_v))``, graded by variable name, with
``d_v`` counting belief-propagation neighbours only ([`counting_number`](@ref)).
"""
function variable_corrections(s::MessageStore, g::FactorGraph)
    names = Tuple(v.name for v in g.variables)
    vals = Tuple(_variable_term(s, g, v.id) for v in g.variables)
    return LenticulumCore.GradedEnergy(NamedTuple{names}(vals))
end

# A variable of degree 1 counts 0, so its entropy is not needed, and is not evaluated: its
# marginal may be one with no entropy in closed form, such as a Gaussian mixture.
function _variable_term(s, g, vid)
    c = counting_number(g, vid)
    return c == 0 ? 0.0 : c * variable_free_energy(marginal(s, g, vid))
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

**With [`MeanField`](@ref) nodes** it is the merged free energy of Riegler et al. (2013), eq. (17):
mean-field nodes contribute [`mean_field_energy`](@ref) and the variable counting numbers
count belief-propagation neighbours only. Its stationary points are the fixed points of the
hybrid message passing (their Theorem 2). When the belief-propagation part is a forest and the
mean-field nodes split it into independent clusters, it is ``\\mathrm{KL}(q \\| p) - \\log Z``
for the factorised ``q``, an upper bound on ``-\\log Z``.

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
    rows = _energy_rows(s, g)
    total = 0.0
    if isempty(rows)
        F, st = _factor_terms(s, g, ps, st)
        for (f, Ff) in zip(g.factors, F)
            total += LenticulumCore.scalarise(LenticulumCore.scalarisation(f.factor), Ff)
        end
    else
        _check_energy_edges(g)
        _check_tied_state(g, st)
        total, st = _factor_terms_rows(s, g, ps, st, rows; scalar = true)
    end
    vtotal, done = _variable_terms_rows(s, g)
    total += vtotal
    for v in g.variables
        done[v.id] && continue
        total += _variable_term(s, g, v.id)
    end
    return total, st
end

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
function _row_items(s, g, key, p, stf, nodes)
    items = Any[]
    covered = Set{Int}()
    for row in _energy_rows(s, g)
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
    adjoint in (:unrolled, :implicit) || throw(ArgumentError("adjoint must be :unrolled or :implicit"))
    coupled = adjoint === :implicit || any(e -> !(e.coupling isa DiagonalCoupling), g.edges)
    # LP-021, option 3: without a replay (no `schedule`) the couplings do not change the free-energy
    # gradient, which at a fixed point is the envelope one; they matter for a replay only.
    if coupled && !(adjoint === :unrolled && schedule === nothing && _deterministic_couplings(g))
        return _coupled_gradient(s, g, ps, st; ad, ε, schedule, nsweeps, rng, nsamples,
            damping, adjoint, maxsweeps, tol, linear_solver, linear_tol, linear_maxiter)
    end
    return _envelope_gradient(s, g, ps, st; ad, ε)
end

# The free-energy gradient at the store's beliefs (LP-020): each factor's term differentiated at
# its fixed belief, tied nodes summed.
function _envelope_gradient(s, g, ps, st; ad = nothing, ε = 1e-6)
    keys_ = parameter_keys(g)
    grads = map(keys_) do key
        nodes = [g.factors[i] for i in tied_factors(g, key)]
        f = first(nodes)
        p = _subtree(ps, key)
        if !LenticulumCore.islearnable(f.factor)
            θ = LenticulumCore._flatten(p)
            return LenticulumCore._unflatten(zero(θ), p)
        end
        stf = _subtree(st, key)
        LenticulumCore._check_factor_ad(ad, f.factor, stf)
        items, rest = _row_items(s, g, key, p, stf, nodes)
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
    parameter_step(store, g, ps, st; ad, rule = GradientDescent(0.01), optimiser_state = nothing)
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
                        ad, rule = GradientDescent(0.01), optimiser_state = nothing,
                        kwargs...)
    fs = Tuple(f for f in map(k -> _key_node(g, k), parameter_keys(g))
               if !isempty(_active_indices(f, _subtree(ps, f.params))))
    names = Tuple(f.params for f in fs)
    indices = map(f -> _active_indices(f, _subtree(ps, f.params)), fs)
    θs = map((f, ii) -> _gather_active(LenticulumCore._flatten(_subtree(ps, f.params)), ii), fs, indices)
    os = optimiser_state === nothing ? NamedTuple{names}(map(zero, θs)) : optimiser_state
    for (f, θ) in zip(fs, θs)
        haskey(os,f.params) && length(os[f.params]) == length(θ) || throw(ArgumentError(
            "optimiser_state for :$(f.params) must contain exactly $(length(θ)) active coordinates; reinitialise it after changing freeze paths"))
    end
    look = map(fs, θs, indices) do f, θ, ii
        _replace_active(_subtree(ps, f.params), ii, rule_get(rule, os[f.params], θ))
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
        rule_put(rule, os[f.params], θ, _gather_active(LenticulumCore._flatten(grad[f.params]), ii))
    end
    newps = map(fs, updates, indices) do f, u, ii
        _replace_active(_subtree(ps, f.params), ii, u[2])
    end
    output = (merge(ps, NamedTuple{names}(newps)), stnew,
              NamedTuple{names}(map(first, updates)))
    return result isa GradientEstimate ? (output..., result) : output
end

function _replace_active(ps, indices, active)
    θ = LenticulumCore._flatten_active(ps)
    return LenticulumCore._unflatten(_replace_coordinates(θ, indices, active), ps)
end

# Tied nodes read one state entry; with a non-empty one, which node's update is kept would depend
# on evaluation order, so it is refused (`plates.md`).
function _check_tied_state(g, st)
    st isa NamedTuple || return nothing
    for k in parameter_keys(g)
        count(f -> f.params === k, g.factors) > 1 || continue
        _empty_graph_state(_subtree(st, k)) || throw(ArgumentError(
            "the factors tied to parameters :$k have a non-empty state; tied factors must be stateless"))
    end
    return nothing
end

# `mean_field = false` for the paths that differentiate through inference or record residual
# moments from cavity messages; they are written for belief propagation only. The default paths
# also take mean-field nodes and expectation-propagation nodes (counted like belief propagation:
# their free energy is the expectation-consistent Bethe free energy, `passing.md`); MixtureReduction()
# nodes are refused everywhere, since no free energy has their fixed points as stationary points.
function _check_energy_edges(g; mean_field::Bool = true, what = "this computation")
    for f in g.factors
        f.rule isa BeliefPropagation || (mean_field && (f.rule isa MeanField || f.rule isa ExpectationPropagationRule)) ||
            throw(ArgumentError(
            "factor :$(f.name) uses $(nameof(typeof(f.rule)))() messages; $what is implemented for \
             belief-propagation $(mean_field ? ", mean-field and expectation-propagation " : "")nodes only, \
             so it is refused rather than misreported"))
    end
    for f in g.factors
        f.rule isa MeanField && continue      # reads marginals, which every edge direction has
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
