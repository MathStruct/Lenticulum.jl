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
variable_entropy(b) = throw(ArgumentError(
    "no `variable_entropy` for $(typeof(b)). The Bethe counting correction needs the \
     entropy of each variable marginal; implement it for your belief representation."))

"""
    factor_free_energies(store, g, ps, st) -> (GradedEnergy, st)

Each factor's **vector** free energy ``\\mathbf{F}^f``, graded by factor name.

This is ``E_G = \\bigoplus_f E_f`` from `Scalar and Multivariate Energy.md`, realised: the
composite energy of a graph *is* the per-factor breakdown, not a number that a logger later
decomposes.
"""
function factor_free_energies(s::MessageStore, g::FactorGraph, ps, st)
    names = Tuple(f.name for f in g.factors)
    vals = Any[]
    for f in g.factors
        beliefs = messages_into(s, g, f.id)
        psf = _subtree(ps, f.name)
        stf = _subtree(st, f.name)
        F, stf = local_free_energy(f.factor, beliefs, psf, stf)
        push!(vals, F)
        st = _setsubtree(st, f.name, stf)
    end
    return LenticulumCore.GradedEnergy(NamedTuple{names}(Tuple(vals))), st
end

"""
    messages_into(store, g, fid) -> NamedTuple

The **incoming messages** ``\\mu_{i \\to a}`` of each variable this factor touches, keyed by
channel. `nothing` for a channel whose edge cannot carry one.

> [!warning] These are messages, not marginals — and the difference is the whole formula
> The Bethe factor belief is ``b_a \\propto f_a \\prod_{i \\in a}\\mu_{i\\to a}``: the
> factor's own potential times the messages *into* it. Using the variable marginals instead
> would multiply in the factor's own outgoing message as well, counting its evidence twice.
>
> An earlier version of this file did exactly that. It was invisible until a factor with a
> non-trivial entropy existed to expose it — see `free_energy.md` §7.
"""
function messages_into(s::MessageStore, g::FactorGraph, fid::Int)
    chans = Tuple(g.edges[ei].channel for ei in g.edges_of_factor[fid])
    vals = Tuple(
        (m = s.to_factor[ei]; m === nothing ? nothing : m.belief)
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
\\mathbf{F}_{\\text{Bethe}} = \\bigoplus_f \\mathbf{F}^f \\ \\oplus\\ \\bigoplus_v (1-d_v)\\,\\mathbf{H}^v
```

**Exact on a tree.** On a loopy graph it is the Bethe approximation, whose stationary points
are exactly the fixed points of loopy belief propagation (Yedidia–Freeman–Weiss) — so running
BP and minimising this quantity are the same activity, which is a genuinely useful thing to
know when the two appear to disagree.

Contrast [`chain_free_energy`](@ref), which follows AutoBayes Theorem 23 literally and is
**schedule-dependent**.
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
