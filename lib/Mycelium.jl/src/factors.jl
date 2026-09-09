# ---------------------------------------------------------------------------
# Structural factors: data, priors, losses, optimisers, relays.
#
# Everything in a Lenticulum graph is a factor. That is not a slogan — it follows from
# AutoBayes Remark 24, which shows that a PRIOR must be a separate factor (with trivial
# inversion, energy -log p, zero entropy) for the composite loss to be the true variational
# free energy. Once priors are factors, data (a Dirac prior) and losses (a factor with energy
# and no inversion) are too; and once parameters can be exposed as channels (Example 3),
# optimisers are factors as well.
#
# See `factors.md` and `Everything is a Factor.md`.
# ---------------------------------------------------------------------------

"""
    local_free_energy(factor, beliefs::NamedTuple, ps, st) -> (𝐅, st)

The factor's **vector** free energy, evaluated at the current beliefs of its neighbouring
variables (keyed by channel).

This is the graph-level entry point corresponding to
`LenticulumCore.free_energy(factor, π, y, ps, st)`; it differs in taking *all* the factor's
channels at once rather than a prior/observation split, because in a graph there is no
distinguished split until a polarity is chosen — and the free energy is a property of the
factor, not of any one message.
"""
function local_free_energy end

local_free_energy(f, beliefs, ps, st) = throw(ArgumentError(
    "no `local_free_energy` for $(typeof(f)). Every factor in a graph must be able to \
     report its own vector free energy; see factors.md."))

"""
    issink(factor) -> Bool

A factor with **no** supported polarity: it contributes energy but never sends a message.

Losses, monitors and pure observations are sinks. This is a third case beyond
`LenticulumCore.isunidirectional` (exactly one polarity) and bidirectional (two or more), and
it is the one that corresponds to Cruttwell et al.'s *learning-rate cap* — the lens
`(L,L') → (1,1)` that terminates a wire.
"""
issink(f) = isempty(LenticulumCore.supported_polarities(f))

"""
    point(belief)

The value of a `DiracBelief`. Energy functions are pointwise
(`Statistical Game.md`, Definition 20), so they need a point, and only a Dirac provides one
unambiguously. Anything else throws rather than silently taking a mean.
"""
point(b::LenticulumCore.DiracBelief) = b.value
point(b) = throw(ArgumentError(
    "cannot extract a point from $(typeof(b)). Pointwise energies need a sample; draw one \
     from the belief, or use an energy that accepts a distribution."))

_polarity(names::Tuple, vals::Tuple) = LenticulumCore.Polarity(NamedTuple{names}(vals))

# =========================================================================== #
# Data
# =========================================================================== #

"""
    DataFactor(channel, value)

A clamped observation: arity one, **emitting only**, non-learnable.

This is the categorical **cup** of `Copiers Cups and Caps.md` and the ``\\rho_{in} = \\infty``
hard clamp of `Channels and Polarity.md`, and it is why `README.md`'s "training data are
variables" is refined here to *training data are factors attached to variables*: a variable
is a wire and has no content of its own, whereas data is evidence, and evidence is a factor.

Energy is `0` and entropy is `0`: a Dirac prior contributes ``-\\log p_\\pi`` which vanishes
at its own point.
"""
struct DataFactor{T} <: LenticulumCore.AbstractLenticulumFactor
    channel::Symbol
    value::T
end

LenticulumCore.channels(f::DataFactor) = (LenticulumCore.Channel(f.channel, nothing),)
LenticulumCore.supported_polarities(f::DataFactor) =
    (_polarity((f.channel,), (LenticulumCore.Unobserved(),)),)
LenticulumCore.supports_polarity(f::DataFactor, p::LenticulumCore.Polarity) =
    keys(p) == (f.channel,) && p[f.channel] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(::DataFactor) = false

factor_message(f::DataFactor, ::Symbol, _, _, _, ps, st) =
    (LenticulumCore.DiracBelief(f.value), st)
local_free_energy(::DataFactor, _, ps, st) = (0.0, st)

# =========================================================================== #
# Prior
# =========================================================================== #

"""
    PriorFactor(channel, belief, nlogp)

A prior over one variable: emits `belief`, and charges energy `nlogp(x)` at the variable's
current point.

This is AutoBayes Remark 24 exactly: give the prior a trivial inversion, set
``l^\\pi(x) = -\\log p_\\pi(x)`` and ``H^\\pi \\equiv 0``, and the composite with a downstream
factor has the **true variational free energy** as its loss. Without a prior factor you get
an "open free energy", missing the ``-\\log p_\\pi(x)`` term.

Learnable if `nlogp` carries parameters — set `learnable = true` and implement
`initialparameters`.
"""
struct PriorFactor{B,E} <: LenticulumCore.AbstractLenticulumFactor
    channel::Symbol
    belief::B
    nlogp::E
    learnable::Bool
end
PriorFactor(channel, belief, nlogp; learnable::Bool = false) =
    PriorFactor(channel, belief, nlogp, learnable)

LenticulumCore.channels(f::PriorFactor) = (LenticulumCore.Channel(f.channel, nothing),)
LenticulumCore.supported_polarities(f::PriorFactor) =
    (_polarity((f.channel,), (LenticulumCore.Unobserved(),)),)
LenticulumCore.supports_polarity(f::PriorFactor, p::LenticulumCore.Polarity) =
    keys(p) == (f.channel,) && p[f.channel] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(f::PriorFactor) = f.learnable

factor_message(f::PriorFactor, ::Symbol, _, _, _, ps, st) = (f.belief, st)
local_free_energy(f::PriorFactor, beliefs, ps, st) =
    (f.nlogp(point(getfield(beliefs, f.channel))), st)

# =========================================================================== #
# Loss
# =========================================================================== #

"""
    LossFactor(channels, loss)

A **sink**: absorbs on every channel, emits nothing, is not learnable, and contributes
`loss(values...)` as its energy.

`loss` may return a scalar **or a vector** — the latter being the multivariate energy of
`Scalar and Multivariate Energy.md`. Pair it with a non-identity `scalarisation` to control
how the vector collapses.

Note that Cruttwell et al.'s Definition 3.3 makes the *label* the loss map's parameter; here
the label is simply another absorbed channel, which is the same statement with the graph
doing the bookkeeping.
"""
struct LossFactor{L,S<:LenticulumCore.AbstractScalarisation,E} <:
       LenticulumCore.AbstractLenticulumFactor
    chans::Tuple{Vararg{Symbol}}
    loss::L
    scalarisation::S
    energyspace::E
end
function LossFactor(chans, loss;
                    scalarisation = LenticulumCore.IdentityScalarisation(),
                    energyspace = LenticulumCore.ScalarEnergySpace())
    return LossFactor(Tuple(chans), loss, scalarisation, energyspace)
end

LenticulumCore.channels(f::LossFactor) =
    Tuple(LenticulumCore.Channel(c, nothing) for c in f.chans)
LenticulumCore.supported_polarities(::LossFactor) = ()   # a sink
LenticulumCore.supports_polarity(::LossFactor, ::LenticulumCore.Polarity) = false
LenticulumCore.islearnable(::LossFactor) = false
LenticulumCore.scalarisation(f::LossFactor) = f.scalarisation
LenticulumCore.energyspace(f::LossFactor) = f.energyspace

local_free_energy(f::LossFactor, beliefs, ps, st) =
    (f.loss(map(c -> point(getfield(beliefs, c)), f.chans)...), st)

# =========================================================================== #
# Relay (the identity game)
# =========================================================================== #

"""
    RelayFactor(a, b)

A bidirectional pass-through: whatever arrives on one channel leaves on the other.

This is the **identity game** of AutoBayes Remark 24 — the identity lens with constantly zero
energy and entropy — and it is the sanity check that inserting a node into a graph changes
nothing. It is also the smallest factor that genuinely has two polarities, so it is what the
polarity machinery is tested against.
"""
struct RelayFactor <: LenticulumCore.AbstractLenticulumFactor
    a::Symbol
    b::Symbol
end

LenticulumCore.channels(f::RelayFactor) =
    (LenticulumCore.Channel(f.a, nothing), LenticulumCore.Channel(f.b, nothing))
function LenticulumCore.supported_polarities(f::RelayFactor)
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    return (_polarity((f.a, f.b), (O, U)), _polarity((f.a, f.b), (U, O)))
end
function LenticulumCore.supports_polarity(f::RelayFactor, p::LenticulumCore.Polarity)
    keys(p) == (f.a, f.b) || return false
    pa, pb = p[f.a], p[f.b]
    return (pa isa LenticulumCore.Observed && pb isa LenticulumCore.Unobserved) ||
           (pa isa LenticulumCore.Unobserved && pb isa LenticulumCore.Observed)
end
LenticulumCore.islearnable(::RelayFactor) = false

function factor_message(f::RelayFactor, target::Symbol, _, inputs, _, ps, st)
    src = target === f.a ? f.b : f.a
    haskey(inputs, src) || return (LenticulumCore.TrivialBelief(), st)
    return (getfield(inputs, src), st)
end
local_free_energy(::RelayFactor, _, ps, st) = (0.0, st)

# =========================================================================== #
# Optimisers as reparametrisations
# =========================================================================== #

"""
    abstract type AbstractUpdateRule

A stateful parameter update: Cruttwell et al.'s Definition 3.14, a lens
``U : (S\\times P,\\ S\\times P) \\to (P, P')``.

Implement [`rule_get`](@ref) (the lens's `get`, ``U``) and [`rule_put`](@ref) (its `put`,
``U^*``).
"""
abstract type AbstractUpdateRule end

"""
    rule_get(rule, state, p)

``U(s,p)`` — the parameter value handed *down* to the factor.

For plain gradient descent and momentum this is just `p`, which makes it look like dead
weight. Nesterov is the counterexample that justifies the whole formulation: its `get` is
`p + γs`, the look-ahead point. "Evaluate the gradient at the look-ahead point" *is*
"the forward part of the optimiser lens is not the identity".
"""
function rule_get end

"""
    rule_put(rule, state, p, p̄) -> (state', p')

``U^*(s,p,p')`` — the new state and the new parameter.
"""
function rule_put end

"""
    GradientDescent(η)

`get`: `p`. `put`: `(s, p - η·p̄)`.

Cruttwell's Definition 3.11 writes `G*(p,p') = p + p'`, with the sign and step size supplied
by the *learning-rate cap* `α*(l) = -ε`. Folding `η` in here is the same map with the cap
absorbed, which is what every real optimiser API does.
"""
struct GradientDescent{T} <: AbstractUpdateRule
    η::T
end
rule_get(::GradientDescent, state, p) = p
rule_put(r::GradientDescent, state, p, p̄) = (state, p .- r.η .* p̄)

"""
    Momentum(η, γ)

`get`: `p`. `put`: `s' = -γ s + (-η p̄)`, then `p + s'`. Recovers gradient descent at `γ = 0`.
"""
struct Momentum{T} <: AbstractUpdateRule
    η::T
    γ::T
end
rule_get(::Momentum, state, p) = p
function rule_put(r::Momentum, state, p, p̄)
    s′ = (-r.γ) .* state .+ (-r.η) .* p̄
    return (s′, p .+ s′)
end

"""
    Nesterov(η, γ)

`get`: `p + γ s` — **a non-trivial forward part**. `put`: as [`Momentum`](@ref).
"""
struct Nesterov{T} <: AbstractUpdateRule
    η::T
    γ::T
end
rule_get(r::Nesterov, state, p) = p .+ r.γ .* state
function rule_put(r::Nesterov, state, p, p̄)
    s′ = (-r.γ) .* state .+ (-r.η) .* p̄
    return (s′, p .+ s′)
end

"""
    OptimiserFactor(channel, rule)

An optimiser, as a **factor attached to an exposed parameter variable**.

Cruttwell et al. present optimisers as *reparametrisations* — boxes sitting above the
parameter wire. In a factor graph a "wire" is a variable, so the box above it is a factor, and
the reparametrisation is an ordinary node. This is only possible because a factor may expose
its parameters as channels rather than hiding them in `ps`, which is exactly the move AutoBayes
Example 3 (VBEM) makes when ``\\Theta`` "migrates from the parameter space into the wire".

Bidirectional by nature: it **emits** the current parameter (`rule_get`) and **absorbs** the
update (`rule_put`). Not learnable — it *holds* state, it does not *have* parameters.
"""
struct OptimiserFactor{R<:AbstractUpdateRule} <: LenticulumCore.AbstractLenticulumFactor
    channel::Symbol
    rule::R
end

LenticulumCore.channels(f::OptimiserFactor) = (LenticulumCore.Channel(f.channel, nothing),)
LenticulumCore.supported_polarities(f::OptimiserFactor) =
    (_polarity((f.channel,), (LenticulumCore.Unobserved(),)),)
LenticulumCore.supports_polarity(f::OptimiserFactor, p::LenticulumCore.Polarity) =
    keys(p) == (f.channel,) && p[f.channel] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(::OptimiserFactor) = false

"""
    optimiser_step(f::OptimiserFactor, state, p, p̄) -> (state', p')

Apply the rule. Separated from message passing because a parameter update is not a belief
message: it happens once per *training* step, whereas messages happen once per *inference*
sweep, and conflating the two schedules is a classic source of silent bugs.
"""
optimiser_step(f::OptimiserFactor, state, p, p̄) = rule_put(f.rule, state, p, p̄)

function factor_message(f::OptimiserFactor, ::Symbol, _, _, _, ps, st)
    p = haskey(st, :p) ? st.p : nothing
    s = haskey(st, :state) ? st.state : nothing
    return (LenticulumCore.DiracBelief(rule_get(f.rule, s, p)), st)
end
local_free_energy(::OptimiserFactor, _, ps, st) = (0.0, st)
