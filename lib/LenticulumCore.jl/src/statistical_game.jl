# ---------------------------------------------------------------------------
# Statistical games (AutoBayes Defs. 20, 22, 27, 28) — the factor interface.
#
#     factor  =  (Bayesian lens, vector energy 𝐥, vector entropy 𝐇, scalarisation σ)
#                parameterized by ps, exactly as a Lux layer is.
#
#     Lux.jl        :  Para(Lens(C))
#     Lenticulum.jl :  Para(StatGame)
#
# See `statistical_game.md`.
# ---------------------------------------------------------------------------

"""
    free_energy(factor, π::AbstractBelief, y, ps, st) -> (𝐅, st)

The **vector** loss

```math
\\mathbf{F}^c(\\pi, y) = \\mathbb{E}_{(x,a)\\sim c'_\\pi(y)}[\\mathbf{l}^c(x,a,y)] - \\mathbf{H}^c(\\pi, y)
```

an element of `energyspace(factor)` — *not* a number. Collapse it with
[`scalar_free_energy`](@ref) when you want a loss.

Concrete families may also implement `free_energy(lens, π, inputs, ps, st)` for an
assembled `AbstractBayesianLens`: this evaluates the expectation under that inversion.
Its domain and prior cost must be specified by the family; no generic lens game is assumed.
For DEQ/NeuralODE, the factor game scores incoming points, while the assembled lens game
scores the solved point plus an anchor cost. Arbitrary point scoring uses `energy` or
`scalar_energy` (L-ADR-0003), not a keyword on `free_energy`.
"""
function free_energy end

"""
    FreeEnergyUpperBound(value)

An evaluated variational upper bound on scalar negative log evidence. This is deliberately
neither a number nor an energy: use `.value` explicitly. It cannot enter `scalarise` or
`scalar_free_energy` accidentally. The model and variational law belong to the producing API.
"""
struct FreeEnergyUpperBound{T<:Real}
    value::T
end

"""
    FreeEnergyBoundEstimate(value, standard_error, nsamples)

A Monte Carlo estimate of a variational upper bound. The expectation is a bound;
the sampled `value` need not lie above negative log evidence. No confidence bound is implied.
"""
struct FreeEnergyBoundEstimate{T<:Real,S<:Real}
    value::T
    standard_error::S
    nsamples::Int
end

function free_energy_upper_bound end
function estimate_free_energy_upper_bound end

"""
    scalar_free_energy(factor, π, y, ps, st; polarity = nothing, σ) -> (Real, st)

``F^c = \\sigma_c \\circ \\mathbf{F}^c``: AutoBayes' Definition 20 loss.

`σ` defaults to [`scalarisation`](@ref)`(factor)`, or to `scalarisation(factor, polarity)` when a
polarity is given; pass another to weigh the summands differently without rebuilding the factor.
Also accepts an assembled lens when its family implements `free_energy` and `scalarisation`.
The lens already fixes a polarity; omit the `polarity` override in that case.
"""
function scalar_free_energy(f::Union{AbstractLenticulumFactor,AbstractBayesianLens}, π, y, ps, st; polarity = nothing,
                            σ = _scalarisation(f, polarity))
    F, st = free_energy(f, π, y, ps, st)
    return scalarise(σ, F), st
end


# --- Learnability traits ---------------------------------------------------

"""
    islearnable(factor) -> Bool

Whether `factor` contributes trainable parameters.

Defaults to `LuxCore.parameterlength(factor) > 0`. Override to `false` when a factor
intrinsically contributes no trainable parameters. Mycelium's `learnable_factors` uses
this trait as the candidate list. Parameters of nonlearnable factors are fixed in graph
parameter gradients, while input derivatives still propagate through their computations.

See also [`isfrozen`](@ref).
"""
islearnable(f::AbstractLenticulumFactor) = LuxCore.parameterlength(f) > 0

"""
    isfrozen(factor) -> Bool

Whether `factor`'s parameters exist but are pinned for this run.

Defaults to `false`. True pins all parameter updates for this run, excludes them
from optimiser state, and preserves input cotangents. Unlike `islearnable`, it does
not remove a factor from `learnable_factors`. Mycelium also accepts `factor!(...;
freeze=(:encoder,))`, a tuple of paths, or `freeze=:all` for graph-local selection.
"""
isfrozen(::AbstractLenticulumFactor) = false

"""
    straight_through(forward, surrogate, x)

Evaluate `forward(x)` with the input Jacobian of `surrogate(x)`. Supported by
ForwardDiff and ChainRules reverse AD; finite differences of this primal do not
compute the surrogate derivative. The callables are fixed, parameterless functions.
"""
straight_through(f, s, x) = f(x)

"""
    ComposedFactor(first, second; coupling = DiagonalCoupling())

``d \\diamond c`` of AutoBayes Definitions 22 and 28.

The composite's parameter space is the product ``\\Phi \\times \\Theta``, realised as the
`NamedTuple` `(first = ..., second = ...)` — literally the same data structure Lux uses for
`Chain`.

Lenticulum implements a pair of GaussianFactors as a collapsed endpoint game, integrating
out the internal channel. Its energy space is that collapsed Gaussian's, distinct from
the generic path-energy pairs of `compose_energy` / `compose_free_energy`.

`coupling` selects the composite parameter-gradient rule. Non-diagonal rules require
an implemented family and otherwise throw at gradient evaluation. Lenticulum's collapsed
Gaussian pair supports analytic exact/pathwise gradients and an explicit posterior
score estimator; its historical diagonal endpoint gradient already marginalises the
internal channel. A collapsed endpoint game is distinct from a graph exposing that wire.
"""
struct ComposedFactor{A,B,C<:AbstractGradientCoupling} <:
       AbstractLenticulumContainerFactor{(:first, :second)}
    first::A   # c : X ⊸ Y
    second::B  # d : Y ⊸ Z
    coupling::C
    function ComposedFactor{A,B,C}(a, b, coupling) where {A,B,C<:AbstractGradientCoupling}
        _check_gradient_coupling(coupling)
        return new{A,B,C}(a, b, coupling)
    end
end
function ComposedFactor(a, b, coupling)
    _check_gradient_coupling(coupling)
    return ComposedFactor{typeof(a),typeof(b),typeof(coupling)}(a, b, coupling)
end
ComposedFactor(a, b; coupling = DiagonalCoupling()) = ComposedFactor(a, b, coupling)

compose(a::AbstractLenticulumFactor, b::AbstractLenticulumFactor; kwargs...) =
    ComposedFactor(a, b; kwargs...)

function free_energy_gradient(f::ComposedFactor, π, y, ps, st; ad=nothing, ε=1e-6, kwargs...)
    return _composed_parameter_gradient(f.coupling, f, π, y, ps, st; ad, ε, kwargs...)
end
function _composed_parameter_gradient(coupling, f, π, y, ps, st; ad, ε, kwargs...)
    coupling isa DiagonalCoupling || throw(ArgumentError(
        "$(typeof(f)) has no compositional gradient rule for $(typeof(coupling)); only the implemented Gaussian pair supplies non-diagonal gradients"))
    isempty(kwargs) || throw(ArgumentError("unsupported composite gradient keywords: $(keys(kwargs))"))
    return _parameter_gradient(ad, _scalar_free_energy_params, ps, f, π, y, st, (;); ε)
end

energyspace(f::ComposedFactor) =
    GradedEnergySpace((; first = energyspace(f.first), second = energyspace(f.second)))

scalarisation(f::ComposedFactor) =
    GradedScalarisation((; first = scalarisation(f.first), second = scalarisation(f.second)))

pointwise_energyspace(f::ComposedFactor) =
    GradedEnergySpace((;
        first = pointwise_energyspace(f.first),
        second = pointwise_energyspace(f.second),
    ))

pointwise_scalarisation(f::ComposedFactor) =
    GradedScalarisation((;
        first = pointwise_scalarisation(f.first),
        second = pointwise_scalarisation(f.second),
    ))

"""
    TensorFactor(parts::NamedTuple)

``c \\otimes d`` of AutoBayes Definition 25. Both energies and entropies add across
summands, because in parallel there is no upstream/downstream.

Exact games accept only explicit independent products: `π` and `y` are NamedTuples keyed
by `parts`, whose entries supply each component's complete channel game. Only declared
exact linear-Gaussian components are supported. Correlated joint beliefs throw, so no
mean-field projection is silently applied. Tensor graph wiring/inversion is not supplied.
"""
struct TensorFactor{names,T<:Tuple} <: AbstractLenticulumFactor
    parts::NamedTuple{names,T}
end

energyspace(f::TensorFactor{names}) where {names} =
    GradedEnergySpace(NamedTuple{names}(map(energyspace, values(f.parts))))
scalarisation(f::TensorFactor{names}) where {names} =
    GradedScalarisation(NamedTuple{names}(map(scalarisation, values(f.parts))))
pointwise_energyspace(f::TensorFactor{names}) where {names} =
    GradedEnergySpace(NamedTuple{names}(map(pointwise_energyspace, values(f.parts))))
pointwise_scalarisation(f::TensorFactor{names}) where {names} =
    GradedScalarisation(NamedTuple{names}(map(pointwise_scalarisation, values(f.parts))))

# Explicit products, keyed by branch, carry independence rather than discarding correlations
# from a joint belief. These are game combinators; they do not invent graph wiring.
function _tensor_game(op, f::TensorFactor{names}, π::NamedTuple{names}, y::NamedTuple{names}, ps, st) where {names}
    all(exact_product_game, values(f.parts)) || throw(ArgumentError(
        "TensorFactor requires components with a declared exact product game; currently only linear-Gaussian components are supported"))
    results = map(names) do n
        op(f.parts[n], π[n], y[n], ps[n], st[n])
    end
    return GradedEnergy(NamedTuple{names}(first.(results))), NamedTuple{names}(last.(results))
end
exact_product_game(::AbstractLenticulumFactor) = false
exact_product_game(f::TensorFactor) = all(exact_product_game, values(f.parts))
for op in (:free_energy, :entropy)
    @eval begin
        $op(f::TensorFactor, π::NamedTuple, y::NamedTuple, ps, st) =
            _tensor_game($op, f, π, y, ps, st)
        $op(::TensorFactor, π, y, ps, st) = throw(ArgumentError(
            "TensorFactor games require independent product beliefs: NamedTuples keyed by parts for both π and y; correlated joint beliefs are unsupported"))
        $op(::ComposedFactor, π, y, ps, st) = throw(ArgumentError(
            "ComposedFactor games require an exact pushforward law; only the linear-Gaussian pair implemented by Lenticulum is supported"))
    end
end
_tensor_game(op, f::TensorFactor, π, y, ps, st) = throw(ArgumentError(
    "TensorFactor product keys must equal keys(parts), in order"))

# --- The composite chain rule ---------------------------------------------

"""
    compose_energy(𝐥c, 𝐥d)

The composite **vector** energy of AutoBayes Definition 22, adapted:

```math
\\mathbf{l}^{dc}(x,a,y,b,z) = (\\mathbf{l}^c(x,a,y),\\ \\mathbf{l}^d(y,b,z))
```

The paper writes `+` here; we write a pair. Applying a `GradedScalarisation` to the pair
recovers the paper's `+` exactly, and does so *strictly* — the direct sum is where no
information is lost. Laxness enters only through [`compose_entropy`](@ref)'s expectation.
"""
compose_energy(lc, ld) = GradedEnergy((; first = lc, second = ld))

"""
    compose_entropy(𝐇c_samples, 𝐇d)

The composite **vector** entropy of AutoBayes Definition 22, adapted:

```math
\\mathbf{H}^{dc}(\\pi,z) = \\Bigl(\\mathbb{E}_{(y,b)\\sim d'_{c_*\\pi}(z)}[\\mathbf{H}^c(\\pi,y)],\\ \\mathbf{H}^d(c_*\\pi,z)\\Bigr)
```

`𝐇c_samples` are evaluations of ``\\mathbf{H}^c(\\pi, y)`` at samples `(y,b)` drawn from the
**downstream** inversion at the **pushforward** prior. Getting those two adjectives right is
the whole content of the definition, and the commonest place to go wrong.
"""
function compose_entropy(Hc_samples, Hd)
    Hc = reduce(+, Hc_samples) * (1 / length(Hc_samples))
    return GradedEnergy((; first = Hc, second = Hd))
end

"""
    compose_free_energy(𝐅c_samples, 𝐅d)

The multivariate chain rule:

```math
\\mathbf{F}^{dc}(\\pi,z) = \\Bigl(\\mathbb{E}_{(y,b)\\sim d'_{c_*\\pi}(z)}[\\mathbf{F}^c(\\pi,y)],\\ \\mathbf{F}^d(c_*\\pi,z)\\Bigr)
```

Compare AutoBayes Theorem 23, which is this composed with a scalarisation. The two agree
strictly iff `σ` is linear; otherwise they differ by [`jensen_gap`](@ref), and the
multivariate one is the smaller (Jensen).
"""
function compose_free_energy(Fc_samples, Fd)
    Fc = reduce(+, Fc_samples) * (1 / length(Fc_samples))
    return GradedEnergy((; first = Fc, second = Fd))
end

"""
    chain_rule_defect(σ, 𝐅c_samples)

How far AutoBayes' Theorem 23 and the multivariate chain rule disagree across one edge:

```math
F^{dc} - \\sigma_{dc}(\\mathbf{F}^{dc}) = \\mathbb{E}[\\sigma_c(\\mathbf{F}^c)] - \\sigma_c(\\mathbb{E}[\\mathbf{F}^c]) \\ \\ge 0
```

Zero for linear `σ`. For `SquaredNorm` it is ``\\tfrac12\\operatorname{tr}\\operatorname{Cov}(\\mathbf{F}^c)``:
how much the downstream posterior disagrees with itself about what the upstream factor
should be doing.
"""
chain_rule_defect(σ::AbstractScalarisation, Fc_samples) = jensen_gap(σ, Fc_samples)

# --- LuxCore interface, inherited -----------------------------------------
#
# The parameter/state machinery is taken over unchanged: a factor's ps/st tree is built
# from its fields exactly as a Lux layer's is. Only the *interpretation* of a factor
# differs, not how its parameters are stored.

for op in (:initialparameters, :initialstates)
    @eval begin
        LuxCore.$(op)(::AbstractRNG, ::AbstractLenticulumFactor) = NamedTuple()
        function LuxCore.$(op)(
            rng::AbstractRNG, f::AbstractLenticulumContainerFactor{factors}
        ) where {factors}
            return NamedTuple{factors}(map(x -> LuxCore.$(op)(rng, x), getfield.((f,), factors)))
        end
        function LuxCore.$(op)(
            rng::AbstractRNG, f::TensorFactor{names}
        ) where {names}
            return NamedTuple{names}(map(x -> LuxCore.$(op)(rng, x), values(f.parts)))
        end
        function LuxCore.$(op)(
            rng::AbstractRNG, f::AbstractLenticulumWrapperFactor{factor}
        ) where {factor}
            return LuxCore.$(op)(rng, getfield(f, factor))
        end
    end
end

function LuxCore.parameterlength(f::AbstractLenticulumContainerFactor{factors}) where {factors}
    return sum(LuxCore.parameterlength, getfield.((f,), factors); init = 0)
end
function LuxCore.statelength(f::AbstractLenticulumContainerFactor{factors}) where {factors}
    return sum(LuxCore.statelength, getfield.((f,), factors); init = 0)
end
function LuxCore.parameterlength(f::TensorFactor)
    return sum(LuxCore.parameterlength, values(f.parts); init = 0)
end
function LuxCore.statelength(f::TensorFactor)
    return sum(LuxCore.statelength, values(f.parts); init = 0)
end
LuxCore.parameterlength(f::AbstractLenticulumWrapperFactor{factor}) where {factor} =
    LuxCore.parameterlength(getfield(f, factor))
LuxCore.statelength(f::AbstractLenticulumWrapperFactor{factor}) where {factor} =
    LuxCore.statelength(getfield(f, factor))

"""
    setup(rng, factor) -> (ps, st)

As `LuxCore.setup`. Provided so that factors and Lux layers are set up identically.
"""
setup(rng::AbstractRNG, f::AbstractLenticulumFactor) =
    (LuxCore.initialparameters(rng, f), LuxCore.initialstates(rng, f))
