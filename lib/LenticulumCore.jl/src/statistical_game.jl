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
"""
function free_energy end

"""
    scalar_free_energy(factor, π, y, ps, st; polarity = nothing, σ) -> (Real, st)

``F^c = \\sigma_c \\circ \\mathbf{F}^c``: AutoBayes' Definition 20 loss.

`σ` defaults to [`scalarisation`](@ref)`(factor)`, or to `scalarisation(factor, polarity)` when a
polarity is given; pass another to weigh the summands differently without rebuilding the factor.
"""
function scalar_free_energy(f::AbstractLenticulumFactor, π, y, ps, st; polarity = nothing,
                            σ = _scalarisation(f, polarity))
    F, st = free_energy(f, π, y, ps, st)
    return scalarise(σ, F), st
end


# --- Learnability traits ---------------------------------------------------

"""
    islearnable(factor) -> Bool

Whether `factor` contributes trainable parameters.

Defaults to `LuxCore.parameterlength(factor) > 0`, but is a *trait*, not a computation:
a factor with parameters that are deliberately held fixed (a pretrained encoder, a physical
constant, a data clamp) should override it to `false`. A graph uses this to decide which
nodes are candidates for an optimiser (`Mycelium.learnable_factors`). This trait does not
implement parameter freezing or control input cotangents; graph backward semantics remain
planned with L-SPEC-002.

See also [`isfrozen`](@ref).
"""
islearnable(f::AbstractLenticulumFactor) = LuxCore.parameterlength(f) > 0

"""
    isfrozen(factor) -> Bool

Whether `factor`'s parameters exist but are pinned for this run.

Defaults to `false`. The planned distinction from [`islearnable`](@ref) is that freezing
pins parameter updates while preserving input cotangents. This backward behavior is not
implemented (LP-011, L-SPEC-002). Mycelium's `build` and `validate` throw `ArgumentError`
naming any factor with `isfrozen(f) == true`, even when `islearnable(f) == false`:
excluding a factor from the optimiser list does not implement frozen cotangent semantics.
"""
isfrozen(::AbstractLenticulumFactor) = false

"""
    ComposedFactor(first, second; coupling = DiagonalCoupling())

``d \\diamond c`` of AutoBayes Definitions 22 and 28.

The composite's parameter space is the product ``\\Phi \\times \\Theta``, realised as the
`NamedTuple` `(first = ..., second = ...)` — literally the same data structure Lux uses for
`Chain`.

`coupling` reserves the choice of gradient terms to restore from Definition 29.
Only `DiagonalCoupling()` is accepted, preserving existing behavior. All other couplings
throw `ArgumentError` at construction (including positional and explicitly parameterized
construction), since gradient couplings are not implemented yet (LP-011, L-SPEC-002)
and would otherwise be silently ignored. The default annotation does not insert AD stops.
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

Subject to the mean-field laxness of `TensorLens`.
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
