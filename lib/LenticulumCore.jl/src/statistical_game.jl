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
    scalar_free_energy(factor, π, y, ps, st) -> (Real, st)

``F^c = \\sigma_c \\circ \\mathbf{F}^c``: AutoBayes' Definition 20 loss.
"""
function scalar_free_energy(f::AbstractLenticulumFactor, π, y, ps, st)
    F, st = free_energy(f, π, y, ps, st)
    return scalarise(scalarisation(f), F), st
end


# --- Learnability traits ---------------------------------------------------

"""
    islearnable(factor) -> Bool

Whether `factor` contributes trainable parameters.

Defaults to `LuxCore.parameterlength(factor) > 0`, but is a *trait*, not a computation:
a factor with parameters that are deliberately held fixed (a pretrained encoder, a physical
constant, a data clamp) should override it to `false`. A graph uses this to decide which
nodes need an optimiser attached and which parameter cotangents may be discarded unread.

See also [`isfrozen`](@ref).
"""
islearnable(f::AbstractLenticulumFactor) = LuxCore.parameterlength(f) > 0

"""
    isfrozen(factor) -> Bool

Whether `factor`'s parameters exist but are pinned for this run.

The distinction from [`islearnable`](@ref) matters for the backward pass: a *frozen* factor
still propagates cotangents through to its inputs (so upstream factors learn), whereas a
factor that is simply not learnable has no parameter wire at all. Confusing the two silently
detaches a subgraph.
"""
isfrozen(::AbstractLenticulumFactor) = false

"""
    ComposedFactor(first, second; coupling = DiagonalCoupling())

``d \\diamond c`` of AutoBayes Definitions 22 and 28.

The composite's parameter space is the product ``\\Phi \\times \\Theta``, realised as the
`NamedTuple` `(first = ..., second = ...)` — literally the same data structure Lux uses for
`Chain`.

`coupling` records which of the gradient terms dropped by Definition 29 this edge restores.
"""
struct ComposedFactor{A,B,C<:AbstractGradientCoupling} <:
       AbstractLenticulumContainerFactor{(:first, :second)}
    first::A   # c : X ⊸ Y
    second::B  # d : Y ⊸ Z
    coupling::C
end
ComposedFactor(a, b; coupling = DiagonalCoupling()) = ComposedFactor(a, b, coupling)

compose(a::AbstractLenticulumFactor, b::AbstractLenticulumFactor; kwargs...) =
    ComposedFactor(a, b; kwargs...)

energyspace(f::ComposedFactor) =
    GradedEnergySpace((; first = energyspace(f.first), second = energyspace(f.second)))

scalarisation(f::ComposedFactor) =
    GradedScalarisation((; first = scalarisation(f.first), second = scalarisation(f.second)))

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
