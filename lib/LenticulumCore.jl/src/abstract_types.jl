# ---------------------------------------------------------------------------
# The abstract type hierarchy.
#
# Three families of types, corresponding to the three layers of theory:
#
#   1. Factors          -- parameterized statistical games (AutoBayes Def. 27).
#                          Mirrors `LuxCore.AbstractLuxLayer` and friends.
#   2. Game components  -- open models, beliefs, inversions, Bayesian lenses.
#                          (AutoBayes Defs. 1, 9, 10.)
#   3. Energy algebra   -- energy spaces and scalarisations.
#                          (Lenticulum's own; see `energy.md`.)
#
# See `abstract_types.md`.
# ---------------------------------------------------------------------------

"""
    abstract type AbstractLenticulumFactor

A **factor**: a parameterized statistical game (AutoBayes, Definition 27).

This is Lenticulum's analogue of `LuxCore.AbstractLuxLayer`, and the parameter/state
interface is inherited verbatim: implementors **must** provide

  - `initialparameters(rng, factor)`
  - `initialstates(rng, factor)`

and additionally, for the statistical-game structure,

  - [`channels`](@ref)          -- the named ports of the factor
  - [`supports_polarity`](@ref) -- which input/output splits it can answer
  - [`assemble`](@ref)          -- produce a Bayesian lens for a chosen polarity
  - [`energyspace`](@ref)       -- the energy space ``E_c``
  - [`energy`](@ref)            -- the vector energy ``\\mathbf{l}^c``
  - [`entropy`](@ref)           -- the vector entropy ``\\mathbf{H}^c``

Optionally `scalarisation`, `parameterlength`, `statelength`, `display_name`.

Unlike a Lux layer, a factor has **no fixed direction**: `assemble(factor, polarity)`
constructs the parametric lens on demand. See `Channels and Polarity.md`.
"""
abstract type AbstractLenticulumFactor end

"""
    abstract type AbstractLenticulumContainerFactor{factors} <: AbstractLenticulumFactor

A factor built from several sub-factors named by the tuple of symbols `factors`.
Mirrors `LuxCore.AbstractLuxContainerLayer`: parameters and states are assembled into a
`NamedTuple` keyed by `factors`.

This is the [`Para`] composition of Definition 28: parameter spaces multiply, and the
product is realised as a `NamedTuple`.
"""
abstract type AbstractLenticulumContainerFactor{factors} <: AbstractLenticulumFactor end

"""
    abstract type AbstractLenticulumWrapperFactor{factor} <: AbstractLenticulumFactor

A factor wrapping exactly one sub-factor, whose parameters and states are **not** nested
under an extra name. Mirrors `LuxCore.AbstractLuxWrapperLayer`.
"""
abstract type AbstractLenticulumWrapperFactor{factor} <: AbstractLenticulumFactor end

# --- Game components -------------------------------------------------------

"""
    abstract type AbstractOpenModel

An open model ``c : X \\nrightarrow Y`` (AutoBayes, Definition 1): a measure kernel
``X \\rightsquigarrow \\llbracket c \\rrbracket \\times Y`` together with its latent space
``\\llbracket c \\rrbracket``.

The latent space is the reason composition of open models needs no integration; see
`open_model.md`.
"""
abstract type AbstractOpenModel end

"""
    abstract type AbstractBelief

An element of ``\\mathcal{P}X``: a distribution over a channel's space.

Beliefs are what flows *forward* along the graph (as priors, propagated by pushforward)
and what inversions consume. Concrete subtypes decide the representation: a Dirac, a
particle set, an exponential-family natural parameter, a Gaussian.
"""
abstract type AbstractBelief end

"""
    abstract type AbstractInversion

The backward half ``c'`` of a Bayesian lens (AutoBayes, Definition 9): a rule that turns a
prior ``\\pi \\in \\mathcal{P}X`` and an observation ``y \\in Y`` into a belief over
``X \\times \\llbracket c \\rrbracket``.

Nothing requires an inversion to be exact. Exact, amortised, mean-field, solver-based and
diffusion-based inversions are all legal inhabitants; the quality of the choice is what the
free energy measures.
"""
abstract type AbstractInversion end

"""
    abstract type AbstractBayesianLens

A Bayesian lens ``(c, c')`` (AutoBayes, Definition 9): an [`AbstractOpenModel`](@ref)
paired with an [`AbstractInversion`](@ref).
"""
abstract type AbstractBayesianLens end

# --- Energy algebra --------------------------------------------------------

"""
    abstract type AbstractEnergySpace

The space ``E_c`` in which a factor's **vector** energy and entropy take values.

AutoBayes fixes ``E_c = [0,\\infty]`` and composes energies by addition. Lenticulum keeps
the vector and composes by direct sum, recovering the paper's law under scalarisation.
See `Scalar and Multivariate Energy.md`.
"""
abstract type AbstractEnergySpace end

"""
    abstract type AbstractScalarisation

A map ``\\sigma : E \\to \\mathbb{R}`` collapsing a vector energy to a loss.

Must satisfy `σ(0) == 0`, non-negativity on the cone, and monotonicity. The critical trait
is [`islinear`](@ref): a linear `σ` commutes with the expectation in the chain rule and so
composition is strict; a merely convex `σ` makes it lax, with the gap given by a Jensen
term (a variance, for a squared norm).
"""
abstract type AbstractScalarisation end

# --- Gradient semantics ----------------------------------------------------

"""
    abstract type AbstractGradientCoupling

How an edge of the graph handles the terms AutoBayes' Definition 29 drops.

Definition 29 composes gradients block-diagonally; the true Jacobian has two extra blocks,
because the parameter of one factor moves the *sampling distribution* and the *pushforward
prior* seen by the other. The paper calls the resulting assignment "lax" and says it "can be
accounted for mechanistically by an implementation". This type is that mechanism: each edge
declares which correction it applies, and the choice is exactly the paper's "different
semantics functors".

See [`DiagonalCoupling`](@ref), [`PathwiseCoupling`](@ref),
[`ScoreFunctionCoupling`](@ref), [`ExactCoupling`](@ref).
"""
abstract type AbstractGradientCoupling end

"""
    DiagonalCoupling()

Keep only the block-diagonal terms of Definition 29; i.e. stop-gradient through the
sampling path and through the pushforward prior. Cheapest, biased.
"""
struct DiagonalCoupling <: AbstractGradientCoupling end

"""
    PathwiseCoupling()

Differentiate through the sampler (the reparametrisation trick). Requires the inversion to
be reparametrisable. Unbiased, low variance.
"""
struct PathwiseCoupling <: AbstractGradientCoupling end

"""
    ScoreFunctionCoupling(baseline=nothing)

REINFORCE: recover the dropped term as `E[F ∇ log q]`, optionally with a control variate.
Unbiased, high variance, works for discrete inversions.
"""
struct ScoreFunctionCoupling{B} <: AbstractGradientCoupling
    baseline::B
end
ScoreFunctionCoupling() = ScoreFunctionCoupling(nothing)

"""
    ExactCoupling()

The conjugate/analytic case: the full Jacobian is available in closed form and no term is
dropped. Composition of gradients is then strictly functorial across this edge.
"""
struct ExactCoupling <: AbstractGradientCoupling end
