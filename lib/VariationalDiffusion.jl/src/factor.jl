# ---------------------------------------------------------------------------
# The diffusion factor: a `LenticulumFactor` whose inversion is a RED-Diff prox.
#
# This is where `ImplicitREDDiff.md`'s selection matrices stop being notation. The factor
# owns one space ℝⁿ, carved into named contiguous blocks (the channels). A `Polarity` assigns
# each block Observed / Unobserved / Latent, and each carries a precision ρ, so
#
#     P = ρ_in P_in + ρ_out P_out + ρ_latent P_latent
#
# is *derived from the polarity*, not configured. `LenticulumCore.channels` already ships
# `default_precision(Observed()) = Inf`, so the hard clamp is the default and the soft clamp
# is what you opt into — which is the right way round.
#
# See `factor.md` and `The Diffusion Factor.md`.
# ---------------------------------------------------------------------------

"""
    DiffusionFactor(blocks::NamedTuple, predictor; prox = REDDiff())

A factor whose prior over its own state space is a diffusion model, and whose Bayesian
inversion is a RED-Diff proximal solve.

`blocks` names the channels and their dimensions; they tile one vector space
``\\mathbb{R}^n``, ``n = \\sum_i \\dim_i``, in declaration order. `predictor` is a
[`NoisePredictor`](@ref) over that whole ``\\mathbb{R}^n`` — **one network for the joint
state**, not one per channel, which is what makes this a relation rather than a collection of
conditionals.

```julia
f = DiffusionFactor((obs = 4, hidden = 4), NoisePredictor(unet, VPSDE());
                    prox = REDDiff(λ = 1.0, steps = 300))
```

This is the third of the three [`Implicit Learners`] families — the diffusion one — and the
only one whose inversion is neither exact nor a root-find. `LenticulumCore` anticipated it:
[`ProximalInversion`](@ref) exists in `lens.jl` and names this package.
"""
struct DiffusionFactor{names,D<:Tuple,P<:NoisePredictor,C<:REDDiff} <:
       LenticulumCore.AbstractLenticulumFactor
    blocks::NamedTuple{names,D}
    predictor::P
    prox::C
end

function DiffusionFactor(blocks::NamedTuple, predictor::NoisePredictor; prox = REDDiff())
    isempty(blocks) && throw(ArgumentError("a DiffusionFactor needs at least one channel"))
    all(d -> d isa Int && d > 0, values(blocks)) ||
        throw(ArgumentError("channel dimensions must be positive Ints; got $blocks"))
    return DiffusionFactor(blocks, predictor, prox)
end

"""
    statedim(f::DiffusionFactor) -> Int

``n``, the dimension of the joint space the diffusion prior lives on.
"""
statedim(f::DiffusionFactor) = sum(values(f.blocks))

"""
    blockranges(f::DiffusionFactor) -> NamedTuple

The index range of each channel inside ``\\mathbb{R}^n``. These *are* the selection matrices
``P_{in}, P_{out}, P_{latent}`` of `ImplicitREDDiff.md`, stored as ranges because a diagonal
0/1 matrix is a wasteful way to write a range.
"""
function blockranges(f::DiffusionFactor{names}) where {names}
    o = 0
    rs = map(values(f.blocks)) do d
        r = (o + 1):(o + d)
        o += d
        r
    end
    return NamedTuple{names}(rs)
end

# Parameters and state are the network's, untouched.
LuxCore.initialparameters(rng::AbstractRNG, f::DiffusionFactor) =
    LuxCore.initialparameters(rng, f.predictor)
LuxCore.initialstates(rng::AbstractRNG, f::DiffusionFactor) =
    LuxCore.initialstates(rng, f.predictor)
LuxCore.parameterlength(f::DiffusionFactor) = LuxCore.parameterlength(f.predictor)
LuxCore.statelength(f::DiffusionFactor) = LuxCore.statelength(f.predictor)

LenticulumCore.channels(f::DiffusionFactor{names}) where {names} =
    map((n, d) -> LenticulumCore.Channel(n, d), names, values(f.blocks))

"""
    LenticulumCore.supported_polarities(f::DiffusionFactor)

The `n` polarities with one channel `Unobserved()` and the rest `Observed()`.

**This under-reports what the factor can do.** [`supports_polarity`](@ref) accepts any
assignment with at least one unobserved channel, including `Latent()` ones, because a
diffusion prior over the joint space can inpaint any subset from any other subset — that is
the entire appeal of using one. The enumeration is truncated because a scheduler needs a
*listable* set and the full set has ``3^n - 2^n`` elements. See `factor.md` §3.
"""
function LenticulumCore.supported_polarities(f::DiffusionFactor{names}) where {names}
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    return map(names) do t
        LenticulumCore.Polarity(NamedTuple{names}(map(n -> n === t ? U : O, names)))
    end
end

function LenticulumCore.supports_polarity(
    f::DiffusionFactor{names}, p::LenticulumCore.Polarity
) where {names}
    keys(p) == names || return false
    return !isempty(LenticulumCore.unobserved_channels(p))
end

LenticulumCore.islearnable(::DiffusionFactor) = true

"""
    LenticulumCore.energyspace(::DiffusionFactor)

``E_c = \\mathbb{R}^2``, graded as `(clamp, score)`:

| summand | is | role |
|---|---|---|
| `clamp` | ``\\tfrac12\\|P(x_0-x)\\|^2`` | the **energy** ``\\mathbf{l}^c`` — data consistency |
| `score` | ``\\mathbb{E}_{t,\\varepsilon}[\\omega(t)\\|\\varepsilon_\\theta - \\varepsilon\\|^2]`` | the **entropy** ``\\mathbf{H}^c`` — the learned prior |

The split is not cosmetic and it is the reading `Implicit Learners.md` §"Diffusion" already
gives: the clamp term is pointwise and depends on the data, so it is an energy; the
score-matching term depends on the *learned distribution* rather than on the data point, so
it is an entropy. Getting this backwards would put the prior in the energy and break the
counting correction of `Bethe Free Energy.md`.
"""
LenticulumCore.energyspace(::DiffusionFactor) = LenticulumCore.GradedEnergySpace((
    clamp = LenticulumCore.ScalarEnergySpace(),
    score = LenticulumCore.ScalarEnergySpace(),
))
LenticulumCore.scalarisation(::DiffusionFactor) = LenticulumCore.GradedScalarisation((
    clamp = LenticulumCore.IdentityScalarisation(),
    score = LenticulumCore.IdentityScalarisation(),
))

# --- Polarity → the matrix P ----------------------------------------------

"""
    precision_vector(f, p::Polarity) -> (ρ, hardmask)

The diagonal of ``P = \\rho_{in}P_{in} + \\rho_{out}P_{out} + \\rho_{latent}P_{latent}``, as a
vector of length `n`, together with a `BitVector` marking the coordinates whose precision is
infinite.

Infinite entries are split off rather than stored as `Inf`: they cannot participate in a
gradient, and the honest treatment is projection. `default_precision(Observed()) == Inf`, so
by default every observed channel lands in `hardmask` and is clamped exactly.
"""
function precision_vector(f::DiffusionFactor{names}, p::LenticulumCore.Polarity) where {names}
    n = statedim(f)
    ρ = zeros(Float64, n)
    hard = falses(n)
    rs = blockranges(f)
    for nm in names
        r = getfield(rs, nm)
        w = float(LenticulumCore.channel_precision(p, nm))
        if isinf(w)
            hard[r] .= true
        else
            ρ[r] .= w
        end
    end
    return (ρ, hard)
end

"""
    assemble_state(f, inputs, π) -> x₀

Build the reference configuration ``x_0`` of ``\\tfrac12\\|P(x_0-x)\\|^2`` by laying each
channel's incoming belief into its block.

A `DiracBelief` contributes its value; anything with a mean contributes that; a channel with
no message contributes zeros. The prior `π` fills the target block when it carries a point.
Channels whose precision is zero never read their entry, so the zeros are not a silent
default — they are multiplied out.
"""
function assemble_state(f::DiffusionFactor{names}, inputs, π) where {names}
    x₀ = zeros(Float64, statedim(f))
    rs = blockranges(f)
    for nm in names
        v = _point(_get(inputs, nm))
        v === nothing && (v = _point(π))
        v === nothing && continue
        r = getfield(rs, nm)
        length(v) == length(r) || throw(DimensionMismatch(
            "channel :$nm has dimension $(length(r)) but its message carries $(length(v))"))
        x₀[r] .= v
    end
    return x₀
end

_get(inputs, k) = (inputs isa NamedTuple && haskey(inputs, k)) ? getfield(inputs, k) : nothing
_point(::Nothing) = nothing
_point(b::LenticulumCore.DiracBelief) = _vec(b.value)
_point(::LenticulumCore.TrivialBelief) = nothing
_point(::LenticulumCore.SampleBelief) = nothing   # a particle set has no single point
_point(b) = hasproperty(b, :η) ? _mean_or_nothing(b) : nothing
_mean_or_nothing(b) = try
    Mycelium.belief_mean(b)
catch
    nothing
end
_vec(v::AbstractVector) = float.(v)
_vec(v::Real) = [float(v)]

# --- The open model and the proximal inversion -----------------------------

"""
    DiffusionModel(factor, polarity)

The open model the polarity selects: an implicit relation on ``\\mathbb{R}^n``, restricted to
predicting the unobserved block from the observed ones.

**Not pure.** The unobserved *and* latent blocks are both reconstructed by the prox, and the
latent ones are exactly AutoBayes' ``\\llbracket c \\rrbracket`` — internal coordinates the
inversion has to fill in and nobody reads.
"""
struct DiffusionModel{F<:DiffusionFactor,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::DiffusionModel) = false
LenticulumCore.latentspace(m::DiffusionModel) =
    LenticulumCore.latent_channels(m.polarity)

LenticulumCore.assemble(f::DiffusionFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(DiffusionModel(f, p), LenticulumCore.ProximalInversion(f.prox)), st)

"""
    LenticulumCore.invert(lens, π, inputs, ps, st) -> (DiracBelief, st)

Run the RED-Diff prox and return the reconstructed unobserved block.

The return type is a **`DiracBelief`**, and that is faithful rather than lazy: RED-Diff's
variational family is ``q = \\mathcal{N}(\\mu, \\sigma^2 I)`` with ``\\sigma \\to 0``, so the
posterior it computes *is* a point mass. The consequence is recorded in `factor.md` §5 —
a Dirac message dominates every `combine` it meets, so a diffusion factor in a graph
overrides its neighbours rather than negotiating with them.
"""
function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:DiffusionModel,<:LenticulumCore.ProximalInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    p = lens.model.polarity
    x, st = _run_prox(f, p, inputs, π, ps, st)
    target = first(LenticulumCore.unobserved_channels(p))
    r = getfield(blockranges(f), target)
    return (LenticulumCore.DiracBelief(x[r]), st)
end

function _run_prox(f::DiffusionFactor, p::LenticulumCore.Polarity, inputs, π, ps, st)
    x₀ = assemble_state(f, inputs, π)
    ρ, hard = precision_vector(f, p)
    ρ² = ρ .^ 2
    datagrad(x) = ρ² .* (x .- x₀)
    return reddiff_solve(f.predictor, f.prox, x₀, datagrad, (hard, x₀), ps, st)
end

"""
    Mycelium.factor_message(f::DiffusionFactor, target, polarity, inputs, prior, ps, st)

The factor → variable message: a `DiracBelief` on `target`, from a RED-Diff prox over the
joint space.

> [!warning] This is the posterior, not the likelihood
> Every other factor in this project returns a *likelihood* here, with the prior divided out,
> because a variable of degree `d` would otherwise count the prior `d` times
> (`gaussian.jl`'s `invert` docstring). **A diffusion factor cannot divide its prior out** —
> the prior is a neural network and there is no subtraction available in canonical form.
>
> So on a graph where the target variable has degree > 1, this message double-counts. It is
> correct for a degree-1 target (the usual inverse-problem setting: one prior, one
> measurement) and approximate otherwise. `factor.md` §5.
"""
function Mycelium.factor_message(
    f::DiffusionFactor, target::Symbol, polarity, inputs, prior, ps, st
)
    haskey(f.blocks, target) || throw(ArgumentError(
        "channel :$target is not a channel of this DiffusionFactor (has $(keys(f.blocks)))"))
    p = polarity isa LenticulumCore.Polarity ? polarity : _default_polarity(f, target, inputs)
    x, st = _run_prox(f, p, inputs, prior, ps, st)
    r = getfield(blockranges(f), target)
    return (LenticulumCore.DiracBelief(x[r]), st)
end

function _default_polarity(f::DiffusionFactor{names}, target::Symbol, inputs) where {names}
    O, U, L = LenticulumCore.Observed(), LenticulumCore.Unobserved(), LenticulumCore.Latent()
    vals = map(names) do n
        n === target ? U : (_get(inputs, n) === nothing ? L : O)
    end
    return LenticulumCore.Polarity(NamedTuple{names}(vals))
end

# --- Energy and the Bethe contribution -------------------------------------

"""
    LenticulumCore.energy(f::DiffusionFactor, x, a, y, ps, st)

The graded energy at a point: `(clamp = ½‖P(x₀-x)‖², score = one MC sample of the
score-matching loss)`.

`x` is the full state vector, `y` the reference ``x_0``, and `a` the polarity (the slot
AutoBayes reserves for ``\\llbracket c \\rrbracket``, used here to carry the thing that
determines ``P``). That reuse is ugly and is discussed in `factor.md` §4.
"""
function LenticulumCore.energy(f::DiffusionFactor, x, a::LenticulumCore.Polarity, y, ps, st)
    ρ, _ = precision_vector(f, a)
    d = ρ .* (y .- x)
    clamp_term = sum(abs2, d) / 2
    t = sample_time(f.prox.rng, f.predictor.schedule)
    ε = randn(f.prox.rng, eltype(x), size(x))
    sc, st = denoising_loss(f.predictor, x, t, ε, ps, st)
    w = reddiff_weight(f.prox, f.predictor.schedule, t)
    return (LenticulumCore.GradedEnergy((clamp = clamp_term, score = w * sc)), st)
end

"""
    Mycelium.local_free_energy(f::DiffusionFactor, msgs, ps, st)

The factor's Bethe contribution, evaluated at the prox's own solution — i.e. at the point the
inversion actually returned, which is the only point where the two summands are comparable.

Because the variational posterior is a Dirac, its entropy is ``-\\infty`` and the
``-H(b_c)`` term of the Bethe free energy is **not** the posterior entropy: the `score`
summand plays that role instead, per `Implicit Learners.md`. This is the one factor in the
project whose free energy is not a closed form, and it is Monte-Carlo noisy by construction.
"""
function Mycelium.local_free_energy(f::DiffusionFactor{names}, msgs, ps, st) where {names}
    target = first(names)
    for n in names
        if _get(msgs, n) === nothing
            target = n
            break
        end
    end
    p = _default_polarity(f, target, msgs)
    x, st = _run_prox(f, p, msgs, LenticulumCore.TrivialBelief(), ps, st)
    x₀ = assemble_state(f, msgs, LenticulumCore.TrivialBelief())
    return LenticulumCore.energy(f, x, p, x₀, ps, st)
end
