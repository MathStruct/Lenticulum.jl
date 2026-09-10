# ---------------------------------------------------------------------------
# The acausal linear constraint:   0 = Σᵢ Aᵢ xᵢ − c + ε,   ε ~ N(0, Q)
#
# This is ModelingToolkit's `0 ~ ...` equation with a noise term bolted on, and it is the
# n-ary, direction-free generalisation of `GaussianFactor`:
#
#     GaussianFactor(A, b)  ==  LinearConstraintFactor with A_x = −A, A_y = I, c = b
#
# and the messages agree on the nose (asserted in the test suite). The difference is not the
# arithmetic — it is that a `GaussianFactor` has two channels named `in` and `out` and admits
# two polarities, whereas this factor has n channels, none of them distinguished, and admits
# one polarity per channel. That is what "acausal" means, and it is what README.md's
# "Symmetry handling: no distinguished input/output" row asks for.
#
# The categorical content: Gaussian RELATIONS (as opposed to Gaussian maps) form a hypergraph
# category, and an improper belief — Λ singular — is what makes that true. `beliefs.jl`
# already represents those, so this factor costs no new belief machinery.
#
# See `constraint.md`, `ModelingToolkit as an Acausal Relation.md` and
# `Acausal Composition is a Hypergraph Category.md`.
# ---------------------------------------------------------------------------

"""
    LinearConstraintFactor(dims::NamedTuple, m::Int; noise = 0.1, init_coeff, init_constant)

The soft linear relation ``0 = \\sum_i A_i x_i - c + \\varepsilon``, ``\\varepsilon \\sim
\\mathcal{N}(0, Q)`` — equivalently the density

```math
f(x_1,\\ldots,x_n) \\;\\propto\\; \\exp\\Bigl(-\\tfrac12 \\bigl\\|\\textstyle\\sum_i A_i x_i - c\\bigr\\|^2_{Q^{-1}}\\Bigr)
```

`dims` names the channels and their dimensions, `m` is the dimension of the residual (the
number of scalar equations). `Q` is a fixed hyperparameter, exactly as in
[`GaussianFactor`](@ref); parameters are `(A, c)`, with `A` a `NamedTuple` keyed by channel.

**Acausal**: no channel is privileged. `supported_polarities` returns one polarity per
channel — "solve this equation for `x_t`, given the others" — and every one of them is
exact. As ``Q \\to 0`` the factor becomes the hard relation
``\\{x : \\sum_i A_i x_i = c\\}``; a finite `Q` is the *soft* relation, which is what makes
it a statistical game rather than a set.

```julia
# Kirchhoff's current law at a three-way node: i₁ + i₂ + i₃ = 0
f  = LinearConstraintFactor((i1 = 1, i2 = 1, i3 = 1), 1; noise = 1e-8)
ps = (A = (i1 = ones(1,1), i2 = ones(1,1), i3 = ones(1,1)), c = [0.0])
```

See `constraint.md` for the message algebra and the honest list of what is missing.
"""
struct LinearConstraintFactor{names,D<:Tuple,Q<:AbstractMatrix,W,B} <:
       LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names,D}
    m::Int
    Q::Q
    init_coeff::W
    init_constant::B
end

function LinearConstraintFactor(
    dims::NamedTuple,
    m::Int;
    noise = 0.1,
    init_coeff = (rng, m, n) -> randn(rng, m, n) ./ sqrt(n),
    init_constant = (rng, m) -> zeros(m),
)
    isempty(dims) && throw(ArgumentError("a constraint needs at least one channel"))
    all(d -> d isa Int && d > 0, values(dims)) ||
        throw(ArgumentError("channel dimensions must be positive Ints; got $(dims)"))
    m > 0 || throw(ArgumentError("the residual dimension m must be positive; got $m"))
    Q = _as_cov(noise, m)
    isposdef(Symmetric(Q)) ||
        throw(ArgumentError("noise covariance Q must be positive definite"))
    return LinearConstraintFactor(dims, m, Q, init_coeff, init_constant)
end

LuxCore.initialparameters(rng::AbstractRNG, f::LinearConstraintFactor{names}) where {names} = (
    A = NamedTuple{names}(map(n -> f.init_coeff(rng, f.m, n), values(f.dims))),
    c = f.init_constant(rng, f.m),
)
LuxCore.initialstates(::AbstractRNG, ::LinearConstraintFactor) = NamedTuple()
LuxCore.parameterlength(f::LinearConstraintFactor) = f.m * sum(values(f.dims)) + f.m

LenticulumCore.channels(f::LinearConstraintFactor{names}) where {names} =
    map((n, d) -> LenticulumCore.Channel(n, d), names, values(f.dims))

"""
    LenticulumCore.supported_polarities(f::LinearConstraintFactor)

One polarity per channel: that channel `Unobserved()`, every other `Observed()`.

Contrast [`GaussianFactor`](@ref), which has exactly two regardless of anything. Here the
count is `n`, and **none of them is the factor's "real" direction** — the equation was never
written in a direction. This is the enumerated form of "a relation, not a function".
"""
function LenticulumCore.supported_polarities(f::LinearConstraintFactor{names}) where {names}
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    return map(names) do t
        LenticulumCore.Polarity(NamedTuple{names}(map(n -> n === t ? U : O, names)))
    end
end

"""
    LenticulumCore.supports_polarity(f::LinearConstraintFactor, p)

Any polarity over exactly this channel set with **exactly one** `Unobserved()` channel.

`Latent()` channels are accepted by this predicate but produce an uninformative message —
see [`Mycelium.factor_message`](@ref) below. Accepting-then-returning-nothing rather than
refusing is deliberate: a latent channel is a legal modelling choice (marginalise it out),
and the answer "this equation tells you nothing about `x_t` once `x_j` is free" is a correct
answer, not an error.
"""
function LenticulumCore.supports_polarity(
    f::LinearConstraintFactor{names}, p::LenticulumCore.Polarity
) where {names}
    keys(p) == names || return false
    return length(LenticulumCore.unobserved_channels(p)) == 1
end

LenticulumCore.islearnable(::LinearConstraintFactor) = true

# The grading is the same (fit, complexity, negentropy) as `GaussianFactor`: same model
# class, same decomposition, and `IdentityScalarisation` on each summand keeps it linear and
# therefore composition strict.
LenticulumCore.energyspace(::LinearConstraintFactor) = LenticulumCore.GradedEnergySpace((
    fit = LenticulumCore.ScalarEnergySpace(),
    complexity = LenticulumCore.ScalarEnergySpace(),
    negentropy = LenticulumCore.ScalarEnergySpace(),
))
LenticulumCore.scalarisation(::LinearConstraintFactor) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(),
    complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation(),
))

"""
    residual(f::LinearConstraintFactor, vals::NamedTuple, ps) -> Vector

``r = \\sum_i A_i x_i - c``, the pointwise vector energy, in the space where the equation's
own noise lives.

Note the signature: a `NamedTuple` of **all** channel values, with no input/output split,
because the factor has none. See `constraint.md` §3 for why `LenticulumCore.energy`'s
`(x, a, y)` signature cannot express this directly.
"""
function residual(f::LinearConstraintFactor{names}, vals::NamedTuple, ps) where {names}
    r = -float.(ps.c)
    for n in names
        haskey(vals, n) || throw(ArgumentError(
            "residual needs a value for every channel; :$n is missing (have $(keys(vals)))"))
        r = r .+ ps.A[n] * _vec(vals[n])
    end
    return r
end

LenticulumCore.energy(f::LinearConstraintFactor, x, a, y, ps, st) =
    (residual(f, _channel_values(x, y), ps), st)

_channel_values(x::NamedTuple, y::NamedTuple) = merge(x, y)
_channel_values(x::NamedTuple, ::Nothing) = x
_channel_values(::Nothing, y::NamedTuple) = y
_channel_values(x, y) = throw(ArgumentError(
    "a LinearConstraintFactor has no input/output split, so `energy` wants NamedTuples of \
     channel values; got $(typeof(x)) and $(typeof(y)). Call `residual(f, vals, ps)` \
     directly with all channels."))

# --- The open model and its exact inversion --------------------------------

"""
    LinearConstraintModel(factor, polarity)

The open model the polarity selects: "solve this equation for the unobserved channel". Pure
— a linear relation with additive Gaussian noise hides nothing.
"""
struct LinearConstraintModel{P} <: LenticulumCore.AbstractOpenModel
    factor::LinearConstraintFactor
    polarity::P
end
LenticulumCore.ispure(::LinearConstraintModel) = true
LenticulumCore.latentspace(::LinearConstraintModel) = nothing

LenticulumCore.assemble(f::LinearConstraintFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(LinearConstraintModel(f, p), LenticulumCore.ExactInversion()), st)

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:LinearConstraintModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    target = only(LenticulumCore.unobserved_channels(lens.model.polarity))
    msg, st = _message(f, target, inputs, ps, st)
    return Mycelium.combine(π, msg), st
end

# --- Messages --------------------------------------------------------------

"""
    Mycelium.factor_message(f::LinearConstraintFactor, target, polarity, inputs, prior, ps, st)

The factor → variable message, in canonical form. Writing
``d = c - \\sum_{i \\ne t} A_i m_i`` and ``R = Q + \\sum_{i \\ne t} A_i S_i A_i^\\top`` for
incoming ``\\mathcal{N}(m_i, S_i)``:

```math
\\mu_{c\\to t} \\;=\\; \\mathcal{N}^{-1}\\bigl(A_t^\\top R^{-1} d,\\ A_t^\\top R^{-1} A_t\\bigr)
```

One formula for every direction — the target channel appears only as "which ``A_i`` is
``A_t``". Specialising to two channels with ``A_x = -A``, ``A_y = I``, ``c = b`` reproduces
both of [`GaussianFactor`](@ref)'s messages exactly, forward and backward.

The message is **improper** whenever ``A_t`` is not full column rank — an equation
constrains only the directions it can see, and with fewer equations than unknowns
(``m < \\dim x_t``) that is the normal case, not an edge case. Only the canonical form can
say this; see `beliefs.md`.

Returns `TrivialBelief()` if any non-target channel carries no information: a free variable
elsewhere in the equation makes the whole thing vacuous. See `constraint.md` §4 for the
sharper answer this gives up.
"""
Mycelium.factor_message(
    f::LinearConstraintFactor, target::Symbol, polarity, inputs, prior, ps, st
) = _message(f, target, inputs, ps, st)

function _message(f::LinearConstraintFactor{names}, target::Symbol, inputs, ps, st) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this LinearConstraintFactor (has $names)"))
    d = float.(ps.c)
    R = float.(f.Q)
    for n in names
        n === target && continue
        m, S = _moments(_get(inputs, n), f.dims[n])
        m === nothing && return (LenticulumCore.TrivialBelief(), st)
        An = ps.A[n]
        d = d .- An * m
        R = R .+ An * S * An'
    end
    At = ps.A[target]
    Ri = inv(_chol(R, "Q + Σ AᵢSᵢAᵢᵀ"))
    return (GaussianBelief(At' * (Ri * d), At' * Ri * At), st)
end

# --- The Bethe contribution ------------------------------------------------

"""
    residual_statistics(f::LinearConstraintFactor, msgs, ps) -> (r̄, Cov_r, H)

Mean and covariance of ``r = \\sum_i A_i x_i - c`` under the factor's own belief
``b_c \\propto f_c \\prod_i \\mu_{i\\to c}``, plus that belief's entropy.

Dirac channels are **substituted out** rather than represented: a clamped channel moves its
contribution into the constant, ``c \\mapsto c - A_j v_j``, and drops out of the joint. That
is what keeps the joint precision finite, and it is the n-ary version of the three hand-written
cases in `gaussian.jl`. With every channel clamped the residual is deterministic and the
entropy is zero, which falls out of the same code rather than needing its own branch.
"""
function residual_statistics(f::LinearConstraintFactor{names}, msgs, ps) where {names}
    c_eff = float.(ps.c)
    free = Symbol[]
    for n in names
        b = _get(msgs, n)
        if b isa LenticulumCore.DiracBelief
            c_eff = c_eff .- ps.A[n] * _vec(b.value)
        else
            push!(free, n)
        end
    end
    isempty(free) && return (-c_eff, zeros(f.m, f.m), 0.0)

    G = reduce(hcat, (ps.A[n] for n in free))
    canon = [_canon(_get(msgs, n), f.dims[n]) for n in free]
    Qi = inv(_chol(f.Q, "Q"))
    Λ = G' * Qi * G + _blockdiag(map(last, canon))
    η = G' * (Qi * c_eff) + reduce(vcat, map(first, canon))
    bel = GaussianBelief(η, Λ)
    m, Σ = belief_mean(bel), belief_cov(bel)
    return (G * m .- c_eff, G * Σ * G', Mycelium.variable_entropy(bel))
end

# n-ary block diagonal. `gaussian.jl`'s `_blockdiag` takes exactly two blocks.
function _blockdiag(blocks::AbstractVector{<:AbstractMatrix})
    n = sum(b -> size(b, 1), blocks; init = 0)
    M = zeros(n, n)
    o = 0
    for b in blocks
        k = size(b, 1)
        M[(o + 1):(o + k), (o + 1):(o + k)] .= b
        o += k
    end
    return M
end

"""
    Mycelium.local_free_energy(f::LinearConstraintFactor, msgs, ps, st)

``F_c = U_c - H(b_c)``, graded as `(fit, complexity, negentropy)` — the same decomposition
as [`GaussianFactor`](@ref), computed from [`residual_statistics`](@ref).
"""
function Mycelium.local_free_energy(f::LinearConstraintFactor, msgs, ps, st)
    r̄, Cr, H = residual_statistics(f, msgs, ps)
    Qi = inv(_chol(f.Q, "Q"))
    fit = (r̄' * (Qi * r̄) + tr(Qi * Cr)) / 2
    complexity = logdet(_chol(2π .* f.Q, "2πQ")) / 2
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -H)), st
end
