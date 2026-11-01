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

**Acausal**: no channel is privileged. Every channel can be solved for — "solve this
equation for `x_t`, given the others" — and every one of these messages is exact; the others
may also be latent (`supported_polarities`). As ``Q \\to 0`` the factor becomes the hard relation
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
    size(Q) == (m, m) || throw(ArgumentError("noise covariance must have size ($m, $m), matching the residual dimension"))
    # Exact `issymmetric` rejects J S Jᵀ, which rounds asymmetrically (LP-019 follow-up).
    LenticulumCore._nearly_symmetric(Q) || throw(ArgumentError("noise covariance must be symmetric"))
    Q = Matrix(LenticulumCore._symmetric_part(Q))   # one consistent matrix from here on
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

# The polarities of a relation: one channel `Unobserved()`, each other `Observed()` or `Latent()`.
# Each assignment is its own `Polarity` type, so the list is capped, as for exposed factors.
const _CONSTRAINT_ENUMERATION_CAP = 6
function _one_unobserved_polarities(names::Tuple)
    n = length(names)
    n <= _CONSTRAINT_ENUMERATION_CAP || throw(ArgumentError(
        "a relation with $n channels has $(n * 2^(n - 1)) polarities; use `supports_polarity` \
         or the lazy `LenticulumCore.polarities` instead of listing them"))
    O, U, L = LenticulumCore.Observed(), LenticulumCore.Unobserved(), LenticulumCore.Latent()
    pols = LenticulumCore.Polarity[]
    for t in names, mask in 0:(2^(n - 1) - 1)
        others = Tuple(c for c in names if c !== t)
        role(c) = c === t ? U : (mask >> (findfirst(==(c), others) - 1)) & 1 == 1 ? L : O
        push!(pols, LenticulumCore.Polarity(NamedTuple{names}(map(role, names))))
    end
    return Tuple(pols)
end

"""
    LenticulumCore.supported_polarities(f::LinearConstraintFactor)

Every assignment with exactly one `Unobserved()` channel and each other channel `Observed()` or
`Latent()`: ``n \\cdot 2^{n-1}`` of them, the set [`supports_polarity`](@ref LenticulumCore.supports_polarity) accepts (L-SPEC-001).
The ``n`` with every other channel observed are the ones that matter most: one per channel, and
**none of them is the factor's "real" direction**. The equation was never written in a
direction. This is the enumerated form of "a relation, not a function".

Contrast [`GaussianFactor`](@ref), which has exactly two regardless of anything. Above
$(_CONSTRAINT_ENUMERATION_CAP) channels the list throws, as the contract permits;
`hasnopolarity` and `isunidirectional` answer without it.
"""
LenticulumCore.supported_polarities(f::LinearConstraintFactor{names}) where {names} =
    _one_unobserved_polarities(names)
LenticulumCore.hasnopolarity(::LinearConstraintFactor) = false
LenticulumCore.isunidirectional(::LinearConstraintFactor{names}) where {names} = length(names) == 1

"""
    LenticulumCore.supports_polarity(f::LinearConstraintFactor, p)

Any polarity over exactly this channel set, in any order, with **exactly one** `Unobserved()`
channel.

A `Latent()` channel is integrated out under a flat prior (see
[`Mycelium.factor_message`](@ref) below): the message keeps what the equation says in the
directions that channel cannot reach, and is uninformative when its `A` has full row rank.
A latent channel is a legal modelling choice, and "this equation tells you nothing more about
``x_t`` once ``x_j`` is free" is a correct answer, not an error.
"""
function LenticulumCore.supports_polarity(
    f::LinearConstraintFactor{names}, p::LenticulumCore.Polarity
) where {names}
    Set(keys(p)) == Set(names) || return false
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

# The pointwise energy is the residual, scalarised by SquaredNorm(Q⁻¹), as for GaussianFactor;
# the graded spaces above are those of the free energy. The equation has no direction, so
# neither depends on the polarity.
LenticulumCore.pointwise_energyspace(f::LinearConstraintFactor) =
    LenticulumCore.EuclideanEnergySpace(size(f.Q, 1))
LenticulumCore.pointwise_scalarisation(f::LinearConstraintFactor) =
    LenticulumCore.SquaredNorm(inv(Symmetric(f.Q)); check = false)

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

function LenticulumCore.likelihood_message(
    lens::LenticulumCore.BayesianLens{<:LinearConstraintModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    target = only(LenticulumCore.unobserved_channels(lens.model.polarity))
    return _message(f, target, inputs, ps, st)
end

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:LinearConstraintModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    msg, st = LenticulumCore.likelihood_message(lens, π, inputs, ps, st)
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

A non-target channel that carries no information (no message, `TrivialBelief()`, a
`Latent()` channel) is integrated out under a flat prior, the limit ``\\tau \\to \\infty`` of
``\\mathcal N(0, \\tau \\Sigma_0)``: ``R^{-1}`` becomes ``R^{-1} - R^{-1} G (G^\\top R^{-1} G)^{+} G^\\top R^{-1}``,
with ``G`` stacking those channels' ``A``'s, the same limit for every ``\\Sigma_0 \\succ 0``. The
equation then still constrains the directions ``G`` cannot reach. The message is
`TrivialBelief()` when nothing is left, as when ``G`` has full row rank (`constraint.md` §4.1).
"""
Mycelium.factor_message(
    f::LinearConstraintFactor, target::Symbol, polarity, inputs, prior, ps, st
) = _message(f, target, inputs, ps, st)

function _message(f::LinearConstraintFactor{names}, target::Symbol, inputs, ps, st) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this LinearConstraintFactor (has $names)"))
    d = float.(ps.c)
    R = float.(f.Q)
    flat = Symbol[]
    for n in names
        n === target && continue
        m, S = _moments(_get(inputs, n), f.dims[n])
        if m === nothing
            push!(flat, n)
            continue
        end
        An = ps.A[n]
        d = d .- An * m
        R = R .+ An * S * An'
    end
    At = ps.A[target]
    if isempty(flat)
        Ri = inv(_chol(R, "Q + Σ AᵢSᵢAᵢᵀ"))
        return (GaussianBelief(At' * (Ri * d), At' * Ri * At), st)
    end
    T, Li = _residual_metric(R, flat, ps.A, "Q + Σ AᵢSᵢAᵢᵀ")
    Ta = T * At
    _projected_rank(Ta, Li * At) == 0 && return (LenticulumCore.TrivialBelief(), st)
    return (GaussianBelief(Ta' * (T * d), Ta' * Ta), st)
end

# The residual metric with the flat channels integrated out, as a factor T with TᵀT the metric.
# For R = LLᵀ and G the stacked coefficients of the channels in `flat`, a prior N(0, τΣ₀) on them
# gives (R + τGΣ₀Gᵀ)⁻¹ → R⁻¹ − R⁻¹G(GᵀR⁻¹G)⁺GᵀR⁻¹ as τ → ∞, for every Σ₀ ≻ 0: the R⁻¹-orthogonal
# projection that removes range(G). So T = U⊥ᵀL⁻¹, with U⊥ an orthonormal basis of the
# complement of range(L⁻¹G); it has no rows when G has full row rank. The normaliser that
# diverges with τ is the same for every value of the target, and for every outcome of a
# switching factor. Also returns L⁻¹, the scale for `_projected_rank`.
function _residual_metric(R, flat, A, what)
    Li = inv(_chol(R, what).L)
    isempty(flat) && return Li, Li
    B = Li * reduce(hcat, (A[n] for n in flat))
    F = LinearAlgebra.svd(B; full = true)
    tol = maximum(size(B)) * eps(Float64) * maximum(F.S; init = 0.0)
    r = count(>(tol), F.S)
    return F.U[:, (r + 1):end]' * Li, Li
end

# The rank of the projected coefficient T Aₜ. Rounding in the projection leaves singular values
# of order eps·‖L⁻¹Aₜ‖ in the directions it removed, so those count as zero.
function _projected_rank(Ta, LiAt)
    s = LinearAlgebra.svdvals(Ta)
    isempty(s) && return 0
    return count(>(maximum(size(LiAt)) * eps(Float64) * maximum(LinearAlgebra.svdvals(LiAt))), s)
end

"""
    Mycelium.mean_field_message(f::LinearConstraintFactor, target, polarity, beliefs, ps, st)

The message of a `MeanField()` node: ``\\exp \\mathbb E_q[\\log f]`` as a function of ``x_t``, over the other
channels' current marginals ``q_i``. ``\\log f`` is quadratic, so only their means enter, and the
message is ``\\mathcal N^{-1}(A_t^\\top Q^{-1} \\bar d, A_t^\\top Q^{-1} A_t)`` with ``\\bar d = c - \\sum_{i \\ne t} A_i\\, \\mathbb E_q[x_i]``:
the belief-propagation message with every input replaced by a point mass at its mean. A mixture
marginal enters by its mean; a channel without a proper marginal yet is integrated out under a
flat prior, as in `factor_message`.
"""
Mycelium.mean_field_message(f::LinearConstraintFactor, target::Symbol, polarity, beliefs, ps, st) =
    _message(f, target, _mean_points(f.dims, target, beliefs), ps, st)

# Point masses at the means of the marginals in `beliefs`, for the channels in `dims` other than
# `target` that have one.
function _mean_points(dims::NamedTuple{names}, target, beliefs) where {names}
    chans, pts = Symbol[], Any[]
    for n in names
        n === target && continue
        μ = _mf_mean(_get(beliefs, n))
        μ === nothing && continue
        push!(chans, n); push!(pts, LenticulumCore.DiracBelief(μ))
    end
    return NamedTuple{Tuple(chans)}(Tuple(pts))
end
_mf_mean(b::LenticulumCore.DiracBelief) = _vec(b.value)
_mf_mean(b::GaussianBelief) = isproper(b) ? belief_mean(b) : nothing
_mf_mean(b::LenticulumCore.MixtureBelief{<:GaussianBelief}) =
    all(isproper, b.components) ? belief_mean(moment_match(b)) : nothing
_mf_mean(b) = nothing

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

Mycelium.uses_cavity(::LinearConstraintFactor) = true

"""
    Mycelium.factor_residual_moments(f::LinearConstraintFactor, msgs, ps) -> (mean, cov)

The mean and full covariance of the residual under the factor's belief, from
[`residual_statistics`](@ref). Mixture messages are collapsed to their moments first. Used by
Mycelium's `SampleBuffer`.
"""
function Mycelium.factor_residual_moments(f::LinearConstraintFactor, msgs, ps)
    r̄, C, _ = residual_statistics(f, map(_collapse, msgs), ps)
    return r̄, C
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

"""
    Mycelium.factor_belief(f::LinearConstraintFactor, msgs, ps, st) -> FrozenGaussianBelief
    Mycelium.belief_free_energy(f::LinearConstraintFactor, b, ps, st)

The factor belief, and the local free energy at a fixed one, with ``r = \\sum_i A_i x_i - c``
averaged under `b`; it agrees with `local_free_energy` at the belief it was evaluated at. Its
parameter derivative is the graph gradient's (LP-020). See [`FrozenGaussianBelief`](@ref).
"""
Mycelium.factor_belief(f::LinearConstraintFactor, msgs, ps, st) =
    _affine_gaussian_belief(ps.A, -ps.c, inv(_chol(f.Q, "Q")), msgs, f.dims)
function Mycelium.belief_free_energy(f::LinearConstraintFactor, b::FrozenGaussianBelief, ps, st)
    r̄, Cr = _frozen_residual(ps.A, -ps.c, b)
    Qi = inv(Symmetric(f.Q))                                   # not inv(::Cholesky): Zygote has no rule
    fit = (r̄' * (Qi * r̄) + tr(Qi * Cr)) / 2
    complexity = logdet(_chol(2π .* f.Q, "2πQ")) / 2
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -b.H)), st
end

function LenticulumCore.free_energy(f::LinearConstraintFactor{names}, π, y::NamedTuple, ps, st) where {names}
    missing_chans = setdiff(names, keys(y))
    msgs = length(missing_chans) == 1 ? merge(NamedTuple{(only(missing_chans),)}((π,)), y) : y
    return Mycelium.local_free_energy(f, msgs, ps, st)
end

function LenticulumCore.entropy(f::LinearConstraintFactor, π, y::NamedTuple, ps, st)
    F, st = LenticulumCore.free_energy(f, π, y, ps, st)
    return LenticulumCore.GradedEnergy((fit = 0.0, complexity = 0.0, negentropy = -F[:negentropy])), st
end
