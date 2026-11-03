# ---------------------------------------------------------------------------
# The switching linear constraint:   0 = Σᵢ Aᵢ xᵢ − b_k + ε,   ε ~ N(0, Q),   k ∈ 1:K, P(k) = w_k
#
# A `LinearConstraintFactor` whose right-hand side is selected by one discrete outcome: a
# measurement of the linear observable Σᵢ Aᵢ xᵢ. It is the smallest factor that leaves the
# Gaussian class (`Switching Constraint Factors.md`): composed, it represents exactly the
# finite Gaussian mixtures, and every message and the local free energy stay closed form.
#
# The A's and Q are shared by the outcomes (the "location-shift" case of §11), so every outcome
# gives a Gaussian with the same precision: the message to a continuous channel is a mixture of
# K Gaussians that differ only in their means, and the factor's belief has a closed-form entropy,
# H(k, x) = H(k) + H(x | k).
#
# See `switching.md`.
# ---------------------------------------------------------------------------

"""
    SwitchingConstraintFactor(dims::NamedTuple, K::Int, m::Int; switch = :k, noise = 0.1,
                              init_coeff, init_offsets)

The switching soft linear relation ``0 = \\sum_i A_i x_i - b_k + \\varepsilon``,
``\\varepsilon \\sim \\mathcal N(0, Q)``, with a discrete outcome ``k \\in \\{1, \\dots, K\\}`` of weight
``w_k``:

```math
f(x_1, \\ldots, x_n, k) \\;=\\; w_k\\, \\mathcal N\\Bigl(\\textstyle\\sum_i A_i x_i - b_k;\\ 0,\\ Q\\Bigr).
```

`dims` names the continuous channels and their dimensions, `m` is the residual dimension, and
the outcome is the channel `switch`. `Q` is a fixed hyperparameter, as in
[`LinearConstraintFactor`](@ref); the parameters are `(A, b, logw)`: `A` a `NamedTuple` keyed by
channel and shared by the outcomes, `b` the `m × K` matrix of right-hand sides, and `logw` the
unnormalised log-weights (`w = softmax(logw)`).

Every channel, the outcome included, can be the target of an exact message, with the others
observed or latent. With `K = 1` it is a `LinearConstraintFactor` with
`c = b₁`, message for message.

```julia
# a two-component mixture on a line in the plane: z = b_k + ε
f  = SwitchingConstraintFactor((z = 2,), 2, 2; noise = 0.1)
ps = (A = (z = Matrix(1.0I, 2, 2),), b = [-1.0 1.0; 0.0 0.0], logw = zeros(2))
```

See `switching.md` for the messages, the free energy and what is not done.
"""
struct SwitchingConstraintFactor{names,D<:Tuple,Q<:AbstractMatrix,W,B} <:
       LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names,D}
    K::Int
    m::Int
    switch::Symbol
    Q::Q
    init_coeff::W
    init_offsets::B
end

function SwitchingConstraintFactor(
    dims::NamedTuple,
    K::Int,
    m::Int;
    switch::Symbol = :k,
    noise = 0.1,
    init_coeff = (rng, m, n) -> randn(rng, m, n) ./ sqrt(n),
    init_offsets = (rng, m, K) -> randn(rng, m, K),
)
    isempty(dims) && throw(ArgumentError("a switching constraint needs at least one continuous channel"))
    all(d -> d isa Int && d > 0, values(dims)) ||
        throw(ArgumentError("channel dimensions must be positive Ints; got $(dims)"))
    switch in keys(dims) && throw(ArgumentError("the switch channel :$switch is also a continuous channel"))
    K > 0 || throw(ArgumentError("the number of outcomes K must be positive; got $K"))
    m > 0 || throw(ArgumentError("the residual dimension m must be positive; got $m"))
    Q = _as_cov(noise, m)
    size(Q) == (m, m) || throw(ArgumentError("noise covariance must have size ($m, $m), matching the residual dimension"))
    LenticulumCore._nearly_symmetric(Q) || throw(ArgumentError("noise covariance must be symmetric"))
    Q = Matrix(LenticulumCore._symmetric_part(Q))
    isposdef(Symmetric(Q)) || throw(ArgumentError("noise covariance Q must be positive definite"))
    return SwitchingConstraintFactor(dims, K, m, switch, Q, init_coeff, init_offsets)
end

LuxCore.initialparameters(rng::AbstractRNG, f::SwitchingConstraintFactor{names}) where {names} = (
    A = NamedTuple{names}(map(n -> f.init_coeff(rng, f.m, n), values(f.dims))),
    b = f.init_offsets(rng, f.m, f.K),
    logw = zeros(f.K),
)
LuxCore.initialstates(::AbstractRNG, ::SwitchingConstraintFactor) = NamedTuple()
LuxCore.parameterlength(f::SwitchingConstraintFactor) = f.m * sum(values(f.dims)) + f.m * f.K + f.K

_allnames(f::SwitchingConstraintFactor{names}) where {names} = (names..., f.switch)

LenticulumCore.channels(f::SwitchingConstraintFactor{names}) where {names} =
    (map((n, d) -> LenticulumCore.Channel(n, d), names, values(f.dims))..., LenticulumCore.Channel(f.switch, f.K))

"""
    LenticulumCore.supported_polarities(f::SwitchingConstraintFactor)

Every assignment with exactly one `Unobserved()` channel, the outcome included, and each other
channel `Observed()` or `Latent()`, as for [`LinearConstraintFactor`](@ref): the set
[`supports_polarity`](@ref LenticulumCore.supports_polarity) accepts.
"""
LenticulumCore.supported_polarities(f::SwitchingConstraintFactor) = _one_unobserved_polarities(_allnames(f))
LenticulumCore.hasnopolarity(::SwitchingConstraintFactor) = false
LenticulumCore.isunidirectional(::SwitchingConstraintFactor) = false

"""
    LenticulumCore.supports_polarity(f::SwitchingConstraintFactor, p)

Any polarity over exactly this channel set, in any order, with exactly one `Unobserved()`
channel. A `Latent()` outcome is marginalised under the factor's own weights; a `Latent()`
continuous channel is integrated out under a flat prior, as for [`LinearConstraintFactor`](@ref).
Either way the channel is a graph variable the message sums or integrates over, not a latent
space of the open model, which stays pure.
"""
function LenticulumCore.supports_polarity(f::SwitchingConstraintFactor, p::LenticulumCore.Polarity)
    Set(keys(p)) == Set(_allnames(f)) || return false
    return length(LenticulumCore.unobserved_channels(p)) == 1
end

LenticulumCore.islearnable(::SwitchingConstraintFactor) = true

LenticulumCore.energyspace(::SwitchingConstraintFactor) = LenticulumCore.GradedEnergySpace((
    fit = LenticulumCore.ScalarEnergySpace(),
    complexity = LenticulumCore.ScalarEnergySpace(),
    negentropy = LenticulumCore.ScalarEnergySpace(),
))
LenticulumCore.scalarisation(::SwitchingConstraintFactor) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(),
    complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation(),
))

_logweights(ps) = (l = float.(ps.logw); l .- _logsumexp(l))

function _outcome(f::SwitchingConstraintFactor, v)
    k = only(_vec(v))
    isinteger(k) && 1 <= k <= f.K || throw(ArgumentError("an observed outcome must be an index in 1:$(f.K); got $v"))
    return Int(k)
end

# log P(k) carried by an incoming outcome belief; `nothing` (latent, or no message) is flat.
_switch_logp(::Nothing, K) = zeros(K)
_switch_logp(::LenticulumCore.TrivialBelief, K) = zeros(K)
_switch_logp(b::LenticulumCore.CategoricalBelief, K) =
    (length(b.logp) == K || throw(DimensionMismatch("outcome belief has $(length(b.logp)) states, the factor $K")); copy(b.logp))
function _switch_logp(b::LenticulumCore.DiracBelief, K)
    k = b.value
    k isa Integer && 1 <= k <= K || throw(ArgumentError("an observed outcome must be an index in 1:$K; got $k"))
    return [i == k ? 0.0 : -Inf for i in 1:K]
end

"""
    residual(f::SwitchingConstraintFactor, vals::NamedTuple, ps) -> Vector

``r = \\sum_i A_i x_i - b_k`` for the outcome `vals[f.switch]` (an index in `1:K`).
"""
function residual(f::SwitchingConstraintFactor{names}, vals::NamedTuple, ps) where {names}
    haskey(vals, f.switch) || throw(ArgumentError("residual needs the outcome :$(f.switch)"))
    k = _outcome(f, vals[f.switch])
    r = -float.(ps.b[:, k])
    for n in names
        haskey(vals, n) || throw(ArgumentError(
            "residual needs a value for every channel; :$n is missing (have $(keys(vals)))"))
        r = r .+ ps.A[n] * _vec(vals[n])
    end
    return r
end

# The pointwise energy is −log f(x, k): the squared residual, the noise normaliser and −log w_k.
function LenticulumCore.energy(f::SwitchingConstraintFactor, x, a, y, ps, st)
    vals = _channel_values(x, y)
    r = residual(f, vals, ps)
    k = _outcome(f, vals[f.switch])
    C = _chol(f.Q, "Q")
    e = LinearAlgebra.dot(r, C \ r) / 2 + logdet(C) / 2 + f.m * log(2π) / 2 - _logweights(ps)[k]
    return e, st
end
LenticulumCore.pointwise_energyspace(::SwitchingConstraintFactor) = LenticulumCore.ScalarEnergySpace()
LenticulumCore.pointwise_scalarisation(::SwitchingConstraintFactor) = LenticulumCore.IdentityScalarisation()

# --- The open model and its exact inversion --------------------------------

"""
    SwitchingConstraintModel(factor, polarity)

The open model the polarity selects: "infer this channel, the outcome or a continuous one,
given the others". Pure, as for [`LinearConstraintModel`](@ref).
"""
struct SwitchingConstraintModel{P} <: LenticulumCore.AbstractOpenModel
    factor::SwitchingConstraintFactor
    polarity::P
end
LenticulumCore.ispure(::SwitchingConstraintModel) = true
LenticulumCore.latentspace(::SwitchingConstraintModel) = nothing

LenticulumCore.assemble(f::SwitchingConstraintFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(SwitchingConstraintModel(f, p), LenticulumCore.ExactInversion()), st)

function LenticulumCore.likelihood_message(
    lens::LenticulumCore.BayesianLens{<:SwitchingConstraintModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    return _switch_message(f, only(LenticulumCore.unobserved_channels(lens.model.polarity)), inputs, ps, st)
end

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:SwitchingConstraintModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    msg, st = LenticulumCore.likelihood_message(lens, π, inputs, ps, st)
    return Mycelium.combine(π, msg), st
end

# --- Messages --------------------------------------------------------------

"""
    Mycelium.factor_message(f::SwitchingConstraintFactor, target, polarity, inputs, prior, ps, st)

With ``d_k = b_k - \\sum_{i \\ne t} A_i m_i`` and ``R = Q + \\sum_{i \\ne t} A_i S_i A_i^\\top`` for incoming
``\\mathcal N(m_i, S_i)``, and ``\\pi_k`` the incoming outcome belief (flat if latent):

- to the outcome: ``P(k) \\propto w_k\\, \\mathcal N(d_k; 0, R)``, a `CategoricalBelief`;
- to a continuous channel ``t``: ``\\sum_k \\pi_k w_k\\, \\mathcal N(A_t x_t; d_k, R)``, a `MixtureBelief`
  of the canonical Gaussians ``\\mathcal N^{-1}(A_t^\\top R^{-1} d_k, A_t^\\top R^{-1} A_t)`` with the
  per-outcome constants in the weights. All components share one precision; they are improper
  when ``A_t`` is not full column rank, declared so (`GaussianBelief(η, Λ, true)`), and weighted
  as the unnormalised exponentials that `Mycelium`'s mixture product expects. A single possible
  outcome gives a `GaussianBelief`.

A Gaussian-mixture input is collapsed to its moments first, as in `LinearConstraintFactor`
(assumed-density filtering; `gaussian.md`, "Mixture inputs"). An uninformative continuous input
(no message, `TrivialBelief()`, `Latent()`) is integrated out under a flat prior, as in
`LinearConstraintFactor`, and so is an improper Gaussian input along its null space.
``R^{-1}`` is replaced by the limit that projects out the range of those flat directions under
the ``A``'s, so only the part of ``d_k`` they cannot reach separates the outcomes. When they
cover every residual direction nothing is left: the outcome gets the factor's own weights and a
continuous target `TrivialBelief()`.
"""
Mycelium.factor_message(f::SwitchingConstraintFactor, target::Symbol, polarity, inputs, prior, ps, st) =
    _switch_message(f, target, inputs, ps, st)

function _switch_message(f::SwitchingConstraintFactor{names}, target::Symbol, inputs, ps, st) where {names}
    target in _allnames(f) || throw(ArgumentError(
        "channel :$target is not a channel of this SwitchingConstraintFactor (has $(_allnames(f)))"))
    lw = _logweights(ps)
    lpk = target === f.switch ? nothing : _switch_logp(_get(inputs, f.switch), f.K)
    parts = [(n, _split_moments(_get(inputs, n), f.dims[n])) for n in names if n !== target]
    s, R, G = _accumulate(ps.A, f.Q, parts)
    T, Li = _residual_metric(R, G, "Q + Σ AᵢSᵢAᵢᵀ")
    Z = T * (ps.b .- s)                                        # column k is T d_k
    quad = vec(sum(abs2, Z; dims = 1))
    if target === f.switch
        l = lw .- quad ./ 2
        return (LenticulumCore.CategoricalBelief(Val(:log), l .- _logsumexp(l), nothing), st)
    end
    l = lpk .+ lw .- quad ./ 2
    ks = [k for k in 1:f.K if l[k] > -Inf]
    isempty(ks) && throw(ArgumentError("the incoming outcome belief gives every outcome zero probability"))
    At = ps.A[target]
    Ta = T * At
    rt = _projected_rank(Ta, Li * At)
    rt == 0 && return (LenticulumCore.TrivialBelief(), st)
    # Properness is decided by the rank, not by rounding, and declared on the components, so that
    # Mycelium's mixture product weights them as likelihoods (switching.md §6).
    proper = rt == size(At, 2)
    Λ = Ta' * Ta
    comps = [GaussianBelief(Ta' * Z[:, k], Λ, !proper) for k in ks]
    length(ks) == 1 && return (only(comps), st)
    lk = l[ks]
    if proper                                                  # normalised components: move ½ηᵀΛ⁻¹η into the weight
        CΛ = _chol(Λ, "AₜᵀR⁻¹Aₜ")
        lk = lk .+ [LinearAlgebra.dot(c.η, CΛ \ c.η) / 2 for c in comps]
    end
    return (LenticulumCore.MixtureBelief(Val(:log), comps, lk .- _logsumexp(lk)), st)
end

"""
    Mycelium.mean_field_message(f::SwitchingConstraintFactor, target, polarity, beliefs, ps, st)

The message of a `MeanField()` node, ``\\exp \\mathbb E_q[\\log f]``, over the other channels' current
marginals. With ``\\bar s = \\sum_{i \\ne t} A_i\\, \\mathbb E_q[x_i]`` and ``q`` the outcome's marginal:

- to a continuous channel: ``\\mathcal N^{-1}(A_t^\\top Q^{-1}(\\bar b - \\bar s),\\ A_t^\\top Q^{-1} A_t)`` with ``\\bar b = \\sum_k q_k b_k``;
- to the outcome: ``P(k) \\propto w_k \\exp(-\\tfrac12 (b_k - \\bar s)^\\top Q^{-1} (b_k - \\bar s))``.

Only means enter, since ``\\log f`` is quadratic in the continuous channels for each outcome and the
covariance terms do not depend on ``x_t`` or ``k``. An outcome without a marginal yet counts with
the factor's weights, which is where the iteration starts. A continuous channel without a proper
marginal yet is integrated out under a flat prior, as in `factor_message`, with ``Q^{-1}`` in place
of ``R^{-1}``; so is an improper marginal along the directions where it is flat.

A rank-deficient precision is declared improper, as in `factor_message`.
"""
function Mycelium.mean_field_message(f::SwitchingConstraintFactor{names}, target::Symbol, polarity, beliefs, ps, st) where {names}
    target in _allnames(f) || throw(ArgumentError(
        "channel :$target is not a channel of this SwitchingConstraintFactor (has $(_allnames(f)))"))
    lw = _logweights(ps)
    parts = [(n, _mf_split(_get(beliefs, n), f.dims[n])) for n in names if n !== target]
    sbar, _, G = _accumulate(ps.A, f.Q, parts)
    T, Li = _residual_metric(f.Q, G, "Q")
    if target === f.switch
        l = lw .- vec(sum(abs2, T * (ps.b .- sbar); dims = 1)) ./ 2
        return (LenticulumCore.CategoricalBelief(Val(:log), l .- _logsumexp(l), nothing), st)
    end
    kb = _get(beliefs, f.switch)
    q = kb === nothing || kb isa LenticulumCore.TrivialBelief ? exp.(lw) : exp.(_switch_logp(kb, f.K))
    At = ps.A[target]
    Ta = T * At
    rt = _projected_rank(Ta, Li * At)
    rt == 0 && return (LenticulumCore.TrivialBelief(), st)
    return (GaussianBelief(Ta' * (T * (ps.b * q .- sbar)), Ta' * Ta, rt < size(At, 2)), st)
end

# --- The Bethe contribution ------------------------------------------------

"""
    switch_statistics(f::SwitchingConstraintFactor, msgs, ps)
        -> (q, r̄, Cov_r, H_x)

The factor's own belief ``b_c(x, k) \\propto f(x, k)\\, \\mu_k(k) \\prod_i \\mu_i(x_i)``, summarised:
the outcome posterior `q`, the per-outcome residual means ``\\bar r_k`` (columns), the residual
covariance (one for all outcomes) and the entropy ``H(x \\mid k)`` (also one for all outcomes).
Dirac channels are substituted out, as in [`residual_statistics`](@ref). A Gaussian-mixture
message is collapsed to its moments first.
"""
function switch_statistics(f::SwitchingConstraintFactor{names}, msgs, ps) where {names}
    shift = zeros(f.m)
    free = Symbol[]
    for n in names
        b = _get(msgs, n)
        if b isa LenticulumCore.DiracBelief
            shift = shift .+ ps.A[n] * _vec(b.value)
        else
            push!(free, n)
        end
    end
    Ceff = ps.b .- shift                                       # column k: c_k = b_k − Σ_Dirac A_j v_j
    CQ = _chol(f.Q, "Q")
    QiC = CQ \ Ceff
    logZ = [-LinearAlgebra.dot(Ceff[:, k], QiC[:, k]) / 2 for k in 1:f.K]
    if isempty(free)
        rbar, Cr, Hx = -Ceff, zeros(f.m, f.m), 0.0
    else
        G = reduce(hcat, (ps.A[n] for n in free))
        canon = [_canon(_collapse(_get(msgs, n)), f.dims[n]) for n in free]
        Λ = G' * (CQ \ G) + _blockdiag(map(last, canon))
        CΛ = _chol(Λ, "factor belief precision")
        etas = G' * QiC .+ reduce(vcat, map(first, canon))    # column k: η_k
        M = CΛ \ etas
        logZ = logZ .+ [LinearAlgebra.dot(etas[:, k], M[:, k]) / 2 for k in 1:f.K]
        rbar = G * M .- Ceff
        Cr = G * (CΛ \ G')
        Hx = Mycelium.variable_entropy(GaussianBelief(zeros(size(Λ, 1)), Λ))
    end
    lp = _switch_logp(_get(msgs, f.switch), f.K)
    all(==(-Inf), lp) && throw(ArgumentError("the incoming outcome belief gives every outcome zero probability"))
    l = lp .+ _logweights(ps) .+ logZ
    q = exp.(l .- _logsumexp(l))
    return q, rbar, Cr, Hx
end

_collapse(b::LenticulumCore.MixtureBelief{<:GaussianBelief}) = moment_match(b)
_collapse(b) = b

Mycelium.uses_cavity(::SwitchingConstraintFactor) = true

"""
    Mycelium.factor_residual_moments(f::SwitchingConstraintFactor, msgs, ps) -> (mean, cov)

The mean and full covariance of the residual ``r = \\sum_i A_i x_i - b_k`` under the factor's belief,
over the outcome as well: ``\\bar r = \\sum_k q_k \\bar r_k`` and
``\\operatorname{Cov} r = \\operatorname{Cov}(r \\mid k) + \\sum_k q_k (\\bar r_k - \\bar r)(\\bar r_k - \\bar r)^\\top``, from
[`switch_statistics`](@ref). Used by Mycelium's `SampleBuffer`.
"""
function Mycelium.factor_residual_moments(f::SwitchingConstraintFactor, msgs, ps)
    q, rbar, Cr, _ = switch_statistics(f, msgs, ps)
    μ = rbar * q
    D = rbar .- μ
    return μ, Cr + D * Diagonal(q) * D'
end

"""
    Mycelium.local_free_energy(f::SwitchingConstraintFactor, msgs, ps, st)

``F_c = \\mathbb E_{b_c}[-\\log f] - H(b_c)``, exact for Gaussian, Dirac and categorical messages,
since ``H(b_c) = H(q) + H(x \\mid k)``. Graded as `(fit, complexity, negentropy)`:
`fit` is ``\\sum_k q_k\\, \\tfrac12 \\mathbb E\\|r_k\\|^2_{Q^{-1}}``, `complexity` is
``\\tfrac12 \\log\\det 2\\pi Q - \\sum_k q_k \\log w_k``, and `negentropy` is ``-H(q) - H(x \\mid k)``.
"""
function Mycelium.local_free_energy(f::SwitchingConstraintFactor, msgs, ps, st)
    q, rbar, Cr, Hx = switch_statistics(f, msgs, ps)
    return _switch_free_energy(f, q, rbar, Cr, Hx, ps), st
end

# E_b[−log f] − H(b) for a belief with outcome posterior q, per-outcome residual means r̄ₖ, the
# shared residual covariance and H(x | k). Outcomes with q_k = 0 are skipped (0·log 0 = 0, and
# 0·log w_k = 0 when w_k = 0). Σ q_k = 1, so the trace term is outside the sum.
function _switch_free_energy(f::SwitchingConstraintFactor, q, rbar, Cr, Hx, ps)
    all(isfinite, q) || throw(ArgumentError("the outcome posterior is not finite: $q"))
    CQ = _chol(f.Q, "Q")
    lw = _logweights(ps)
    Qr = CQ \ rbar
    ks = [k for k in 1:f.K if q[k] > 0]
    fit = (tr(CQ \ Cr) + sum(q[k] * LinearAlgebra.dot(rbar[:, k], Qr[:, k]) for k in ks)) / 2
    complexity = (f.m * log(2π) + logdet(CQ)) / 2 - sum(q[k] * lw[k] for k in ks)
    Hq = -sum(q[k] * log(q[k]) for k in ks)
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -Hq - Hx))
end

"""
    Mycelium.factor_belief(f::SwitchingConstraintFactor, msgs, ps, st) -> FrozenGaussianBelief
    Mycelium.belief_free_energy(f::SwitchingConstraintFactor, b, ps, st)

The factor belief ``b_c(x, k)``, held fixed: the outcome posterior `q`, one mean per outcome and
the shared covariance. The local free energy at a fixed `b` averages ``r_k = \\sum_i A_i x_i - b_k``
and ``-\\log w_k`` under it, so `A`, `b` and `logw` all move and the entropy ``H(q) + H(x \\mid k)``
does not. Agrees with `local_free_energy` at the belief it was evaluated at (LP-020).
"""
function Mycelium.factor_belief(f::SwitchingConstraintFactor, msgs, ps, st)
    q, _, _, _ = switch_statistics(f, msgs, ps)
    return _affine_gaussian_belief(ps.A, -float.(ps.b), inv(_chol(f.Q, "Q")), msgs, f.dims; q)
end
function Mycelium.belief_free_energy(f::SwitchingConstraintFactor, b::FrozenGaussianBelief, ps, st)
    rbar, Cr = _frozen_residual(ps.A, -ps.b, b)
    return _switch_free_energy(f, b.q, rbar, Cr, b.H, ps), st
end

"""
    Mycelium.mean_field_energy(f::SwitchingConstraintFactor, marginals, ps, st)

The average energy of a `MeanField()` node under independent marginals on the continuous
channels and the outcome ``q``: ``\\sum_k q_k \\bigl(\\tfrac12 \\mathbb E\\lVert r_k\\rVert^2_{Q^{-1}} - \\log w_k\\bigr)``
plus the normaliser, with no entropy (Riegler et al. 2013, eq. 17). The outcome's marginal must be
a categorical or a point; the continuous ones as for `LinearConstraintFactor`.
"""
function Mycelium.mean_field_energy(f::SwitchingConstraintFactor{names}, marginals, ps, st) where {names}
    b = _switch_product_belief(f, marginals)
    F, st = Mycelium.belief_free_energy(f, b, ps, st)
    return _without_entropy(F), st
end
function _switch_product_belief(f::SwitchingConstraintFactor{names}, marginals) where {names}
    kb = _get(marginals, f.switch)
    kb isa Union{LenticulumCore.CategoricalBelief,LenticulumCore.DiracBelief} || throw(ArgumentError(
        "a MeanField() switching node's energy needs a categorical or point marginal on the outcome \
         :$(f.switch), got $(nameof(typeof(kb)))"))
    lp = _switch_logp(kb, f.K)
    q = exp.(lp .- _logsumexp(lp))
    return _product_belief(names, f.dims, marginals; K = f.K, q)
end

function LenticulumCore.free_energy(f::SwitchingConstraintFactor, π, y::NamedTuple, ps, st)
    missing_chans = setdiff(_allnames(f), keys(y))
    msgs = length(missing_chans) == 1 ? merge(NamedTuple{(only(missing_chans),)}((π,)), y) : y
    return Mycelium.local_free_energy(f, msgs, ps, st)
end

function LenticulumCore.entropy(f::SwitchingConstraintFactor, π, y::NamedTuple, ps, st)
    F, st = LenticulumCore.free_energy(f, π, y, ps, st)
    return LenticulumCore.GradedEnergy((fit = 0.0, complexity = 0.0, negentropy = -F[:negentropy])), st
end

# --- Mixture inputs expanded (MixtureReduction nodes, Extending Mycelium §4) -------------------

"""
    Mycelium.mixture_message(f::SwitchingConstraintFactor, target, polarity, inputs, prior, ps, st)
    Mycelium.mixture_message(f::LinearConstraintFactor, target, polarity, inputs, prior, ps, st)

The exact message when inputs are Gaussian mixtures, for `MixtureReduction()` nodes: each mixture input
``\\sum_j v_{ij}\\, \\mathcal N(m_{ij}, S_{ij})`` is expanded instead of collapsed. For a combination
``c`` of components, ``R_c = Q + \\sum_{i \\ne t} A_i S_{i c_i} A_i^\\top`` and ``d_{kc} = b_k - \\sum_{i \\ne t} A_i m_{i c_i}``,
and the combination carries the evidence ``\\prod_i v_{i c_i}\\, \\mathcal N(d_{kc}; 0, R_c)``, with the
determinant of ``R_c``, which now differs between combinations:

- to the outcome: ``P(k) \\propto w_k \\sum_c \\prod_i v_{i c_i}\\, \\mathcal N(d_{kc}; 0, R_c)``;
- to a continuous channel: the mixture over ``(k, c)`` of ``\\mathcal N(A_t x_t; d_{kc}, R_c)``, weighted
  by ``\\pi_k w_k \\prod_i v_{i c_i}``, one component each (a `GaussianBelief` if only one).

Components are normalised when proper and weighted as unnormalised exponentials when not, as in
`factor_message`. Flat directions of non-mixture inputs are integrated out as there; the
diverging normaliser is the same for every combination. The components of a mixture input must
be proper Gaussians. A `LinearConstraintFactor` is the case ``K = 1``, ``b_1 = c``. Without mixture
inputs the message is `factor_message`'s.
"""
function Mycelium.mixture_message(f::SwitchingConstraintFactor{names}, target::Symbol, polarity, inputs, prior, ps, st) where {names}
    target in _allnames(f) || throw(ArgumentError(
        "channel :$target is not a channel of this SwitchingConstraintFactor (has $(_allnames(f)))"))
    lw = _logweights(ps)
    to_switch = target === f.switch
    lpk = to_switch ? zeros(f.K) : _switch_logp(_get(inputs, f.switch), f.K)
    others = [n for n in names if n !== target]
    return _expanded_message(ps.A, f.Q, float.(ps.b), lw, lpk, f.dims, others, target, inputs, to_switch), st
end

function Mycelium.mixture_message(f::LinearConstraintFactor{names}, target::Symbol, polarity, inputs, prior, ps, st) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this LinearConstraintFactor (has $names)"))
    others = [n for n in names if n !== target]
    c = float.(ps.c)
    return _expanded_message(ps.A, f.Q, reshape(c, length(c), 1), [0.0], [0.0], f.dims, others, target, inputs, false), st
end

# An input as the expansion reads it: a list of (log weight, (mean, cov, flat basis)).
function _expansion_options(b, n, dim)
    b isa LenticulumCore.MixtureBelief || return [(0.0, _split_moments(b, dim))]
    all(c -> c isa GaussianBelief && isproper(c), b.components) || throw(ArgumentError(
        "a MixtureReduction() node expands mixture inputs into proper Gaussian components; the input on :$n \
         has a component that is not one"))
    return [(b.logw[j], (belief_mean(c), belief_cov(c), nothing)) for (j, c) in enumerate(b.components)]
end

# The residual metric of `_residual_metric` and the log normaliser it implies, up to a constant
# shared by every combination: log det R plus, with flat directions G, the log pseudo-determinant
# of GᵀR⁻¹G (the finite part of log det(R + τGΣ₀Gᵀ) as τ → ∞; `switching.md` §9).
function _residual_metric_logdet(R, G, what)
    C = _chol(R, what)
    Li = inv(C.L)
    ld = logdet(C)
    isempty(G) && return Li, Li, ld
    B = Li * reduce(hcat, G)
    F = LinearAlgebra.svd(B; full = true)
    tol = maximum(size(B)) * eps(Float64) * maximum(F.S; init = 0.0)
    r = count(>(tol), F.S)
    return F.U[:, (r + 1):end]' * Li, Li, ld + 2 * sum(log, F.S[1:r]; init = 0.0)
end

function _expanded_message(A, Q, B, lw, lpk, dims, others, target, inputs, to_switch::Bool)
    K = size(B, 2)
    options = [_expansion_options(_get(inputs, n), n, dims[n]) for n in others]
    lks = Vector{Float64}[]                    # to the outcome: one vector over k per combination
    comps, logws = GaussianBelief[], Float64[]
    nt = to_switch ? 0 : dims[target]
    for combo in Iterators.product(options...)
        lc = sum(first, combo; init = 0.0)
        s, R, G = _accumulate(A, Q, [(n, last(o)) for (n, o) in zip(others, combo)])
        T, Li, ld = _residual_metric_logdet(R, G, "Q + Σ AᵢSᵢAᵢᵀ")
        Z = T * (B .- s)
        l = lw .+ lpk .+ (lc - ld / 2) .- vec(sum(abs2, Z; dims = 1)) ./ 2
        if to_switch
            push!(lks, l); continue
        end
        At = A[target]
        Ta = T * At
        rt = _projected_rank(Ta, Li * At)
        if rt == 0                              # flat in the target: one flat component, the outcomes summed
            any(>(-Inf), l) || continue         # no possible outcome: as `factor_message`, refused below
            lf = _logsumexp(l)
            push!(comps, GaussianBelief(zeros(nt), zeros(nt, nt), true)); push!(logws, lf)
            continue
        end
        proper = rt == nt
        Λ = Ta' * Ta
        CΛ = proper ? _chol(Λ, "AₜᵀR⁻¹Aₜ") : nothing
        for k in 1:K
            l[k] == -Inf && continue
            η = Ta' * Z[:, k]
            # A proper component is normalised: its mass ∫exp(ηᵀx − xᵀΛx/2) moves into the weight.
            lk = proper ? l[k] + LinearAlgebra.dot(η, CΛ \ η) / 2 - logdet(CΛ) / 2 + nt * log(2π) / 2 : l[k]
            push!(comps, GaussianBelief(η, Λ, !proper)); push!(logws, lk)
        end
    end
    if to_switch
        L = reduce(hcat, lks)
        lk = [any(>(-Inf), view(L, k, :)) ? _logsumexp(L[k, :]) : -Inf for k in 1:K]   # an outcome of weight 0: −Inf, not NaN
        all(==(-Inf), lk) && throw(ArgumentError("every outcome has zero probability"))
        return LenticulumCore.CategoricalBelief(Val(:log), lk .- _logsumexp(lk), nothing)
    end
    isempty(comps) && throw(ArgumentError("the incoming outcome belief gives every outcome zero probability"))
    all(c -> all(iszero, c.Λ), comps) && return LenticulumCore.TrivialBelief()
    length(comps) == 1 && return only(comps)
    return LenticulumCore.MixtureBelief(Val(:log), comps, logws .- _logsumexp(logws))
end

# --- Message blocks (a plate of tied switching factors, L-ADR-0010) ---------------------

"""
    Mycelium.factor_messages(fs::AbstractVector{<:SwitchingConstraintFactor}, target, polarity, inputs::NamedTuple, priors, ps, st)

The messages of tied replicas from message blocks. To the outcome, when the replicas share the
noise covariance and each continuous input block is points, trivial, or proper Gaussians with
one shared precision (as `LinearConstraintFactor`'s block method), the metric is factorised once
and the result is one `CategoricalBlock`: ``\\log P_i(k) = \\log w_k - \\tfrac12 \\lVert T(b_k - s_i)\\rVert^2`` up to
normalisation, with ``s_i = \\sum_n A_n m_{n,i}``. A mixture message to a continuous channel, and
any other input, goes replica by replica through `factor_message`.
"""
function Mycelium.factor_messages(fs::AbstractVector{<:SwitchingConstraintFactor{names}}, target::Symbol, polarity,
                                  inputs::NamedTuple, priors::AbstractVector, ps, st) where {names}
    f, n = first(fs), length(fs)
    if target !== f.switch && _same_switching(fs)
        out = _switch_mixture_block(f, target, inputs, n, ps)
        out === nothing || return out, st
    end
    parts = target === f.switch ? map(c -> (c, _block_parts(_get(inputs, c), f.dims[c], n)), names) : nothing
    if parts === nothing || any(p -> p[2] === nothing, parts) || !_same_switching(fs)
        cols = [NamedTuple{keys(inputs)}(map(b -> Mycelium.replica(b, i), values(inputs))) for i in 1:n]
        return Mycelium.belief_block(first(Mycelium.factor_messages(fs, target, polarity, cols, priors, ps, st))), st
    end
    _, R, G = _accumulate(ps.A, f.Q, [(ch, (M[:, 1], S, N)) for (ch, (M, S, N)) in parts])
    T, _ = _residual_metric(R, G, "Q + Σ AᵢSᵢAᵢᵀ")
    # s_i = Σₙ Aₙ mₙ,ᵢ as columns, then ‖T b_k − T s_i‖² for every (k, i)
    S = foldl((acc, (ch, (M, _, _))) -> acc .+ ps.A[ch] * M, parts; init = zeros(f.m, n))
    TB, TS = T * float.(ps.b), T * S
    lw = _logweights(ps)
    L = zeros(f.K, n)
    for k in 1:f.K                          # the difference first: no cancellation of ‖Tb‖² against ‖Ts‖²
        L[k, :] .= lw[k] .- vec(sum(abs2, TS .- TB[:, k]; dims = 1)) ./ 2
    end
    m = maximum(L; dims = 1)
    L = L .- m
    return Mycelium.CategoricalBlock(L .- log.(sum(exp.(L); dims = 1))), st
end

"""
    FrozenSwitchingBeliefs

The factor beliefs of `n` tied switching replicas held fixed: the point channels as `d × n`
matrices, the free channels' joint means per outcome and replica (`M[:, k, i]`), one covariance
`Σ` and conditional entropy ``H(x \\mid k)`` shared by every replica, and the outcome posteriors as
the columns of `q`. Built from message blocks whose Gaussian inputs share one precision; the
batched counterpart of the `FrozenGaussianBelief` that `factor_belief` returns.
"""
struct FrozenSwitchingBeliefs{A<:AbstractArray,S<:AbstractMatrix,H<:Real,Q<:AbstractMatrix}
    points::Vector{Pair{Symbol,Matrix{Float64}}}
    free::Vector{Symbol}
    M::A
    Σ::S
    H::H
    q::Q
end

_same_switching(fs) = (f = first(fs); all(h -> h === f || (isequal(h.Q, f.Q) && h.dims == f.dims && h.K == f.K && h.switch === f.switch), fs))

# log P(k) of every replica from the outcome's block: flat without a message, or a categorical block.
_switch_logp_block(::Union{Nothing,Mycelium.TrivialBlock}, K, n) = zeros(K, n)
_switch_logp_block(b::Mycelium.CategoricalBlock, K, n) = size(b.logp, 1) == K ? Matrix{Float64}(b.logp) : nothing
_switch_logp_block(b, K, n) = nothing

# R0[:, k, i] = Σ_points Aₙ xₙ,ᵢ − b_k, the residual with the free channels at zero.
function _switch_offsets(A, B, points, n)
    # no `foldl` with `init`: Zygote's pullback fails on an empty list of point channels
    S = isempty(points) ? zeros(eltype(B), size(B, 1), n) : reduce(+, [A[c] * X for (c, X) in points])
    return reshape(S, size(S, 1), 1, n) .- reshape(B, size(B, 1), size(B, 2), 1)
end

"""
    Mycelium.factor_beliefs(fs::AbstractVector{<:SwitchingConstraintFactor}, msgs::NamedTuple, ps, st)
    Mycelium.belief_free_energies(fs::AbstractVector{<:SwitchingConstraintFactor}, b::FrozenSwitchingBeliefs, ps, st)
    Mycelium.local_free_energies(fs::AbstractVector{<:SwitchingConstraintFactor}, msgs::NamedTuple, ps, st)

The free energy of a plate row of tied switching factors on message blocks (L-ADR-0010): every
continuous channel a point block, or a Gaussian block with one shared precision, trivial or
absent, and the outcome a categorical block, trivial or absent. Then the factor belief's
precision, covariance and ``H(x \\mid k)`` are computed once, and each replica costs its means
and its outcome posterior ``q_i``, as [`switch_statistics`](@ref) computes them one replica at a
time. `factor_beliefs` returns `nothing` otherwise, and the energies go replica by replica.
"""
function Mycelium.factor_beliefs(fs::AbstractVector{<:SwitchingConstraintFactor{names}}, msgs::NamedTuple, ps, st) where {names}
    f, n = first(fs), length(fs)
    _same_switching(fs) || return nothing
    lp = _switch_logp_block(_get(msgs, f.switch), f.K, n)
    lp === nothing && return nothing
    points, free = Pair{Symbol,Matrix{Float64}}[], Symbol[]
    for c in names
        b = _get(msgs, c)
        if b isa Mycelium.PointBlock
            push!(points, c => Matrix{Float64}(b.X))
        elseif b === nothing || b isa Mycelium.TrivialBlock || (b isa Mycelium.GaussianBlock && b.Λ isa AbstractMatrix)
            push!(free, c)
        else
            return nothing
        end
    end
    CQ = _chol(f.Q, "Q")
    R0 = _switch_offsets(ps.A, float.(ps.b), points, n)           # m × K × n
    C = reshape(-R0, f.m, f.K * n)                                 # c_{k,i} = b_k − Σ_points Aₙxₙ,ᵢ
    QiC = CQ \ C
    logZ = -vec(sum(C .* QiC; dims = 1)) ./ 2
    if isempty(free)
        M, Σ, H = zeros(0, f.K, n), zeros(0, 0), 0.0
    else
        G = reduce(hcat, (ps.A[c] for c in free))
        canon = map(free) do c
            b = _get(msgs, c)
            b isa Mycelium.GaussianBlock ? (Matrix{Float64}(b.η), b.Λ) : (zeros(f.dims[c], n), zeros(f.dims[c], f.dims[c]))
        end
        Λ = G' * (CQ \ G) + _blockdiag(map(last, canon))
        CΛ = _chol(Λ, "factor belief precision")
        Ecanon = reduce(vcat, map(first, canon))                   # p × n
        η = G' * QiC .+ repeat(Ecanon; inner = (1, f.K))           # column (k, i): η_{k,i}
        Mf = CΛ \ η
        logZ = logZ .+ vec(sum(η .* Mf; dims = 1)) ./ 2
        M, Σ = reshape(Mf, size(Λ, 1), f.K, n), Matrix(inv(CΛ))
        H = Mycelium.variable_entropy(GaussianBelief(zeros(size(Λ, 1)), Λ))
    end
    L = lp .+ _logweights(ps) .+ reshape(logZ, f.K, n)
    all(i -> any(>(-Inf), view(L, :, i)), 1:n) ||
        throw(ArgumentError("the incoming outcome belief gives every outcome zero probability"))
    L = L .- maximum(L; dims = 1)
    q = exp.(L); q = q ./ sum(q; dims = 1)
    return FrozenSwitchingBeliefs(points, free, M, Σ, H, q)
end

function Mycelium.belief_free_energies(fs::AbstractVector{<:SwitchingConstraintFactor}, b::FrozenSwitchingBeliefs, ps, st)
    f, n = first(fs), size(b.q, 2)
    R = _switch_offsets(ps.A, ps.b, b.points, n)                   # m × K × n at the current parameters
    if isempty(b.free)
        Cr = zeros(eltype(R), f.m, f.m)
    else
        G = reduce(hcat, (ps.A[c] for c in b.free))
        R = R .+ reshape(G * reshape(b.M, size(b.M, 1), :), f.m, f.K, n)
        Cr = G * b.Σ * G'
    end
    CQ = _chol(f.Q, "Q")
    Rf = reshape(R, f.m, :)
    quad = reshape(vec(sum(Rf .* (CQ \ Rf); dims = 1)), f.K, n)     # rₖᵢᵀQ⁻¹rₖᵢ
    lw = _logweights(ps)
    fit = (tr(CQ \ Cr) .+ vec(sum(b.q .* quad; dims = 1))) ./ 2
    complexity = (f.m * log(2π) + logdet(CQ)) / 2 .- vec(sum(b.q .* ifelse.(b.q .> 0, lw, 0.0); dims = 1))
    Hq = -vec(sum(ifelse.(b.q .> 0, b.q .* log.(max.(b.q, floatmin(Float64))), 0.0); dims = 1))
    return [LenticulumCore.GradedEnergy((fit = fit[i], complexity = complexity[i], negentropy = -Hq[i] - b.H)) for i in 1:n], st
end

function Mycelium.local_free_energies(fs::AbstractVector{<:SwitchingConstraintFactor}, msgs::NamedTuple, ps, st)
    b = Mycelium.factor_beliefs(fs, msgs, ps, st)
    b isa FrozenSwitchingBeliefs ||
        return invoke(Mycelium.local_free_energies, Tuple{AbstractVector,NamedTuple,Any,Any}, fs, msgs, ps, st)
    return Mycelium.belief_free_energies(fs, b, ps, st)
end

# The mixture messages of tied replicas to a continuous channel as one `MixtureBlock`, or
# `nothing` for the per-replica path: inputs as for the outcome's block method, the outcome's input
# trivial, absent or a categorical block, and every outcome possible in every replica (an
# impossible one is dropped from that replica's mixture, so the replicas would differ in K), with
# K > 1 (one outcome gives a Gaussian). Component k of replica i: η = (TAₜ)ᵀT(b_k − s_i), one
# precision (TAₜ)ᵀ(TAₜ), the weights as in `_switch_message`.
function _switch_mixture_block(f::SwitchingConstraintFactor{names}, target, inputs, n, ps) where {names}
    f.K > 1 || return nothing
    lpk = _switch_logp_block(_get(inputs, f.switch), f.K, n)
    lpk === nothing && return nothing
    others = [c for c in names if c !== target]
    parts = [(c, _block_parts(_get(inputs, c), f.dims[c], n)) for c in others]
    any(p -> p[2] === nothing, parts) && return nothing
    _, R, G = _accumulate(ps.A, f.Q, [(ch, (M[:, 1], S, N)) for (ch, (M, S, N)) in parts])
    T, Li = _residual_metric(R, G, "Q + Σ AᵢSᵢAᵢᵀ")
    At = ps.A[target]
    Ta = T * At
    rt = _projected_rank(Ta, Li * At)
    rt == 0 && return Mycelium.TrivialBlock(n)
    S = isempty(parts) ? zeros(f.m, n) : reduce(+, [ps.A[ch] * M for (ch, (M, _, _)) in parts])
    TB, TS = T * float.(ps.b), T * S
    Z = reshape(TB, size(TB, 1), f.K, 1) .- reshape(TS, size(TS, 1), 1, n)        # T(b_k − s_i)
    L = lpk .+ _logweights(ps) .- dropdims(sum(abs2, Z; dims = 1); dims = 1) ./ 2   # K × n
    all(isfinite, L) || return nothing
    η = reshape(Ta' * reshape(Z, size(Z, 1), :), size(At, 2), f.K, n)
    proper = rt == size(At, 2)
    Λ = Ta' * Ta
    if proper                                              # normalised components: ½ηᵀΛ⁻¹η into the weight
        CΛ = _chol(Λ, "AₜᵀR⁻¹Aₜ")
        Ef = reshape(η, size(η, 1), :)
        L = L .+ reshape(vec(sum(Ef .* (CΛ \ Ef); dims = 1)) ./ 2, f.K, n)
    end
    L = L .- maximum(L; dims = 1)
    L = L .- log.(sum(exp.(L); dims = 1))
    return Mycelium.MixtureBlock(η, Λ, L, !proper)
end

# --- Cluster nodes (Extending Mycelium §5, L-SPEC-009) ---------------------------

Mycelium.cluster_switch(f::SwitchingConstraintFactor) = f.switch
Mycelium.cluster_form(f::SwitchingConstraintFactor, ps) = Mycelium.ClusterForm(ps.A, ps.b, f.Q, _logweights(ps), f.switch)
Mycelium.cluster_form(f::LinearConstraintFactor, ps) = Mycelium.ClusterForm(ps.A, reshape(ps.c, :, 1), f.Q, [0.0], nothing)
function Mycelium.cluster_form(f::GaussianPrior, ps)
    f.noise_learning === nothing || throw(ArgumentError("a GaussianPrior with learnable noise cannot be put in a cluster node"))
    n = length(f.μ)
    return Mycelium.ClusterForm(NamedTuple{(f.channel,)}((Matrix(1.0I, n, n),)), reshape(float.(f.μ), n, 1), Matrix(f.Σ), [0.0], nothing)
end

"""
    switch_overlaps(g, ps, sum::Symbol, switches) -> Matrix

The pairwise overlaps of switching factors feeding one linear factor `sum`, for grouping them
into cluster nodes (`Mycelium.overlap_groups`, E7's rule). Switch ``l`` sets its channel ``u_l``
(``A_{u,l} u_l = b_{l,k} + \\varepsilon_l``, ``A_{u,l}`` square and invertible), which enters `sum` as
``A_{\\mathrm{sum},l}\\, u_l``; its effect on the sum's residual is the matrix of directions
``V_l = A_{\\mathrm{sum},l} A_{u,l}^{-1} (b_{l,k} - b_{l,j})_{j < k}`` (every pair of outcomes). In the metric of the noise of the sum and the listed switches (other inputs of `sum` are not counted)
``R = Q_{\\mathrm{sum}} + \\sum_l A_{\\mathrm{sum},l} A_{u,l}^{-1} Q_l A_{u,l}^{-\\top} A_{\\mathrm{sum},l}^\\top``, the overlap of two switches is the
largest ``\\lvert\\cos\\rvert`` between a direction of one and a direction of the other; the diagonal is 1. For
binary switches feeding ``z = \\sum_l W_l k_l + \\varepsilon`` this is E7's ``\\lvert G_{lm}\\rvert / \\sqrt{G_{ll} G_{mm}}``, ``G = W^\\top R^{-1} W``.
"""
function switch_overlaps(g::Mycelium.FactorGraph, ps, sum::Symbol, switches)
    snode = Mycelium.factornode(g, g.factor_index[sum])
    snode.factor isa LinearConstraintFactor || throw(ArgumentError("switch_overlaps: :$sum is not a LinearConstraintFactor"))
    psum = Mycelium._subtree(ps, snode.params)
    R = Matrix{Float64}(snode.factor.Q)
    Vs = Matrix{Float64}[]
    for name in switches
        node = Mycelium.factornode(g, g.factor_index[name])
        f = node.factor
        f isa SwitchingConstraintFactor && length(f.dims) == 1 || throw(ArgumentError(
            "switch_overlaps: :$name must be a SwitchingConstraintFactor with one continuous channel"))
        p = Mycelium._subtree(ps, node.params)
        uch = only(keys(f.dims))
        e = only(ei for ei in g.edges_of_factor[node.id] if g.edges[ei].channel === uch)
        v = g.edges[e].variable
        es = [ei for ei in g.edges_of_factor[snode.id] if g.edges[ei].variable == v]
        length(es) == 1 || throw(ArgumentError("switch_overlaps: :$name's channel does not feed :$sum once"))
        As = psum.A[g.edges[only(es)].channel]
        Au = p.A[uch]
        size(Au, 1) == size(Au, 2) && LinearAlgebra.rank(Au) == size(Au, 1) || throw(ArgumentError(
            "switch_overlaps: :$name's coefficient must be square and invertible"))
        M = As / Au
        K = size(p.b, 2)
        push!(Vs, M * reduce(hcat, [p.b[:, k] .- p.b[:, j] for j in 1:K for k in (j + 1):K]))   # every pair of outcomes
        R = R .+ M * f.Q * M'
    end
    C = cholesky(Symmetric(R))
    Ws = [C.L \ V for V in Vs]                            # directions whitened by R
    L = length(Ws)
    O = Matrix{Float64}(I, L, L)
    for l in 1:L, m in (l + 1):L
        o = 0.0
        for a in eachcol(Ws[l]), b in eachcol(Ws[m])
            na, nb = LinearAlgebra.norm(a), LinearAlgebra.norm(b)
            na > 0 && nb > 0 && (o = max(o, abs(LinearAlgebra.dot(a, b)) / (na * nb)))
        end
        O[l, m] = O[m, l] = o
    end
    return O
end
