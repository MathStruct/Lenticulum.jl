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
