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
observed (or, for the outcome, latent). With `K = 1` it is a `LinearConstraintFactor` with
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

One polarity per channel, the outcome included: that channel `Unobserved()`, every other
`Observed()`.
"""
function LenticulumCore.supported_polarities(f::SwitchingConstraintFactor)
    all = _allnames(f)
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    return map(t -> LenticulumCore.Polarity(NamedTuple{all}(map(n -> n === t ? U : O, all))), all)
end

"""
    LenticulumCore.supports_polarity(f::SwitchingConstraintFactor, p)

Any polarity over exactly this channel set with exactly one `Unobserved()` channel. A
`Latent()` outcome is marginalised under the factor's own weights; a `Latent()` continuous
channel gives the uninformative answer of [`LinearConstraintFactor`](@ref).
"""
function LenticulumCore.supports_polarity(f::SwitchingConstraintFactor, p::LenticulumCore.Polarity)
    keys(p) == _allnames(f) || return false
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
    k = _vec(vals[f.switch]) |> only |> Int
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
    k = Int(only(_vec(vals[f.switch])))
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
  when ``A_t`` is not full column rank, and are then weighted as the unnormalised exponentials
  that `Mycelium`'s mixture product expects. A single possible outcome gives a `GaussianBelief`.

A Gaussian-mixture input is collapsed to its moments first, as in `LinearConstraintFactor`
(assumed-density filtering; `gaussian.md`, "Mixture inputs"). An uninformative continuous input
makes a continuous target uninformative; for the outcome it gives the factor's own weights when
that channel's `A` has full row rank (the integral is then the same for every outcome), and
`TrivialBelief()` otherwise.
"""
Mycelium.factor_message(f::SwitchingConstraintFactor, target::Symbol, polarity, inputs, prior, ps, st) =
    _switch_message(f, target, inputs, ps, st)

function _switch_message(f::SwitchingConstraintFactor{names}, target::Symbol, inputs, ps, st) where {names}
    target in _allnames(f) || throw(ArgumentError(
        "channel :$target is not a channel of this SwitchingConstraintFactor (has $(_allnames(f)))"))
    lw = _logweights(ps)
    d0 = zeros(f.m)
    R = float.(f.Q)
    for n in names
        n === target && continue
        m, S = _moments(_get(inputs, n), f.dims[n])
        if m === nothing
            target === f.switch && LinearAlgebra.rank(ps.A[n]) == f.m &&
                return (LenticulumCore.CategoricalBelief(Val(:log), lw, nothing), st)
            return (LenticulumCore.TrivialBelief(), st)
        end
        d0 = d0 .- ps.A[n] * m
        R = R .+ ps.A[n] * S * ps.A[n]'
    end
    C = _chol(R, "Q + Σ AᵢSᵢAᵢᵀ")
    D = ps.b .+ d0                                            # column k is d_k
    quad = [LinearAlgebra.dot(D[:, k], C \ D[:, k]) for k in 1:f.K]
    if target === f.switch
        l = lw .- quad ./ 2
        return (LenticulumCore.CategoricalBelief(Val(:log), l .- _logsumexp(l), nothing), st)
    end
    l = _switch_logp(_get(inputs, f.switch), f.K) .+ lw .- quad ./ 2
    At = ps.A[target]
    Λ = At' * (C \ At)
    H = C \ D
    proper = LinearAlgebra.rank(At) == size(At, 2)
    proper || (Λ = _singular_precision(Λ, size(At, 2) - LinearAlgebra.rank(At)))
    ks = [k for k in 1:f.K if l[k] > -Inf]
    isempty(ks) && throw(ArgumentError("the incoming outcome belief gives every outcome zero probability"))
    comps = [GaussianBelief(At' * H[:, k], Λ) for k in ks]
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
the factor's weights, which is where the iteration starts. Continuous channels without a proper
marginal are handled as in `factor_message`.
"""
function Mycelium.mean_field_message(f::SwitchingConstraintFactor{names}, target::Symbol, polarity, beliefs, ps, st) where {names}
    target in _allnames(f) || throw(ArgumentError(
        "channel :$target is not a channel of this SwitchingConstraintFactor (has $(_allnames(f)))"))
    lw = _logweights(ps)
    sbar = zeros(f.m)
    for n in names
        n === target && continue
        μ = _mf_mean(_get(beliefs, n))
        if μ === nothing
            target === f.switch && LinearAlgebra.rank(ps.A[n]) == f.m &&
                return (LenticulumCore.CategoricalBelief(Val(:log), lw, nothing), st)
            return (LenticulumCore.TrivialBelief(), st)
        end
        sbar = sbar .+ ps.A[n] * μ
    end
    CQ = _chol(f.Q, "Q")
    if target === f.switch
        D = ps.b .- sbar
        l = lw .- [LinearAlgebra.dot(D[:, k], CQ \ D[:, k]) for k in 1:f.K] ./ 2
        return (LenticulumCore.CategoricalBelief(Val(:log), l .- _logsumexp(l), nothing), st)
    end
    kb = _get(beliefs, f.switch)
    q = kb === nothing || kb isa LenticulumCore.TrivialBelief ? exp.(lw) : exp.(_switch_logp(kb, f.K))
    At = ps.A[target]
    return (GaussianBelief(At' * (CQ \ (ps.b * q .- sbar)), At' * (CQ \ At)), st)
end

# A rank-deficient AₜᵀR⁻¹Aₜ is positive definite after rounding often enough that `isproper`
# would call it a (very wide) distribution, and the mixture product would then weight its
# components by ½ηᵀΛ⁻¹η, which is rounding noise. Its null-space eigenvalues are set to zero,
# and to −eps·λ_max if the result still factorises, so that the components stay improper.
function _singular_precision(Λ, nullity)
    E = LinearAlgebra.eigen(Symmetric(Λ))
    λ = copy(E.values)
    λ[1:nullity] .= 0
    S = E.vectors * Diagonal(λ) * E.vectors'
    S = (S + S') / 2
    if isposdef(S)
        λ[1:nullity] .= -eps(maximum(abs, λ))
        S = E.vectors * Diagonal(λ) * E.vectors'
        S = (S + S') / 2
    end
    return S
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
        Ηs = G' * QiC .+ reduce(vcat, map(first, canon))      # column k: η_k
        M = CΛ \ Ηs
        logZ = logZ .+ [LinearAlgebra.dot(Ηs[:, k], M[:, k]) / 2 for k in 1:f.K]
        rbar = G * M .- Ceff
        Cr = G * (CΛ \ G')
        Hx = Mycelium.variable_entropy(GaussianBelief(zeros(size(Λ, 1)), Λ))
    end
    l = _switch_logp(_get(msgs, f.switch), f.K) .+ _logweights(ps) .+ logZ
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
    CQ = _chol(f.Q, "Q")
    lw = _logweights(ps)
    trQC = tr(CQ \ Cr)
    fit = sum(q[k] * (LinearAlgebra.dot(rbar[:, k], CQ \ rbar[:, k]) + trQC) / 2 for k in 1:f.K if q[k] > 0)
    complexity = logdet(_chol(2π .* f.Q, "2πQ")) / 2 - sum(q[k] * lw[k] for k in 1:f.K if q[k] > 0)
    Hq = -sum(qk * log(qk) for qk in q if qk > 0)
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -Hq - Hx)), st
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
