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
with ``G`` stacking those channels' ``A``'s, the same limit for every ``\\Sigma_0 \\succ 0``. An
improper Gaussian input splits into a proper part on the range of its precision, which enters
``d`` and ``R``, and a flat part on its null space ``N``, which adds ``A_i N`` to ``G``. The
equation then still constrains the directions ``G`` cannot reach. The message is
`TrivialBelief()` when nothing is left, as when ``G`` has full row rank, and is declared improper
(`GaussianBelief(η, Λ, true)`) whenever the (projected) ``A_t`` is rank-deficient
(`constraint.md` §4.1). The flat case is not differentiable by ForwardDiff: the projection
needs an SVD, which has no method for dual numbers, and it fails rather than give a derivative
that ignores how the flat directions move.
"""
Mycelium.factor_message(
    f::LinearConstraintFactor, target::Symbol, polarity, inputs, prior, ps, st
) = _message(f, target, inputs, ps, st)

function _message(f::LinearConstraintFactor{names}, target::Symbol, inputs, ps, st) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this LinearConstraintFactor (has $names)"))
    parts = [(n, _split_moments(_get(inputs, n), f.dims[n])) for n in names if n !== target]
    return _linear_message(ps.A, float.(ps.c), f.Q, target, parts, st)
end

# The message to `target` from Σᵢ Aᵢxᵢ = c + ε, the other channels given as `parts`.
function _linear_message(A, c, Q, target, parts, st)
    s, R, G = _accumulate(A, Q, parts)
    d = c .- s
    At = A[target]
    if isempty(G)
        Ri = inv(_chol(R, "Q + Σ AᵢSᵢAᵢᵀ"))
        improper = _projected_rank(At, At) < size(At, 2)          # rank(L⁻¹Aₜ) = rank(Aₜ)
        return (GaussianBelief(At' * (Ri * d), At' * Ri * At, improper), st)
    end
    T, Li = _residual_metric(R, G, "Q + Σ AᵢSᵢAᵢᵀ")
    Ta = T * At
    rt = _projected_rank(Ta, Li * At)
    rt == 0 && return (LenticulumCore.TrivialBelief(), st)
    return (GaussianBelief(Ta' * (T * d), Ta' * Ta, rt < size(At, 2)), st)
end

"""
    Mycelium.factor_messages(fs::AbstractVector{<:LinearConstraintFactor}, target, polarity, inputs, priors, ps, st)

The messages of tied replicas (a plate). When the replicas have one noise covariance and every
replica's inputs have the same covariances and flat directions (points, or Gaussians of one
covariance, as with data and a shared prior), the metric, its factorisation and the rank test
depend on the parameters only and are computed once; each replica then costs one
matrix-vector product. The messages are those of `factor_message`, term for term. Otherwise each
replica is computed on its own.
"""
function Mycelium.factor_messages(fs::AbstractVector{<:LinearConstraintFactor{names}}, target::Symbol, polarity,
                                  inputs::AbstractVector, priors::AbstractVector, ps, st) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this LinearConstraintFactor (has $names)"))
    f = first(fs)
    others = _other_channels(Val(names), Val(target))
    partss = [map(n -> (n, _split_moments(_get(inp, n), h.dims[n])), others) for (h, inp) in zip(fs, inputs)]
    p1 = first(partss)
    same(a, b) = isequal(a[2][2], b[2][2]) && isequal(a[2][3], b[2][3])
    c = float.(ps.c)
    if !all(h -> h === f || (isequal(h.Q, f.Q) && h.dims == f.dims), fs) ||
       !all(p -> all(splat(same), zip(p, p1)), partss)
        beliefs = Vector{Any}(undef, length(partss))
        for (i, p) in enumerate(partss)
            beliefs[i], st = _linear_message(ps.A, c, fs[i].Q, target, p, st)
        end
        return beliefs, st
    end
    A, At = ps.A, ps.A[target]
    _, R, G = _accumulate(A, f.Q, p1)
    if isempty(G)
        Ri = inv(_chol(R, "Q + Σ AᵢSᵢAᵢᵀ"))
        improper = _projected_rank(At, At) < size(At, 2)
        K, Λ = At' * Ri, At' * Ri * At
    else
        T, Li = _residual_metric(R, G, "Q + Σ AᵢSᵢAᵢᵀ")
        Ta = T * At
        rt = _projected_rank(Ta, Li * At)
        rt == 0 && return Any[LenticulumCore.TrivialBelief() for _ in partss], st
        K, Λ, improper = Ta' * T, Ta' * Ta, rt < size(At, 2)
    end
    # every replica at once: H = K (c 1ᵀ − Σₙ Aₙ Mₙ), with Mₙ the stacked input means; one precision
    # shared by all the messages (beliefs are not mutated in place)
    Mc = repeat(c, 1, length(partss))
    for (k, (n, _)) in enumerate(p1)
        Mc = Mc .- A[n] * reduce(hcat, [p[k][2][1] for p in partss])
    end
    H = K * Mc
    return Any[GaussianBelief(H[:, i], Λ, improper) for i in axes(H, 2)], st
end

# The channels other than the target, as a tuple known to the compiler.
@generated _other_channels(::Val{names}, ::Val{target}) where {names,target} =
    Tuple(n for n in names if n !== target)

# The non-target channels, each as (n, (m, S, N)): a Gaussian part with mean m and covariance S
# (`nothing`: a point), flat along the columns of N (`nothing`: nowhere). Returns Σ Aₙmₙ,
# R = Q + Σ AₙSₙAₙᵀ, and the coefficients AₙNₙ of the flat directions.
function _accumulate(A, Q, parts)
    s = zeros(size(Q, 1))
    R = float.(Q)
    G = Any[]
    for (n, (m, S, N)) in parts
        s = s .+ A[n] * m
        S === nothing || (R = R .+ A[n] * S * A[n]')
        N === nothing || push!(G, A[n] * N)
    end
    return s, R, G
end

# The residual metric with the flat directions integrated out, as a factor T with TᵀT the metric.
# For R = LLᵀ and G the stacked coefficients of the flat directions, a prior N(0, τΣ₀) along them
# gives (R + τGΣ₀Gᵀ)⁻¹ → R⁻¹ − R⁻¹G(GᵀR⁻¹G)⁺GᵀR⁻¹ as τ → ∞, for every Σ₀ ≻ 0: the R⁻¹-orthogonal
# projection that removes range(G). So T = U⊥ᵀL⁻¹, with U⊥ an orthonormal basis of the
# complement of range(L⁻¹G); it has no rows when G has full row rank. The normaliser that
# diverges with τ is the same for every value of the target, and for every outcome of a
# switching factor. Also returns L⁻¹, the scale for `_projected_rank`.
function _residual_metric(R, G, what)
    Li = inv(_chol(R, what).L)
    isempty(G) && return Li, Li
    B = Li * reduce(hcat, G)
    F = LinearAlgebra.svd(B; full = true)
    tol = maximum(size(B)) * eps(Float64) * maximum(F.S; init = 0.0)
    r = count(>(tol), F.S)
    return F.U[:, (r + 1):end]' * Li, Li
end

# The rank of the projected coefficient T Aₜ. Rounding in the projection leaves singular values
# of order eps·‖L⁻¹Aₜ‖ in the directions it removed, so those count as zero. A rank has no
# derivative, so it is computed on values (`stop_gradient` strips ForwardDiff duals).
function _projected_rank(Ta, LiAt)
    s = LinearAlgebra.svdvals(LenticulumCore.stop_gradient(Ta))
    isempty(s) && return 0
    scale = maximum(LinearAlgebra.svdvals(LenticulumCore.stop_gradient(LiAt)))
    return count(>(maximum(size(LiAt)) * eps(Float64) * scale), s)
end

"""
    Mycelium.mean_field_message(f::LinearConstraintFactor, target, polarity, beliefs, ps, st)

The message of a `MeanField()` node: ``\\exp \\mathbb E_q[\\log f]`` as a function of ``x_t``, over the other
channels' current marginals ``q_i``. ``\\log f`` is quadratic, so only their means enter, and the
message is ``\\mathcal N^{-1}(A_t^\\top Q^{-1} \\bar d, A_t^\\top Q^{-1} A_t)`` with ``\\bar d = c - \\sum_{i \\ne t} A_i\\, \\mathbb E_q[x_i]``:
the belief-propagation message with every input replaced by a point mass at its mean. A mixture
marginal enters by its mean. A channel without a marginal yet is integrated out under a flat
prior, as in `factor_message`, and so is an improper marginal along the directions where it is
flat; its mean elsewhere enters as usual.
"""
function Mycelium.mean_field_message(f::LinearConstraintFactor{names}, target::Symbol, polarity, beliefs, ps, st) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this LinearConstraintFactor (has $names)"))
    parts = [(n, _mf_split(_get(beliefs, n), f.dims[n])) for n in names if n !== target]
    return _linear_message(ps.A, float.(ps.c), f.Q, target, parts, st)
end

# A marginal as the mean-field rule reads it: (mean, nothing, N), flat along the columns of N.
_mf_split(b, n) = (zeros(n), nothing, Matrix(1.0LinearAlgebra.I, n, n))
_mf_split(b::LenticulumCore.DiracBelief, n) = (_vec(b.value), nothing, nothing)
_mf_split(b::GaussianBelief, n) = ((m, _, N) = _split_moments(b, n); (m, nothing, N))
_mf_split(b::LenticulumCore.MixtureBelief{<:GaussianBelief}, n) =
    all(isproper, b.components) ? (belief_mean(moment_match(b)), nothing, nothing) : _mf_split(nothing, n)

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

# --- The mean-field node's average energy (Riegler et al. 2013, eq. 17) --------------------

# The product of the marginals as a frozen belief with no entropy. A mean-field node's term is
# the factor free energy at it with the entropy left out. Point marginals are substituted, as in
# `_affine_gaussian_belief`; a mixture is collapsed to its moments, which is exact here because
# the energy is quadratic in each channel. `K` columns of the mean for a switching factor.
# Built from the marginals alone, so it is held fixed under differentiation (the ChainRulesCore
# extension marks it non-differentiable).
function _product_belief(names, dims, marginals; K = nothing, q = nothing)
    points = Pair{Symbol,Vector{Float64}}[]
    free = Symbol[]
    means = Vector{Float64}[]
    covs = Matrix{Float64}[]
    for n in names
        b = _collapse(_get(marginals, n))
        if b isa LenticulumCore.DiracBelief
            push!(points, n => _vec(b.value))
        elseif b isa GaussianBelief && isproper(b)
            length(b.η) == dims[n] || throw(DimensionMismatch(
                "the marginal on :$n has dimension $(length(b.η)), the channel $(dims[n])"))
            push!(free, n)
            push!(means, LenticulumCore.belief_mean(b))
            push!(covs, Matrix(LenticulumCore.belief_cov(b)))
        else
            throw(ArgumentError(
                "a MeanField() node's energy needs a point or a proper Gaussian marginal on :$n, got \
                 $(nameof(typeof(b))); run inference until every variable has one"))
        end
    end
    cols = K === nothing ? 1 : K
    m = isempty(free) ? zeros(0) : reduce(vcat, means)
    M = K === nothing ? m : repeat(m, 1, cols)
    Σ = isempty(free) ? zeros(0, 0) : _blockdiag(covs)
    return FrozenGaussianBelief(points, free, isempty(free) && K !== nothing ? zeros(0, cols) : M, Σ, 0.0, q)
end

# A mean-field node contributes an average energy and no entropy of its own.
_without_entropy(F) = LenticulumCore.GradedEnergy(
    (fit = F[:fit], complexity = F[:complexity], negentropy = zero(F[:fit])))

"""
    Mycelium.mean_field_energy(f::LinearConstraintFactor, marginals, ps, st)

The average energy of a `MeanField()` node, ``\\mathbb E_{\\prod_i q_i}[-\\log f]`` with
``r = \\sum_i A_i x_i - c`` under independent marginals: `fit` is
``\\tfrac12(\\bar r^\\top Q^{-1}\\bar r + \\operatorname{tr}(Q^{-1}\\sum_i A_i \\Sigma_i A_i^\\top))``,
`complexity` the normaliser, and `negentropy` zero, since the variables' entropies are counted
at the variables. Marginals must be points or proper Gaussians (mixtures by their moments, which
is exact for this quadratic energy).
"""
function Mycelium.mean_field_energy(f::LinearConstraintFactor{names}, marginals, ps, st) where {names}
    b = _product_belief(names, f.dims, marginals)
    F, st = Mycelium.belief_free_energy(f, b, ps, st)
    return _without_entropy(F), st
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
