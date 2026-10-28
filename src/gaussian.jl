# ---------------------------------------------------------------------------
# The linear-Gaussian factor:  p(y | x) = N(y; Ax + b, Q)
#
# The workhorse conditional — Kalman filters, linear regression, factor analysis and GP
# regression are all built from it — and the first factor in this project that implements the
# WHOLE statistical-game interface with nothing stubbed:
#
#   * a real open model (the kernel),
#   * an EXACT Bayesian inversion (conjugacy),
#   * a genuine vector energy and entropy,
#   * bidirectional polarity (it is a relation, not a function),
#   * learnable parameters (A, b), with opt-in noise covariance.
#
# Its purpose is to be a test oracle: everything it computes has a closed form, so the
# framework's claims can be checked against arithmetic rather than against itself.
#
# See `gaussian.md` and `Scalar and Multivariate Energy.md`.
# ---------------------------------------------------------------------------

"""
    GaussianFactor(nin => nout; noise = 0.1, learn_noise = false,
                   noise_prior = nothing, eigenvalue_floor = 1e-6,
                   channels = (:x, :y), init_weight, init_bias)

The linear-Gaussian conditional ``p(y \\mid x) = \\mathcal{N}(y;\\ Ax + b,\\ Q)``.

`noise` is the initial covariance `Q`: a matrix, a vector of variances, or a scalar
variance. By default it is fixed and parameters are `(A, b)`. With `learn_noise=true`
(or `:full`), `ps.noise` stores a precision Cholesky's `log_diagonal` and packed strict
`lower` entries; `:diagonal` stores only `log_precision`. The initial covariance must
be diagonal in diagonal mode. Use [`noise_covariance`](@ref) to read the current value.

Train noise with `free_energy` / `local_free_energy`, which include the Gaussian
normaliser. The residual `energy` and its quadratic `scalar_energy` omit it: minimising
those alone inflates noise without bound. An optional [`WishartPrecisionPrior`](@ref)
(full) or [`LogNormalVariancePrior`](@ref) (diagonal) adds its negative log density to
free energy. Otherwise covariance eigenvalues are clipped at `eigenvalue_floor`.
The hard floor is continuous with a kink; its derivative is zero in saturated directions.
See `noise.md` and `gaussian.md` for density coordinates and prior multiplicity.

Parameters are produced by `initialparameters` exactly as a Lux layer's are:
the factor is a *description*, `ps` is an *inhabitant*.

**Bidirectional**: both `(x observed, y unobserved)` and `(x unobserved, y observed)` are
supported, and each is exact. That is what makes it a relation rather than a function, and it
is the smallest non-trivial witness that `assemble` — the operation with no Lux counterpart —
does real work.
"""
struct GaussianFactor{Q<:AbstractMatrix,W,B,N} <: LenticulumCore.AbstractLenticulumFactor
    nin::Int
    nout::Int
    in_channel::Symbol
    out_channel::Symbol
    Q::Q
    init_weight::W
    init_bias::B
    noise_learning::N
end

# Preserve the fixed-noise constructor used by nonlinear Gaussian local models.
GaussianFactor(nin::Int, nout::Int, input::Symbol, output::Symbol, Q::AbstractMatrix, iw, ib) =
    GaussianFactor(nin, nout, input, output, Q, iw, ib, nothing)

function GaussianFactor(
    dims::Pair{Int,Int};
    noise = 0.1,
    learn_noise = false,
    noise_prior = nothing,
    eigenvalue_floor = 1e-6,
    channels::Tuple{Symbol,Symbol} = (:x, :y),
    init_weight = (rng, o, i) -> randn(rng, o, i) ./ sqrt(i),
    init_bias = (rng, o) -> zeros(o),
)
    nin, nout = dims
    Q = _as_cov(noise, nout)
    size(Q) == (nout, nout) || throw(DimensionMismatch("noise must have size ($nout, $nout)"))
    issymmetric(Q) || throw(ArgumentError("noise covariance must be symmetric"))
    isposdef(Symmetric(Q)) || throw(ArgumentError("noise covariance Q must be positive definite"))
    learning = _noise_learning(Q, learn_noise, noise_prior, eigenvalue_floor)
    return GaussianFactor(nin, nout, channels[1], channels[2], Q, init_weight, init_bias, learning)
end

_as_cov(q::AbstractMatrix, n) = Matrix(float.(q))
_as_cov(q::AbstractVector, n) = Matrix(Diagonal(float.(q)))
_as_cov(q::Real, n) = Matrix(float(q) * I, n, n)

function LuxCore.initialparameters(rng::AbstractRNG, f::GaussianFactor)
    base = (A = f.init_weight(rng, f.nout, f.nin), b = f.init_bias(rng, f.nout))
    return merge(base, _noise_parameters(f.noise_learning, f.Q))
end
LuxCore.initialstates(::AbstractRNG, ::GaussianFactor) = NamedTuple()
LuxCore.parameterlength(f::GaussianFactor) = f.nout * f.nin + f.nout + _noise_parameterlength(f.noise_learning, f.nout)

LenticulumCore.channels(f::GaussianFactor) = (
    LenticulumCore.Channel(f.in_channel, f.nin),
    LenticulumCore.Channel(f.out_channel, f.nout),
)
function LenticulumCore.supported_polarities(f::GaussianFactor)
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    ks = (f.in_channel, f.out_channel)
    return (LenticulumCore.Polarity(NamedTuple{ks}((O, U))),
            LenticulumCore.Polarity(NamedTuple{ks}((U, O))))
end
function LenticulumCore.supports_polarity(f::GaussianFactor, p::LenticulumCore.Polarity)
    keys(p) == (f.in_channel, f.out_channel) || return false
    pi_, po = p[f.in_channel], p[f.out_channel]
    return (pi_ isa LenticulumCore.Observed && po isa LenticulumCore.Unobserved) ||
           (pi_ isa LenticulumCore.Unobserved && po isa LenticulumCore.Observed)
end
LenticulumCore.islearnable(::GaussianFactor) = true

"""
    energyspace(f::GaussianFactor)

``E_c = \\mathbb{R}^3``, graded as `(fit, complexity, negentropy)`:

| summand | is | role |
|---|---|---|
| `fit` | ``\\mathbb{E}_{b_c}[\\tfrac12 r^\\top Q^{-1} r]`` | how badly the residual misses |
| `complexity` | ``\\tfrac12\\log\\det(2\\pi Q)`` | the log-normaliser; penalises variance inflation |
| `negentropy` | ``-H(b_c)`` | the entropy term of the free energy |

The grading is not decoration: it is the fit/complexity/entropy decomposition every Gaussian
model has, made visible for free by ``E_G = \\bigoplus_f E_f``. Summing it recovers the exact
free energy, so the scalarisation is `IdentityScalarisation` on each summand and is therefore
**linear** — hence composition is *strict*, per `Scalar and Multivariate Energy.md` §5.

This is the space of the **free energy**. The pointwise `energy`
is the residual, in its own space: see `pointwise_energyspace` below.

The alternative grading — keep the residual ``r`` as a vector with
`SquaredNorm(Q⁻¹)` — is **lax**, and by exactly ``\\tfrac12\\operatorname{tr}(Q^{-1}\\operatorname{Cov}(r))``.
[`residual_statistics`](@ref) exposes both sides of that identity; see `gaussian.md` §2.
"""
LenticulumCore.energyspace(::GaussianFactor) = LenticulumCore.GradedEnergySpace((
    fit = LenticulumCore.ScalarEnergySpace(),
    complexity = LenticulumCore.ScalarEnergySpace(),
    negentropy = LenticulumCore.ScalarEnergySpace(),
))
LenticulumCore.scalarisation(::GaussianFactor) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(),
    complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation(),
))

"""
    residual(f, x, y, ps) -> Vector

``r = y - Ax - b``: the **pointwise vector energy** of
`Scalar and Multivariate Energy.md`, in the residual space where the observation noise lives
(`Algebraic versus Geometric Distance.md` §"what the energy space is").

`Q⁻¹` is the noise precision, hence the natural scalarisation's metric — derived, not chosen.
"""
residual(f::GaussianFactor, x, y, ps) = y .- ps.A * x .- ps.b

LenticulumCore.energy(f::GaussianFactor, x, a, y, ps, st) = (residual(f, x, y, ps), st)

"""
    pointwise_energyspace(f::GaussianFactor), pointwise_scalarisation(f::GaussianFactor)

The pointwise energy is the residual ``r \\in \\mathbb{R}^{n_{out}}``, signed, scalarised by
`SquaredNorm(Q⁻¹)`: ``\\tfrac12 r^\\top Q^{-1} r``, zero at a perfect fit. Its expectation under
the factor belief is the `fit` summand of the free energy (the graded
`energyspace` above); the `complexity`
``\\tfrac12\\log\\det(2\\pi Q)`` is the normaliser it leaves out, a constant while `Q` is fixed.
The residual is the same in both polarities, so neither depends on one.
"""
LenticulumCore.pointwise_energyspace(f::GaussianFactor) = LenticulumCore.EuclideanEnergySpace(f.nout)
LenticulumCore.pointwise_scalarisation(f::GaussianFactor) =
    f.noise_learning === nothing ? LenticulumCore.SquaredNorm(inv(Symmetric(f.Q)); check = false) :
    LearnedPrecisionNorm()

# --- The open model and its exact inversion --------------------------------

"""
    LinearGaussianModel(factor, polarity)

The open model ``c : X \\nrightarrow Y`` of AutoBayes Definition 1, in the direction the
polarity selects. Pure (``\\llbracket c \\rrbracket \\cong 1``): a linear-Gaussian conditional
hides nothing.
"""
struct LinearGaussianModel{P} <: LenticulumCore.AbstractOpenModel
    factor::GaussianFactor
    polarity::P
end
LenticulumCore.ispure(::LinearGaussianModel) = true
LenticulumCore.latentspace(::LinearGaussianModel) = nothing

LenticulumCore.assemble(f::GaussianFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(LinearGaussianModel(f, p), LenticulumCore.ExactInversion()), st)

"""
    LenticulumCore.invert(lens, π, inputs, ps, st) -> (belief, st)

The **exact** Bayesian inversion ``c^\\dagger_\\pi(y)`` of AutoBayes Definition 10 — the
posterior, *including* the prior.

> [!warning] This is not the belief-propagation message
> ``c^\\dagger_\\pi(y)`` is the posterior; a BP message is the **likelihood**, with the prior
> divided out. If a factor sent the posterior, a variable of degree ``d`` would count the prior
> ``d`` times. In canonical form the two differ by one subtraction, and
> [`Mycelium.factor_message`](@ref) returns the likelihood while this returns the posterior.
>
> The relation between them is exactly `combine(π, message) == posterior`, which is asserted in
> the test suite. On a chain (every variable of degree 2) the distinction is invisible, which is
> why it does not appear in the paper. See `gaussian.md` §4.
"""
function LenticulumCore.likelihood_message(
    lens::LenticulumCore.BayesianLens{<:LinearGaussianModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    target = only(LenticulumCore.unobserved_channels(lens.model.polarity))
    return _message(f, target, inputs, ps, st)
end

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:LinearGaussianModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    msg, st = LenticulumCore.likelihood_message(lens, π, inputs, ps, st)
    return Mycelium.combine(π, msg), st
end

# --- Messages (BP semantics: likelihoods, not posteriors) ------------------

"""
    Mycelium.factor_message(f::GaussianFactor, target, polarity, inputs, prior, ps, st)

The factor → variable message: the **likelihood** contribution, in canonical form.

Forward (``x`` observed, target ``y``), for an incoming ``\\mathcal{N}(m, S)`` on ``x``:

```math
\\mu_{c\\to y} = \\mathcal{N}\\bigl(Am + b,\\ ASA^\\top + Q\\bigr)
```

Backward (``y`` observed, target ``x``), for an incoming ``\\mathcal{N}(m_y, S_y)`` on ``y``,
writing ``R = Q + S_y``:

```math
\\mu_{c\\to x} = \\mathcal{N}^{-1}\\bigl(A^\\top R^{-1}(m_y - b),\\ A^\\top R^{-1}A\\bigr)
```

Note the backward message is generally **improper** — ``A^\\top R^{-1} A`` is rank-deficient
whenever `A` is not full column rank, because a likelihood constrains only the directions the
model can see. The moment form cannot express this; the canonical form does, which is the
second reason `beliefs.jl` uses it.

The `prior` argument is *deliberately unused*: including it would double-count. See
[`LenticulumCore.invert`](@ref) above.
"""
Mycelium.factor_message(f::GaussianFactor, target::Symbol, polarity, inputs, prior, ps, st) =
    _message(f, target, inputs, ps, st)

function _message(f::GaussianFactor, target::Symbol, inputs, ps, st)
    A, b = ps.A, ps.b
    if target === f.out_channel
        src = _get(inputs, f.in_channel)
        m, S = _moments(src, f.nin)
        m === nothing && return (LenticulumCore.TrivialBelief(), st)
        Σ = A * S * A' + noise_covariance(f, ps)
        Λ = inv(_chol(Σ, "ASAᵀ + Q"))
        return (GaussianBelief(Λ * (A * m .+ b), Λ), st)
    elseif target === f.in_channel
        src = _get(inputs, f.out_channel)
        m, S = _moments(src, f.nout)
        m === nothing && return (LenticulumCore.TrivialBelief(), st)
        R = noise_covariance(f, ps) + S
        Ri = inv(_chol(R, "Q + Sᵧ"))
        return (GaussianBelief(A' * (Ri * (m .- b)), A' * Ri * A), st)
    end
    throw(ArgumentError("channel :$target is not a channel of this GaussianFactor"))
end

_get(inputs, k) = haskey(inputs, k) ? getfield(inputs, k) : nothing

# Functional schedule AD avoids the mutating inverse-of-Cholesky implementation.
# Ordinary message passing retains its established arithmetic.
function Mycelium._gradient_message(ad, f::GaussianFactor, target::Symbol, pol, inputs, prior, ps, st)
    if target === f.out_channel
        m,S=_moments(_get(inputs,f.in_channel),f.nin)
        m === nothing && return LenticulumCore.TrivialBelief(),st
        V=ps.A*S*ps.A'+noise_covariance(f,ps)
        Λ=inv(Symmetric(V))
        return GaussianBelief(Λ*(ps.A*m+ps.b),Λ),st
    elseif target === f.in_channel
        m,S=_moments(_get(inputs,f.out_channel),f.nout)
        m === nothing && return LenticulumCore.TrivialBelief(),st
        Ri=inv(Symmetric(noise_covariance(f,ps)+S))
        return GaussianBelief(ps.A'*(Ri*(m-ps.b)),ps.A'*Ri*ps.A),st
    end
    throw(ArgumentError("unknown Gaussian target :$target"))
end

# (mean, covariance) of an incoming message; `nothing` means "carries no information".
_moments(b::LenticulumCore.DiracBelief, n) = (_vec(b.value), zeros(n, n))
_moments(::LenticulumCore.TrivialBelief, n) = (nothing, nothing)
_moments(::Nothing, n) = (nothing, nothing)
function _moments(b::GaussianBelief, n)
    all(iszero, b.Λ) && return (nothing, nothing)
    return (belief_mean(b), belief_cov(b))
end
_vec(v::AbstractVector) = float.(v)
_vec(v::Real) = [float(v)]

# --- The Bethe contribution ------------------------------------------------

"""
    residual_statistics(f, msgs, ps) -> (r̄, Cov_r)

The mean and covariance of the residual ``r = y - Ax - b`` under the factor's own belief
``b_c \\propto f_c \\prod_i \\mu_{i\\to c}``.

These are what make the **two energies** comparable on this factor:

```math
\\underbrace{\\mathbb{E}\\bigl[\\tfrac12 r^\\top Q^{-1} r\\bigr]}_{\\text{scalar (exact)}}
\\;-\\;
\\underbrace{\\tfrac12 \\bar r^\\top Q^{-1}\\bar r}_{\\text{multivariate (lax)}}
\\;=\\;
\\tfrac12\\operatorname{tr}\\bigl(Q^{-1}\\operatorname{Cov}(r)\\bigr)
```

which is precisely the Jensen gap ``\\tfrac12\\operatorname{tr}\\operatorname{Cov}`` of
`Scalar and Multivariate Energy.md` §5, here in closed form on a real model. Asserted in the
test suite.
"""
function residual_statistics(f::GaussianFactor, msgs, ps)
    Q = noise_covariance(f, ps)
    A, b = ps.A, ps.b
    mx = _get(msgs, f.in_channel)
    my = _get(msgs, f.out_channel)
    xd, yd = mx isa LenticulumCore.DiracBelief, my isa LenticulumCore.DiracBelief
    if xd && yd
        r = _vec(my.value) .- A * _vec(mx.value) .- b
        return r, zeros(f.nout, f.nout), 0.0
    elseif yd
        ηx, Λx = _canon(mx, f.nin)
        y0 = _vec(my.value)
        Λ = A' * (Q \ A) + Λx
        η = A' * (Q \ (y0 .- b)) + ηx
        m, Σ, H = _posterior_statistics(f.noise_learning, η, Λ)
        return (y0 .- A * m .- b), A * Σ * A', H
    elseif xd
        ηy, Λy = _canon(my, f.nout)
        x0 = _vec(mx.value)
        ŷ = A * x0 .+ b
        Λ = inv(Symmetric(Q)) + Λy
        η = (Q \ ŷ) + ηy
        m, Σ, H = _posterior_statistics(f.noise_learning, η, Λ)
        return (m .- ŷ), Σ, H
    else
        ηx, Λx = _canon(mx, f.nin)
        ηy, Λy = _canon(my, f.nout)
        G = hcat(-A, Matrix{Float64}(I, f.nout, f.nout))
        Qi = inv(Symmetric(Q))
        Λ = G' * Qi * G + _joint_precision(f.noise_learning, Λx, Λy)
        η = G' * (Qi * b) + vcat(ηx, ηy)
        m, Σ, H = _posterior_statistics(f.noise_learning, η, Λ)
        return (G * m .- b), G * Σ * G', H
    end
end

_canon(b::GaussianBelief, n) = (b.η, b.Λ)
_canon(::LenticulumCore.TrivialBelief, n) = (zeros(n), zeros(n, n))
_canon(::Nothing, n) = (zeros(n), zeros(n, n))

function _blockdiag(A1, A2)
    n1, n2 = size(A1, 1), size(A2, 1)
    T = promote_type(eltype(A1), eltype(A2))
    return vcat(hcat(A1, zeros(T,n1,n2)),hcat(zeros(T,n2,n1),A2))
end

# Retain fixed-noise arithmetic, while avoiding inverse-of-Cholesky mutation in AD.
function _posterior_statistics(::Nothing, η, Λ)
    bel = GaussianBelief(η, Λ)
    return belief_mean(bel), belief_cov(bel), Mycelium.variable_entropy(bel)
end
function _posterior_statistics(::NoiseLearning, η, Λ)
    C = _chol(Λ, "factor posterior precision")
    Σ = inv(Matrix(Λ))
    return C \ η, Σ, (length(η) * (1 + log(2π)) - logdet(Matrix(Λ))) / 2
end
_joint_precision(::Nothing, A, B) = _blockdiag(A, B)
_joint_precision(::NoiseLearning, A, B) =
    vcat(hcat(A, zeros(eltype(A), size(A, 1), size(B, 2))),
         hcat(zeros(eltype(B), size(B, 1), size(A, 2)), B))

Mycelium.uses_cavity(::GaussianFactor) = true

"""
    Mycelium.local_free_energy(f::GaussianFactor, msgs, ps, st)

The factor's Bethe contribution ``F_c = U_c - H(b_c)``, as a graded energy
`(fit, complexity, negentropy)`, where

```math
b_c(x,y) \\;\\propto\\; \\mathcal{N}(y;\\,Ax+b,\\,Q)\\ \\mu_{x\\to c}(x)\\ \\mu_{y\\to c}(y)
```

Note `msgs` are the **incoming messages** ``\\mu_{i\\to c}``, not the variable marginals. That
is what the Bethe formula asks for, and getting it wrong is the bug recorded in
`free_energy.md` §7.
"""
function Mycelium.local_free_energy(f::GaussianFactor, msgs, ps, st)
    r̄, Cr, H = residual_statistics(f, msgs, ps)
    Qi, logdetQ = _noise_metrics(f.noise_learning, f.Q, ps)
    fit = (r̄' * (Qi * r̄) + tr(Qi * Cr)) / 2
    complexity = f.noise_learning === nothing ? logdet(_chol(2π .* f.Q, "2πQ")) / 2 :
        (f.nout * log(2π) + logdetQ) / 2 + _noise_penalty(f.noise_learning.prior, Qi, logdetQ)
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -H)), st
end

function _game_messages(f, π, y)
    by = y isa NamedTuple ? _get(y, f.out_channel) :
         y isa LenticulumCore.AbstractBelief ? y :
         y === nothing ? nothing : LenticulumCore.DiracBelief(y)
    return NamedTuple{(f.in_channel, f.out_channel)}((π, by))
end

function LenticulumCore.free_energy(f::GaussianFactor, π, y, ps, st)
    return Mycelium.local_free_energy(f, _game_messages(f, π, y), ps, st)
end

function LenticulumCore.entropy(f::GaussianFactor, π, y, ps, st)
    F, st = LenticulumCore.free_energy(f, π, y, ps, st)
    return LenticulumCore.GradedEnergy((fit = 0.0, complexity = 0.0, negentropy = -F[:negentropy])), st
end

# =========================================================================== #
# A Gaussian prior — AutoBayes Remark 24, exactly
# =========================================================================== #

"""
    GaussianPrior(channel, μ, Σ; learn_noise = false, noise_prior = nothing, eigenvalue_floor = 1e-6)

The prior factor ``\\pi : 1 \\multimap X`` of AutoBayes Remark 24: emits
``\\mathcal{N}(\\mu,\\Sigma)`` and charges energy ``-\\log p_\\pi(x)``. Its game-level entropy is
the entropy of the factor belief after pooling with incoming information (zero for a point
observation).

Without it the composite free energy is an **open** free energy, missing the
``-\\log p_\\pi(x)`` term. With it, ``F^{c\\pi}(\\ast, y) = \\mathrm{VFE}(c,c')(\\pi,y)``, and
when the inversion is exact that equals ``-\\log p_{c_*\\pi}(y)`` on the nose. That identity
is the project's main correctness test.

The mean `μ` stays fixed. Covariance learning is opt-in with the same `learn_noise`,
`noise_prior` and `eigenvalue_floor` options as [`GaussianFactor`](@ref). Its existing
pointwise energy is the normalised negative log-density, including the normaliser for
both fixed and learned covariance. A noise hyperprior is charged only in free energy.
"""
struct GaussianPrior{V<:AbstractVector,M<:AbstractMatrix,N} <: LenticulumCore.AbstractLenticulumFactor
    channel::Symbol
    μ::V
    Σ::M
    noise_learning::N
end
function GaussianPrior(channel::Symbol, μ::AbstractVector, Σ::AbstractMatrix;
                       learn_noise = false, noise_prior = nothing, eigenvalue_floor = 1e-6)
    if learn_noise === false
        noise_prior === nothing || throw(ArgumentError("noise_prior requires learn_noise"))
        return GaussianPrior(channel, μ, Σ, nothing)
    end
    Q = _as_cov(Σ, length(μ))
    size(Q) == (length(μ), length(μ)) || throw(DimensionMismatch("prior covariance size"))
    issymmetric(Q) && isposdef(Symmetric(Q)) || throw(ArgumentError("Σ must be symmetric positive definite"))
    return GaussianPrior(channel, float.(μ), Q, _noise_learning(Q, learn_noise, noise_prior, eigenvalue_floor))
end
GaussianPrior(channel::Symbol, μ::Real, σ²::Real; kwargs...) =
    GaussianPrior(channel, [float(μ)], fill(float(σ²), 1, 1); kwargs...)

LenticulumCore.channels(f::GaussianPrior) = (LenticulumCore.Channel(f.channel, length(f.μ)),)
LenticulumCore.supported_polarities(f::GaussianPrior) =
    (LenticulumCore.Polarity(NamedTuple{(f.channel,)}((LenticulumCore.Unobserved(),))),)
LenticulumCore.supports_polarity(f::GaussianPrior, p::LenticulumCore.Polarity) =
    keys(p) == (f.channel,) && p[f.channel] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(f::GaussianPrior) = f.noise_learning !== nothing
LenticulumCore.energyspace(::GaussianPrior) = LenticulumCore.energyspace(GaussianFactor(1 => 1))
LenticulumCore.scalarisation(::GaussianPrior) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(),
    complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation(),
))

LuxCore.initialparameters(::AbstractRNG, f::GaussianPrior) = _noise_parameters(f.noise_learning, f.Σ)
LuxCore.parameterlength(f::GaussianPrior) = _noise_parameterlength(f.noise_learning, length(f.μ))

# The source factor's pointwise energy includes its normalizer, in the same grading as F.
function LenticulumCore.energy(f::GaussianPrior, x, a, y, ps, st)
    d = _vec(y) .- f.μ
    if f.noise_learning === nothing
        C = _chol(f.Σ, "Σ")
        fit = (d' * (C \ d)) / 2
        complexity = (length(d) * log(2π) + logdet(C)) / 2
    else
        P, logdetQ = _noise_metrics(f.noise_learning, f.Σ, ps)
        fit = (d' * P * d) / 2
        complexity = (length(d) * log(2π) + logdetQ) / 2
    end
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = 0.0)), st
end

struct GaussianPriorModel{F<:GaussianPrior} <: LenticulumCore.AbstractOpenModel
    factor::F
end
LenticulumCore.ispure(::GaussianPriorModel) = true
LenticulumCore.latentspace(::GaussianPriorModel) = nothing
LenticulumCore.assemble(f::GaussianPrior, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(GaussianPriorModel(f), LenticulumCore.ExactInversion()), st)

function LenticulumCore.likelihood_message(
    lens::LenticulumCore.BayesianLens{<:GaussianPriorModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    return (Gaussian(f.μ, noise_covariance(f, ps)), st)
end

function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:GaussianPriorModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    msg, st = LenticulumCore.likelihood_message(lens, π, inputs, ps, st)
    return (Mycelium.combine(π, msg), st)
end

Mycelium.factor_message(f::GaussianPrior, ::Symbol, _, _, _, ps, st) =
    (Gaussian(f.μ, noise_covariance(f, ps)), st)
function Mycelium._gradient_message(ad, f::GaussianPrior, ::Symbol, _, _, _, ps, st)
    P=inv(Symmetric(noise_covariance(f,ps)))
    return GaussianBelief(P*f.μ,P),st
end

Mycelium.uses_cavity(::GaussianPrior) = true
_prior_precision(Σ) = inv(_chol(Σ, "Σ"))

function Mycelium.local_free_energy(f::GaussianPrior, msgs, ps, st)
    Λ0, logdetQ = f.noise_learning === nothing ?
        (_prior_precision(f.Σ), zero(eltype(f.Σ))) : _noise_metrics(f.noise_learning, f.Σ, ps)
    incoming = _get(msgs, f.channel)
    complexity = f.noise_learning === nothing ? logdet(_chol(2π .* f.Σ, "2πΣ")) / 2 :
        (length(f.μ) * log(2π) + logdetQ) / 2 + _noise_penalty(f.noise_learning.prior, Λ0, logdetQ)
    if incoming isa LenticulumCore.DiracBelief
        d = _vec(incoming.value) .- f.μ
        fit = (d' * (Λ0 * d)) / 2
        return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = 0.0)), st
    end
    ηi, Λi = _canon(incoming, length(f.μ))
    m, Σb, H = _posterior_statistics(f.noise_learning, Λ0 * f.μ + ηi, Λ0 + Λi)
    d = m .- f.μ
    fit = (d' * (Λ0 * d) + tr(Λ0 * Σb)) / 2
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -H)), st
end

function LenticulumCore.free_energy(f::GaussianPrior, π, y, ps, st)
    by = y isa NamedTuple ? _get(y, f.channel) :
         y isa LenticulumCore.AbstractBelief ? y :
         y === nothing ? π : LenticulumCore.DiracBelief(y)
    return Mycelium.local_free_energy(f, NamedTuple{(f.channel,)}((by,)), ps, st)
end

function LenticulumCore.entropy(f::GaussianPrior, π, y, ps, st)
    F, st = LenticulumCore.free_energy(f, π, y, ps, st)
    return LenticulumCore.GradedEnergy((fit = 0.0, complexity = 0.0, negentropy = -F[:negentropy])), st
end

"""
    noise_covariance(f::Union{GaussianFactor,GaussianPrior}, ps)

Current covariance reconstructed from opt-in precision parameters (including the
eigenvalue floor when no noise prior is supplied). Fixed-noise factors return their
original covariance. The initial covariance on the struct is never updated.
"""
noise_covariance(f::GaussianFactor, ps) = f.noise_learning === nothing ? f.Q :
    inv(Symmetric(first(_noise_metrics(f.noise_learning, f.Q, ps))))
noise_covariance(f::GaussianPrior, ps) = f.noise_learning === nothing ? f.Σ :
    inv(Symmetric(first(_noise_metrics(f.noise_learning, f.Σ, ps))))
