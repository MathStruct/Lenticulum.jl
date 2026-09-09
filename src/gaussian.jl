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
#   * learnable parameters (A, b).
#
# Its purpose is to be a test oracle: everything it computes has a closed form, so the
# framework's claims can be checked against arithmetic rather than against itself.
#
# See `gaussian.md` and `The Gaussian Factor.md`.
# ---------------------------------------------------------------------------

"""
    GaussianFactor(nin => nout; noise = 0.1, channels = (:x, :y), init_weight, init_bias)

The linear-Gaussian conditional ``p(y \\mid x) = \\mathcal{N}(y;\\ Ax + b,\\ Q)``.

`noise` is `Q`: a matrix, a vector of variances, or a scalar variance. It is a **fixed
hyperparameter**, not a parameter — learning `Q` needs a positive-definite parametrisation,
which is a separate concern (see `gaussian.md` §3).

Parameters are `(A, b)`, produced by `initialparameters` exactly as a Lux layer's are:
the factor is a *description*, `ps` is an *inhabitant*.

**Bidirectional**: both `(x observed, y unobserved)` and `(x unobserved, y observed)` are
supported, and each is exact. That is what makes it a relation rather than a function, and it
is the smallest non-trivial witness that `assemble` — the operation with no Lux counterpart —
does real work.
"""
struct GaussianFactor{Q<:AbstractMatrix,W,B} <: LenticulumCore.AbstractLenticulumFactor
    nin::Int
    nout::Int
    in_channel::Symbol
    out_channel::Symbol
    Q::Q
    init_weight::W
    init_bias::B
end

function GaussianFactor(
    dims::Pair{Int,Int};
    noise = 0.1,
    channels::Tuple{Symbol,Symbol} = (:x, :y),
    init_weight = (rng, o, i) -> randn(rng, o, i) ./ sqrt(i),
    init_bias = (rng, o) -> zeros(o),
)
    nin, nout = dims
    Q = _as_cov(noise, nout)
    isposdef(Symmetric(Q)) || throw(ArgumentError("noise covariance Q must be positive definite"))
    return GaussianFactor(nin, nout, channels[1], channels[2], Q, init_weight, init_bias)
end

_as_cov(q::AbstractMatrix, n) = Matrix(float.(q))
_as_cov(q::AbstractVector, n) = Matrix(Diagonal(float.(q)))
_as_cov(q::Real, n) = Matrix(float(q) * I, n, n)

LuxCore.initialparameters(rng::AbstractRNG, f::GaussianFactor) =
    (A = f.init_weight(rng, f.nout, f.nin), b = f.init_bias(rng, f.nout))
LuxCore.initialstates(::AbstractRNG, ::GaussianFactor) = NamedTuple()
LuxCore.parameterlength(f::GaussianFactor) = f.nout * f.nin + f.nout

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
| `complexity` | ``\\tfrac12\\log\\det(2\\pi Q)`` | the log-normaliser; what stops ``Q \\to 0`` |
| `negentropy` | ``-H(b_c)`` | the entropy term of the free energy |

The grading is not decoration: it is the fit/complexity/entropy decomposition every Gaussian
model has, made visible for free by ``E_G = \\bigoplus_f E_f``. Summing it recovers the exact
free energy, so the scalarisation is `IdentityScalarisation` on each summand and is therefore
**linear** — hence composition is *strict*, per `Scalar and Multivariate Energy.md` §5.

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
function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:LinearGaussianModel,LenticulumCore.ExactInversion},
    π, inputs, ps, st,
)
    f = lens.model.factor
    target = only(LenticulumCore.unobserved_channels(lens.model.polarity))
    msg, st = _message(f, target, inputs, ps, st)
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
        Σ = A * S * A' + f.Q
        Λ = inv(_chol(Σ, "ASAᵀ + Q"))
        return (GaussianBelief(Λ * (A * m .+ b), Λ), st)
    elseif target === f.in_channel
        src = _get(inputs, f.out_channel)
        m, S = _moments(src, f.nout)
        m === nothing && return (LenticulumCore.TrivialBelief(), st)
        R = f.Q + S
        Ri = inv(_chol(R, "Q + Sᵧ"))
        return (GaussianBelief(A' * (Ri * (m .- b)), A' * Ri * A), st)
    end
    throw(ArgumentError("channel :$target is not a channel of this GaussianFactor"))
end

_get(inputs, k) = haskey(inputs, k) ? getfield(inputs, k) : nothing

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
        Λ = A' * (f.Q \ A) + Λx
        η = A' * (f.Q \ (y0 .- b)) + ηx
        bel = GaussianBelief(η, Λ)
        m, Σ = belief_mean(bel), belief_cov(bel)
        return (y0 .- A * m .- b), A * Σ * A', Mycelium.variable_entropy(bel)
    elseif xd
        ηy, Λy = _canon(my, f.nout)
        x0 = _vec(mx.value)
        ŷ = A * x0 .+ b
        Λ = inv(_chol(f.Q, "Q")) + Λy
        η = (f.Q \ ŷ) + ηy
        bel = GaussianBelief(η, Λ)
        m, Σ = belief_mean(bel), belief_cov(bel)
        return (m .- ŷ), Σ, Mycelium.variable_entropy(bel)
    else
        ηx, Λx = _canon(mx, f.nin)
        ηy, Λy = _canon(my, f.nout)
        G = hcat(-A, Matrix{Float64}(I, f.nout, f.nout))
        Qi = inv(_chol(f.Q, "Q"))
        Λ = G' * Qi * G + _blockdiag(Λx, Λy)
        η = G' * (Qi * b) + vcat(ηx, ηy)
        bel = GaussianBelief(η, Λ)
        m, Σ = belief_mean(bel), belief_cov(bel)
        return (G * m .- b), G * Σ * G', Mycelium.variable_entropy(bel)
    end
end

_canon(b::GaussianBelief, n) = (b.η, b.Λ)
_canon(::LenticulumCore.TrivialBelief, n) = (zeros(n), zeros(n, n))
_canon(::Nothing, n) = (zeros(n), zeros(n, n))

function _blockdiag(A1, A2)
    n1, n2 = size(A1, 1), size(A2, 1)
    M = zeros(n1 + n2, n1 + n2)
    M[1:n1, 1:n1] .= A1
    M[(n1 + 1):end, (n1 + 1):end] .= A2
    return M
end

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
    Qi = inv(_chol(f.Q, "Q"))
    fit = (r̄' * (Qi * r̄) + tr(Qi * Cr)) / 2
    complexity = logdet(_chol(2π .* f.Q, "2πQ")) / 2
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -H)), st
end

# =========================================================================== #
# A Gaussian prior — AutoBayes Remark 24, exactly
# =========================================================================== #

"""
    GaussianPrior(channel, μ, Σ)

The prior factor ``\\pi : 1 \\multimap X`` of AutoBayes Remark 24: emits
``\\mathcal{N}(\\mu,\\Sigma)``, charges energy ``-\\log p_\\pi(x)``, and has zero entropy of
its own.

Without it the composite free energy is an **open** free energy, missing the
``-\\log p_\\pi(x)`` term. With it, ``F^{c\\pi}(\\ast, y) = \\mathrm{VFE}(c,c')(\\pi,y)``, and
when the inversion is exact that equals ``-\\log p_{c_*\\pi}(y)`` on the nose. That identity
is the project's main correctness test.

Non-learnable in v0: a learnable prior wants its natural parameters in `ps`, and `Σ` needs a
positive-definite parametrisation — the same deferred concern as `Q` in
[`GaussianFactor`](@ref).
"""
struct GaussianPrior{V<:AbstractVector,M<:AbstractMatrix} <: LenticulumCore.AbstractLenticulumFactor
    channel::Symbol
    μ::V
    Σ::M
end
GaussianPrior(channel::Symbol, μ::Real, σ²::Real) =
    GaussianPrior(channel, [float(μ)], fill(float(σ²), 1, 1))

LenticulumCore.channels(f::GaussianPrior) = (LenticulumCore.Channel(f.channel, length(f.μ)),)
LenticulumCore.supported_polarities(f::GaussianPrior) =
    (LenticulumCore.Polarity(NamedTuple{(f.channel,)}((LenticulumCore.Unobserved(),))),)
LenticulumCore.supports_polarity(f::GaussianPrior, p::LenticulumCore.Polarity) =
    keys(p) == (f.channel,) && p[f.channel] isa LenticulumCore.Unobserved
LenticulumCore.islearnable(::GaussianPrior) = false
LenticulumCore.energyspace(::GaussianPrior) = LenticulumCore.energyspace(GaussianFactor(1 => 1))
LenticulumCore.scalarisation(::GaussianPrior) = LenticulumCore.GradedScalarisation((
    fit = LenticulumCore.IdentityScalarisation(),
    complexity = LenticulumCore.IdentityScalarisation(),
    negentropy = LenticulumCore.IdentityScalarisation(),
))

Mycelium.factor_message(f::GaussianPrior, ::Symbol, _, _, _, ps, st) =
    (Gaussian(f.μ, f.Σ), st)

function Mycelium.local_free_energy(f::GaussianPrior, msgs, ps, st)
    Λ0 = inv(_chol(f.Σ, "Σ"))
    incoming = _get(msgs, f.channel)
    complexity = logdet(_chol(2π .* f.Σ, "2πΣ")) / 2
    if incoming isa LenticulumCore.DiracBelief
        d = _vec(incoming.value) .- f.μ
        fit = (d' * (Λ0 * d)) / 2
        return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = 0.0)), st
    end
    ηi, Λi = _canon(incoming, length(f.μ))
    bel = GaussianBelief(Λ0 * f.μ + ηi, Λ0 + Λi)
    m, Σb = belief_mean(bel), belief_cov(bel)
    d = m .- f.μ
    fit = (d' * (Λ0 * d) + tr(Λ0 * Σb)) / 2
    H = Mycelium.variable_entropy(bel)
    return LenticulumCore.GradedEnergy((fit = fit, complexity = complexity, negentropy = -H)), st
end
