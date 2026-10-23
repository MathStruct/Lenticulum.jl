# ---------------------------------------------------------------------------
# Beyond one point: every stable answer a query has, and how sharply each is determined.
#
#   implicit_roots    runs `implicit_infer` from several starts (the clamped inputs fixed),
#                     keeps the converged, stable answers, removes duplicates. A multivalued
#                     relation then returns all its branches instead of the one whose basin
#                     the start happened to lie in.
#   implicit_laplace  the Laplace approximation at an answer: the inverse of the symmetric
#                     part of the residual's Jacobian on the free coordinates, which the
#                     solver and the adjoint already compute.
#
# See `implicit.md` §5 and `Open Problems in Implicit Diffusion Learning.md` (T1, I2, I4).
# ---------------------------------------------------------------------------

"""
    implicit_roots(m, z₀, ρ, ps, st; nstarts = 16, spread = 1.0, starts = nothing,
                   rng = Xoshiro(0), unique_tol = 1e-3, kwargs...) -> (Vector{ImplicitSolution}, st)

All distinct **stable** answers to a query that a set of starting points reaches. Starts are
`z₀` itself and `nstarts` perturbations of it on the free coordinates (normal, standard
deviation `spread`), or the vectors in `starts`; hard-clamped coordinates stay fixed. Each start
is solved with [`implicit_infer`](@ref) (`kwargs` are passed on); converged, stable solutions
are kept, and two are the same answer if they differ by less than `unique_tol` (relative).

The answers are ordered by the query's energy ``U(z) + \\tfrac12\\lVert P(z - z_0)\\rVert^2`` when the
predictor has an energy, i.e. is an [`EnergyNetwork`](@ref) or a closed-form
[`GaussianMixtureEps`](@ref) (lowest first, i.e. the most plausible answer first),
and otherwise in the order found. There is no guarantee that every stable answer is found;
more starts and a larger `spread` find more.
"""
function implicit_roots(m::AbstractImplicitRelation, z₀, ρ, ps, st; nstarts::Integer = 16, spread = 1.0,
                        starts = nothing, rng::AbstractRNG = Random.Xoshiro(0), unique_tol = 1e-3, kwargs...)
    free = .!isinf.(ρ)
    inits = if starts === nothing
        perturbed() = (z = float.(copy(z₀)); z[free] .+= spread .* randn(rng, count(free)); z)
        [float.(copy(z₀)), (perturbed() for _ in 1:nstarts)...]
    else
        collect(starts)
    end
    roots = ImplicitSolution[]
    for z_init in inits
        sol, st = implicit_infer(m, z₀, ρ, ps, st; z_init, kwargs...)
        (sol.converged && sol.stable) || continue
        any(r -> norm(r.z .- sol.z) ≤ unique_tol * (1 + norm(sol.z)), roots) && continue
        push!(roots, sol)
    end
    roots = [r for r in roots]                                   # concrete element type
    if _has_energy(m) && length(roots) > 1
        energies = [_query_energy(m, r.z, z₀, ρ, ps, st) for r in roots]
        roots = roots[sortperm(energies)]
    end
    return (roots, st)
end

_has_energy(m) = false
_has_energy(::ImplicitDiffusion{<:NoisePredictor{<:EnergyNetwork}}) = true
_has_energy(::ImplicitDiffusion{<:NoisePredictor{<:GaussianMixtureEps}}) = true

function _query_energy(m, z, z₀, ρ, ps, st)
    U, _ = implicit_energy(m, z, ps, st)
    soft = .!isinf.(ρ)
    return U + sum(abs2, ρ[soft] .* (z[soft] .- z₀[soft])) / 2
end

"""
    density_lambda(schedule, nodes::FieldNodes) -> λ

The weighting ``\\lambda = 1 / \\sum_k w_k\\,\\sigma_k^2/\\alpha_k^2`` under which the implicit learner's prior
term is a weighted **average** of smoothed negative log-densities,
``U(z) \\approx \\sum_k \\bar w_k\\,(-\\log p_{t_k}(\\alpha_k z + \\sigma_k\\varepsilon_k))`` with ``\\sum_k \\bar w_k = 1``: a proper
negative log-density, in the same units as the clamp's Gaussian log-likelihood. With it,
[`implicit_laplace`](@ref) returns covariances in data units. For queries whose coordinates are
all hard inputs or free outputs (``\\rho_i \\in \\{0, \\infty\\}``), λ does not change the answers, only
the curvature; with soft evidence it sets the balance between prior and evidence.
"""
density_lambda(s::AbstractNoiseSchedule, nodes::FieldNodes) =
    1 / sum(nodes.w[k] * sigma(s, t)^2 / alpha(s, t)^2 for (k, t) in enumerate(nodes.t))

"""
    implicit_laplace(m, sol, ρ, ps, st) -> (mean, cov, free)

The Laplace approximation of the query at the answer `sol`: mean `sol.z`, and on the free
coordinates the covariance

```math
\\Sigma_{FF} = \\bigl(\\operatorname{sym} J_{FF}\\bigr)^{-1}, \\qquad J_{FF} = \\partial_{z_F} r_F(z^\\star) = \\partial_{z_F} g_F + \\operatorname{diag}(\\rho_F^2),
```

the inverse curvature of the query's energy (for an energy-parametrised predictor ``J_{FF}`` is
already symmetric; otherwise its symmetric part is used). Hard-clamped coordinates have zero
variance. `free` marks the free coordinates.

**Units.** The energy is the clamp ``\\tfrac12\\lVert P(z - z_0)\\rVert^2`` (a Gaussian log-likelihood with
precision ``\\rho^2``) plus the prior term, a weighted sum of smoothed negative log-densities
scaled by `m.λ`. `cov` is a calibrated posterior covariance when that prior term is a calibrated
negative log-prior. Build the model with `λ = density_lambda(schedule, nodes)` for that: the
covariance is then calibrated to the density *smoothed at the field's noise levels* (wider than
the data distribution by those levels). With another λ it is a relative measure of how sharply
the answer is determined. Throws unless `sol` converged to a stable answer.
"""
function implicit_laplace(m::AbstractImplicitRelation, sol::ImplicitSolution, ρ, ps, st)
    (sol.converged && sol.stable) ||
        throw(ArgumentError("the Laplace approximation needs a converged, stable answer (a local minimum)"))
    free = .!isinf.(ρ)
    J = prior_jacobian(m, sol.z, ps, st; coords = free) .+ Diagonal(ρ[free] .^ 2)
    C = LinearAlgebra.cholesky(Symmetric((J .+ J') ./ 2); check = false)
    LinearAlgebra.issuccess(C) ||
        throw(ArgumentError("the symmetric part of J_FF is not positive definite at the solution"))
    ΣF = inv(C)
    n = length(sol.z)
    Σ = zeros(eltype(ΣF), n, n)
    Σ[free, free] .= ΣF
    return (mean = sol.z, cov = Σ, free = free)
end

# --- Answers as beliefs ----------------------------------------------------------------------

_indices(coords::AbstractVector{Bool}) = findall(coords)
_indices(coords) = collect(coords)

"""
    laplace_belief(m, sol, ρ, ps, st; coords = free, linsolve = DenseSolve()) -> GaussianBelief

The Laplace approximation at the answer `sol` ([`implicit_laplace`](@ref)) as a
`GaussianBelief`, marginalised onto `coords` (indices or a mask; default: all free
coordinates). Hard-clamped coordinates have no variance and cannot be among `coords`.
Calibrated in data units when the model is built with [`density_lambda`](@ref).

With `linsolve = KrylovSolve()` only the marginal is computed, one conjugate-gradient solve per
coordinate in `coords`, so a label's covariance can be read off a query whose image is free.
"""
function laplace_belief(m::AbstractImplicitRelation, sol::ImplicitSolution, ρ, ps, st;
                        coords = .!isinf.(ρ), linsolve::AbstractLinearSolve = DenseSolve())
    idx = _indices(coords)
    Σ, _ = _laplace_marginal(linsolve, m, sol, ρ, ps, st, idx; logdet = false)
    return LenticulumCore.Gaussian(sol.z[idx], Σ)
end

_check_free(free, idx) = all(free[idx]) ||
    throw(ArgumentError("`coords` must be free coordinates; hard-clamped ones have no variance"))

# The Laplace covariance on the coordinates `idx` (all free), and ½ log det Σ_FF of the whole
# free block (the mixture weights need it).
function _laplace_marginal(::DenseSolve, m, sol, ρ, ps, st, idx; logdet::Bool)
    (sol.converged && sol.stable) ||
        throw(ArgumentError("the Laplace approximation needs a converged, stable answer (a local minimum)"))
    free = .!isinf.(ρ)
    _check_free(free, idx)
    J = prior_jacobian(m, sol.z, ps, st; coords = free) .+ Diagonal(ρ[free] .^ 2)
    C = LinearAlgebra.cholesky(Symmetric((J .+ J') ./ 2); check = false)
    LinearAlgebra.issuccess(C) ||
        throw(ArgumentError("the symmetric part of J_FF is not positive definite at the solution"))
    ΣF = inv(C)
    pos = cumsum(free)[idx]
    hld = logdet ? -LinearAlgebra.logdet(C) / 2 : NaN
    return (Matrix(Symmetric(ΣF[pos, pos])), hld)
end

# matrix-free: Σ e_i by conjugate gradients on sym J_FF for each i in idx, log det by stochastic
# Lanczos quadrature (Σ_FF = (sym J_FF)⁻¹, so ½ log det Σ_FF = -½ log det sym J_FF)
function _laplace_marginal(ls::KrylovSolve, m, sol, ρ, ps, st, idx; logdet::Bool)
    (sol.converged && sol.stable) ||
        throw(ArgumentError("the Laplace approximation needs a converged, stable answer (a local minimum)"))
    free = .!isinf.(ρ)
    _check_free(free, idx)
    op = _free_operators(m, sol.z, ρ, ps, st)
    pos = cumsum(free)[idx]                            # positions among the free coordinates
    Σ = zeros(eltype(float(sol.z)), length(idx), length(idx))
    for (a, i) in enumerate(pos)
        e = zeros(eltype(Σ), op.n); e[i] = 1
        x, ok = _cg(op.S, e; rtol = ls.rtol, maxiters = ls.maxiters)
        ok || @warn "laplace: conjugate gradients stopped short of rtol = $(ls.rtol); the covariance is approximate"
        Σ[:, a] .= x[pos]
    end
    hld = logdet ? -_logdet(op.S, op.n, ls) / 2 : NaN
    return (Matrix(Symmetric((Σ .+ Σ') ./ 2)), hld)
end

"""
    implicit_mixture(m, z₀, ρ, ps, st; coords = free, linsolve = DenseSolve(), kwargs...) -> (MixtureBelief, st)

Every answer to a query as one belief: a mixture of the Laplace Gaussians at the stable roots
that [`implicit_roots`](@ref) finds (`kwargs` are passed on), marginalised onto `coords`.

The weights are the Laplace estimates of each answer's probability mass,
``w_k \\propto e^{-E(z_k)}\\,\\det(\\Sigma_k)^{1/2}``, with ``E`` the query's energy (the relation's
plus the clamp's) and ``\\Sigma_k`` the full free-coordinate covariance, when the relation has an
energy: an [`EnergyNetwork`](@ref), a closed-form [`GaussianMixtureEps`](@ref), or a product of
them. Otherwise the field determines no relative heights and the weights are equal. Build the
model with [`density_lambda`](@ref) so that the energy is a calibrated negative log-density.
Throws if no stable answer is found.

With `linsolve = KrylovSolve()` the roots are found matrix-free, the covariances are computed
on `coords` only, and ``\\log\\det\\Sigma_k`` is a stochastic Lanczos estimate (exact when at most
`probes` coordinates are free).
"""
function implicit_mixture(m::AbstractImplicitRelation, z₀, ρ, ps, st; coords = .!isinf.(ρ),
                          linsolve::AbstractLinearSolve = DenseSolve(), kwargs...)
    roots, st = implicit_roots(m, z₀, ρ, ps, st; linsolve, kwargs...)
    isempty(roots) && throw(ArgumentError("no stable answer found; try more starts or a larger spread"))
    idx = _indices(coords)
    comps = map(roots) do r
        Σ, hld = _laplace_marginal(linsolve, m, r, ρ, ps, st, idx; logdet = _has_energy(m))
        (LenticulumCore.Gaussian(r.z[idx], Σ), hld)
    end
    logw = if _has_energy(m)
        [-_query_energy(m, r.z, z₀, ρ, ps, st) + c[2] for (r, c) in zip(roots, comps)]
    else
        zeros(length(roots))
    end
    w = exp.(logw .- maximum(logw))
    return (LenticulumCore.MixtureBelief([c[1] for c in comps], w), st)
end

# --- Categorical answers ---------------------------------------------------------------------

"""
    categorical_answer(mix::MixtureBelief, classes::AbstractMatrix; labels = nothing, radius = nothing,
                       temperature = nothing, cov_scale = 1) -> (belief, unassigned, assignment)

A mixture answer on label coordinates read as a class: each component is assigned to the
nearest column of `classes` (the class codes, e.g. scaled one-hot vectors, one column per
class), and the weights of the components assigned to a class are summed.

- `belief`: a `CategoricalBelief` over the classes (with `labels`), normalised over the
  **assigned** mass; `nothing` if no component is assigned. With `temperature = τ` it is
  instead the posterior of the code given each component's position and covariance,
  ``p(k) = \\sum_j w_j\\,\\mathcal N(c_k;\\ \\mu_j,\\ \\kappa\\Sigma_j + \\tau^2 I) / \\sum_l \\mathcal N(c_l;\\ \\mu_j,\\ \\kappa\\Sigma_j + \\tau^2 I)``
  with ``\\kappa`` = `cov_scale`, which gives graded probabilities also when the query has a
  single root (a root between two codes splits its mass between them). Fit τ and κ on held-out
  answers with [`calibrate_temperature`](@ref): κ < 1 sharpens a Laplace covariance that is too
  wide, τ widens one that is too narrow.
- `unassigned`: the mixture weight of the components at least `radius` from every class
  code, reported rather than dropped. A root between the codes is an answer the relation gives
  but no class describes (a smoothing artefact, or a query outside the data).
- `assignment`: the class of each component, `0` for unassigned.

`radius` defaults to half the smallest distance between two class codes, so that the regions
do not overlap. The means are compared, not the Gaussians: a component's own spread does not
split its mass between classes.
"""
function categorical_answer(mix::LenticulumCore.MixtureBelief, classes::AbstractMatrix; labels = nothing,
                            radius::Union{Nothing,Real} = nothing, temperature::Union{Nothing,Real} = nothing,
                            cov_scale::Real = 1)
    K = size(classes, 2)
    K ≥ 1 || throw(ArgumentError("need at least one class code"))
    r = radius === nothing ? (K == 1 ? Inf :
        minimum(norm(classes[:, a] .- classes[:, b]) for a in 1:K for b in (a + 1):K) / 2) : float(radius)
    w = LenticulumCore.mixture_weights(mix)
    assignment = map(mix.components) do c
        μ = LenticulumCore.belief_mean(c)
        length(μ) == size(classes, 1) || throw(DimensionMismatch(
            "components on $(length(μ)) coordinates, class codes on $(size(classes, 1))"))
        d = [norm(μ .- view(classes, :, k)) for k in 1:K]
        k = argmin(d)
        d[k] < r ? k : 0
    end
    p = zeros(eltype(w), K)
    for (a, wi) in zip(assignment, w)
        a > 0 && (p[a] += wi)
    end
    unassigned = max(one(eltype(w)) - sum(p), zero(eltype(w)))
    if temperature !== nothing
        q = _code_mixture(mix, classes, temperature, cov_scale)
        return (belief = LenticulumCore.CategoricalBelief(q; labels), unassigned = unassigned, assignment = assignment)
    end
    belief = sum(p) > 0 ? LenticulumCore.CategoricalBelief(p; labels) : nothing
    return (belief = belief, unassigned = unassigned, assignment = assignment)
end

"""
    code_posterior(g, classes, τ; cov_scale = 1) -> Vector

The probability of each class code (column of `classes`) given a Gaussian answer `g` on the
label coordinates, with the codes equally likely a priori: ``\\propto \\mathcal N(c_k;\\ \\mu,\\ \\kappa\\Sigma + \\tau^2 I)``,
``\\kappa`` = `cov_scale`.
"""
function code_posterior(g, classes::AbstractMatrix, τ::Real; cov_scale::Real = 1)
    μ = LenticulumCore.belief_mean(g)
    Σ = Symmetric(cov_scale .* Matrix(LenticulumCore.belief_cov(g)) .+ τ^2 .* I(length(μ)))
    F = LinearAlgebra.cholesky(Σ)
    ℓ = [-sum(abs2, F.L \ (view(classes, :, k) .- μ)) / 2 for k in axes(classes, 2)]
    q = exp.(ℓ .- maximum(ℓ))
    return q ./ sum(q)
end

_code_mixture(mix, classes, τ, κ) =
    (q = sum(wi .* code_posterior(c, classes, τ; cov_scale = κ) for (wi, c) in zip(LenticulumCore.mixture_weights(mix), mix.components));
     q ./ sum(q))

"""
    calibrate_temperature(answers, classes, truth; bounds = (1e-3, 10), scale_bounds = (1e-4, 10))
        -> (temperature = τ, cov_scale = κ, nll)

The temperature τ and covariance scale κ of [`categorical_answer`](@ref) that maximise the
likelihood of the true classes `truth` (indices into the columns of `classes`) on held-out
answers: `answers` are the mixtures (`MixtureBelief`s on the label coordinates) of held-out
queries. Alternating golden-section searches over ``\\log\\tau`` and ``\\log\\kappa`` within the bounds
(τ in the units of the codes). `nll` is the mean negative log-likelihood at the optimum. Pass
`scale_bounds = (1, 1)` to fit τ alone.
"""
function calibrate_temperature(answers::AbstractVector, classes::AbstractMatrix, truth::AbstractVector;
                               bounds = (1e-3, 10.0), scale_bounds = (1e-4, 10.0), rounds::Integer = 4)
    length(answers) == length(truth) || throw(DimensionMismatch("one true class per answer"))
    nll(lτ, lκ) = -sum(log(max(_code_mixture(a, classes, exp(lτ), exp(lκ))[k], 1e-300))
                       for (a, k) in zip(answers, truth)) / length(truth)
    lτ, lκ = log(sqrt(prod(float.(bounds)))), 0.0
    lκ = clamp(lκ, log.(float.(scale_bounds))...)
    for _ in 1:rounds
        lτ = _golden(x -> nll(x, lκ), log.(float.(bounds))...)
        scale_bounds[1] == scale_bounds[2] && break
        lκ = _golden(x -> nll(lτ, x), log.(float.(scale_bounds))...)
    end
    return (temperature = exp(lτ), cov_scale = exp(lκ), nll = nll(lτ, lκ))
end

# the minimiser of a unimodal f on [a, b]
function _golden(f, a, b; iters::Integer = 50)
    a == b && return a
    φ = (sqrt(5) - 1) / 2
    c, d = b - φ * (b - a), a + φ * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:iters
        if fc < fd
            b, d, fd = d, c, fc
            c = b - φ * (b - a); fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + φ * (b - a); fd = f(d)
        end
    end
    return (a + b) / 2
end

"""
    implicit_classify(m, z₀, ρ, ps, st; label, classes, labels = nothing, radius = nothing,
                      starts = class codes, kwargs...) -> (answer, st)

Classification as a query: the coordinates `label` (indices) are free and read as a class,
the others are clamped or soft as `ρ` says. Runs [`implicit_mixture`](@ref) on the label
coordinates, by default started from every class code (the columns of `classes` placed in
the label coordinates, the rest of `z₀` unchanged), and reads the answer with
[`categorical_answer`](@ref) (with `radius`, `temperature` and `cov_scale`). `kwargs` go to the solver
(`maxiters`, `linsolve`, …).

`answer` holds `belief` (a `CategoricalBelief`, or `nothing`), `unassigned`, `assignment` and
`mixture`, the Gaussian mixture it was read from. The class probabilities are the Laplace masses
of the roots, so they are calibrated only when the relation has an energy (an
[`EnergyNetwork`](@ref) or a closed-form mixture) and λ is [`density_lambda`](@ref); without
an energy every root weighs the same.
"""
function implicit_classify(m::AbstractImplicitRelation, z₀, ρ, ps, st; label, classes::AbstractMatrix,
                           labels = nothing, radius = nothing, temperature = nothing, cov_scale = 1, starts = nothing,
                           kwargs...)
    idx = _indices(label)
    length(idx) == size(classes, 1) || throw(DimensionMismatch(
        "$(length(idx)) label coordinates, class codes of length $(size(classes, 1))"))
    all(!isinf, ρ[idx]) || throw(ArgumentError("the label coordinates must not be hard-clamped"))
    inits = starts === nothing ? [(z = float.(copy(z₀)); z[idx] .= c; z) for c in eachcol(classes)] : starts
    mix, st = implicit_mixture(m, z₀, ρ, ps, st; coords = idx, starts = inits, kwargs...)
    a = categorical_answer(mix, classes; labels, radius, temperature, cov_scale)
    return ((a..., mixture = mix), st)
end
