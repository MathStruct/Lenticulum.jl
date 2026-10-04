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
predictor is an [`EnergyNetwork`](@ref) (lowest first, i.e. the most plausible answer first),
and otherwise in the order found. There is no guarantee that every stable answer is found;
more starts and a larger `spread` find more.
"""
function implicit_roots(m::ImplicitDiffusion, z₀, ρ, ps, st; nstarts::Integer = 16, spread = 1.0,
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
    if m.predictor isa NoisePredictor{<:EnergyNetwork} && length(roots) > 1
        energies = [_query_energy(m, r.z, z₀, ρ, ps, st) for r in roots]
        roots = roots[sortperm(energies)]
    end
    return (roots, st)
end

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
function implicit_laplace(m::ImplicitDiffusion, sol::ImplicitSolution, ρ, ps, st)
    (sol.converged && sol.stable) ||
        throw(ArgumentError("the Laplace approximation needs a converged, stable answer (a local minimum)"))
    free = .!isinf.(ρ)
    J = prior_jacobian(m, sol.z, ps, st)[free, free] .+ Diagonal(ρ[free] .^ 2)
    ΣF = inv(Symmetric((J .+ J') ./ 2))
    n = length(sol.z)
    Σ = zeros(eltype(ΣF), n, n)
    Σ[free, free] .= ΣF
    return (mean = sol.z, cov = Σ, free = free)
end
