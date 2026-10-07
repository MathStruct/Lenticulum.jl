# ---------------------------------------------------------------------------
# Gaussian beliefs: projections and divergences.
#
# The type `GaussianBelief` and its accessors live in LenticulumCore (`gaussian_belief.jl`), its
# message rules in Mycelium (`gaussian.jl`). What stays here are the operations that map other
# beliefs *onto* Gaussians: `moment_match` (the projection step of expectation propagation),
# `reduce_mixture`, and `kl_divergence`. See `Gaussian Belief.md` and `Belief Algebra.md`.
# ---------------------------------------------------------------------------

"""
    kl_divergence(q::GaussianBelief, p::GaussianBelief) -> Real

``D_{KL}(q \\,\\|\\, p)``. Both must be proper. Used to measure the laxness quantities the
framework keeps insisting should be reported rather than hidden — see
`Composition of Bayesian Lenses.md` Remark 16.
"""
function kl_divergence(q::GaussianBelief, p::GaussianBelief)
    Cq, Cp = _improper_check(q, "kl_divergence"), _improper_check(p, "kl_divergence")
    n = dimension(q)
    Σq = inv(Cq)
    d = (Cq \ q.η) .- (Cp \ p.η)
    return (tr(p.Λ * Σq) + d' * (p.Λ * d) - n + logdet(Cq) - logdet(Cp)) / 2
end

"""
    moment_match(b) -> GaussianBelief

The Gaussian with the same mean and covariance as `b`: the minimiser of ``\\mathrm{KL}(b \\,\\|\\, q)``
over Gaussians ``q``, i.e. the projection step of expectation propagation. Defined for a
`SampleBelief` (weighted sample mean and covariance; vector or matrix samples), a
`MixtureBelief` of Gaussians (total mean and covariance), and a Gaussian (itself).
"""
moment_match(b::GaussianBelief) = b

function moment_match(s::LenticulumCore.SampleBelief)
    X = s.samples isa AbstractMatrix ? Matrix{Float64}(s.samples) : reduce(hcat, [collect(float.(x)) for x in s.samples])
    n = size(X, 2)
    w = s.weights === nothing ? fill(1 / n, n) : collect(float.(s.weights)) ./ sum(s.weights)
    μ = X * w
    D = X .- μ
    Σ = (D .* w') * D'
    isposdef(Symmetric(Σ)) || throw(ArgumentError(
        "the weighted sample covariance is singular (too few distinct samples for the dimension)"))
    return Gaussian(μ, Matrix(Symmetric(Σ)))
end

function moment_match(m::LenticulumCore.MixtureBelief{<:GaussianBelief})
    w = LenticulumCore.mixture_weights(m)
    μs = [belief_mean(c) for c in m.components]
    μ = sum(w[k] .* μs[k] for k in eachindex(w))
    Σ = sum(w[k] .* (belief_cov(m.components[k]) .+ (μs[k] .- μ) * (μs[k] .- μ)') for k in eachindex(w))
    return Gaussian(μ, Matrix(Symmetric(Σ)))
end

"""
    reduce_mixture(m::MixtureBelief{<:GaussianBelief}; max_components, prune = 1e-8)
        -> MixtureBelief

Bound the size of a Gaussian mixture, which products make grow multiplicatively. Components
with weight below `prune` (relative to the largest) are dropped; then, while there are more
than `max_components`, the pair whose merge costs least is replaced by the single Gaussian
with their combined weight, mean and covariance. The cost is Runnalls' upper bound on the
Kullback–Leibler divergence the merge introduces,
``\\tfrac12\\bigl[(w_i + w_j)\\log\\det\\Sigma_{ij} - w_i\\log\\det\\Sigma_i - w_j\\log\\det\\Sigma_j\\bigr]``.
Merging preserves the mixture's overall mean and covariance exactly; pruning changes them by
at most the pruned weight.
"""
function reduce_mixture(m::LenticulumCore.MixtureBelief{<:GaussianBelief}; max_components::Integer, prune::Real = 1e-8)
    max_components ≥ 1 || throw(ArgumentError("max_components must be at least 1"))
    w = LenticulumCore.mixture_weights(m)
    keep = w .≥ prune * maximum(w)
    ws = w[keep]
    μs = [belief_mean(c) for c in m.components[keep]]
    Σs = [belief_cov(c) for c in m.components[keep]]
    function merged(i, j)
        wij = ws[i] + ws[j]
        μ = (ws[i] .* μs[i] .+ ws[j] .* μs[j]) ./ wij
        Σ = (ws[i] .* (Σs[i] .+ (μs[i] .- μ) * (μs[i] .- μ)') .+ ws[j] .* (Σs[j] .+ (μs[j] .- μ) * (μs[j] .- μ)')) ./ wij
        return wij, μ, Matrix(Symmetric(Σ))
    end
    while length(ws) > max_components
        best, bi, bj = Inf, 0, 0
        for i in eachindex(ws), j in (i + 1):length(ws)
            wij, _, Σ = merged(i, j)
            cost = (wij * logdet(Σ) - ws[i] * logdet(Σs[i]) - ws[j] * logdet(Σs[j])) / 2
            cost < best && ((best, bi, bj) = (cost, i, j))
        end
        wij, μ, Σ = merged(bi, bj)
        ws[bi], μs[bi], Σs[bi] = wij, μ, Σ
        deleteat!(ws, bj); deleteat!(μs, bj); deleteat!(Σs, bj)
    end
    return LenticulumCore.MixtureBelief([Gaussian(μs[k], Σs[k]) for k in eachindex(ws)], ws)
end
