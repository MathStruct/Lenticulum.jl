# ---------------------------------------------------------------------------
# Gaussian beliefs: projections and divergences.
#
# The type `GaussianBelief` and its accessors live in LenticulumCore (`gaussian_belief.jl`), its
# message rules in Mycelium (`gaussian.jl`). What stays here are the operations that map other
# beliefs *onto* Gaussians: `moment_match` (the projection step of expectation propagation) and
# `kl_divergence`. `reduce_mixture` and the `reduce_message` of Gaussian mixtures moved to Mycelium
# (`gaussian.jl`, P17); Lenticulum re-exports `reduce_mixture`. See `Gaussian Belief.md` and `Belief Algebra.md`.
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
    w = s.weights === nothing ? fill(1 / n, n) : collect(float.(s.weights))
    maximum(w) > 0 || throw(ArgumentError("all sample weights are zero"))
    w ./= maximum(w)
    w ./= sum(w)
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
