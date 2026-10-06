# ---------------------------------------------------------------------------
# Kernel models: a Gaussian kernel density estimate is a Gaussian mixture with one centre per
# sample, so `GaussianMixtureEps` already is one, with exact derivatives. This file adds
#
#   kde_predictor / kde_bandwidth   the batch KDE and its bandwidth by held-out likelihood;
#   OnlineKDE / observe!            a KDE that learns from a stream: new samples are added or
#                                   merged into nearby centres, old ones are forgotten
#                                   geometrically, and the number of centres has a budget.
#
# See `kernel.md` and `Kernel Methods for Implicit Learning.md`.
# ---------------------------------------------------------------------------

"""
    kde_predictor(schedule, data::AbstractMatrix; bandwidth) -> NoisePredictor

A Gaussian **kernel density estimate** of the samples in the columns of `data`, as a noise
predictor: [`GaussianMixtureEps`](@ref) with one component per sample and width `bandwidth`.
Its ``\\varepsilon`` is the exact optimal noise predictor for that density, so every query function
([`implicit_infer`](@ref), [`implicit_roots`](@ref), [`implicit_laplace`](@ref)) works on it with
exact derivatives and no training. It is the classical baseline for a diffusion network: the
relation it defines is the ridge of the KDE. Query cost grows linearly with the number of
samples. Choose the bandwidth with [`kde_bandwidth`](@ref).
"""
kde_predictor(s::AbstractNoiseSchedule, data::AbstractMatrix; bandwidth::Real) =
    NoisePredictor(GaussianMixtureEps(s, float.(data); s = bandwidth), s)

"""
    kde_bandwidth(data; candidates = nothing, holdout = 0.2, rng = Xoshiro(0)) -> h

The bandwidth of an isotropic Gaussian KDE that maximises the **held-out log-likelihood**: a
random fraction `holdout` of the columns of `data` is set aside, a KDE on the rest is evaluated
on it, for each candidate. The default candidates span 0.5 % to 100 % of the data's scale
(the root mean coordinate variance) on a log grid. This chooses the best *density*, not the
best answers to queries, and uses no query information.
"""
function kde_bandwidth(data::AbstractMatrix; candidates = nothing, holdout = 0.2,
                       rng::AbstractRNG = Random.Xoshiro(0))
    n, d = size(data, 2), size(data, 1)
    perm = Random.randperm(rng, n)
    m = max(1, round(Int, holdout * n))
    test, train = data[:, perm[1:m]], data[:, perm[(m + 1):end]]
    μ = sum(data; dims = 2) ./ n
    scale = sqrt(sum(abs2, data .- μ) / (n * d))
    hs = candidates === nothing ? scale .* exp.(range(log(0.005), 0; length = 25)) : collect(candidates)
    D = [sum(abs2, view(test, :, j) .- view(train, :, i)) for i in axes(train, 2), j in axes(test, 2)]
    function heldout(h)
        ℓ = -D ./ (2h^2)
        mx = maximum(ℓ; dims = 1)
        return sum(mx .+ log.(sum(exp.(ℓ .- mx); dims = 1) ./ size(train, 2))) / m - d / 2 * log(2π * h^2)
    end
    return hs[argmax(heldout.(hs))]
end

"""
    OnlineKDE(dim; bandwidth, forget = 1.0, merge_radius = bandwidth / 2, budget = 1000,
              prune = 1e-4)

A Gaussian kernel density estimate that learns from a **stream** of samples and can **forget**.
Feed it with [`observe!`](@ref); query it through [`kde_predictor`](@ref)`(schedule, kde)`, which
returns a weighted [`GaussianMixtureEps`](@ref), so every query function works on the current
state.

Per observed sample:
1. every existing weight is multiplied by `forget` (``\\le 1``): a sample observed ``k`` samples
   ago carries weight ``\\text{forget}^k``, so the effective memory is about
   ``1/(1 - \\text{forget})`` samples (`forget = 1` remembers everything, equally);
2. the sample is **merged** into the nearest centre if it lies within `merge_radius`
   (the centre moves to the weighted mean, the weights add), and becomes a new centre otherwise;
3. centres whose weight fell below `prune` times the largest are dropped (forgotten), and while
   more than `budget` remain, the lightest centre is merged into its nearest neighbour
   (weighted mean, weights added), so compression never discards the newest samples.

Merging keeps the cost of a query bounded by the budget instead of growing with the stream;
it changes the density by at most about `merge_radius` in each centre's position. The
bandwidth is fixed; choose it on a first batch with [`kde_bandwidth`](@ref).
"""
mutable struct OnlineKDE{T<:Real}
    centres::Matrix{T}
    weights::Vector{T}
    bandwidth::T
    forget::T
    merge_radius::T
    budget::Int
    prune::T
    seen::Int
end
function OnlineKDE(dim::Integer; bandwidth::Real, forget::Real = 1.0, merge_radius::Real = bandwidth / 2,
                   budget::Integer = 1000, prune::Real = 1e-4)
    0 < forget ≤ 1 || throw(ArgumentError("forget must lie in (0, 1]"))
    T = float(promote_type(typeof(bandwidth), typeof(forget)))
    return OnlineKDE{T}(zeros(T, dim, 0), T[], T(bandwidth), T(forget), T(merge_radius), Int(budget), T(prune), 0)
end

"""
    observe!(kde::OnlineKDE, samples::AbstractMatrix) -> kde
    observe!(kde::OnlineKDE, sample::AbstractVector) -> kde

Add samples (columns) to the stream, in order: forget, merge or insert, prune. See
[`OnlineKDE`](@ref).
"""
observe!(k::OnlineKDE, X::AbstractMatrix) = (foreach(x -> observe!(k, x), eachcol(X)); k)
function observe!(k::OnlineKDE, x::AbstractVector)
    k.weights .*= k.forget
    k.seen += 1
    J = size(k.centres, 2)
    if J > 0
        d2 = vec(sum(abs2, k.centres .- x; dims = 1))
        j = argmin(d2)
        if d2[j] ≤ k.merge_radius^2
            w = k.weights[j]
            k.centres[:, j] .= (w .* k.centres[:, j] .+ x) ./ (w + 1)
            k.weights[j] = w + 1
            return _prune!(k)
        end
    end
    k.centres = hcat(k.centres, float.(x))
    push!(k.weights, one(eltype(k.weights)))
    return _prune!(k)
end

function _prune!(k::OnlineKDE)
    keep = k.weights .≥ k.prune * maximum(k.weights)          # forgotten: negligible weight
    if !all(keep)
        k.centres = k.centres[:, keep]
        k.weights = k.weights[keep]
    end
    # over budget: merge the lightest centre into its nearest neighbour (mass and mean kept),
    # rather than dropping it, which would discard the newest samples first
    while size(k.centres, 2) > k.budget
        j = argmin(k.weights)
        d2 = vec(sum(abs2, k.centres .- k.centres[:, j]; dims = 1))
        d2[j] = Inf
        i = argmin(d2)
        wi, wj = k.weights[i], k.weights[j]
        k.centres[:, i] .= (wi .* k.centres[:, i] .+ wj .* k.centres[:, j]) ./ (wi + wj)
        k.weights[i] = wi + wj
        k.centres = k.centres[:, setdiff(1:end, j)]
        deleteat!(k.weights, j)
    end
    return k
end

"""
    kde_predictor(schedule, kde::OnlineKDE) -> NoisePredictor

The current state of an [`OnlineKDE`](@ref) as a noise predictor: a weighted
[`GaussianMixtureEps`](@ref) with the centres as parameters (`ps.μ`, from `LuxCore.setup`).
"""
kde_predictor(s::AbstractNoiseSchedule, k::OnlineKDE) =
    NoisePredictor(GaussianMixtureEps(s, copy(k.centres); s = k.bandwidth, weights = k.weights), s)
