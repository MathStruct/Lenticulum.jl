# ---------------------------------------------------------------------------
# Coverage diagnostics: does an answer lie where the model has seen data?
#
# Inference always returns something, also for a query the training data never covered
# (`Open Problems in Implicit Diffusion Learning.md` T5); a root there is an extrapolation of the
# network, not a statement about the data. Three scores flag it, each turned into an empirical
# (conformal) p-value against reference points from the data distribution:
#
#   distance        to the nearest data point (needs the data)
#   reconstruction  how far Tweedie's denoiser moves the answer at a small noise level (any predictor)
#   energy          -log p_t at that noise level (energy networks, closed-form mixtures and KDEs)
#
# Large scores are atypical; a small p-value is the flag. See `coverage.md`.
# ---------------------------------------------------------------------------

"""
    CoverageReference

Reference scores of in-distribution points for [`coverage`](@ref), built by
[`coverage_reference`](@ref). It holds the relation and its parameters, the data for the
nearest-neighbour distance, the fixed noise draws and the scores of the calibration points.
"""
struct CoverageReference{R<:ImplicitDiffusion,P,S,M<:AbstractMatrix,T<:Real,E}
    m::R
    ps::P
    st::S
    data::M
    scale::Vector{T}
    t::T
    ξ::Matrix{T}
    distance::Vector{T}
    reconstruction::Vector{T}
    energy::E                       # Vector{T}, or `nothing` when the predictor has no energy
end

"""
    coverage_reference(m::ImplicitDiffusion, data, ps, st; t = 0.05, draws = 8,
                       calibration = nothing, ncal = 500, scale = 1, rng = Xoshiro(0))
        -> CoverageReference

Scores of in-distribution points, against which [`coverage`](@ref) ranks an answer. `data`
(``\\dim z\\times N``, one point per column) is the set the nearest-neighbour distance is
measured to, normally the training data.

- `calibration`: held-out points (``\\dim z\\times N_c``) whose scores are the reference. By
  default `ncal` columns of `data` are used, each with its own column left out of the
  nearest-neighbour search. Held-out points are better: a network scores its own training
  points too well, which makes every answer look slightly atypical.
- `t`, `draws`: the noise level of the reconstruction and energy scores, and the number of
  noise draws they average over. The draws are fixed, the same for the reference and every
  query, so the scores are deterministic.
- `scale`: a scalar or one value per coordinate; distances are measured in `(z - x) ./ scale`.
  Use it when coordinates have different units (pixels and a label, say).

Cost: `draws` predictor calls per calibration point (one call with `batched = true`), and a
pass over `data` for each distance.
"""
function coverage_reference(m::ImplicitDiffusion, data::AbstractMatrix, ps, st; t::Real = 0.05, draws::Integer = 8,
                            calibration::Union{Nothing,AbstractMatrix} = nothing, ncal::Integer = 500,
                            scale = 1, rng::AbstractRNG = Random.Xoshiro(0))
    n, N = size(data)
    T = float(eltype(data))
    sc = scale isa Real ? fill(T(scale), n) : T.(collect(scale))
    length(sc) == n && all(>(0), sc) || throw(ArgumentError("scale must be positive, a scalar or one value per coordinate"))
    if calibration === nothing
        idx = Random.randperm(rng, N)[1:min(ncal, N)]
        C = data[:, idx]
    else
        size(calibration, 1) == n || throw(DimensionMismatch("calibration has $(size(calibration, 1)) rows, data $n"))
        idx = fill(0, size(calibration, 2))          # 0: no column of `data` to leave out
        C = calibration
    end
    ξ = randn(rng, T, n, draws)
    X = T.(data)
    dist = [_nearest(X, sc, view(C, :, j), idx[j]) for j in axes(C, 2)]
    recon = similar(dist)
    en = _has_energy(m) ? similar(dist) : nothing
    for j in axes(C, 2)
        r, e, st = _model_scores(m, C[:, j], T(t), ξ, ps, st)
        recon[j] = r
        en === nothing || (en[j] = e)
    end
    return CoverageReference(m, ps, st, X, sc, T(t), ξ, dist, recon, en)
end

"""
    coverage(ref::CoverageReference, z) -> NamedTuple
    coverage(ref::CoverageReference, sol::ImplicitSolution) -> NamedTuple
    coverage(ref::CoverageReference, Z::AbstractMatrix) -> Vector{NamedTuple}

How typical the answer `z` is of the data, by three scores and their p-values:

| field | score |
|---|---|
| `distance`, `p_distance` | distance to the nearest data point |
| `reconstruction`, `p_reconstruction` | ``\\tfrac1K\\sum_k \\lVert\\hat x_0(\\alpha_t z + \\sigma_t\\xi_k) - z\\rVert^2``, Tweedie's denoiser at level `t` |
| `energy`, `p_energy` | ``\\tfrac1K\\sum_k E(\\alpha_t z + \\sigma_t\\xi_k, t)``, ``E = -\\log p_t``; `nothing` without an energy |
| `p` | the combined p-value: the smallest one times the number of scores (Bonferroni), at most 1 |

Each p-value is ``(1 + \\#\\{\\text{reference scores} \\ge s\\}) / (1 + N_c)``: the fraction of
calibration points at least as atypical. For an in-distribution point exchangeable with the
calibration points it is uniform, so `p < 0.05` flags about 5% of them by chance and an
answer outside the data with high probability. A matrix scores each column.

The scores read the model at the answer and the data near it; they do not certify the answer.
A root on the relation can still be the wrong branch (T1), and an answer of the smoothed
relation sits slightly off the data (T2), which the reconstruction score sees.

For a classification answer pass `unassigned`, the mass of [`categorical_answer`](@ref) that
lies between the class codes: it is returned, and `flagged` is true when `p < alpha` or at least
half the answer is unassigned, so that a root between two codes is flagged rather than read as
a class.
"""
function coverage(ref::CoverageReference, z::AbstractVector; unassigned::Real = 0.0, alpha::Real = 0.05)
    T = eltype(ref.distance)
    d = _nearest(ref.data, ref.scale, z, 0)
    r, e, _ = _model_scores(ref.m, collect(T.(z)), ref.t, ref.ξ, ref.ps, ref.st)
    pd, pr = _pvalue(ref.distance, d), _pvalue(ref.reconstruction, r)
    pe = ref.energy === nothing ? nothing : _pvalue(ref.energy, e)
    ps = pe === nothing ? (pd, pr) : (pd, pr, pe)
    p = min(one(T), length(ps) * minimum(ps))
    return (distance = d, reconstruction = r, energy = ref.energy === nothing ? nothing : e,
            p_distance = pd, p_reconstruction = pr, p_energy = pe, p = p,
            unassigned = unassigned, flagged = p < alpha || unassigned ≥ 0.5)
end
coverage(ref::CoverageReference, sol::ImplicitSolution; kwargs...) = coverage(ref, sol.z; kwargs...)
coverage(ref::CoverageReference, Z::AbstractMatrix; unassigned = zeros(size(Z, 2)), alpha::Real = 0.05) =
    [coverage(ref, Z[:, j]; unassigned = unassigned[j], alpha) for j in axes(Z, 2)]

_pvalue(reference, s) = (1 + count(≥(s), reference)) / (1 + length(reference))

# distance from z to the nearest column of X, leaving column `skip` out (0: none)
function _nearest(X, scale, z, skip)
    best = typemax(eltype(X))
    for j in axes(X, 2)
        j == skip && continue
        d = zero(eltype(X))
        for i in axes(X, 1)
            d += abs2((z[i] - X[i, j]) / scale[i])
        end
        best = min(best, d)
    end
    return sqrt(best)
end

# the reconstruction error and the energy at z, averaged over the fixed noise draws ξ
function _model_scores(m::ImplicitDiffusion, z, t, ξ, ps, st)
    s = m.predictor.schedule
    Xt = alpha(s, t) .* z .+ sigma(s, t) .* ξ
    X̂, st = _denoise_columns(m, Xt, t, ps, st)
    r = sum(abs2, X̂ .- z) / size(ξ, 2)
    e = _has_energy(m) ? _mean_energy(m.predictor, Xt, t, ps, st) : nothing
    return (r, e, st)
end

_mean_energy(p::NoisePredictor{<:EnergyNetwork}, Xt, t, ps, st) =
    sum(first(energy(p, Xt[:, k], t, ps, st)) for k in axes(Xt, 2)) / size(Xt, 2)
_mean_energy(p::NoisePredictor{<:GaussianMixtureEps}, Xt, t, ps, st) =
    -sum(mixture_logdensity(p.model, Xt[:, k], t, ps) for k in axes(Xt, 2)) / size(Xt, 2)
