"""
A linear factor read like a quantum measurement: the factor's state is a density matrix
ρ = Σₖ λₖ uₖuₖᵀ over z = (x, y), fitted in closed form as the normalised second moment of the
data. Clamping x and measuring returns eigenvector k with the Born weight λₖ (uₖˣ·x)², and the
point on that eigen-line that matches x: one shot, no iterative denoising, and as many modes
as eigenvectors. Run with `julia --project=. examples/measurement_factor.jl`; the test suite
includes it. See `measurement_factor.md`.

Two experiments against the exact conditional of the data-generating model:
1. two orthogonal lines: the eigenvectors are the lines, and the measurement is close to exact;
2. two lines 60° apart: eigenvectors are orthogonal, so they cannot be the lines.
"""
module MeasurementFactors

using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, Printf
using LenticulumCore: DiracBelief, MixtureBelief, Observed, Unobserved, Polarity, Channel

# --- data: noisy lines through the origin ----------------------------------------------------

"""
    lines_data(θs, wts, n; σ, rng) -> (X, Y)

`n` points z = t·(cos θₖ, sin θₖ) + σ ε with k ~ wts, t ~ N(0, 1), ε ~ N(0, I₂).
"""
function lines_data(θs, wts, n; σ = 0.05, rng = Xoshiro(1))
    X, Y = zeros(n), zeros(n)
    c = cumsum(wts)
    for i in 1:n
        k = findfirst(>=(rand(rng)), c)
        t = randn(rng)
        X[i] = t * cos(θs[k]) + σ * randn(rng)
        Y[i] = t * sin(θs[k]) + σ * randn(rng)
    end
    return X, Y
end

"""
    oracle_logpdf(θs, wts, σ, x, y)

log p(y | x) under the true model: per line, (x, y) ~ N(0, uuᵀ + σ²I); the conditional is a
mixture of the per-line Gaussian conditionals, weighted by wₖ N(x; 0, Σₖ,ₓₓ).
"""
function oracle_logpdf(θs, wts, σ, x, y)
    terms = map(eachindex(θs)) do k
        c, s = cos(θs[k]), sin(θs[k])
        Sxx, Sxy, Syy = c^2 + σ^2, c * s, s^2 + σ^2
        m, v = Sxy / Sxx * x, Syy - Sxy^2 / Sxx
        log(wts[k]) - (x^2 / Sxx + log(2π * Sxx)) / 2 - ((y - m)^2 / v + log(2π * v)) / 2
    end
    lx = log(sum(wts[k] * exp(-(x^2 / (cos(θs[k])^2 + σ^2) + log(2π * (cos(θs[k])^2 + σ^2))) / 2) for k in eachindex(θs)))
    M = maximum(terms)
    return M + log(sum(exp.(terms .- M))) - lx
end

# --- the factor --------------------------------------------------------------------------------

"""
    MeasurementFactor(λ, U, nx, width)

A density matrix ρ = U diag(λ) Uᵀ over z = (x, y) with `nx` x-coordinates, read as a
measurement: with x clamped, the message to y is a mixture with one component per eigenvector,
weight ∝ λₖ (uₖˣ·x)² and mean uₖʸ (uₖˣ·x)/‖uₖˣ‖² (the point on eigen-line k whose x-part best
matches x), each with covariance `width·I`.
"""
struct MeasurementFactor <: LenticulumCore.AbstractLenticulumFactor
    λ::Vector{Float64}
    U::Matrix{Float64}
    nx::Int
    width::Float64
end

"Fit ρ = ZᵀZ / tr(ZᵀZ) from rows z = (x, y), and the component width from the y-residuals."
function fit_measurement(X::AbstractMatrix, Y::AbstractMatrix)
    Z = hcat(X, Y)
    M = Z' * Z
    E = eigen(Symmetric(M / tr(M)))
    f0 = MeasurementFactor(E.values, E.vectors, size(X, 2), 1.0)
    # width: mean squared y-residual to the nearest component, per y-coordinate (not the most
    # probable one: with scalar x the Born weights do not depend on x)
    r2 = sum(axes(Z, 1)) do i
        _, m = born(f0, X[i, :])
        minimum(sum(abs2, Y[i, :] - mk) for mk in m)
    end
    return MeasurementFactor(E.values, E.vectors, size(X, 2), r2 / (size(Z, 1) * size(Y, 2)))
end
fit_measurement(x::AbstractVector, y::AbstractVector) = fit_measurement(reshape(x, :, 1), reshape(y, :, 1))

"Born weights and per-component y-means for a clamped x."
function born(f::MeasurementFactor, x::AbstractVector)
    Ux, Uy = f.U[1:f.nx, :], f.U[f.nx+1:end, :]
    a = Ux' * x
    w = f.λ .* a .^ 2
    m = [Uy[:, k] * a[k] / max(sum(abs2, Ux[:, k]), eps()) for k in axes(f.U, 2)]
    return w ./ sum(w), m
end

"One measurement: draw an outcome by its Born weight, return that eigen-line's point."
function measure(f::MeasurementFactor, x::AbstractVector; rng = Random.default_rng())
    w, m = born(f, x)
    return m[findfirst(>=(rand(rng)), cumsum(w))]
end

logpdf(f::MeasurementFactor, x, y) = Mycelium.belief_logdensity(message(f, x), y)
function message(f::MeasurementFactor, x::AbstractVector)
    w, m = born(f, x)
    keep = findall(>(0), w)
    ny = size(f.U, 1) - f.nx
    return MixtureBelief([Gaussian(m[k], Matrix(f.width * I, ny, ny)) for k in keep], w[keep])
end

# As a graph factor: x observed, y the target.
LenticulumCore.channels(f::MeasurementFactor) = (Channel(:x, f.nx), Channel(:y, size(f.U, 1) - f.nx))
LenticulumCore.supported_polarities(::MeasurementFactor) = (Polarity(x = Observed(), y = Unobserved()),)
LenticulumCore.supports_polarity(::MeasurementFactor, p::Polarity) =
    keys(p) == (:x, :y) && p[:x] isa Observed && p[:y] isa Unobserved
function Mycelium.factor_message(f::MeasurementFactor, target::Symbol, pol, inputs, prior, ps, st)
    target === :y && inputs.x isa DiracBelief || throw(ArgumentError("the measurement factor answers y from a point x"))
    return message(f, collect(float.(inputs.x.value))), st
end

# --- the baseline: one linear-Gaussian conditional, by least squares ---------------------------

function gaussian_baseline(x, y)
    β = dot(x, y) / dot(x, x)
    v = sum(abs2, y - β * x) / length(x)
    return (β = β, v = v, logpdf = (xi, yi) -> -((yi - β * xi)^2 / v + log(2π * v)) / 2)
end

# --- experiments ------------------------------------------------------------------------------

angle_deg(u) = mod(rad2deg(atan(u[2], u[1])), 180)           # a line's angle in [0, 180)
line_error(a, b) = min(abs(a - b), 180 - abs(a - b))

"""
    experiment(θs, wts; σ, n, ntest, seed) -> NamedTuple

Fit on `n` points, score mean log p(y | x) on `ntest` held-out points for the measurement
factor, the Gaussian baseline and the true model; report eigen-line angle errors.
"""
function experiment(θs, wts; σ = 0.05, n = 20_000, ntest = 5_000, seed = 1)
    x, y = lines_data(θs, wts, n; σ, rng = Xoshiro(seed))
    xt, yt = lines_data(θs, wts, ntest; σ, rng = Xoshiro(seed + 1000))
    f = fit_measurement(x, y)
    g = gaussian_baseline(x, y)
    angles = sort(angle_deg.(eachcol(f.U)))
    truth = sort(mod.(rad2deg.(θs), 180))
    err = [minimum(line_error(a, t) for a in angles) for t in truth]
    score(h) = sum(h(xt[i], yt[i]) for i in eachindex(xt)) / ntest
    return (; f, g, angles, truth, err,
            ll_measure = score((a, b) -> logpdf(f, [a], [b])),
            ll_gauss = score(g.logpdf),
            ll_oracle = score((a, b) -> oracle_logpdf(θs, wts, σ, a, b)))
end

function main()
    rng = Xoshiro(7)
    println("1. Two orthogonal lines, y = x (weight 0.3) and y = -x (weight 0.7), σ = 0.05")
    A = experiment([π / 4, -π / 4], [0.3, 0.7])
    @printf("   eigen-lines at %s°, true %s°; eigenvalues %s\n", round.(A.angles; digits = 2), A.truth, round.(A.f.λ; digits = 3))
    w, m = born(A.f, [1.0])
    @printf("   at x = 1: weights %s, y-means %s\n", round.(w; digits = 3), round.(only.(m); digits = 3))
    draws = [only(measure(A.f, [1.0]; rng)) for _ in 1:10_000]
    @printf("   10000 one-shot measurements at x = 1: %.3f land on y = +1\n", count(>(0), draws) / 10_000)
    @printf("   held-out mean log p(y|x): measurement %.3f, Gaussian %.3f, true model %.3f\n", A.ll_measure, A.ll_gauss, A.ll_oracle)
    println("2. Two lines at 0° and 60°, equal weights")
    B = experiment([0.0, π / 3], [0.5, 0.5])
    @printf("   eigen-lines at %s°, true %s°: angle errors %s°\n", round.(B.angles; digits = 2), B.truth, round.(B.err; digits = 2))
    @printf("   held-out mean log p(y|x): measurement %.3f, Gaussian %.3f, true model %.3f\n", B.ll_measure, B.ll_gauss, B.ll_oracle)
    return (; A, B)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    MeasurementFactors.main()
end
