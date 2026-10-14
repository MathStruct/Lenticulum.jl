# Where a model's samples go wrong: its denoising error at each noise level, and where the
# sampler's speckles come from.
#
#     MODEL=… KIND=… BLOCKS=… REF=… julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/denoise_eval.jl
#
# 1. The ε error per pixel on test images noised to level t, label clean.
# 2. Test images noised to t₀ (label held clean), then the sampler's ancestral steps down to tmin
#    (as many per unit log-SNR as a full run of 200 steps). Reported: the fraction of the 3-pixel
#    border, background in every digit, that ends brighter than -0.5, and the RMSE to the
#    original. From t₀ = 1 this is plain generation. Recorded in `MNIST in Both Directions.md` §9.

include("common.jl")
using VariationalDiffusion: alpha, sigma

const BORDER = vec([min(r, c, 29 - r, 29 - c) ≤ 3 for c in 1:28, r in 1:28])
ps, st = load_model(); ps = GDEV(ps)
pred = predictor(network())
const S = SCHED
times(t, B; label = t) = BLOCKS ? vcat(fill(t, 1, B), fill(label, 1, B)) : fill(t, 1, B)
eps(X, T) = reduce(hcat, [first(epsilon(pred, X[:, r], T[:, r], ps, st)) for r in Iterators.partition(1:size(X, 2), CHUNK)])

ℓ(t) = log(alpha(S, t) / sigma(S, t))
function time_of(l)
    lo, hi = S.tmin, 1.0
    for _ in 1:100
        mid = (lo + hi) / 2
        ℓ(mid) > l ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
function grid(t₀; steps = 200)
    k = max(round(Int, steps * (ℓ(S.tmin) - ℓ(t₀)) / (ℓ(S.tmin) - ℓ(1.0))), 5)
    ls = range(ℓ(t₀), ℓ(S.tmin); length = k + 1)
    return [t₀; time_of.(ls[2:(end - 1)]); S.tmin]
end

# with blocks the label is held clean (at tmin); without, it is noised with the image
function from(Z₀, t₀; rng = Xoshiro(0))
    d, n = size(Z₀)
    lab(t) = BLOCKS ? S.tmin : t
    hold(X, t) = (X[LABEL, :] .= alpha(S, lab(t)) .* Z₀[LABEL, :] .+ sigma(S, lab(t)) .* randn(rng, NY, n); X)
    X = alpha(S, t₀) .* Z₀ .+ sigma(S, t₀) .* randn(rng, d, n)
    ts = grid(t₀)
    for k in 1:(length(ts) - 1)
        t, u = ts[k], ts[k + 1]
        hold(X, t)
        X̂ = (X .- sigma(S, t) .* eps(X, times(t, n; label = lab(t)))) ./ alpha(S, t)
        a = alpha(S, t) / alpha(S, u)
        σt², σu² = sigma(S, t)^2, sigma(S, u)^2
        σtu² = σt² - a^2 * σu²
        X = (a * σu² / σt²) .* X .+ (alpha(S, u) * σtu² / σt²) .* X̂ .+ sqrt(σtu² * σu² / σt²) .* randn(rng, d, n)
    end
    t = last(ts)
    hold(X, t)
    return (X .- sigma(S, t) .* eps(X, times(t, n; label = lab(t)))) ./ alpha(S, t)
end

name = get(ENV, "MODEL", "energy")
Z, _ = joint(:test)
rng = Xoshiro(1)
x₀ = Z[:, 1:200]
for t in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.7)
    T = times(t, 200; label = BLOCKS ? S.tmin : t)
    ε = randn(rng, N, 200)
    xt = copy(x₀)
    xt[IMAGE, :] .= alpha(S, t) .* x₀[IMAGE, :] .+ sigma(S, t) .* ε[IMAGE, :]
    l = BLOCKS ? S.tmin : t
    xt[LABEL, :] .= alpha(S, l) .* x₀[LABEL, :] .+ sigma(S, l) .* ε[LABEL, :]
    @printf("%-20s t = %.3f (σ %.3f): image ε MSE per pixel %.3f\n", name, t, sigma(S, t),
            mean(abs2, eps(xt, T)[IMAGE, :] .- ε[IMAGE, :]))
    flush(stdout)
end
for t₀ in (0.1, 0.3, 0.5, 1.0)
    X = from(Z[:, 1:50], t₀)
    @printf("%-20s from t₀ = %.1f (σ %.2f): border speckles %.4f, RMSE to the original %.3f\n", name, t₀, sigma(S, t₀),
            mean(X[IMAGE[BORDER], :] .> -0.5), sqrt(mean(abs2, clamp.(X[IMAGE, :], -1, 1) .- Z[IMAGE, 1:50])))
    flush(stdout)
end
