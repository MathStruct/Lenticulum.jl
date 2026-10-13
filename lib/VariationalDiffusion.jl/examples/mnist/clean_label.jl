# Is the label ignored, or only noised away? The reverse diffusion with the label clamped at its
# clean value at every step (instead of its forward marginal α_t y + σ_t ξ, the replacement method),
# scored like `sample_eval.jl`. A model trained with one noise level for image and label never saw
# a clean label beside a noisy image, so this probes whether it uses the label at all.
#
#     MODEL=… KIND=… julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/clean_label.jl

include("sample_eval.jl")
const VD = VariationalDiffusion

function clean_sample(m, y, ps, st; n = 20, steps = 200, rng = Xoshiro(0))
    s = SCHED
    ts = VD._logsnr_grid(s, 1.0, s.tmin, steps)
    X = randn(rng, N, n); X[LABEL, :] .= y
    for k in 1:steps
        X, st = VD._ancestral(m, X, ts[k], ts[k + 1], ps, st, rng)
        X[LABEL, :] .= y
    end
    X̂, _ = VD._denoise_columns(m, X, last(ts), ps, st)
    return X̂
end

hits, conf = 0, 0.0
for k in 0:9
    Z = clean_sample(m, codes()[:, k + 1], ps, st; rng = Xoshiro(k))
    P = softmax_of(Z)
    global hits += count(==(k), vec(map(argmax, eachcol(P))) .- 1); global conf += sum(P[k + 1, :])
end
@printf("%-8s clean label at every step: recognised as the clamped label %.3f, mean probability %.3f\n",
        get(ENV, "MODEL", "energy"), hits / 200, conf / 200)
