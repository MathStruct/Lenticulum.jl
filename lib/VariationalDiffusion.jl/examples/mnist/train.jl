# Train the energy network of `common.jl` by denoising score matching on (image, label).
#
#     julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/train.jl [epochs]
#
# The loss is the denoising loss with the label coordinates weighted by `LABEL_WEIGHT` (env,
# default 1): the label is 10 of 794 coordinates, and unweighted it barely enters the loss. Its
# parameter gradient is one mixed second derivative (`epsilon_vjp_params` of the weighted
# residual), forward over reverse on the GPU. Writes ~/.cache/lenticulum/mnist/$MODEL.jld2.

include("common.jl")

# the denoising loss Σ_i w_i (ε̂_i - ε_i)² / B and its parameter gradient
function weighted_gradient(pred, x₀, t, ε, ps, st, w)
    s = pred.schedule
    xt = alpha.(Ref(s), t) .* x₀ .+ sigma.(Ref(s), t) .* ε
    ε̂, _ = epsilon(pred, xt, t, ps, st)
    r = ε̂ .- ε
    B = size(x₀, 2)
    return (sum(w .* r .^ 2) / B, epsilon_vjp_params(pred, xt, t, ps, st, (2 / B) .* w .* r))
end

function train(; epochs = 30, batch = 256, lr = 1e-3, rng = Xoshiro(0), label_weight = 1.0)
    Z, _ = joint(:train)
    wts = [ones(NX); fill(label_weight, NY)]
    net = network()
    ps, st = Lux.setup(rng, net)
    psd = GDEV(ps)
    pred = predictor(net)
    opt = Optimisers.setup(Optimisers.AdamW(lr, (0.9, 0.999), 1e-5), psd)
    nb = size(Z, 2) ÷ batch
    steps = epochs * nb
    k = 0
    for ep in 1:epochs
        perm = randperm(rng, size(Z, 2))
        tot = 0.0
        t0 = time()
        for b in 1:nb
            k += 1
            Optimisers.adjust!(opt, lr * (1 + cos(π * k / steps)) / 2)          # cosine decay
            idx = perm[((b - 1) * batch + 1):(b * batch)]
            x₀ = Z[:, idx]
            t = SCHED.tmin .+ (1 - SCHED.tmin) .* rand(rng, 1, batch)
            ε = randn(rng, N, batch)
            L, g = weighted_gradient(pred, x₀, t, ε, psd, st, wts)
            opt, psd = Optimisers.update!(opt, psd, g)
            tot += L
        end
        @printf("epoch %3d  weighted loss per coordinate %.4f  (%.0f s)\n", ep, tot / nb / sum(wts), time() - t0)
        flush(stdout)
    end
    JLD2.save(modelfile(), Dict("ps" => CDEV(psd), "st" => st, "label_weight" => label_weight, "epochs" => epochs))
    return psd
end

train(; epochs = parse(Int, get(ARGS, 1, "30")), label_weight = parse(Float64, get(ENV, "LABEL_WEIGHT", "1")))
