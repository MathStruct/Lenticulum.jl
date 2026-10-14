# Train the energy network of `common.jl` by denoising score matching on (image, label).
#
#     julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/train.jl [epochs]
#
# The loss is the denoising loss with the label coordinates weighted by `LABEL_WEIGHT` (env,
# default 1): the label is 10 of 794 coordinates, and unweighted it barely enters the loss. Its
# parameter gradient is one mixed second derivative (`denoising_gradient` with `weights`), forward
# over reverse on the GPU. With BLOCKS=1 image and label get independent times (`block_times`),
# a fraction JOINT (env, default 0.2) of the batch a shared one, and a fraction CLEAN (env, default
# 0) of the rest one block held clean at tmin, as a query holds it; with HELD_LOSS=0 that block
# is left out of the loss. Writes
# ~/.cache/lenticulum/mnist/$MODEL.jld2.

include("common.jl")

function train(; epochs = 30, batch = 256, lr = 1e-3, rng = Xoshiro(0), label_weight = 1.0,
               joint_fraction = parse(Float64, get(ENV, "JOINT", "0.2")), clean = parse(Float64, get(ENV, "CLEAN", "0")),
               held_loss = get(ENV, "HELD_LOSS", "1") == "1")
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
            t = BLOCKS ? block_times(rng, pred, batch; joint = joint_fraction, clean) : SCHED.tmin .+ (1 - SCHED.tmin) .* rand(rng, 1, batch)
            ε = randn(rng, N, batch)
            # HELD_LOSS=0: a block held clean is a condition, not a target, and leaves the loss
            w = BLOCKS && !held_loss ? wts .* (t[pred.blocks.owner, :] .> SCHED.tmin) : wts
            L, g, _ = denoising_gradient(pred, x₀, t, ε, psd, st; weights = w)
            opt, psd = Optimisers.update!(opt, psd, g)
            tot += L
        end
        @printf("epoch %3d  weighted loss per coordinate %.4f  (%.0f s)\n", ep, tot / nb / sum(wts), time() - t0)
        flush(stdout)
    end
    JLD2.save(modelfile(), Dict("ps" => CDEV(psd), "st" => st, "label_weight" => label_weight, "epochs" => epochs,
                                "blocks" => BLOCKS, "joint" => joint_fraction, "clean" => clean,
                                "held_loss" => held_loss))
    return psd
end

train(; epochs = parse(Int, get(ARGS, 1, "30")), label_weight = parse(Float64, get(ENV, "LABEL_WEIGHT", "1")))
