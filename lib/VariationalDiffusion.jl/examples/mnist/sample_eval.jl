# How well the label-clamped sampler generates: 20 draws per class, scored by a softmax MLP
# trained on the real data (the fraction it recognises as the clamped label, and its mean
# confidence), for several sampler settings.
#
#     MODEL=… KIND=energy|eps julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/sample_eval.jl

push!(ARGS, "none")
include("generate.jl")

function judge(; epochs = 5, rng = Xoshiro(0))
    Ztr, ytr = joint(:train)
    net = Chain(Dense(NX => WIDTH, swish), Dense(WIDTH => WIDTH, swish), Dense(WIDTH => NY))
    ps, st = Lux.setup(rng, net); ps = GDEV(ps)
    opt = Optimisers.setup(Optimisers.Adam(1e-3), ps)
    X = Float32.(Ztr[IMAGE, :])
    for _ in 1:epochs, idx in Iterators.partition(randperm(rng, size(X, 2)), 128)
        x, yy = GDEV(X[:, idx]), GDEV(Float32.(Matrix(1.0I, NY, NY)[:, ytr[idx] .+ 1]))
        g = only(Zygote.gradient(p -> (l = first(net(x, p, st)); -sum(yy .* (l .- log.(sum(exp.(l); dims = 1)))) / size(x, 2)), ps))
        opt, ps = Optimisers.update!(opt, ps, g)
    end
    return Z -> (l = CDEV(first(net(GDEV(Float32.(clamp.(Z[IMAGE, :], -1, 1))), ps, st)));
                 P = exp.(l .- maximum(l; dims = 1)); P ./ sum(P; dims = 1))
end

softmax_of = judge()
m, ps, st = relation(range(0.002, 0.05; length = 4))
ρ = [zeros(NX); fill(Inf, NY)]
configs = ["plain" => (;), "resample 3" => (resample = 3,),
           "Langevin ×2" => (corrector = :langevin, corrector_steps = 2, corrector_step = 0.1),
           "resample 3 + Langevin ×2" => (resample = 3, corrector = :langevin, corrector_steps = 2, corrector_step = 0.1)]
get(ENV, "SAMPLE_EVAL", "1") == "1" && for (name, kw) in configs
    t = @elapsed begin
        hits, conf = 0, 0.0
        for k in 0:9
            Z, _ = implicit_sample(m, [zeros(NX); codes()[:, k + 1]], ρ, ps, st; n = 20, steps = 200, rng = Xoshiro(k), kw...)
            P = softmax_of(Z)
            hits += count(==(k), vec(map(argmax, eachcol(P))) .- 1); conf += sum(P[k + 1, :])
            k == 0 && JLD2.save(joinpath(OUT, "samples_$(get(ENV, "MODEL", "energy"))_$(replace(name, " " => "_")).jld2"), Dict("Z" => Z))
        end
    end
    @printf("%-8s %-26s recognised as the clamped label %.3f, mean probability %.3f  (%.1f s per class of 20)\n",
            get(ENV, "MODEL", "energy"), name, hits / 200, conf / 200, t / 10)
    flush(stdout)
end
