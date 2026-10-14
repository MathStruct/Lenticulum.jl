# The cost model against measurement: time per classification query of root finding (one start,
# Newton-or-descent and trust region, batches of 1000) and of the diffusion classifier (10
# classes × M draws, chunks of 100; DRAWS=32,64,… env), measured on the tuning images 9001–10000, beside the
# prediction from the per-column cost of each network pass (`pass_costs`, `root_cost`,
# `classifier_cost`). Then where the model puts the break-even.
#
#     MODEL=… BLOCKS=… julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/cost.jl

push!(ARGS, "none")
include("classify.jl")

Z, y = joint(:test)
X, lab = Z[IMAGE, 9001:10000], y[9001:10000]
B = size(X, 2)
levels = range(0.05, 0.2; length = 4)
m, ps, st = model(levels)
pred = m.predictor
τ = BLOCKS ? [SCHED.tmin, 0.1] : 0.1            # the field's times at a node (image held clean with blocks)
npl = count(==(first(m.nodes.t)), m.nodes.t)    # nodes per noise level
Zq = vcat(X, zeros(NY, B))

# per-column costs at the widths the solvers use (queries in chunks of CHUNK): the field CHUNK·npl columns, the Jacobian block
# |F| times that; the diffusion classifier calls chunks of 100 columns
u = pass_costs(pred, repeat(Zq[:, 1:CHUNK], 1, npl), τ, ps, st; directions = NY)
ud = pass_costs(pred, Zq[:, 1:100], 0.1, ps, st)
@printf("per column: ε %.2f µs, energy %.2f µs, derivative %.2f µs (field width %d, derivative width %d); ε at width 100 %.2f µs\n",
        1e6u.epsilon, 1e6u.energy, 1e6u.derivative, CHUNK * npl, CHUNK * npl * NY, 1e6ud.epsilon)
flush(stdout)

for method in (:newton, :trust_region)
    implicit_infer_many(m, Zq[:, 1:10], RHO, ps, st; maxiters = 50, method)        # compile
    t = @elapsed sols = reduce(vcat, [first(implicit_infer_many(m, Zq[:, c], RHO, ps, st; maxiters = 50, method))
                                      for c in Iterators.partition(1:B, CHUNK)])
    its = mean(s.iters for s in sols)
    acc = mean(argmaxclass(reduce(hcat, [s.z[LABEL] for s in sols])) .== lab)
    @printf("root finding, %-12s measured %.1f ms per query, predicted %.1f ms (%.2f iterations, accuracy %.4f)\n",
            method, 1000t / B, 1000root_cost(m, NY, u; iterations = its, method), its, acc)
    flush(stdout)
end

function dc(Zc, lab; K = 32, rng = Xoshiro(3))
    ts = SCHED.tmin .+ (1 - SCHED.tmin) .* rand(rng, K)
    E = randn(rng, N, K)
    err = zeros(NY, size(Zc, 2))
    for c in Iterators.partition(axes(Zc, 2), 100), k in 1:NY
        Zk = vcat(Zc[IMAGE, c], repeat(codes()[:, k], 1, length(c)))
        for j in 1:K
            ε̂, _ = epsilon(pred, alpha(SCHED, ts[j]) .* Zk .+ sigma(SCHED, ts[j]) .* E[:, j], ts[j], ps, st)
            err[k, c] .+= vec(sum(abs2, ε̂ .- E[:, j]; dims = 1)) ./ K
        end
    end
    return mean(vec(map(argmin, eachcol(err))) .- 1 .== lab)
end
dc(Zq[:, 1:100], lab[1:100]; K = 2)                                                   # compile
for M in parse.(Int, split(get(ENV, "DRAWS", "32"), ","))
    t = @elapsed acc = dc(Zq, lab; K = M)
    @printf("diffusion classifier (10 × %d): measured %.1f ms per query, predicted %.1f ms (accuracy %.4f)\n",
            M, 1000t / B, 1000classifier_cost(NY, M, ud), acc)
    flush(stdout)
end
