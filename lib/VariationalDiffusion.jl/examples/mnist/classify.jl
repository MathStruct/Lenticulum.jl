# Classification as a query: the image clamped, the ten label coordinates free.
#
#     julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/classify.jl [stage]
#
# stages:
#   tune      the field's noise levels, chosen on test images 9001–10000
#   test      accuracy, calibration and cost on test images 1–9000 with the chosen levels
#   baselines a softmax MLP of the same width, and the diffusion classifier of Li et al. (2023)
#             with the same energy network (Bayes' rule over per-class denoising errors)
# Results go to ~/.cache/lenticulum/mnist/ and are recorded in `MNIST in Both Directions.md` §6.

include("common.jl")

const RHO = [fill(Inf, NX); zeros(NY)]

# every query of the columns of X, in chunks, started from every class code: the categorical
# answer (nothing when the roots are all unassigned), the label coordinates of the heaviest root,
# the unassigned mass and the number of distinct roots.
# tol: the field is computed in Float32 on the GPU, so residuals stop near 1e-6 (relative to |z| ≈ 28)
function classify(m, ps, st, X; chunk = max(CHUNK ÷ 10, 1), maxiters = 50, tol = 1e-6)
    B = size(X, 2)
    pred = fill(-1, B); P = fill(NaN, NY, B); Y = zeros(NY, B); un = zeros(B); nroots = zeros(Int, B)
    for c in Iterators.partition(1:B, chunk)
        Z₀ = vcat(X[:, c], zeros(NY, length(c)))
        ans, _ = implicit_classify(m, Z₀, RHO, ps, st; label = LABEL, classes = codes(), maxiters, tol)
        for (j, b) in enumerate(c)
            a = ans[j]
            un[b] = a.unassigned; nroots[b] = length(a.mixture.components)
            Y[:, b] .= belief_mean(a.mixture.components[argmax(a.mixture.logw)])
            a.belief === nothing && continue
            P[:, b] .= probabilities(a.belief); pred[b] = argmax(P[:, b]) - 1
        end
    end
    return (pred = pred, P = P, Y = Y, unassigned = un, nroots = nroots)
end

# one solve per query, from the label 0 (all class codes equally far)
function classify_single(m, ps, st, X; chunk = CHUNK, maxiters = 50, tol = 1e-6)
    Y = zeros(NY, size(X, 2)); ok = trues(size(X, 2))
    for c in Iterators.partition(axes(X, 2), chunk)
        sols, _ = implicit_infer_many(m, vcat(X[:, c], zeros(NY, length(c))), RHO, ps, st; maxiters, tol)
        for (j, b) in enumerate(c)
            Y[:, b] .= sols[j].z[LABEL]; ok[b] = sols[j].converged && sols[j].stable
        end
    end
    return (Y = Y, ok = ok)
end

argmaxclass(Y) = vec(map(argmax, eachcol(Y))) .- 1

function report(name, r, lab, t)
    am = argmaxclass(r.Y) .== lab
    assigned = r.pred .>= 0
    @printf("%-18s argmax of the root %.4f | assigned %.3f, accurate when assigned %.4f | >1 root %.3f | %.1f ms per query\n",
            name, mean(am), mean(assigned), mean(r.pred[assigned] .== lab[assigned]), mean(r.nroots .> 1), 1000t / length(lab))
    flush(stdout)
end

# expected calibration error over 15 confidence bins
function ece(P, labels; bins = 15)
    conf = vec(maximum(P; dims = 1)); hit = (vec(map(argmax, eachcol(P))) .- 1) .== labels
    e = 0.0
    for k in 1:bins
        sel = (conf .> (k - 1) / bins) .& (conf .≤ k / bins)
        any(sel) && (e += count(sel) / length(conf) * abs(mean(hit[sel]) - mean(conf[sel])))
    end
    return e
end

function model(levels; samples = 2)
    nodes = query_nodes(levels; samples)
    ps, st = load_model()
    pred = predictor(network())
    return ImplicitDiffusion(pred, nodes; λ = density_lambda(SCHED, nodes), batched = true,
                             levels = query_levels(pred, RHO)), GDEV(ps), st
end

const LEVELSETS = [
    "low (0.002–0.05)" => range(0.002, 0.05; length = 4),
    "mid (0.05–0.2)" => range(0.05, 0.2; length = 4),
    "high (0.1–0.4)" => range(0.1, 0.4; length = 4),
    "wide (0.02–0.5)" => range(0.02, 0.5; length = 6),
]

function tune(; n = 1000)
    Z, y = joint(:test)
    X, lab = Z[IMAGE, 9001:(9000 + n)], y[9001:(9000 + n)]
    for (name, lv) in LEVELSETS
        m, ps, st = model(lv)
        t = @elapsed r = classify(m, ps, st, X)
        report(name, r, lab, t)
        t1 = @elapsed r1 = classify_single(m, ps, st, X)
        @printf("%-18s one start: argmax %.4f, converged and stable %.3f, %.1f ms per query\n", "",
                mean(argmaxclass(r1.Y) .== lab), mean(r1.ok), 1000t1 / n)
    end
end

function test(levels; n = 9000)
    Z, y = joint(:test)
    X, lab = Z[IMAGE, 1:n], y[1:n]
    m, ps, st = model(levels)
    t = @elapsed r = classify(m, ps, st, X)
    t1 = @elapsed r1 = classify_single(m, ps, st, X)
    JLD2.save(joinpath(OUT, "classify_test.jld2"), Dict("P" => r.P, "Y" => r.Y, "pred" => r.pred, "labels" => lab,
              "unassigned" => r.unassigned, "nroots" => r.nroots, "seconds" => t, "Y1" => r1.Y, "seconds1" => t1))
    report("test, ten starts", r, lab, t)
    @printf("test, one start: argmax %.4f, converged and stable %.4f, %.1f ms per query\n",
            mean(argmaxclass(r1.Y) .== lab), mean(r1.ok), 1000t1 / n)
end

# --- baselines ----------------------------------------------------------------------------------

function softmax_baseline(; epochs = 10, rng = Xoshiro(0))
    Ztr, ytr = joint(:train); Zte, yte = joint(:test)
    net = Chain(Dense(NX => WIDTH, swish), Dense(WIDTH => WIDTH, swish), Dense(WIDTH => NY))
    ps, st = Lux.setup(rng, net); ps = GDEV(ps)
    opt = Optimisers.setup(Optimisers.Adam(1e-3), ps)
    Xtr = Float32.(Ztr[IMAGE, :])
    for ep in 1:epochs, idx in Iterators.partition(randperm(rng, size(Xtr, 2)), 128)
        x, yy = GDEV(Xtr[:, idx]), GDEV(Float32.(codes()[:, ytr[idx] .+ 1] ./ C))
        g = only(Zygote.gradient(p -> (l = first(net(x, p, st)); -sum(yy .* (l .- log.(sum(exp.(l); dims = 1)))) / size(x, 2)), ps))
        opt, ps = Optimisers.update!(opt, ps, g)
    end
    l = CDEV(first(net(GDEV(Float32.(Zte[IMAGE, 1:9000])), ps, st)))
    P = exp.(l .- maximum(l; dims = 1)); P ./= sum(P; dims = 1)
    @printf("softmax MLP (%d epochs): accuracy %.4f  ECE %.4f\n", epochs, mean(vec(map(argmax, eachcol(P))) .- 1 .== yte[1:9000]),
            ece(Float64.(P), yte[1:9000]))
end

# Li et al.: per class, the mean denoising error over K shared (t, ε) draws; p ∝ exp(-K̄·err/τ)
function diffusion_classifier(; n = 9000, K = 32, rng = Xoshiro(3))
    Z, y = joint(:test)
    ps, st = load_model(); ps = GDEV(ps)
    pred = predictor(network())
    ts = SCHED.tmin .+ (1 - SCHED.tmin) .* rand(rng, K)
    E = randn(rng, N, K)
    err = zeros(NY, n)
    t0 = time()
    for c in Iterators.partition(1:n, 100), k in 1:NY
        Zc = vcat(Z[IMAGE, c], repeat(codes()[:, k], 1, length(c)))
        for j in 1:K
            Xt = alpha(SCHED, ts[j]) .* Zc .+ sigma(SCHED, ts[j]) .* E[:, j]
            ε̂, _ = epsilon(pred, Xt, ts[j], ps, st)
            err[k, c] .+= vec(sum(abs2, ε̂ .- E[:, j]; dims = 1)) ./ K
        end
    end
    t = time() - t0
    acc = mean(vec(map(argmin, eachcol(err))) .- 1 .== y[1:n])
    @printf("diffusion classifier (K = %d draws, 10 classes): accuracy %.4f  (%.1f ms per query, %d ε passes)\n",
            K, acc, 1000t / n, 10K)
end

stage = get(ARGS, 1, "tune")
stage == "tune" && tune()
stage == "test" && test(eval(Meta.parse(get(ARGS, 2, "range(0.05, 0.2; length = 4)"))))
stage == "baselines" && (softmax_baseline(); diffusion_classifier())
