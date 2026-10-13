# The other directions of the same model: the label clamped and the image free (a mode and
# samples), half an image clamped with the other half and the label free (inpainting and
# classifying at once), and the ambiguous digits of the classification run.
#
#     julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/generate.jl [stage]
#
# stages: generate (samples, then the modes reached from them), inpaint, ambiguous, all. Figures go to docs/src/assets/mnist_*.png,
# numbers to stdout; recorded in `MNIST in Both Directions.md` §6.

include("common.jl")
using CairoMakie

const ASSETS = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "docs", "src", "assets"))

img(z) = reverse(permutedims(reshape(clamp.((z[IMAGE] .+ 1) ./ 2, 0, 1), 28, 28)); dims = 1)

function grid!(fig, pos, Zs; title = "", labels = nothing)
    g = fig[pos...] = GridLayout()
    for (k, z) in enumerate(Zs)
        ax = Axis(g[1, k]; aspect = DataAspect(), title = labels === nothing ? "" : labels[k], titlesize = 11)
        hidedecorations!(ax); hidespines!(ax)
        heatmap!(ax, permutedims(img(z)); colormap = :grays, colorrange = (0, 1))
    end
    title == "" || Label(g[0, :], title; tellwidth = false, fontsize = 13)
    colgap!(g, 2)
    return g
end

function relation(levels; samples = 2)
    nodes = query_nodes(levels; samples)
    ps, st = load_model()
    m = ImplicitDiffusion(predictor(network()), nodes; λ = density_lambda(SCHED, nodes), batched = true)
    return m, GDEV(ps), st
end

# the label clamped, the image drawn by the reverse diffusion
function samples(; n = 8, steps = 200)
    m, ps, st = relation(range(0.002, 0.05; length = 4))
    ρ = [zeros(NX); fill(Inf, NY)]
    t = @elapsed rows = map(0:9) do k
        Z, _ = implicit_sample(m, [zeros(NX); codes()[:, k + 1]], ρ, ps, st; n, steps, rng = Xoshiro(k))
        [Z[:, j] for j in 1:n]
    end
    @printf("samples: %d per class, %d steps, %.1f s per class\n", n, steps, t / 10)
    return rows
end

# the label clamped, the 784 pixels free: the mode of the basin a sample lies in, solved
# matrix-free (KrylovSolve). From a blank image the solve is not stable: the descent steps do not
# reach a basin in 100 iterations (residual 170 for the digit 0).
function prototypes(rows; levels = range(0.002, 0.05; length = 4))
    m, ps, st = relation(levels)
    ρ = [zeros(NX); fill(Inf, NY)]
    Zs = []
    for k in 0:9
        z₀ = [rows[k + 1][1][IMAGE]; codes()[:, k + 1]]
        t = @elapsed sol, _ = implicit_infer(m, z₀, ρ, ps, st; linsolve = KrylovSolve(), maxiters = 50, tol = 1e-6)
        push!(Zs, sol.z)
        @printf("prototype %d: converged %s stable %s, %d iterations, residual %.1e, moved %.2f from the sample, %.1f s\n",
                k, sol.converged, sol.stable, sol.iters, sol.residual, norm(sol.z[IMAGE] .- z₀[IMAGE]) / sqrt(NX), t)
        flush(stdout)
    end
    return Zs
end

# the top half clamped; the bottom half and the label free. A draw by clamped sampling (the
# label's frequencies are p(y | top half)), then the answer of the joint solve started there.
function inpaint(; n = 100, steps = 200, levels = range(0.05, 0.2; length = 4))   # the classification levels
    Z, y = joint(:test)
    m, ps, st = relation(levels)
    ms, _, _ = relation(range(0.002, 0.05; length = 4))
    top = vec([r ≤ 14 for c in 1:28, r in 1:28])               # rows 1–14 (MLDatasets stores 28×28 as width × height)
    ρ = [ifelse.(top, Inf, 0.0); zeros(NY)]
    ok_s, ok_a, conv, shown, ts, ta = 0, 0, 0, [], 0.0, 0.0
    for b in 1:n
        z₀ = [ifelse.(top, Z[IMAGE, b], 0.0); zeros(NY)]
        ts += @elapsed S, _ = implicit_sample(ms, z₀, ρ, ps, st; n = 1, steps, rng = Xoshiro(b))
        ta += @elapsed sol, _ = implicit_infer(m, z₀, ρ, ps, st; z_init = S[:, 1], linsolve = KrylovSolve(), maxiters = 50, tol = 1e-6)
        ŷs, ŷa = argmax(S[LABEL, 1]) - 1, argmax(sol.z[LABEL]) - 1
        ok_s += ŷs == y[b]; ok_a += ŷa == y[b]; conv += sol.converged && sol.stable
        b ≤ 8 && push!(shown, (Z[:, b], [ifelse.(top, Z[IMAGE, b], -1.0); zeros(NY)], S[:, 1], sol.z, ŷs, ŷa))
    end
    @printf("inpainting (top half clamped, %d test images): label of the draw correct %.3f (%.1f s per draw); label of the solved answer correct %.3f, converged and stable %.3f (%.1f s per solve)\n",
            n, ok_s / n, ts / n, ok_a / n, conv / n, ta / n)
    return shown
end

# every classification query had a single root; ambiguity shows in where it lies: the images whose
# root's label coordinates are most evenly split between two classes
function ambiguous(; k = 8)
    d = JLD2.load(joinpath(OUT, "classify_test.jld2"))
    Y, lab = d["Y1"], d["labels"]
    Z, _ = joint(:test)
    top2(v) = (o = sortperm(v; rev = true); (o[1], o[2]))
    gap = [(o = top2(Y[:, b]); Y[o[1], b] - Y[o[2], b]) for b in axes(Y, 2)]
    idx = sortperm(gap)[1:k]
    for b in idx
        o = top2(Y[:, b])
        @printf("test image %d (label %d): root label %d: %.2f, %d: %.2f\n", b, lab[b], o[1] - 1, Y[o[1], b], o[2] - 1, Y[o[2], b])
    end
    return [(Z[:, b], Y[:, b], lab[b]) for b in idx]
end

stage = get(ARGS, 1, "all")
if stage in ("generate", "all")
    rows = samples()
    fig = Figure(size = (720, 900))
    for (k, r) in enumerate(rows); grid!(fig, (k, 1), r); end
    save(joinpath(ASSETS, "mnist_samples.png"), fig)
    Zs = prototypes(rows)
    fig = Figure(size = (900, 220))
    grid!(fig, (1, 1), [r[1] for r in rows]; title = "a sample of each class", labels = ["$k" for k in 0:9])
    grid!(fig, (2, 1), Zs; title = "the mode of its basin (the image solved, the label clamped)")
    save(joinpath(ASSETS, "mnist_prototypes.png"), fig)
end
if stage in ("inpaint", "all")
    shown = inpaint()
    fig = Figure(size = (900, 480))
    grid!(fig, (1, 1), [s[1] for s in shown]; title = "test images")
    grid!(fig, (2, 1), [s[2] for s in shown]; title = "the query: the top half clamped, the rest and the label free")
    grid!(fig, (3, 1), [s[3] for s in shown]; title = "a draw", labels = ["$(s[5])" for s in shown])
    grid!(fig, (4, 1), [s[4] for s in shown]; title = "the answer solved from it", labels = ["$(s[6])" for s in shown])
    save(joinpath(ASSETS, "mnist_inpaint.png"), fig)
end
if stage in ("ambiguous", "all") && isfile(joinpath(OUT, "classify_test.jld2"))
    am = ambiguous()
    labels = [(o = sortperm(v; rev = true); @sprintf("%d: %.2f · %d: %.2f (is %d)", o[1] - 1, v[o[1]], o[2] - 1, v[o[2]], l)) for (_, v, l) in am]
    fig = Figure(size = (1300, 170)); grid!(fig, (1, 1), [a[1] for a in am]; labels)
    save(joinpath(ASSETS, "mnist_ambiguous.png"), fig)
end
