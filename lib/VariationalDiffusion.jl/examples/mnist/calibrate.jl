# Class probabilities from single roots: every classification query of the block model has (as on
# the joint model) one stable root, so the Laplace mixture gives probabilities 0 or 1. Read the
# root's position instead: `categorical_answer(...; temperature = τ)`, the posterior of the code
# given the root's Gaussian, with τ (and a covariance scale κ) fitted on the tuning images (9001–10000) and scored on test
# images 1–N (env, default 3000): accuracy, negative log-likelihood and calibration error.
#
#     MODEL=… BLOCKS=… julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/calibrate.jl

push!(ARGS, "none")
include("classify.jl")

function mixtures(m, ps, st, X; chunk = 100)
    out = Any[]
    for c in Iterators.partition(axes(X, 2), chunk)
        ans, _ = implicit_classify(m, vcat(X[:, c], zeros(NY, length(c))), RHO, ps, st; label = LABEL, classes = codes(),
                                   maxiters = 50)
        append!(out, [a.mixture for a in ans])
    end
    return out
end

function scores(name, mixes, lab; temperature = nothing, cov_scale = 1)
    P = reduce(hcat, [(a = categorical_answer(x, codes(); temperature, cov_scale); a.belief === nothing ? fill(0.1, NY) : probabilities(a.belief))
                      for x in mixes])
    acc = mean(vec(map(argmax, eachcol(P))) .- 1 .== lab)
    nll = -mean(log(max(P[l + 1, b], 1e-12)) for (b, l) in enumerate(lab))
    @printf("%-34s accuracy %.4f  NLL %.3f  ECE %.4f  mean max-probability %.3f\n", name, acc, nll, ece(P, lab),
            mean(maximum(P; dims = 1)))
    flush(stdout)
end

levels = eval(Meta.parse(get(ENV, "LEVELS", "range(0.05, 0.2; length = 4)")))
Z, y = joint(:test)
ntest = parse(Int, get(ENV, "N", "3000"))
# the solved answers are cached (means, covariances, weights), so the reading can be rerun
cache = joinpath(OUT, "answers_$(get(ENV, "MODEL", "energy"))_$(hash(levels))_$ntest.jld2")
pack(xs) = [([belief_mean(c) for c in x.components], [Matrix(belief_cov(c)) for c in x.components], mixture_weights(x)) for x in xs]
unpack(ps) = [LenticulumCore.MixtureBelief([LenticulumCore.Gaussian(μ, Σ) for (μ, Σ) in zip(M, S)], w) for (M, S, w) in ps]
if isfile(cache)
    d = JLD2.load(cache); tune_mix, test_mix = unpack(d["tune"]), unpack(d["test"])
else
    m, ps, st = model(levels)
    t = @elapsed tune_mix = mixtures(m, ps, st, Z[IMAGE, 9001:10000])
    @printf("tuning answers: %.1f ms per query, %.3f with one root\n", t, mean(length(x.components) == 1 for x in tune_mix))
    test_mix = mixtures(m, ps, st, Z[IMAGE, 1:ntest])
    JLD2.save(cache, Dict("tune" => pack(tune_mix), "test" => pack(test_mix)))
end
c1 = calibrate_temperature(tune_mix, codes(), y[9001:10000] .+ 1; scale_bounds = (1, 1))
c2 = calibrate_temperature(tune_mix, codes(), y[9001:10000] .+ 1)
@printf("calibrated: τ alone %.3f (tuning NLL %.3f); τ %.3f with κ %.4f (tuning NLL %.3f)\n",
        c1.temperature, c1.nll, c2.temperature, c2.cov_scale, c2.nll)
scores("Laplace weights (hard assignment)", test_mix, y[1:ntest])
scores("code posterior, τ alone", test_mix, y[1:ntest]; temperature = c1.temperature)
scores("code posterior, τ and κ", test_mix, y[1:ntest]; temperature = c2.temperature, cov_scale = c2.cov_scale)
