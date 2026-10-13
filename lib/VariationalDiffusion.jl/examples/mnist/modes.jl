# Mode finding with 784 free coordinates: the label clamped (held clean with noise blocks), the
# image solved matrix-free (KrylovSolve) from a blank image and from a sample: Newton-or-descent
# without and with a continuation in the noise level, and the trust-region method (METHODS, env,
# default both). Each mode is scored by the softmax judge of `sample_eval.jl`.
#
#     MODEL=… BLOCKS=… julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/modes.jl

ENV["SAMPLE_EVAL"] = "0"
include("sample_eval.jl")

ρm = [zeros(NX); fill(Inf, NY)]
mm, _, _ = relation(range(0.002, 0.05; length = 4); ρ = ρm)
runs = [(start, cont, meth) for meth in Symbol.(split(get(ENV, "METHODS", "newton,trust_region"), ","))
        for start in ("blank", "sample") for cont in (meth === :newton && get(ENV, "CONTINUATION", "1") == "1" ? ((), (8, 3)) : ((),))]
for (start, cont, meth) in runs
    ok, hits, res, its, tt = 0, 0, Float64[], 0, 0.0
    for k in 0:9
        z₀ = [zeros(NX); codes()[:, k + 1]]
        zi = if start == "blank"
            z₀
        else
            S, _ = implicit_sample(mm, z₀, ρm, ps, st; n = 1, steps = 100, rng = Xoshiro(k), sample_kw()...)
            S[:, 1]
        end
        tt += @elapsed sol, _ = implicit_infer(mm, z₀, ρm, ps, st; z_init = zi, linsolve = KrylovSolve(), maxiters = parse(Int, get(ENV, "MAXITERS", "50")),
                                               continuation = cont, method = meth)
        ok += sol.converged && sol.stable
        hits += argmax(softmax_of(reshape(sol.z, :, 1))[:, 1]) - 1 == k
        push!(res, sol.residual); its += sol.iters
    end
    @printf("%-12s %-6s continuation %-7s converged and stable %d/10, judged as the clamped label %d/10, median residual %.1e, %.1f iterations, %.1f s per mode\n",
            meth, start, string(cont), ok, hits, median(res), its / 10, tt / 10)
    flush(stdout)
end
