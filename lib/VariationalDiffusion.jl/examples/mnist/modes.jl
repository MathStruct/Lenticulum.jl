# Mode finding with 784 free coordinates: the label clamped (held clean with noise blocks), the
# image solved matrix-free (KrylovSolve) from a blank image and from a sample, without and with a
# continuation in the noise level. Each mode is scored by the softmax judge of `sample_eval.jl`.
#
#     MODEL=… BLOCKS=… julia --project=lib/VariationalDiffusion.jl/examples/mnist lib/VariationalDiffusion.jl/examples/mnist/modes.jl

ENV["SAMPLE_EVAL"] = "0"
include("sample_eval.jl")

ρm = [zeros(NX); fill(Inf, NY)]
mm, _, _ = relation(range(0.002, 0.05; length = 4); ρ = ρm)
for (start, cont) in (("blank", ()), ("blank", (8, 3)), ("sample", ()), ("sample", (8, 3)))
    ok, hits, res, its, tt = 0, 0, Float64[], 0, 0.0
    for k in 0:9
        z₀ = [zeros(NX); codes()[:, k + 1]]
        zi = if start == "blank"
            z₀
        else
            S, _ = implicit_sample(mm, z₀, ρm, ps, st; n = 1, steps = 100, rng = Xoshiro(k), sample_kw()...)
            S[:, 1]
        end
        tt += @elapsed sol, _ = implicit_infer(mm, z₀, ρm, ps, st; z_init = zi, linsolve = KrylovSolve(), maxiters = 50,
                                               continuation = cont)
        ok += sol.converged && sol.stable
        hits += argmax(softmax_of(reshape(sol.z, :, 1))[:, 1]) - 1 == k
        push!(res, sol.residual); its += sol.iters
    end
    @printf("%-6s continuation %-7s converged and stable %d/10, judged as the clamped label %d/10, median residual %.1e, %.1f iterations, %.1f s per mode\n",
            start, string(cont), ok, hits, median(res), its / 10, tt / 10)
    flush(stdout)
end
