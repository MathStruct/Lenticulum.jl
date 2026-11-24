# julia --project=bench bench/minres_energy.jl
@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

using VariationalDiffusion, Lux, Random, LinearAlgebra, Printf
using ADTypes: AutoZygote
using DifferentiationInterface, Zygote

const VD = VariationalDiffusion
const IS = VD.ImplicitSolvers
const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
const FREQ = [1.0, 2.0]
const RTOL = 1e-8
const MAX_LINEAR_STEPS = 2000
const MAX_OUTER_STEPS = 8

function time_input(x, t)
    phases = (2π .* t) .* FREQ
    features = vcat(sin.(phases), cos.(phases))
    return vcat(x, features .* ones(eltype(x), 1, size(x, 2)))
end

function setup(n, width; seed = 0)
    net = Chain(Dense(n + 4 => width, swish), Dense(width => width, swish), Dense(width => 1))
    ps0, st = Lux.setup(Xoshiro(seed), net)
    pred = NoisePredictor(EnergyNetwork(net), VPSDE(); input = time_input, ad = AutoZygote())
    m = ImplicitDiffusion(pred, noisefree_nodes(n); batched = true)
    z = randn(Xoshiro(seed + 1), n)
    b = randn(Xoshiro(seed + 2), n)
    return (; m, z, b, ps = Lux.f64(ps0), st)
end

function linear_benchmark(problem, ρval)
    (; m, z, b, ps, st) = problem
    A = VD._free_operators(m, z, fill(ρval, length(z)), ps, st).Jt
    tol = IS.Tolerance(atol = 0, rtol = RTOL)
    for (method, solver) in (("GMRES(40)", IS.GMRES(restart = 40, maxiters = MAX_LINEAR_STEPS)),
                             ("MINRES", IS.MINRES(maxiters = MAX_LINEAR_STEPS)))
        t = @elapsed x, report = IS.linsolve(A, b, solver; tol)
        relative = norm(b - A(x)) / norm(b)
        @printf("| %d | %.0f | %s | %s | %d | %d | %.3e | %.3f |\n",
            length(z), ρval, method, report.reason, report.iterations, report.nfe, relative, t)
    end
end

function inference_benchmark(problem, ρval; print_rows = true)
    (; m, z, b, ps, st) = problem
    ρ = fill(ρval, length(z))
    algorithms = (KrylovSolve(method = :gmres, maxiters = MAX_LINEAR_STEPS),
                  KrylovSolve(method = :auto, maxiters = MAX_LINEAR_STEPS))
    common_root = nothing
    for method in (:trust_region, :newton), alg in algorithms
        seconds = @elapsed sol, _ = implicit_infer(m, z, ρ, ps, st;
            method, maxiters = MAX_OUTER_STEPS, linsolve = alg)
        if print_rows
            @printf("| %d | %.0f | %s | %s | %d | %s | %.3f |\n",
                length(z), ρval, method, alg.method, sol.iters, sol.converged, seconds)
        end
        if common_root === nothing && sol.converged
            common_root = sol
        end
    end

    # Use the same converged state for both pullbacks. The API rejects non-roots.
    for alg in algorithms
        if common_root === nothing
            print_rows && @printf("| %d | %.0f | pullback | %s | — | no root | — |\n",
                length(z), ρval, alg.method)
        else
            seconds = @elapsed implicit_pullback(m, common_root, z, ρ, b, ps, st; linsolve = alg)
            print_rows && @printf("| %d | %.0f | pullback | %s | — | returned | %.3f |\n",
                length(z), ρval, alg.method, seconds)
        end
    end
end

function main()
    BLAS.set_num_threads(1)
    setup_sizes = SMOKE ? ((16, 32),) : ((64, 256), (128, 256), (256, 512))
    problems = map(size -> setup(size...), setup_sizes)
    warm = setup(16, 32)
    apply = VD._free_operators(warm.m, warm.z, ones(16), warm.ps, warm.st).Jt
    IS.linsolve(apply, warm.b, IS.GMRES())
    IS.linsolve(apply, warm.b, IS.MINRES())
    inference_benchmark(warm, 1.0; print_rows = false)

    println("| n | ρ | linear method | reason | steps | products | true relative residual | seconds |")
    println("|---:|---:|---|---|---:|---:|---:|---:|")
    for problem in problems
        linear_benchmark(problem, 0.0)
        (length(problem.z) == 128 || SMOKE) && linear_benchmark(problem, 1.0)
        flush(stdout)
    end

    println("\n| n | ρ | operation | Krylov method | iterations | result | seconds |")
    println("|---:|---:|---|---|---:|---|---:|")
    for problem in problems, ρval in (0.0, 1.0)
        inference_benchmark(problem, ρval)
        flush(stdout)
    end
end

main()
