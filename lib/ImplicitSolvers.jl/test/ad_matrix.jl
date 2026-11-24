module SolverADMatrix

using ImplicitSolvers, LinearAlgebra, Test
import Krylov

include("../../LenticulumCore.jl/test/support/ad_matrix.jl")
using .Q5ADMatrix

const SLOW_BACKENDS = ("Zygote", "ForwardDiff", "Enzyme", "Mooncake")

@testset verbose=true "ImplicitSolvers AD backend matrix" begin
    # The SPD system from the ForwardDiff tests in krylovjl.jl. Perturb the
    # diagonal and the right-hand side together to exercise both arguments.
    A = [4.0 1.0; 1.0 3.0]
    p = [0.1, -0.2, 1.0, 2.0]
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    for alg in (DenseSolve(), GMRES(), MINRES(), KrylovJL())
        name = "linsolve/$(nameof(typeof(alg)))"
        if alg isa KrylovJL && Base.get_extension(ImplicitSolvers, :ImplicitSolversKrylovExt) === nothing
            skip_row(name; slow=true)
            continue
        end
        loss(v) = sum(first(linsolve(A + Diagonal(v[1:2]), v[3:4], alg; tol)))
        # Q5 baseline: Zygote ErrorException, "Mutating arrays is not supported"
        # (copyto! on DenseSolve/GMRES) or "Can't differentiate gc_preserve_end" (KrylovJL).
        # Q5 baseline: Enzyme (both modes) EnzymeRuntimeActivityError,
        # "Detected potential need for runtime activity."
        # Q5 baseline: Mooncake KrylovJL MooncakeRuleCompilationError,
        # "Mooncake failed to differentiate the following method:" (unsupported try/catch in _attempt).
        broken = alg isa DenseSolve ? ("Zygote", "Enzyme", "Enzyme.Forward") :
                 alg isa GMRES ? ("Zygote", "Enzyme") :
                 alg isa MINRES ? ("Enzyme",) : ("Zygote", "Enzyme", "Mooncake")
        # KrylovJL and the column driver account for the tail of the measured
        # 5m13s all-backend run.
        slow = alg isa KrylovJL ? SLOW_BACKENDS : ()
        gradient_row(name, loss, p; broken, forward=alg isa DenseSolve, slow)
    end

    # The linear residual from roots.jl, with both RHS coordinates active.
    B = [0.5 0.125; 0.125 0.75]
    for alg in (Picard(damping=0.5), Broyden(init_scale=0.5))
        loss(v) = sum(first(solve_root(u -> B * u .- v, zeros(2), alg; tol)))
        # Q5 baseline: Zygote Broyden ErrorException, "Mutating arrays is not
        # supported -- called copyto!(Matrix{Float64}, ...)" in the solver loop.
        # Q5 baseline: Enzyme EnzymeRuntimeActivityError, "Detected potential
        # need for runtime activity."
        broken = alg isa Picard ? ("Enzyme",) : ("Zygote", "Enzyme")
        gradient_row("solve_root/$(nameof(typeof(alg)))", loss, p[3:4]; broken)
    end

    # The quadratic problem from descent.jl. The starting column must have
    # the input element type because the Newton driver writes into it.
    H = [2.0 1.0; 1.0 3.0]
    function column_loss(v)
        problem = (
            residual = (U, cols) -> H * (U .- v),
            scale = (U, cols) -> map(norm, eachcol(U)),
            system = (U, cols) -> ([true], (r -> H \ r,)),
        )
        return sum(first(solve_columns(problem, reshape(zero(v), :, 1), NewtonDescent();
                                       tol, maxiters=4)))
    end
    # Q5 baseline: Zygote ErrorException, "Mutating arrays is not supported --
    # called setindex!(Vector{Bool}, ...)" in the solver loop.
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError, "Detected potential need
    # for runtime activity."
    gradient_row("solve_columns/NewtonDescent", column_loss, p[3:4];
                 broken=("Zygote", "Enzyme"), slow=SLOW_BACKENDS)
end

end
