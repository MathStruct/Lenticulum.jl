module LearningADMatrix

using MyceliumLearning, Mycelium, LenticulumCore, LinearAlgebra, Test
using MyceliumLearning: MomentStats
include("../../LenticulumCore.jl/test/support/ad_matrix.jl")
using .Q5ADMatrix
include("support/adjoint_gaussian.jl")
include("support/normal_m_step.jl")

@testset verbose=true "MyceliumLearning AD backend matrix" begin
    g, ps, st = adjoint_gaussian_graph()
    # Vary the first unary potential's precision and natural mean.
    p = [ps.u.q, ps.u.h]
    parameters(v) = merge(ps, (u=(q=v[1], h=v[2]),))
    # Two sequential sweeps reach the fixed point of this short Gaussian tree.
    sched = SequentialSchedule(tasks(tree_schedule(g)))
    maxsweeps = 2
    _, _, _, store = infer!(g, sched, ps, st; maxsweeps, tol=0.0)
    for adjoint in (:unrolled, :implicit)
        gradient(ad) = adjoint_flat(free_energy_gradient(store, g, ps, st;
                                        ad, adjoint, schedule=sched, nsweeps=maxsweeps))
        # Q5 baseline: Enzyme and Mooncake ArgumentError, "schedule differentiation
        # supports reference differences, AutoForwardDiff and AutoZygote".
        matrix_row("free_energy_gradient/$adjoint", gradient, () -> gradient(nothing);
                   broken=("Enzyme", "Mooncake"))
    end

    function loss(v)
        q = parameters(v)
        _, _, state, s = infer!(g, sched, q, st; maxsweeps, tol=0.0)
        return first(scalar_free_energy(s, g, q, state))
    end
    reference = free_energy_gradient(store, g, ps, st; ad=nothing, schedule=sched, nsweeps=maxsweeps)
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError, "Detected potential need
    # for runtime activity." Mooncake MooncakeRuleCompilationError, "Mooncake
    # failed to differentiate the following method:" (try/catch in _emit_plan).
    gradient_row("free_energy after infer!", loss, p; reference=[reference.u.q, reference.u.h],
                 broken=("Enzyme", "Mooncake"))
end

# The EM statistics dominated the 6m50s standalone matrix.
slow_tier() && @testset verbose=true "MyceliumLearning EM AD backend matrix" begin
    # A latent normal combined with a fixed Gaussian prior. Its sufficient
    # statistics genuinely depend on ps because the latent variable is unobserved.
    b = GraphBuilder()
    variable!(b, :x, 1)
    factor!(b, :normal, P32Normal()); connect!(b, :normal, :x, :x)
    prior = PriorFactor(:x, Gaussian(0.2, 2.0), _ -> false)
    factor!(b, :prior, prior); connect!(b, :prior, :x, :x)
    g = cluster(build(b), :joint, [:normal])
    p = [0.7, -0.1]
    parameters(v) = (joint=(normal=(μ=[sum(v)],),),)
    nt = (;)
    _, _, st, store = infer!(g, tree_schedule(g), parameters(p), nt)
    function moment(v)
        stats = moment_statistics(store, g, parameters(v), st)
        return only(stats[(:joint, :normal)][2].m)
    end
    # Q5 baseline: Zygote MethodError, "no method matching rrule(::typeof(sum),
    # ::Vector{Int64}; init::Int64)". Enzyme EnzymeRuntimeActivityError, "Detected
    # potential need for runtime activity." Mooncake MissingForeigncallRuleError,
    # "No rrule!! available for foreigncall with primal argument types".
    gradient_row("moment_statistics", moment, p; broken=("Zygote", "Enzyme", "Mooncake"),
                 slow=("Zygote", "ForwardDiff", "Enzyme", "Mooncake"))
end

end
