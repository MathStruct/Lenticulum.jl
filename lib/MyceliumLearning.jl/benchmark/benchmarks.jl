using BenchmarkTools
using ADTypes: AutoForwardDiff, AutoZygote
using DifferentiationInterface, ForwardDiff, Zygote
using LenticulumCore: free_energy_gradient
using MyceliumLearning: em_step

include(joinpath(@__DIR__, "../../../benchmark/fixtures.jl"))
include(joinpath(@__DIR__, "../../../benchmark/kalman.jl"))
include(joinpath(@__DIR__, "../../../benchmark/regime.jl"))

const SUITE::BenchmarkGroup = BenchmarkGroup()
learning = SUITE["MyceliumLearning"] = BenchmarkGroup()
gradients = learning["kalman"] = BenchmarkGroup()

function kalman_gradient_inputs(steps)
    g, ps, st = kalman_chain(steps; coupling = ExactCoupling())
    return g, tree_schedule(g), ps, st
end

function kalman_first_call(g, sched, ps, st)
    _, _, poststate, store = infer!(g, sched, ps, st)
    return free_energy_gradient(store, g, ps, poststate; ad = AutoZygote(), schedule = sched)
end

const kalman = cached_fixture(() -> kalman_gradient_inputs(3))
for (key, ad) in (("zygote_gradient", AutoZygote()), ("forwarddiff_gradient", AutoForwardDiff()))
    gradients[key] = @benchmarkable free_energy_gradient(store, g, ps, st; ad=$ad, schedule=sched) setup=((g, sched, ps, st) = $kalman(); store = MessageStore(g)) evals=1
end
# Forward mode keeps both adjoint cases small on this short chain.
const adjoint_ad = AutoForwardDiff()
for solver in (:dense, :gmres)
    gradients["adjoint_$solver"] = @benchmarkable free_energy_gradient(
        store, g, ps, st; ad=$adjoint_ad, schedule=sched, adjoint=:implicit, linear_solver=$solver
    ) setup=((g, sched, ps, st) = $kalman(); store = MessageStore(g)) evals=1
end

const em_inputs = cached_fixture() do
    g, sched, ps, st = regime_plate()
    _, _, poststate, store = infer!(g, sched, ps, st)
    return store, g, ps, poststate
end
learning["regime"] = BenchmarkGroup()
learning["regime"]["mstep"] = @benchmarkable em_step(store, g, ps, st) setup=((store, g, ps, st) = $em_inputs())
