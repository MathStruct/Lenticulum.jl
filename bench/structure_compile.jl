# The cost of the first gradient on a new graph structure, through the graph drivers (I1, I30): the
# position-velocity chain of `kalman_chain.jl` with every edge `ExactCoupling()` and the tree schedule.
# Each size is a new type of `ps` and `st` (a field per factor), so its first call compiles whatever
# is specialised on them. A chain of 5 steps (16 factors) is run first, so that what every structure
# shares (the factors' own code) is compiled before the first timed size. Recorded in
# `lib/Mycelium.jl/src/passing.md`, "The AD paths (I1)", and `lib/MyceliumLearning.jl/src/gradient_couplings.md`.
# Run one mode per process, since modes share compiled code:
#   MODE=zygote   julia --project=bench bench/structure_compile.jl      (TS="10 20 40" to choose the steps)
#   MODE=fd       julia --project=bench bench/structure_compile.jl
#   MODE=implicit julia --project=bench bench/structure_compile.jl      (Zygote through adjoint = :implicit)
# Smoke: LENTICULUM_SMOKE=true (bench/smoke.jl) runs every mode at 2 and 3 steps, after 1.

@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore, Printf,
    Zygote, ForwardDiff, ADTypes, DifferentiationInterface

include(joinpath(@__DIR__, "kalman_chain.jl"))

const _SC_STEPS = SMOKE ? (2, 3) : Tuple(parse.(Int, split(get(ENV, "TS", "10 20 40"))))
const _SC_MODES = SMOKE ? ("zygote", "fd", "implicit") : (get(ENV, "MODE", "zygote"),)

function _sc_gradient(mode, g, ps, st, sched)
    ad = mode == "fd" ? AutoForwardDiff() : AutoZygote()
    adjoint = mode == "implicit" ? :implicit : :unrolled
    return free_energy_gradient(MessageStore(g), g, ps, st; ad, schedule = sched, adjoint)
end

for mode in _SC_MODES
    g0, ps0, st0 = chain(SMOKE ? 1 : 5; coupling = ExactCoupling())
    _sc_gradient(mode, g0, ps0, st0, tree_schedule(g0))
    for T in _SC_STEPS
        g, ps, st = chain(T; coupling = ExactCoupling())
        sched = tree_schedule(g)
        first = @elapsed _sc_gradient(mode, g, ps, st, sched)
        warm = minimum(_ -> (GC.gc(); @elapsed _sc_gradient(mode, g, ps, st, sched)), 1:(SMOKE ? 1 : 3))
        @printf("%-8s T=%4d factors=%5d  first %8.2f s  warm %9.4f s\n", mode, T, length(g.factors), first, warm)
    end
end
