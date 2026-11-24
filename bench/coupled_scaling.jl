# How the cost of Zygote through the coupled gradient drivers grows with the number of factors, on the
# position-velocity Kalman chain of `state_scaling.jl` with every edge `ExactCoupling()` and the tree
# schedule. Before I17 every task of a replay read `ps` and the message store as a whole and copied the
# store, so the cotangents added up as O(tasks × keys + tasks × edges). Recorded in
# `lib/MyceliumLearning.jl/src/gradient_couplings.md`, "Reverse mode by message versions (I17)".
# Run:  julia --project=bench bench/coupled_scaling.jl              (TS="10 20 40" to choose the steps)
#       MODE=implicit julia --project=bench bench/coupled_scaling.jl
# Smoke: LENTICULUM_SMOKE=true (bench/smoke.jl) runs both modes with 2 chain steps and 1 timed run.
#
# `unrolled`: `free_energy_gradient(…; ad = AutoZygote(), schedule)`, one sweep replayed, and the same
# with reference differences (one replay per parameter coordinate, so quadratic by nature).
# `implicit`: the pieces of `adjoint = :implicit` at a converged store: the coordinate store, one update,
# one Krylov product (Zygote of `_graph_map_dot` in the messages and `ps`) and Zygote of the loss. The
# dense linear solve (one reverse pass per coordinate) is not measured. Zygote compiles its pullback once
# per type of `ps`, so every size compiles anew ("first").

@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore, Printf,
    Zygote, ADTypes, DifferentiationInterface

include(joinpath(@__DIR__, "kalman_chain.jl"))
const ML = MyceliumLearning

best(f; n = (SMOKE ? 1 : 3)) = minimum(_ -> (GC.gc(); @elapsed f()), 1:n)
steps = SMOKE ? (2,) : Tuple(parse.(Int, split(get(ENV, "TS", "10 20 40"))))
modes = SMOKE ? ("unrolled", "implicit") : (get(ENV, "MODE", "unrolled"),)

for mode in modes, T in steps
    g, ps, st = chain(T; coupling = ExactCoupling())
    sched = tree_schedule(g)
    if mode == "unrolled"
        G() = free_energy_gradient(MessageStore(g), g, ps, st; ad = AutoZygote(), schedule = sched)
        tc = @elapsed G()
        t = best(G); bytes = @allocated G()
        tr = best(() -> free_energy_gradient(MessageStore(g), g, ps, st; schedule = sched))
        @printf("unrolled T=%4d factors=%5d tasks=%6d  first %7.1f s  zygote %9.2f ms  %9.1f MB  refdiff %9.2f ms\n",
            T, length(g.factors), length(tasks(sched)), tc, t * 1e3, bytes / 2^20, tr * 1e3)
    else
        s = MessageStore(g); for _ in 1:3; sweep!(s, g, sched, ps, st); end
        ts = all_tasks(g)
        beliefs = map(t -> ML._stored_belief(s, t), ts)
        parts = map(ML._message_coordinates, beliefs)
        lengths = length.(parts); starts = cumsum(vcat(1, lengths[1:end-1]))
        ranges = map((i, n) -> i:i+n-1, starts, lengths); x = reduce(vcat, parts)
        frozen = ML._frozen_beliefs(s, g, ps, st)
        ctx = (; g, ps, st, ts, beliefs, ranges, x, ad = AutoZygote(), damping = 0.0, ε = 1e-6, frozen)
        tcs = best(() -> ML._coordinate_store(x, ctx))
        tu = best(() -> ML._graph_update(x, ps, ctx))
        v = randn(Xoshiro(1), length(x))
        P() = Zygote.gradient((x, p) -> ML._graph_map_dot(x, p, ctx, v), x, ps)
        tc = @elapsed P(); tp = best(P); pbytes = @allocated P()
        R() = Zygote.gradient((x, p) -> ML._graph_loss(x, p, ctx), x, ps)
        tr = best(R)
        @printf("implicit T=%4d tasks=%6d n=%6d  store %8.2f ms  update %8.2f ms  product first %6.1f s  product %9.2f ms %8.1f MB  loss grad %9.2f ms\n",
            T, length(ts), length(x), tcs * 1e3, tu * 1e3, tc, tp * 1e3, pbytes / 2^20, tr * 1e3)
    end
    flush(stdout)
end
