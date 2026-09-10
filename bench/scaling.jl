# Scaling and allocation behaviour of `infer!` on a linear-Gaussian chain.
# Run:  julia --project=. bench/scaling.jl
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Printf

const Id = fill(1.0, 1, 1)
const Q  = fill(0.25, 1, 1)

function chain(n::Int)
    b = GraphBuilder()
    for k in 1:n; variable!(b, Symbol(:x, k), 1); end
    factor!(b, :prior, GaussianPrior(:x1, [0.0], fill(1.0, 1, 1)))
    connect!(b, :prior, :x1, :x1; direction = Bidirectional())
    for k in 1:(n-1)
        nm = Symbol(:odo, k)
        factor!(b, nm, GaussianFactor(1 => 1; noise = Q, channels = (:x, :y)))
        connect!(b, nm, :x, Symbol(:x, k);   direction = Bidirectional())
        connect!(b, nm, :y, Symbol(:x, k+1); direction = Bidirectional())
    end
    g = validate(build(b))
    ps = merge((prior = NamedTuple(),),
               NamedTuple(Symbol(:odo,k) => (A = Id, b = [1.0]) for k in 1:(n-1)))
    st = merge((prior = NamedTuple(),),
               NamedTuple(Symbol(:odo,k) => NamedTuple() for k in 1:(n-1)))
    return g, ps, st
end

println("chain scaling: infer! on a tree schedule\n")
@printf("%6s %8s %10s %12s %14s %12s\n", "n", "edges", "msgs", "time (ms)", "alloc (MiB)", "alloc/msg")
for n in (10, 25, 50, 100, 200, 400)
    g, ps, st = chain(n)
    sched = tree_schedule(g)
    nmsg = length(tasks(sched))
    infer!(g, sched, ps, st)                       # warm up
    t = @elapsed infer!(g, sched, ps, st)
    a = @allocated infer!(g, sched, ps, st)
    @printf("%6d %8d %10d %12.3f %14.2f %12.0f\n",
            n, nedges(g), nmsg, 1000t, a/2^20, a/nmsg)
end

# --- isolate the cost of the untyped store -------------------------------
println("\nstore access: Vector{Any} vs a concretely typed vector")
struct Msg; belief::GaussianBelief{Vector{Float64},Matrix{Float64}}; iteration::Int; end
const N = 200_000
gb = GaussianBelief([1.0], fill(2.0, 1, 1))

any_store = Vector{Any}(undef, N); fill!(any_store, Msg(gb, 0))
typed     = [Msg(gb, 0) for _ in 1:N]

sum_any(v)   = (s = 0.0; for i in eachindex(v); s += v[i].belief.η[1]; end; s)
sum_typed(v) = (s = 0.0; for i in eachindex(v); s += v[i].belief.η[1]; end; s)

sum_any(any_store); sum_typed(typed)              # warm up
ta = @elapsed for _ in 1:20; sum_any(any_store); end
tt = @elapsed for _ in 1:20; sum_typed(typed); end
aa = @allocated sum_any(any_store)
at = @allocated sum_typed(typed)
@printf("  Vector{Any}: %7.2f ms   %9d bytes allocated\n", 1000ta, aa)
@printf("  typed      : %7.2f ms   %9d bytes allocated\n", 1000tt, at)
@printf("  ratio      : %7.1fx slower\n", ta/tt)
