using BenchmarkTools
using Lenticulum: LinearConstraintFactor
using LenticulumCore: Gaussian, scalar_free_energy
using Mycelium
using LinearAlgebra: I
using Random: Xoshiro

include(joinpath(@__DIR__, "../../../benchmark/fixtures.jl"))
include(joinpath(@__DIR__, "../../../benchmark/kalman.jl"))

const SUITE::BenchmarkGroup = BenchmarkGroup()
mycelium = SUITE["Mycelium"] = BenchmarkGroup()
chain = mycelium["kalman"] = BenchmarkGroup()

const kalman = cached_fixture() do
    g, ps, st = kalman_chain(96)
    return g, ps, st, tree_schedule(g), flooding_schedule(g)
end

chain["tree_infer"] = @benchmarkable infer!(g, sched, ps, st) setup=((g, ps, st, sched, flood) = $kalman()) evals=1
# Fixed sweeps measure flooding throughput on the same chain.
chain["flooding_infer"] = @benchmarkable infer!(g, sched, ps, st; maxsweeps=4, tol=0.0) setup=((g, ps, st, tree, sched) = $kalman()) evals=1

const energy_inputs = cached_fixture() do
    g, ps, st, sched, _ = kalman()
    _, _, poststate, store = infer!(g, sched, ps, st)
    return store, g, ps, poststate
end
chain["free_energy"] = @benchmarkable scalar_free_energy(store, g, ps, st) setup=((store, g, ps, st) = $energy_inputs())

function gaussian_grid(n)
    builder = GraphBuilder()
    parameters = Dict{Symbol,Any}()
    rng = Xoshiro(23)
    node(i, j) = Symbol(:x, i, :_, j)
    for j in 1:n, i in 1:n
        variable!(builder, node(i, j), 1)
        prior = Symbol(:prior, i, :_, j)
        factor!(builder, prior, GaussianPrior(:x, [randn(rng)], ones(1, 1)))
        connect!(builder, prior, :x, node(i, j))
        parameters[prior] = (;)
    end
    for j in 1:n, i in 1:n, (di, dj) in ((1, 0), (0, 1))
        (i + di <= n && j + dj <= n) || continue
        edge = Symbol(:edge, i, :_, j, :_, di)
        factor!(builder, edge, LinearConstraintFactor((a = 1, b = 1), 1; noise = 1.0))
        connect!(builder, edge, :a, node(i, j))
        connect!(builder, edge, :b, node(i + di, j + dj))
        parameters[edge] = (A = (a = ones(1, 1), b = -ones(1, 1)), c = zeros(1))
    end
    g = build(builder)
    return g, flooding_schedule(g), (; parameters...), (;)
end

const grid = cached_fixture(() -> gaussian_grid(10))
mycelium["grid"] = BenchmarkGroup()
mycelium["grid"]["flooding_sweep"] = @benchmarkable sweep!(s, g, sched, ps, st) setup=((g, sched, ps, st) = $grid(); s = MessageStore(g)) evals=1

function factor_analysis_plate()
    rng = Xoshiro(24)
    X = randn(rng, 3, 24)
    builder = GraphBuilder()
    plate!(builder, :observations, size(X, 2)) do p, i
        variable!(p, :z, 2)
        variable!(p, :x, 3)
        factor!(p, :prior, GaussianPrior(:z, zeros(2), Matrix(1.0I, 2, 2)))
        factor!(p, :lik, LinearConstraintFactor((x = 3, z = 2), 3; noise = 0.7))
        factor!(p, :data, DataFactor(:x, X[:, i]))
        connect!(p, :prior, :z, :z)
        connect!(p, :lik, :z, :z)
        connect!(p, :lik, :x, :x; direction = Absorbing())
        connect!(p, :data, :x, :x; direction = Emitting())
    end
    g = build(builder)
    ps = (lik = (A = (x = Matrix(1.0I, 3, 3), z = -randn(rng, 3, 2)), c = randn(rng, 3)),)
    return g, plate_schedule(g), ps, (;)
end

const plate_inputs = cached_fixture(factor_analysis_plate)
mycelium["plate"] = BenchmarkGroup()
mycelium["plate"]["factor_analysis_estep"] = @benchmarkable sweep!(s, g, sched, ps, st) setup=((g, sched, ps, st) = $plate_inputs(); s = MessageStore(g)) evals=1
mycelium["belief"] = BenchmarkGroup()
mycelium["belief"]["gaussian_product"] = @benchmarkable Mycelium.combine(a, b) setup=((a, b) = (Gaussian([0.3, -0.4], [0.5 0.1; 0.1 0.4]), Gaussian([0.0, 1.0], [0.6 0.2; 0.2 0.7])))
