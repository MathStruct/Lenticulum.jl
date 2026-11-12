# The position-velocity Kalman chain of `state_scaling.jl` and `sweep_profile.jl`: T steps, three
# factors per step (transition, observation, data). `chain(T)` returns the graph, `ps` and `st`.
using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore

function chain(T; nx = 2, H = [1.0 0.0; 0.0 1.0], A = [1.0 1.0; 0.0 1.0])
    b = GraphBuilder()
    for t in 0:T; variable!(b, Symbol(:x, t), nx); end
    factor!(b, :prior, GaussianPrior(:x, zeros(nx), Matrix(1.0I, nx, nx))); connect!(b, :prior, :x, :x0)
    for t in 1:T
        factor!(b, Symbol(:f, t), GaussianFactor(nx => nx; noise = 0.1, channels = (:a, :b)))
        connect!(b, Symbol(:f, t), :a, Symbol(:x, t - 1)); connect!(b, Symbol(:f, t), :b, Symbol(:x, t))
        variable!(b, Symbol(:y, t), size(H, 1))
        factor!(b, Symbol(:h, t), GaussianFactor(nx => size(H, 1); noise = 0.5, channels = (:a, :b)))
        connect!(b, Symbol(:h, t), :a, Symbol(:x, t)); connect!(b, Symbol(:h, t), :b, Symbol(:y, t))
        factor!(b, Symbol(:d, t), DataFactor(:v, fill(0.3t, size(H, 1))))
        connect!(b, Symbol(:d, t), :v, Symbol(:y, t); direction = Emitting())
    end
    g = build(b)
    rng = Xoshiro(1)
    ps = LuxCore.initialparameters(rng, g); st = LuxCore.initialstates(rng, g)
    kv = Dict{Symbol,Any}(pairs(ps))
    for t in 1:T
        kv[Symbol(:f, t)] = (A = A, b = zeros(nx))
        kv[Symbol(:h, t)] = (A = H, b = zeros(size(H, 1)))
    end
    ks = keys(ps)
    return g, NamedTuple{ks}(Tuple(kv[k] for k in ks)), st
end
