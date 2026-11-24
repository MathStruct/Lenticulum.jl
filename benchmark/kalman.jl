# Position and velocity with Gaussian transitions and two observed coordinates
# per step, following the model in bench/kalman_chain.jl.
using Lenticulum: GaussianFactor, GaussianPrior
using LenticulumCore: DiagonalCoupling, ExactCoupling
using Mycelium
using LinearAlgebra: I

function kalman_chain(steps; coupling = DiagonalCoupling())
    builder = GraphBuilder()
    state(t) = Symbol(:x, t)
    parameters = Dict{Symbol,Any}(:prior => (;))
    transition = [1.0 1.0; 0.0 1.0]
    identity = Matrix(1.0I, 2, 2)
    for t in 0:steps
        variable!(builder, state(t), 2)
    end
    factor!(builder, :prior, GaussianPrior(:x, zeros(2), identity))
    connect!(builder, :prior, :x, state(0); coupling)
    for t in 1:steps
        f, h, d, y = Symbol.((:f, :h, :d, :y), t)
        variable!(builder, y, 2)
        factor!(builder, f, GaussianFactor(2 => 2; noise = 0.1, channels = (:a, :b)))
        connect!(builder, f, :a, state(t - 1); coupling)
        connect!(builder, f, :b, state(t); coupling)
        factor!(builder, h, GaussianFactor(2 => 2; noise = 0.5, channels = (:a, :b)))
        connect!(builder, h, :a, state(t); coupling)
        connect!(builder, h, :b, y; coupling)
        factor!(builder, d, DataFactor(:v, fill(0.3t, 2)))
        connect!(builder, d, :v, y; direction = Emitting(), coupling)
        parameters[f] = (A = transition, b = zeros(2))
        parameters[h] = (A = identity, b = zeros(2))
        parameters[d] = (;)
    end
    g = build(builder)
    ps = (; (node.name => parameters[node.name] for node in g.factors)...)
    return g, ps, (;)
end
