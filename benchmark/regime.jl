# A seeded plate of two-regime scalar regressions, following
# bench/regime_plate_profile.jl.
using Lenticulum: RegimeFactor
using Mycelium
using Random: Xoshiro

function regime_plate()
    rng = Xoshiro(11)
    replicas = 8
    x = 6 .* rand(rng, replicas) .- 3
    labels = rand(rng, 1:2, replicas)
    y = [1.0, -1.0][labels] .* x .+ [0.0, 0.5][labels] .+ 0.2 .* randn(rng, replicas)
    builder = GraphBuilder()
    for i in eachindex(x)
        dx, dy, reg, xi, yi, k = (replica_name(s, i) for s in (:dx, :dy, :reg, :x, :y, :k))
        variable!(builder, xi, 1)
        variable!(builder, yi, 1)
        variable!(builder, k, 2)
        factor!(builder, dx, DataFactor(:x, [x[i]]))
        factor!(builder, dy, DataFactor(:y, [y[i]]))
        connect!(builder, dx, :x, xi; direction = Emitting())
        connect!(builder, dy, :y, yi; direction = Emitting())
        factor!(builder, reg, RegimeFactor((y = 1, x = 1), 2, 1; learn_noise = true, output = :y);
                params = :reg)
        connect!(builder, reg, :x, xi)
        connect!(builder, reg, :y, yi)
        connect!(builder, reg, :k, k)
    end
    g = as_plate(build(builder), :points, replicas)
    data = (; (replica_name(s, i) => (;) for i in eachindex(x) for s in (:dx, :dy))...)
    ps = merge(data, (reg = (A = (x = reshape([-0.5, 0.5], 1, 1, 2),), b = zeros(1, 2),
                             logw = fill(log(0.5), 2), L = zeros(1, 1, 2)),))
    return g, plate_schedule(g), ps, (;)
end
