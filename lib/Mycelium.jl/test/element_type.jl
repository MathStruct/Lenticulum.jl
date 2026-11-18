# P8 (LP-024 item 7): the graph free energy is in the element type of its messages. A graph whose
# messages are all Float32 has a Float32 free energy, on the node-by-node path and on a plate's
# rows, and it agrees with the same graph in Float64. Before P8 every total started at `0.0`, and
# the zero energies of data factors and points, the categorical entropy's `0 log 0` and the
# Gaussian entropy's `log(2π)` were Float64 constants, so a Float32 graph summed to Float64.

using LinearAlgebra
using ForwardDiff, Zygote

# x (Gaussian, degree 2: prior and relay), y (Gaussian, degree 2: relay and prior), c (categorical,
# degree 2: two priors), z (a point, degree 2: data and prior).
function p8_graph(T; plate::Union{Nothing,Int} = nothing)
    gauss(μ, λ) = LenticulumCore.GaussianBelief(T.(λ .* μ), Matrix{T}(λ * I, 2, 2))
    quad(μ, λ) = x -> λ * sum(abs2, x .- T.(μ)) / 2 - log(λ) + log(2 * T(π))
    function add!(b, i)
        variable!(b, :x, 2); variable!(b, :y, 2); variable!(b, :c); variable!(b, :z, 2)
        factor!(b, :px, PriorFactor(:x, gauss([1, -1], T(2)), quad([1, -1], T(2))))
        factor!(b, :r, RelayFactor(:x, :y))
        factor!(b, :py, PriorFactor(:y, gauss([0, 1], T(1)), quad([0, 1], T(1))))
        factor!(b, :pc1, PriorFactor(:c, CategoricalBelief(T[0.25, 0.75]), k -> T(k)))
        factor!(b, :pc2, PriorFactor(:c, CategoricalBelief(T[0.6, 0.4]), k -> T(2) * k))
        factor!(b, :dz, DataFactor(:z, T[0.5, i]))
        factor!(b, :pz, PriorFactor(:z, gauss([0, 0], T(1)), quad([0, 0], T(1))))
        connect!(b, :px, :x, :x); connect!(b, :r, :x, :x); connect!(b, :r, :y, :y)
        connect!(b, :py, :y, :y); connect!(b, :pc1, :c, :c); connect!(b, :pc2, :c, :c)
        connect!(b, :dz, :z, :z; direction = Emitting()); connect!(b, :pz, :z, :z)
    end
    b = GraphBuilder()
    plate === nothing ? add!(b, 0) : plate!((p, i) -> add!(p, i), b, :rep, plate)
    return validate(build(b))
end

function p8_energy(T; plate = nothing)
    g = p8_graph(T; plate)
    nt = NamedTuple()
    sched = plate === nothing ? spanning_tree_schedule(g) : plate_schedule(g)
    _, report, _, s = infer!(g, sched, nt, nt)
    report.converged || error("P8 test graph did not converge")
    F, _ = LenticulumCore.scalar_free_energy(s, g, nt, nt)
    Ff, _ = factor_free_energies(s, g, nt, nt)
    V = variable_corrections(s, g)
    return F, Ff, V
end

@testset "P8: the graph free energy keeps the element type of its messages" begin
    @test Mycelium.variable_entropy(DiracBelief(Float32[1, 2])) + 1f0 isa Float32
    @test Mycelium.variable_entropy(TrivialBelief()) + 1f0 isa Float32
    @test Mycelium.variable_entropy(CategoricalBelief(Float32[0, 1])) isa Float32
    @test Mycelium.variable_entropy(LenticulumCore.GaussianBelief(Float32[0, 0], Matrix{Float32}(I, 2, 2))) isa Float32
    @test LenticulumCore.support_entropy(LenticulumCore.ExtendedGaussian(Float32[0, 0], Matrix{Float32}(I, 2, 2))) isa Float32
    @test first(local_free_energy(DataFactor(:z, Float32[1]), NamedTuple(), nothing, nothing)) + 1f0 isa Float32

    for plate in (nothing, 3)
        F32, Ff32, V32 = p8_energy(Float32; plate)
        F64, _, _ = p8_energy(Float64; plate)
        @test F32 isa Float32
        @test F32 ≈ F64 rtol = 1e-5
        @test all(e -> !(e isa Float64), values(Ff32.parts))
        @test all(e -> !(e isa Float64), values(V32.parts))
    end

    # review (Codex): a graph whose every term is an exact zero still returns a float
    b = GraphBuilder(); variable!(b, :x, 1)
    factor!(b, :d, DataFactor(:x, Float32[1])); connect!(b, :d, :x, :x; direction = Emitting())
    g = validate(build(b)); nt = NamedTuple()
    _, _, _, s = infer!(g, spanning_tree_schedule(g), nt, nt)
    @test first(LenticulumCore.scalar_free_energy(s, g, nt, nt)) isa AbstractFloat
    # review (Codex): a mixture of categoricals of different element types keeps the widest
    m = LenticulumCore.MixtureBelief(CategoricalBelief[CategoricalBelief(Float32[1, 0]), CategoricalBelief([1e-100, 1.0])], Float32[0, 1])
    @test Mycelium.variable_entropy(m) ≈ 1e-100 * log(1e100) rtol = 1e-6   # 1 − 1e-100 rounds to 1 in Float64; was -0f0
    # review (Codex): derivatives in the precision, ForwardDiff and Zygote, against the closed forms
    # H = (1 + log 2π)/2 − log(λ)/2 and log Z = η²/(2λ) − log(λ)/2 + log(2π)/2
    H(λ) = Mycelium.variable_entropy(LenticulumCore.GaussianBelief([0.3], fill(λ, 1, 1)))
    logZ(λ) = LenticulumCore.logpartition(LenticulumCore.GaussianBelief([0.3], fill(λ, 1, 1)))
    for λ in (0.5, 2.0)
        @test ForwardDiff.derivative(H, λ) ≈ -1 / (2λ)
        @test only(Zygote.gradient(H, λ)) ≈ -1 / (2λ)
        @test ForwardDiff.derivative(logZ, λ) ≈ -0.09 / (2λ^2) - 1 / (2λ)
        @test only(Zygote.gradient(logZ, λ)) ≈ -0.09 / (2λ^2) - 1 / (2λ)
    end
end
