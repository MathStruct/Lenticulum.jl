# The Reactant backend, kept out of the main suite because Reactant downloads XLA. Run with
#     julia --project=test/reactant -e 'using Pkg; Pkg.instantiate(); include("test/reactant/runtests.jl")'
# from lib/VariationalDiffusion.jl. Uses Reactant's CPU backend; the same code runs on a GPU.
using Test, Random, LinearAlgebra
using VariationalDiffusion, LuxCore, Reactant
using Lux: Lux, Chain, Dense, swish
using ADTypes: AutoReactant
const VD = VariationalDiffusion
Reactant.set_default_backend("cpu")

const FREQ = [1.0, 2.0, 4.0]
_emb(t) = vcat(sin.((2π * t) .* FREQ), cos.((2π * t) .* FREQ))
_input(x, t) = vcat(x, _emb(t) .* ones(eltype(x), 1, size(x, 2)))
_f64(ps) = Lux.Functors.fmap(a -> a isa AbstractArray ? Float64.(a) : a, ps)

@testset "Reactant backend" begin
    sched = VPSDE()
    mlp = Chain(Dense(8 => 32, swish), Dense(32 => 2))
    plain = NoisePredictor(mlp, sched; input = _input)
    rx = NoisePredictor(mlp, sched; input = _input, ad = AutoReactant())
    ps, st = LuxCore.setup(Xoshiro(0), plain)
    ps = _f64(ps)
    x = reshape([0.3, -0.4], 2, 1)
    ε(p, x, t, ps = ps) = first(epsilon(p, x, t, ps, st))

    @testset "the compiled forward pass agrees with plain Lux, for any t" begin
        for t in (0.1, 0.37, 0.9)
            @test ε(rx, x, t) ≈ ε(plain, x, t) atol = 1e-12
        end
    end

    h = 1e-6
    @testset "input Jacobian and parameter VJP agree with finite differences" begin
        J = VD.epsilon_jacobian(rx, vec(x), 0.37, ps, st)
        Jfd = hcat([(ε(plain, vec(x) .+ h .* (1:2 .== i), 0.37) .- ε(plain, vec(x) .- h .* (1:2 .== i), 0.37)) ./ 2h for i in 1:2]...)
        @test J ≈ Jfd atol = 1e-8
        w = [0.7, -1.3]
        g = VD.epsilon_vjp_params(rx, x, 0.37, ps, st, w)
        p₊ = deepcopy(ps); p₊.layer_2.weight[1, 3] += h
        p₋ = deepcopy(ps); p₋.layer_2.weight[1, 3] -= h
        fd = (sum(ε(plain, x, 0.37, p₊) .* w) - sum(ε(plain, x, 0.37, p₋) .* w)) / 2h
        @test g.layer_2.weight[1, 3] ≈ fd atol = 1e-8
        @test VD.epsilon_vjp_params(rx, vec(x), 0.37, ps, st, w).layer_2.weight[1, 3] ≈ fd atol = 1e-8  # 2×1 output, vector w
    end

    @testset "implicit inference and its adjoint run through the compiled network" begin
        nodes = field_nodes(Xoshiro(1), 2; samples = 2)
        mp, mr = ImplicitDiffusion(plain, nodes), ImplicitDiffusion(rx, nodes)
        z₀, ρ = [0.6, 0.5], [Inf, 1.0]
        sp, _ = implicit_infer(mp, z₀, ρ, ps, st)
        sr, _ = implicit_infer(mr, z₀, ρ, ps, st)
        @test sr.converged
        @test sr.z ≈ sp.z atol = 1e-8
        b = implicit_pullback(mr, sr, z₀, ρ, [0.0, 1.0], ps, st)
        z(x) = first(implicit_infer(mp, [x, 0.5], ρ, ps, st)).z[2]
        @test b.z₀[1] ≈ (z(0.6 + h) - z(0.6 - h)) / 2h atol = 1e-6
    end
end
