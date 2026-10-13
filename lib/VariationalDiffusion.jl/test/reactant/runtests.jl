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

    @testset "free-coordinate blocks, input VJPs and batched nodes, compiled" begin
        nodes = field_nodes(Xoshiro(1), 2; samples = 2)
        mp = ImplicitDiffusion(plain, nodes)
        z, w = [0.6, 0.5], [0.3, -0.8]
        J = prior_jacobian(mp, z, ps, st)
        for batched in (false, true)
            mr = ImplicitDiffusion(rx, nodes; batched)
            @test first(prior_field(mr, z, ps, st)) ≈ first(prior_field(mp, z, ps, st)) atol = 1e-10
            @test prior_jacobian(mr, z, ps, st; coords = [2]) ≈ J[[2], [2]] rtol = 1e-6
            @test prior_vjp(mr, z, w, ps, st) ≈ J' * w rtol = 1e-6
            @test prior_jvp(mr, z, w, ps, st) ≈ J * w rtol = 1e-6     # a difference through the compiled forward
        end
        # matrix-free inference and adjoint through the compiled network
        mr = ImplicitDiffusion(rx, nodes; batched = true)
        ρ = [Inf, 1.0]
        sp, _ = implicit_infer(mp, z, ρ, ps, st)
        sk, _ = implicit_infer(mr, z, ρ, ps, st; linsolve = KrylovSolve())
        @test sk.converged && sk.stable
        @test sk.z ≈ sp.z atol = 1e-8
        bd = implicit_pullback(mr, sk, z, ρ, [0.0, 1.0], ps, st)
        bk = implicit_pullback(mr, sk, z, ρ, [0.0, 1.0], ps, st; linsolve = KrylovSolve())
        @test bk.z₀ ≈ bd.z₀ rtol = 1e-6
        @test bk.ps.layer_2.weight ≈ bd.ps.layer_2.weight rtol = 1e-6 atol = 1e-12
    end

    @testset "an energy network, second derivatives compiled (nested Enzyme)" begin
        enet = Chain(Dense(8 => 32, tanh), Dense(32 => 1))
        ep = NoisePredictor(EnergyNetwork(enet), sched; input = _input)            # no ad: the forward energy only
        er = NoisePredictor(EnergyNetwork(enet), sched; input = _input, ad = AutoReactant())
        qs, qt = LuxCore.setup(Xoshiro(2), ep)
        qs = _f64(qs)
        t, s = 0.37, sigma(sched, 0.37)
        E(x, q = qs) = first(energy(ep, x, t, q, qt))
        e(i, n) = Float64.(1:n .== i)
        ∇E(x, q = qs) = [(E(x .+ h .* e(i, 2), q) - E(x .- h .* e(i, 2), q)) / 2h for i in 1:2]
        x = [0.3, -0.4]
        εr(x) = vec(first(epsilon(er, x, t, qs, qt)))
        @test εr(x) ≈ s .* ∇E(x) atol = 1e-8
        X = [0.3 -0.2 0.9; -0.4 0.5 0.1]
        @test first(epsilon(er, X, t, qs, qt)) ≈ s .* hcat(∇E.(eachcol(X))...) atol = 1e-8   # per column
        J = VD.epsilon_jacobian(er, x, t, qs, qt)
        @test J ≈ J' atol = 1e-12                                                    # a Hessian
        @test J ≈ hcat([(εr(x .+ h .* e(i, 2)) .- εr(x .- h .* e(i, 2))) ./ 2h for i in 1:2]...) atol = 1e-7
        W = [0.7 -1.3 0.2; 0.4 0.1 -0.5]
        HW = VD.epsilon_vjp_input(er, X, t, qs, qt, W)
        @test HW ≈ hcat([VD.epsilon_jacobian(er, X[:, b], t, qs, qt) * W[:, b] for b in 1:3]...) atol = 1e-10
        @test VD.epsilon_jvp_input(er, X, t, qs, qt, W) ≈ HW atol = 1e-12
        # the block: Σ_b w_b ∂ε(x_b) on the coordinates idx
        wb = [0.5, 1.0, 2.0]
        Jb = sum(wb[b] .* VD.epsilon_jacobian(er, X[:, b], t, qs, qt) for b in 1:3)
        @test VD.epsilon_jacobian_block(er, X, t, qs, qt, [2], wb) ≈ Jb[[2], [2]] atol = 1e-10
        @test VD.epsilon_jacobian_block(er, X, t, qs, qt, [1, 2], wb) ≈ Jb atol = 1e-10
        # the mixed derivative ∂_θ⟨ε, w⟩ against finite differences of ⟨σ∇E, w⟩ in one weight
        g = VD.epsilon_vjp_params(er, X, t, qs, qt, W)
        q₊ = deepcopy(qs); q₊.layer_1.weight[3, 2] += 1e-4
        q₋ = deepcopy(qs); q₋.layer_1.weight[3, 2] -= 1e-4
        pairing(q) = sum(s .* hcat([∇E(X[:, b], q) for b in 1:3]...) .* W)
        @test g.layer_1.weight[3, 2] ≈ (pairing(q₊) - pairing(q₋)) / 2e-4 atol = 1e-6
        loss, ḡ, _ = denoising_gradient(er, X, t, W, qs, qt)
        @test ḡ.layer_2.weight ≈ VD.epsilon_vjp_params(er, alpha(sched, t) .* X .+ s .* W, t, qs, qt,
                                                    (2 / 3) .* (first(epsilon(er, alpha(sched, t) .* X .+ s .* W, t, qs, qt)) .- W)).layer_2.weight atol = 1e-12
        # inference, its energy and its adjoint through the compiled second derivatives
        nodes = field_nodes(Xoshiro(1), 2; samples = 2)
        z₀, ρ = [0.6, 0.5], [Inf, 1.0]
        for batched in (false, true)
            m = ImplicitDiffusion(er, nodes; batched)
            sol, _ = implicit_infer(m, z₀, ρ, qs, qt)
            @test sol.converged
            U(z) = first(implicit_energy(m, z, qs, qt))
            @test first(prior_field(m, sol.z, qs, qt)) ≈ [(U(sol.z .+ h .* e(i, 2)) - U(sol.z .- h .* e(i, 2))) / 2h for i in 1:2] atol = 1e-6
            b = implicit_pullback(m, sol, z₀, ρ, [0.0, 1.0], qs, qt)
            z(x) = first(implicit_infer(m, [x, 0.5], ρ, qs, qt)).z[2]
            @test b.z₀[1] ≈ (z(0.6 + 1e-5) - z(0.6 - 1e-5)) / 2e-5 atol = 1e-6
            sk, _ = implicit_infer(m, z₀, ρ, qs, qt; linsolve = KrylovSolve())
            @test sk.z ≈ sol.z atol = 1e-8
        end
    end
end
