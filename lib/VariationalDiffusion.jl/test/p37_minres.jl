@testset "P37 an energy's field is symmetric at unequal block times" begin
    # The predictor's Jacobian diag(σ) H diag(α) is not symmetric under unequal block times (LP-018),
    # but the field weights coordinate i by c α_i/σ_i, so J_FF = Σ c diag(α) H diag(α) is: MINRES and
    # the one-product symmetric part (`op.S = op.J`) are right for an energy at any block times.
    VD = VariationalDiffusion
    A = [1.0 0.3; 0.3 2.0]
    pred = NoisePredictor(EnergyNetwork(LP018Energy(A)), VPSDE(); blocks = [[1], [2]], ad = AutoZygote())
    z, ρ, ps, st = [0.2, 0.3], [0.4, 0.7], (;), (;)
    for levels in ((0.1, 0.6), (nothing, 0.6))
        m = ImplicitDiffusion(pred, noisefree_nodes(2); levels)
        J = prior_jacobian(m, z, ps, st) + Diagonal(ρ .^ 2)
        X, t = VD._group_input(m, z, [1]), VD._node_times(m, m.nodes.t[1])
        D = epsilon_jacobian_block(pred, X, t, ps, st, [1, 2], [1.0])
        @test !isapprox(D, D'; rtol = 1e-6)    # the predictor has unequal row scales
        @test J ≈ J' rtol = 1e-12              # the field weights cancel them
        @test VD._symmetric_block(m)
        op = VD._free_operators(m, z, ρ, ps, st)
        @test stack(op.S(v) for v in eachcol(Matrix{Float64}(I, 2, 2))) ≈ (J + J') / 2 rtol = 1e-10
    end
    # a closed-form mixture's exact ε, two blocks at unequal levels, noisy nodes
    ring = ImplicitDiffusion(NoisePredictor(RING.model, SCHED; blocks = [[1], [2]]),
                             field_nodes(Xoshiro(1), 2; samples = 2); levels = (0.1, 0.6))
    Jr = prior_jacobian(ring, [0.6, 0.5], RPS, RST)
    @test Jr ≈ Jr' rtol = 1e-10
    @test VD._symmetric_block(ring)
end

@testset "P37 matrix-free MINRES selection and inference" begin
    VD = VariationalDiffusion
    IS = VD.ImplicitSolvers
    z, ρ, cotangent = [0.6, 0.5], [2.0, 2.0], [0.2, 1.0]
    nodes = field_nodes(Xoshiro(1), 2; samples = 2)
    epred = NoisePredictor(EnergyNetwork(ENERGY_MLP), SCHED; input = _batch_input, ad = AutoZygote())
    energy = ImplicitDiffusion(epred, nodes)
    eps_pred = NoisePredictor(MLP, SCHED; input = _batch_input, ad = AutoZygote())
    eps_model = ImplicitDiffusion(eps_pred, nodes)
    ring = ImplicitDiffusion(RING, nodes)
    ring_blocks = NoisePredictor(RING.model, SCHED; blocks = [[1], [2]])
    mixed_ring = ImplicitDiffusion(ring_blocks, nodes; levels = (0.1, 0.6))
    block_pred = NoisePredictor(EnergyNetwork(LP018Energy([1.0 0.3; 0.3 2.0])), SCHED;
                               blocks = [[1], [2]], ad = AutoZygote())
    mixed = ImplicitDiffusion(block_pred, nodes; levels = (0.1, 0.6))
    mixed_annealed = ImplicitDiffusion(block_pred, nodes; levels = (nothing, 0.6))
    all_annealed = ImplicitDiffusion(block_pred, nodes; levels = (nothing, nothing))
    all_fixed = ImplicitDiffusion(block_pred, nodes; levels = (0.3, 0.3))
    factor = DiffusionFactor((x = 1, y = 1), RING; prox = ImplicitProx(nodes))
    for (model, precisions, symmetric) in (
            (energy, ρ, true), (ring, ρ, true), (mixed_ring, ρ, true), (mixed, ρ, true),
            (mixed_annealed, ρ, true), (all_annealed, ρ, true), (all_fixed, ρ, true),
            (mixed, [Inf, 0.0], true), (mixed_annealed, [0.0, Inf], true),
            (mixed, [Inf, Inf], true), (eps_model, ρ, false),
            (ProductRelation(energy, energy), ρ, false), (factor, ρ, false))
        @test VD._symmetric_block(model) === symmetric
        @test VD._krylov_solver(KrylovSolve(), model, precisions) isa (symmetric ? IS.MINRES : IS.GMRES)
        @test VD._krylov_solver(KrylovSolve(method = :gmres), model, precisions) isa IS.GMRES
        if symmetric
            @test VD._krylov_solver(KrylovSolve(method = :minres, restart = 0, maxiters = 5), model, precisions) === IS.MINRES(maxiters = 5)
        else
            @test_throws ArgumentError VD._krylov_solver(KrylovSolve(method = :minres), model, precisions)
        end
    end
    @test_throws ArgumentError KrylovSolve(method = :bogus)
    old = KrylovSolve(1f-6, 10, 3, 4, 5, 6)
    @test old isa KrylovSolve{Float32}
    @test (old.rtol, old.maxiters, old.restart, old.lanczos, old.probes, old.seed, old.method) ==
          (1f-6, 10, 3, 4, 5, 6, :auto)
    ps0, st = Lux.setup(Xoshiro(0), MLP)
    @test_throws ArgumentError implicit_infer(eps_model, z, ρ, _f64(ps0), st;
        method = :newton, linsolve = KrylovSolve(method = :minres))

    ps0, st = Lux.setup(Xoshiro(0), ENERGY_MLP)
    ps = _f64(ps0)
    dense, _ = implicit_infer(energy, z, ρ, ps, st; method = :newton)
    ref = implicit_pullback(energy, dense, z, ρ, cotangent, ps, st)
    auto, _ = implicit_infer(energy, z, ρ, ps, st; method = :newton, linsolve = KrylovSolve())
    back = implicit_pullback(energy, auto, z, ρ, cotangent, ps, st; linsolve = KrylovSolve())
    gmres = KrylovSolve(method = :gmres)
    gm, _ = implicit_infer(energy, z, ρ, ps, st; method = :newton, linsolve = gmres)
    gmback = implicit_pullback(energy, gm, z, ρ, cotangent, ps, st; linsolve = gmres)
    @test auto.converged && auto.stable
    for (sol, adj) in ((dense, ref), (gm, gmback))
        @test auto.z ≈ sol.z atol = 1e-10
        @test back.z₀ ≈ adj.z₀ atol = 1e-8
        @test back.ρ ≈ adj.ρ atol = 1e-8
        for layer in keys(back.ps), parameter in keys(back.ps[layer])
            @test back.ps[layer][parameter] ≈ adj.ps[layer][parameter] atol = 1e-8
        end
    end

    prox = MixtureProx(RING.model)
    t, λ, ρprox = 0.01, 0.1, [Inf, 0.0]
    sol, _ = prox_infer(prox, z, ρprox, t, RPS, RST; λ, maxiters = 5000)
    @test sol.converged
    @test_throws ArgumentError prox_pullback(prox, sol, z, ρprox, t, cotangent, RPS, RST;
        λ, linsolve = KrylovSolve(method = :minres))
end
