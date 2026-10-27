using ADTypes, DifferentiationInterface, ForwardDiff, Zygote
using Distributions: MvNormal, Wishart, LogNormal, logpdf

# Each store is one independent observation. Averaging candidate GD steps from
# the SAME parameters is exactly a step on the mean sample free energy. This
# exercises parameter_step without pretending that graph factors share parameters.
function noise_sample_graph(f, x, y)
    b = GraphBuilder()
    variable!(b, :x, length(x)); variable!(b, :y, length(y))
    factor!(b, :model, f)
    factor!(b, :input, DataFactor(:x, x))
    factor!(b, :target, DataFactor(:y, y))
    connect!(b, :model, :x, :x; direction = Absorbing())
    connect!(b, :model, :y, :y; direction = Absorbing())
    connect!(b, :input, :x, :x; direction = Emitting())
    connect!(b, :target, :y, :y; direction = Emitting())
    g = build(b)
    ps, st = LuxCore.setup(Xoshiro(5), g)
    store = MessageStore(g)
    _, st = propagate!(store, g, tree_schedule(g), ps, st)
    return store, g, ps, st
end

function noise_batch_step(samples, ps; ad, rate)
    candidates = map(samples) do (store, g, _, st)
        p, _, _ = parameter_step(store, g, ps, st; ad, rule = GradientDescent(rate))
        LenticulumCore._flatten(p)
    end
    return LenticulumCore._unflatten(sum(candidates) / length(samples), ps)
end

@testset "LP-005 learnable noise" begin
    @testset "initialisation, opt-in and parameter layout" begin
        Q0 = [1.2 0.3 -0.1; 0.3 0.8 0.2; -0.1 0.2 1.1]
        for mode in (false, :full, :diagonal)
            Qinit = mode === :diagonal ? Matrix(Diagonal(diag(Q0))) : Q0
            f = GaussianFactor(2 => 3; noise = Qinit, learn_noise = mode)
            ps, st = LuxCore.setup(Xoshiro(1), f)
            @test noise_covariance(f, ps) ≈ Qinit
            @test LuxCore.parameterlength(f) == length(LenticulumCore._flatten(ps))
            @test keys(ps) == (mode === false ? (:A, :b) : (:A, :b, :noise))
            @test first(energy(f, [0.2, -0.1], nothing, ones(3), ps, st)) ≈
                  ones(3) - ps.A * [0.2, -0.1] - ps.b
            prior = GaussianPrior(:z, zeros(3), Qinit; learn_noise = mode)
            pp, sp = LuxCore.setup(Xoshiro(1), prior)
            @test noise_covariance(prior, pp) ≈ Qinit
            @test islearnable(prior) == (mode !== false)
            @test LuxCore.parameterlength(prior) == length(LenticulumCore._flatten(pp))
        end
        @test_throws ArgumentError GaussianFactor(1 => 2; noise = Q0[1:2, 1:2], learn_noise = :diagonal)
        @test_throws ArgumentError GaussianFactor(1 => 1; noise = 0.001, learn_noise = true, eigenvalue_floor = 0.01)
        @test_throws ArgumentError GaussianFactor(1 => 1; learn_noise = true, eigenvalue_floor = 0)
        @test_throws ArgumentError GaussianFactor(1 => 1; learn_noise = :bad)
        @test_throws ArgumentError WishartPrecisionPrior(1, Matrix{Float64}(I, 2, 2))
        @test_throws ArgumentError LogNormalVariancePrior(0.0, -1.0)
    end

    @testset "normalisers, prior densities and parameter gradients" begin
        Q0 = [1.3 0.2; 0.2 0.9]
        x, y = [0.3], [0.7, -0.4]
        for mode in (:full, :diagonal), regularised in (false, true)
            Qinit = mode === :full ? Q0 : Matrix(Diagonal(diag(Q0)))
            prior = !regularised ? nothing : mode === :full ?
                WishartPrecisionPrior(6, [0.8 0.1; 0.1 0.6]) :
                LogNormalVariancePrior([0.1, -0.2], [0.7, 0.8])
            f = GaussianFactor(1 => 2; noise = Qinit, learn_noise = mode, noise_prior = prior)
            ps, st = LuxCore.setup(Xoshiro(2), f)
            r = residual(f, x, y, ps)
            F = first(scalar_free_energy(f, DiracBelief(x), DiracBelief(y), ps, st))
            nll = -logpdf(MvNormal(zeros(2), Qinit), r)
            penalty = prior === nothing ? 0.0 : prior isa WishartPrecisionPrior ?
                -logpdf(Wishart(prior.df, prior.scale), inv(Qinit)) :
                -sum(logpdf.(LogNormal.(prior.logmean, prior.logstd), diag(Qinit)))
            @test F ≈ nll + penalty
            @test first(scalar_energy(f, x, nothing, y, ps, st)) ≈ dot(r, Qinit \ r) / 2
            for ad in (AutoForwardDiff(), AutoZygote())
                @test LenticulumCore._flatten(energy_gradient(f, x, nothing, y, ps, st; ad)) ≈
                      LenticulumCore._flatten(energy_gradient(f, x, nothing, y, ps, st)) atol = 2e-7
                @test LenticulumCore._flatten(free_energy_gradient(f, DiracBelief(x), DiracBelief(y), ps, st; ad)) ≈
                      LenticulumCore._flatten(free_energy_gradient(f, DiracBelief(x), DiracBelief(y), ps, st)) atol = 2e-7
            end
            p = GaussianPrior(:z, [0.1, -0.2], Qinit; learn_noise = mode, noise_prior = prior)
            pp, sp = LuxCore.setup(Xoshiro(2), p)
            @test first(scalar_free_energy(p, TrivialBelief(), DiracBelief(y), pp, sp)) ≈
                  -logpdf(MvNormal(p.μ, Qinit), y) + penalty
            for ad in (AutoForwardDiff(), AutoZygote())
                @test LenticulumCore._flatten(free_energy_gradient(p, TrivialBelief(), DiracBelief(y), pp, sp; ad)) ≈
                      LenticulumCore._flatten(free_energy_gradient(p, TrivialBelief(), DiracBelief(y), pp, sp)) atol = 2e-7
            end
        end
    end

    @testset "current covariance in messages and distributional free energies" begin
        f = GaussianFactor(1 => 2; noise = 1.0, learn_noise = true)
        ps, st = LuxCore.setup(Xoshiro(9), f)
        ps = merge(ps, (noise = (log_diagonal = [0.2, -0.3], lower = [0.1]),))
        Q = noise_covariance(f, ps)
        fixed = GaussianFactor(1 => 2; noise = Q)
        gx, gy = Gaussian([0.2], fill(0.7, 1, 1)), Gaussian([0.5, -0.1], [0.9 0.1; 0.1 0.8])
        for (bx, by) in ((gx, gy), (gx, DiracBelief([0.5, -0.1])), (DiracBelief([0.2]), gy))
            @test first(scalar_free_energy(f, bx, by, ps, st)) ≈ first(scalar_free_energy(fixed, bx, by, ps, st))
            for ad in (AutoForwardDiff(), AutoZygote())
                @test LenticulumCore._flatten(free_energy_gradient(f, bx, by, ps, st; ad)) ≈
                      LenticulumCore._flatten(free_energy_gradient(f, bx, by, ps, st)) atol = 5e-7
            end
        end
        for target in (:x, :y)
            m, _ = factor_message(f, target, nothing, (x = gx, y = gy), TrivialBelief(), ps, st)
            reference, _ = factor_message(fixed, target, nothing, (x = gx, y = gy), TrivialBelief(), ps, st)
            @test m.η ≈ reference.η
            @test m.Λ ≈ reference.Λ
        end
        p = GaussianPrior(:y, zeros(2), Matrix{Float64}(I, 2, 2); learn_noise = true)
        pp = (noise = ps.noise,)
        for ad in (AutoForwardDiff(), AutoZygote())
            @test LenticulumCore._flatten(free_energy_gradient(p, TrivialBelief(), gy, pp, st; ad)) ≈
                  LenticulumCore._flatten(free_energy_gradient(p, TrivialBelief(), gy, pp, st)) atol = 5e-7
        end
        # GaussianPrior's opt-in learnability participates in graph parameter steps.
        b = GraphBuilder()
        variable!(b, :y, 2)
        factor!(b, :prior, p); factor!(b, :data, DataFactor(:y, [0.5, -0.1]))
        connect!(b, :prior, :y, :y; direction = Bidirectional())
        connect!(b, :data, :y, :y; direction = Emitting())
        g = build(b)
        gps, gst = LuxCore.setup(Xoshiro(9), g)
        store = MessageStore(g)
        _, gst = propagate!(store, g, tree_schedule(g), gps, gst)
        for ad in (AutoForwardDiff(), AutoZygote())
            newps, _, _ = parameter_step(store, g, gps, gst; ad, rule = GradientDescent(0.01))
            @test first(scalar_free_energy(store, g, newps, gst)) < first(scalar_free_energy(store, g, gps, gst))
        end
    end

    @testset "residual sample covariance MLE through parameter_step" begin
        S = [0.8 0.3; 0.3 1.4]
        C = sqrt(2) .* cholesky(Symmetric(S)).L
        A0, b0 = reshape([0.6, -0.4], 2, 1), [0.1, -0.2]
        data = [(x = [x], r = sign .* C[:, j]) for x in (-1.0, 1.0), sign in (-1, 1), j in 1:2]
        samplecov = sum(d.r * d.r' for d in data) / length(data)
        @test samplecov ≈ S
        for mode in (:full, :diagonal), ad in (AutoForwardDiff(), AutoZygote())
            f = GaussianFactor(1 => 2; noise = 2.0, learn_noise = mode,
                init_weight = (rng, o, i) -> copy(A0), init_bias = (rng, o) -> copy(b0))
            samples = [noise_sample_graph(f, d.x, A0 * d.x + b0 + d.r) for d in data]
            ps = first(samples)[3]
            for _ in 1:220
                ps = noise_batch_step(samples, ps; ad, rate = 0.15)
            end
            target = mode === :full ? S : Matrix(Diagonal(diag(S)))
            @test noise_covariance(f, ps.model) ≈ target atol = 2e-6
            @test ps.model.A ≈ A0 atol = 1e-12
            @test ps.model.b ≈ b0 atol = 1e-12
        end
    end

    @testset "pointwise variance inflation" begin
        f = GaussianFactor(1 => 1; noise = 1.0, learn_noise = :diagonal)
        ps, st = LuxCore.setup(Xoshiro(1), f)
        energies, variances = Float64[], Float64[]
        for k in 1:5
            p = merge(ps, (noise = (log_precision = [-10.0k],),))
            push!(energies, first(scalar_energy(f, [0.0], nothing, [1.0], p, st)))
            push!(variances, only(noise_covariance(f, p)))
        end
        @test all(diff(energies) .< 0)
        @test all(diff(variances) .> 0)
        # Gradient descent on noise only, keeping the nonzero residual fixed.
        for _ in 1:1000
            grad = energy_gradient(f, [0.0], nothing, [1.0], ps, st; ad = AutoForwardDiff())
            ps = merge(ps, (noise = (log_precision = ps.noise.log_precision - 0.2grad.noise.log_precision,),))
        end
        @test only(noise_covariance(f, ps)) > 100
        # F, unlike fit alone, diverges upward along the same inflation sequence.
        p = merge(ps, (noise = (log_precision = [-50.0],),))
        @test first(scalar_free_energy(f, DiracBelief([0.0]), DiracBelief([1.0]), p, st)) > 20
    end

    @testset "exact fit: floor, Wishart MAP and log-normal MAP" begin
        for mode in (:full, :diagonal), ad in (AutoForwardDiff(), AutoZygote())
            f = GaussianFactor(1 => 2; noise = 1.0, learn_noise = mode, eigenvalue_floor = 0.05,
                init_bias = (rng, o) -> zeros(o))
            store, g, ps, st = noise_sample_graph(f, [0.0], zeros(2))
            for _ in 1:80
                ps, st, _ = parameter_step(store, g, ps, st; ad, rule = GradientDescent(0.2))
            end
            @test noise_covariance(f, ps.model) ≈ 0.05I atol = 1e-12
            @test eigmin(Symmetric(noise_covariance(f, ps.model))) >= 0.05 - 1e-12
        end
        for ad in (AutoForwardDiff(), AutoZygote())
            V = [0.5 0.08; 0.08 0.3]
            prior = WishartPrecisionPrior(7, V)
            f = GaussianFactor(1 => 2; noise = 1.0, learn_noise = true, noise_prior = prior,
                eigenvalue_floor = 0.9, init_bias = (rng, o) -> zeros(o))
            store, g, ps, st = noise_sample_graph(f, [0.0], zeros(2))
            for _ in 1:250
                ps, st, _ = parameter_step(store, g, ps, st; ad, rule = GradientDescent(0.03))
            end
            # One exact-fit sample: P_MAP = (ν - d - 1 + N) V = 5V.
            @test noise_covariance(f, ps.model) ≈ inv(5V) atol = 2e-6
            prior_d = LogNormalVariancePrior([0.1, -0.2], [0.4, 0.5])
            p = GaussianPrior(:y, zeros(2), Matrix{Float64}(I, 2, 2);
                learn_noise = :diagonal, noise_prior = prior_d)
            pp, sp = LuxCore.setup(Xoshiro(1), p)
            for _ in 1:150
                grad = free_energy_gradient(p, TrivialBelief(), DiracBelief(zeros(2)), pp, sp; ad)
                pp = (noise = (log_precision = pp.noise.log_precision - 0.1grad.noise.log_precision,),)
            end
            @test diag(noise_covariance(p, pp)) ≈ exp.(prior_d.logmean - 1.5prior_d.logstd.^2) atol = 1e-9
        end
    end

    @testset "spectral floor derivatives away from the kink" begin
        # Rotated eigenspaces; one eigenvalue clipped, one free. Also repeated
        # eigenvalues entirely inside each smooth branch (no 0/0 in the rule).
        R = [cos(0.4) -sin(0.4); sin(0.4) cos(0.4)]
        for vals in ([0.1, 2.0], [0.1, 0.1], [2.0, 2.0])
            Q0 = R * Diagonal(vals) * R'
            f = GaussianFactor(1 => 2; noise = Q0, learn_noise = true, eigenvalue_floor = 0.01)
            ps, st = LuxCore.setup(Xoshiro(8), f)
            # Change the configured floor while retaining the same raw parameters.
            f2 = GaussianFactor(1 => 2; noise = 1.0, learn_noise = true, eigenvalue_floor = 0.5)
            for ad in (AutoForwardDiff(), AutoZygote())
                @test LenticulumCore._flatten(free_energy_gradient(f2, DiracBelief([0.3]), DiracBelief([0.7, -0.4]), ps, st; ad)) ≈
                      LenticulumCore._flatten(free_energy_gradient(f2, DiracBelief([0.3]), DiracBelief([0.7, -0.4]), ps, st)) atol = 5e-7
            end
        end
    end
end
