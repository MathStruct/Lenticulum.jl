module ScaleShiftOracle

using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Random, LuxCore, Test
using LenticulumCore: DiracBelief, TrivialBelief, Observed, Unobserved, Latent
using ADTypes: AutoForwardDiff, AutoZygote
using DifferentiationInterface, ForwardDiff, Zygote

include(joinpath(@__DIR__, "../../LenticulumCore.jl/test/support/factor_conformance.jl"))

const tree_kinds = (:forward_point, :reverse_point, :noisy, :chain, :unobserved)
const backends = (nothing, AutoForwardDiff(), AutoZygote())

function spd(rng, n)
    M = randn(rng, n, n) / sqrt(n)
    return Matrix(Symmetric(M * M' + 0.5I))
end

nlogpdf(y, μ, Σ) = nlogpdf(y, μ, cholesky(Symmetric(Σ)))
function nlogpdf(y, μ, C::Cholesky)
    r = y - μ
    return (dot(r, C \ r) + logdet(C) + length(r) * log(2π)) / 2
end

diagscale(p, groups) = Diagonal(exp.(p.logs[groups]))
shift(p, groups) = p.b[groups]
random_parameters(rng, groups) =
    (b = randn(rng, maximum(groups)), logs = 0.4 .* randn(rng, maximum(groups)))

function random_case(rng, kind, grouped)
    d = rand(rng, 3:5)
    groups = grouped ? shuffle(rng, [mod1(i, 2) for i in 1:d]) : collect(1:d)
    groups2 = grouped ? collect(1:d) : fill(1, d)
    root = kind === :reverse_point ? :x : :z
    prior = (μ = randn(rng, d), Σ = spd(rng, d))
    obs = if kind === :unobserved
        nothing
    else
        name = kind === :reverse_point ? :z : kind in (:noisy, :chain) ? :y : :x
        (name = name, value = randn(rng, kind === :noisy ? d - 1 : d))
    end
    link = kind === :noisy ?
        (A = randn(rng, d - 1, d) / sqrt(d), b = randn(rng, d - 1), Q = spd(rng, d - 1)) : nothing
    p = (ss = random_parameters(rng, groups),)
    if kind === :chain
        p = merge(p, (ss2 = random_parameters(rng, groups2),))
    end
    return (; kind, groups, groups2, root, prior, obs, link), p
end

function add_prior!(gb, name, v, prior)
    factor!(gb, name, GaussianPrior(:out, prior.μ, prior.Σ))
    connect!(gb, name, :out, v)
end

function observe!(gb, name, v, value)
    factor!(gb, name, DataFactor(:out, value))
    connect!(gb, name, :out, v; direction = Emitting())
end

function graph(c; rule = BeliefPropagation())
    n = length(c.groups)
    gb = GraphBuilder()
    variable!(gb, :z, n)
    variable!(gb, :x, n)
    factor!(gb, :ss, ScaleShiftFactor(:x, :z, n; groups = c.groups); rule)
    connect!(gb, :ss, :x, :x)
    connect!(gb, :ss, :z, :z)
    if c.kind === :chain
        variable!(gb, :y, n)
        factor!(gb, :ss2, ScaleShiftFactor(:x, :z, n; groups = c.groups2))
        connect!(gb, :ss2, :z, :x)
        connect!(gb, :ss2, :x, :y)
    elseif c.kind === :noisy
        m = length(c.link.b)
        variable!(gb, :y, m)
        factor!(gb, :link, GaussianFactor(n => m; noise = c.link.Q,
            init_weight = (rng, o, i) -> c.link.A, init_bias = (rng, o) -> c.link.b))
        connect!(gb, :link, :x, :x)
        connect!(gb, :link, :y, :y)
    end
    add_prior!(gb, :prior, c.root, c.prior)
    if c.obs !== nothing
        observe!(gb, :data, c.obs.name, c.obs.value)
    end
    return validate(build(gb))
end

function run(g, p = (;); schedule = tree_schedule(g))
    ps, st = LuxCore.setup(Xoshiro(0), g)
    ps = merge(ps, p)
    marg, report, st, store = infer!(g, schedule, ps, st)
    @test report.converged
    F = first(LenticulumCore.scalar_free_energy(store, g, ps, st))
    return (; marg, F, store, ps, st)
end

# Stack each variable's affine map from the Gaussian root. The noisy link adds an
# independent Gaussian innovation. The joint may be singular, but its observed block
# has positive definite covariance. Condition the full joint on that block.
function dense(c, p)
    d = length(c.groups)
    S, b = diagscale(p.ss, c.groups), shift(p.ss, c.groups)
    identity = Matrix{Float64}(I, d, d)
    if c.root === :x
        A = vcat(inv(S), identity)
        offset = vcat(-(S \ b), zeros(d))
        # Integrating δ(x − Sz − b) over z gives 1/det(S). The Gaussian
        # obtained by pushing the x prior to z must carry that total mass.
        nlogZ = sum(p.ss.logs[c.groups])
    else
        A = vcat(identity, S)
        offset = vcat(zeros(d), b)
        nlogZ = zero(eltype(p.ss.logs))
    end
    ranges = (z = 1:d, x = (d + 1):(2d))
    if c.kind === :chain
        S2, b2 = diagscale(p.ss2, c.groups2), shift(p.ss2, c.groups2)
        A = vcat(A, S2 * S)
        offset = vcat(offset, S2 * b + b2)
        ranges = merge(ranges, (y = (2d + 1):(3d),))
    elseif c.kind === :noisy
        A = vcat(A, c.link.A * S)
        offset = vcat(offset, c.link.A * b + c.link.b)
        ranges = merge(ranges, (y = (2d + 1):length(offset),))
    end
    μ = A * c.prior.μ + offset
    Σ = A * c.prior.Σ * A'
    if c.kind === :noisy
        Σ[ranges.y, ranges.y] += c.link.Q
    end
    if c.obs !== nothing
        ix = ranges[c.obs.name]
        C = cholesky(Symmetric(Σ[ix, ix]))
        nlogZ += nlogpdf(c.obs.value, μ[ix], C)
        cross = Σ[:, ix]
        μ += cross * (C \ (c.obs.value - μ[ix]))
        Σ -= cross * (C \ cross')
    end
    return (; μ, Σ, nlogZ, ranges)
end

function check_marginals(marg, ref)
    for (v, ix) in pairs(ref.ranges)
        e = extended(marg[v], length(ix))
        @test size(e.D, 2) == 0
        @test e.μ ≈ ref.μ[ix] atol = 1e-7 rtol = 1e-9
        @test e.Σ ≈ ref.Σ[ix, ix] atol = 1e-7 rtol = 1e-9
    end
end

function dense_gradient(c, p, key)
    q = p[key]
    G = length(q.b)
    function objective(θ)
        params = (b = θ[1:G], logs = θ[(G + 1):end])
        return dense(c, merge(p, NamedTuple{(key,)}((params,)))).nlogZ
    end
    ∇ = ForwardDiff.gradient(objective, vcat(q.b, q.logs))
    return (b = ∇[1:G], logs = ∇[(G + 1):end])
end

function polarity(f, target, role)
    names = Tuple(channelname(c) for c in LenticulumCore.channels(f))
    return LenticulumCore.Polarity(NamedTuple{names}(map(n -> n === target ? Unobserved() : role, names)))
end

function message(f, target, inputs, ps; role = Observed())
    p = polarity(f, target, role)
    return first(Mycelium.factor_message(f, target, p, inputs, nothing, ps, (;)))
end

function random_input(rng, n, kind)
    μ = randn(rng, n)
    kind === :point && return DiracBelief(μ)
    kind === :proper && return Gaussian(μ, spd(rng, n))
    kind === :trivial && return TrivialBelief()
    kind in (:nothing, :omitted) && return nothing
    if kind === :mixture
        return MixtureBelief([Gaussian(μ, spd(rng, n)), Gaussian(randn(rng, n), spd(rng, n))],
                             [0.3, 0.7])
    end
    Q = Matrix(qr(randn(rng, n, n)).Q)
    if kind === :improper
        V = Q[:, 2:end]
        Λ = Matrix(Symmetric(V * spd(rng, n - 1) * V'))
        return GaussianBelief(Λ * μ, Λ, true)
    end
    # One flat direction and one hard direction, both oblique to the coordinates.
    V = Q[:, 3:end]
    return ExtendedGaussian(μ, Matrix(Symmetric(V * spd(rng, n - 2) * V')), Q[:, 1:1])
end

input_moments(input, d) = extended(input, d)
function input_moments(input::MixtureBelief, d)
    # Include the between-component covariance when collapsing the mixture.
    components = zip(mixture_weights(input), input.components)
    μ = sum(w * belief_mean(b) for (w, b) in components)
    Σ = sum(components) do (w, b)
        r = belief_mean(b) - μ
        return w * (belief_cov(b) + r * r')
    end
    return (μ = μ, Σ = Σ, D = zeros(d, 0))
end

function affine_message(input, S, b)
    e = input_moments(input, length(b))
    # Compare canonical moments on the orthogonal complement of the transported flat space.
    D = S * e.D
    flat = D * pinv(D)
    P = I - flat
    return (μ = P * (S * e.μ + b), Σ = P * S * e.Σ * S' * P', flat = flat)
end

function check_message(actual, expected)
    @test actual isa ExtendedGaussian
    @test actual.μ ≈ expected.μ atol = 1e-7 rtol = 1e-9
    @test actual.Σ ≈ expected.Σ atol = 1e-7 rtol = 1e-9
    @test actual.D * actual.D' ≈ expected.flat atol = 1e-7 rtol = 1e-9
end

function iid_graph(f, X)
    gb = GraphBuilder()
    for (i, sample) in enumerate(eachcol(X))
        x, z = Symbol(:x, i), Symbol(:z, i)
        ss = Symbol(:ss, i)
        d = length(sample)
        variable!(gb, x, d)
        variable!(gb, z, d)
        factor!(gb, ss, f; params = :shared)
        connect!(gb, ss, :x, x)
        connect!(gb, ss, :z, z)
        add_prior!(gb, Symbol(:prior, i), z, (μ = zeros(d), Σ = Matrix{Float64}(I, d, d)))
        observe!(gb, Symbol(:data, i), x, collect(sample))
    end
    return validate(build(gb))
end

@testset "ScaleShiftFactor" begin
    @testset "Random trees against dense Gaussian marginals and evidence" begin
        rng = Xoshiro(301)
        counts = Dict((kind, grouped) => 0 for kind in tree_kinds for grouped in (false, true))
        for kind in tree_kinds, grouped in (false, true), trial in 1:4
            @testset "$kind grouped=$grouped trial=$trial" begin
                c, p = random_case(rng, kind, grouped)
                counts[(c.kind, maximum(c.groups) < length(c.groups))] += 1
                result = run(graph(c), p)
                ref = dense(c, p)
                check_marginals(result.marg, ref)
                @test result.F ≈ ref.nlogZ atol = 1e-7 rtol = 1e-9
            end
        end
        for kind in tree_kinds, grouped in (false, true)
            @test counts[(kind, grouped)] == 4
        end
    end

    @testset "Observed x contributes the coordinate log-Jacobian" begin
        groups = [2, 1, 2]
        prior = (μ = [0.6, -0.2, 0.7], Σ = [1.5 0.2 0.4; 0.2 0.9 0.1; 0.4 0.1 1.2])
        p = (b = [0.3, -0.6], logs = log.([0.7, 1.8]))
        x0 = [0.1, 0.8, 0.2]
        z0 = diagscale(p, groups) \ (x0 - shift(p, groups))
        c = (kind = :forward_point, root = :z, groups = groups, prior = prior,
             obs = (name = :x, value = x0))
        Fx = run(graph(c), (ss = p,)).F
        gb = GraphBuilder()
        variable!(gb, :z, length(z0))
        add_prior!(gb, :prior, :z, prior)
        observe!(gb, :data, :z, z0)
        Fz = run(validate(build(gb))).F
        logjac = sum(p.logs[groups])
        ref = nlogpdf(z0, prior.μ, prior.Σ)
        @test Fz ≈ ref atol = 1e-9
        @test Fx ≈ ref + logjac atol = 1e-9
        @test Fx - Fz ≈ logjac atol = 1e-9
    end

    @testset "Parameter gradients against dense negative log evidence" begin
        rng = Xoshiro(302)
        for kind in tree_kinds, trial in 1:3
            @testset "$kind trial=$trial" begin
                c, p = random_case(rng, kind, isodd(trial))
                g = graph(c)
                result = run(g, p)
                reference = map(key -> dense_gradient(c, p, key), keys(p))
                for ad in backends
                    @testset "$(typeof(ad))" begin
                        ∇ = LenticulumCore.free_energy_gradient(result.store, g, result.ps, result.st; ad)
                        for (key, exact) in zip(keys(p), reference), field in (:b, :logs)
                            @test ∇[key][field] ≈ exact[field] atol = 1e-7 rtol = 2e-6
                        end
                    end
                end
            end
        end
    end

    @testset "Tied maximum likelihood on independent observations" begin
        rng = Xoshiro(303)
        groups = [2, 1, 2, 1, 2]
        d, N = length(groups), 32
        f = ScaleShiftFactor(:x, :z, d; groups)
        truth = (b = [-0.4, 1.2], logs = log.([0.7, 1.6]))
        X = shift(truth, groups) .+ diagscale(truth, groups) * randn(rng, d, N)
        g = iid_graph(f, X)
        # The replicas form a forest, so use flooding to obtain their tree fixed points.
        fit(p) = run(g, (shared = p,); schedule = flooding_schedule(g))
        gradient(result; ad = nothing) =
            LenticulumCore.free_energy_gradient(result.store, g, result.ps, result.st; ad).shared
        objective(p) = sum(eachcol(X)) do sample
            z = diagscale(p, groups) \ (sample - shift(p, groups))
            return (dot(z, z) + d * log(2π)) / 2 + sum(p.logs[groups])
        end
        p = scale_shift_parameters(f, X)
        optimum = fit(p)
        @test optimum.F ≈ objective(p) atol = 1e-7 rtol = 1e-9
        for ad in backends
            ∇ = gradient(optimum; ad)
            @test ∇.b ≈ zeros(2) atol = 1e-7
            @test ∇.logs ≈ zeros(2) atol = 1e-7
        end

        q = (b = p.b + [0.35, -0.25], logs = p.logs + [0.2, -0.15])
        perturbed = fit(q)
        ∇ = gradient(perturbed; ad = AutoForwardDiff())
        @test norm(vcat(∇.b, ∇.logs)) > 1e-2
        step = 0.1 / (d * N)
        updated = (b = q.b - step * ∇.b, logs = q.logs - step * ∇.logs)
        after = fit(updated)
        @test perturbed.F ≈ objective(q) atol = 1e-7 rtol = 1e-9
        @test after.F ≈ objective(updated) atol = 1e-7 rtol = 1e-9
        @test optimum.F < after.F < perturbed.F
    end

    @testset "Extrinsic messages and factor free energy" begin
        rng = Xoshiro(304)
        groups = [2, 1, 3, 1, 2]
        d = length(groups)
        f = ScaleShiftFactor(:x, :z, d; groups)
        h = HardConstraintFactor((x = d, z = d), d)
        kinds = (:point, :proper, :improper, :extended, :mixture, :trivial, :nothing, :omitted)
        for trial in 1:5
            p = random_parameters(rng, groups)
            S, b = diagscale(p, groups), shift(p, groups)
            for (target, source) in ((:x, :z), (:z, :x)), kind in kinds
                @testset "$kind to $target trial=$trial" begin
                    input = random_input(rng, d, kind)
                    inputs = kind === :omitted ? (;) : NamedTuple{(source,)}((input,))
                    role = kind in (:nothing, :omitted, :trivial) ? Latent() : Observed()
                    M, offset = target === :x ? (S, b) : (inv(S), -(S \ b))
                    check_message(message(f, target, inputs, p; role), affine_message(input, M, offset))
                end
            end

            # One or two soft cavities keep the relation proper. A single point
            # fixes its support, exercising both directions of the reduction.
            x = random_input(rng, d, :proper)
            z = random_input(rng, d, :proper)
            hp = (A = (x = Matrix{Float64}(I, d, d), z = -Matrix(S)), c = b)
            for msgs in ((; x, z), (; z), (; x),
                         (x = DiracBelief(randn(rng, d)), z = z),
                         (x = x, z = DiracBelief(randn(rng, d))))
                local_F = first(Mycelium.local_free_energy(f, msgs, p, (;)))
                belief = Mycelium.factor_belief(f, msgs, p, (;))
                frozen_F = first(Mycelium.belief_free_energy(f, belief, p, (;)))
                hard_F = first(Mycelium.local_free_energy(h, msgs, hp, (;)))
                @test _energy_isapprox(frozen_F, local_F)
                @test scalarise(scalarisation(f), local_F) ≈
                    scalarise(scalarisation(h), hard_F) atol = 1e-9 rtol = 1e-9
            end
        end
    end

    @testset "Grouped population moments and invalid data" begin
        groups = [2, 1, 2, 1, 2]
        f = ScaleShiftFactor(:x, :z, length(groups); groups)
        X = [2.0 4 1; 3 3 3; -1 2 0; 1 3 2; 6 3 5]
        p = scale_shift_parameters(f, X)
        @test keys(p) == (:b, :logs)
        @test p.b isa Vector && p.logs isa Vector
        @test length(p.b) == length(p.logs) == 2
        for g in 1:2
            samples = X[groups .== g, :]
            m = sum(samples) / length(samples)
            v = sum(abs2, samples .- m) / length(samples)
            @test p.b[g] ≈ m atol = 1e-12
            @test exp(p.logs[g]) ≈ sqrt(v) atol = 1e-12
        end
        constant_group = copy(X)
        constant_group[groups .== 2, :] .= 4.0
        @test_throws ArgumentError scale_shift_parameters(f, constant_group)
        @test_throws DimensionMismatch scale_shift_parameters(f, X[1:4, :])
        @test_throws ArgumentError scale_shift_parameters(f, X[:, 1:0])
    end

    @testset "Construction, traits, polarities, conformance and mean field" begin
        @test_throws ArgumentError ScaleShiftFactor(:x, :x, 2)
        @test_throws ArgumentError ScaleShiftFactor(:x, :z, 0)
        @test_throws ArgumentError ScaleShiftFactor(:x, :z, -1)
        @test_throws ArgumentError ScaleShiftFactor(:x, :z, 1.5)
        for groups in ((1, 2), reshape([1, 2], 1, 2), [1.0, 2.0], [true, true],
                       Integer[1, true], [0, 1], [-1, 1], [2, 2], [1, 3])
            @test_throws ArgumentError ScaleShiftFactor(:x, :z, 2; groups)
        end
        for groups in (Int[], [1], [1, 2, 3])
            @test_throws Union{ArgumentError, DimensionMismatch} ScaleShiftFactor(:x, :z, 2; groups)
        end

        rng = Xoshiro(0)
        @test LuxCore.initialparameters(rng, ScaleShiftFactor(:x, :z, 3)) ==
            (b = zeros(3), logs = zeros(3))
        f = ScaleShiftFactor(:x, :z, 3; groups = [2, 1, 2])
        ps, st = LuxCore.setup(rng, f)
        @test ps == (b = zeros(2), logs = zeros(2))
        @test isempty(st)
        @test LuxCore.parameterlength(f) == 4
        @test LenticulumCore.islearnable(f) === true
        @test Mycelium.exact_messages(f) === true
        @test coupling_precision(f, ps) === :hard

        chans = LenticulumCore.channels(f)
        names = Tuple(channelname(c) for c in chans)
        @test length(chans) == 2
        @test Dict(channelname(c) => channelspace(c) for c in chans) == Dict(:x => 3, :z => 3)
        allowed = supported_polarities(f)
        @test length(allowed) == 4
        roles = (Observed(), Unobserved(), Latent())
        for assignment in Iterators.product(roles, roles)
            p = LenticulumCore.Polarity(NamedTuple{names}(assignment))
            valid = count(r -> r isa Unobserved, assignment) == 1
            @test supports_polarity(f, p) == valid
            @test (p in allowed) == valid
        end

        p = (b = [0.5, -0.25], logs = log.([0.75, 1.5]))
        groups = [2, 1, 2]
        S, b = diagscale(p, groups), shift(p, groups)
        x0 = [0.2, 0.7, -0.4]
        inputs = (x = DiracBelief(x0),)
        π = Gaussian(zeros(3), Matrix{Float64}(I, 3, 3))
        expected = ExtendedGaussian(S \ (x0 - b), zeros(3, 3))
        check_factor_conformance(f, polarity(f, :z, Observed()); inputs, π, ps = p, st,
            msgs = (x = inputs.x, z = π), direct_message = true,
            expected_site = expected, expected_posterior = expected,
            pointwise_gap = "an exact relation has no finite pointwise energy",
            game_gap = "the specification defines the graph free energy")

        c = (kind = :forward_point, root = :z, groups = groups, obs = (name = :x, value = x0),
             prior = (μ = zeros(3), Σ = Matrix{Float64}(I, 3, 3)))
        @test_throws ArgumentError begin
            g = graph(c; rule = MeanField())
            params, state = LuxCore.setup(rng, g)
            infer!(g, tree_schedule(g), merge(params, (ss = p,)), state)
        end
    end

    @testset "Float32 direct messages, stored messages and marginals" begin
        groups = [2, 1, 2]
        f = ScaleShiftFactor(:x, :z, 3; groups)
        p = (b = Float32[0.3, -0.5], logs = Float32[0.2, -0.4])
        μ = Float32[0.4, -0.3, 0.2]
        Σ = Float32[1.2 0.2 0.1; 0.2 0.9 -0.1; 0.1 -0.1 0.8]
        for (target, source) in ((:x, :z), (:z, :x))
            for input in (Gaussian(μ, Σ), DiracBelief(μ), ExtendedGaussian(μ, Σ))
                inputs = NamedTuple{(source,)}((input,))
                @test message(f, target, inputs, p) isa ExtendedGaussian{Float32}
            end
            @test message(f, target, (;), p; role = Latent()) isa ExtendedGaussian{Float32}
        end
        for obs in (nothing, (name = :x, value = Float32[0.2, 0.9, -0.7]))
            kind = obs === nothing ? :unobserved : :forward_point
            c = (; kind, root = :z, groups, obs, prior = (; μ, Σ))
            result = run(graph(c), (ss = p,))
            for v in (:x, :z)
                @test extended(result.marg[v], 3) isa ExtendedGaussian{Float32}
            end
            for messages in (result.store.to_variable, result.store.to_factor), slot in messages
                slot === nothing && continue
                belief = slot.belief
                if belief isa GaussianBelief
                    @test eltype(belief.η) === eltype(belief.Λ) === Float32
                elseif belief isa ExtendedGaussian
                    @test belief isa ExtendedGaussian{Float32}
                elseif belief isa DiracBelief
                    @test eltype(belief.value) === Float32
                else
                    @test belief isa TrivialBelief || belief === nothing
                end
            end
        end
    end
end


# Regression tests of the review of U3 (`../Lenticulum.jl-Review-2026-11-20/u3`, problem note LP-070),
# each failing on the code before its fix.
@testset "ScaleShiftFactor: review of U3" begin
    # finding 2: a Float32 scale of 1e-3 kept a variance of 1e-6 next to 1 (the pushforward and
    # preimage of HardConstraintFactor cut it to a hard direction, and the reverse to a flat one)
    f = ScaleShiftFactor(:x, :z, 2)
    p = (b = Float32[0, 0], logs = Float32[log(1f-3), 0])
    Σ = Float32[1 0; 0 1]
    mx = message(f, :x, (z = Gaussian(Float32[1, 1], Σ),), p)
    @test mx isa ExtendedGaussian{Float32}
    @test size(mx.D, 2) == 0
    @test mx.Σ[1, 1] ≈ 1f-6 rtol = 1f-4
    mz = message(f, :z, (x = Gaussian(Float32[1, 1], Float32[4f-6 0; 0 1]),), p)
    @test size(mz.D, 2) == 0
    @test mz.Σ[1, 1] ≈ 4 rtol = 1f-4

    # finding 3: the inputs' energy, uncentred, lost its change to cancellation at a mean of 1e8
    # (the default gradient was 5e5 for −1) and was NaN at 1e155
    for (c, x0) in ((1e8, 1e8 + 1), (1e155, 1e155))
        g = let gb = GraphBuilder()
            variable!(gb, :x, 1); variable!(gb, :z, 1)
            factor!(gb, :ss, ScaleShiftFactor(:x, :z, 1)); connect!(gb, :ss, :x, :x); connect!(gb, :ss, :z, :z)
            factor!(gb, :pz, GaussianPrior(:out, [c], fill(1.0, 1, 1))); connect!(gb, :pz, :out, :z)
            factor!(gb, :data, DataFactor(:out, [x0])); connect!(gb, :data, :out, :x)
            validate(build(gb))
        end
        ps, st = LuxCore.setup(Xoshiro(0), g)
        _, _, st, s = infer!(g, tree_schedule(g), ps, st)
        msgs = (x = DiracBelief([x0]), z = Gaussian([c], fill(1.0, 1, 1)))
        ff = ScaleShiftFactor(:x, :z, 1)
        local_F = first(Mycelium.local_free_energy(ff, msgs, ps.ss, NamedTuple()))
        belief_F = first(Mycelium.belief_free_energy(ff, Mycelium.factor_belief(ff, msgs, ps.ss, NamedTuple()), ps.ss, NamedTuple()))
        @test LenticulumCore.scalarise(LenticulumCore.scalarisation(ff), belief_F) ==
              LenticulumCore.scalarise(LenticulumCore.scalarisation(ff), local_F)
        # −log p = ½ ((x0 − b) e^{−logs} − c)² + logs + const: at b = logs = 0, with r = x0 − c,
        # ∂b = −r and ∂logs = 1 − r x0
        r = x0 - c
        gr = LenticulumCore.free_energy_gradient(s, g, ps, st; ad = AutoForwardDiff()).ss
        @test gr.b[1] ≈ -r atol = 1e-8
        @test gr.logs[1] ≈ 1 - r * x0 rtol = 1e-12
        c == 1e8 && @test LenticulumCore.free_energy_gradient(s, g, ps, st).ss.b[1] ≈ -r atol = 1e-2
    end

    # finding 5: scale_shift_parameters overflowed an Int64 sum and squared deviations of 1e200,
    # underflowed at 1e-200, and took a rounded spread of equal entries for a scale
    f1 = ScaleShiftFactor(:x, :z, 1)
    for (row, m, σ) in (([9223372036854773760, 9223372036854771712], 9.223372036854772736e18, 1024.0),
                        ([-1e200, 1e200], 0.0, 1e200), ([-1e-200, 1e-200], 0.0, 1e-200))
        q = scale_shift_parameters(f1, reshape(row, 1, :))
        @test q.b[1] ≈ m rtol = 1e-15
        @test exp(q.logs[1]) ≈ σ rtol = 1e-12
    end
    @test_throws ArgumentError scale_shift_parameters(f1, [0.1 0.1 0.1])

    # finding 6: arguments the constructor took or refused by dispatch
    @test_throws ArgumentError ScaleShiftFactor(:x, :z, 2; groups = nothing)
    for dim in (nothing, "2", 2 + im)
        @test_throws ArgumentError ScaleShiftFactor(:x, :z, dim)
    end
    @test ScaleShiftFactor(:x, :z, 2; groups = Any[1, 2]).groups == [1, 2]
end

end # module ScaleShiftOracle
