# I26 (Feature Backlog): the Bethe free energy where exact messages meet at a variable, against
# the dense −log Z (Mycelium `free_energy.md`, "Exact messages that meet at a variable"). Written
# by Codex from the specification alone (`../Lenticulum.jl-Review-2026-11-20/i26/spec.md`).

module HardOverlapOracle

using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning
using LinearAlgebra, Random, LuxCore, Test
using DifferentiationInterface, ForwardDiff, Zygote
using ADTypes: AutoForwardDiff, AutoZygote

function spd(rng, d)
    A = randn(rng, d, d) / sqrt(d)
    return Matrix(Symmetric(A * A' + 0.7I))
end

basis(rng, d) = Matrix(qr(randn(rng, d, d)).Q)
prior(rng, d) = (μ = randn(rng, d), Σ = spd(rng, d))
eye(T, d) = Matrix{T}(I, d, d)

# Condition a whitened Gaussian by thin QR. Working with L*sqrt(Σ) instead
# of its Gram matrix avoids squaring the condition number in the angle test.
function conditioned(p, L, y)
    S = cholesky(Symmetric(p.Σ)).L
    F = qr(Matrix((L * S)'))
    Q, R = Matrix(F.Q), F.R
    innovation = R' \ (y - L * p.μ)
    shift = S * Q
    μ = p.μ + shift * innovation
    residual = S - shift * Q'
    Σ = residual * residual'
    nlogZ = (length(y) * log(2π) + sum(abs2, innovation)) / 2 +
            sum(log ∘ abs, diag(R))
    return (; μ, Σ, nlogZ)
end

# Integrate unregularised coordinates using the coarea factor and project the
# right-hand side onto null(Cf'). Recover their moments by the affine solve.
function eliminated(p, Cp, Cf, c)
    P = nullspace(Cf')
    ref = conditioned(p, P' * Cp, P' * c)
    M, b = -(Cf \ Cp), Cf \ c
    free = (μ = M * ref.μ + b, Σ = M * ref.Σ * M')
    nlogZ = ref.nlogZ + sum(log, svdvals(Cf))
    return (; ref, free, nlogZ)
end

function add_prior!(gb, p)
    variable!(gb, :z, length(p.μ))
    factor!(gb, :prior, GaussianPrior(:out, p.μ, p.Σ))
    connect!(gb, :prior, :out, :z)
end

function clamp!(gb, v, y)
    factor!(gb, Symbol(:data_, v), DataFactor(:out, y))
    connect!(gb, Symbol(:data_, v), :out, v; direction = Emitting())
end

function relation!(gb, name, v, w, A, B, c)
    factor!(gb, name, HardConstraintFactor((u = size(A, 2), y = size(B, 2)), length(c);
                                          noise = zero(eltype(A))))
    connect!(gb, name, :u, v)
    connect!(gb, name, :y, w)
    return (A = (u = A, y = B), c = c)
end

# Observe a block of rows through its own relation and DataFactor.
function project!(gb, name, v, w, A, y)
    T = eltype(A)
    variable!(gb, w, length(y))
    clamp!(gb, w, y)
    return relation!(gb, name, v, w, A, -eye(T, length(y)), zeros(T, length(y)))
end

function observations!(gb, v, blocks, data)
    params = map(eachindex(blocks)) do i
        project!(gb, Symbol(:rel, i), v, Symbol(:y, i), blocks[i], data[i])
    end
    return NamedTuple{ntuple(i -> Symbol(:rel, i), length(blocks))}(Tuple(params))
end

function run(g, params; schedule = tree_schedule(g))
    ps, st = LuxCore.setup(Xoshiro(0), g)
    ps = merge(ps, params)
    marg, report, st, store = infer!(g, schedule, ps, st)
    @test report.converged
    return (; g, ps, st, marg, store)
end

function run(gb::GraphBuilder, params)
    return run(validate(build(gb)), params)
end

flatten(x::Real) = x
flatten(x::GradedEnergy) = sum(flatten, values(x.parts))

function check_energy(r, nlogZ; atol = 1e-8, clamps = ())
    F, _ = LenticulumCore.scalar_free_energy(r.store, r.g, r.ps, r.st)
    B, _ = Mycelium.bethe_free_energy(r.store, r.g, r.ps, r.st)
    @test F ≈ nlogZ atol = atol rtol = 0
    @test flatten(B[:factors]) + flatten(B[:variables]) ≈ F atol = atol rtol = 0
    for v in clamps
        @test iszero(B[:variables][v])
    end
    return F
end

function check_marginal(marg, ref; atol = 1e-8)
    e = extended(marg, length(ref.μ))
    @test size(e.D, 2) == 0
    @test e.μ ≈ ref.μ atol = atol rtol = 0
    @test e.Σ ≈ ref.Σ atol = atol rtol = 0
end

function direct(p, blocks, data)
    gb = GraphBuilder()
    add_prior!(gb, p)
    return run(gb, observations!(gb, :z, blocks, data))
end

# Generate oblique, full-row-rank blocks in a random orthonormal frame.
function oblique_blocks(rng, d, ranks)
    Q = basis(rng, d)
    L = copy(Q[:, 1:sum(ranks)]')
    L[2:end, :] .+= (0.3 .+ 0.4 .* rand(rng, size(L, 1) - 1)) .* L[1:1, :]
    L .*= 0.6 .+ rand(rng, size(L, 1))
    ends = cumsum([0; collect(ranks)])
    return ntuple(i -> L[(ends[i] + 1):ends[i + 1], :], length(ranks))
end

@testset verbose = true "I26 hard-message overlap" begin
    @testset "Two exact projections and an exact observation chain" begin
        rng = Xoshiro(2601)
        for d in (2, 3, 5), _ in 1:10
            p = prior(rng, d)
            k, h = oblique_blocks(rng, d, (1, 1))
            y, v, c = randn(rng, 1), randn(rng, 1), randn(rng, 1)
            ref = conditioned(p, [k; h], [y; v])
            r = direct(p, (k, h), (y, v))
            check_marginal(r.marg.z, ref)
            check_energy(r, ref.nlogZ; clamps = (:y1, :y2))

            gb = GraphBuilder()
            add_prior!(gb, p)
            rel1 = project!(gb, :rel1, :z, :y1, k, y)
            variable!(gb, :u, 1)
            rel2 = relation!(gb, :rel2, :z, :u, h, fill(-1.0, 1, 1), c)
            a = (rand(rng, Bool) ? 1 : -1) * (0.5 + rand(rng))
            tail = project!(gb, :tail, :u, :v, fill(a, 1, 1), v)
            r = run(gb, (; rel1, rel2, tail))
            ref = eliminated(p, [k; h; zeros(1, d)], reshape([0, -1, a], 3, 1),
                             [y; c; v])
            check_marginal(r.marg.z, ref.ref)
            check_marginal(r.marg.u, ref.free)
            check_energy(r, ref.nlogZ; clamps = (:y1, :v))
        end
    end

    @testset "Three and four messages of mixed rank" begin
        rng = Xoshiro(2602)
        for ranks in ((2, 1, 1), (2, 1, 1, 1)), extra in (0, 2), _ in 1:10
            d = sum(ranks) + extra
            p = prior(rng, d)
            blocks = oblique_blocks(rng, d, ranks)
            data = map(m -> randn(rng, m), ranks)
            ref = conditioned(p, vcat(blocks...), vcat(data...))
            r = direct(p, blocks, data)
            check_marginal(r.marg.z, ref)
            @test size(hard_directions(extended(r.marg.z, d)), 2) == sum(ranks)
            check_energy(r, ref.nlogZ; clamps = ntuple(i -> Symbol(:y, i), length(ranks)))
        end
    end

    @testset "Overlap at a variable without a prior" begin
        rng = Xoshiro(2603)
        for _ in 1:16
            d = rand(rng, 3:5)
            p = prior(rng, d)
            M = basis(rng, d) * Diagonal(0.5 .+ rand(rng, d)) * basis(rng, d)
            k, h = oblique_blocks(rng, d, (1, 1))
            y, v = randn(rng, 1), randn(rng, 1)
            gb = GraphBuilder()
            add_prior!(gb, p)
            variable!(gb, :w, d)
            link = relation!(gb, :link, :z, :w, -M, eye(Float64, d), zeros(d))
            params = observations!(gb, :w, (k, h), (y, v))
            r = run(gb, merge(params, (; link)))
            ref = eliminated(p, [-M; zeros(2, d)], [eye(Float64, d); k; h], [zeros(d); y; v])
            check_marginal(r.marg.z, ref.ref)
            check_marginal(r.marg.w, ref.free)
            check_energy(r, ref.nlogZ; clamps = (:y1, :y2))
        end
    end

    @testset "Overlap with SplitFactor" begin
        rng = Xoshiro(2604)
        for _ in 1:16
            d = rand(rng, 3:5)
            p = prior(rng, d)
            i, j = randperm(rng, d)[1:2]
            E = eye(Float64, d)[[i], :]
            k = reshape(randn(rng, d), 1, d)
            k[i], k[j] = 0.8, 1.1
            y, v = randn(rng, 1), randn(rng, 1)
            gb = GraphBuilder()
            add_prior!(gb, p)
            variable!(gb, :piece, 1)
            factor!(gb, :split, SplitFactor(:z, (piece = [i],); dim = d))
            connect!(gb, :split, :z, :z)
            connect!(gb, :split, :piece, :piece)
            clamp!(gb, :piece, y)
            obs = project!(gb, :obs, :z, :y, k, v)
            r = run(gb, (; obs))
            ref = conditioned(p, [E; k], [y; v])
            check_marginal(r.marg.z, ref)
            check_energy(r, ref.nlogZ; clamps = (:piece, :y))
        end
    end

    @testset "Overlap with ScaleShiftFactor and its gradient" begin
        rng = Xoshiro(2605)
        for trial in 1:12
            d = 4
            p = prior(rng, d)
            k, h = oblique_blocks(rng, d, (1, 1))
            y, v = randn(rng, 1), randn(rng, 1)
            ss = (b = randn(rng, d), logs = 0.25 .* randn(rng, d))
            gb = GraphBuilder()
            add_prior!(gb, p)
            variable!(gb, :x, d)
            factor!(gb, :ss, ScaleShiftFactor(:x, :z, d))
            connect!(gb, :ss, :x, :x)
            connect!(gb, :ss, :z, :z)
            rel1 = project!(gb, :rel1, :z, :y1, k, y)
            rel2 = project!(gb, :rel2, :x, :y2, h, v)
            r = run(gb, (; ss, rel1, rel2))

            # Integrate x directly in δ(x − Sz − b). Its coefficient is the
            # identity, so substitution leaves unit mass.
            S = Diagonal(exp.(ss.logs))
            ref = conditioned(p, [k; h * S], [y; v - h * ss.b])
            check_marginal(r.marg.z, ref)
            check_marginal(r.marg.x, (μ = S * ref.μ + ss.b, Σ = S * ref.Σ * S'))
            check_energy(r, ref.nlogZ; clamps = (:y1, :y2))

            if trial <= 6
                # Use the observation covariance directly for the independent
                # ForwardDiff objective, differentiating through both rows.
                function objective(θ)
                    b, logs = θ[1:d], θ[(d + 1):end]
                    L = [k; h * Diagonal(exp.(logs))]
                    C = cholesky(Symmetric(L * p.Σ * L'))
                    δ = [y; v - h * b] - L * p.μ
                    return (dot(δ, C \ δ) + logdet(C) + length(δ) * log(2π)) / 2
                end

                θ = [ss.b; ss.logs]
                @test objective(θ) ≈ ref.nlogZ atol = 1e-10
                expected = ForwardDiff.gradient(objective, θ)
                for ad in (nothing, AutoForwardDiff(), AutoZygote())
                    @testset "$ad" begin
                        ∇ = LenticulumCore.free_energy_gradient(r.store, r.g, r.ps, r.st; ad).ss
                        @test ∇.b ≈ expected[1:d] atol = 1e-7 rtol = 1e-8
                        @test ∇.logs ≈ expected[(d + 1):end] atol = 1e-7 rtol = 1e-8
                    end
                end
            end
        end
    end

    @testset "Overlap at a higher-degree variable with a soft child" begin
        rng = Xoshiro(2606)
        for observed in (false, true), _ in 1:10
            d, m = 4, 2
            p = prior(rng, d)
            k, h = oblique_blocks(rng, d, (1, 1))
            y, v, t = randn(rng, 1), randn(rng, 1), randn(rng, m)
            A, b, Q = randn(rng, m, d), randn(rng, m), spd(rng, m)
            gb = GraphBuilder()
            add_prior!(gb, p)
            params = observations!(gb, :z, (k, h), (y, v))
            variable!(gb, :w, m)
            factor!(gb, :soft, GaussianFactor(d => m; noise = Q))
            connect!(gb, :soft, :x, :z)
            connect!(gb, :soft, :y, :w)
            observed && clamp!(gb, :w, t)
            r = run(gb, merge(params, (soft = (; A, b),)))

            # Build the dense joint (z, w) before conditioning.
            joint = (μ = [p.μ; A * p.μ + b],
                     Σ = [p.Σ p.Σ * A'; A * p.Σ A * p.Σ * A' + Q])
            L = [k zeros(1, m); h zeros(1, m)]
            ref = observed ? conditioned(joint, [L; zeros(m, d) eye(Float64, m)], [y; v; t]) :
                             conditioned(joint, L, [y; v])
            for (name, ix) in ((:z, 1:d), (:w, (d + 1):(d + m)))
                check_marginal(r.marg[name], (μ = ref.μ[ix], Σ = ref.Σ[ix, ix]))
            end
            check_energy(r, ref.nlogZ; clamps = observed ? (:y1, :y2, :w) : (:y1, :y2))
        end
    end

    @testset "Orthogonal and nearly parallel exact directions" begin
        rng = Xoshiro(2607)
        for angle in (π / 2, 1e-3), _ in 1:12
            d = rand(rng, 3:5)
            p = prior(rng, d)
            Q = basis(rng, d)
            k, h = Matrix(Q[:, 1]'), Matrix((cos(angle) * Q[:, 1] + sin(angle) * Q[:, 2])')
            z = randn(rng, d)
            # A common witness keeps the innovation bounded even as the rows
            # approach one another. Gram solves may amplify roundoff by 1/sin²θ.
            y, v = k * z, h * z
            ref = conditioned(p, [k; h], [y; v])
            r = direct(p, (k, h), (y, v))
            atol = angle < 0.01 ? 2e-6 : 1e-8
            check_marginal(r.marg.z, ref; atol)
            check_energy(r, ref.nlogZ; atol, clamps = (:y1, :y2))
        end
    end

    @testset "Plate with tied exact relations" begin
        rng = Xoshiro(2608)
        for _ in 1:8
            d, replicas = 4, 5
            p = prior(rng, d)
            k, h = oblique_blocks(rng, d, (1, 1))
            Y, V = randn(rng, 1, replicas), randn(rng, 1, replicas)
            gb = GraphBuilder()
            plate!(gb, :r, replicas) do plate, i
                add_prior!(plate, p)
                observations!(plate, :z, (k, h), (Y[:, i], V[:, i]))
            end
            g = validate(build(gb))
            # Plate replicas have the same coefficients and distinct clamps.
            params = (rel1 = (A = (u = k, y = fill(-1.0, 1, 1)), c = [0.0]),
                      rel2 = (A = (u = h, y = fill(-1.0, 1, 1)), c = [0.0]))
            r = run(g, params; schedule = plate_schedule(g))
            refs = [conditioned(p, [k; h], [Y[:, i]; V[:, i]]) for i in 1:replicas]
            for i in 1:replicas
                check_marginal(r.marg[replica_name(:z, i)], refs[i])
            end
            clamps = Tuple(replica_name(v, i) for i in 1:replicas for v in (:y1, :y2))
            check_energy(r, sum(ref -> ref.nlogZ, refs); clamps)
        end
    end

    @testset "Shared exact direction is singular" begin
        rng = Xoshiro(2609)
        for trial in 1:9
            p = prior(rng, 3)
            k = if trial <= 8
                first(oblique_blocks(rng, 3, (1, 1)))
            else
                [0.0 1.0 0.0]
            end
            y = randn(rng, 1)
            r = try
                direct(p, (k, 2k), (y, 2y))
            catch err
                # Predicting a redundant clamp can round its zero variance
                # below zero. Which oblique draws hit this depends on BLAS.
                if trial <= 8 && err isa ArgumentError &&
                   startswith(err.msg, "Σ must be positive semidefinite; it has the eigenvalue -")
                    nothing
                else
                    rethrow()
                end
            end
            if r === nothing
                @test_broken r !== nothing
                continue
            end
            check_marginal(r.marg.z, conditioned(p, k, y))
            @test_throws ArgumentError LenticulumCore.scalar_free_energy(r.store, r.g, r.ps, r.st)
            @test_throws ArgumentError Mycelium.bethe_free_energy(r.store, r.g, r.ps, r.st)
        end
    end

    @testset "Float32 overlap" begin
        rng = Xoshiro(2610)
        for _ in 1:4
            p0 = prior(rng, 3)
            p32 = (μ = Float32.(p0.μ), Σ = Float32.(p0.Σ))
            blocks32 = map(A -> Float32.(A), oblique_blocks(rng, 3, (1, 1)))
            data32 = (randn(rng, Float32, 1), randn(rng, Float32, 1))
            p64 = (μ = Float64.(p32.μ), Σ = Float64.(p32.Σ))
            blocks64, data64 = map(A -> Float64.(A), blocks32), map(y -> Float64.(y), data32)
            ref = conditioned(p64, vcat(blocks64...), vcat(data64...))
            r64 = direct(p64, blocks64, data64)
            F64 = check_energy(r64, ref.nlogZ)
            r32 = direct(p32, blocks32, data32)
            e = extended(r32.marg.z, 3)
            @test eltype(e.μ) === eltype(e.Σ) === Float32
            check_marginal(r32.marg.z, ref; atol = 5e-5)
            F32 = check_energy(r32, ref.nlogZ; atol = 5e-5)
            @test F32 isa Float32
            @test F32 ≈ F64 atol = 5e-5 rtol = 0
        end
    end
end

# Review of I26 (Codex): regression tests of findings 1 and 3.
@testset verbose = true "I26: review" begin
    # Finding 1: two independent directions at a small angle t were refused as one common
    # direction (the Gram matrix's eigenvalue t²/2 fell below the rank tolerance at t = 1e-7).
    # Identity prior, zero data, rows [1 0 0] and [1 t 0]: det LLᵀ = t², −log Z = log(2π t).
    @testset "near-parallel directions, $T, t = $t0" for (T, t0) in ((Float64, 1e-6), (Float64, 1e-7), (Float32, 1e-3))
        t = T(t0)
        p = (μ = zeros(T, 3), Σ = eye(T, 3))
        r = direct(p, (T[1 0 0], T[1 t 0]), (zeros(T, 1), zeros(T, 1)))
        F = check_energy(r, log(2π * Float64(t)); atol = T == Float64 ? 1e-6 : 1e-2)
        @test F isa T
    end

    # Finding 3: Zygote through the coupled replay (`ExactCoupling`, `schedule`) failed in the
    # overlap term, a closure over a rebound tuple. Oblique exact observations meet at u, on the
    # latent side of a ScaleShiftFactor.
    @testset "coupled replay through the overlap, $(something(ad, "nothing"))" for ad in (nothing, AutoForwardDiff(), AutoZygote())
        d = 3
        μ, C = [0.4, -0.6, 0.1], [1.2 0.5 0.0; 0.5 1.8 0.3; 0.0 0.3 0.8]
        k, h = [1.0 0.2 0.6], [0.4 1.0 0.5]
        a, v = [0.1], [-0.6]
        b, logs = [0.2, 0.7, -0.3], [-0.4, 0.3, 0.1]
        coupling = ExactCoupling()
        gb = GraphBuilder()
        variable!(gb, :u, d); variable!(gb, :x, d)
        factor!(gb, :prior, GaussianPrior(:out, μ, C)); connect!(gb, :prior, :out, :u; coupling)
        factor!(gb, :scale, ScaleShiftFactor(:x, :u, d))
        connect!(gb, :scale, :u, :u; coupling); connect!(gb, :scale, :x, :x; coupling)
        params = (scale = (; b, logs),)
        for (rel, var, obs, A, y) in ((:one_obs, :u, :ya, k, a), (:two_obs, :x, :yv, h, v))
            variable!(gb, obs, 1)
            factor!(gb, Symbol(:data_, obs), DataFactor(:out, y))
            connect!(gb, Symbol(:data_, obs), :out, obs; direction = Emitting())
            factor!(gb, rel, HardConstraintFactor((z = d, y = 1), 1))
            connect!(gb, rel, :z, var; coupling); connect!(gb, rel, :y, obs; coupling)
            params = merge(params, NamedTuple{(rel,)}(((A = (z = A, y = -eye(Float64, 1)), c = [0.0]),)))
        end
        r = run(gb, params)
        function dense(θ)
            L = [k; h * Diagonal(exp.(θ[(d + 1):end]))]
            δ = [a; v - h * θ[1:d]] - L * μ
            V = cholesky(Symmetric(L * C * L'))
            return (dot(δ, V \ δ) + logdet(V) + 2 * log(2π)) / 2
        end
        θ = [b; logs]
        check_energy(r, dense(θ))
        expected = ForwardDiff.gradient(dense, θ)
        ∇ = LenticulumCore.free_energy_gradient(r.store, r.g, r.ps, r.st; ad, schedule = tree_schedule(r.g)).scale
        @test [∇.b; ∇.logs] ≈ expected atol = 1e-7 rtol = 1e-8
    end
end

end # module
