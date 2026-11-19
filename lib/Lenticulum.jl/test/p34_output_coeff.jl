# P34: the fixed output coefficient (`output_coeff`, default `I`) takes its number type from the
# parameters at use, not from the constructor's `noise`. A Boolean coefficient (`I`) is stored Boolean;
# other integers keep the rule before P34; a floating one keeps its own type and promotion decides.
using Zygote, ForwardDiff

p34_mean(b::DiracBelief) = b.value
p34_mean(b::MixtureBelief) = belief_mean(moment_match(b))
p34_mean(b) = belief_mean(b)
p34_cov(b::MixtureBelief) = belief_cov(moment_match(b))
p34_cov(b) = belief_cov(b)

@testset "P34: the fixed output coefficient's number type" begin
    nt = NamedTuple()
    # the three constructors, with the default noise (Float64) and the default `output_coeff = I`
    mk = (
        constraint = () -> LinearConstraintFactor((x = 2, y = 2), 2; output = :y),
        switching = () -> SwitchingConstraintFactor((x = 2, y = 2), 2, 2; output = :y),
        regime = () -> RegimeFactor((x = 2, y = 2), 2, 2; output = :y, learn_noise = true),
    )
    params(f::LinearConstraintFactor, T) = (A = (x = T[0.5 -0.2; 0.1 0.7],), c = T[0.1, -0.3])
    params(f::SwitchingConstraintFactor, T) =
        (A = (x = T[0.5 -0.2; 0.1 0.7],), b = T[0.1 -0.2; -0.3 0.4], logw = T[0.2, -0.1])
    params(f::RegimeFactor, T) = (A = (x = cat(T[0.5 -0.2; 0.1 0.7], T[-0.4 0.3; 0.2 0.6]; dims = 3),),
                                  b = T[0.1 -0.2; -0.3 0.4], logw = T[0.2, -0.1], L = zeros(T, 2, 2, 2))

    @testset "$name" for (name, make) in pairs(mk)
        f = make()
        @test f.output_coeff == Matrix(I, 2, 2)
        @test eltype(f.output_coeff) === Bool
        for T in (Float32, Float64)
            p = params(f, T)
            A = Lenticulum._coeffs(f, p)
            @test eltype(A.y) === T
            @test A.y == (f isa RegimeFactor ? repeat(Matrix{T}(I, 2, 2), 1, 1, 2) : Matrix{T}(I, 2, 2))
        end
        # Float64: the same messages as with an explicit Float64 identity (the stored type before P34), bit for bit
        f64 = f isa LinearConstraintFactor ? LinearConstraintFactor((x = 2, y = 2), 2; output = :y, output_coeff = Matrix(1.0I, 2, 2)) :
              f isa SwitchingConstraintFactor ? SwitchingConstraintFactor((x = 2, y = 2), 2, 2; output = :y, output_coeff = Matrix(1.0I, 2, 2)) :
              RegimeFactor((x = 2, y = 2), 2, 2; output = :y, learn_noise = true, output_coeff = Matrix(1.0I, 2, 2))
        @test eltype(f64.output_coeff) === Float64
        p = params(f, Float64)
        y = DiracBelief([0.3, -0.2])
        inputs = f isa LinearConstraintFactor ? (y = y,) : (y = y, (f.switch => CategoricalBelief([0.4, 0.6]),)...)
        x = Gaussian([0.2, 0.1], [0.8 0.1; 0.1 0.5])
        m1 = first(Mycelium.factor_message(f, :x, nothing, inputs, x, p, nt))
        m2 = first(Mycelium.factor_message(f64, :x, nothing, inputs, x, p, nt))
        @test p34_mean(m1) == p34_mean(m2)
        @test p34_cov(m1) == p34_cov(m2)
        E1 = first(Mycelium.local_free_energy(f, merge(inputs, (x = x,)), p, nt))
        E2 = first(Mycelium.local_free_energy(f64, merge(inputs, (x = x,)), p, nt))
        @test E1.parts == E2.parts
        # Float32 parameters and data with the default Float64 noise: a Float32 message
        if f isa RegimeFactor   # with learned noise the parameter L, not the noise, is the covariance
            p32 = params(f, Float32)
            inputs32 = (y = DiracBelief(Float32[0.3, -0.2]), k = CategoricalBelief(Float32[0.4, 0.6]))
            x32 = Gaussian(Float32[0.2, 0.1], Float32[0.8 0.1; 0.1 0.5])
            m32 = first(Mycelium.factor_message(f, :x, nothing, inputs32, x32, p32, nt))
            @test eltype(p34_mean(m32)) === Float32
            @test eltype(p34_cov(m32)) === Float32
            @test p34_mean(m32) ≈ p34_mean(m1) rtol = 1e-5
            E32 = first(Mycelium.local_free_energy(f, merge(inputs32, (x = x32,)), p32, nt))
            @test all(v -> v isa Float32 || v === false, values(E32.parts))
        end
        # Zygote: the fixed coefficient is a constant; the gradient is that of the explicit identity
        g1 = Zygote.gradient(q -> sum(values(first(Mycelium.local_free_energy(f, merge(inputs, (x = x,)), q, nt)).parts)), p)[1]
        g2 = Zygote.gradient(q -> sum(values(first(Mycelium.local_free_energy(f64, merge(inputs, (x = x,)), q, nt)).parts)), p)[1]
        @test g1.A.x ≈ g2.A.x
        @test !haskey(g1.A, :y) || g1.A.y === nothing
        # ForwardDiff: dual coefficients and offsets; the Boolean identity becomes duals with zero partials
        off = f isa LinearConstraintFactor ? :c : :b
        function energy(ff, v)
            na = length(p.A.x)
            q = merge(p, (A = (x = reshape(v[1:na], size(p.A.x)),),),
                      NamedTuple{(off,)}((reshape(v[(na + 1):end], size(p[off])),)))
            return sum(values(first(Mycelium.local_free_energy(ff, merge(inputs, (x = x,)), q, nt)).parts))
        end
        v0 = [vec(p.A.x); vec(p[off])]
        d1 = ForwardDiff.gradient(v -> energy(f, v), v0)
        d2 = ForwardDiff.gradient(v -> energy(f64, v), v0)
        @test d1 == d2
        @test d1[1:length(p.A.x)] ≈ vec(g2.A.x)
    end

    @testset "other integer and floating coefficients" begin
        # other integers keep the rule before P34: the noise's type for `2I`, Float64 for an integer matrix
        f = LinearConstraintFactor((x = 1, y = 1), 1; output = :y, output_coeff = 2I)
        @test eltype(f.output_coeff) === Float64
        f = LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.1f0, output = :y, output_coeff = 2I)
        @test eltype(f.output_coeff) === Float32
        @test Lenticulum._coeffs(f, (A = (x = fill(0.5f0, 1, 1),), c = Float32[0])).y == fill(2.0f0, 1, 1)
        # review of P34 (Codex): an integer cast to Float32 at use rounds (16777217 → 16777216), so two
        # factors with equal M-step signatures (pooled when tied) would get different coefficients
        fi = LinearConstraintFactor((x = 1, y = 1), 1; output = :y, output_coeff = fill(16777217, 1, 1))
        ff = LinearConstraintFactor((x = 1, y = 1), 1; output = :y, output_coeff = fill(16777217.0, 1, 1))
        p32 = (A = (x = fill(0.5f0, 1, 1),), c = Float32[0])
        @test MyceliumLearning.m_step_signature(fi) == MyceliumLearning.m_step_signature(ff)
        @test Lenticulum._coeffs(fi, p32).y == Lenticulum._coeffs(ff, p32).y
        # a Boolean identity and a floating one: equal signatures, equal coefficients at use
        fb = LinearConstraintFactor((x = 1, y = 1), 1; output = :y)
        fo = LinearConstraintFactor((x = 1, y = 1), 1; output = :y, output_coeff = fill(1.0, 1, 1))
        @test MyceliumLearning.m_step_signature(fb) == MyceliumLearning.m_step_signature(fo)
        @test Lenticulum._coeffs(fb, p32).y == Lenticulum._coeffs(fo, p32).y
        @test eltype(LinearConstraintFactor((x = 1, y = 1), 1; output = :y, output_coeff = trues(1, 1)).output_coeff) === Bool
        # a floating coefficient keeps its type; with Float32 parameters promotion makes the message Float64
        f = LinearConstraintFactor((x = 1, y = 1), 1; noise = 0.1f0, output = :y, output_coeff = fill(2.0, 1, 1))
        @test eltype(f.output_coeff) === Float64
        @test eltype(Lenticulum._coeffs(f, (A = (x = fill(0.5f0, 1, 1),), c = Float32[0])).y) === Float64
        f = RegimeFactor((x = 1, y = 1), 2, 1; output = :y, output_coeff = 0.5f0 * I)
        @test eltype(f.output_coeff) === Float32
        # no learned channel: the offsets decide
        f = LinearConstraintFactor((y = 1,), 1; output = :y)
        @test eltype(Lenticulum._coeffs(f, (A = NamedTuple(), c = Float32[0])).y) === Float32
        @test eltype(Lenticulum._coeffs(f, (A = NamedTuple(), c = [0.0])).y) === Float64
        @test_throws ArgumentError LinearConstraintFactor((x = 1, y = 1), 1; output = :y, output_coeff = 0I)
        @test_throws ArgumentError LinearConstraintFactor((x = 2, y = 2), 2; output = :y, output_coeff = [1 1; 1 1])
    end

    @testset "Float32 EM on a regime plate with the default noise" begin
        # the plate of `bench/float32_cluster_plates.jl` (:reg1) without `noise = T(0.1)`
        function plate(T, N)
            rng = Xoshiro(11)
            x = 6 .* rand(rng, N) .- 3
            k = [rand(rng) < 0.5 ? 1 : 2 for _ in 1:N]
            y = [1.0, -1.0][k] .* x .+ [0.0, 0.5][k] .+ [0.2, 0.5][k] .* randn(rng, N)
            b = GraphBuilder()
            for i in 1:N
                r(s) = replica_name(s, i)
                variable!(b, r(:x), 1); variable!(b, r(:y), 1); variable!(b, r(:k), 2)
                factor!(b, r(:dx), DataFactor(:x, T[x[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
                factor!(b, r(:dy), DataFactor(:y, T[y[i]])); connect!(b, r(:dy), :y, r(:y); direction = Emitting())
                factor!(b, r(:reg), RegimeFactor((y = 1, x = 1), 2, 1; learn_noise = true, output = :y); params = :reg)
                connect!(b, r(:reg), :y, r(:y)); connect!(b, r(:reg), :x, r(:x)); connect!(b, r(:reg), :k, r(:k))
            end
            g = as_plate(build(b), :pts, N)
            ps = merge(NamedTuple(s => nt for i in 1:N for s in (replica_name(:dx, i), replica_name(:dy, i))),
                       (reg = (A = (x = reshape(T[-0.5, 0.5], 1, 1, 2),), b = zeros(T, 1, 2), logw = log.(T[0.5, 0.5]),
                               L = zeros(T, 1, 1, 2)),))
            return g, ps
        end
        res = map((Float64, Float32)) do T
            g, ps = plate(T, 60)
            fit_em(g, ps, nt; schedule = plate_schedule(g), iters = 10, tol = 0)
        end
        (p64, F64, _), (p32, F32, _) = res
        @test eltype(F32) === Float32
        @test eltype(p32.reg.A.x) === Float32
        @test eltype(p32.reg.L) === Float32
        @test last(F32) ≈ last(F64) rtol = 1e-4
        @test p32.reg.A.x ≈ p64.reg.A.x rtol = 1e-3
    end
end
