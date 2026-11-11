# The relation factors take ExtendedGaussian inputs (gaussian.md, "Extended Gaussian inputs").
#
# Oracle: HardConstraintFactor with a positive definite noise computes the same message by an
# independent path (pushforward and preimage of extended Gaussians, hard_constraint.md). Inputs
# are random extended Gaussians with flat, hard and Gaussian directions at once, which no
# GaussianBelief can represent. Second, for every relation factor, an input given as a
# GaussianBelief (proper or improper) or a DiracBelief and the same input as `extended(b)` give
# the same message. Third, a graph of GaussianFactors and one HardConstraintFactor against the
# dense conditioned Gaussian.

module ExtendedInputsOracle
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# an extended Gaussian on ℝⁿ with kf flat, kh hard and n − kf − kh Gaussian directions
function random_eg(rng, n; kf = rand(rng, 0:n), kh = rand(rng, 0:(n - kf)))
    B = Matrix(qr(randn(rng, n, n)).Q)
    D = B[:, 1:kf]
    V = B[:, (kf + kh + 1):n]
    Σ = V * spd(rng, size(V, 2)) * V'
    return ExtendedGaussian(randn(rng, n), (Σ + Σ') / 2, D)
end

# the messages as extended Gaussians, compared on μ, Σ and the projector onto the flat directions
asext(b, n) = extended(b, n)
same(a::ExtendedGaussian, b::ExtendedGaussian; atol, rtol = 1e-9) =
    isapprox(a.μ, b.μ; atol, rtol) && isapprox(a.Σ, b.Σ; atol, rtol) && isapprox(a.D * a.D', b.D * b.D'; atol)
same(a, b; atol) = same(asext(a, 0), asext(b, 0); atol)
function same_belief(a, b; atol = 1e-8)
    a isa LenticulumCore.MixtureBelief && b isa LenticulumCore.MixtureBelief &&
        return isapprox(a.logw, b.logw; atol) && all(same_belief(x, y; atol) for (x, y) in zip(a.components, b.components))
    a isa LenticulumCore.CategoricalBelief && return isapprox(LenticulumCore.probabilities(a), LenticulumCore.probabilities(b); atol)
    a isa LenticulumCore.TrivialBelief && return b isa LenticulumCore.TrivialBelief
    return isapprox(a.η, b.η; atol) && isapprox(a.Λ, b.Λ; atol) && isproper(a) == isproper(b)
end

function random_input(rng, n, kind)
    kind === :dirac && return DiracBelief(randn(rng, n))
    kind === :proper && return Gaussian(randn(rng, n), spd(rng, n))
    kind === :trivial && return TrivialBelief()
    V = Matrix(qr(randn(rng, n, n)).Q)[:, 1:max(1, n - 1)]
    Λ = V * spd(rng, size(V, 2)) * V'
    return GaussianBelief(Λ * randn(rng, n), (Λ + Λ') / 2, true)
end
end # module

@testset "LinearConstraintFactor with extended-Gaussian inputs: HardConstraintFactor's messages" begin
    EO = ExtendedInputsOracle
    rng = Xoshiro(51)
    nmsg = 0; nhard = 0; nflat = 0
    for trial in 1:80
        k = rand(rng, 2:3)
        names = (:a, :b, :c)[1:k]
        dims = NamedTuple{names}(Tuple(rand(rng, 1:3) for _ in 1:k))
        m = rand(rng, 1:3)
        Q = EO.spd(rng, m)
        soft = LinearConstraintFactor(dims, m; noise = Q)
        hard = HardConstraintFactor(dims, m; noise = Q)
        ps = LuxCore.initialparameters(Xoshiro(trial), soft)
        for t in names
            others = Tuple(n for n in names if n !== t)
            ins = Tuple(EO.random_eg(rng, dims[n]) for n in others)
            nhard += count(e -> size(e.D, 2) + rank(e.Σ) < length(e.μ), ins)
            nflat += count(e -> 0 < size(e.D, 2) < length(e.μ), ins)
            inputs = NamedTuple{others}(ins)
            ms, _ = Mycelium.factor_message(soft, t, nothing, inputs, nothing, ps, NamedTuple())
            mh, _ = Mycelium.factor_message(hard, t, nothing, inputs, nothing, ps, NamedTuple())
            @test EO.same(extended(ms, dims[t]), mh; atol = 1e-7)
            nmsg += 1
        end
    end
    @test nmsg > 150 && nhard > 30 && nflat > 30
end

@testset "GaussianFactor with extended-Gaussian inputs: HardConstraintFactor's messages" begin
    EO = ExtendedInputsOracle
    rng = Xoshiro(52)
    for trial in 1:60
        nin, nout = rand(rng, 1:3), rand(rng, 1:3)
        Q = EO.spd(rng, nout)
        f = GaussianFactor(nin => nout; noise = Q)
        ps = LuxCore.initialparameters(Xoshiro(trial), f)
        hard = HardConstraintFactor((x = nin, y = nout), nout; noise = Q)
        hps = (A = (x = -ps.A, y = Matrix(1.0I, nout, nout)), c = ps.b)
        for (t, src, n) in ((:y, :x, nin), (:x, :y, nout))
            e = EO.random_eg(rng, n)
            inputs = NamedTuple{(src,)}((e,))
            mg, _ = Mycelium.factor_message(f, t, nothing, inputs, nothing, ps, NamedTuple())
            mh, _ = Mycelium.factor_message(hard, t, nothing, inputs, nothing, hps, NamedTuple())
            @test EO.same(extended(mg, t === :x ? nin : nout), mh; atol = 1e-7)
        end
    end
end

@testset "relation factors: an input and the same input as an extended Gaussian give one message" begin
    EO = ExtendedInputsOracle
    rng = Xoshiro(53)
    kinds = (:dirac, :proper, :improper, :trivial)
    for trial in 1:40
        # LinearConstraintFactor and SwitchingConstraintFactor
        dims = (a = rand(rng, 1:3), b = rand(rng, 1:3))
        m = rand(rng, 1:3)
        for f in (LinearConstraintFactor(dims, m; noise = EO.spd(rng, m)),
                  SwitchingConstraintFactor(dims, 2, m; noise = EO.spd(rng, m)))
            ps = LuxCore.initialparameters(Xoshiro(trial), f)
            for t in (:a, :b)
                o = t === :a ? :b : :a
                b = EO.random_input(rng, dims[o], rand(rng, kinds))
                b isa TrivialBelief && continue
                m1, _ = Mycelium.factor_message(f, t, nothing, NamedTuple{(o,)}((b,)), nothing, ps, NamedTuple())
                m2, _ = Mycelium.factor_message(f, t, nothing, NamedTuple{(o,)}((extended(b, dims[o]),)), nothing, ps, NamedTuple())
                @test EO.same_belief(m1, m2)
            end
        end
        # GaussianFactor, both directions
        nin, nout = rand(rng, 1:3), rand(rng, 1:3)
        g = GaussianFactor(nin => nout; noise = EO.spd(rng, nout))
        gps = LuxCore.initialparameters(Xoshiro(trial), g)
        for (t, src, n) in ((:y, :x, nin), (:x, :y, nout))
            b = EO.random_input(rng, n, rand(rng, (:dirac, :proper, :improper)))
            m1, _ = Mycelium.factor_message(g, t, nothing, NamedTuple{(src,)}((b,)), nothing, gps, NamedTuple())
            m2, _ = Mycelium.factor_message(g, t, nothing, NamedTuple{(src,)}((extended(b, n),)), nothing, gps, NamedTuple())
            @test EO.same_belief(m1, m2)
        end
        # NonlinearFactor: forwards from a proper input, backwards from an improper one around a prior
        h = NonlinearFactor(x -> [sin(x[1]) + x[2], x[1] * x[2]], 2 => 2; noise = 0.3)
        b = EO.random_input(rng, 2, :proper)
        m1 = first(Mycelium.factor_message(h, :y, nothing, (x = b,), nothing, NamedTuple(), NamedTuple()))
        m2 = first(Mycelium.factor_message(h, :y, nothing, (x = extended(b),), nothing, NamedTuple(), NamedTuple()))
        @test EO.same_belief(m1, m2)
        b = EO.random_input(rng, 2, rand(rng, (:proper, :improper, :dirac)))
        π = Gaussian(randn(rng, 2), EO.spd(rng, 2))
        m1 = first(Mycelium.factor_message(h, :x, nothing, (y = b,), π, NamedTuple(), NamedTuple()))
        m2 = first(Mycelium.factor_message(h, :x, nothing, (y = extended(b),), π, NamedTuple(), NamedTuple()))
        @test EO.same_belief(m1, m2)
    end
end

@testset "a GaussianFactor under a HardConstraintFactor: infer! against the conditioned Gaussian" begin
    EO = ExtendedInputsOracle
    rng = Xoshiro(54)
    for trial in 1:20
        # a tree: x ~ N(μ₀, Σ₀);  u = A x + b + ε;  v ~ N(μᵥ, Σᵥ);  Cu u + Cv v = c exactly
        n, nu, nv = rand(rng, 1:3), rand(rng, 1:3), rand(rng, 1:2)
        m = rand(rng, 1:(nu + nv))
        μ0, Σ0, μv, Σv = randn(rng, n), EO.spd(rng, n), randn(rng, nv), EO.spd(rng, nv)
        Q = EO.spd(rng, nu)
        gb = GraphBuilder()
        variable!(gb, :x, n); variable!(gb, :u, nu); variable!(gb, :v, nv)
        factor!(gb, :px, GaussianPrior(:x, μ0, Σ0)); connect!(gb, :px, :x, :x)
        factor!(gb, :pv, GaussianPrior(:x, μv, Σv)); connect!(gb, :pv, :x, :v)
        factor!(gb, :g, GaussianFactor(n => nu; noise = Q)); connect!(gb, :g, :x, :x); connect!(gb, :g, :y, :u)
        factor!(gb, :rel, HardConstraintFactor((u = nu, v = nv), m))
        connect!(gb, :rel, :u, :u); connect!(gb, :rel, :v, :v)
        g = validate(build(gb))
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        # the dense joint of (x, u, v), conditioned on Cu u + Cv v = c
        A, b = ps.g.A, ps.g.b
        L = vcat(Matrix(1.0I, n, n), A)
        μ = vcat(μ0, A * μ0 + b, μv)
        Σ = zeros(n + nu + nv, n + nu + nv)
        Σ[1:(n + nu), 1:(n + nu)] = L * Σ0 * L'
        Σ[(n + 1):(n + nu), (n + 1):(n + nu)] += Q
        Σ[(n + nu + 1):end, (n + nu + 1):end] = Σv
        C = hcat(zeros(m, n), ps.rel.A.u, ps.rel.A.v)
        K = Σ * C' * pinv(C * Σ * C')
        μp, Σp = μ - K * (C * μ - ps.rel.c), Σ - K * C * Σ
        marg, rep, _ = infer!(g, tree_schedule(g), ps, st)
        @test rep.converged
        for (v, r) in ((:x, 1:n), (:u, (n + 1):(n + nu)), (:v, (n + nu + 1):(n + nu + nv)))
            e = extended(marg[v], length(r))
            @test size(e.D, 2) == 0
            @test e.μ ≈ μp[r] atol = 1e-7 rtol = 1e-9
            @test e.Σ ≈ Σp[r, r] atol = 1e-7 rtol = 1e-9
        end
    end
end

@testset "extended-Gaussian inputs: refusals" begin
    f = GaussianFactor(2 => 1; noise = 0.5)
    ps = (A = ones(1, 2), b = [0.0])
    @test_throws DimensionMismatch Mycelium.factor_message(f, :y, nothing, (x = ExtendedGaussian(zeros(3), Matrix(1.0I, 3, 3)),), nothing, ps, NamedTuple())
    # the free energy's canonical form refuses hard directions, takes flat ones as zero precision
    @test_throws ArgumentError Lenticulum._canon(ExtendedGaussian([1.0], zeros(1, 1)), 1)
    η, Λ = Lenticulum._canon(ExtendedGaussian([1.0, 0.0], [2.0 0; 0 0], reshape([0.0, 1.0], 2, 1)), 2)
    @test η ≈ [0.5, 0.0] && Λ ≈ [0.5 0; 0 0]
end
