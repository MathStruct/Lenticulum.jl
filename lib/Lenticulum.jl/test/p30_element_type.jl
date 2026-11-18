# P30: paired draws, independently assembled dense precision, and all stored messages.
module P30ElementType
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, LuxCore
using ForwardDiff, Zygote
const nt = NamedTuple()

# `upstream` marks the checks that went through LenticulumCore's extended-Gaussian algebra (Float64
# before P30); they are strict now, and the flag only documents where each check's type comes from.
checktype(ok, upstream = false) = @test ok
function checkbelief(b, T; upstream = false)
    if b isa GaussianBelief
        checktype(eltype(b.η) === T, upstream)
        checktype(eltype(b.Λ) === T, upstream)
    elseif b isa ExtendedGaussian
        checktype(b isa ExtendedGaussian{T}, upstream)
        checktype(eltype(b.μ) === T, upstream)
        checktype(eltype(b.Σ) === T, upstream)
        checktype(eltype(b.D) === T, upstream)
    elseif b isa DiracBelief
        checktype(eltype(b.value) === T, upstream)
    else
        @test b isa TrivialBelief || b === nothing
    end
end

# Full-rank square tree edges; an observed leaf supplies data, with no Float64 literals in
# the Float32 graph. The dense oracle is built only in Float64, from the same random draw.
function tree(T, seed; rectangular = false, mixed = false, learn = false)
    rng = Xoshiro(seed)
    dims = rectangular ? [2, 3, 2, 2] : fill(2, 4)
    parents = rectangular ? [0, 1, 1, 2] : [0; [rand(rng, 1:i-1) for i in 2:4]]
    A = [i == 1 ? zeros(0, 0) : randn(rng, dims[i], dims[parents[i]]) / 3 for i in 1:4]
    b = [randn(rng, d) / 3 for d in dims]
    μ = randn(rng, 2) / 3
    y = randn(rng, 2)
    gb = GraphBuilder()
    for i in 1:4; variable!(gb, Symbol(:x, i), dims[i]); end
    factor!(gb, :p, GaussianPrior(:x, T.(μ), Matrix{T}(I, 2, 2)))
    connect!(gb, :p, :x, :x1)
    for i in 2:4
        f = Symbol(:f, i)
        factor!(gb, f, GaussianFactor(dims[parents[i]] => dims[i]; noise = T(0.7), learn_noise = learn))
        connect!(gb, f, :x, Symbol(:x, parents[i])); connect!(gb, f, :y, Symbol(:x, i))
    end
    factor!(gb, :d, DataFactor(:x, mixed ? y : T.(y))); connect!(gb, :d, :x, :x4; direction = Emitting())
    g = validate(build(gb)); ps = LuxCore.initialparameters(Xoshiro(0), g)
    ps = merge(ps, NamedTuple{(:f2, :f3, :f4)}(Tuple(merge(getfield(ps, Symbol(:f, i)), (A = T.(A[i]), b = T.(b[i]))) for i in 2:4)))
    st = LuxCore.initialstates(Xoshiro(0), g)
    off = cumsum([0; dims]); rg(i) = off[i]+1:off[i+1]
    Λ = zeros(off[end], off[end]); η = zeros(off[end]); c = dot(μ, μ)/2 + log(2π)
    Λ[rg(1), rg(1)] = I(2); η[rg(1)] = μ
    for i in 2:4
        G = zeros(dims[i], off[end]); G[:, rg(parents[i])] = -A[i]; G[:, rg(i)] = I(dims[i])
        Λ += G' * G / 0.7; η += G' * b[i] / 0.7
        c += dot(b[i], b[i])/1.4 + dims[i]*log(2π*0.7)/2
    end
    free = 1:off[4]; obs = rg(4)
    P = Symmetric(Λ[free, free]); h = η[free] - Λ[free, obs]*y
    m = P \ h; S = inv(P)
    F = c + dot(y, Λ[obs, obs]*y)/2 - dot(η[obs], y) - dot(h, m)/2 + (logdet(P)-length(free)*log(2π))/2
    return g, ps, st, m, S, F, rg
end

function rungraph(g, ps, st; plate = false)
    m, rep, _, s = infer!(g, plate ? plate_schedule(g) : tree_schedule(g), ps, st)
    @test rep.converged
    return m, s, first(LenticulumCore.scalar_free_energy(s, g, ps, st))
end
function energies(s, g, ps, st, T; upstream_variables = false, upstream_factors = false)
    ff = first(factor_free_energies(s, g, ps, st)); vc = variable_corrections(s, g)
    scalar(x) = x isa T || x === false
    for e in values(ff.parts)
        if e isa GradedEnergy
            for part in values(e.parts); checktype(scalar(part), upstream_factors); end
        else
            checktype(scalar(e), upstream_factors)
        end
    end
    for e in values(vc.parts)
        checktype(scalar(e), upstream_variables)
    end
end

# Two independent priors conditioned on a noisy or exact linear relation.
function relation(T; hard = false, plate = false, gaussian = false)
    gb = GraphBuilder()
    function add!(p, i)
        variable!(p, :x, 2); variable!(p, :y, 2)
        for c in (:x, :y)
            factor!(p, Symbol(:p, c), GaussianPrior(c, T[0.1, -0.2], Matrix{T}(I, 2, 2)))
            connect!(p, Symbol(:p, c), c, c)
        end
        f = gaussian ? GaussianFactor(2 => 2; noise = T(0.4)) : hard ?
            HardConstraintFactor((x = 2, y = 2), 1; noise = zero(T)) :
            LinearConstraintFactor((x = 2, y = 2), 1; noise = T(0.4))
        factor!(p, :f, f); connect!(p, :f, :x, :x); connect!(p, :f, :y, :y)
    end
    plate ? plate!((p, i) -> add!(p, i), gb, :rep, 3) : add!(gb, 1)
    g = validate(build(gb)); ps = LuxCore.initialparameters(Xoshiro(9), g)
    ps = merge(ps, (f = gaussian ? (A = T[0.8 0.1; 0.2 0.7], b = T[0.1, 0.2]) :
                                   (A = (x = T[1 0.2], y = T[-0.3 1]), c = T[0.3]),))
    return g, ps, LuxCore.initialstates(Xoshiro(9), g)
end

@testset "P30 linear-Gaussian element types" begin
@testset "P30 constraints and plates" begin
    for hard in (false, true), plate in (false, true), gaussian in (false, true)
        hard && gaussian && continue
        g32, ps32, st32 = relation(Float32; hard, plate, gaussian)
        g64, ps64, st64 = relation(Float64; hard, plate, gaussian)
        m32, s32, F32 = rungraph(g32, ps32, st32; plate)
        m64, _, F64 = rungraph(g64, ps64, st64; plate)
        # Dense conditioning of the independent priors; exact equality uses its conditional
        # Gaussian formula, noisy relations use the joint canonical solve.
        μ0 = repeat([0.1, -0.2], 2)
        G = gaussian ? hcat(-ps64.f.A, Matrix{Float64}(I, 2, 2)) : hcat(ps64.f.A.x, ps64.f.A.y)
        c = gaussian ? ps64.f.b : ps64.f.c
        V = G * G' + (hard ? 0I : 0.4I)
        μdense = μ0 + G' * (V \ (c - G * μ0))
        Σdense = Matrix{Float64}(I, 4, 4) - G' * (V \ G)
        Fdense = (dot(c - G * μ0, V \ (c - G * μ0)) + logdet(2π .* V)) / 2
        @test F64 ≈ (plate ? 3 : 1) * Fdense atol=1e-9
        for v in g32.variables
            r = startswith(String(v.name), "x") ? (1:2) : (3:4)
            @test belief_mean(m64[v.name]) ≈ μdense[r] atol=1e-9
            @test belief_cov(m64[v.name]) ≈ Σdense[r, r] atol=1e-9
            checkbelief(m32[v.name], Float32; upstream = !gaussian)
            @test belief_mean(m32[v.name]) ≈ belief_mean(m64[v.name]) rtol=1e-3 atol=1e-5
            @test belief_cov(m32[v.name]) ≈ belief_cov(m64[v.name]) rtol=1e-3 atol=1e-5
        end
        for (to_variable, msgs) in ((true, s32.to_variable), (false, s32.to_factor)), (i, msg) in enumerate(msgs)
            # Only hard-factor outputs and the cavities they supply to prior factors traverse
            # Core.preimage. The priors' own outputs and the hard factor's inputs stay strict.
            is_relation = g32.factors[g32.edges[i].factor].factor isa HardConstraintFactor
            upstream = hard && (to_variable == is_relation)
            checkbelief(msg === nothing ? nothing : msg.belief, Float32; upstream)
        end
        checktype(F32 isa Float32, !gaussian)
        @test F32 ≈ F64 rtol=1e-3 atol=1e-5
        energies(s32, g32, ps32, st32, Float32; upstream_variables = !gaussian, upstream_factors = hard)
    end
    # No input: Gaussian/soft-constraint flat buffers must use the factor's scalar type.
    f = GaussianFactor(1 => 2; noise = 0.4f0)
    ps = (A = reshape(Float32[1, 0], 2, 1), b = zeros(Float32, 2))
    msg = first(Mycelium.factor_message(f, :y, nothing, nt, nothing, ps, nt))
    checkbelief(msg, Float32)
    @test msg isa ExtendedGaussian && size(msg.D, 2) == 1

    h = HardConstraintFactor((x = 1, y = 2), 2; noise = 0f0)
    hp = (A = (x = reshape(Float32[1, 0], 2, 1), y = Matrix{Float32}(I, 2, 2)), c = zeros(Float32, 2))
    hm = first(Mycelium.factor_message(h, :y, nothing, nt, nothing, hp, nt))
    @test hm isa ExtendedGaussian{Float32}
end

@testset "P30 frozen beliefs, mean field and noise" begin
    for T in (Float32, Float64)
        scalar(e) = e isa T || e === false
        a = Matrix{T}(I, 2, 2)
        x = DiracBelief(T[0.1, 0.2]); y = DiracBelief(T[0.3, -0.1])
        ps = (A = a, b = zeros(T, 2))
        for f in (GaussianFactor(2 => 2; noise = T(0.4)),)
            msgs = (x = x, y = y)
            b = Mycelium.factor_belief(f, msgs, ps, nt)
            @test eltype(last(first(b.points))) === T
            F = first(Mycelium.belief_free_energy(f, b, ps, nt))
            @test all(scalar, values(F.parts))
            @test all(scalar, values(first(Mycelium.local_free_energy(f, msgs, ps, nt)).parts))
            blocks = (x = Mycelium.PointBlock(repeat(x.value, 1, 3)), y = Mycelium.PointBlock(repeat(y.value, 1, 3)))
            bs = Mycelium.factor_beliefs(fill(f, 3), blocks, ps, nt)
            pointsets = bs isa AbstractVector ? (b.points for b in bs) : (bs.points,)
            @test all(points -> eltype(last(first(points))) === T, pointsets)
            Fs = first(Mycelium.belief_free_energies(fill(f, 3), bs, ps, nt))
            @test all(E -> all(scalar, values(E.parts)), Fs)
        end
        default_output = LinearConstraintFactor((x = 2, y = 2), 2; noise = T(0.4), output = :y)
        @test eltype(default_output.output_coeff) === T
        # Shared helper compatibility: RegimeFactor's field remains Float64 (out of scope).
        @test eltype(RegimeFactor((x = 2, y = 2), 2, 2; output = :y).output_coeff) === Float64
        f = LinearConstraintFactor((x = 2, y = 2), 2; noise = T(0.4), output = :y, output_coeff = a)
        p = (A = (x = a,), c = zeros(T, 2))
        checkbelief(first(Mycelium.mean_field_message(f, :x, nothing, (y = y,), p, nt)), T)
        @test all(scalar, values(first(Mycelium.mean_field_energy(f, (x = x, y = y), p, nt)).parts))
        # Unequal covariances force the per-replica buffer kernel, rather than the shared metric.
        S = [a .* T(i) for i in 1:3]
        input = Mycelium.GaussianBlock(ones(T, 2, 3), S, falses(3))
        block = first(Mycelium.factor_messages(fill(f, 3), :x, nothing, (y = input,), fill(nothing, 3), p, nt))
        for i in 1:3
            msg = Mycelium.replica(block, i)
            checkbelief(msg, T)
            oracle = first(Mycelium.factor_message(f, :x, nothing, (y = Mycelium.replica(input, i),), nothing, p, nt))
            @test msg.η ≈ oracle.η
            @test msg.Λ ≈ oracle.Λ
        end
        for mode in (true, :diagonal), hyper in (false, true)
            prior = !hyper ? nothing : mode === true ? WishartPrecisionPrior(T(4), a) : LogNormalVariancePrior(T(0), T(1))
            f = GaussianPrior(:x, zeros(T, 2), a; learn_noise = mode, noise_prior = prior)
            p = LuxCore.initialparameters(Xoshiro(1), f)
            @test eltype(noise_covariance(f, p)) === T
            F = first(Mycelium.local_free_energy(f, (x = x,), p, nt))
            @test all(scalar, values(F.parts))
            blocks = (x = Mycelium.PointBlock(repeat(x.value, 1, 3)),)
            bs = Mycelium.factor_beliefs(fill(f, 3), blocks, p, nt)
            @test eltype(last(first(bs.points))) === T
            Fs = first(Mycelium.belief_free_energies(fill(f, 3), bs, p, nt))
            @test all(E -> all(scalar, values(E.parts)), Fs)
            @test all(scalar, values(first(LenticulumCore.energy(f, nothing, nothing, x.value, p, nt)).parts))
        end
    end
end

@testset "P30 exact integer inputs" begin
    # review (Mistral, probe 3): an Int df took Float64 and widened a Float32 Wishart prior
    w = WishartPrecisionPrior(4, Matrix{Float32}(I, 2, 2))
    @test w.df === 4f0
    @test WishartPrecisionPrior(4, Matrix{Float64}(I, 2, 2)).df === 4.0
    f = GaussianPrior(:x, zeros(Float32, 2), Matrix{Float32}(I, 2, 2); learn_noise = true, noise_prior = w)
    p = LuxCore.initialparameters(Xoshiro(1), f)
    F = first(Mycelium.local_free_energy(f, (x = DiracBelief(Float32[0.1, 0.2]),), p, nt))
    @test F[:complexity] isa Float32
    f = GaussianPrior(:x, [0], ones(Int, 1, 1))
    F = first(Mycelium.local_free_energy(f, (x = DiracBelief([0]),), nt, nt))
    @test F[:complexity] ≈ log(2π) / 2
    @test F[:negentropy] === false
end

@testset "P30 Float32 derivatives" begin
    f = GaussianFactor(2 => 2; noise = 0.4f0)
    msgs = (x = DiracBelief(Float32[0.1, 0.2]), y = DiracBelief(Float32[0.3, -0.1]))
    fit(b) = first(Mycelium.local_free_energy(f, msgs, (A = Matrix{Float32}(I, 2, 2), b = fill(b, 2)), nt))[:fit]
    b = 0.15f0
    expected = -sum(msgs.y.value .- msgs.x.value .- b) / 0.4f0
    for d in (ForwardDiff.derivative(fit, b), only(Zygote.gradient(fit, b)))
        @test d isa Float32
        @test d ≈ expected rtol=1e-5
    end
end

@testset "P30 element type" begin
    for rectangular in (false, true), learn in (false, true), seed in 1:5
        g32, p32, st32, _, _, _, _ = tree(Float32, seed; rectangular, learn)
        g64, p64, st64, μ, Σ, F, rg = tree(Float64, seed; rectangular, learn)
        m32, s32, F32 = rungraph(g32, p32, st32)
        m64, _, F64 = rungraph(g64, p64, st64)
        for i in 1:4
            checkbelief(m32[Symbol(:x, i)], Float32; upstream = rectangular && i == 2)
            if i == 4
                @test m32[:x4].value ≈ m64[:x4].value rtol=1e-3 atol=1e-5
                continue
            end
            @test belief_mean(m32[Symbol(:x, i)]) ≈ belief_mean(m64[Symbol(:x, i)]) rtol=1e-3 atol=1e-5
            @test belief_cov(m32[Symbol(:x, i)]) ≈ belief_cov(m64[Symbol(:x, i)]) rtol=1e-3 atol=1e-5
            if i < 4
                @test belief_mean(m64[Symbol(:x, i)]) ≈ μ[rg(i)] atol=1e-9
                @test belief_cov(m64[Symbol(:x, i)]) ≈ Σ[rg(i), rg(i)] atol=1e-9
            end
        end
        if rectangular
            @test any(msg -> msg !== nothing && msg.belief isa ExtendedGaussian && size(msg.belief.D, 2) > 0, s32.to_variable)
        end
        for msg in s32.to_factor; checkbelief(msg === nothing ? nothing : msg.belief, Float32); end
        for msg in s32.to_variable; checkbelief(msg === nothing ? nothing : msg.belief, Float32); end
        checktype(F32 isa Float32, rectangular)
        @test F32 ≈ F64 rtol=1e-3 atol=1e-5
        @test F64 ≈ F atol=1e-9
        energies(s32, g32, p32, st32, Float32; upstream_variables = rectangular)
    end
    A = Float32[1 0; 0 0]
    @test Lenticulum._rank_deficient(A)
    entry = Lenticulum._RANK_MEMO[objectid(A)]
    @test entry[1].value === A && entry[2] == A
    @test Lenticulum._rank_deficient(A)
    @test Lenticulum._RANK_MEMO[objectid(A)][2] === entry[2]
    A[2, 2] = 1f0
    @test !Lenticulum._rank_deficient(A) # the copied-value check invalidates a stale rank
    g, ps, st, _, _, _, _ = tree(Float32, 2; mixed = true)
    m, _, F = rungraph(g, ps, st)
    checkbelief(m[:x1], Float64)
    @test F isa Float64
end
end
end # module P30ElementType
