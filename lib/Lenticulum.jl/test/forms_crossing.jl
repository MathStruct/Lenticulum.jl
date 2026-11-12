# P1 (Feature Backlog): one edge carries a flat likelihood in two forms. With plain values a
# rank-deficient backward message is an `ExtendedGaussian`; under ForwardDiff, where the values
# are dual numbers, it stays a flagged `GaussianBelief(η, Λ, true)` (gaussian.md, "Improper
# messages as extended Gaussians"). Checked here:
#
# 1. the dual path's values are the value path's message, and its derivatives are central
#    differences of the value path: for GaussianFactor and LinearConstraintFactor with every
#    parameter dual, and for these and NonlinearFactor with only η dual (`b`, `c` or an input's
#    mean; A and the expansion point plain `Float64`);
# 2. a flagged input and its extended form split by one rule (`_split_moments`);
# 3. an inference through such edges is differentiable by ForwardDiff (derivative of a marginal
#    mean against the closed form), also where a dual flagged and a plain extended message meet;
# 4. the known limits: a dual flagged message entering a relation factor's flat path, and
#    NonlinearFactor's backward message at a dual expansion point.
#
# Run against the code before P1, the η-only cases of 1 and the testset of 3 fail:
# `_likelihood_message` tested only Λ's element type and built the extended form of a dual η
# (MethodError: no method matching svd!(::Matrix{ForwardDiff.Dual...})).
# Combine, distance and damping across the forms are Mycelium's `test/forms_crossing.jl`.

using ForwardDiff

module FormsCrossingLenticulum
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, ForwardDiff, LuxCore
const LC = LenticulumCore

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))
dual(x) = ForwardDiff.Dual{Nothing}(x, 1.0)
val(x) = ForwardDiff.value.(x)
der(x) = eltype(x) <: ForwardDiff.Dual ? ForwardDiff.partials.(x, 1) : zero(x)

same(a::ExtendedGaussian, b::ExtendedGaussian; atol = 1e-8) =
    isapprox(a.μ, b.μ; atol, rtol = 1e-8) && isapprox(a.Σ, b.Σ; atol, rtol = 1e-8) &&
    isapprox(a.D * a.D', b.D * b.D'; atol)

# A message as a function of one scalar t, the parameters moving along a fixed direction. `full`
# moves the coefficients too, along a direction that keeps their rank; otherwise only η moves.
function gaussian_case(rng, full)
    nin = rand(rng, 2:4); nout = rand(rng, 1:4); r = rand(rng, 1:min(nout, nin - 1))
    B, B2, C = randn(rng, nout, r), randn(rng, nout, r), randn(rng, r, nin)
    b0, b1, y0, y1 = randn(rng, nout), randn(rng, nout), randn(rng, nout), randn(rng, nout)
    Sy = spd(rng, nout)
    f = GaussianFactor(nin => nout; noise = spd(rng, nout))
    return t -> first(Mycelium.factor_message(f, :x, nothing, (y = LC.Gaussian(y0 .+ t .* y1, Sy),), nothing,
        (A = full ? (B .+ t .* B2) * C : B * C, b = b0 .+ t .* b1), NamedTuple()))
end

function constraint_case(rng, full)
    na = rand(rng, 2:3); nb = rand(rng, 1:3); m = rand(rng, 1:(na - 1))   # fewer equations than :a's dimension
    f = LinearConstraintFactor((a = na, b = nb), m; noise = spd(rng, m))
    Aa, Ab, Aa2, Ab2 = randn(rng, m, na), randn(rng, m, nb), randn(rng, m, na), randn(rng, m, nb)
    c0, c1, μb, μb1, Sb = randn(rng, m), randn(rng, m), randn(rng, nb), randn(rng, nb), spd(rng, nb)
    return t -> first(Mycelium.factor_message(f, :a, nothing, (b = LC.Gaussian(μb .+ t .* μb1, Sb),), nothing,
        (A = full ? (a = Aa .+ t .* Aa2, b = Ab .+ t .* Ab2) : (a = Aa, b = Ab), c = c0 .+ t .* c1), NamedTuple()))
end

# the Jacobian of the expansion point's linear model has rank nout < nin; the expansion point
# (the prior argument) is plain, so only η is dual
function nonlinear_case(rng, _)
    nin = rand(rng, 2:3); nout = rand(rng, 1:(nin - 1))
    W = randn(rng, nout, nin)
    f = NonlinearFactor(x -> tanh.(W * x), nin => nout; noise = spd(rng, nout))
    x̄, P = randn(rng, nin), spd(rng, nin)
    y0, y1, Sy = randn(rng, nout), randn(rng, nout), spd(rng, nout)
    return t -> first(Mycelium.factor_message(f, :x, nothing, (y = LC.Gaussian(y0 .+ t .* y1, Sy),),
        LC.Gaussian(x̄, P), NamedTuple(), NamedTuple()))
end
end # module

@testset "P1: the dual path's flagged message is the value path's extended one" begin
    FL = FormsCrossingLenticulum
    rng = Xoshiro(2612)
    # h = 1e-4, not smaller: with a saturated tanh the message has a variance of order 1e9 in
    # some direction, the extended -> canonical round trip of the value path then carries
    # rounding of order 1e-12, and at h = 1e-6 that was 7.6e-5 of the derivative (once in 100)
    h = 1e-4
    counts = Dict{Symbol,Int}()
    for (name, case) in ((:gaussian, FL.gaussian_case), (:constraint, FL.constraint_case), (:nonlinear, FL.nonlinear_case)),
        full in (true, false), trial in 1:20
        name === :nonlinear && full && continue
        msg = case(rng, full)
        v = msg(0.0)
        d = msg(FL.dual(0.0))
        @test v isa ExtendedGaussian
        @test d isa GaussianBelief && d.improper
        # values: the flagged message read back is the extended one, and the other way round
        @test FL.same(extended(GaussianBelief(FL.val(d.η), FL.val(d.Λ), true)), v)
        gv = GaussianBelief(v)
        @test gv.η ≈ FL.val(d.η) atol = 1e-7 rtol = 1e-7
        @test gv.Λ ≈ FL.val(d.Λ) atol = 1e-7 rtol = 1e-7
        # derivatives: central differences of the value path, read in canonical form
        gp, gm = GaussianBelief(msg(h)), GaussianBelief(msg(-h))
        @test FL.der(d.η) ≈ (gp.η .- gm.η) ./ 2h atol = 1e-5 rtol = 1e-5
        @test FL.der(d.Λ) ≈ (gp.Λ .- gm.Λ) ./ 2h atol = 1e-5 rtol = 1e-5
        full || @test eltype(d.Λ) <: AbstractFloat               # the η-only case is the one tested
        counts[name] = get(counts, name, 0) + 1
    end
    @test counts == Dict(:gaussian => 40, :constraint => 40, :nonlinear => 20)
end

@testset "P1: a flagged input and its extended form split by one rule" begin
    FL = FormsCrossingLenticulum
    rng = Xoshiro(2613)
    for trial in 1:100
        n = rand(rng, 1:4)
        kind = rand(rng, (:proper, :flagged, :flagged_full, :zero))
        V = Matrix(qr(randn(rng, n, n)).Q)[:, 1:(kind === :flagged ? rand(rng, 0:(n - 1)) : n)]
        Λ = kind === :zero ? zeros(n, n) : V * FL.spd(rng, size(V, 2)) * V'
        Λ = (Λ + Λ') / 2
        g = GaussianBelief(Λ * randn(rng, n), Λ, kind !== :proper)
        μ1, Σ1, N1 = Lenticulum._split_moments(g, n)
        μ2, Σ2, N2 = Lenticulum._split_moments(extended(g), n)
        @test μ1 ≈ μ2 atol = 1e-8 rtol = 1e-8
        @test Σ1 ≈ Σ2 atol = 1e-8 rtol = 1e-8
        @test (N1 === nothing) == (N2 === nothing)
        N1 === nothing || @test N1 * N1' ≈ N2 * N2' atol = 1e-8
        @test (N1 === nothing) == (kind in (:proper, :flagged_full))
    end
end

@testset "P1: ForwardDiff through an inference with flat backward messages" begin
    FL = FormsCrossingLenticulum
    rng = Xoshiro(2614)
    for nfac in (1, 2), trial in 1:10
        # x ~ N(μ₀, Σ₀);  uₖ = Aₖ x + bₖ + εₖ with rank(Aₖ) < dim x;  uₖ ~ N(μₖ, Σₖ) observed
        # softly. The derivative of x's marginal mean by b₁; every message gₖ -> x is flat along
        # null(Aₖ). With two factors, x combines a dual flagged message (g₁, whose b is
        # differentiated) with a plain extended one (g₂): the crossing of Mistral's review of P1
        # (vault LP-028, finding 1), where `combine` built the extended form of the dual one.
        n = rand(rng, 2:3)
        nus = [rand(rng, 1:(n - 1)) for _ in 1:nfac]
        μ0, Σ0 = randn(rng, n), FL.spd(rng, n)
        μs, Σs, Qs = [randn(rng, k) for k in nus], [FL.spd(rng, k) for k in nus], [FL.spd(rng, k) for k in nus]
        As, bs = [randn(rng, k, n) for k in nus], [randn(rng, k) for k in nus]
        b1 = randn(rng, nus[1])
        gb = GraphBuilder()
        variable!(gb, :x, n)
        factor!(gb, :px, GaussianPrior(:x, μ0, Σ0)); connect!(gb, :px, :x, :x)
        for k in 1:nfac
            u, pu, gk = Symbol(:u, k), Symbol(:pu, k), Symbol(:g, k)
            variable!(gb, u, nus[k])
            factor!(gb, pu, GaussianPrior(:x, μs[k], Σs[k])); connect!(gb, pu, :x, u)
            factor!(gb, gk, GaussianFactor(n => nus[k]; noise = Qs[k])); connect!(gb, gk, :x, :x); connect!(gb, gk, :y, u)
        end
        g = validate(build(gb))
        ps0 = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        for k in 2:nfac
            ps0 = merge(ps0, NamedTuple{(Symbol(:g, k),)}((merge(getfield(ps0, Symbol(:g, k)), (A = As[k], b = bs[k])),)))
        end
        function mean_x(t)
            ps = merge(ps0, (g1 = merge(ps0.g1, (A = As[1], b = bs[1] .+ t .* b1)),))
            marg, rep, _ = infer!(g, tree_schedule(g), ps, st)
            rep.converged || error("not converged")
            return LenticulumCore.belief_mean(marg[:x])
        end
        # the closed form: x given every u-observation, by the dense joint
        L = vcat(Matrix(1.0I, n, n), As...)
        Σ = L * Σ0 * L'
        Σ[(n + 1):end, (n + 1):end] += cat(Qs...; dims = (1, 2))
        K = Σ[1:n, (n + 1):end] / (Σ[(n + 1):end, (n + 1):end] + cat(Σs...; dims = (1, 2)))
        mu = vcat((As[k] * μ0 + bs[k] for k in 1:nfac)...)
        @test mean_x(0.0) ≈ μ0 + K * (vcat(μs...) - mu) atol = 1e-8 rtol = 1e-8
        @test ForwardDiff.derivative(mean_x, 0.0) ≈ -K * vcat(b1, zeros(sum(nus) - nus[1])) atol = 1e-7 rtol = 1e-7
    end
end

@testset "P1: known limit, a dual flagged message into a relation factor's flat path" begin
    # η-only dual: the flat directions do not move, so a derivative exists, but the flat path
    # computes the split and the residual metric by SVD and eigendecomposition, which have no
    # methods for dual numbers (constraint.jl, `factor_message` docstring)
    A = [1.0 1.0]
    f = GaussianFactor(2 => 1; noise = fill(0.5, 1, 1))
    msg, _ = Mycelium.factor_message(f, :x, nothing, (y = LenticulumCore.Gaussian([1.0], fill(1.0, 1, 1)),), nothing,
        (A = A, b = [FormsCrossingLenticulum.dual(0.3)]), NamedTuple())
    @test msg isa GaussianBelief && msg.improper
    f2 = GaussianFactor(2 => 2; noise = Matrix(0.3I, 2, 2))
    ok = try
        Mycelium.factor_message(f2, :y, nothing, (x = msg,), nothing, (A = [1.0 0.5; 0 1], b = zeros(2)), NamedTuple())
        true
    catch err
        err isa MethodError || rethrow()
        false
    end
    @test_broken ok
end

@testset "P1: known limit, NonlinearFactor's backward message at a dual expansion point" begin
    # `_jacobian` writes the finite-difference Jacobian into a `Float64` matrix, so a dual
    # expansion point throws `MethodError: no method matching Float64(::Dual)` before the message
    # is formed (pre-existing; found by Mistral's review of P1, vault LP-028, finding 3)
    W = [1.0 2.0]
    f = NonlinearFactor(x -> tanh.(W * x), 2 => 1; noise = fill(0.3, 1, 1))
    xd = FormsCrossingLenticulum.dual.([0.1, -0.2])
    ok = try
        Mycelium.factor_message(f, :x, nothing, (y = LenticulumCore.Gaussian([0.5], fill(1.0, 1, 1)),),
            LenticulumCore.Gaussian(xd, [1.0 0; 0 1]), NamedTuple(), NamedTuple())
        true
    catch err
        err isa MethodError || rethrow()
        false
    end
    @test_broken ok
end
