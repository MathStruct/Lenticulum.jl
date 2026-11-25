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
# 4. (P21) a dual flagged message entering a relation factor's flat path, and NonlinearFactor's
#    message at a dual expansion point, against central differences of the value path;
# 5. (P25) a flat input whose precision carries a derivative, and dual coefficients on the flat
#    path, at constant rank; a derivative that changes the rank is refused.
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
        @test eltype(d.Λ) === eltype(d.η)
        # In the offset-only case precision is constant, even though the
        # working type carries the same dual tag in both canonical arrays.
        full || @test all(iszero, FL.der(d.Λ))
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

# P21: a flagged input of dual numbers on a relation factor's flat path. With only η dual (a
# derivative by `b`, `c` or an observation upstream) the flat directions do not move: the split
# is taken on Λ's values and the duals pass through the mean. The oracle is the value path, where
# the same input is an `ExtendedGaussian`: the dual message's values are its message at t = 0, and
# its partials are central differences of it, read in canonical form.
module FormsCrossingP21
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, ForwardDiff, LuxCore
using ..FormsCrossingLenticulum: spd, dual, val, der
const LC = LenticulumCore

# (η, Λ) of a message; `nothing` for "no message"
canon(m::ExtendedGaussian) = (g = GaussianBelief(m); (g.η, g.Λ))
canon(m::GaussianBelief) = (m.η, m.Λ)
canon(::LC.TrivialBelief) = nothing


# A flat message on a channel of dimension n as a function of t: GaussianFactor's backward
# message through a rank-deficient A, its `b` moving with t (η-only under ForwardDiff).
function flat_source(rng, n)
    nout = rand(rng, 1:(n - 1))
    A = randn(rng, nout, n)
    b0, b1, y0, Sy = randn(rng, nout), randn(rng, nout), randn(rng, nout), spd(rng, nout)
    f = GaussianFactor(n => nout; noise = spd(rng, nout))
    return t -> first(Mycelium.factor_message(f, :x, nothing, (y = LC.Gaussian(y0, Sy),), nothing,
        (A = A, b = b0 .+ t .* b1), NamedTuple()))
end

# The relation factors that take the flat input; each returns the message as a function of the input.
function consumer(rng, kind, n)
    if kind === :forward                  # GaussianFactor, x flat -> y
        k = rand(rng, 1:4)
        f = GaussianFactor(n => k; noise = spd(rng, k))
        ps = (A = randn(rng, k, n), b = randn(rng, k))
        return src -> first(Mycelium.factor_message(f, :y, nothing, (x = src,), nothing, ps, NamedTuple()))
    elseif kind === :backward             # GaussianFactor, y flat -> x
        k = rand(rng, 1:4)
        f = GaussianFactor(k => n; noise = spd(rng, n))
        ps = (A = randn(rng, n, k), b = randn(rng, n))
        return src -> first(Mycelium.factor_message(f, :x, nothing, (y = src,), nothing, ps, NamedTuple()))
    else                                  # LinearConstraintFactor, a flat and c proper -> b
        nb, m = rand(rng, 1:3), rand(rng, 1:4)
        f = LinearConstraintFactor((a = n, b = nb, c = 2), m; noise = spd(rng, m))
        ps = (A = (a = randn(rng, m, n), b = randn(rng, m, nb), c = randn(rng, m, 2)), c = randn(rng, m))
        μc, Sc = randn(rng, 2), spd(rng, 2)
        return src -> first(Mycelium.factor_message(f, :b, nothing, (a = src, c = LC.Gaussian(μc, Sc)), nothing,
            ps, NamedTuple()))
    end
end
end # module

@testset "P21: a dual flagged message on a relation factor's flat path" begin
    P = FormsCrossingP21
    FL = FormsCrossingLenticulum
    rng = Xoshiro(2615)
    h = 1e-4
    counts = Dict{Tuple{Symbol,Symbol},Int}()
    for kind in (:forward, :backward, :constraint), trial in 1:30
        n = rand(rng, 2:4)
        src = P.flat_source(rng, n)
        msg = P.consumer(rng, kind, n)
        s0 = src(FL.dual(0.0))
        @test s0 isa GaussianBelief && s0.improper
        @test eltype(s0.Λ) === eltype(s0.η) && all(iszero, FL.der(s0.Λ))
        # Also feed an externally supplied belief whose constant precision has
        # primal storage, alongside the factor's dual precision with zero partials.
        if iseven(trial)
            s0 = GaussianBelief(s0.η, FL.val(s0.Λ), true)
        end
        d = msg(s0)
        v, vp, vm = P.canon(msg(src(0.0))), P.canon(msg(src(h))), P.canon(msg(src(-h)))
        @test (P.canon(d) === nothing) == (v === nothing)
        v === nothing && (counts[(kind, :trivial)] = get(counts, (kind, :trivial), 0) + 1; continue)
        η, Λ = P.canon(d)
        @test FL.val(η) ≈ v[1] atol = 1e-7 rtol = 1e-7
        @test FL.val(Λ) ≈ v[2] atol = 1e-7 rtol = 1e-7
        @test FL.der(η) ≈ (vp[1] .- vm[1]) ./ 2h atol = 1e-5 rtol = 1e-5
        @test FL.der(Λ) ≈ (vp[2] .- vm[2]) ./ 2h atol = 1e-5 rtol = 1e-5
        @test all(iszero, FL.der(Λ))          # the flat input's precision does not move
        @test norm(FL.der(η)) > 0
        key = (kind, d isa GaussianBelief && d.improper ? :flat : :proper)
        counts[key] = get(counts, key, 0) + 1
    end
    # the outcomes each consumer reaches here: the forward message is flat along A N (trivial
    # when A N spans y); the backward and the constraint message are flat or proper, the
    # constraint one trivial when the flat input's G spans the equations
    @test all(get(counts, ks, 0) > 0 for ks in ((:forward, :flat), (:forward, :trivial), (:backward, :flat),
        (:backward, :proper), (:constraint, :flat), (:constraint, :proper), (:constraint, :trivial)))
    @test sum(values(counts)) == 90
end

@testset "P21: the switching factor and two flat inputs" begin
    # SwitchingConstraintFactor with a flat x, to the continuous channel z (a mixture) and to the
    # outcome k; a LinearConstraintFactor with two flat inputs, each with up to three flat
    # directions. Values and central differences of the value path, as above, over every number
    # a message carries (log-weights relative to the first, then each component's η and Λ)
    P = FormsCrossingP21
    FL = FormsCrossingLenticulum
    flat(m::ExtendedGaussian) = (g = GaussianBelief(m); vcat(g.η, vec(g.Λ)))
    flat(m::GaussianBelief) = vcat(m.η, vec(m.Λ))
    flat(::TrivialBelief) = Float64[]
    flat(m::LenticulumCore.CategoricalBelief) = m.logp .- m.logp[1]
    flat(m::LenticulumCore.MixtureBelief) = vcat(m.logw .- m.logw[1], (flat(c) for c in m.components)...)
    h = 1e-4
    seen = Dict{Symbol,Int}()
    function check(msg)
        d, v = msg(FL.dual(0.0)), msg(0.0)
        @test typeof(d).name.wrapper == typeof(v).name.wrapper ||
              (d isa GaussianBelief && v isa ExtendedGaussian)
        fd, fv = flat(d), flat(v)
        @test FL.val(fd) ≈ fv atol = 1e-7 rtol = 1e-7
        @test FL.der(fd) ≈ (flat(msg(h)) .- flat(msg(-h))) ./ 2h atol = 1e-5 rtol = 1e-5
        k = Symbol(nameof(typeof(v)))
        seen[k] = get(seen, k, 0) + 1
    end
    rng = Xoshiro(2619)
    for trial in 1:30
        n = rand(rng, 2:4); src = P.flat_source(rng, n)
        nz, K, m = rand(rng, 1:3), rand(rng, 2:3), rand(rng, 1:4)
        f = SwitchingConstraintFactor((x = n, z = nz), K, m; noise = FL.spd(rng, m))
        ps = (A = (x = randn(rng, m, n), z = randn(rng, m, nz)), b = randn(rng, m, K), logw = randn(rng, K))
        kmsg = LenticulumCore.CategoricalBelief(rand(rng, K) .+ 0.1)
        zmsg = LenticulumCore.Gaussian(randn(rng, nz), FL.spd(rng, nz))
        check(t -> first(Mycelium.factor_message(f, :z, nothing, (x = src(t), k = kmsg), nothing, ps, NamedTuple())))
        check(t -> first(Mycelium.factor_message(f, :k, nothing, (x = src(t), z = zmsg), nothing, ps, NamedTuple())))
    end
    for trial in 1:30
        na, nb, nc, m = rand(rng, 2:4), rand(rng, 2:4), rand(rng, 1:3), rand(rng, 1:5)
        sa, sb = P.flat_source(rng, na), P.flat_source(rng, nb)
        f = LinearConstraintFactor((a = na, b = nb, c = nc), m; noise = FL.spd(rng, m))
        ps = (A = (a = randn(rng, m, na), b = randn(rng, m, nb), c = randn(rng, m, nc)), c = randn(rng, m))
        check(t -> first(Mycelium.factor_message(f, :c, nothing, (a = sa(t), b = sb(t)), nothing, ps, NamedTuple())))
    end
    # mixtures, outcomes, and flat, proper and trivial messages all reached
    @test all(get(seen, k, 0) > 0 for k in (:MixtureBelief, :CategoricalBelief, :ExtendedGaussian, :GaussianBelief, :TrivialBelief))
end

@testset "P21: ForwardDiff through an inference whose flat message enters a relation factor" begin
    FL = FormsCrossingLenticulum
    rng = Xoshiro(2616)
    h = 1e-4
    for trial in 1:15
        # u ~ N(μu, Σu) observed softly;  u = A₁x + b₁ + ε₁ with rank(A₁) < dim x;  x has no prior;
        # z = A₂x + b₂ + ε₂;  z ~ N(μz, Σz). The message g₁ -> x is flat, x passes it to g₂
        # (the flat path), and z's marginal mean is differentiated by b₁.
        n = rand(rng, 2:3); k = rand(rng, 1:(n - 1)); p = rand(rng, 1:3)
        gb = GraphBuilder()
        variable!(gb, :x, n); variable!(gb, :u, k); variable!(gb, :z, p)
        factor!(gb, :pu, GaussianPrior(:x, randn(rng, k), FL.spd(rng, k))); connect!(gb, :pu, :x, :u)
        factor!(gb, :pz, GaussianPrior(:x, randn(rng, p), FL.spd(rng, p))); connect!(gb, :pz, :x, :z)
        factor!(gb, :g1, GaussianFactor(n => k; noise = FL.spd(rng, k))); connect!(gb, :g1, :x, :x); connect!(gb, :g1, :y, :u)
        factor!(gb, :g2, GaussianFactor(n => p; noise = FL.spd(rng, p))); connect!(gb, :g2, :x, :x); connect!(gb, :g2, :y, :z)
        g = validate(build(gb))
        ps0 = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        A1, b1, d1, A2, b2 = randn(rng, k, n), randn(rng, k), randn(rng, k), randn(rng, p, n), randn(rng, p)
        function mean_z(t)
            ps = merge(ps0, (g1 = merge(ps0.g1, (A = A1, b = b1 .+ t .* d1)), g2 = merge(ps0.g2, (A = A2, b = b2))))
            marg, rep, _ = infer!(g, tree_schedule(g), ps, st)
            rep.converged || error("not converged")
            return LenticulumCore.belief_mean(marg[:z])
        end
        @test FL.val(mean_z(FL.dual(0.0))) ≈ mean_z(0.0) atol = 1e-8 rtol = 1e-8
        @test ForwardDiff.derivative(mean_z, 0.0) ≈ (mean_z(h) .- mean_z(-h)) ./ 2h atol = 1e-6 rtol = 1e-6
    end
end

# P25: a flat input whose precision carries a derivative (a dual coefficient upstream), and dual
# coefficients of the relation factor itself on its flat path: the flat directions move. The
# derivative is taken at constant rank (`LenticulumForwardDiffExt`, gaussian.md "Flat directions
# that move"). Oracle: central differences of the value path, as for P21. Before P25 every case
# here threw (`_improper_split_dual` refused a moving Λ; `svd` has no method for duals).
module FormsCrossingP25
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, ForwardDiff
using ..FormsCrossingLenticulum: spd
const LC = LenticulumCore

# GaussianFactor's backward message through A(t) of constant rank r < n: its range moves
# (A = (B + tB₂)C), its null space moves (A = B(C + tC₂)), or both
function moving_source(rng, n; mode = rand(rng, (:range, :null, :both)))
    nout = rand(rng, 1:(n - 1)); r = rand(rng, 1:nout)
    B, B2, C, C2 = randn(rng, nout, r), randn(rng, nout, r), randn(rng, r, n), randn(rng, r, n)
    b0, y0, Sy = randn(rng, nout), randn(rng, nout), spd(rng, nout)
    f = GaussianFactor(n => nout; noise = spd(rng, nout))
    A(t) = mode === :range ? (B .+ t .* B2) * C : mode === :null ? B * (C .+ t .* C2) :
           (B .+ t .* B2) * (C .+ t .* C2)
    return t -> first(Mycelium.factor_message(f, :x, nothing, (y = LC.Gaussian(y0, Sy),), nothing,
        (A = A(t), b = b0), NamedTuple()))
end

# every number a message carries (log-weights relative to the first, then η and Λ)
flat(m::ExtendedGaussian) = (g = GaussianBelief(m); vcat(g.η, vec(g.Λ)))
flat(m::GaussianBelief) = vcat(m.η, vec(m.Λ))
flat(::LC.TrivialBelief) = Float64[]
flat(m::LC.CategoricalBelief) = m.logp .- m.logp[1]
flat(m::LC.MixtureBelief) = vcat(m.logw .- m.logw[1], (flat(c) for c in m.components)...)
end # module

@testset "P25: flat inputs whose precision moves, and dual coefficients on the flat path" begin
    FL, P, Q = FormsCrossingLenticulum, FormsCrossingP21, FormsCrossingP25
    LC = LenticulumCore
    h = 1e-5
    seen = Dict{Tuple{Symbol,Symbol},Int}()
    function check(msg, key)
        d, v = msg(FL.dual(0.0)), msg(0.0)
        fd, fv = Q.flat(d), Q.flat(v)
        @test FL.val(fd) ≈ fv atol = 1e-7 rtol = 1e-7
        @test FL.der(fd) ≈ (Q.flat(msg(h)) .- Q.flat(msg(-h))) ./ 2h atol = 1e-5 rtol = 1e-5
        k = (key, Symbol(nameof(typeof(v))))
        seen[k] = get(seen, k, 0) + 1
    end
    rng = Xoshiro(2625)
    for kind in (:forward, :backward, :constraint), trial in 1:20
        n = rand(rng, 2:4); src = Q.moving_source(rng, n); msg = P.consumer(rng, kind, n)
        check(t -> msg(src(t)), kind)
    end
    for trial in 1:20                     # the switching factor, to z (a mixture) and to k
        n = rand(rng, 2:4); src = Q.moving_source(rng, n)
        nz, K, m = rand(rng, 1:3), rand(rng, 2:3), rand(rng, 1:4)
        f = SwitchingConstraintFactor((x = n, z = nz), K, m; noise = FL.spd(rng, m))
        ps = (A = (x = randn(rng, m, n), z = randn(rng, m, nz)), b = randn(rng, m, K), logw = randn(rng, K))
        kmsg = LC.CategoricalBelief(rand(rng, K) .+ 0.1); zmsg = LC.Gaussian(randn(rng, nz), FL.spd(rng, nz))
        check(t -> first(Mycelium.factor_message(f, :z, nothing, (x = src(t), k = kmsg), nothing, ps, NamedTuple())), :switch_z)
        check(t -> first(Mycelium.factor_message(f, :k, nothing, (x = src(t), z = zmsg), nothing, ps, NamedTuple())), :switch_k)
    end
    for trial in 1:20                     # two moving flat inputs
        na, nb, nc, m = rand(rng, 2:4), rand(rng, 2:4), rand(rng, 1:3), rand(rng, 1:5)
        sa, sb = Q.moving_source(rng, na), Q.moving_source(rng, nb)
        f = LinearConstraintFactor((a = na, b = nb, c = nc), m; noise = FL.spd(rng, m))
        ps = (A = (a = randn(rng, m, na), b = randn(rng, m, nb), c = randn(rng, m, nc)), c = randn(rng, m))
        check(t -> first(Mycelium.factor_message(f, :c, nothing, (a = sa(t), b = sb(t)), nothing, ps, NamedTuple())), :two)
    end
    for trial in 1:20                     # a fixed flat input, the factor's coefficients dual
        na, nb, nc, m = rand(rng, 2:4), rand(rng, 1:3), rand(rng, 1:3), rand(rng, 1:5)
        s0 = P.flat_source(rng, na)(0.0)
        f = LinearConstraintFactor((a = na, b = nb, c = nc), m; noise = FL.spd(rng, m))
        A0 = (a = randn(rng, m, na), b = randn(rng, m, nb), c = randn(rng, m, nc))
        A1 = map(x -> randn(rng, size(x)), A0)
        c, μb, Sb = randn(rng, m), randn(rng, nb), FL.spd(rng, nb)
        check(t -> first(Mycelium.factor_message(f, :c, nothing, (a = s0, b = LC.Gaussian(μb, Sb)), nothing,
            (A = map((x, y) -> x .+ t .* y, A0, A1), c = c), NamedTuple())), :coefficients)
    end
    # flat, proper and trivial messages, mixtures and outcomes all reached
    @test all(get(seen, k, 0) > 0 for k in ((:forward, :ExtendedGaussian), (:forward, :TrivialBelief),
        (:backward, :ExtendedGaussian), (:backward, :GaussianBelief), (:constraint, :ExtendedGaussian),
        (:constraint, :GaussianBelief), (:switch_z, :MixtureBelief), (:switch_k, :CategoricalBelief),
        (:two, :GaussianBelief), (:coefficients, :GaussianBelief)))
end

@testset "P25: a derivative that changes the rank is refused" begin
    LT = Lenticulum
    d(x, p) = ForwardDiff.Dual{Nothing}(x, p)
    rng = Xoshiro(2626)
    for trial in 1:20
        # a flagged input: a tangent with no NᵀΔN part is accepted, one with it is refused
        n = rand(rng, 2:4); r = rand(rng, 1:(n - 1))
        U = Matrix(qr(randn(rng, n, n)).Q); Ur, N = U[:, 1:r], U[:, (r + 1):end]
        Λ0 = Ur * Diagonal(rand(rng, r) .+ 0.5) * Ur'; η0 = Λ0 * randn(rng, n)
        Δ = Ur * randn(rng, r, n); Δ = Δ + Δ'
        _, _, Nd = LT._split_moments(GaussianBelief(d.(η0, 0.0), d.(Λ0, Δ), true), n)
        @test size(Nd, 2) == n - r
        @test maximum(abs, ForwardDiff.value.(Nd' * Nd) - I) < 1e-12
        @test maximum(abs, ForwardDiff.partials.(Nd' * Nd, 1)) < 1e-12       # orthonormal to first order
        @test maximum(abs, ForwardDiff.partials.(d.(Λ0, Δ) * Nd, 1)) < 1e-10  # Λ(ε)N(ε) = 0 to first order
        @test_throws ArgumentError LT._split_moments(GaussianBelief(d.(η0, 0.0), d.(Λ0, Δ + N[:, 1] * N[:, 1]'), true), n)
        # nested duals: refused by the fallback
        @test_throws ArgumentError LT._split_moments(GaussianBelief(η0, ForwardDiff.Dual{:outer}.(d.(Λ0, Δ), 1.0), true), n)
        # the flat directions' coefficients: two equal columns that a tangent separates
        m = rand(rng, 2:4); v, w = randn(rng, m), randn(rng, m)
        @test_throws ArgumentError LT._range_complement(d.(hcat(v, v), hcat(zeros(m), w)))
        # nested duals (found by the review, LP-042: was a MethodError of `svd`)
        @test_throws ArgumentError LT._range_complement(ForwardDiff.Dual{:outer}.(d.(hcat(v, w), hcat(w, v)), 1.0))
        W, _ = LT._range_complement(d.(hcat(v, v), hcat(w, w)))              # kept parallel: accepted
        @test maximum(abs, ForwardDiff.partials.(W' * d.(hcat(v, v), hcat(w, w)), 1)) < 1e-10
        # the log pseudo-determinant (the switching factor's normaliser) against central differences
        Bv, Bw = randn(rng, m + 1, m), randn(rng, m + 1, m)
        lp(t) = LT._range_complement(Bv .+ t .* Bw)[2]
        @test ForwardDiff.partials(LT._range_complement(d.(Bv, Bw))[2], 1) ≈ (lp(1e-6) - lp(-1e-6)) / 2e-6 rtol = 1e-6
    end
end

@testset "P21: NonlinearFactor's message at a dual expansion point" begin
    # The linear model at x̄ + t x̄₁: the finite-difference Jacobian of duals (or a given Jacobian
    # function) against central differences of the value path, both methods
    FL = FormsCrossingLenticulum
    P = FormsCrossingP21
    rng = Xoshiro(2618)
    h = 1e-4
    for method in (Linearization(), Unscented()), given in (false, true), trial in 1:10
        nin = rand(rng, 2:3); nout = rand(rng, 1:3)
        W = randn(rng, nout, nin)
        jac = given ? (x -> (1 .- tanh.(W * x) .^ 2) .* W) : nothing
        f = NonlinearFactor(x -> tanh.(W * x), nin => nout; noise = FL.spd(rng, nout), method, jacobian = jac)
        x̄, x̄1, Px = randn(rng, nin), randn(rng, nin), FL.spd(rng, nin)
        y0, Sy = randn(rng, nout), FL.spd(rng, nout)
        msg(t) = first(Mycelium.factor_message(f, :x, nothing, (y = LenticulumCore.Gaussian(y0, Sy),), LenticulumCore.Gaussian(x̄ .+ t .* x̄1, Px),
            NamedTuple(), NamedTuple()))
        d = msg(FL.dual(0.0))
        v, vp, vm = P.canon(msg(0.0)), P.canon(msg(h)), P.canon(msg(-h))
        η, Λ = P.canon(d)
        @test FL.val(η) ≈ v[1] atol = 1e-7 rtol = 1e-7
        @test FL.val(Λ) ≈ v[2] atol = 1e-7 rtol = 1e-7
        @test FL.der(η) ≈ (vp[1] .- vm[1]) ./ 2h atol = 1e-5 rtol = 1e-5
        @test FL.der(Λ) ≈ (vp[2] .- vm[2]) ./ 2h atol = 1e-5 rtol = 1e-5
    end
end
