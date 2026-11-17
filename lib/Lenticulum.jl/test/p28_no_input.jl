# P28: a NonlinearFactor whose input x carries no information sends the forward message of its
# linearisation at `x0`, with x flat, as GaussianFactor does since P3 (nonlinear.md, "No input on x").
#
# Oracles. (1) The P3 closed form for the linear model at x0, with the Jacobian written out by hand:
# for f(x) = tanh.(W x + c), A = Diagonal(1 - tanh²(W x0 + c)) W and b = f(x0) - A x0; with
# U an orthonormal basis of null(Aᵀ) the message has precision U (UᵀQU)⁻¹ Uᵀ and information Λ b,
# and is `TrivialBelief()` when U is empty. (2) An affine f, for which the message is exact: in a
# graph where x has no other factor, the marginal of y against the dense posterior under
# x ~ N(0, τI), τ large, in BigFloat.

module P28NoInput
using Lenticulum, LenticulumCore, LinearAlgebra, Random

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

function random_W(rng, nout, nin, r)
    r == min(nout, nin) && return randn(rng, nout, nin)
    return randn(rng, nout, r) * randn(rng, r, nin)
end

function closed_form(A, b, Q)
    F = svd(A; full = true)
    r = count(>(maximum(size(A)) * eps() * maximum(F.S; init = 0.0)), F.S)
    U = F.U[:, (r + 1):end]
    size(U, 2) == 0 && return nothing
    Λ = U * inv(Symmetric(U' * Q * U)) * U'
    return (Λ * b, Λ)
end

canon(::TrivialBelief) = nothing
canon(m::ExtendedGaussian) = (g = GaussianBelief(m); (g.η, g.Λ))
canon(m::GaussianBelief) = (m.η, m.Λ)

function empty_inputs(nin)
    flat = ExtendedGaussian(zeros(nin), zeros(nin, nin), Matrix(1.0I, nin, nin))
    z = GaussianBelief(zeros(nin), zeros(nin, nin))
    return (NamedTuple(), (x = nothing,), (x = TrivialBelief(),), (x = z,), (x = flat,),
            (x = MixtureBelief([z, z], [0.3, 0.7]),), (x = MixtureBelief([flat]),))
end
end # module

@testset "P28: a NonlinearFactor with no input on x linearises at x0" begin
    PN = P28NoInput
    rng = Xoshiro(2801)
    nmsg = 0; ntrivial = 0; ndeficient = 0
    nfd = 0; nfd_agree = 0                 # central differences with a rank-deficient W
    for trial in 1:60
        nin, nout = rand(rng, 1:4), rand(rng, 1:4)
        r = rand(rng, 1:min(nin, nout))
        deficient = r < min(nin, nout)
        ndeficient += deficient
        Q, W, c, x0 = PN.spd(rng, nout), PN.random_W(rng, nout, nin, r), randn(rng, nout), randn(rng, nin)
        fx(x) = tanh.(W * x .+ c)
        A = Diagonal(1 .- tanh.(W * x0 .+ c) .^ 2) * W
        b = fx(x0) .- A * x0
        expected = PN.closed_form(A, b, Q)
        # an exact Jacobian (tolerance 1e-8) and central differences (1e-5); both methods
        for (jac, tol) in ((x -> Diagonal(1 .- tanh.(W * x .+ c) .^ 2) * W, 1e-8), (nothing, 1e-5)),
            method in (Linearization(), Unscented())
            f = NonlinearFactor(fx, nin => nout; noise = Q, x0, jacobian = jac, method)
            for inputs in PN.empty_inputs(nin)
                nmsg += 1
                got = PN.canon(first(Mycelium.factor_message(f, :y, nothing, inputs, nothing, NamedTuple(), NamedTuple())))
                if jac === nothing && deficient
                    # The difference error (~1e-10) gave the Jacobian full numerical rank, so the
                    # rank test lost directions of null(Aᵀ); since P29 the singular values below
                    # the difference error are cut (test/p29_rank_tolerance.jl). Never more than exact:
                    nfd += 1
                    Λe = expected === nothing ? zeros(nout, nout) : expected[2]
                    Λg = got === nothing ? zeros(nout, nout) : got[2]
                    nfd_agree += isapprox(Λg, Λe; atol = tol, rtol = tol)
                    @test minimum(eigvals(Symmetric(Λe - Λg))) >= -1e-5 * (1 + opnorm(Λe))
                elseif expected === nothing
                    ntrivial += 1
                    @test got === nothing
                else
                    @test got !== nothing
                    got === nothing && continue
                    @test got[2] ≈ expected[2] atol = tol rtol = tol
                    @test got[1] ≈ expected[1] atol = tol rtol = tol
                end
            end
        end
        # without x0 there is no point to linearise at: uninformative, as before P28
        f = NonlinearFactor(fx, nin => nout; noise = Q)
        for inputs in PN.empty_inputs(nin)
            @test first(Mycelium.factor_message(f, :y, nothing, inputs, nothing, NamedTuple(), NamedTuple())) isa TrivialBelief
        end
    end
    @test nmsg == 60 * 4 * 7
    @test 0 < ntrivial < nmsg
    @test ndeficient > 0
    @test nfd > 0
    @test nfd_agree == nfd                 # the rank tolerance of difference Jacobians (P29)
end

@testset "P28: the polarity x Latent(), y the target, only with x0" begin
    f = NonlinearFactor(x -> [sin(x[1]), x[1]^2], 1 => 2; noise = 0.2)
    g = NonlinearFactor(x -> [sin(x[1]), x[1]^2], 1 => 2; noise = 0.2, x0 = [0.4])
    p = Polarity(x = Latent(), y = Unobserved())
    @test length(supported_polarities(f)) == 2 && !supports_polarity(f, p)
    @test length(supported_polarities(g)) == 3 && supports_polarity(g, p) && p in supported_polarities(g)
    @test !supports_polarity(g, Polarity(x = Unobserved(), y = Latent()))
end

@testset "P28: an affine NonlinearFactor, x with no other factor: infer! against the dense limit" begin
    PN = P28NoInput
    rng = Xoshiro(2802)
    τ = big(1e30)
    for trial in 1:20
        nin, nout = rand(rng, 1:3), rand(rng, 2:4)
        nin = min(nin, nout - 1)                      # range(A) is a proper subspace
        Q, μy, Σy = PN.spd(rng, nout), randn(rng, nout), PN.spd(rng, nout)
        A, b, x0 = randn(rng, nout, nin), randn(rng, nout), randn(rng, nin)
        nl = NonlinearFactor(x -> A * x .+ b, nin => nout; noise = Q, x0, jacobian = x -> A)
        # the same message as GaussianFactor's (P3) for (A, b)
        gf = GaussianFactor(nin => nout; noise = Q)
        gps = merge(LuxCore.initialparameters(Xoshiro(trial), gf), (A = A, b = b))
        mg = PN.canon(first(Mycelium.factor_message(gf, :y, nothing, NamedTuple(), nothing, gps, NamedTuple())))
        mn = PN.canon(first(Mycelium.factor_message(nl, :y, nothing, NamedTuple(), nothing, NamedTuple(), NamedTuple())))
        @test mn[2] ≈ mg[2] rtol = 1e-10 atol = 1e-10
        @test mn[1] ≈ mg[1] rtol = 1e-10 atol = 1e-10
        gb = GraphBuilder()
        variable!(gb, :x, nin); variable!(gb, :y, nout)
        factor!(gb, :g, nl); connect!(gb, :g, :x, :x); connect!(gb, :g, :y, :y)
        factor!(gb, :py, GaussianPrior(:x, μy, Σy)); connect!(gb, :py, :x, :y)
        g = validate(build(gb))
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        Ab, Qb, Σb = big.(A), big.(Q), big.(Σy)
        V = τ * Ab * Ab' + Qb
        Λ = inv(Σb) + inv(V)
        μ = Float64.(Λ \ (Σb \ big.(μy) + V \ big.(b)))
        marg, rep, _ = infer!(g, tree_schedule(g), ps, st)
        @test rep.converged
        @test belief_cov(marg.y) ≈ Float64.(inv(Λ)) rtol = 1e-8
        @test belief_mean(marg.y) ≈ μ rtol = 1e-8 atol = 1e-9
        @test !isapprox(belief_cov(marg.y), Σy; rtol = 1e-3)
    end
end
