# P3: a GaussianFactor whose input x carries no information (no message, `TrivialBelief()`, a zero
# precision, an everywhere-flat extended Gaussian) sends the exact forward message, not the
# `TrivialBelief()` shortcut (gaussian.md, "No input on x").
#
# Oracle: x integrated out under a flat prior leaves y flat along range(A) and Gaussian off it.
# With U an orthonormal basis of null(Aᵀ) (from a full SVD), the message has precision
# Λ = U (UᵀQU)⁻¹ Uᵀ and information vector Λ b; it is `TrivialBelief()` when U is empty. Second,
# a graph in which x has no other factor, against the dense posterior of y under the prior
# x ~ N(0, τI) with τ large (the limit that defines the flat input).

module P3NoInput
using Lenticulum, LenticulumCore, LinearAlgebra, Random

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# A of size nout × nin with rank r
function random_A(rng, nout, nin, r)
    r == min(nout, nin) && return randn(rng, nout, nin)
    return randn(rng, nout, r) * randn(rng, r, nin)
end

# the forward message for a flat x, from the left null space of A
function closed_form(A, b, Q)
    F = svd(A; full = true)
    r = count(>(maximum(size(A)) * eps() * maximum(F.S; init = 0.0)), F.S)
    U = F.U[:, (r + 1):end]
    size(U, 2) == 0 && return nothing
    Λ = U * inv(Symmetric(U' * Q * U)) * U'
    return (Λ * b, Λ)
end

# a message as (η, Λ) with zero precision along its flat directions; `nothing` for TrivialBelief()
canon(::TrivialBelief) = nothing
canon(m::ExtendedGaussian) = (g = GaussianBelief(m); (g.η, g.Λ))
canon(m::GaussianBelief) = (m.η, m.Λ)
end # module

@testset "P3: a GaussianFactor with no input on x sends the exact forward message" begin
    PN = P3NoInput
    rng = Xoshiro(301)
    nmsg = 0; ntrivial = 0; ndeficient = 0
    for trial in 1:80
        nin, nout = rand(rng, 1:4), rand(rng, 1:4)
        r = rand(rng, 1:min(nin, nout))
        ndeficient += r < min(nin, nout)
        Q = PN.spd(rng, nout)
        f = GaussianFactor(nin => nout; noise = Q)
        ps = LuxCore.initialparameters(Xoshiro(trial), f)
        ps = merge(ps, (A = PN.random_A(rng, nout, nin, r), b = randn(rng, nout)))
        expected = PN.closed_form(ps.A, ps.b, Q)
        flat = ExtendedGaussian(zeros(nin), zeros(nin, nin), Matrix(1.0I, nin, nin))
        z = GaussianBelief(zeros(nin), zeros(nin, nin))
        # the mixtures are LP-048 F1 (Mistral): they threw in `moment_match`
        for inputs in (NamedTuple(), (x = nothing,), (x = TrivialBelief(),), (x = z,), (x = flat,),
                       (x = MixtureBelief([z, z], [0.3, 0.7]),), (x = MixtureBelief([flat]),))
            for msg in (first(Mycelium.factor_message(f, :y, nothing, inputs, nothing, ps, NamedTuple())),
                        first(MyceliumLearning._gradient_message(nothing, f, :y, nothing, inputs, nothing, ps, NamedTuple())))
                nmsg += 1
                got = PN.canon(msg)
                if expected === nothing
                    ntrivial += 1
                    @test got === nothing
                else
                    @test got !== nothing
                    got === nothing && continue
                    @test got[2] ≈ expected[2] atol = 1e-8 rtol = 1e-8
                    @test got[1] ≈ expected[1] atol = 1e-8 rtol = 1e-8
                end
            end
        end
        # backwards, no input on y: nothing reaches x
        for inputs in (NamedTuple(), (y = TrivialBelief(),))
            @test first(Mycelium.factor_message(f, :x, nothing, inputs, nothing, ps, NamedTuple())) isa TrivialBelief
        end
    end
    @test nmsg == 1120
    @test 0 < ntrivial < nmsg        # both kinds of A were drawn
    @test ndeficient > 0
end

@testset "P3: x with no other factor, y with a prior: infer! against the dense limit" begin
    PN = P3NoInput
    rng = Xoshiro(302)
    τ = big(1e30)
    for trial in 1:20
        nin, nout = rand(rng, 1:3), rand(rng, 2:4)
        nin = min(nin, nout - 1)                      # range(A) is a proper subspace
        Q, μy, Σy = PN.spd(rng, nout), randn(rng, nout), PN.spd(rng, nout)
        gb = GraphBuilder()
        variable!(gb, :x, nin); variable!(gb, :y, nout)
        factor!(gb, :g, GaussianFactor(nin => nout; noise = Q)); connect!(gb, :g, :x, :x); connect!(gb, :g, :y, :y)
        factor!(gb, :py, GaussianPrior(:x, μy, Σy)); connect!(gb, :py, :x, :y)
        g = validate(build(gb))
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        A, b = ps.g.A, ps.g.b
        # y's posterior: the prior times N(y; b, τAAᵀ + Q), the message of x ~ N(0, τI), in BigFloat
        # so that τ = 1e30 leaves an error far below the tolerance
        Ab, Qb, Σb = big.(A), big.(Q), big.(Σy)
        V = τ * Ab * Ab' + Qb
        Λ = inv(Σb) + inv(V)
        μ = Float64.(Λ \ (Σb \ big.(μy) + V \ big.(b)))
        marg, rep, _ = infer!(g, tree_schedule(g), ps, st)
        @test rep.converged
        @test belief_cov(marg.y) ≈ Float64.(inv(Λ)) rtol = 1e-8
        @test belief_mean(marg.y) ≈ μ rtol = 1e-8 atol = 1e-9
        # and it is not the prior alone, which the shortcut gave
        @test !isapprox(belief_cov(marg.y), Σy; rtol = 1e-3)
    end
end
