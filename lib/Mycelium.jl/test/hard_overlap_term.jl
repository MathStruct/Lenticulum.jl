# I26: the variable term where exact messages meet (`free_energy.md`, "Exact messages that meet at
# a variable"). Sources that send fixed messages into one variable; `variable_corrections` is
# (1 − d)(−H) plus the overlap −½[(p − 1) log det G − Σᵢ log det G₋ᵢ] of the hard messages. The
# oracle that this is −log Z on trees is in Lenticulum (`test/hard_overlap.jl`,
# `test/hard_free_energy.jl`); here the term's own cases.

using LinearAlgebra, Random

struct I26Source <: AbstractLenticulumFactor end
LenticulumCore.islearnable(::I26Source) = true            # its message is a parameter
LenticulumCore.channels(::I26Source) = (LenticulumCore.Channel(:x, nothing),)
LenticulumCore.supported_polarities(::I26Source) = (Polarity(x = Unobserved()),)
LenticulumCore.supports_polarity(::I26Source, p::Polarity) = p[:x] isa Unobserved
Mycelium.factor_message(::I26Source, ::Symbol, _, _, _, ps, st) = (ps.msg, st)

# exact along the columns of B, flat along the rest: an observation of Bᵀz
function i26_observation(B, μ)
    D = nullspace(Matrix(B'))
    return LenticulumCore.ExtendedGaussian(μ, zeros(length(μ), length(μ)), D)
end

function i26_terms(msgs)
    b = GraphBuilder()
    variable!(b, :z)
    for (i, _) in enumerate(msgs)
        factor!(b, Symbol(:s, i), I26Source()); connect!(b, Symbol(:s, i), :x, :z)
    end
    g = build(b); s = MessageStore(g)
    ps = NamedTuple{Tuple(Symbol(:s, i) for i in eachindex(msgs))}(Tuple((msg = m,) for m in msgs))
    st = NamedTuple{keys(ps)}(Tuple((;) for _ in msgs))
    propagate!(s, g, tree_schedule(g), ps, st)
    m = marginal(s, g, 1)
    H = m isa LenticulumCore.DiracBelief ? 0.0 : LenticulumCore.support_entropy(m)
    return variable_corrections(s, g).parts.z, (1 - length(msgs)) * -H
end

@testset "I26: the overlap of exact messages at a variable" begin
    rng = Random.Xoshiro(26)
    prior = Gaussian(zeros(3), Matrix(1.0I, 3, 3))
    for _ in 1:20
        k, h = normalize(randn(rng, 3)), normalize(randn(rng, 3))
        sinθ = sqrt(1 - dot(k, h)^2)
        # two oblique observations and a prior: −log sin θ on top of the entropy terms
        V, Hterm = i26_terms((prior, i26_observation(k, randn(rng, 3)), i26_observation(h, randn(rng, 3))))
        @test V ≈ Hterm - log(sinθ) rtol = 1e-10
        # the order of the messages does not matter, nor does the length of the projection
        V2, _ = i26_terms((i26_observation(3h, randn(rng, 3)), prior, i26_observation(-k, randn(rng, 3))))
        @test V2 - i26_terms((i26_observation(3h, zeros(3)), prior, i26_observation(-k, zeros(3))))[2] ≈ -log(sinθ) rtol = 1e-10
        # one exact message: no overlap
        V1, H1 = i26_terms((prior, i26_observation(k, randn(rng, 3))))
        @test V1 ≈ H1 rtol = 1e-12
    end
    # orthogonal exact messages: no overlap
    V, Hterm = i26_terms((prior, i26_observation([1.0, 0, 0], randn(rng, 3)), i26_observation([0, 1.0, 0], randn(rng, 3))))
    @test V ≈ Hterm atol = 1e-12
    # p = 3 of mixed ranks in ℝ⁴: the formula, with the Gram determinants by hand
    B1, b2, b3 = Matrix(qr(randn(rng, 4, 2)).Q), normalize(randn(rng, 4)), normalize(randn(rng, 4))
    gram(M) = det(M' * M)
    want = -(2 * log(gram([B1 b2 b3])) - log(gram([b2 b3])) - log(gram([B1 b3])) - log(gram([B1 b2]))) / 2
    V, Hterm = i26_terms((Gaussian(zeros(4), Matrix(1.0I, 4, 4)), i26_observation(B1, randn(rng, 4)),
                          i26_observation(b2, randn(rng, 4)), i26_observation(b3, randn(rng, 4))))
    @test V ≈ Hterm + want rtol = 1e-10
    # a point is a clamp, not counted (else its basis I would overlap every other exact message)
    k, h = normalize(randn(rng, 3)), normalize(randn(rng, 3)); x = randn(rng, 3)
    V, _ = i26_terms((DiracBelief(x), i26_observation(k, x)))
    @test iszero(V)
    # two exact messages along a common direction: the free energy is infinite
    @test_throws ArgumentError i26_terms((prior, i26_observation(k, zeros(3)), i26_observation(-2k, zeros(3))))
    @test_throws ArgumentError i26_terms((prior, i26_observation([k h], zeros(3)), i26_observation(k + h, zeros(3))))
end
