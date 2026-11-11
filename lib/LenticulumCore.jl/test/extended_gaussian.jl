# Extended Gaussians against the limit of proper Gaussians (extended_gaussian.md).
#
# Oracle: an extended Gaussian N(μ, Σ) + D is the limit of the proper Gaussian with covariance
# Σ + ε I + τ D Dᵀ as ε → 0 (hard directions) and τ → ∞ (flat directions). The reference is that
# Gaussian in 512-bit BigFloat with ε = 1e-20, τ = 1e20: the limit then errs by about 1e-20, and
# the two inversions (condition about 1e40 each) keep some 70 digits. With τ/ε = 1e60 at the default
# 256 bits the reference itself was wrong (a covariance with an eigenvalue of -1e13). The product is
# the product of densities (precisions add).

module ExtendedGaussianOracle
using LenticulumCore, LinearAlgebra, Random

# a random extended Gaussian of dimension n with k flat and h hard directions
function random_extended(rng, n, k, h)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    D = Q[:, 1:k]
    G = Q[:, (k + h + 1):n]                          # the Gaussian directions
    M = randn(rng, size(G, 2), size(G, 2))
    Σ = G * (M * M' + 0.3I) * G'
    return ExtendedGaussian(randn(rng, n), Σ, D)
end

const ε, τ = setprecision(() -> (big"1e-20", big"1e20"), BigFloat, 512)
# called inside the 512-bit block. The projector onto the flat directions is formed in BigFloat:
# formed in Float64, its rounding (1e-16) times τ puts variance of order 1e4 in directions that
# are not flat.
approx_cov(b) = (D = BigFloat.(b.D); BigFloat.(b.Σ) + ε * I + τ * D * D')

# the reference product (moments), from the two proper approximations
reference_product(a, b) = setprecision(() -> _reference_product(a, b), BigFloat, 512)
function _reference_product(a, b)
    Λ = inv(approx_cov(a)) + inv(approx_cov(b))
    Σ = inv(Λ)
    μ = Σ * (inv(approx_cov(a)) * BigFloat.(a.μ) + inv(approx_cov(b)) * BigFloat.(b.μ))
    return μ, Σ, Λ
end
projector(D) = D * D'
end # module

using LenticulumCore, LinearAlgebra, Random, Test
# `pushforward` is qualified: DifferentiationInterface, in this test environment, exports one too

@testset "extended Gaussians: the product is the limit of proper Gaussians" begin
    EO = ExtendedGaussianOracle
    rng = Xoshiro(2026)
    ncases = 0
    for trial in 1:200
        n = rand(rng, 2:5)
        ka, kb = rand(rng, 0:(n - 1)), rand(rng, 0:(n - 1))
        ha = rand(rng, 0:(n - ka)); hb = rand(rng, 0:(n - kb))
        ha + hb <= n - 1 || continue                 # two generic constraint sets that do not overdetermine
        a, b = EO.random_extended(rng, n, ka, ha), EO.random_extended(rng, n, kb, hb)
        p = LenticulumCore._conjoin(a, b)
        μr, Σr, Λr = EO.reference_product(a, b)
        P = I - EO.projector(p.D)
        # off the flat directions: the same mean and covariance
        @test P * μr ≈ p.μ atol = 1e-9 * (1 + norm(p.μ))
        @test P * Σr * P ≈ p.Σ atol = 1e-9 * (1 + norm(p.Σ))
        # along them the reference variance grows with τ (Dᵀ Λ D is no test: the hard directions have
        # precision 1/ε, so a rounding of 1e-16 in D leaves 1e-11 there)
        size(p.D, 2) > 0 && @test minimum(eigvals(Symmetric(Float64.(p.D' * Σr * p.D)))) > 1e15
        ncases += 1
    end
    @test ncases > 100
end

@testset "extended Gaussians: algebraic laws" begin
    EO = ExtendedGaussianOracle
    rng = Xoshiro(7)
    same(x, y) = isapprox(x.μ, y.μ; atol = 1e-9) && isapprox(x.Σ, y.Σ; atol = 1e-9) &&
                 isapprox(EO.projector(x.D), EO.projector(y.D); atol = 1e-9)
    for _ in 1:50
        n = 4
        a, b, c = EO.random_extended(rng, n, 1, 0), EO.random_extended(rng, n, 1, 1), EO.random_extended(rng, n, 0, 1)
        @test same(LenticulumCore._conjoin(a, b), LenticulumCore._conjoin(b, a))
        @test same(LenticulumCore._conjoin(LenticulumCore._conjoin(a, b), c),
                   LenticulumCore._conjoin(a, LenticulumCore._conjoin(b, c)))
        @test same(LenticulumCore._conjoin(a, extended(TrivialBelief(), n)), a)
        # pushforward composes, marginalise is a pushforward
        M1, M2 = randn(rng, 3, n), randn(rng, 2, 3)
        @test same(LenticulumCore.pushforward(LenticulumCore.pushforward(b, M1), M2), LenticulumCore.pushforward(b, M2 * M1))
        @test same(marginalise(b, [1, 3]), LenticulumCore.pushforward(b, Matrix(I, n, n)[[1, 3], :]))
    end
end

@testset "extended Gaussians: pushforward of flat and hard directions" begin
    # x₃ is flat and x₂ is fixed at 2: then x₁ + x₃ is flat and x₂ stays fixed
    b = ExtendedGaussian([1.0, 2.0, 3.0], Matrix(Diagonal([1.0, 0.0, 0.0])), reshape([0.0, 0.0, 1.0], 3, 1))
    @test size(hard_directions(b), 2) == 1 && abs(hard_directions(b)[2]) ≈ 1
    p = LenticulumCore.pushforward(b, [1.0 0 1; 0 1 0])
    @test LenticulumCore.isuninformative(marginalise(p, [1]))
    @test p.μ ≈ [0.0, 2.0] && all(iszero, p.Σ)
    # added noise makes the hard direction Gaussian
    q = LenticulumCore.pushforward(b, [0.0 1 0], 0, fill(0.5, 1, 1))
    @test q.μ ≈ [2.0] && q.Σ ≈ fill(0.5, 1, 1) && size(q.D, 2) == 0
end

@testset "extended Gaussians: conversions and refusals" begin
    rng = Xoshiro(3)
    # a proper Gaussian round trip
    M = randn(rng, 3, 3); Σ = M * M' + I; μ = randn(rng, 3)
    g = Gaussian(μ, Σ)
    e = extended(g)
    @test e.μ ≈ μ && e.Σ ≈ Σ && size(e.D, 2) == 0
    g2 = GaussianBelief(e)
    @test g2.η ≈ g.η && g2.Λ ≈ g.Λ && !g2.improper
    # an improper Gaussian: flat along null(Λ), and declared improper on the way back
    gi = GaussianBelief([2.0, 0.0], [2.0 0.0; 0.0 0.0], true)
    ei = extended(gi)
    @test ei.μ ≈ [1.0, 0.0] && ei.Σ ≈ [0.5 0.0; 0.0 0.0] && abs(ei.D[2]) ≈ 1
    gb = GaussianBelief(ei)
    @test gb.improper && gb.Λ ≈ gi.Λ && gb.η ≈ gi.η
    # a clamp is hard everywhere and has no canonical form
    ed = extended(DiracBelief([1.0, 2.0]))
    @test size(hard_directions(ed), 2) == 2
    @test_throws ArgumentError GaussianBelief(ed)
    @test LenticulumCore.isuninformative(extended(TrivialBelief(), 3))
    # refusals: exp(ηᵀx) along a flat direction; Σ not positive semidefinite; contradictory constraints
    @test_throws ArgumentError extended(GaussianBelief([1.0, 1.0], [1.0 0.0; 0.0 0.0], true))
    @test_throws ArgumentError ExtendedGaussian([0.0, 0.0], [1.0 0.0; 0.0 -1.0])
    a = ExtendedGaussian([0.0, 1.0], [1.0 0.0; 0.0 0.0])
    b = ExtendedGaussian([0.0, 2.0], [1.0 0.0; 0.0 0.0])
    @test_throws ArgumentError LenticulumCore._conjoin(a, b)
    # two consistent constraints on different coordinates give a point
    c = ExtendedGaussian([5.0, 0.0], [0.0 0.0; 0.0 1.0])
    p = LenticulumCore._conjoin(a, c)
    @test p.μ ≈ [5.0, 1.0] && all(x -> abs(x) < 1e-12, p.Σ)
end
