# P1 (Feature Backlog): where the two forms of a flat likelihood meet. An `improper`
# `GaussianBelief` (the canonical form with the flag) and an `ExtendedGaussian` (the host form)
# describe the same belief when there are no hard directions; `combine`, `belief_distance` and
# `damp` must give the same answer whichever form each side comes in. The oracle is the operation
# with both sides in one form: canonical for `combine` and `damp`, extended for the distance.
#
# Run against the code before P1, the damping testset fails: `can_damp` had no method for a mixed
# pair, so `damp` returned the new message undamped.

using LinearAlgebra, Random, ForwardDiff

module FormsCrossing
using LenticulumCore, LinearAlgebra, Random
const LC = LenticulumCore

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# a Gaussian belief on ℝⁿ of the given kind: proper; flagged and flat along n − k directions;
# or no information at all
function random_gb(rng, n, kind)
    kind === :proper && return LC.Gaussian(randn(rng, n), spd(rng, n))
    kind === :zero && return GaussianBelief(zeros(n), zeros(n, n), true)
    k = rand(rng, 1:(n - 1))
    V = Matrix(qr(randn(rng, n, n)).Q)[:, 1:k]
    Λ = V * spd(rng, k) * V'
    Λ = (Λ + Λ') / 2
    return GaussianBelief(Λ * randn(rng, n), Λ, true)
end

# an extended Gaussian with hard directions as well (no canonical form)
function random_hard(rng, n)
    B = Matrix(qr(randn(rng, n, n)).Q)
    kh = rand(rng, 1:n); kf = rand(rng, 0:(n - kh))
    V = B[:, (kh + kf + 1):n]
    Σ = V * spd(rng, size(V, 2)) * V'
    return ExtendedGaussian(randn(rng, n), (Σ + Σ') / 2, B[:, (kh + 1):(kh + kf)])
end

same(a::ExtendedGaussian, b::ExtendedGaussian; atol = 1e-8) =
    isapprox(a.μ, b.μ; atol, rtol = 1e-8) && isapprox(a.Σ, b.Σ; atol, rtol = 1e-8) &&
    isapprox(a.D * a.D', b.D * b.D'; atol)
end # module

@testset "P1: combine, distance and damping across the canonical and the extended form" begin
    FC = FormsCrossing
    LC = LenticulumCore
    rng = Xoshiro(2611)
    kinds = (:proper, :flagged, :zero)
    ncomb = 0; ndamp = 0; nflat = 0
    for trial in 1:150
        n = rand(rng, 2:4)
        a = FC.random_gb(rng, n, rand(rng, kinds))
        b = FC.random_gb(rng, n, rand(rng, kinds))
        ea, eb = extended(a), extended(b)
        nflat += size(ea.D, 2) + size(eb.D, 2)

        # combine: canonical sum, then extended, equals the product with either side extended
        ref = extended(combine(a, b))
        for c in (combine(ea, b), combine(a, eb), combine(ea, eb))
            @test c isa ExtendedGaussian
            @test FC.same(c, ref)
            ncomb += 1
        end
        # with a hard belief on the other side, the flagged and the extended form give one product
        h = FC.random_hard(rng, n)
        @test FC.same(combine(h, a), combine(h, ea))
        @test FC.same(combine(a, h), combine(ea, h))

        # the store's distance: a belief and its extended form are at distance zero, and a mixed
        # pair is compared as two extended Gaussians, from either side
        @test belief_distance(a, ea) <= 1e-9
        @test belief_distance(ea, a) <= 1e-9
        @test belief_distance(ea, b) == belief_distance(ea, eb)
        @test belief_distance(a, eb) == belief_distance(ea, eb)
        @test belief_distance(ea, b) == belief_distance(b, ea)

        # damping is the geometric mixture new^α · old^(1-α) in both forms: canonical convex
        # combination, then extended, equals the damping of any mixed pair
        α = rand(rng)
        @test can_damp(ea, b) && can_damp(a, eb)
        dref = extended(damp(a, b, α))
        for d in (damp(ea, b, α), damp(a, eb, α), damp(ea, eb, α))
            @test d isa ExtendedGaussian
            @test FC.same(d, dref; atol = 1e-7)
            ndamp += 1
        end
    end
    @test ncomb == 450 && ndamp == 450
    @test nflat > 100                              # flat directions were exercised

end

@testset "P1: a mixed pair with dual numbers goes through the canonical form" begin
    # ForwardDiff: a `GaussianBelief` whose η (or Λ) holds dual numbers has no extended form. Met
    # with a plain extended Gaussian without hard directions, `combine`, `belief_distance` and
    # `damp` take the extended side in canonical form; with hard directions there is no common
    # form, an `ArgumentError`, and `can_damp` is false (LP-028, finding 1: `combine` threw a
    # MethodError inside `infer!`; Lenticulum's `test/forms_crossing.jl` has the inference)
    FC = FormsCrossing
    LC = LenticulumCore
    rng = Xoshiro(2615)
    for trial in 1:60
        n = rand(rng, 2:4)
        a = FC.random_gb(rng, n, rand(rng, (:proper, :flagged)))
        dη = randn(rng, n)
        ad = GaussianBelief(ForwardDiff.Dual{Nothing}.(a.η, dη), a.Λ, a.improper)
        e = extended(FC.random_gb(rng, n, rand(rng, (:proper, :flagged, :zero))))
        ge = GaussianBelief(e)
        for c in (combine(e, ad), combine(ad, e))
            @test c isa GaussianBelief
            ref = combine(ge, ad)
            @test ForwardDiff.value.(c.η) ≈ ForwardDiff.value.(ref.η) && c.Λ ≈ ref.Λ
            @test ForwardDiff.partials.(c.η, 1) ≈ dη                        # combine adds η
            # its values are the plain product of the extended forms
            @test FC.same(extended(GaussianBelief(ForwardDiff.value.(c.η), c.Λ, c.improper)), combine(e, a))
        end
        @test belief_distance(e, ad) == belief_distance(ad, e) == belief_distance(ge, ad)
        α = rand(rng)
        @test can_damp(e, ad) && can_damp(ad, e)
        d1, d2 = damp(e, ad, α), damp(ad, e, α)
        @test d1 isa GaussianBelief && d2 isa GaussianBelief
        @test ForwardDiff.value.(d1.η) ≈ ForwardDiff.value.(damp(ge, ad, α).η) && d1.Λ ≈ damp(ge, ad, α).Λ
        @test FC.same(extended(GaussianBelief(ForwardDiff.value.(d2.η), d2.Λ, d2.improper)), damp(a, e, α); atol = 1e-7)
        # an extended Gaussian with hard directions has no common form with it
        h = FC.random_hard(rng, n)
        @test_throws ArgumentError combine(h, ad)
        @test_throws ArgumentError combine(ad, h)
        @test_throws ArgumentError belief_distance(h, ad)
        @test !can_damp(h, ad) && !can_damp(ad, h)
        @test damp(h, ad, α) === h
    end
end
