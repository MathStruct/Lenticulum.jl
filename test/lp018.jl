using Lenticulum, LenticulumCore, Mycelium, LuxCore, Random, LinearAlgebra, Test
import SpecialFunctions

# Independent arbitrary-precision probability recurrence, starting at zero, with a fixed
# generously oversized support (no floating mass stopping condition).
function lp018_poisson_entropy(λ)
    setprecision(256) do
        l = BigFloat(λ)
        p = exp(-l)
        H = -p * log(p)
        for n in 1:ceil(Int, l + 50sqrt(l) + 300)
            p *= l / n
            H -= p * log(p)
        end
        Float64(H)
    end
end

@testset "LP-018 top-level regressions" begin
@testset "LP-018.1 bounded Poisson predictive entropy" begin
    errors = Float64[]
    for λ in sort!(unique!(vcat(10.0 .^ range(-3, 4; length = 31), 500.)))
        actual = Lenticulum._poisson_entropy(log(λ); maxterms = 100_000)
        reference = lp018_poisson_entropy(exp(log(λ)))
        push!(errors, abs(actual-reference))
        @test actual ≈ reference atol = 1e-12 rtol = 0
    end
    @info "LP-018 Poisson entropy" maximum_absolute_error = maximum(errors) entropy_at_500 = Lenticulum._poisson_entropy(log(500.); maxterms = 100_000)
    @test_throws ErrorException Lenticulum._poisson_entropy(log(500.); maxterms = 10)
    L = BivariatePoissonLikelihood()
    @test_throws ErrorException Lenticulum._conditional_entropy(L, log.([500., 1., 1.]); maxterms = 10)
    @test_throws ErrorException Lenticulum._conditional_entropy(L, log.([1., 2., 0.5]); maxcells = 1)
    # Independent convolution of three BigFloat Poisson pmfs.
    rates = [1., 2., 0.5]
    reference = setprecision(256) do
        pmfs = [[exp(-BigFloat(l)) * BigFloat(l)^n / factorial(big(n)) for n in 0:60] for l in rates]
        -sum(begin
            p = sum(pmfs[1][i-k+1] * pmfs[2][j-k+1] * pmfs[3][k+1] for k in 0:min(i,j))
            p * log(p)
        end for i in 0:60, j in 0:60)
    end
    @test Lenticulum._conditional_entropy(L, log.(rates); maxterms = 1000) ≈ Float64(reference) atol = 1e-12
end

@testset "LP-018.7 residual covariance validation" begin
    constructors = (Q -> LinearConstraintFactor((x = 2, y = 2), 2; noise = Q),
                    Q -> NonlinearFactor(identity, 2 => 2; noise = Q))
    for make in constructors
        for Q in ([1. 0.2; 10. 1.], [1.], ones(1, 1), ones(2, 3))
            @test_throws ArgumentError make(Q)
        end
        @test make([1. 0.2; 0.2 1.]).Q == [1. 0.2; 0.2 1.]
        @test_throws ArgumentError make([1. 2.; 2. 1.])
    end
end

@testset "LP-018.8 ordinal narrow intervals and both tails" begin
    for (lo, hi) in ((-1e-20, 1e-20), (8., 9.), (-9., -8.), (8., 8. + 1e-12), (-9., -9. + 1e-12)), predictor in (0., 0.7)
        L = OrdinalProbitLikelihood([lo, hi])
        ref = setprecision(256) do
            a, b = BigFloat(lo)-BigFloat(predictor), BigFloat(hi)-BigFloat(predictor)
            Φ(x) = SpecialFunctions.erfc(-x / sqrt(big(2))) / 2
            ϕ(x) = exp(-x*x/2) / sqrt(2big(π))
            p = Φ(b) - Φ(a)
            g = (ϕ(a)-ϕ(b)) / p
            h = (a*ϕ(a)-b*ϕ(b)) / p - g*g
            Float64.((log(p), g, h))
        end
        ell, g, H = loglikelihood_derivatives(L, [2.], [predictor])
        @test ell ≈ ref[1] atol = 2e-13
        @test only(g) ≈ ref[2] atol = 2e-12
        @test only(H) ≈ ref[3] atol = 2e-11
        @test loglikelihood(L, [2.], [predictor]) == ell
    end
    # A vanishing-width category observes z + N(0,1) = 0. Its tilted moments
    # are the ordinary Gaussian conditional, independent of the interval routines.
    f = LikelihoodFactor(OrdinalProbitLikelihood([-1e-20, 1e-20]), 1; method = ExpectationPropagation())
    prior = Gaussian(0.7, 1.)
    site = first(Mycelium.factor_message(f, :x, nothing, (y = DiracBelief([2.]),), prior, (;), (;)))
    post = combine(prior, site)
    @test belief_mean(post) ≈ [0.35] atol = 1e-14
    @test belief_cov(post) ≈ fill(0.5, 1, 1) atol = 1e-14
end

@testset "LP-018.9 finite particle weights in moment matching" begin
    for samples in ([[1.], [3.]], [1. 3.])
        b = moment_match(SampleBelief(samples, [1e308, 1e308]))
        @test belief_mean(b) ≈ [2.]
        @test belief_cov(b) ≈ ones(1, 1)
    end
end

@testset "LP-018.13 pruning bounds mass, not moments" begin
    m = MixtureBelief([Gaussian(0., 1.), Gaussian(1e12, 1.)], [1-1e-10, 1e-10])
    pruned = reduce_mixture(m; max_components = 2, prune = 1e-8)
    @test only(belief_mean(moment_match(m))) ≈ 100.
    @test only(belief_mean(moment_match(pruned))) == 0.
    @test occursin("total variation", string(@doc reduce_mixture))
end

end
