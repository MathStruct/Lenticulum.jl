using Test
isdefined(@__MODULE__, :HierarchyInference) ||
    include(joinpath(@__DIR__, "..", "examples", "hierarchy_inference.jl"))

@testset "Structured inference in the hierarchy (examples/hierarchy_inference.jl)" begin
    using LinearAlgebra, Random
    H = HierarchyInference
    # the exact E-step's likelihood against brute-force enumeration
    h = H.truth(8, 3, 4); X = H.sample(h, 300, Xoshiro(1)); Kc = H.configs(4)
    function lp(i, j, c)
        r = X[i, :] - h.W[j] * vcat(1.0, Kc[:, c])
        log(h.w[j]) + sum(Kc[l, c] == 1 ? log(h.p[j][l]) : log(1 - h.p[j][l]) for l in 1:4) -
            0.5 * dot(r, h.Σ[j] \ r) - 0.5 * logdet(2π * h.Σ[j])
    end
    bf = sum(log(sum(exp(lp(i, j, c)) for j in 1:3, c in 1:16)) for i in 1:300) / 300
    exact = H.exact_loglik(h, X)
    @test exact ≈ bf rtol = 1e-12
    # the bounds: mean field ≤ blocks of 2 ≤ exact
    mf = H.estep(h, X, G -> [[l] for l in 1:4])[2]; b2 = H.estep(h, X, G -> H.fixed_blocks(4, 2))[2]
    @test mf <= b2 <= exact + 1e-10
    # a cheap version of the run: overlap blocks have a fixed point much nearer the truth than mean field
    rows = H.run(; seeds = (1,), Ls = (6,), random_seeds = (), ntrain = 2000, ntest = 1000, ngen = 200, iters = 100)
    r(m) = only(x for x in rows if x.method == m)
    @test r("exact").gap < 0.1
    @test r("overlap blocks of ≤ 3").gap < r("mean field").gap - 0.2
end
