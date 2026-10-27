include("../examples/gaussian_structure_search.jl")
using .GaussianStructureSearch: chain_sample, fit_precision, bethe_score, greedy_search,
    graphical_lasso, edge_metrics, exact_nll

@testset "LP-010 Gaussian structure search" begin
    data = chain_sample(; d = 6, n = 20000, seed = 1011)
    recovered = greedy_search(data.S, data.n)
    @test recovered.edges == data.edges
    @test edge_metrics(recovered.edges, data.edges) == (precision = 1.0, recall = 1.0)
    @test all(diff([row.score for row in recovered.trajectory]) .< 0)
    # Every intermediate graph is a forest: BP normalisers must match dense integration.
    @test all(abs(row.nll - row.exact) < 1e-5 for row in recovered.trajectory)
    @test inv(Symmetric(recovered.K))[1, 2] ≈ data.S[1, 2] atol = 1e-8

    # Independent optimisation oracles: unconstrained MLE and glasso KKT conditions.
    d = size(data.S, 1)
    complete = [(i, j) for i in 1:d for j in (i + 1):d]
    @test fit_precision(data.S, complete) ≈ inv(data.S) atol = 1e-8
    @test graphical_lasso(data.S, 0.0).K ≈ inv(data.S) atol = 1e-7
    fit = graphical_lasso(data.S, 0.06)
    G = data.S - inv(Symmetric(fit.K))
    @test maximum(abs, diag(G)) < 1e-7
    @test all(abs(fit.Z[i, j]) > 1e-6 ?
              abs(G[i, j] + 0.06 * sign(fit.Z[i, j])) < 1e-7 :
              abs(G[i, j]) <= 0.06 + 1e-7 for (i, j) in complete)

    # A loop really exercises the Bethe approximation; do not claim exact evidence.
    loopedges = vcat(data.edges, [(1, 3)])
    K = copy(data.K)
    K[1, 3] = K[3, 1] = -0.1
    loopy = bethe_score(K, loopedges, data.S, data.n)
    @test isfinite(loopy.score)
    @test abs(loopy.nll - exact_nll(K, data.S, data.n)) > 1e-3
end
