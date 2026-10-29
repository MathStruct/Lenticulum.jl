include("../examples/hierarchical_models.jl")
using .HierarchicalModels: schools_graph_posterior, schools_dense, slopes_data, slopes_graph_posterior,
    slopes_dense, crossed_data, crossed_graph_posterior, crossed_dense

# Every reference is a dense Gaussian conditioned by plain linear algebra, not by Lenticulum.
@testset "Hierarchical models (examples/hierarchical_models.jl)" begin
    @testset "eight schools: a tree, exact at every τ" begin
        for τ in (0.5, 5.0, 25.0)
            G, D = schools_graph_posterior(τ), schools_dense(τ)
            @test G.mean ≈ D.mean rtol = 1e-10
            @test G.var ≈ D.var rtol = 1e-10
            @test G.F ≈ D.F atol = 1e-10              # −log p(y | τ)
        end
        # τ → 0 pools completely: every θⱼ → the precision-weighted mean of y (and μ's prior)
        y, s = HierarchicalModels.Y8, HierarchicalModels.S8
        pooled = sum(y ./ s .^ 2) / (sum(1 ./ s .^ 2) + 1 / HierarchicalModels.V0)
        @test all(abs.(schools_graph_posterior(1e-3).mean[2:end] .- pooled) .< 1e-4)
        G9, D9 = schools_graph_posterior(10.0; newschool = true), schools_dense(10.0; newschool = true)
        @test G9.mean ≈ D9.mean rtol = 1e-10
        @test G9.var ≈ D9.var rtol = 1e-10
        @test G9.var[end] ≈ G9.var[1] + 10.0^2 rtol = 1e-10   # new school: Var θ₉ = Var μ + τ²
    end
    @testset "random slopes: weights as variables" begin
        d = slopes_data()
        G, D = slopes_graph_posterior(d), slopes_dense(d)
        @test G.mean ≈ D.mean rtol = 1e-8
        @test all(isapprox.(G.cov, D.cov; rtol = 1e-8))
        @test G.F ≈ D.F atol = 1e-8
    end
    @testset "crossed effects: loopy means exact, variances not" begin
        c = crossed_data()
        G, D = crossed_graph_posterior(c), crossed_dense(c)
        @test G.report.converged
        @test G.loops == (c.R - 1) * (c.C - 1)
        @test G.mean ≈ D.mean rtol = 1e-9            # Weiss & Freeman (2001): converged means are exact
        @test all(G.var .< D.var)                    # overconfident on this graph
    end
end
