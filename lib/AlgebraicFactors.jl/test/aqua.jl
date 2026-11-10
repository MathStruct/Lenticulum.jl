# Static checks (Aqua.jl): method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks. An exception is listed
# with its reason.
using AlgebraicFactors, Aqua, Test

@testset "Aqua" begin
    # the packages of this repository are found by [sources] paths and have no released versions
    # to bound; their compat entries come with registration
    # persistent_tasks walks every dependency's Project.toml; TreeViews (through HomotopyContinuation)
    # has none, and Aqua throws on it (2026-11-10, Aqua 0.8.18)
    Aqua.test_all(AlgebraicFactors;
        persistent_tasks = false,
        deps_compat = (ignore = [:LenticulumCore, :Mycelium],))
end
