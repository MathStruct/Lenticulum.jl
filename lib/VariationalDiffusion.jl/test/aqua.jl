# Static checks (Aqua.jl): method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks. An exception is listed
# with its reason.
using Aqua, Test, VariationalDiffusion

@testset "Aqua" begin
    # the packages of this repository are found by [sources] paths and have no released versions
    # to bound; their compat entries come with registration
    Aqua.test_all(VariationalDiffusion;
        deps_compat = (ignore = [:LenticulumCore, :Mycelium],))
end
