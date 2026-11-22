# Static checks (Aqua.jl): method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks. An exception is listed
# with its reason.
using Aqua, ImplicitSolvers, Test

@testset "Aqua" begin
    Aqua.test_all(ImplicitSolvers)
end
