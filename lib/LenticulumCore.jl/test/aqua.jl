# Static checks (Aqua.jl): method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks. An exception is listed
# with its reason.
using Aqua, LenticulumCore, Test

@testset "Aqua" begin
    Aqua.test_all(LenticulumCore)
end
