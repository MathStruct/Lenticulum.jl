# Static checks (Aqua.jl): method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks. An exception is listed
# with its reason.
using Aqua, Lenticulum, Mycelium, Test

@testset "Aqua" begin
    # the packages of this repository are found by [sources] paths and have no released versions
    # to bound; their compat entries come with registration
    # Known: Mycelium.reduce_message for a mixture of Gaussians (LenticulumCore's types) is defined
    # here, beside Runnalls' merge (beliefs.jl). It belongs in Mycelium with the Gaussian belief
    # algebra; until it moves, Mycelium without Lenticulum refuses to reduce a Gaussian mixture.
    Aqua.test_all(Lenticulum;
        deps_compat = (ignore = [:LenticulumCore, :Mycelium, :MyceliumLearning],),
        piracies = (treat_as_own = [Mycelium.reduce_message],))
end
