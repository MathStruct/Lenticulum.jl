# Static checks (Aqua.jl): method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks. An exception is listed
# with its reason.
using Aqua, Mycelium, MyceliumLearning, Test

@testset "Aqua" begin
    # the packages of this repository are found by [sources] paths and have no released versions
    # to bound; their compat entries come with registration
    # MyceliumLearning is the learning half of Mycelium (split in 2026-11): its methods of
    # LenticulumCore.free_energy_gradient on Mycelium's store and graph are the split, not piracy.
    Aqua.test_all(MyceliumLearning;
        deps_compat = (ignore = [:LenticulumCore, :Mycelium],),
        piracies = (treat_as_own = [Mycelium.MessageStore, Mycelium.FactorGraph],))
end
