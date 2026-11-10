# Static checks (Aqua.jl): method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks. An exception is listed
# with its reason.
using Adversarial, Aqua, LenticulumCore, Test

@testset "Aqua" begin
    # the packages of this repository are found by [sources] paths and have no released versions
    # to bound; their compat entries come with registration
    # Known: belief_mean and belief_cov of LenticulumCore's SampleBelief are defined here (ratio.jl),
    # so they exist only with Adversarial loaded. They belong in LenticulumCore beside the type;
    # move them when the package comes back (vault/Project/Refactor 2026-11.md).
    Aqua.test_all(Adversarial;
        deps_compat = (ignore = [:LenticulumCore, :Mycelium],),
        piracies = (treat_as_own = [LenticulumCore.SampleBelief],))
end
