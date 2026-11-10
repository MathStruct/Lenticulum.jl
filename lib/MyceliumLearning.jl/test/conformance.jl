include(joinpath(@__DIR__, "../../LenticulumCore.jl/test/support/factor_conformance.jl"))
using LuxCore, Random
@testset "StraightThroughFactor conformance" begin
    nt = (;)
    f = StraightThroughFactor(round,identity)
    check_factor_conformance(f,only(supported_polarities(f));
        π=TrivialBelief(),inputs=(in=DiracBelief(1.4),),ps=nt,st=nt,
        pointwise=(1.4,nothing,1.0),game_prior=DiracBelief(1.4),game_y=DiracBelief(1.0),
        msgs=(in=DiracBelief(1.4),out=DiracBelief(1.0)),direct_message=true,
        expected_site=DiracBelief(1.0),expected_posterior=DiracBelief(1.0),
        expected_free_energy=0.0,expected_entropy=0.0)
end
