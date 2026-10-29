using Mycelium, LenticulumCore, Test

@testset "LP-018.9 particle distance normalises finite overflowing weights" begin
    points = [[1.], [3.]]
    ordinary = SampleBelief(points, [1., 1.])
    for container in (identity, Tuple)
        @test belief_distance(SampleBelief(points, container([1e308, 1e308])), ordinary) == 0.
        # Independent total variation between (2/3,1/3) and (1/2,1/2).
        @test belief_distance(SampleBelief(points, container([1.6e308, 8e307])), ordinary) ≈ 1/6
    end
end
