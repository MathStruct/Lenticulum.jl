# LP-022 for toric factors: unnormalised, a learned potential on observed states has no lower bound;
# normalised (with its log partition) it is −log of a probability, never below 0.
using AlgebraicFactors, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Test

@testset "LP-022: toric factors on observed data" begin
    A = vcat(independence((:a, :b), (2, 2)), [0 1 1 0])           # with one interaction row
    function graph(normalised)
        b = GraphBuilder()
        variable!(b, :a); variable!(b, :b)
        factor!(b, :ab, ToricFactor((:a, :b), (2, 2), A; normalised))
        connect!(b, :ab, :a, :a; direction = Bidirectional()); connect!(b, :ab, :b, :b; direction = Bidirectional())
        factor!(b, :da, DataFactor(:a, 1)); connect!(b, :da, :a, :a; direction = Emitting())
        factor!(b, :db, DataFactor(:b, 2)); connect!(b, :db, :b, :b; direction = Emitting())
        return build(b)
    end
    x = LinearIndices((2, 2))[1, 2]                                   # the observed joint state
    F(g, t) = (ps = (ab = (η = t .* Float64.(A[:, x]),), da = NamedTuple(), db = NamedTuple());
               s = MessageStore(g); sweep!(s, g, SequentialSchedule(all_tasks(g)), ps, NamedTuple());
               first(scalar_free_energy(s, g, ps, NamedTuple())))
    gu, gn = graph(false), graph(true)
    Fu = [F(gu, t) for t in (1.0, 10.0, 100.0)]
    @test issorted(Fu; rev = true) && Fu[3] < -100                  # unbounded below
    @test all(F(gn, t) >= -1e-12 for t in (1.0, 10.0, 100.0))       # −log p(x) ≥ 0
    @test Mycelium._unnormalised_nodes(gu) == [:ab] && isempty(Mycelium._unnormalised_nodes(gn))
end
