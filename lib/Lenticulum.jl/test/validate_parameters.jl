# validate_parameters (review of 2026-11-04, LP-023): a graph-wide parameter tree is checked against
# the nodes' initial parameters before inference reads a missing key as an empty tree.
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using MyceliumLearning

@testset "validate_parameters" begin
    function graph(f; freeze = ())
        b = GraphBuilder(); variable!(b, :y, 1); variable!(b, :x, 1)
        factor!(b, :dy, DataFactor(:y, [0.3])); connect!(b, :dy, :y, :y; direction = Emitting())
        factor!(b, :rel, f; freeze); connect!(b, :rel, :y, :y); connect!(b, :rel, :x, :x)
        return build(b)
    end
    g = graph(LinearConstraintFactor((y = 1, x = 1), 1))
    good = (rel = (A = (y = ones(1, 1), x = ones(1, 1)), c = [0.0]),)
    @test validate_parameters(g, good) === nothing                        # the data node needs no entry
    @test_throws ArgumentError validate_parameters(g, NamedTuple())        # missing :rel
    @test_throws ArgumentError validate_parameters(g, (rel = (A = (y = ones(1, 1), x = ones(2, 1)), c = [0.0]),))   # a size
    @test_throws ArgumentError validate_parameters(g, (rel = (A = (y = ones(1, 1),), c = [0.0]),))                   # a key
    @test_logs (:warn, r"no node reads: rell") validate_parameters(g, merge(good, (rell = NamedTuple(),)))
    # a declared output's coefficient passed as a parameter
    go = graph(LinearConstraintFactor((y = 1, x = 1), 1; output = :y))
    @test validate_parameters(go, (rel = (A = (x = ones(1, 1),), c = [0.0]),)) === nothing
    @test_throws ArgumentError validate_parameters(go, good)
    # frozen paths must exist
    @test validate_parameters(graph(LinearConstraintFactor((y = 1, x = 1), 1); freeze = (:A, :x)), good) === nothing
    @test_throws ArgumentError validate_parameters(graph(LinearConstraintFactor((y = 1, x = 1), 1); freeze = (:A, :z)), good)
end
