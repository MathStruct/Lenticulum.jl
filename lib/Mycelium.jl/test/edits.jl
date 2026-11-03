# Structure edits on Mycelium's own factors (`edits.md`); Lenticulum's `test/structure_edits.jl`
# checks edited graphs against graphs built directly and their exact free energies.
@testset "structure edits: Mycelium's side" begin
    G(μ, v) = LenticulumCore.Gaussian([μ], fill(v, 1, 1))
    b = GraphBuilder(); variable!(b, :x, 1); variable!(b, :y, 1)
    factor!(b, :px, PriorFactor(:x, G(0.0, 1.0), _ -> 0.0)); connect!(b, :px, :x, :x)
    factor!(b, :py, PriorFactor(:y, G(1.0, 2.0), _ -> 0.0)); connect!(b, :py, :y, :y)
    factor!(b, :qx, PriorFactor(:x, G(2.0, 4.0), _ -> 0.0)); connect!(b, :qx, :x, :x)
    g = build(b)
    s = MessageStore(g); sweep!(s, g, SequentialSchedule(all_tasks(g)), NamedTuple(), NamedTuple())
    # remove a variable with its factor: ids are renumbered, names kept
    e = GraphBuilder(g); remove_factor!(e, :py); remove_variable!(e, :y); g2 = build(e)
    @test nvariables(g2) == 1 && nfactors(g2) == 2 && all(f -> g2.factors[f.id] === f, g2.factors)
    @test Set(f.name for f in g2.factors) == Set([:px, :qx])
    t = transfer(s, g, g2)
    @test count(i -> t.to_variable[i] !== nothing, 1:nedges(g2)) == 2
    @test LenticulumCore.belief_mean(marginal(t, g2, g2.variable_index[:x])) ≈ LenticulumCore.belief_mean(marginal(s, g, g.variable_index[:x]))
    # replace a factor: edges on channels it keeps stay
    e = GraphBuilder(g); replace_factor!(e, :qx, PriorFactor(:x, G(-1.0, 1.0), _ -> 0.0)); g3 = build(e)
    @test nedges(g3) == nedges(g) && g3.factors[g3.factor_index[:qx]].factor.belief.η ≈ [-1.0]
    # disconnect, and add back
    e = GraphBuilder(g); disconnect!(e, :qx, :x); @test length(e.edges) == 2
    connect!(e, :qx, :x, :x); @test nedges(build(e)) == 3
    @test_throws ArgumentError remove_variable!(GraphBuilder(g), :nope)
    @test_throws ArgumentError replace_factor!(GraphBuilder(g), :nope, PriorFactor(:x, G(0.0, 1.0), _ -> 0.0))
end
