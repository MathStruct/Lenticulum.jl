# LP-022 for DEQs: ½‖z − g(z, x)‖² is not a density of z, and with z observed a cell that is the
# identity in z puts every pair on the relation (`vault/Energy-Based/Why Learned Relations Do Not
# Collapse.md`). Oracle: the residual energy in closed form.
using ImplicitLayers, LenticulumCore, Mycelium, LuxCore, LinearAlgebra, Random, Test
using ADTypes, DifferentiationInterface, ForwardDiff

struct LP22Cell <: LuxCore.AbstractLuxLayer end
LuxCore.initialparameters(::AbstractRNG, ::LP22Cell) = (W = zeros(1, 1), U = ones(1, 1), b = zeros(1))
LuxCore.initialstates(::AbstractRNG, ::LP22Cell) = NamedTuple()
(::LP22Cell)(inp, ps, st) = (ps.W * inp[1] .+ ps.U * inp[2] .+ ps.b, st)

@testset "LP-022: a DEQ trained by its residual energy with the state observed" begin
    rng = Xoshiro(22)
    N = 50
    x = randn(rng, N); z = 2 .* x .+ 0.3 .+ 0.1 .* randn(rng, N)
    function graph(observe_z)
        b = GraphBuilder()
        for i in 1:N
            r(s) = replica_name(s, i)
            variable!(b, r(:x), 1); variable!(b, r(:z), 1)
            factor!(b, r(:dx), DataFactor(:x, [x[i]])); connect!(b, r(:dx), :x, r(:x); direction = Emitting())
            if observe_z
                factor!(b, r(:dz), DataFactor(:z, [z[i]])); connect!(b, r(:dz), :z, r(:z); direction = Emitting())
            end
            factor!(b, r(:deq), DEQFactor(LP22Cell(), (x = 1, z = 1)); params = :deq)
            connect!(b, r(:deq), :x, r(:x)); connect!(b, r(:deq), :z, r(:z))
        end
        return build(b)
    end
    g = graph(true)
    data = NamedTuple(s => NamedTuple() for i in 1:N for s in (replica_name(:dx, i), replica_name(:dz, i)))
    # only the data's messages are needed: the free energy reads the DEQ's incoming points
    tasks = [t for t in all_tasks(g) if !(t.kind === :to_variable && g.factors[g.edges[t.edge].factor].factor isa DEQFactor)]
    function F(p)
        local ps, s                                  # not the testset's `ps` and `s`
        ps = merge(data, (deq = p,)); s = MessageStore(g)
        for _ in 1:2; sweep!(s, g, SequentialSchedule(tasks), ps, NamedTuple()); end
        return first(scalar_free_energy(s, g, ps, NamedTuple())), s, ps
    end
    fit = (W = zeros(1, 1), U = fill(2.0, 1, 1), b = [0.3])       # the generating relation
    ident = (W = ones(1, 1), U = zeros(1, 1), b = [0.0])         # g(z, x) = z
    Ffit, s, ps = F(fit)
    @test Ffit ≈ sum((z .- 2 .* x .- 0.3) .^ 2) / 2 rtol = 1e-10
    @test first(F(ident)) == 0.0 < Ffit                           # every pair on the relation
    # the gradient at the generating relation points towards it (−Σ rᵢ zᵢ in W)
    G = free_energy_gradient(s, g, ps, NamedTuple(); ad = AutoForwardDiff()).deq
    r = z .- 2 .* x .- 0.3
    @test only(G.W) ≈ -sum(r .* z) rtol = 1e-8
    # the warning fires with the state observed, not with the input alone
    @test length(MyceliumLearning._unnormalised_nodes(g)) == N              # one per replica node
    @test isempty(MyceliumLearning._unnormalised_nodes(graph(false)))
end
