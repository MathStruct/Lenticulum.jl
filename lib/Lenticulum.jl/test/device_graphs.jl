using Adapt, JLArrays, LinearAlgebra, Test, Lenticulum, LenticulumCore, Mycelium
isdefined(@__MODULE__, :EltypeMatrix) || include("../../LenticulumCore.jl/test/support/eltype_matrix.jl")
isdefined(@__MODULE__, :DeviceMatrix) || include("../../LenticulumCore.jl/test/support/device_matrix.jl")

function q7_chain(; rule = BeliefPropagation(), freeze = ())
    b = GraphBuilder()
    variable!(b, :x, 2)
    variable!(b, :y, 2)
    variable!(b, :z, 1)
    factor!(b, :prior, GaussianPrior(:x, [0.2, 0.4], [1.0 0.1; 0.1 0.8]))
    transition = rule isa MeanField ?
        LinearConstraintFactor((x = 2, y = 2), 2; noise = [0.4 0.1; 0.1 0.6]) :
        GaussianFactor(2 => 2; noise = [0.4 0.1; 0.1 0.6])
    factor!(b, :transition, transition; rule, freeze)
    factor!(b, :observation, GaussianFactor(2 => 1; noise=0.3))
    factor!(b, :data, DataFactor(:z, [0.8]))
    connect!(b, :prior, :x, :x)
    connect!(b, :transition, :x, :x)
    connect!(b, :transition, :y, :y)
    connect!(b, :observation, :x, :y)
    connect!(b, :observation, :y, :z)
    connect!(b, :data, :z, :z; direction=Emitting())
    ps = (transition=(A=[0.9 0.1; 0.0 0.8], b=[0.0, 0.1]), observation=(A=[0.3 0.5], b=[0.1]))
    if rule isa MeanField
        ps = merge(ps, (transition = (A = (x = ps.transition.A, y = -Matrix(1.0I, 2, 2)), c = ps.transition.b),))
    end
    return build(b), ps
end

function q7_switch(; rule = BeliefPropagation())
    b = GraphBuilder()
    variable!(b, :x, 1)
    variable!(b, :y, 1)
    variable!(b, :k, 2)
    factor!(b, :switch, SwitchingConstraintFactor((x=1, y=1), 2, 1; noise=0.4); rule)
    factor!(b, :data, DataFactor(:x, [0.8]))
    connect!(b, :switch, :x, :x)
    connect!(b, :switch, :y, :y)
    connect!(b, :switch, :k, :k)
    connect!(b, :data, :x, :x; direction=Emitting())
    if rule isa ExpectationPropagationRule
        factor!(b, :cavity, GaussianPrior(:y, 0.2, 1.5))
        connect!(b, :cavity, :y, :y)
    end
    ps = (switch=(A=(x=fill(-0.9, 1, 1), y=ones(1, 1)), b=[-0.3 0.5], logw=[0.2, -0.5]),)
    return build(b), ps
end

isdefined(DeviceMatrix, :device_block_payload) || include("../../Mycelium.jl/test/support/device_blocks.jl")

function check_adapted_graph(g, ps, schedule)
    D = DeviceMatrix.device_array()
    cpu, report, _, s = infer!(g, schedule(g), ps, (;))
    dg, dp = adapt(D, (g, ps))
    gpu, dev_report, _, ds = infer!(dg, schedule(dg), dp, (;))
    @test report.converged && dev_report.converged
    @test keys(gpu) == keys(cpu)
    for n in keys(cpu)
        @test DeviceMatrix.device_agrees(gpu[n], cpu[n], 1e-10)
        @test DeviceMatrix.on_device(DeviceMatrix.device_payload(gpu[n]))
    end
    @test first(LenticulumCore.scalar_free_energy(ds, dg, dp, (;))) ≈
          first(LenticulumCore.scalar_free_energy(s, g, ps, (;))) rtol=1e-10 atol=1e-12
    for (host, device) in ((s.to_variable, ds.to_variable), (s.to_factor, ds.to_factor))
        @test map(isnothing, host.blocks) == map(isnothing, device.blocks)
        for (a, b) in zip(host.blocks, device.blocks)
            b === nothing && continue
            @test DeviceMatrix.device_agrees(b.belief, a.belief, 1e-10)
            @test DeviceMatrix.on_device(DeviceMatrix.device_payload(b.belief))
        end
    end
    copied = adapt(D, s)
    @test DeviceMatrix.on_device(DeviceMatrix.device_payload(marginal(copied, dg, 1)))
    return dg, dp, ds
end

function q20_plate(; clustered)
    b = GraphBuilder()
    plate!(b, :replicas, 2) do p, i
        variable!(p, :x, 1)
        variable!(p, :y, 1)
        factor!(p, :data, DataFactor(:x, [0.4i]); tie = false)
        connect!(p, :data, :x, :x; direction = Emitting())
        if clustered
            variable!(p, :k, 2)
            factor!(p, :transition, SwitchingConstraintFactor((x = 1, y = 1), 2, 1; noise = 0.4))
            connect!(p, :transition, :k, :k)
        else
            factor!(p, :transition, GaussianFactor(1 => 1; noise = 0.4))
        end
        connect!(p, :transition, :x, :x)
        connect!(p, :transition, :y, :y)
    end
    g = build(b)
    if clustered
        ps = (transition = (A = (x = [-0.9;;], y = [1.0;;]), b = [-0.3 0.5], logw = [0.2, -0.5]),)
        cg = cluster(g, :replicas, :cluster, [:transition])
        return cg, cluster_parameters(cg, ps)
    end
    return g, (transition = (A = [0.9;;], b = [0.1]),)
end

@testset "Adapted factor graphs" begin
    JLArrays.allowscalar(false)
    @testset "Belief propagation" begin
        for (g, ps) in (q7_chain(), q7_switch())
            check_adapted_graph(g, ps, tree_schedule)
        end
    end
    @testset "Mean field" begin
        check_adapted_graph(q7_chain(; rule = MeanField())..., g -> SequentialSchedule(all_tasks(g)))
    end
    @testset "EP mixture site" begin
        check_adapted_graph(q7_switch(; rule = ExpectationPropagationRule())...,
                            g -> SequentialSchedule(all_tasks(g)))
    end
    @testset "Frozen coordinates" begin
        g, ps = q7_chain(; freeze = (:b,))
        dg, dp, _ = check_adapted_graph(g, ps, tree_schedule)
        node = dg.factors[dg.factor_index[:transition]]
        pinned = Mycelium._factor_parameters(node, dp.transition)
        @test DeviceMatrix.device_agrees(pinned, ps.transition, 1e-10)
        @test DeviceMatrix.on_device(pinned)
    end
    @testset "Plate and batched cluster" begin
        for clustered in (false, true)
            g, ps = q20_plate(; clustered)
            _, _, store = check_adapted_graph(g, ps, plate_schedule)
            @test !isempty(Mycelium._energy_rows(store, g))
        end
    end
end
