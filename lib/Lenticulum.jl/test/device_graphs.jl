using Adapt, JLArrays, LinearAlgebra, Test, Lenticulum, LenticulumCore, Mycelium
const DeviceArray = isdefined(Main, :LENTICULUM_DEVICE_ARRAY) ? Main.LENTICULUM_DEVICE_ARRAY : JLArray  # CuArray under gpu/runtests.jl (Q8)

function q7_chain()
    b = GraphBuilder()
    variable!(b, :x, 2)
    variable!(b, :y, 2)
    variable!(b, :z, 1)
    factor!(b, :prior, GaussianPrior(:x, [0.2, 0.4], [1.0 0.1; 0.1 0.8]))
    factor!(b, :transition, GaussianFactor(2 => 2; noise=[0.4 0.1; 0.1 0.6]))
    factor!(b, :observation, GaussianFactor(2 => 1; noise=0.3))
    factor!(b, :data, DataFactor(:z, [0.8]))
    connect!(b, :prior, :x, :x)
    connect!(b, :transition, :x, :x)
    connect!(b, :transition, :y, :y)
    connect!(b, :observation, :x, :y)
    connect!(b, :observation, :y, :z)
    connect!(b, :data, :z, :z; direction=Emitting())
    ps = (transition=(A=[0.9 0.1; 0.0 0.8], b=[0.0, 0.1]), observation=(A=[0.3 0.5], b=[0.1]))
    return build(b), ps
end

function q7_switch()
    b = GraphBuilder()
    variable!(b, :x, 1)
    variable!(b, :y, 1)
    variable!(b, :k, 2)
    factor!(b, :switch, SwitchingConstraintFactor((x=1, y=1), 2, 1; noise=0.4))
    factor!(b, :data, DataFactor(:x, [0.8]))
    connect!(b, :switch, :x, :x)
    connect!(b, :switch, :y, :y)
    connect!(b, :switch, :k, :k)
    connect!(b, :data, :x, :x; direction=Emitting())
    ps = (switch=(A=(x=fill(-0.9, 1, 1), y=ones(1, 1)), b=[-0.3 0.5], logw=[0.2, -0.5]),)
    return build(b), ps
end

@testset "Adapted factor graphs" begin
    JLArrays.allowscalar(false)
    for (g, ps) in (q7_chain(), q7_switch())
        cpu, report, _, s = infer!(g, tree_schedule(g), ps, (;))
        dg, dp = adapt(DeviceArray, (g, ps))
        gpu, dev_report, _, ds = infer!(dg, tree_schedule(dg), dp, (;))
        @test report.converged && dev_report.converged
        @test keys(gpu) == keys(cpu)
        for n in keys(cpu)
            @test DeviceMatrix.device_agrees(gpu[n], cpu[n], 1e-10)
            @test DeviceMatrix.on_device(DeviceMatrix.device_payload(gpu[n]))
        end
        @test first(LenticulumCore.scalar_free_energy(ds, dg, dp, (;))) ≈
              first(LenticulumCore.scalar_free_energy(s, g, ps, (;))) rtol=1e-10
        copied = adapt(DeviceArray, s)
        @test DeviceMatrix.on_device(DeviceMatrix.device_payload(marginal(copied, dg, 1)))
    end
end
