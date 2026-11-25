# GH-19/GH-18 on a device (JLArrays emulation, `LENTICULUM_DEVICE_TESTS=true`): the diagonal device
# moment methods apply the same declared-improper guard as the CPU ones — a tagged belief is a
# likelihood, and reading its moments or normaliser is an error there too — and device damping
# decides properness from the combined support, as on the CPU. Oracles: the CPU beliefs themselves.
if get(ENV, "LENTICULUM_DEVICE_TESTS", "false") == "true"
    using JLArrays
    DeviceArray = isdefined(Main, :LENTICULUM_DEVICE_ARRAY) ? Main.LENTICULUM_DEVICE_ARRAY : JLArray  # CuArray under gpu/runtests.jl (Q8)

    @testset "GH-19: device diagonal Gaussians keep the improper contract" begin
        JLArrays.allowscalar(false)
        for T in (Float32, Float64)
            b = GaussianBelief(DeviceArray(T[1]), Diagonal(DeviceArray(T[1])), true)
            c = GaussianBelief([T(1)], Diagonal([T(1)]), true)
            @test !isproper(b) && !isproper(c)
            @test_throws ArgumentError belief_mean(b)
            @test_throws ArgumentError belief_cov(b)
            @test_throws ArgumentError logpartition(b)
            @test_throws ArgumentError belief_mean(c)
            @test_throws ArgumentError belief_cov(c)
            @test_throws ArgumentError logpartition(c)
            # an untagged device belief still answers, and agrees with the CPU one
            p = GaussianBelief(DeviceArray(T[1]), Diagonal(DeviceArray(T[2])))
            q = GaussianBelief([T(1)], Diagonal([T(2)]))
            @test Array(belief_mean(p)) ≈ belief_mean(q)
            # the device covariance stays diagonal; Array(::Diagonal{<:GPUArray}) would scalar-index
            @test Array(belief_cov(p).diag) ≈ LinearAlgebra.diag(belief_cov(q))
            @test logpartition(p) ≈ logpartition(q) rtol = sqrt(eps(T))
        end
    end

    @testset "GH-18 device: damping decides properness from the combined support" begin
        JLArrays.allowscalar(false)
        a = GaussianBelief(DeviceArray([0.0, 0.0]), Diagonal(DeviceArray([1.0, 0.0])), true)
        b = GaussianBelief(DeviceArray([0.0, 0.0]), Diagonal(DeviceArray([0.0, 1.0])), true)
        @test !damp(a, b, 0.5).improper                       # complementary nullspaces: proper
        @test damp(a, GaussianBelief(DeviceArray([0.0, 0.0]), Diagonal(DeviceArray([2.0, 0.0])), true), 0.5).improper
    end
end
