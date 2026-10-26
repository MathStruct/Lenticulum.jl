using ADTypes
struct ParameterResidual <: AbstractLenticulumFactor end
LenticulumCore.energy(::ParameterResidual, x, a, y, ps, st) =
    (y .- ps.block.A * x .- ps.b .- ps.c[1], (calls = st.calls + 1,))
LenticulumCore.pointwise_scalarisation(::ParameterResidual) = SquaredNorm()
LenticulumCore.free_energy(f::ParameterResidual, x, y, ps, st) = scalar_energy(f, x, nothing, y, ps, st)
@testset "LP-002 unloaded AD extension hint" begin
    @test Base.get_extension(LenticulumCore, :LenticulumCoreDifferentiationInterfaceExt) === nothing
    err = try
        energy_jacobian(ParameterResidual(), [1.0], nothing, [2.0],
            (block=(A=ones(1,1),), b=[0.0], c=(0.0,)), (calls=0,);
            wrt=:ps, ad=AutoForwardDiff())
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("using DifferentiationInterface", sprint(showerror, err))
end
using DifferentiationInterface, ForwardDiff, Zygote, Enzyme
@testset "LP-002 parameter trees and scalar gradients" begin
    f = ParameterResidual()
    ps = (block = (A = [0.3 0.2; 0.1 -0.4], empty = (;)), b = [0.2, -0.1], c = (0.4,))
    x, y, st = [1.2, -0.7], [0.4, 0.9], (calls = 0,)
    Id = Matrix{Float64}(I, 2, 2)
    J = -hcat(kron(x', Id), Id, ones(2))
    r = y - ps.block.A * x - ps.b .- ps.c[1]
    for ad in (nothing, AutoForwardDiff(), AutoZygote(), AutoEnzyme())
        got = energy_jacobian(f, x, nothing, y, ps, st; wrt = :ps, ad)
        @test got ≈ J atol = 2e-9
        @test gauss_newton_metric(f, x, nothing, y, ps, st; wrt = :ps, ad) ≈ J'J atol = 1e-8
        g = energy_gradient(f, x, nothing, y, ps, st; ad)
        @test g.block.A ≈ -r * x' atol = 2e-9
        @test g.b ≈ -r atol = 2e-9
        @test g.c[1] ≈ -sum(r) atol = 2e-9
        @test g.block.empty === (;)
        @test LenticulumCore._flatten(free_energy_gradient(f, x, y, ps, st; ad)) ≈ J'r atol = 2e-9
        @test energy_jacobian(f, x, nothing, y, ps, st; wrt = :x, ad) ≈ -ps.block.A atol = 2e-9
        @test energy_jacobian(f, x, nothing, y, ps, st; wrt = :y, ad) ≈ Id atol = 2e-9
        @test parameter_gradient(_ -> 1.0, (;); ad) === (;)
    end
    @test st.calls == 0
    @test_throws ArgumentError energy_jacobian(f, x, nothing, y, ps, st; wrt = :bad)
    @test_throws ArgumentError energy_gradient(f, x, nothing, y, ps, st; ad = :unknown)
end

@testset "LP-002 empty subtrees do not widen Float32 parameters" begin
    ps = (empty=(;), block=(w=Float32[1,2],))
    @test eltype(LenticulumCore._flatten(ps)) === Float32
    for ad in (AutoForwardDiff(), AutoZygote(), AutoEnzyme())
        g = parameter_gradient(p -> sum(abs2, p.block.w), ps; ad)
        @test eltype(g.block.w) === Float32
        @test g.block.w ≈ Float32[2,4]
        @test g.empty === (;)
    end
end

@testset "LP-002 graded scalarisation and inactive parameters" begin
    σ = GradedScalarisation((a=IdentityScalarisation(), b=IdentityScalarisation()))
    for ad in (nothing, AutoForwardDiff(), AutoZygote(), AutoEnzyme())
        ps = (w=0.7, v=[0.2, -0.3])
        g = parameter_gradient(ps; ad) do p
            scalarise(σ, GradedEnergy((a=p.w^2, b=4sum(p.v))))
        end
        @test g.w ≈ 2ps.w atol=1e-8
        @test g.v ≈ [4,4] atol=1e-8
        zero_grad = parameter_gradient(_ -> 0.0, ps; ad)
        @test zero_grad.w == 0
        @test iszero(zero_grad.v)
        empty = parameter_gradient(_ -> scalarise(GradedScalarisation((;)), GradedEnergy((;))), ps; ad)
        @test empty.w == 0
        @test iszero(empty.v)
    end
end
