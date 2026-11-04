# Review of 2026-11-04: the implicit pullback at a singular free-block Jacobian (a fold or branch
# point) is refused instead of returning Inf/NaN; the Laplace covariance of a non-conservative
# field warns that it is a curvature measure. A stub relation supplies the Jacobian.
struct _StubRelation <: VariationalDiffusion.AbstractImplicitRelation
    J::Matrix{Float64}
end
VariationalDiffusion.prior_jacobian(m::_StubRelation, z, ps, st; coords = Colon()) = m.J[coords, coords]

@testset "implicit pullback and Laplace at singular or non-conservative Jacobians" begin
    z, ρ, b = zeros(2), zeros(2), [1.0, -1.0]
    @test VariationalDiffusion._adjoint_solve(DenseSolve(), _StubRelation([2.0 0.5; 0.5 1.0]), z, ρ, b, NamedTuple(), NamedTuple()) ≈
          [2.0 0.5; 0.5 1.0]' \ b
    @test_throws ArgumentError VariationalDiffusion._adjoint_solve(DenseSolve(), _StubRelation([1.0 1.0; 1.0 1.0]), z, ρ, b,
                                                                   NamedTuple(), NamedTuple())
    sol = VariationalDiffusion.ImplicitSolution(z, 0.0, 1, true, true)
    @test_logs (:warn, r"not symmetric") implicit_laplace(_StubRelation([2.0 0.5; -0.5 1.0]), sol, ρ, NamedTuple(), NamedTuple())
    @test_logs implicit_laplace(_StubRelation([2.0 0.5; 0.5 1.0]), sol, ρ, NamedTuple(), NamedTuple())
end
