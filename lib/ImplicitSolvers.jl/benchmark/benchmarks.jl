using BenchmarkTools
using ForwardDiff, ImplicitSolvers, LinearAlgebra
import Krylov # activates ImplicitSolvers' weak-dependency extension
using Random: Xoshiro

include(joinpath(@__DIR__, "../../../benchmark/fixtures.jl"))

const SUITE::BenchmarkGroup = BenchmarkGroup()
solvers = SUITE["ImplicitSolvers"] = BenchmarkGroup()
linear = solvers["linear"] = BenchmarkGroup()
nonlinear = solvers["nonlinear"] = BenchmarkGroup()

const systems = cached_fixture() do
    n = 256
    rng = Xoshiro(4242)
    W = randn(rng, n, n) / (4sqrt(n))
    b = randn(rng, n)
    return Matrix(I + W), Symmetric(Matrix(I + W' * W)), b, ForwardDiff.Dual.(b, randn(rng, n))
end

linear["dense"] = @benchmarkable linsolve(A, b, DenseSolve()) setup=((A, symmetric, b, dual) = $systems())
linear["gmres"] = @benchmarkable linsolve(A, b, GMRES()) setup=((A, symmetric, b, dual) = $systems())
linear["minres"] = @benchmarkable linsolve(A, b, MINRES()) setup=((general, A, b, dual) = $systems())
linear["krylovjl_gmres"] = @benchmarkable linsolve(A, b, ImplicitSolvers.KrylovJL(:gmres)) setup=((A, symmetric, b, dual) = $systems())
linear["krylovjl_minres"] = @benchmarkable linsolve(A, b, ImplicitSolvers.KrylovJL(:minres)) setup=((general, A, b, dual) = $systems())
linear["forwarddiff_gmres"] = @benchmarkable linsolve(A, b, GMRES()) setup=((A, symmetric, rhs, b) = $systems())

root_residual(u) = [u[1]^2 - 2, u[2] + sin(u[1])]
nonlinear["broyden"] = @benchmarkable solve_root(root_residual, u, Broyden()) setup=(u = [1.0, 0.0]) evals=1

function column_problem()
    H = Symmetric([2.0 1.0; 1.0 3.0])
    root = [1.0, -2.0]
    residual(u) = H * (u - root)
    energy(u) = dot(u - root, residual(u)) / 2
    return (residual = (U, cols) -> stack(map(residual, eachcol(U))),
            scale = (U, cols) -> map(norm, eachcol(U)),
            system = (U, cols) -> (fill(true, size(U, 2)), fill(r -> H \ r, size(U, 2))),
            merit = (U, R, cols) -> map(energy, eachcol(U)),
            model = (U, R, cols) -> [(R[:, b], H) for b in axes(U, 2)],
            stable = (U, cols) -> fill(true, size(U, 2)))
end

nonlinear["columns_newton"] = @benchmarkable solve_columns(p, U, NewtonDescent(); tol=Tolerance(), maxiters=12) setup=((p, U) = (column_problem(), zeros(2, 8))) evals=1
nonlinear["columns_trust_region"] = @benchmarkable solve_columns(p, U, TrustRegion(); tol=Tolerance(), maxiters=12) setup=((p, U) = (column_problem(), zeros(2, 8))) evals=1
