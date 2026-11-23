@testset "D4 linear systems against backslash" begin
    rng = Xoshiro(2810)
    for T in (Float32, Float64), n in (1, 2, 50), spd in (false, true)
        R = randn(rng, T, n, n)
        R = spd ? R' * R : R
        A = Matrix{T}(I, n, n) + R / (3 * norm(R))
        b = randn(rng, T, n)
        xref = A \ b
        bound = max(T(1e-9), 30 * eps(T)) * (1 + norm(b))
        error_bound = opnorm(inv(A)) * bound + 8 * eps(T) * norm(xref)
        for (solver, operator) in Iterators.product(
            (DenseSolve(; spd), GMRES(restart=3)), (A, v -> A * v))
            x, report = linsolve(operator, b, solver)
            @test report.converged
            @test norm(x - xref) <= error_bound
            @test norm(b - A * x) <= bound
            @test report.residual ≈ norm(b - A * x)
            @test report.residual isa T
            @test eltype(x) === T
        end
    end

    A = [1.0 0; 0 2]
    b = ones(2)
    tol = Tolerance(atol=0, rtol=1e-12)
    calls = Ref(0)
    apply = v -> (calls[] += 1; A * v)
    x, report = linsolve(apply, b, GMRES(maxiters=1); tol)
    check_report(report, :maxiters, 1, 3)
    @test calls[] == report.nfe
    # Minimize ||b - α A b|| over the first Krylov space.
    @test x ≈ [3 / 5, 3 / 5]
    @test report.residual ≈ sqrt(1 / 5)
    x, report = linsolve(A, b, GMRES(restart=1); tol)
    @test report.iterations > 1
    @test report.residual <= 1e-12 * sqrt(2)
    @test x ≈ [1, 1 / 2] atol=2e-12

    x, report = linsolve(zeros(2, 2), b, GMRES())
    check_report(report, :stalled, 1, 3)
    @test x == zeros(2)
    @test report.residual == sqrt(2)
    x, report = linsolve(A, b, GMRES(maxiters=0))
    check_report(report, :maxiters, 0, 1)
    @test x == zeros(2)
    @test report.residual == sqrt(2)
    x, report = linsolve(A, zeros(2), GMRES())
    check_report(report, :converged, 0, 1)
    @test x == zeros(2)
    @test report.residual === 0.0

    for (rtol, reason) in ((1.0, :converged), (prevfloat(1.0), :maxiters))
        # At x=0 the residual equals the RHS scale exactly.
        _, report = linsolve(identity, [2.0], GMRES(maxiters=0);
                             tol=Tolerance(; atol=0, rtol))
        check_report(report, reason, 0, 1)
        @test report.residual === 2.0
    end

    # Scaling A and b together must not trigger an absolute Arnoldi breakdown test.
    for magnitude in (1e-80, 1e80)
        x, report = linsolve(magnitude .* A, magnitude .* b, GMRES(); tol)
        @test report.converged
        @test x ≈ [1.0, 0.5] atol=2e-12
        @test report.residual <= 1e-12 * magnitude * sqrt(2)
    end

    for solver in (DenseSolve(), DenseSolve(spd=true), GMRES())
        x, report = linsolve(zeros(Float32, 0, 0), Float32[], solver)
        check_report(report, :converged, 0, 0)
        @test isempty(x)
        @test report.residual === 0.0f0
        x, report = linsolve(fill(NaN, 1, 1), [1.0], solver)
        @test report.reason === :nonfinite
        @test isnan(report.residual)
        @test_throws DimensionMismatch linsolve(ones(2, 1), b, solver)
    end

    for solver in (DenseSolve(), DenseSolve(spd=true), GMRES())
        rhs = [0.5, 0.25]
        p = ForwardDiff.Dual{:p}(rhs[1], 1.0)
        x, report = linsolve(A, [p, rhs[2]], solver; tol)
        _, reference = linsolve(A, rhs, solver; tol)
        check_report(report, :converged, solver isa GMRES ? 2 : 1, solver isa GMRES ? 4 : 1)
        @test report.iterations == reference.iterations
        @test report.residual isa Float64
        @test primal.(x) ≈ [0.5, 0.125]
        @test ForwardDiff.partials.(x, 1) ≈ [1.0, 0.0]
    end

    for solver in (DenseSolve(), GMRES())
        x, report = linsolve(BigFloat[4 1; 1 3], BigFloat[2, 1], solver;
                             tol=Tolerance(atol=big"1e-65", rtol=0))
        @test report.converged
        @test report.residual isa BigFloat
        @test x ≈ BigFloat[5 // 11, 2 // 11] atol=big"1e-65" rtol=0
    end
end

@testset "GMRES preconditions the operator and checks the original residual" begin
    A = [4.0 -1 0.5; 1 3 0; 0.5 -1 2]
    b = [2.0, -1, 3]
    tol = Tolerance(atol=1e-13, rtol=1e-12)
    jacobi = v -> v ./ diag(A)
    for (left, right) in ((jacobi, nothing), (nothing, jacobi), (jacobi, jacobi))
        calls = Ref(0)
        apply = v -> (calls[] += 1; A * v)
        x, report = linsolve(apply, b, GMRES(restart=2); tol, left, right)
        @test report.converged
        @test x ≈ A \ b atol=1e-11
        @test report.residual == norm(b - A*x)
        @test report.residual <= tol.atol + tol.rtol*norm(b)
        @test calls[] == report.nfe > report.iterations + 2
    end

    # The Arnoldi estimate accepts the first step, but the true residual
    # requires another cycle even when the full two-vector basis fits.
    A, b = diagm([1.0, 2.0]), ones(2)
    left = v -> [1.0, 1e-4] .* v
    tol = Tolerance(atol=1e-2, rtol=0)
    x, report = linsolve(A, b, GMRES(restart=2, maxiters=1); tol, left)
    @test norm(left(b - A*x)) < tol.atol < report.residual
    @test report.residual == norm(b - A*x)
    check_report(report, :maxiters, 1, 3)
    x, report = linsolve(A, b, GMRES(restart=2); tol, left)
    @test report.converged
    @test report.iterations > 1
    @test report.nfe > report.iterations + 2
    @test x ≈ A \ b atol=tol.atol

    @test_throws ArgumentError linsolve(A, b, DenseSolve(); left)
    @test_throws ArgumentError linsolve(A, b, DenseSolve(); right=identity)
    @test_throws ArgumentError linsolve(zeros(0, 0), Float64[], DenseSolve(); left)
end

@testset "GMRES solves complex systems" begin
    # W1 review F2 reproduced failure on the diagonal case. The nonsymmetric
    # case exercises multiple complex rotations and the one-vector restart.
    for A in (diagm(ComplexF64[2 + 3im, 1 + 0.5im]), ComplexF64[2+im 0.5im; -0.2+im 3-im]),
            b in (ComplexF64[1-im, 2], [1.0, 2.0]), restart in (1, 2)
        x, report = linsolve(A, b, GMRES(; restart))
        @test report.converged
        @test report.residual == norm(b - A*x)
        @test report.residual isa Float64
        @test eltype(x) === ComplexF64
        @test x ≈ A \ b atol=4e-9
    end
end

@testset "GMRES on complex dual numbers (LP-082, Gemini's finding 2)" begin
    # `primal` did not unwrap Complex{Dual}: the scale stayed a Dual and `_bound`
    # threw a MethodError. Tangents must not change the iteration either.
    rng = Xoshiro(82)
    n = 6
    A = Matrix{ComplexF64}(I, n, n) .+ 0.3 .* randn(rng, ComplexF64, n, n)
    b0 = randn(rng, ComplexF64, n)
    db = randn(rng, ComplexF64, n)
    dual(z, t) = Complex(ForwardDiff.Dual{:lp082}(real(z), real(t)), ForwardDiff.Dual{:lp082}(imag(z), imag(t)))
    val(z) = complex(ForwardDiff.value(real(z)), ForwardDiff.value(imag(z)))
    tan(z) = complex(ForwardDiff.partials(real(z), 1), ForwardDiff.partials(imag(z), 1))
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    xp, rp = linsolve(A, b0, GMRES(restart=3, maxiters=60); tol)
    @test primal(dual(1.0 + 2.0im, 3.0 - 1.0im)) === 1.0 + 2.0im
    for scale in (0.0, 1.0, 1e8)
        b = dual.(b0, scale .* db)
        x, report = linsolve(A, b, GMRES(restart=3, maxiters=60); tol)
        @test report.converged
        @test report.residual isa Float64
        @test report.iterations == rp.iterations
        # Complex dual arithmetic rounds differently from ComplexF64 in the last bits.
        @test val.(x) ≈ xp rtol=1e-13
        @test tan.(x) ≈ A \ (scale .* db) rtol=1e-8 atol=1e-8 * scale
    end
end

@testset "GMRES with degenerate preconditioners (LP-082, Gemini's finding 3)" begin
    # Behaviour checked by Mistral's probe 1 of the W5 review; kept as tests.
    rng = Xoshiro(821)
    n = 5
    A = Matrix(1.0I, n, n) .+ 0.2 .* randn(rng, n, n)
    b = randn(rng, n)
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    bound = 1e-12 + 1e-12 * norm(b)
    for (name, kw, reason) in (("left annihilates", (; left = v -> zero(v)), :stalled),
                               ("right annihilates", (; right = v -> zero(v)), :stalled),
                               ("left overflows", (; left = v -> floatmax(Float64) .* 4 .* v), :nonfinite),
                               ("left nonfinite", (; left = v -> fill(NaN, length(v))), :nonfinite))
        x, report = linsolve(A, b, GMRES(restart=3, maxiters=20); tol, kw...)
        @test !report.converged
        @test report.reason === reason
        @test report.residual == norm(b - A * x)
        @test report.iterations <= 20
    end
    # A left preconditioner that makes the preconditioned residual tiny decides nothing:
    # the true residual is still above the bound after the one allowed step.
    x, report = linsolve(A, b, GMRES(restart=1, maxiters=1); tol, left = v -> 1e-20 .* v)
    @test report.residual == norm(b - A * x)
    @test report.converged == (report.residual <= bound)
    @test report.iterations == 1
end

@testset "P37 MINRES against backslash" begin
    rng = Xoshiro(3701)
    for T in (Float32, Float64, ComplexF64), n in (1, 2, 10, 50), indefinite in (false, true)
        R = real(T)
        Q = Matrix(qr(randn(rng, T, n, n)).Q)
        spectrum = collect(range(R(1), R(3); length=max(n, 2)))[1:n]
        indefinite && (spectrum[1:2:end] .*= -one(R))
        A = Matrix(Hermitian(Q * Diagonal(spectrum) * Q'))
        b = randn(rng, T, n)
        rtol = T === Float32 ? 1f-5 : 1e-12
        tol = Tolerance(; atol=0, rtol)
        for op in (A, v -> A * v)
            x, report = linsolve(op, b, MINRES(); tol)
            @test report.converged
            @test eltype(x) === T
            @test report.residual == norm(b - A*x) <= rtol * norm(b)
            @test x ≈ A \ b rtol=4rtol
        end
    end

    A, b = diagm([1.0, 2.0]), ones(2)
    tol = Tolerance(; atol=0, rtol=1e-12)
    @test_throws ArgumentError MINRES(maxiters=-1)
    @test_throws ArgumentError linsolve(A, b, MINRES(); left=identity)
    @test_throws ArgumentError linsolve(A, b, MINRES(); right=identity)
    @test_throws ArgumentError linsolve(zeros(0, 0), Float64[], MINRES(); right=identity)
    x, report = linsolve(zeros(Float32, 0, 0), Float32[], MINRES())
    @test isempty(x)
    check_report(report, :converged, 0, 0)
    x, report = linsolve(A, zeros(2), MINRES(); tol)
    @test x == zeros(2)
    check_report(report, :converged, 0, 0)
    x, report = linsolve(A, b, MINRES(maxiters=0); tol)
    @test x == zeros(2)
    check_report(report, :maxiters, 0, 1)
    @test report.residual == norm(b)

    calls = Ref(0)
    apply = v -> (calls[] += 1; A * v)
    x, report = linsolve(apply, b, MINRES(maxiters=1); tol)
    check_report(report, :maxiters, 1, 3)
    @test x ≈ [3/5, 3/5]
    @test report.nfe == calls[]
    @test report.residual == norm(b - A*x)
    calls[] = 0
    x, report = linsolve(apply, b, MINRES(); tol)
    check_report(report, :converged, 2, 4)
    @test report.nfe == calls[]
    @test report.residual == norm(b - A*x)

    # One Lanczos step gives a zero estimate, but rounding leaves a true
    # residual. With a second step available, MINRES restarts on it.
    scalar, rhs = fill(3.0, 1, 1), [7.0]
    exact_tol = Tolerance(; atol=0, rtol=0)
    x, report = linsolve(scalar, rhs, MINRES(maxiters=1); tol=exact_tol)
    check_report(report, :maxiters, 1, 3)
    @test report.residual == norm(rhs - scalar*x) > 0
    x, report = linsolve(scalar, rhs, MINRES(maxiters=2); tol=exact_tol)
    check_report(report, :converged, 2, 5)
    @test report.residual == norm(rhs - scalar*x) == 0

    A = diagm([2.0, 0.0])
    x, report = linsolve(A, [1.0, 0.0], MINRES(); tol)
    @test report.converged
    @test x ≈ [0.5, 0.0]
    for A in (A, [1.0 4.0; 0.0 2.0])
        x, report = linsolve(A, b, MINRES(maxiters=4); tol)
        @test report.reason in (:stalled, :maxiters)
        @test !report.converged
        @test report.residual == norm(b - A*x) > tol.rtol * norm(b)
    end
    x, report = linsolve(zeros(2, 2), b, MINRES(); tol)
    check_report(report, :stalled, 1, 3)
    @test report.residual == norm(b)
    _, report = linsolve(fill(NaN, 2, 2), b, MINRES(); tol)
    @test report.reason === :nonfinite && isnan(report.residual)
    x, report = linsolve(v -> iszero(norm(v)) ? zero(v) : fill(Inf, 2), b, MINRES(); tol)
    check_report(report, :nonfinite, 1, 2)
    @test x == zeros(2) && report.residual == norm(b)

    # Differentiate both the matrix and the right-hand side through each recurrence.
    for T in (Float64, ComplexF64)
        Q = Matrix(qr(randn(rng, T, 6, 6)).Q)
        A = Matrix(Hermitian(Q * Diagonal([-4.0, -2.0, -1.0, 1.5, 3.0, 5.0]) * Q'))
        W = randn(rng, T, 6, 6)
        dA = (W + W') / 2
        b0, db = randn(rng, T, 6), randn(rng, T, 6)
        function solve(θ, alg)
            x, report = linsolve(A .+ θ .* dA, b0 .+ θ .* db, alg; tol)
            @test report.converged
            @test report.residual isa Float64
            return vcat(real.(x), imag.(x))
        end
        @test ForwardDiff.derivative(θ -> solve(θ, MINRES()), 0.0) ≈
              ForwardDiff.derivative(θ -> solve(θ, DenseSolve()), 0.0) rtol=1e-9
    end

    J = [0.25 0.5; 0.5 -0.5]
    λ, report = adjoint_solve(J', b, MINRES(); tol)
    @test report.converged && λ ≈ J' \ b
    λ, report = fixed_point_adjoint(v -> J' * v, b, MINRES(); tol)
    @test report.converged && λ ≈ (I - J') \ b
end
