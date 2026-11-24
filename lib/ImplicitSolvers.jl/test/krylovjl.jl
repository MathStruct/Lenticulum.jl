function krylovjl_check(A, b, reference, x, report, tol)
    @test report.converged
    @test eltype(x) === eltype(b)
    @test report.residual == norm(b - A * x)
    @test report.residual <= tol.atol + tol.rtol * norm(b)
    xwide = widen.(x)
    residual = norm(reference.b - reference.A * xwide)
    @test residual <= tol.atol + tol.rtol * norm(reference.b)
    @test norm(xwide - reference.x) <=
        reference.inverse_norm * (residual + reference.residual) * (1 + 1e-6)
end

function krylovjl_reference(A, b)
    Aw, bw = widen.(A), widen.(b)
    x, report = linsolve(Aw, bw, DenseSolve())
    @test report.converged
    inverse_norm = inv(last(svdvals(Matrix{eltype(A) <: Real ? Float64 : ComplexF64}(A))))
    return (; A=Aw, b=bw, x, inverse_norm, residual=norm(bw - Aw * x))
end

@testset "P41 GMRES backends against DenseSolve" begin
    rng = Xoshiro(1405)
    for T in (Float32, Float64, ComplexF64), family in (:nonsymmetric, :spd, :ill_conditioned),
            n in (1, 2, 10, 50, 200)
        A, b = krylov_oracle_system(rng, T, n, family)
        reference = krylovjl_reference(A, b)
        tol = Tolerance(; atol=0, rtol=T === Float32 ? 2f-5 : 2e-9)
        jacobi = inv(Diagonal(diag(A)))
        action = v -> jacobi * v
        linop = v -> A * v
        memories = n == 1 ? (1,) : (min(n - 1, 12), n)
        for memory in memories, op in (A, linop), (use_left, use_right) in
                ((false, false), (true, false), (false, true), (true, true))
            @testset "$T $family n=$n memory=$memory matrix=$(op === A) left=$use_left right=$use_right" begin
                maxiters = 4n + 20
                x, shared = linsolve(op, b, GMRES(; restart=memory, maxiters);
                    tol, left=use_left ? action : nothing, right=use_right ? action : nothing)
                y, backend = linsolve(op, b, KrylovJL(:gmres; memory, maxiters);
                    tol, left=use_left ? jacobi : nothing, right=use_right ? action : nothing)
                @test shared.converged == backend.converged
                krylovjl_check(A, b, reference, x, shared, tol)
                krylovjl_check(A, b, reference, y, backend, tol)
            end
        end
    end
end

@testset "P41 MINRES and CG against DenseSolve" begin
    rng = Xoshiro(3702)
    for T in (Float32, Float64, ComplexF64), n in (1, 2, 10, 50, 200),
            family in (:spd, :indefinite)
        if family === :spd
            A, b = krylov_oracle_system(rng, T, n, family)
        else
            R = real(T)
            Q = Matrix(qr(randn(rng, T, n, n)).Q)
            spectrum = collect(range(R(0.8), one(R); length=max(n, 2)))[1:n]
            spectrum[1:2:end] .*= -one(R)
            A = Matrix(Hermitian(Q * Diagonal(spectrum) * Q'))
            b = randn(rng, T, n)
        end
        reference = krylovjl_reference(A, b)
        tol = Tolerance(; atol=0, rtol=T === Float32 ? 2f-5 : 2e-9)
        jacobi = inv(Diagonal(abs.(real.(diag(A)))))
        methods = family === :spd ? (:minres, :cg) : (:minres,)
        for op in (A, v -> A*v)
            maxiters = 4n + 20
            x, shared = linsolve(op, b, MINRES(; maxiters); tol)
            krylovjl_check(A, b, reference, x, shared, tol)
            for method in methods, left in (nothing, v -> jacobi * v)
                @testset "$method $T $family n=$n matrix=$(op === A) preconditioned=$(left !== nothing)" begin
                    y, backend = linsolve(op, b, KrylovJL(method; maxiters); tol, left)
                    @test shared.converged == backend.converged
                    krylovjl_check(A, b, reference, y, backend, tol)
                end
            end
        end
    end
end

@testset "P41 true residual and work accounting" begin
    # LP-082's two-dimensional failure with a nonuniform left Jacobi operator.
    A = [1.0 0.5; 0.5 100.0]
    b = ones(2)
    M = inv(Diagonal(diag(A)))
    tol = Tolerance(atol=0.01, rtol=0)
    raw, stats = Krylov.gmres(A, b; M, atol=tol.atol, rtol=0.0, memory=2, restart=true)
    @test stats.solved
    @test norm(M * (b - A * raw)) <= tol.atol < norm(b - A * raw)
    calls = Ref(0)
    op = v -> (calls[] += 1; A * v)
    x, report = linsolve(op, b, KrylovJL(:gmres; memory=2); tol, left=M)
    @test report.converged
    @test report.residual == norm(b - A * x) <= tol.atol
    @test report.iterations > stats.niter
    @test report.nfe == calls[] > report.iterations + 2

    # A small absolute estimate can also stop the very first attempt at x = 0.
    x, report = linsolve(A, b, KrylovJL(); tol, left=v -> 1e-8 .* v)
    @test report.reason === :stalled
    @test !report.converged
    @test report.residual == norm(b - A * x) > tol.atol

    for method in (:gmres, :minres, :cg)
        alg = KrylovJL(method)
        exact = Tolerance(atol=1e-12, rtol=0)
        for budget in (0, 1)
            calls[] = 0
            x, report = linsolve(op, b, KrylovJL(method; maxiters=budget); tol=exact)
            check_report(report, :maxiters, budget, calls[])
            @test report.residual == norm(b - A * x)
        end
        calls[] = 0
        x, report = linsolve(op, zero(b), alg)
        check_report(report, :converged, 0, calls[])
        @test iszero(x) && iszero(report.residual)
        for (bad_A, bad_b) in ((fill(NaN, 2, 2), b), (A, [NaN, 1.0]))
            _, report = linsolve(bad_A, bad_b, alg)
            @test report.reason === :nonfinite && isnan(report.residual)
        end
    end
end

@testset "P41 preconditioner protocols and errors" begin
    A = [3.0 1.0; 1.0 2.0]
    b = [1.0, 2.0]
    P = cholesky(Symmetric([4.0 1.0; 1.0 3.0]))
    tol = Tolerance(atol=1e-12, rtol=0)
    @test KrylovJL().method === :gmres
    @test_throws ArgumentError KrylovJL(:other)
    @test_throws ArgumentError KrylovJL(memory=0)
    @test_throws ArgumentError KrylovJL(maxiters=-1)
    for method in (:gmres, :minres, :cg)
        if method !== :gmres
            @test_throws ArgumentError linsolve(A, b, KrylovJL(method); right=identity)
        end
        sides = method === :gmres ? ((P, nothing), (nothing, P), (P, P)) : ((P, nothing),)
        for (left, right) in sides
            x, report = linsolve(A, b, KrylovJL(method; ldiv=true); tol, left, right)
            @test report.converged
            @test x ≈ A \ b
        end
        bd = ForwardDiff.Dual{:p41}.(b, ones(2))
        @test_throws ArgumentError("KrylovJL with ldiv=true does not support dual numbers") linsolve(
            A, bd, KrylovJL(method; ldiv=true); left=P)
    end

    # The subprocess uses the package project and loads only ImplicitSolvers.
    code = """
        using ImplicitSolvers, Test
        @test Base.get_extension(ImplicitSolvers, :ImplicitSolversKrylovExt) === nothing
        @test_throws ErrorException("KrylovJL needs Krylov.jl: run \`using Krylov\`") linsolve(
            identity, [1.0], KrylovJL())
        """
    project = dirname(@__DIR__)
    @test success(run(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`))
end

@testset "LP-086: review of P41" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    # F1: a nonfinite product during CG is a :nonfinite report, as for GMRES and MINRES;
    # Krylov.jl's cg throws on it (its curvature check sees NaN).
    for bad in (NaN, Inf), method in (:gmres, :minres, :cg)
        k = Ref(0)
        op = v -> (k[] += 1; k[] > 2 ? fill(bad, 2) : A * v)
        x, report = linsolve(op, b, KrylovJL(method; maxiters=50))
        @test report.reason === :nonfinite
        @test report.nfe == k[]
    end
    # A finite preconditioner that is not positive definite is the caller's error.
    @test_throws ArgumentError linsolve(A, b, KrylovJL(:cg); left=-1.0 * I(2))
    # F2: a UniformScaling preconditioner, with and without ldiv, and under the tangent correction.
    tol = Tolerance(atol=1e-12, rtol=0)
    for method in (:gmres, :minres, :cg), ldiv in (false, true), P in (I, 2.0I)
        x, report = linsolve(A, b, KrylovJL(method; ldiv); tol, left=P)
        @test report.converged && x ≈ A \ b
    end
    bd = ForwardDiff.Dual{:lp086}.(zeros(2), b)
    x, report = linsolve(A, bd, KrylovJL(:gmres); tol, left=I, right=2.0I)
    @test report.converged
    @test ForwardDiff.partials.(x, 1) ≈ A \ b
end

@testset "P41 integer and BigFloat systems" begin
    for T in (Int, BigFloat), method in (:gmres, :minres, :cg)
        A, b = T[4 1; 1 3], T[2, 1]
        atol = T === BigFloat ? big"1e-65" : 1e-12
        x, report = linsolve(A, b, KrylovJL(method); tol=Tolerance(; atol, rtol=0))
        @test eltype(x) === float(T)
        @test report.converged && report.residual <= atol
        @test x ≈ float(T)[5//11, 2//11] atol=atol rtol=0
    end
end

@testset "P41 ForwardDiff" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    gradient_rhs(alg) = ForwardDiff.gradient(v -> sum(first(linsolve(A, v, alg; tol))), b)
    gradient_matrix(alg) = ForwardDiff.gradient(
        v -> sum(first(linsolve(A + Diagonal(v), b, alg; tol))), zeros(2))
    nested(alg) = ForwardDiff.derivative(0.0) do t
        ForwardDiff.derivative(0.0) do s
            sum(first(linsolve(A + (t * s) * I, b, alg; tol)))
        end
    end

    # LP-082's complex fixture for restarted GMRES.
    rng = Xoshiro(82)
    C = Matrix{ComplexF64}(I, 6, 6) .+ 0.3 .* randn(rng, ComplexF64, 6, 6)
    c, dc = randn(rng, ComplexF64, 6), randn(rng, ComplexF64, 6)
    complex_tangent(A, alg) = ForwardDiff.derivative(0.0) do t
        x, _ = linsolve(A, c .+ t .* dc, alg; tol)
        return vcat(real.(x), imag.(x))
    end
    H = Matrix(Hermitian(C' * C + I))

    for method in (:gmres, :minres, :cg)
        @testset "$method" begin
            alg = KrylovJL(method; memory=3)
            dense = DenseSolve()
            complex_operator = method === :gmres ? C : H
            rhs_reference = gradient_rhs(dense)
            matrix_reference = gradient_matrix(dense)
            nested_reference = nested(dense)
            complex_reference = complex_tangent(complex_operator, dense)
            # Krylov 0.10 raises MethodError for a Dual right-hand side (AbstractFloat bound).
            @test_broken gradient_rhs(alg) ≈ rhs_reference
            # Krylov 0.10 raises MethodError when a dual matrix promotes the RHS to Dual.
            @test_broken gradient_matrix(alg) ≈ matrix_reference
            # Krylov 0.10 raises MethodError for nested Dual workspace elements.
            @test_broken nested(alg) ≈ nested_reference
            # Krylov 0.10 raises MethodError for workspace elements Complex{Dual}.
            @test_broken complex_tangent(complex_operator, alg) ≈ complex_reference

            # When the initial primal residual is zero, P38 corrects the tangent on a plain RHS.
            bd = ForwardDiff.Dual{:p41}.(zero(b), b)
            calls = Ref(0)
            op = v -> (calls[] += 1; A * v)
            left = inv(Diagonal(diag(A)))
            x, report = linsolve(op, bd, alg; tol, left)
            @test report.converged
            @test report.residual === 0.0
            @test report.nfe == calls[]
            @test primal.(x) == zero(b)
            @test ForwardDiff.partials.(x, 1) ≈ A \ b
        end
    end
end

@testset "P41 Krylov.jl on JLArrays" begin
    JLArrays.allowscalar(false)
    rng = Xoshiro(4102)
    for T in (Float32, Float64, ComplexF64), n in (2, 50), method in (:gmres, :minres, :cg)
        A, b = krylov_oracle_system(rng, T, n, :spd)
        dA, db = JLArray(A), JLArray(b)
        tol = Tolerance(atol=T === Float32 ? 1f-5 : 1e-9, rtol=0)
        xref, host = linsolve(A, b, KrylovJL(method); tol)
        @test host.converged
        for op in (dA, v -> dA * v)
            x, report = linsolve(op, db, KrylovJL(method); tol)
            @test x isa JLArray{T,1}
            @test report.converged
            @test Array(x) ≈ xref atol=2tol.atol rtol=0
            @test report.residual == norm(db - dA*x) <= tol.atol
        end
    end
end
