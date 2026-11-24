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
            @test gradient_rhs(alg) ≈ rhs_reference
            @test gradient_matrix(alg) ≈ matrix_reference
            @test nested(alg) ≈ nested_reference
            @test complex_tangent(complex_operator, alg) ≈ complex_reference

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

krylov_dual(x::Real, parts) = ForwardDiff.Dual{:p42}(x, parts)
krylov_dual(z::Complex, parts) =
    complex(krylov_dual(real(z), map(real, parts)), krylov_dual(imag(z), map(imag, parts)))
krylov_partial(x::ForwardDiff.Dual, i) = ForwardDiff.partials(x, i)
krylov_partial(z::Complex, i) = complex(krylov_partial(real(z), i), krylov_partial(imag(z), i))

@testset "P42 random dual systems against DenseSolve" begin
    tol = Tolerance(atol=1e-12, rtol=2e-9)
    for method in (:gmres, :minres, :cg), T in (Float64, ComplexF64)
        @testset "$method $T" begin
            rng = Xoshiro(4242)
            for sample in 1:50
                n, N = mod1(sample, 30), rand(rng, 1:4)
                A, b = krylov_oracle_system(rng, T, n, method === :gmres ? :nonsymmetric : :spd)
                dA = ntuple(N) do _
                    W = randn(rng, T, n, n) / sqrt(n)
                    return method === :gmres ? W : (W + W') / 2
                end
                db = ntuple(_ -> randn(rng, T, n), N)
                x, report = linsolve(krylov_dual.(A, tuple.(dA...)),
                    krylov_dual.(b, tuple.(db...)), KrylovJL(method); tol)
                @test report.converged

                Aw, bw = widen.(A), widen.(b)
                dAw, dbw = map(M -> widen.(M), dA), map(v -> widen.(v), db)
                ref, dense = linsolve(krylov_dual.(Aw, tuple.(dAw...)),
                    krylov_dual.(bw, tuple.(dbw...)), DenseSolve(); tol)
                @test dense.converged
                inverse_norm = inv(last(svdvals(A)))
                value, value_ref = widen.(primal.(x)), primal.(ref)
                @test norm(bw - Aw * value) <= tol.atol + tol.rtol * norm(bw)
                for i in 1:N
                    tangent, tangent_ref = widen.(krylov_partial.(x, i)), krylov_partial.(ref, i)
                    rhs = dbw[i] - dAw[i] * value
                    r = norm(rhs - Aw * tangent)
                    rr = norm(dbw[i] - dAw[i] * value_ref - Aw * tangent_ref)
                    # The two tangent solves linearize at different primal iterates.
                    shift = norm(dAw[i] * (value - value_ref))
                    @test r <= tol.atol + tol.rtol * norm(rhs)
                    @test norm(tangent - tangent_ref) <= inverse_norm * (r + rr + shift) * (1 + 1e-6)
                end
            end
        end
    end
end

@testset "P42 callable dual operators, preconditioners and reports" begin
    A, b = [4.0 1.0; 1.0 3.0], [1.0, 2.0]
    db = [2.0, -1.0]
    dA = [1.0 0.5; 0.5 0.0]
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    for method in (:gmres, :minres, :cg)
        alg = KrylovJL(method)
        calls = Ref(0)
        t = krylov_dual(0.0, (1.0,))
        op(v) = (calls[] += 1; A * v .+ t .* (dA * v))
        x, report = linsolve(op, b, alg; tol)
        reference, _ = linsolve(A .+ t .* dA, b, DenseSolve(); tol)
        xp, primal_report = linsolve(A, b, alg; tol)
        _, correction = linsolve(A, -dA * xp, alg; tol)
        check_report(report, :converged, primal_report.iterations + correction.iterations,
            primal_report.nfe + correction.nfe + 2) # discovery probe and dual residual
        @test report.nfe == calls[]
        @test primal.(x) ≈ primal.(reference)
        @test krylov_partial.(x, 1) ≈ krylov_partial.(reference, 1)
        @test_throws ArgumentError("KrylovJL with ldiv=true does not support dual numbers") linsolve(
            op, b, KrylovJL(method; ldiv=true); tol)

        # The captured value also appears in the preconditioners' output.
        left(v) = v ./ (diag(A) .+ t)
        right = method === :gmres ? v -> v .* (0.5 + t) : nothing
        bd = krylov_dual.(b, tuple.(db))
        x, report = linsolve(A, bd, alg; tol, left, right)
        @test report.converged
        @test primal.(x) ≈ A \ b
        @test krylov_partial.(x, 1) ≈ A \ db

        short = KrylovJL(method; maxiters=1)
        xp, primal_report = linsolve(A, b, short; tol)
        x, report = linsolve(A, bd, short; tol)
        check_report(report, :maxiters, primal_report.iterations, primal_report.nfe)
        @test report.residual === primal_report.residual
        @test primal.(x) == xp
        @test iszero(krylov_partial.(x, 1))

        # The operator captures only the inner value's tag. Its contribution to the
        # mixed derivative survives stripping the outer tag from the right-hand side.
        mixed(solver) = ForwardDiff.derivative(0.0) do a
            ForwardDiff.derivative(0.0) do s
                first(linsolve(v -> A * v .+ a .* (dA * v), b .+ s .* db, solver; tol))
            end
        end
        expected = mixed(DenseSolve())
        @test norm(expected) > 0
        @test mixed(alg) ≈ expected
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

@testset "P42 KrylovJL dual solve on JLArrays" begin
    JLArrays.allowscalar(false)
    A, b, db = [4.0 1.0; 1.0 3.0], [1.0, 2.0], [-1.0, 3.0]
    x, report = linsolve(JLArray(A), JLArray(krylov_dual.(b, tuple.(db))), KrylovJL())
    @test x isa JLArray
    @test report.converged
    @test primal.(Array(x)) ≈ A \ b
    @test krylov_partial.(Array(x), 1) ≈ A \ db
end

@testset "LP-087: review of P42" begin
    # F1 (Mistral): a dual carried only by a preconditioner, with A and b plain. The solution does
    # not depend on the preconditioner, so its derivative is zero; it used to raise a MethodError.
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    tol = Tolerance(atol=1e-12, rtol=1e-12)
    for method in (:gmres, :minres, :cg), kind in (:matrix, :function)
        alg = KrylovJL(method)
        value = nothing
        derivative = ForwardDiff.derivative(0.0) do t
            P = Diagonal([1 / 4 + t, 1 / 3])
            left = kind === :matrix ? Matrix(P) : (v -> P * v)
            x, report = linsolve(A, b, alg; tol, left)
            value = (x, report)
            return sum(x)
        end
        x, report = value
        @test report.converged
        @test primal.(x) ≈ A \ b
        @test derivative == 0
    end
    # GMRES also with a dual right preconditioner.
    derivative = ForwardDiff.derivative(0.0) do t
        sum(first(linsolve(A, b, KrylovJL(:gmres); tol, right=Diagonal([1 + t, 1.0]))))
    end
    @test derivative == 0
    # Found beside F1, older than P42: the in-house GMRES with a dual matrix preconditioner and plain
    # A and b raises a MethodError (a function preconditioner works). No row; LP-087.
    @test_broken try
        ForwardDiff.derivative(0.0) do t
            sum(first(linsolve(A, b, GMRES(); tol, left=Matrix(Diagonal([1 / 4 + t, 1 / 3])))))
        end
        true
    catch err
        err isa MethodError || rethrow()
        false
    end
    # With ldiv = true a dual factorisation cannot be stripped: the documented ArgumentError.
    @test_throws ArgumentError ForwardDiff.derivative(0.0) do t
        sum(first(linsolve(A, b, KrylovJL(:cg; ldiv=true); tol,
                           left=cholesky(Symmetric(A + t * I)))))
    end
end
