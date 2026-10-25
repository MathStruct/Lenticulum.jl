@testset "Polynomial basis tables" begin
    expected = [0 1 0 2 1 0; 0 0 1 0 1 2]
    @test ExponentTable(2,2).exponents == expected
    for N in 1:4, d in 0:4
        t = ExponentTable(N,d)
        @test size(t.exponents) == (N,binomial(N+d,d))
        indices = [Tuple(t.exponents[:,i]) for i in axes(t.exponents,2)]
        @test issorted(indices;by=α->(sum(α),(-).(α)))
        @test length(unique(indices)) == length(indices)
        h = ExponentTable(N,d;homogeneous=true)
        @test size(h.exponents,2) == binomial(N+d-1,d)
        @test all(==(d),vec(sum(h.exponents;dims=1)))
    end
    for args in ((0,2),(-1,2),(2,-1),(true,2),(2,true))
        @test_throws ArgumentError ExponentTable(args...)
    end
    @test_throws ArgumentError ExponentTable(2,2;basis=:other)
    t = ExponentTable(2,2)
    @test veronese(t,[2,3]) == [1,2,3,4,6,9]
    @test veronese_jacobian(t,[2,3]) == [0 0;1 0;0 1;4 0;3 2;0 6]
    @test veronese_jacobian(t,[0,0]) == [0 0;1 0;0 1;0 0;0 0;0 0]
    @test eltype(veronese(t,[2,3])) == Int
    @test veronese(t,[1//2,2//3]) == [1,1//2,2//3,1//4,1//3,4//9]
    @test eltype(veronese(t,Float32[1,2])) == Float32
    @test_throws DimensionMismatch veronese(t,[1])
    @test_throws DimensionMismatch veronese_jacobian(t,[1])
    rng = Xoshiro(3001)
    for d in 0:6
        mon,cheb = ExponentTable(3,d),ExponentTable(3,d;basis=:chebyshev)
        C = basis_change(mon,cheb)
        @test rank(Float64.(C)) == size(C,1)
        for x in (2rand(rng,3).-1,[-1.,0,1],[1//3,-2//5,3//4])
            E = cheb.exponents
            @test veronese(cheb,x) ≈ [prod(cos(E[j,a]*acos(x[j])) for j in 1:3) for a in axes(E,2)] atol=2e-14
            @test veronese(cheb,x) ≈ C*veronese(mon,x) atol=2e-13
            x isa Vector{Rational{Int}} && @test veronese(cheb,x) == C*veronese(mon,x)
            for tab in (mon,cheb)
                h = 1e-6
                fd = hcat([(veronese(tab,x+h*I(3)[:,j])-veronese(tab,x-h*I(3)[:,j]))/(2h) for j in 1:3]...)
                @test veronese_jacobian(tab,x) ≈ fd atol=2e-7 rtol=2e-8
            end
        end
    end
    @test_throws ArgumentError basis_change(t,t)
    @test_throws ArgumentError basis_change(t,ExponentTable(2,1;basis=:chebyshev))
    @test_throws ArgumentError basis_change(ExponentTable(2,2;homogeneous=true),ExponentTable(2,2;homogeneous=true,basis=:chebyshev))
end

@testset "PolynomialFactor and Sampson invariance" begin
    rng = Xoshiro(3002)
    tab = ExponentTable(3,2)
    f = PolynomialFactor((a=1,b=2),tab,2)
    ps,st = LuxCore.setup(Xoshiro(3002),f)
    @test ps == LuxCore.initialparameters(Xoshiro(3002),f)
    @test st == (;)
    @test ispoint(Grassmann(2,10),ps.Θ)
    @test LuxCore.parameterlength(f) == 20
    @test LuxCore.statelength(f) == 0
    @test islearnable(f)
    @test isempty(supported_polarities(f))
    @test Tuple(channelname(c) for c in LenticulumCore.channels(f)) == (:a,:b)
    @test Tuple(c.space for c in LenticulumCore.channels(f)) == (1,2)
    @test pointwise_energyspace(f) == EuclideanEnergySpace(2)
    B = Matrix(qr(randn(rng,2,2)).Q)*Diagonal([0.8,1.6])*Matrix(qr(randn(rng,2,2)).Q)'
    cheb = ExponentTable(3,2;basis=:chebyshev)
    C = Float64.(basis_change(tab,cheb))
    Θc = rand(rng,Grassmann(2,10)); Θm = Θc*C
    for _ in 1:20
        z = randn(rng,3)
        @test energy(f,(a=z[1],),nothing,(b=z[2:3],),ps,st) == (ps.Θ*veronese(tab,z),st)
        @test energy(f,nothing,nothing,(a=z[1],b=z[2:3]),ps,st) == energy(f,nothing,nothing,z,ps,st)
        J = energy_jacobian(f,nothing,nothing,z,ps,st;wrt=:y)
        Jfd = invoke(energy_jacobian,Tuple{AbstractLenticulumFactor,Any,Any,Any,Any,Any},f,nothing,nothing,z,ps,st;wrt=:y)
        @test J ≈ Jfd atol=2e-9 rtol=2e-10
        for λ in (0.,0.3), Σ in (I,[2.0 0.2 0;0.2 0.7 0;0 0 1])
            g = PolynomialFactor(f.dims,tab,2;λ,Σ)
            score(Θ) = first(scalar_energy(g,nothing,nothing,z,(;Θ),st))
            @test score(ps.Θ) ≈ score(B*ps.Θ) rtol=1e-10
            if λ == 0
                gc = PolynomialFactor(f.dims,cheb,2;Σ)
                @test score(Θm) ≈ first(scalar_energy(gc,nothing,nothing,z,(Θ=Θc,),st)) rtol=1e-10
            end
        end
    end
    t = ExponentTable(2,2)
    Θ = [-1. 0 0 1 0 0;-2 0 0 0 0 1]
    g = PolynomialFactor((z=2,),t,2)
    @test_throws ArgumentError scalar_energy(g,nothing,nothing,zeros(2),(;Θ),st)
    gd = PolynomialFactor((z=2,),t,2;λ=0.3)
    @test isfinite(first(scalar_energy(gd,nothing,nothing,zeros(2),(;Θ),st)))
    @test first(scalar_energy(gd,nothing,nothing,zeros(2),(;Θ),st)) ≈ first(scalar_energy(gd,nothing,nothing,zeros(2),(Θ=B*Θ,),st)) rtol=1e-10
    for kwargs in ((λ=-1,),(λ=NaN,),(Σ=-I,),(Σ=[1. 2;0 1],))
        @test_throws ArgumentError PolynomialFactor((z=2,),t,1;kwargs...)
    end
    @test_throws DimensionMismatch PolynomialFactor((z=3,),t,1)
    @test_throws DimensionMismatch PolynomialFactor((z=2,),t,1;Σ=I(3))
    @test_throws ArgumentError PolynomialFactor((z=0,),t,1)
    @test_throws ArgumentError PolynomialFactor((z=2,),t,true)
    @test_throws ArgumentError energy(g,(z=zeros(2),),nothing,(z=zeros(2),),(;Θ),st)
    @test_throws ArgumentError energy(g,(other=zeros(2),),nothing,nothing,(;Θ),st)
    @test_throws DimensionMismatch energy(g,(z=[0.],),nothing,nothing,(;Θ),st)
    @test_throws ArgumentError energy_jacobian(g,nothing,nothing,zeros(2),(;Θ),st;wrt=:x)
    @test_throws ArgumentError scalar_energy(g,nothing,nothing,(z=zeros(2),),(;Θ),st)
    @test_throws ArgumentError energy(g,nothing,nothing,ComplexF64[1,2],(;Θ),st)
    # λΘΘ′ is invariant to row gauges, but specifies a basis-dependent
    # coefficient metric. Verify the documented limitation explicitly.
    tc = ExponentTable(2,2;basis=:chebyshev)
    θc = reshape([-1.,0,0,1,0,1],1,:)
    θm = θc*Float64.(basis_change(t,tc))
    z = [0.2,0.4]
    sm = first(scalar_energy(PolynomialFactor((z=2,),t,1;λ=0.3),nothing,nothing,z,(Θ=θm,),st))
    sc = first(scalar_energy(PolynomialFactor((z=2,),tc,1;λ=0.3),nothing,nothing,z,(Θ=θc,),st))
    @test abs(sm-sc)>1e-3
end

# Independent objective oracle, using only public evaluations, Cholesky and scalar sums.
function sampson_oracle(tab,X,Θ;Σ=I,λ=0)
    sum(axes(X,2)) do i
        v,D = veronese(tab,X[:,i]),veronese_jacobian(tab,X[:,i])
        r,J = Θ*v,Θ*D
        dot(r,cholesky(Symmetric(J*Σ*J'+λ*Θ*Θ'))\r)/2
    end
end

@testset "Klein quadric fit_nullspace" begin
    rng = Xoshiro(5106)
    X = reduce(hcat,[plucker(randn(rng,4),randn(rng,4)) for _ in 1:400])
    mon,cheb = ExponentTable(6,2),ExponentTable(6,2;basis=:chebyshev)
    q = only(plucker_relations(4))
    klein = zeros(28)
    for ((i,j),c) in zip(q.monomials,q.coefficients)
        α = zeros(Int,6); α[i]+=1; α[j]+=1
        klein[only(findall(a->mon.exponents[:,a]==α,1:28))] = c
    end
    for tab in (mon,cheb)
        fit = fit_nullspace(tab,X,1;rtol=1e-12)
        @test fit.nullity == 1
        @test fit.numerical_rank == 27
        @test ispoint(Grassmann(1,28),fit.Θ)
        θ = tab.basis === :monomial ? vec(fit.Θ) : vec(fit.Θ*Float64.(basis_change(mon,cheb)))
        @test abs(dot(θ/norm(θ),klein/norm(klein))) ≈ 1 atol=2e-14
        @test norm([dot(θ,veronese(mon,X[:,i])) for i in 1:400]) < 2e-12
        s = fit.singular_values/fit.singular_values[1]
        @info "Klein fit (seed 5106, n=400, rtol=1e-12)" basis=tab.basis smallest=s[end] next=s[end-1] gap=s[end-1]/s[end]
    end
end

@testset "Polynomial fitting and full Sampson gradient" begin
    rng = Xoshiro(3003)
    tab = ExponentTable(2,2)
    # Seeded ellipse: rotate axes (1.3,0.8), translate by a seeded centre.
    φ = rand(rng)*2π
    R = [cos(φ) -sin(φ);sin(φ) cos(φ)]
    A = R*Diagonal([1.3,0.8]); c = 0.2randn(rng,2)
    angles = 2π*rand(rng,200)
    X = c .+ A*vcat(cos.(angles)',sin.(angles)')
    Q = inv(A*A')
    truth = [dot(c,Q*c)-1; -2Q*c; Q[1,1];2Q[1,2];Q[2,2]]
    fit = fit_nullspace(tab,X,1)
    @test fit.nullity == 1
    @test abs(dot(vec(fit.Θ),truth/norm(truth))) ≈ 1 atol=2e-14
    noisy = X+0.04randn(rng,size(X))
    init = fit_nullspace(tab,noisy,1).Θ
    initialcost = sampson_oracle(tab,noisy,init)
    result = fit_sampson(tab,noisy,1;tol=2e-6,maxiter=20000)
    @test result.converged
    @test result.gradient_norm <= 2e-6
    @test result.cost ≈ sampson_oracle(tab,noisy,result.Θ) rtol=2e-13
    @test result.cost < initialcost
    fns = fit_sampson(tab,noisy,1;method=:fns,tol=2e-6,maxiter=1000)
    @test fns.converged
    @test fns.cost ≈ result.cost atol=1e-11
    @test abs(dot(vec(fns.Θ),vec(result.Θ))) ≈ 1 atol=1e-10
    @info "Conic Sampson (seed 3003, n=200, noise sigma=0.04)" initialcost cost=result.cost gradient=result.gradient_norm iterations=result.iterations fns_cost=fns.cost fns_gradient=fns.gradient_norm fns_iterations=fns.iterations
    for _ in 1:25
        Δ = project_tangent(Grassmann(1,6),result.Θ,randn(rng,1,6)); Δ/=norm(Δ)
        for h in (-1e-4,1e-4)
            @test sampson_oracle(tab,noisy,retract(Grassmann(1,6),result.Θ,h*Δ)) >= result.cost-1e-11
        end
    end
    # Central differences on retracted curves for k=1 AND k=2, anisotropic Σ,
    # with and without damping; independent objective includes metric derivatives.
    for k in (1,2), λ in (0.,0.2)
        M = Grassmann(k,6); Θ = rand(rng,M)
        Σ = [1.4 0.2;0.2 0.8]
        data = AlgebraicFactors._sampson_data(tab,noisy,Σ,λ)
        cost,G = AlgebraicFactors._sampson_cost_gradient(Θ,data)
        @test cost ≈ sampson_oracle(tab,noisy,Θ;Σ,λ) rtol=2e-12
        grad = project_tangent(M,Θ,G)
        for _ in 1:5
            Δ = project_tangent(M,Θ,randn(rng,k,6)); Δ/=norm(Δ)
            h = 1e-6
            fd = (sampson_oracle(tab,noisy,retract(M,Θ,h*Δ);Σ,λ)-sampson_oracle(tab,noisy,retract(M,Θ,-h*Δ);Σ,λ))/(2h)
            @test dot(grad,Δ) ≈ fd rtol=2e-5 atol=1e-4
        end
    end
    # Two quadrics in R³: z=x², y²+x²=1. Degree-two vanishing space is exactly 2D.
    t3 = ExponentTable(3,2)
    θ = 2π*rand(rng,200)
    curve = vcat(cos.(θ)',sin.(θ)',(cos.(θ).^2)')
    fitted = fit_nullspace(t3,curve,2)
    expected = [ex[3]==1 && sum(ex)==1 ? 1. : ex==[2,0,0] ? -1. : 0. for ex in eachcol(t3.exponents)]
    expected2 = [sum(ex)==0 ? -1. : ex==[2,0,0] || ex==[0,2,0] ? 1. : 0. for ex in eachcol(t3.exponents)]
    rows = vcat(expected',expected2')
    target = retract(Grassmann(2,10),rows,zero(rows))
    @test fitted.nullity == 2
    @test fitted.Θ'*fitted.Θ ≈ target'*target atol=2e-13
    refined = fit_sampson(t3,curve,2;tol=1e-8)
    @test refined.converged
    @test refined.cost < 1e-24
    @test ispoint(Grassmann(2,10),refined.Θ)
    # Underdetermined V must include the unreturned structural zeros of a thin SVD.
    tiny = fit_nullspace(tab,zeros(2,1),3)
    @test length(tiny.singular_values)==6 && tiny.nullity==5
    @test ispoint(Grassmann(3,6),tiny.Θ)
    @test fit_nullspace(ExponentTable(1,1;homogeneous=true),zeros(1,3),1).nullity == 1
    @test !fit_sampson(tab,noisy,1;maxiter=0).converged
    for kwargs in ((method=:other,),(tol=0.,),(tol=NaN,),(maxiter=-1,),(λ=-1.,),(Θ0=zeros(1,6),))
        @test_throws ArgumentError fit_sampson(tab,noisy,1;kwargs...)
    end
    @test_throws ArgumentError fit_sampson(tab,noisy,2;method=:fns)
    @test_throws ArgumentError fit_nullspace(tab,noisy,0)
    @test_throws ArgumentError fit_nullspace(tab,noisy,1;rtol=-1)
    @test_throws ArgumentError fit_nullspace(tab,zeros(2,0),1)
    @test_throws DimensionMismatch fit_nullspace(tab,zeros(3,10),1)
    @test_throws ArgumentError fit_nullspace(tab,fill(NaN,2,10),1)
    singular = reshape([-1.,0,0,1,0,1],1,:); singular/=norm(singular)
    err = try fit_sampson(tab,zeros(2,1),1;Θ0=singular); nothing catch e; e end
    @test err isa ArgumentError && occursin("point 1",sprint(showerror,err))
    @test isfinite(fit_sampson(tab,zeros(2,1),1;Θ0=singular,λ=0.1,maxiter=0).cost)
end
