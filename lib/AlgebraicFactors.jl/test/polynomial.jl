include("legacy_sampson.jl")

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
    @test length(supported_polarities(f)) == length(f.dims)
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
    # Defaults and Symbol aliases preserve every field of the original solver.
    for (symbol,method,limit) in ((:riemannian,RiemannianGradient(),20000),(:fns,FNS(),1000))
        old = legacy_fit_sampson(tab,noisy,1;method=symbol,tol=2e-6,maxiter=limit)
        @test fit_sampson(tab,noisy,1;method=symbol,tol=2e-6,maxiter=limit) == old
        @test fit_sampson(tab,noisy,1;method,tol=2e-6,maxiter=limit) == old
    end
    result = fit_sampson(tab,noisy,1;method=RiemannianTrustRegion(),tol=1e-7,maxiter=20)
    @test result.converged
    @test result.gradient_norm <= 1e-7
    @test result.iterations <= 6
    @test fit_sampson(tab,noisy,1;method=:trust_region,tol=1e-7,maxiter=20) == result
    @test result.cost ≈ sampson_oracle(tab,noisy,result.Θ) rtol=2e-13
    @test result.cost < initialcost
    fns = fit_sampson(tab,noisy,1;method=:fns,tol=1e-7,maxiter=1000)
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
    refined = fit_sampson(t3,curve,2;method=:riemannian,tol=1e-8)
    for symbol in (:riemannian,:fns)
        if symbol === :fns
            @test fit_sampson(tab,X,1;method=symbol) == legacy_fit_sampson(tab,X,1;method=symbol)
        else
            @test refined == legacy_fit_sampson(t3,curve,2;tol=1e-8)
            @test fit_sampson(tab,X,1;method=:riemannian) == legacy_fit_sampson(tab,X,1)
            @test fit_sampson(tab,X,1) == fit_sampson(tab,X,1;method=RiemannianTrustRegion())
        end
    end
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

# Load the actual linear templates, without adding a test/package dependency on
# Lenticulum (or its unrelated SpecialFunctions dependency). No copied oracle code.
module PolynomialLinearOracle
using LinearAlgebra, Random, LuxCore
using LenticulumCore: LenticulumCore, GaussianBelief, Gaussian, uninformative,
    belief_mean, belief_cov, isproper, logpartition, dimension, _chol
using Mycelium: Mycelium
include(joinpath(@__DIR__, "../../../src/gaussian.jl"))
include(joinpath(@__DIR__, "../../../src/constraint.jl"))
end

@testset "Polynomial degree-one messages and lens contract" begin
    rng = Xoshiro(3004)
    dims = (a=1,b=2,c=1); names = keys(dims)
    ranges = (a=1:1,b=2:3,c=4:4)
    errors_h, errors_Λ = Float64[], Float64[]
    for k in 1:3, λ in (0.,0.2)
        # v₁(z) = [1; z]; Θ = [-c B_a B_b B_c].
        Θ = randn(rng,k,5)
        J = Θ[:,2:end]
        S = [0.4 0.1 0 0;0.1 0.3 0 0;0 0 0.2 0;0 0 0 0.5]
        f = PolynomialFactor(dims,ExponentTable(4,1),k;Σ=S,λ,iterations=4)
        # Constraint Q is the factor-only residual covariance. Its own input
        # propagation then adds Σ_o J_o C_o J_o', producing exactly our A.
        Q = J*S*J'+λ*Θ*Θ'
        linear = PolynomialLinearOracle.LinearConstraintFactor(dims,k;noise=Q)
        coeff = NamedTuple{names}(map(n->J[:,ranges[n]],names))
        linear_ps = (A=coeff,c=-Θ[:,1])
        @test length(supported_polarities(f)) == 3
        for (t,p) in zip(names,supported_polarities(f)), mask in 0:7
            @test supports_polarity(f,p)
            @test only(unobserved_channels(p)) == t
            incoming = NamedTuple{names}(map(enumerate(names)) do (i,n)
                μ = randn(rng,dims[n]); C = Matrix(0.1I,dims[n],dims[n])
                iszero(mask & (1<<(i-1))) ? DiracBelief(μ) : Gaussian(μ,C)
            end |> Tuple)
            π = Gaussian(randn(rng,dims[t]),Matrix(0.7I,dims[t],dims[t]))
            site,st = factor_message(f,t,p,incoming,π,(;Θ),(;))
            exact,_ = factor_message(linear,t,p,incoming,π,linear_ps,(;))
            push!(errors_h,norm(site.η-exact.η,Inf))
            push!(errors_Λ,norm(site.Λ-exact.Λ,Inf))
            @test site.η ≈ exact.η atol=3e-13 rtol=3e-13
            @test site.Λ ≈ exact.Λ atol=3e-13 rtol=3e-13
            @test st === (;)
            @test rank(site.Λ;atol=1e-11) == min(k,dims[t])
            lens,_ = assemble(f,p,(;Θ),(;))
            @test lens isa BayesianLens{<:PolynomialModel,<:SolverInversion}
            via_lens,_ = likelihood_message(lens,π,incoming,(;Θ),(;))
            posterior,_ = invert(lens,π,incoming,(;Θ),(;))
            @test belief_distance(via_lens,site) < 1e-12
            @test belief_distance(posterior,combine(π,site)) < 1e-12
            @test belief_mean(posterior) ≈ belief_mean(combine(π,exact)) atol=2e-12
            # Target input ignored, including unsupported target belief.
            replaced = merge(incoming,NamedTuple{(t,)}((SampleBelief([[99.]]),)))
            @test belief_distance(first(factor_message(f,t,p,replaced,π,(;Θ),(;))),site) < 1e-12
            @test belief_distance(first(factor_message(f,t,p,incoming,nothing,(;Θ),(;))),site) < 2e-12
        end
    end
    @info "Polynomial degree-one agreement (seed 3004, 144 cases)" max_information_error=maximum(errors_h) max_precision_error=maximum(errors_Λ)
end

# Scalar circle oracle: derivatives written independently, with a fixed scalar A.
# This is the frozen-metric objective, NOT the varying-metric Sampson cost.
circle_objective(y,x,A,μ,P) = (x*x+y*y-1)^2/(2A)+(y-μ)^2/(2P)
function circle_newton(x,A,μ,P; start=0.8)
    y = start
    for _ in 1:100
        g = 2y*(x*x+y*y-1)/A+(y-μ)/P
        H = (2(x*x+y*y-1)+4y*y)/A+1/P
        step = g/H
        y -= step
        abs(step)<1e-15 && break
    end
    return y
end

@testset "Polynomial circle single-branch posterior" begin
    Θ = reshape([-1.,0,0,1,0,1],1,:); tab = ExponentTable(2,2)
    x,μ,P,δ = 0.6,0.9,0.16,0.0025
    f = PolynomialFactor((x=1,y=1),tab,1;Σ=δ*I,iterations=60)
    p = last(supported_polarities(f)); π = Gaussian(μ,P)
    lens,_ = assemble(f,p,(;Θ),(;))
    site,_ = likelihood_message(lens,π,(x=DiracBelief(x),),(;Θ),(;))
    posterior,_ = invert(lens,π,(x=DiracBelief(x),),(;Θ),(;))
    y = only(belief_mean(posterior))
    A = 4δ*(x*x+y*y)
    yref = circle_newton(x,A,μ,P)
    grid = range(0.5,1.1;length=120001)
    ygrid = grid[argmin([circle_objective(t,x,A,μ,P) for t in grid])]
    @test y ≈ yref atol=2e-12
    @test abs(y-ygrid) <= 2.5e-6
    @test site.Λ[1] ≈ 4y*y/A rtol=2e-12
    @test site.η[1] ≈ site.Λ[1]*y-2y*(x*x+y*y-1)/A rtol=2e-12
    @test belief_cov(posterior)[1] ≈ inv(4y*y/A+1/P) rtol=2e-12
    @test belief_distance(first(factor_message(f,:y,p,(x=DiracBelief(x),),π,(;Θ),(;))),site) < 1e-12
    # Same mean, different cavity precision: one-step site excludes the cavity.
    f1 = PolynomialFactor(f.dims,tab,1;Σ=δ*I)
    s1 = first(factor_message(f1,:y,p,(x=DiracBelief(x),),π,(;Θ),(;)))
    s2 = first(factor_message(f1,:y,p,(x=DiracBelief(x),),Gaussian(μ,10P),(;Θ),(;)))
    @test s1.η ≈ s2.η rtol=2e-15
    @test s1.Λ ≈ s2.Λ rtol=2e-15
    @test abs(only(belief_mean(combine(π,s1)))-yref) > 1e-3
    @info "Circle frozen-metric posterior" y yref ygrid A site_precision=site.Λ[1] posterior_variance=belief_cov(posterior)[1]
    errors = Float64[]
    for noise in (1e-2,1e-4,1e-6,1e-8)
        fn = PolynomialFactor(f.dims,tab,1;Σ=noise*I,iterations=60)
        sn = first(factor_message(fn,:y,p,(x=DiracBelief(x),),π,(;Θ),(;)))
        yn = only(belief_mean(combine(π,sn)))
        @test yn > 0
        push!(errors,abs(yn-sqrt(1-x*x)))
    end
    @test issorted(errors;rev=true)
    @test last(errors)<1e-8
    @info "Circle upper-branch zero-noise errors" errors
    precisions = Float64[]
    for y0 in (1e-1,1e-3,1e-5,0.)
        x0 = sqrt(1-y0*y0)
        s = first(factor_message(f1,:y,p,(x=DiracBelief(x0),),DiracBelief(y0),(;Θ),(;)))
        @test all(isfinite,s.Λ) && all(isfinite,s.η)
        @test s.Λ[1] ≈ y0*y0/δ atol=1e-16 rtol=2e-15
        push!(precisions,s.Λ[1])
    end
    @test issorted(precisions;rev=true) && last(precisions)==0
    # Full J is nonzero at tangency, although J_y is zero. No NaN or inverse of Λ.
    flat = first(factor_message(f,:y,p,(x=DiracBelief(1.),),nothing,(;Θ),(;)))
    @test isuninformative(flat)
    @test !isproper(flat)
    @test first(invert(lens,Gaussian(0.,P),(x=DiracBelief(1.),),(;Θ),(;))).Λ == Gaussian(0.,P).Λ
    @info "Circle tangency precisions" precisions
end

@testset "Polynomial rank cases and seeded posterior descent" begin
    tab = ExponentTable(3,2)
    Θ = reshape([sum(α)==0 ? -1. : α==[1,0,0] || α==[0,2,0] || α==[0,0,2] ? 1. : 0. for α in eachcol(tab.exponents)],1,:)
    f = PolynomialFactor((o=1,u=2),tab,1;Σ=0.02I,iterations=1)
    p = last(supported_polarities(f)); π = Gaussian([0.3,0.4],Matrix(0.1I,2,2))
    s = first(factor_message(f,:u,p,(o=DiracBelief(0.2),),π,(;Θ),(;)))
    Ju = [0.6 0.8]; tangent = [-0.8,0.6]
    @test rank(s.Λ;atol=1e-12) == 1
    @test norm(Ju*tangent)<1e-15
    @test norm(s.Λ*tangent)<1e-13
    @test abs(dot(s.η,tangent))<1e-13
    @test isproper(combine(π,s))
    # With no cavity and an affine rank-one target, iteration stops at the
    # improper site instead of attempting a pseudoinverse posterior mode.
    affine = PolynomialFactor((u=2,),ExponentTable(2,1),1;iterations=10)
    sa = first(factor_message(affine,:u,only(supported_polarities(affine)),(;),nothing,
                             (Θ=[-1. 1 2],),(;)))
    @test rank(sa.Λ;atol=1e-12)==1
    @test sa.η ≈ [0.2,0.4]
    # A constant metric makes a single objective meaningful across iterations.
    # This is a finite seeded descent check; unsafeguarded GN is not monotone in general.
    rng = Xoshiro(3005); θ = reshape([-1.,0,0,1,0,1],1,:)
    improvements = Float64[]
    for _ in 1:20
        x = 0.2+0.5rand(rng); μ = 0.65+0.5rand(rng); P = 0.1+0.2rand(rng)
        ys = Float64[]
        for it in (1,30)
            g = PolynomialFactor((x=1,y=1),ExponentTable(2,2),1;Σ=zeros(2,2),λ=0.01,iterations=it)
            prior = Gaussian(μ,P)
            site = first(factor_message(g,:y,last(supported_polarities(g)),(x=DiracBelief(x),),prior,(Θ=θ,),(;)))
            push!(ys,only(belief_mean(combine(prior,site))))
        end
        F(y) = circle_objective(y,x,0.03,μ,P)
        push!(improvements,F(ys[1])-F(ys[2]))
        @test F(ys[2])<=F(ys[1])+1e-13
        @test ys[2] ≈ circle_newton(x,0.03,μ,P;start=ys[2]) atol=1e-10
    end
    @info "Polynomial seeded descent (3005, 20 cases)" minimum_improvement=minimum(improvements)
end

@testset "Polynomial Mycelium graph and independent joint Laplace" begin
    # A genuinely quadratic relation a*b+c=0. Linearise at E[a],E[b]; the
    # independent joint objective's MAP has a*b+c exactly zero when c is free.
    # Joint Laplace marginal variance differs from input propagation by no
    # second-order product term, so these Gaussian/GN covariances agree here.
    tab = ExponentTable(3,2)
    Θ = reshape([α==[1,1,0] || α==[0,0,1] ? 1. : 0. for α in eachcol(tab.exponents)],1,:)
    μa,μb,Va,Vb,Q = 0.7,-0.4,0.03,0.02,0.01
    f = PolynomialFactor((a=1,b=1,c=1),tab,1;Σ=zeros(3,3),λ=Q/2,iterations=8)
    builder = GraphBuilder()
    for n in (:a,:b,:c); variable!(builder,n,1); end
    factor!(builder,:pa,PriorFactor(:a,Gaussian(μa,Va),x->0.0))
    factor!(builder,:pb,PriorFactor(:b,Gaussian(μb,Vb),x->0.0))
    factor!(builder,:poly,f)
    connect!(builder,:pa,:a,:a;direction=Emitting())
    connect!(builder,:pb,:b,:b;direction=Emitting())
    connect!(builder,:poly,:a,:a;direction=Absorbing())
    connect!(builder,:poly,:b,:b;direction=Absorbing())
    connect!(builder,:poly,:c,:c;direction=Emitting())
    graph = build(builder)
    ps = (pa=(;),pb=(;),poly=(;Θ)); st = (pa=(;),pb=(;),poly=(;))
    beliefs,report,_,_ = infer!(graph,SequentialSchedule(tasks(forward_backward_schedule(graph))),ps,st;tol=1e-11)
    @test report.converged
    # Independently differentiated joint objective:
    # (a*b+c)^2/(2Q)+(a-μa)^2/(2Va)+(b-μb)^2/(2Vb).
    map_joint = [μa,μb,-μa*μb]
    joint_objective(z) = (z[1]*z[2]+z[3])^2/(2Q)+(z[1]-μa)^2/(2Va)+(z[2]-μb)^2/(2Vb)
    @test joint_objective(map_joint)==0
    gradient_r = [map_joint[2],map_joint[1],1.]
    H = gradient_r*gradient_r'/Q+Diagonal([1/Va,1/Vb,0.])
    laplace_cov = inv(H)
    @test only(belief_mean(beliefs.c)) ≈ map_joint[3] atol=2e-12
    @test only(belief_cov(beliefs.c)) ≈ laplace_cov[3,3] atol=2e-12
    @info "Polynomial graph joint Laplace" report mean=belief_mean(beliefs.c) variance=belief_cov(beliefs.c)[1] laplace_variance=laplace_cov[3,3]
end

@testset "Polynomial invalid messages and conformance scope" begin
    tab = ExponentTable(2,2); Θ = reshape([-1.,0,0,1,0,1],1,:)
    f = PolynomialFactor((x=1,y=1),tab,1); p = last(supported_polarities(f))
    site(inputs,prior=Gaussian(0.8,0.1);g=f,pol=p,θ=Θ,t=:y) = first(factor_message(g,t,pol,inputs,prior,(Θ=θ,),(;)))
    for b in (TrivialBelief(),nothing,uninformative(1),SampleBelief([[0.6]]),CategoricalBelief([0.2,0.8]),DiracBelief(1+im))
        @test_throws ArgumentError site((x=b,))
    end
    @test_throws ArgumentError site((;))
    @test_throws DimensionMismatch site((x=DiracBelief([0.6,0.7]),))
    @test_throws DimensionMismatch site((x=Gaussian([0.6,0.7],I(2)),))
    @test_throws DimensionMismatch site((x=DiracBelief(0.6),),Gaussian([0.8,0.9],I(2)))
    @test_throws DimensionMismatch site((x=DiracBelief(0.6),);θ=zeros(2,6))
    @test_throws ArgumentError site((x=DiracBelief(0.6),);θ=fill(NaN,1,6))
    @test_throws ArgumentError site((x=DiracBelief(NaN),))
    @test_throws ArgumentError site((x=DiracBelief(0.),),Gaussian(0.,0.1))
    damped = PolynomialFactor(f.dims,tab,1;λ=0.1)
    @test isuninformative(site((x=DiracBelief(0.),),Gaussian(0.,0.1);g=damped))
    @test_throws ArgumentError site((x=DiracBelief(0.),),Gaussian(0.,0.1);g=damped,θ=zeros(1,6))
    @test_throws ArgumentError site((x=DiracBelief(0.6),);t=:bad)
    @test_throws ArgumentError site((x=DiracBelief(0.6),);pol=first(supported_polarities(f)))
    @test_throws ArgumentError assemble(f,Polarity(x=Latent(),y=Unobserved()),(;Θ),(;))
    @test !supports_polarity(f,Polarity(x=Latent(),y=Unobserved()))
    @test !supports_polarity(f,Polarity(a=Observed(),y=Unobserved()))
    @test !supports_polarity(f,Polarity(x=Unobserved(),y=Unobserved()))
    for it in (0,-1,true)
        @test_throws ArgumentError PolynomialFactor(f.dims,tab,1;iterations=it)
    end
    # The shared checker unconditionally requires game-level free_energy/entropy
    # and graph-local free energy. Those are intentionally absent (LP-009).
    # The degree-one test above checks the lens/site/posterior subset on EVERY
    # polarity with independent fixtures; do not advertise full conformance.
    @test !applicable(free_energy,f,TrivialBelief(),(;),(;Θ),(;))
    @test !applicable(LenticulumCore.entropy,f,TrivialBelief(),(;),(;Θ),(;))
end

@testset "Polynomial nonlinear covariance and row-gauge sites" begin
    rng = Xoshiro(3006)
    tab = ExponentTable(3,2); Θ = rand(rng,Grassmann(2,10))
    Σ = [0.3 0.05 0.01;0.05 0.2 0.02;0.01 0.02 0.4]
    f = PolynomialFactor((a=1,b=2),tab,2;Σ,λ=0.2)
    μ = [0.4,-0.2]; π = Gaussian(μ,[0.2 0.03;0.03 0.1])
    inputs = (a=Gaussian(0.6,0.07),); p = last(supported_polarities(f))
    z = [0.6;μ]
    J = Θ*veronese_jacobian(tab,z); Ju = J[:,2:3]; r = Θ*veronese(tab,z)
    # Include target AND observed blocks of Σ; add observed belief variance once.
    A = J*Σ*J'+0.07J[:,1]*J[:,1]'+0.2Θ*Θ'
    Λ = Ju'*(A\Ju); h = Λ*μ-Ju'*(A\r)
    s = first(factor_message(f,:b,p,inputs,π,(;Θ),(;)))
    @test s.Λ ≈ Λ rtol=2e-13
    @test s.η ≈ h rtol=2e-13
    B = [1.3 0.2;-0.1 0.7]
    sb = first(factor_message(f,:b,p,inputs,π,(Θ=B*Θ,),(;)))
    @test sb.Λ ≈ s.Λ rtol=2e-13
    @test sb.η ≈ s.η rtol=2e-13
    # A Chebyshev residual follows the same exact-Jacobian message path.
    tc = ExponentTable(3,2;basis=:chebyshev)
    Θc = Θ/Float64.(basis_change(tab,tc))
    for t in (tab,tc)
        θ = t === tab ? Θ : Θc
        g = PolynomialFactor(f.dims,t,2;Σ)
        st = first(factor_message(g,:b,p,inputs,π,(Θ=θ,),(;)))
        A0 = J*Σ*J'+0.07J[:,1]*J[:,1]'
        @test st.Λ ≈ Ju'*(A0\Ju) rtol=2e-13
        @test st.η ≈ st.Λ*μ-Ju'*(A0\r) rtol=2e-13
    end
end

@testset "Polynomial note corrections" begin
    # At circle tangency a scalar condition number is always 1: singular value,
    # not condition number alone, detects the collapsing target derivative.
    for ε in (1e-2,1e-4,1e-6,1e-8)
        x = 1-ε; y = sqrt(1-x*x)
        @test cond(reshape([2y],1,1)) == 1
        @test abs((-x/y)*sqrt(ε)+1/sqrt(2)) < 0.006
    end
    # Overdetermined fixed-metric IFT includes the mixed residual Hessian.
    # r = [u²+o*u-1, u-o]; independently solve its scalar stationarity equation.
    function min_u(o)
        u = 0.8
        for _ in 1:30
            r = u*u+o*u-1
            u -= ((2u+o)*r+u-o)/((2u+o)^2+2r+1)
        end
        u
    end
    o = 0.4; u = min_u(o); r = u*u+o*u-1
    Huu = (2u+o)^2+1+2r
    Huo = (2u+o)*u-1+r
    fd = (min_u(o+1e-6)-min_u(o-1e-6))/2e-6
    @test -Huo/Huu ≈ fd atol=1e-9
    @test abs(-((2u+o)*u-1)/Huu-fd)>1e-3
    # Tikhonov regularises J_u' * λ = -cotangent, including its transpose.
    J = [1. 2;0.3 -0.7]; cotangent = [0.4,-0.8]; ε = 0.1
    adjoint = -(J*J'+ε*I)\(J*cotangent)
    @test norm(J*(J'*adjoint+cotangent)+ε*adjoint)<1e-14
    # Nonnegative polynomial least squares need not attain its infimum.
    # r(x,y)=[xy-1,x] has no zero, yet at (1/n,n) its squared norm is 1/n².
    @test [(1/n*n-1)^2+(1/n)^2 for n in (1,10,100)] ≈ [1.,0.01,0.0001]
end

# A downstream method only implements this documented dispatch hook.
struct FixedSampsonDescent <: AbstractSampsonMethod
    step::Float64
end
function AlgebraicFactors.sampson_solve(method::FixedSampsonDescent,problem::SampsonProblem,Θ;
                                        tol,maxiter)
    M = problem.manifold
    cost,G = problem.cost_gradient(Θ)
    grad = project_tangent(M,Θ,G)
    iterations = 0
    while norm(grad)>tol && iterations<maxiter
        Θ = retract(M,Θ,-method.step*grad)
        cost,G = problem.cost_gradient(Θ)
        grad = project_tangent(M,Θ,G)
        iterations += 1
    end
    gradient_norm = norm(grad)
    return (;Θ,cost,gradient_norm,converged=gradient_norm<=tol,iterations)
end

@testset "Exchangeable Sampson solvers and RTR" begin
    rng = Xoshiro(3003)
    tab = ExponentTable(2,2)
    φ=rand(rng)*2π; R=[cos(φ) -sin(φ);sin(φ) cos(φ)]
    A=R*Diagonal([1.3,0.8]); c=0.2randn(rng,2); angles=2π*rand(rng,200)
    noisy=c .+ A*vcat(cos.(angles)',sin.(angles)')+0.04randn(rng,2,200)
    for (label,Y) in (("unscaled ellipse",noisy),("ellipse / 2 (Σ=I)",noisy/2))
        fits = [fit_sampson(tab,Y,1;method,tol=1e-7,maxiter=1000)
                for method in (RiemannianGradient(),FNS(),RiemannianTrustRegion())]
        for (method,r) in zip((:riemannian,:fns,:trust_region),fits)
            @test r.converged
            @test r.gradient_norm<=1e-7
            @test ispoint(Grassmann(1,6),r.Θ)
            @test r.cost ≈ sampson_oracle(tab,Y,r.Θ) rtol=2e-13
            @info "Sampson comparison" problem=label method iterations=r.iterations cost=r.cost gradient_norm=r.gradient_norm hessian_products=get(r,:hessian_products,0)
        end
        @test fits[3].iterations<=6
        @test fits[3].cost ≈ fits[2].cost atol=1e-12
        @test fits[3].cost <= fits[1].cost+1e-14
    end
    # Coordinates of size ~30 (ellipse × 20, Σ=I): the quadratic Veronese columns (~10³) dwarf the
    # constant one, gradient descent stalls, the second-order methods do not.
    big = 20noisy
    stalled = fit_sampson(tab,big,1;method=RiemannianGradient(),tol=1e-7,maxiter=2000)
    fnsbig = fit_sampson(tab,big,1;method=FNS(),tol=1e-7,maxiter=100)
    rtrbig = fit_sampson(tab,big,1;method=RiemannianTrustRegion(),tol=1e-7,maxiter=100)
    @test !stalled.converged && stalled.gradient_norm>1e-3
    @test fnsbig.converged && rtrbig.converged
    @test rtrbig.iterations<=10
    @test rtrbig.cost ≈ fnsbig.cost rtol=1e-12
    @test rtrbig.cost < stalled.cost
    for (method,r) in ((:riemannian,stalled),(:fns,fnsbig),(:trust_region,rtrbig))
        @info "Sampson comparison" problem="ellipse × 20 (Σ=I)" method iterations=r.iterations cost=r.cost gradient_norm=r.gradient_norm hessian_products=get(r,:hessian_products,0)
    end
    # Well-conditioned affine line: a simple external fixed-step solver reaches
    # the same stationary subspace without relying on any private cache layout.
    line = vcat(collect(range(-1.,1.;length=40))', (0.7collect(range(-1.,1.;length=40)).+0.2)')
    line += 0.04randn(rng,2,40)
    linear = ExponentTable(2,1)
    reference = fit_sampson(linear,line,1;method=RiemannianTrustRegion())
    initial = fit_nullspace(linear,line,1).Θ
    initialcopy = copy(initial)
    custom = fit_sampson(linear,line,1;Θ0=initial,method=FixedSampsonDescent(0.005),maxiter=1000)
    @test custom.converged && reference.converged
    @test custom.cost ≈ reference.cost atol=1e-12
    @test custom.Θ'*custom.Θ ≈ reference.Θ'*reference.Θ atol=1e-7
    @test initial == initialcopy
    rng2 = Xoshiro(3010)
    t3 = ExponentTable(3,2); t=2π*rand(rng2,200)
    curve=vcat(cos.(t)',sin.(t)',(cos.(t).^2)')+0.02randn(rng2,3,200)
    gradient = fit_sampson(t3,curve,2;method=RiemannianGradient(),tol=1e-7,maxiter=1000)
    trust = fit_sampson(t3,curve,2;method=RiemannianTrustRegion(),tol=1e-7,maxiter=20)
    @test trust.converged && trust.gradient_norm<=1e-7
    @test trust.iterations<=6
    @test trust.cost <= gradient.cost
    @test trust.cost ≈ sampson_oracle(t3,curve,trust.Θ) rtol=2e-12
    @test ispoint(Grassmann(2,10),trust.Θ)
    for (method,r) in ((:riemannian,gradient),(:trust_region,trust))
        @info "Sampson comparison" problem="noisy two quadrics, k=2 (seed 3010)" method iterations=r.iterations cost=r.cost gradient_norm=r.gradient_norm hessian_products=get(r,:hessian_products,0)
    end
    # Non-default radii force boundary truncation; still converge on the ellipse.
    small = fit_sampson(tab,noisy,1;method=RiemannianTrustRegion(Δ0=1e-4,Δmax=0.01,tcg_maxiter=20),maxiter=100)
    @test small.converged
    @test small.iterations>4
    for method in (RiemannianGradient(),FNS(),RiemannianTrustRegion(),FixedSampsonDescent(0.005))
        stopped = fit_sampson(tab,noisy,1;method,maxiter=0)
        @test !stopped.converged && stopped.iterations==0
        @test stopped.Θ == fit_nullspace(tab,noisy,1).Θ
    end
    @test_throws ArgumentError fit_sampson(tab,noisy,1;method=1)
    @test_throws ArgumentError fit_sampson(tab,noisy,2;method=FNS())
    for kwargs in ((c=0.,),(c=1.,),(c=NaN,),(shrink=0.,),(shrink=1.,),(shrink=Inf,),
                   (max_backtracks=0,),(max_backtracks=true,),(max_backtracks=1.5,),(initial_step=0.,),(initial_step=Inf,))
        @test_throws ArgumentError RiemannianGradient(;kwargs...)
    end
    for kwargs in ((Δ0=0.,),(Δ0=NaN,),(Δmax=Inf,),(Δmax=0.1,Δ0=1.,),
                   (ρ_accept=-0.1,),(ρ_accept=0.25,),(ρ_accept=NaN,),
                   (tcg_maxiter=0,),(tcg_maxiter=true,),(tcg_maxiter=1.5,),(κ=0.,),(κ=1.,),(κ=NaN,),
                   (θ=0.,),(θ=Inf,))
        @test_throws ArgumentError RiemannianTrustRegion(;kwargs...)
    end
    # Negative curvature and positive-curvature boundary intersections, plus interior CG.
    method = RiemannianTrustRegion()
    g = [1. 0.]
    for H in (ξ->-ξ, ξ->ξ)
        η,boundary = AlgebraicFactors._sampson_tcg(method,g,H,0.1)
        @test boundary && norm(η)≈0.1
        @test dot(g,η)+dot(η,H(η))/2<0
    end
    η,boundary = AlgebraicFactors._sampson_tcg(method,g,ξ->ξ,2.)
    @test !boundary && η == -g
end

@testset "Analytic Sampson Riemannian Hessian" begin
    rng = Xoshiro(3011)
    tab = ExponentTable(3,2)
    t = range(0,2π;length=80)
    X = vcat(cos.(t)',sin.(t)',(cos.(t).^2)')+0.02randn(rng,3,80)
    for k in (1,2), λ in (0.,0.2)
        M = Grassmann(k,10)
        Θ = fit_nullspace(tab,X,k).Θ
        data = AlgebraicFactors._sampson_data(tab,X,[1.4 0.2 0.;0.2 0.8 0.;0. 0. 1.1],λ)
        cost,G = AlgebraicFactors._sampson_cost_gradient(Θ,data)
        grad = project_tangent(M,Θ,G)
        for _ in 1:4
            ξ = project_tangent(M,Θ,randn(rng,k,10)); ξ/=norm(ξ)
            ζ = project_tangent(M,Θ,randn(rng,k,10)); ζ/=norm(ζ)
            Hξ = AlgebraicFactors._sampson_hessian(M,Θ,ξ,data)
            Hζ = AlgebraicFactors._sampson_hessian(M,Θ,ζ,data)
            @test norm(Hξ*Θ') < 1e-10*max(1,norm(Hξ))
            @test dot(ζ,Hξ) ≈ dot(ξ,Hζ) rtol=1e-10 atol=1e-9
            h = 1e-6
            Θp,Θm = retract(M,Θ,h*ξ),retract(M,Θ,-h*ξ)
            gp = project_tangent(M,Θp,last(AlgebraicFactors._sampson_cost_gradient(Θp,data)))
            gm = project_tangent(M,Θm,last(AlgebraicFactors._sampson_cost_gradient(Θm,data)))
            @test Hξ ≈ project_tangent(M,Θ,(gp-gm)/(2h)) rtol=2e-6 atol=1e-6
            # Independent cost oracle: symmetric second differences cancel odd terms.
            h = 2e-4
            second = (sampson_oracle(tab,X,retract(M,Θ,h*ξ);Σ=[1.4 0.2 0.;0.2 0.8 0.;0. 0. 1.1],λ)+
                      sampson_oracle(tab,X,retract(M,Θ,-h*ξ);Σ=[1.4 0.2 0.;0.2 0.8 0.;0. 0. 1.1],λ)-2cost)/h^2
            @test dot(ξ,Hξ) ≈ second rtol=2e-5 atol=1e-5
            errors = [abs(sampson_oracle(tab,X,retract(M,Θ,h*ξ);Σ=[1.4 0.2 0.;0.2 0.8 0.;0. 0. 1.1],λ)-
                          cost-h*dot(grad,ξ)-h^2*dot(ξ,Hξ)/2) for h in (0.002,0.001)]
            @test errors[2] <= 0.2errors[1]+1e-12 # cubic Taylor remainder
        end
    end
end
