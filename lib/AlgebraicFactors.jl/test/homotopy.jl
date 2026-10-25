# This check precedes loading HC, in the same process used by Pkg.test().
@testset "Homotopy optional dependency absent" begin
    @test Base.get_extension(AlgebraicFactors,:AlgebraicFactorsHomotopyContinuationExt) === nothing
    f=PolynomialFactor((x=1,y=1),ExponentTable(2,2),1;branches=HomotopyBranches())
    p=only(filter(p->only(unobserved_channels(p))===:y,supported_polarities(f)))
    ps=(Θ=reshape([-1.,0,0,1,0,1],1,:),)
    @test_throws r"load HomotopyContinuation" polynomial_branches(f,:y,(x=DiracBelief(0.6),),nothing,ps)
    @test_throws r"load HomotopyContinuation" factor_message(f,:y,p,(x=DiracBelief(0.6),),nothing,ps,(;))
end

# Include loading and first compilation in the reported extension cost.
const homotopy_test_started = time()
using HomotopyContinuation

hcprior(μ,P) = Gaussian([μ],reshape([P],1,1))
hcpolarity(f) = only(filter(p->only(unobserved_channels(p))===:y,supported_polarities(f)))
hcmean(m) = sum(w*belief_mean(c)[1] for (w,c) in zip(mixture_weights(m),m.components))
hcsorted(d) = sort(filter(r->r.weight>0,d.roots);by=r->real(r.root[1]))

@testset "Homotopy branches (§4)" begin
    tab=ExponentTable(2,2)
    ps=(Θ=reshape([-1.,0,0,1,0,1],1,:),)
    f=PolynomialFactor((x=1,y=1),tab,1;Σ=0.0025I,branches=HomotopyBranches())
    inputs=(x=DiracBelief(0.6),)
    d=polynomial_branches(f,:y,inputs,hcprior(0.,0.4),ps)
    roots=hcsorted(d)
    @test real.(first.(getproperty.(roots,:root))) ≈ [-0.8,0.8] atol=1e-10
    @test d.weights ≈ [0.5,0.5]
    @test all(r->r.certified===true,roots)
    @test all(p->!p.failed&&!p.at_infinity,d.paths)
    @test all(r->r.A≈reshape([0.01],1,1),roots)
    prior=hcprior(0.3,0.4)
    a=polynomial_branches(f,:y,inputs,prior,ps)
    ar=hcsorted(a)
    S=0.01/1.6^2
    @test ar[2].weight/ar[1].weight ≈ exp(2*0.8*0.3/(0.4+S)) rtol=1e-10
    @test ar[2].detJu ≈ 1.6
    @test belief_cov(ar[2].component)[1] ≈ 1/(1/0.4+1/S)
    @test belief_mean(ar[2].component)[1] ≈ (0.3/0.4+0.8/S)/(1/0.4+1/S)
    lens,_=assemble(f,hcpolarity(f),ps,(;))
    msg,_=likelihood_message(lens,prior,inputs,ps,(;))
    post,_=invert(lens,prior,inputs,ps,(;))
    pooled=combine(prior,msg)
    @test pooled.logw ≈ post.logw
    @test hcmean(post) ≈ hcmean(a.posterior)
    @test mixture_weights(msg) ≈ [0.5,0.5]
    @test first(factor_message(f,:y,hcpolarity(f),inputs,prior,ps,(;))).logw ≈ msg.logw
    for (c,e) in zip(pooled.components,post.components)
        @test c.η ≈ e.η
        @test c.Λ ≈ e.Λ
    end
    flat=polynomial_branches(f,:y,inputs,TrivialBelief(),ps)
    @test flat.posterior.logw≈flat.likelihood.logw
    @test all(r->belief_mean(r.component)≈real.(r.root),hcsorted(flat))
    no=polynomial_branches(f,:y,(x=DiracBelief(1.2),),nothing,ps)
    @test isempty(no.weights) && no.posterior===nothing
    @test count(r->!r.real,no.roots)==2
    @test_throws r"no regular real root" factor_message(f,:y,hcpolarity(f),(x=DiracBelief(1.2),),nothing,ps,(;))
    tangent=polynomial_branches(f,:y,(x=DiracBelief(1.),),nothing,ps)
    @test any(r->r.singular && abs(r.root[1])<1e-6,tangent.roots)
    @test all(r->r.weight==0,tangent.roots)
    @test_throws r"no regular real root" factor_message(f,:y,hcpolarity(f),(x=DiracBelief(1.),),nothing,ps,(;))
    @test all(r->r.certified===nothing,polynomial_branches(f,:y,inputs,nothing,ps;method=HomotopyBranches(certify=false)).roots)
    @test_throws ArgumentError HomotopyBranches(real_tol=-1)
    @test_throws ArgumentError polynomial_branches(f,:y,inputs,DiracBelief(0.),ps)
    @test_throws ArgumentError polynomial_branches(f,:y,(x=TrivialBelief(),),nothing,ps)

    # Observed Gaussian uncertainty and full-data factor noise, unchanged from §3(b).
    ginputs=(x=hcprior(0.6,0.03),)
    dg=polynomial_branches(f,:y,ginputs,nothing,ps)
    @test all(r->r.A≈reshape([0.01+1.2^2*0.03],1,1),hcsorted(dg))

    # Cubic coefficients (1,x,y,x²,xy,y²,x³,x²y,xy²,y³).
    cp=(Θ=reshape([0.,-1,-1,0,0,0,0,0,0,1],1,:),)
    cf=PolynomialFactor((x=1,y=1),ExponentTable(2,3),1;Σ=0.001I,branches=HomotopyBranches())
    critical=2/(3sqrt(3))
    for (x,n) in ((0.,3),(0.2,3),(critical-1e-4,3),(critical+1e-4,1),(-critical+1e-4,3),(-critical-1e-4,1),(0.6,1))
        cd=polynomial_branches(cf,:y,(x=DiracBelief(x),),nothing,cp)
        @test length(cd.weights)==n
        @test all(r->abs(real(r.root[1])^3-real(r.root[1])-x)<1e-10,hcsorted(cd))
        @test all(r->r.certified===true,hcsorted(cd))
    end

    # Product conics: u₁²=1+x, u₂²=4-x, so four closed-form roots.
    t3=ExponentTable(3,2)
    Θ=zeros(2,size(t3.exponents,2))
    for (row,ex,c) in ((1,[0,0,0],-1.),(1,[1,0,0],-1.),(1,[0,2,0],1.),(2,[0,0,0],-4.),(2,[1,0,0],1.),(2,[0,0,2],1.))
        Θ[row,only(findall(j->t3.exponents[:,j]==ex,axes(Θ,2)))]=c
    end
    ff=PolynomialFactor((x=1,y=2),t3,2;Σ=0.001I,branches=HomotopyBranches())
    prior2=Gaussian([0.2,-0.3],[0.7 0.1;0.1 0.9])
    dd=polynomial_branches(ff,:y,(x=DiracBelief(0.2),),prior2,(;Θ))
    @test length(dd.weights)==4
    @test all(r->abs.(real.(r.root))≈sqrt.([1.2,3.8]),hcsorted(dd))
    @test all(r->r.certified===true,hcsorted(dd))
    B=[1.5 0.3;-0.2 0.8]
    gauge=polynomial_branches(ff,:y,(x=DiracBelief(0.2),),prior2,(Θ=B*Θ,))
    # Match by root rather than HC's path order.
    for r in hcsorted(dd)
        s=only(filter(s->norm(s.root-r.root)<1e-7,gauge.roots))
        @test r.weight≈s.weight rtol=1e-10
        @test r.site.Λ≈s.site.Λ rtol=1e-10
    end
    # Two finite roots and projective paths at infinity: u₁²=1, u₁u₂=1.
    infinityf=PolynomialFactor((y=2,),ExponentTable(2,2),2;Σ=0.001I,branches=HomotopyBranches())
    infinityps=(Θ=[-1. 0 0 1 0 0;-1. 0 0 0 1 0],)
    infinitydata=polynomial_branches(infinityf,:y,(;),nothing,infinityps)
    @test length(infinitydata.weights)==2
    @test any(p->p.at_infinity,infinitydata.paths)
    @test all(r->!r.at_infinity&&!r.failed,infinitydata.roots)
    cheb=ExponentTable(2,3;basis=:chebyshev)
    Θc=cp.Θ/Float64.(basis_change(cf.table,cheb))
    chf=PolynomialFactor(cf.dims,cheb,1;Σ=cf.Σ,branches=HomotopyBranches())
    cm=polynomial_branches(cf,:y,(x=DiracBelief(0.2),),prior,cp)
    cc=polynomial_branches(chf,:y,(x=DiracBelief(0.2),),prior,(Θ=Θc,))
    for (r,s) in zip(hcsorted(cm),hcsorted(cc))
        @test r.root≈s.root atol=1e-10
        @test r.weight≈s.weight rtol=1e-9
    end
    # Homogeneous Chebyshev indices include constant terms outside that span.
    cht=ExponentTable(2,2;basis=:chebyshev,homogeneous=true)
    chhom=PolynomialFactor((x=1,y=1),cht,1;Σ=0.001I,branches=HomotopyBranches())
    hh=polynomial_branches(chhom,:y,(x=DiracBelief(0.),),nothing,(Θ=reshape([1.,0,1],1,:),))
    @test [real(r.root[1]) for r in hcsorted(hh)]≈[-1.,1.] atol=1e-10
    for (dims,k) in (((x=1,y=2),1),((x=1,y=1),2))
        nf=PolynomialFactor(dims,ExponentTable(sum(dims),2),k;branches=HomotopyBranches())
        @test_throws r"§3's single-branch site" polynomial_branches(nf,:y,inputs,nothing,(Θ=zeros(k,size(nf.table.exponents,2)),))
    end
end

@testset "Homotopy small-noise quadrature" begin
    # Constant residual covariance A=δ via Σ=0, λ=δ/||Θ||².
    # Thus the normalised residual Gaussian and exp(-r²/(2A)) have the
    # same posterior. With varying A, the latter needs a det(A)^(-1/2)
    # factor to have the §4.3 limit; the implementation note explains this.
    prior=hcprior(0.3,0.7)
    grid=collect(range(-3.,3.;length=300001))
    for (name,degree,θ,x,residual,cuts) in (
        (:circle,2,[-1.,0,0,1,0,1],0.6,y->y^2-0.64,[0.]),
        (:cubic,3,[0.,-1,-1,0,0,0,0,0,0,1],0.2,y->y^3-y-0.2,[-1/sqrt(3),1/sqrt(3)]))
        ps=(Θ=reshape(θ,1,:),)
        meanerr,masserr,geminierr=Float64[],Float64[],Float64[]
        for δ in (1e-3,1e-4,1e-5)
            f=PolynomialFactor((x=1,y=1),ExponentTable(2,degree),1;Σ=zeros(2,2),λ=δ/sum(abs2,θ),branches=HomotopyBranches())
            d=polynomial_branches(f,:y,(x=DiracBelief(x),),prior,ps)
            rs=hcsorted(d)
            density=exp.([-residual(y)^2/(2δ)-(y-0.3)^2/(2*0.7) for y in grid])
            density./=sum(density)
            exactmean=sum(grid.*density)
            bins=[searchsortedlast(cuts,y)+1 for y in grid]
            masses=[sum(density[bins.==j]) for j in 1:length(rs)]
            gemini=[exp(Mycelium.belief_logdensity(prior,real.(r.root)))/abs(r.detJu) for r in rs]
            gemini./=sum(gemini)
            push!(meanerr,abs(hcmean(d.posterior)-exactmean))
            push!(masserr,maximum(abs.([r.weight for r in rs].-masses)))
            push!(geminierr,maximum(abs.([r.weight for r in rs].-gemini)))
        end
        @info "Small-noise quadrature" name meanerr=repr(meanerr) masserr=repr(masserr) geminierr=repr(geminierr) grid_points=length(grid)
        @test all(diff(meanerr).<0)
        @test all(diff(masserr).<0)
        @test all(diff(geminierr).<0)
        @test last(meanerr)<5e-5
        @test last(masserr)<5e-5
        @test last(geminierr)<5e-5
    end
end

@testset "Homotopy mixture entropy bounds" begin
    # Reproducible MC standard-error tolerance, both overlapping and separated.
    rng=Xoshiro(4404)
    N=100000
    for variance in (1.0,0.01)
        f=PolynomialFactor((x=1,y=1),ExponentTable(2,2),1;Σ=zeros(2,2),λ=variance*2.56/3,branches=HomotopyBranches())
        d=polynomial_branches(f,:y,(x=DiracBelief(0.6),),nothing,(Θ=reshape([-1.,0,0,1,0,1],1,:),))
        m=d.posterior
        bounds=branch_entropy_bounds(m)
        w=mixture_weights(m)
        samples=map(1:N) do _
            c=m.components[rand(rng)<w[1] ? 1 : 2]
            [belief_mean(c)[1]+sqrt(belief_cov(c)[1])*randn(rng)]
        end
        surprise=[-Mycelium.belief_logdensity(m,x) for x in samples]
        estimate=sum(surprise)/N
        se=sqrt(sum(abs2,surprise.-estimate)/(N-1)/N)
        @info "Entropy MC" variance N estimate se bounds
        @test bounds.label_entropy≈log(2)
        @test bounds.lower-5se≤estimate≤bounds.upper+5se
        @test variance==1 ? estimate<bounds.upper-0.1 : abs(estimate-bounds.upper)<5se
    end
end
@info "Homotopy extension test elapsed (load + compilation + checks)" seconds=time()-homotopy_test_started
