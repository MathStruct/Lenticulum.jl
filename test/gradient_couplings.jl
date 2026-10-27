using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Test, Random
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

function gaussian_coupled_tree(c)
    b=GraphBuilder()
    for v in (:x,:u,:y); variable!(b,v); end
    factor!(b,:prior,GaussianPrior(:x,0.4,0.7))
    factor!(b,:a,GaussianFactor(1=>1;noise=0.3,channels=(:x,:u)))
    factor!(b,:b,GaussianFactor(1=>1;noise=0.2,channels=(:u,:y)))
    factor!(b,:data,DataFactor(:y,[1.2]))
    for (f,ch,v) in ((:prior,:x,:x),(:a,:x,:x),(:a,:u,:u),(:b,:u,:u),(:b,:y,:y))
        connect!(b,f,ch,v;coupling=c)
    end
    connect!(b,:data,:y,:y;direction=Emitting(),coupling=c)
    g=build(b)
    p=(prior=(;),a=(A=fill(0.8,1,1),b=[0.1]),b=(A=fill(1.1,1,1),b=[-0.2]),data=(;))
    st=(prior=(;),a=(;),b=(;),data=(;))
    return g,p,st,tree_schedule(g)
end
function fixed_cavity_gradient(A,b,Q,bx,by)
    mx,Sx=only(belief_mean(bx)),only(belief_cov(bx))
    if by isa DiracBelief
        y=only(by.value); C=1/(A^2/Q+1/Sx); m=C*(A*(y-b)/Q+mx/Sx)
        r=y-A*m-b
        dΛ=2A/Q; dC=-C^2*dΛ; dm=C*((y-b)/Q-dΛ*m)
        dA=r*(-m-A*dm)/Q+(2A*C+A^2*dC)/(2Q)-dC/(2C)
        db=r*(A^2*C/Q-1)/Q
        return [dA,db]
    end
    my,Sy=only(belief_mean(by)),only(belief_cov(by))
    Λ=[1/Sx+A^2/Q -A/Q;-A/Q 1/Sy+1/Q]
    C=inv(Λ); η=[mx/Sx-A*b/Q,my/Sy+b/Q]; m=C*η; G=[-A,1.0]
    r=dot(G,m)-b
    return map((true,false)) do isA
        dΛ=isA ? [2A/Q -1/Q;-1/Q 0] : zeros(2,2)
        dη=isA ? [-b/Q,0.0] : [-A/Q,1/Q]
        dG=isA ? [-1.0,0.0] : zeros(2)
        dC=-C*dΛ*C; dm=C*(dη-dΛ*m)
        dr=dot(dG,m)+dot(G,dm)-(isA ? 0 : 1)
        dV=2dot(dG,C*G)+dot(G,dC*G)
        r*dr/Q+dV/(2Q)-tr(Λ*dC)/2
    end |> collect
end
@testset "LP-011 Gaussian marginal likelihood, fixed cavities and EM" begin
    for ad in (nothing,AutoForwardDiff(),AutoZygote())
        gradients=[]
        for c in (DiagonalCoupling(),ExactCoupling(),PathwiseCoupling())
            g,p,st,sched=gaussian_coupled_tree(c)
            _,_,_,s=infer!(g,sched,p,st)
            grad=free_energy_gradient(s,g,p,st;ad,schedule=sched)
            # y mean = B(A*m+a)+b; variance = B²(A²*S+Q)+R.
            m,S,A,B,a,b,Q,R,y=0.4,0.7,0.8,1.1,0.1,-0.2,0.3,0.2,1.2
            M=B*(A*m+a)+b; V=B^2*(A^2*S+Q)+R; d=M-y
            dM=d/V; dV=(1/V-d^2/V^2)/2
            expected_aA=dM*B*m+dV*2B^2*A*S
            expected_bA=dM*(A*m+a)+dV*2B*(A^2*S+Q)
            if c isa DiagonalCoupling
                ca,cb=messages_into(s,g,2),messages_into(s,g,3)
                oa=fixed_cavity_gradient(A,a,Q,ca.x,ca.u)
                ob=fixed_cavity_gradient(B,b,R,cb.u,cb.y)
                @test [only(grad.a.A),only(grad.a.b)] ≈ oa atol=2e-7
                @test [only(grad.b.A),only(grad.b.b)] ≈ ob atol=2e-7
                @test abs(only(grad.a.A)-expected_aA) > 0.1
            else
                @test only(grad.a.A) ≈ expected_aA atol=2e-7
                @test only(grad.b.A) ≈ expected_bA atol=2e-7
                @test only(grad.a.b) ≈ dM*B atol=2e-7
                @test only(grad.b.b) ≈ dM atol=2e-7
            end
            # True EM holds the full joint posterior fixed, rather than cavities.
            Λ=[1/S+A^2/Q -A/Q;-A/Q 1/Q+B^2/R]
            C=inv(Λ); mpost=C*[m/S-A*a/Q,a/Q+B*(y-b)/R]
            emA=((A*mpost[1]+a-mpost[2])*mpost[1]+A*C[1,1]-C[1,2])/Q
            @test emA ≈ expected_aA atol=2e-12
            push!(gradients,grad)
        end
        @test LenticulumCore._flatten(gradients[2]) ≈ LenticulumCore._flatten(gradients[3]) atol=2e-7
    end
end

@testset "LP-011 collapsed composite couplings" begin
    a=GaussianFactor(1=>1;noise=0.3,channels=(:x,:u))
    b=GaussianFactor(1=>1;noise=0.2,channels=(:u,:y))
    p=(first=(A=fill(0.8,1,1),b=[0.1]),second=(A=fill(1.1,1,1),b=[-0.2]))
    st=(first=(;),second=(;)); π=DiracBelief([0.4]); obs=DiracBelief([1.2])
    reference=free_energy_gradient(ComposedFactor(a,b),π,obs,p,st)
    for ad in (nothing,AutoForwardDiff(),AutoZygote()), c in (DiagonalCoupling(),ExactCoupling(),PathwiseCoupling())
        g=free_energy_gradient(ComposedFactor(a,b;coupling=c),π,obs,p,st;ad)
        @test LenticulumCore._flatten(g) ≈ LenticulumCore._flatten(reference) atol=2e-7
    end
    F=first(scalar_free_energy(ComposedFactor(a,b),π,obs,p,st))
    for baseline in (nothing,F), ad in (AutoForwardDiff(),AutoZygote())
        f=ComposedFactor(a,b;coupling=ScoreFunctionCoupling(;baseline))
        g=free_energy_gradient(f,π,obs,p,st;ad,rng=Xoshiro(47),nsamples=3000)
        v,se=LenticulumCore._flatten(g.value),LenticulumCore._flatten(g.standard_error)
        @test all(abs.(v-LenticulumCore._flatten(reference)) .< 6se .+ 2e-7)
        if baseline !== nothing
            @test maximum(se) < 1e-12
        end
        @info "LP-011 composite score measurement" ad baseline estimate=g
    end
end
