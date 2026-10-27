@testset "LP-009 normalised degree-one polynomial games" begin
    A,b = 1.3,0.4
    Θ = [-b -A 2.0] # y=(A*x+b)/2; the output Jacobian must enter the normaliser.
    π = Gaussian(0.2,0.7); obs = DiracBelief([0.8])
    for basis in (:monomial,:chebyshev), scale in (1.0,-2.0), λ in (0.0,0.2)
        tab = ExponentTable(2,1;basis)
        f = PolynomialFactor((x=1,y=1),tab,1;Σ=[0.03 0.01;0.01 0.05],λ)
        ps,st = (Θ=scale*Θ,),(;)
        J = [-A 2.0]
        Qr = only(J*f.Σ*J')+λ*sum(abs2,Θ)
        # Exact conditional is N(y;(A*x+b)/2,Qr/4), independent of row gauge.
        m = [0.2]; S = fill(0.7,1,1); a = fill(A/2,1,1); bias = [b/2]; Q = fill(Qr/4,1,1)
        postvar = inv(inv(S)+a'*(Q\a))
        postmean = postvar*(S\m+a'*(Q\(obs.value-bias)))
        r = obs.value-a*postmean-bias
        fit = (dot(r,Q\r)+tr(Q\(a*postvar*a')))/2
        H = (1+log((2 * Base.MathConstants.pi))+logdet(postvar))/2
        F = first(free_energy(f,π,(y=obs,),ps,st))
        @test F[:fit] ≈ fit
        @test F[:complexity] ≈ logdet((2 * Base.MathConstants.pi)*Q)/2
        @test F[:negentropy] ≈ -H
        @test first(LenticulumCore.entropy(f,π,(y=obs,),ps,st))[:negentropy] ≈ H
        # Both polarities use the same incoming factor game, as L-SPEC-001 specifies.
        for p in supported_polarities(f)
            forward = only(unobserved_channels(p)) === :y
            inputs = forward ? (x=DiracBelief([0.2]),) : (y=obs,)
            prior = forward ? TrivialBelief() : π
            site = forward ? Gaussian((A*0.2+b)/2,Qr/4) :
                GaussianBelief([A*(2*0.8-b)/Qr],fill(A^2/Qr,1,1))
            post = forward ? site : Gaussian(vec(postmean),postvar)
            check_factor_conformance(f,p;π=prior,inputs,ps,st,
                pointwise=(nothing,nothing,[0.2,0.8]),game_prior=π,game_y=(y=obs,),
                msgs=(x=π,y=obs),expected_site=site,expected_posterior=post)
        end
        for px in (DiracBelief([0.2]),π), py in (obs,Gaussian(0.8,0.2))
            out = first(free_energy(f,px,(y=py,),ps,st))
            # The independently implemented top-level Gaussian factor is already included
            # by polynomial.jl's degree-one message tests, without a package dependency.
            g = PolynomialLinearOracle.GaussianFactor(1=>1;noise=Q)
            expected = first(free_energy(g,px,py,(A=a,b=bias),(;)))
            @test all(out[k] ≈ expected[k] for k in keys(out))
        end
    end
    f = PolynomialFactor((x=1,y=1),ExponentTable(2,1),1)
    @test_throws ArgumentError free_energy(f,TrivialBelief(),(y=TrivialBelief(),),(Θ=Θ,),(;))
    @test_throws ArgumentError free_energy(f,π,(y=obs,),(Θ=[0.0 1.0 0.0],),(;))
    nonlinear = PolynomialFactor((x=1,y=1),ExponentTable(2,2),1)
    @test_throws ArgumentError local_free_energy(nonlinear,(x=π,y=obs),(Θ=zeros(1,6),),(;))
end

@testset "LP-009 exact scalar Plucker image" begin
    nt = (;); f = PluckerFactor(2)
    z = DiracBelief([2.0,-1.0]); w = Gaussian([0.3,0.5],[0.7 0.1;0.1 0.4])
    # p=z₁*w₂-z₂*w₁ = [1,2]w, mean=1.3, variance=2.7.
    expected = Gaussian(1.3,2.7)
    H = log((2 * Base.MathConstants.pi)*exp(1)*2.7)/2
    inputs = (z=z,w=w)
    @test first(LenticulumCore.entropy(f,TrivialBelief(),inputs,nt,nt)) ≈ H
    @test first(free_energy(f,nothing,inputs,nt,nt)) ≈ -H
    check_factor_conformance(f,only(supported_polarities(f));π=TrivialBelief(),inputs,
        ps=nt,st=nt,pointwise=(nothing,nothing,nothing),game_prior=TrivialBelief(),
        game_y=inputs,msgs=inputs,direct_message=true,expected_site=expected,expected_posterior=expected)
    @test first(LenticulumCore.entropy(f,nothing,(z=w,w=z),nt,nt)) ≈ H
    @test_throws ArgumentError LenticulumCore.entropy(f,nothing,(z=w,w=w),nt,nt)
    @test_throws ArgumentError LenticulumCore.entropy(f,nothing,(z=DiracBelief(zeros(2)),w=w),nt,nt)
    @test_throws ArgumentError LenticulumCore.entropy(PluckerFactor(2;covariance_jitter=0.1),nothing,inputs,nt,nt)
    f3 = PluckerFactor(3)
    @test_throws ArgumentError LenticulumCore.entropy(f3,nothing,(z=DiracBelief(ones(3)),w=Gaussian(zeros(3),I(3))),nt,nt)
end
