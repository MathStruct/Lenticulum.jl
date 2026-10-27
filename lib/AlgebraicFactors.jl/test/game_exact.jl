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
                msgs=(x=π,y=obs),expected_site=site,expected_posterior=post,
                expected_free_energy=GradedEnergy((fit=fit,complexity=logdet((2 * Base.MathConstants.pi)*Q)/2,negentropy=-H)),
                expected_entropy=GradedEnergy((fit=0.0,complexity=0.0,negentropy=H)))
        end
        for px in (DiracBelief([0.2]),π), py in (obs,Gaussian(0.8,0.2))
            # Independent scalar Gaussian conditioning, including a soft y observation.
            # The prior conditional joint has Cov(x,y)=a*S and Var(y)=a²*S+Q.
            q = only(Q); gain = A/2; offset = b/2
            fixed_x = px isa DiracBelief; fixed_y = py isa DiracBelief
            if fixed_x && fixed_y
                residual_mean = 0.8-gain*0.2-offset
                residual_var = 0.0; h = 0.0
            elseif fixed_y
                residual_mean = only(r); residual_var = only(a*postvar*a')
                h = H
            elseif fixed_x
                v = inv(inv(q)+inv(0.2))
                my = v*((gain*0.2+offset)/q+0.8/0.2)
                residual_mean = my-gain*0.2-offset
                residual_var = v; h = (1+log((2 * Base.MathConstants.pi)*v))/2
            else
                joint_cov = [0.7 gain*0.7; gain*0.7 gain^2*0.7+q]
                gain_y = joint_cov[:,2]/(joint_cov[2,2]+0.2)
                joint_mean = [0.2,gain*0.2+offset]+gain_y*(0.8-gain*0.2-offset)
                joint_post = joint_cov-gain_y*joint_cov[2,:]'
                residual_mean = dot([-gain,1.0],joint_mean)-offset
                residual_var = dot([-gain,1.0],joint_post*[-gain,1.0])
                h = (2*(1+log((2 * Base.MathConstants.pi)))+logdet(joint_post))/2
            end
            expected_F = GradedEnergy((fit=(residual_mean^2+residual_var)/(2q),
                complexity=log((2 * Base.MathConstants.pi)*q)/2,negentropy=-h))
            expected_H = GradedEnergy((fit=0.0,complexity=0.0,negentropy=h))
            check_factor_conformance(f,nothing;scope=:game,ps,st,game_prior=px,game_y=(y=py,),
                expected_free_energy=expected_F,expected_entropy=expected_H)
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
        game_y=inputs,msgs=inputs,direct_message=true,expected_site=expected,expected_posterior=expected,
        expected_free_energy=-H,expected_entropy=H)
    swapped = (z=w,w=z)
    check_factor_conformance(f,only(supported_polarities(f));π=TrivialBelief(),inputs=swapped,
        ps=nt,st=nt,pointwise=(nothing,nothing,nothing),game_prior=TrivialBelief(),
        game_y=swapped,msgs=swapped,direct_message=true,
        expected_site=Gaussian(-1.3,2.7),expected_posterior=Gaussian(-1.3,2.7),
        expected_free_energy=-H,expected_entropy=H)
    @test first(LenticulumCore.entropy(f,nothing,(z=w,w=z),nt,nt)) ≈ H
    @test_throws ArgumentError LenticulumCore.entropy(f,nothing,(z=w,w=w),nt,nt)
    @test_throws ArgumentError LenticulumCore.entropy(f,nothing,(z=DiracBelief(zeros(2)),w=w),nt,nt)
    @test_throws ArgumentError LenticulumCore.entropy(PluckerFactor(2;covariance_jitter=0.1),nothing,inputs,nt,nt)
    f3 = PluckerFactor(3)
    @test_throws ArgumentError LenticulumCore.entropy(f3,nothing,(z=DiracBelief(ones(3)),w=Gaussian(zeros(3),I(3))),nt,nt)
end
