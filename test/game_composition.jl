@testset "LP-009 exact Gaussian composition and independent tensor games" begin
    rng = Xoshiro(9005)
    a = GaussianFactor(2=>2; noise=[0.3 0.04;0.04 0.5],channels=(:x,:u))
    b = GaussianFactor(2=>1; noise=0.2,channels=(:u,:y))
    f = ComposedFactor(a,b)
    ps = (first=(A=[1.0 0.3;-0.2 0.8],b=[0.4,-0.1]),second=(A=[0.7 -1.2],b=[0.2]))
    st = (first=(;),second=(;))
    # Independent Gaussian elimination, including propagation of first-stage noise.
    A = ps.second.A*ps.first.A
    bias = ps.second.A*ps.first.b+ps.second.b
    Q = ps.second.A*a.Q*ps.second.A'+b.Q
    g = GaussianFactor(2=>1;noise=Q)
    gp = (A=A,b=bias)
    π = Gaussian([0.2,-0.5],[0.8 0.1;0.1 0.6])
    obs = DiracBelief([0.7])
    for (px,py) in ((π,obs),(DiracBelief([0.2,-0.5]),obs),(π,Gaussian(0.7,0.4)))
        F,s = free_energy(f,px,py,ps,st)
        G = first(free_energy(g,px,py,gp,(;)))
        @test all(F[k] ≈ G[k] for k in keys(F))
        @test s === st
        H = first(LenticulumCore.entropy(f,px,py,ps,st))
        @test H[:negentropy] ≈ -F[:negentropy]
    end
    m = belief_mean(π); S = belief_cov(π); y = obs.value
    Λ = inv(S)+A'*(Q\A); C = inv(Λ)
    post = C*(S\m+A'*(Q\(y-bias)))
    r = y-A*post-bias
    expected_fit = (dot(r,Q\r)+tr(Q\(A*C*A')))/2
    expected_H = (2*(1+log((2 * Base.MathConstants.pi)))+logdet(C))/2
    F = first(free_energy(f,π,obs,ps,st))
    @test F[:fit] ≈ expected_fit
    @test F[:complexity] ≈ logdet((2 * Base.MathConstants.pi)*Q)/2
    @test F[:negentropy] ≈ -expected_H
    pushed = first(LenticulumCore.pushforward(f,π,ps,st))
    @test belief_mean(pushed) ≈ A*m+bias
    @test belief_cov(pushed) ≈ A*S*A'+Q
    @test belief_cov(first(LenticulumCore.pushforward(b,first(LenticulumCore.pushforward(a,π,ps.first,st.first)),ps.second,st.second))) ≈ belief_cov(pushed)
    for p in supported_polarities(f)
        forward = only(unobserved_channels(p)) === :y
        inputs = forward ? (x=DiracBelief([0.2,-0.5]),) : (y=obs,)
        prior = forward ? Gaussian(0.1,0.7) : π
        gl = first(assemble(g,p,gp,(;)))
        site = first(likelihood_message(gl,prior,inputs,gp,(;)))
        posterior = first(invert(gl,prior,inputs,gp,(;)))
        check_factor_conformance(f,p;π=prior,inputs,ps,st,
            pointwise=([0.2,-0.5],nothing,[0.7]), game_prior=π,game_y=obs,
            msgs=(x=π,y=obs),expected_site=site,expected_posterior=posterior)
    end
    tensor = TensorFactor((left=g,right=g))
    tp = (left=gp,right=gp); ts = (left=(;),right=(;))
    priors = (left=π,right=DiracBelief([0.2,-0.5]))
    ys = (left=obs,right=DiracBelief([-0.3]))
    TF,next = free_energy(tensor,priors,ys,tp,ts)
    @test keys(TF) == (:left,:right)
    @test next == ts
    expected = sum(first(scalar_free_energy(g,priors[n],ys[n],gp,(;))) for n in keys(priors))
    @test first(scalar_free_energy(tensor,priors,ys,tp,ts)) ≈ expected
    point_residual = [-0.3]-A*[0.2,-0.5]-bias
    closed_form = expected_fit-expected_H+logdet((2 * Base.MathConstants.pi)*Q) +
        dot(point_residual,Q\point_residual)/2
    @test first(scalar_free_energy(tensor,priors,ys,tp,ts)) ≈ closed_form
    TH = first(LenticulumCore.entropy(tensor,priors,ys,tp,ts))
    @test TH[:left][:negentropy] ≈ expected_H
    @test TH[:right][:negentropy] == 0
    @test_throws ArgumentError free_energy(tensor,π,obs,tp,ts)
    @test_throws ArgumentError free_energy(tensor,(wrong=π,),ys,tp,ts)
    unsupported = TensorFactor((branch=ComposedFactor(tensor,g),))
    @test_throws ArgumentError free_energy(unsupported,(branch=π,),(branch=obs,),(;),(;))
    @test_throws DimensionMismatch free_energy(ComposedFactor(g,g),π,obs,(first=gp,second=gp),st)
    @test_throws DimensionMismatch LenticulumCore.pushforward(ComposedFactor(b,a),π,ps,st)
end
