include("conformance_capture.jl")

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
        # Endpoint sites from Gaussian multiplication, without calling the collapsed factor.
        site = forward ? Gaussian(A*[0.2,-0.5]+bias,Q) :
            GaussianBelief(vec(A'*(Q\(y-bias))),A'*(Q\A))
        if forward
            covariance = inv(inv(fill(0.7,1,1))+inv(Q))
            posterior = Gaussian(vec(covariance*([0.1/0.7]+Q\(A*[0.2,-0.5]+bias))),covariance)
        else
            posterior = Gaussian(vec(post),C)
        end
        check_factor_conformance(f,p;π=prior,inputs,ps,st,
            pointwise=([0.2,-0.5],nothing,[0.7]), game_prior=π,game_y=obs,
            msgs=(x=π,y=obs),expected_site=site,expected_posterior=posterior,
            expected_free_energy=GradedEnergy((fit=expected_fit,complexity=logdet((2 * Base.MathConstants.pi)*Q)/2,negentropy=-expected_H)),
            expected_entropy=GradedEnergy((fit=0.0,complexity=0.0,negentropy=expected_H)))
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
    normaliser = logdet((2 * Base.MathConstants.pi)*Q)/2
    left_F = GradedEnergy((fit=expected_fit,complexity=normaliser,negentropy=-expected_H))
    right_F = GradedEnergy((fit=dot(point_residual,Q\point_residual)/2,complexity=normaliser,negentropy=0.0))
    left_H = GradedEnergy((fit=0.0,complexity=0.0,negentropy=expected_H))
    zero_H = GradedEnergy((fit=0.0,complexity=0.0,negentropy=0.0))
    check_factor_conformance(tensor,nothing;scope=:game,ps=tp,st=ts,
        game_prior=priors,game_y=ys,
        expected_free_energy=GradedEnergy((left=left_F,right=right_F)),
        expected_entropy=GradedEnergy((left=left_H,right=zero_H)))
    @test_throws ArgumentError check_factor_conformance(tensor,nothing;scope=:game,ps=tp,st=ts)
    @test_throws ArgumentError check_factor_conformance(tensor,nothing;scope=:unknown,ps=tp,st=ts)
    @test_throws ArgumentError check_factor_conformance(tensor,nothing;scope=:game,ps=tp,st=ts,
        game_gap="cannot omit a game-only check")
    # Same scalar sum, wrong coordinates: the checker must reject the fixture.
    wrong_left = GradedEnergy((fit=expected_fit+1,complexity=normaliser-1,negentropy=-expected_H))
    captured = capture_conformance_checks() do
        check_factor_conformance(tensor,nothing;scope=:game,ps=tp,st=ts,
            game_prior=priors,game_y=ys,
            expected_free_energy=GradedEnergy((left=wrong_left,right=right_F)),
            expected_entropy=GradedEnergy((left=left_H,right=zero_H)))
    end
    @test conformance_failures(captured) == 1
    # The product API also admits Gaussian priors, collapsed pairs and nested products.
    prior_factor = GaussianPrior(:q,0.3,0.8)
    prior_F = GradedEnergy((fit=0.5,complexity=log((2 * Base.MathConstants.pi)*0.8)/2,negentropy=-log((2 * Base.MathConstants.pi)*exp(1)*0.8)/2))
    prior_H = GradedEnergy((fit=0.0,complexity=0.0,negentropy=log((2 * Base.MathConstants.pi)*exp(1)*0.8)/2))
    nested = TensorFactor((product=tensor,collapsed=f,source=prior_factor))
    check_factor_conformance(nested,nothing;scope=:game,
        ps=(product=tp,collapsed=ps,source=(;)),st=(product=ts,collapsed=st,source=(;)),
        game_prior=(product=priors,collapsed=π,source=TrivialBelief()),
        game_y=(product=ys,collapsed=obs,source=nothing),
        expected_free_energy=GradedEnergy((product=GradedEnergy((left=left_F,right=right_F)),collapsed=left_F,source=prior_F)),
        expected_entropy=GradedEnergy((product=GradedEnergy((left=left_H,right=zero_H)),collapsed=left_H,source=prior_H)))
    point_r = y-A*[0.2,-0.5]-bias
    check_factor_conformance(f,nothing;scope=:game,ps,st,game_prior=DiracBelief([0.2,-0.5]),game_y=obs,
        expected_free_energy=GradedEnergy((fit=dot(point_r,Q\point_r)/2,complexity=normaliser,negentropy=0.0)),
        expected_entropy=zero_H)
    # Soft Gaussian observations pool the full joint (x,y), rather than fixing y at its mean.
    V = fill(0.4,1,1)
    precision = [inv(S)+A'*(Q\A) -A'/Q; -Q\A inv(Q)+inv(V)]
    covariance = inv(precision)
    mean = covariance*vcat(S\m-A'*(Q\bias),Q\bias+V\[0.7])
    residual_map = hcat(-A,ones(1,1))
    residual = residual_map*mean-bias
    fit = (dot(residual,Q\residual)+tr(Q\(residual_map*covariance*residual_map')))/2
    entropy = (3*(1+log((2 * Base.MathConstants.pi)))+logdet(covariance))/2
    check_factor_conformance(f,nothing;scope=:game,ps,st,game_prior=π,game_y=Gaussian(0.7,0.4),
        expected_free_energy=GradedEnergy((fit=fit,complexity=normaliser,negentropy=-entropy)),
        expected_entropy=GradedEnergy((fit=0.0,complexity=0.0,negentropy=entropy)))
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
