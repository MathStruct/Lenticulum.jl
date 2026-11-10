struct ShearScale{T} <: LuxCore.AbstractLuxLayer
    c::T
end
(f::ShearScale)(z,ps,st) = ([z[1],exp(f.c*z[1])*z[2]],st)

@testset "LP-009 Lebesgue entropy under invertible transport" begin
    nt = (;)
    A = [2.0 0.5;0.0 -0.7]; b = [0.2,-1.0]
    f = GeneratorFactor(AffineGenerator(A,b),(z=2,x=2))
    ps,st = LuxCore.setup(Xoshiro(9003),f)
    source = Gaussian([0.3,-0.4],[0.8 0.1;0.1 0.5])
    exact = (2*(1+log(2π))+logdet(A*belief_cov(source)*A'))/2
    @test first(pushforward_entropy(f,source,ps,st)) ≈ exact
    @test first(LenticulumCore.entropy(f,TrivialBelief(),source,ps,st)) ≈ exact
    @test first(free_energy(f,nothing,source,ps,st)) ≈ -exact
    pushed = first(pushforward(f,source,ps,st))
    @test belief_mean(pushed) ≈ A*belief_mean(source)+b
    @test belief_cov(pushed) ≈ A*belief_cov(source)*A'
    check_factor_conformance(f,only(supported_polarities(f));π=TrivialBelief(),
        inputs=(z=source,),ps,st,pointwise=(zeros(2),nothing,b),
        game_prior=TrivialBelief(),game_y=source,msgs=(z=source,),
        expected_site=Gaussian(A*[0.3,-0.4]+b,A*[0.8 0.1;0.1 0.5]*A'),
        expected_posterior=Gaussian(A*[0.3,-0.4]+b,A*[0.8 0.1;0.1 0.5]*A'),
        expected_free_energy=-exact,expected_entropy=exact)
    @test_throws ArgumentError pushforward_entropy(f,DiracBelief([0.0,0.0]),ps,st)
    @test_throws ArgumentError pushforward_entropy(f,SampleBelief([[0.0,0.0]]),ps,st)
    @test_throws ArgumentError pushforward_entropy(f,source,(A=zeros(2,2),b=b),st)
    @test_throws ArgumentError LenticulumCore.entropy(f,DiracBelief(b),source,ps,st)
    rect = GeneratorFactor(AffineGenerator(ones(1,2),[0.0]),(z=2,x=1))
    rp,rs = LuxCore.setup(Xoshiro(1),rect)
    @test_throws ArgumentError pushforward_entropy(rect,source,rp,rs)
    @test_throws ArgumentError pushforward_entropy(f,uninformative(2),ps,st)
    # g(z)=(z₁, exp(c*z₁)*z₂) is a global nonlinear bijection of R².
    # log|det J|=c*z₁, so its expectation and sampling standard error are closed form.
    c = 0.6
    net = ShearScale(c)
    nf = GeneratorFactor(InvertibleGenerator(net,(z,ps,st)->c*z[1]),(z=2,x=2))
    np,ns = LuxCore.setup(Xoshiro(9003),nf)
    est = first(estimate_pushforward_entropy(nf,source,np,ns;nsamples=16000,rng=Xoshiro(9003)))
    H = (2*(1+log(2π))+logdet(belief_cov(source)))/2
    @test est isa EntropyEstimate
    @test !(est isa Real)
    @test abs(est.value-(H+c*0.3)) < 5est.standard_error
    @test est.standard_error ≈ abs(c)*sqrt(0.8/est.nsamples) rtol=0.03
    @test_throws ArgumentError LenticulumCore.entropy(nf,nothing,source,np,ns)
    @test_throws ArgumentError estimate_pushforward_entropy(nf,DiracBelief([0.0,1.0]),np,ns;rng=Xoshiro(1))
    for d in (1,3)
        noise = NoiseSource(:z,d;nsamples=7)
        check_factor_conformance(noise,nothing;scope=:game,ps=nt,st=nt,
            game_prior=TrivialBelief(),game_y=nothing,
            expected_free_energy=-d*(1+log(2π))/2,expected_entropy=d*(1+log(2π))/2)
        @test first(LenticulumCore.entropy(noise,TrivialBelief(),nothing,nt,nt)) ≈ d*(1+log(2π))/2
        @test first(free_energy(noise,TrivialBelief(),nothing,nt,nt)) ≈ -d*(1+log(2π))/2
        @test first(local_free_energy(noise,nt,nt,nt)) ≈ -d*(1+log(2π))/2
    end
    unknown = NoiseSource(:z,1;dist=(rng,d,n)->[[0.0] for _ in 1:n])
    @test_throws ArgumentError LenticulumCore.entropy(unknown,TrivialBelief(),nothing,nt,nt)
end
