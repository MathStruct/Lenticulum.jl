using ForwardDiff, Zygote
# ChainRulesCore is a Mycelium weak dependency loaded by Zygote; no extra test dependency needed.
const P36CRC = Base.get_extension(Mycelium, :MyceliumForwardDiffExt).ChainRulesCore

include("support/switched_normal.jl")

@testset "P36 overlay: enumerate switch and dense Gaussian products" begin
    f = P36Switched(); nt = (;)
    p = LuxCore.initialparameters(Xoshiro(36), f)
    prior = P36Fixed([0.2],fill(1.3,1,1))
    for normalise in (nothing, (:y,))
        o = OverlayFactor(:sw=>f, :prior=>prior; normalise)
        ps = (sw=p, prior=nt)
        F = cluster_form(o, ps)
        @test islearnable(o)
        @test cluster_switch(o) == (:k,)
        @test length(channels(o)) == 2 && !hasnopolarity(o) && !isunidirectional(o)
        @test !Mycelium.learns_unnormalised(o, (:y,))
        @test keys(LuxCore.initialparameters(Xoshiro(0),o)) == (:sw,:prior)
        @test keys(LuxCore.initialstates(Xoshiro(0),o)) == (:sw,:prior)
        pol = Polarity(y=Observed(), k=Unobserved())
        @test supports_polarity(o, pol) && pol in supported_polarities(o)
        @test !supports_polarity(o, Polarity(y=Unobserved(), k=Unobserved()))
        y = 0.6
        weights = Float64[]
        for k in 1:2
            precision = 1/p.q[k]+1/1.3
            μ = (p.mu[k]/p.q[k]+0.2/1.3)/precision
            if normalise === nothing
                push!(weights, exp(p.logw[k]-(y-p.mu[k])^2/(2p.q[k])-log(2π*p.q[k])/2-(y-0.2)^2/(2*1.3)-log(2π*1.3)/2))
                @test F.C[:,k] ≈ [p.mu[k],0.2]
                @test F.Q[:,:,k] ≈ Diagonal([p.q[k],1.3])
            else
                push!(weights, exp(p.logw[k]-(y-μ)^2*precision/2+log(precision/(2π))/2))
                @test F.C[:,k] ≈ [μ]
                @test F.Q[:,:,k] ≈ fill(1/precision,1,1)
            end
        end
        m, _ = factor_message(o,:k,pol,(y=DiracBelief([y]),),nothing,ps,nt)
        @test probabilities(m) ≈ weights/sum(weights)
        msgs=(y=DiracBelief([y]),k=CategoricalBelief([0.5,0.5]))
        # F_c = E_b[log incoming] - log Z; the uniform switch message cancels.
        E,_=local_free_energy(o,msgs,ps,nt)
        @test scalarise(scalarisation(o), E) ≈ -log(sum(weights))
        frozen = factor_belief(o,msgs,ps,nt)
        @test scalarise(scalarisation(o), first(belief_free_energy(o,frozen,ps,nt))) ≈ -log(sum(weights))
    end
    @test_throws ArgumentError OverlayFactor()
    @test_throws ArgumentError OverlayFactor(:a=>f,:a=>f)
    @test_throws ArgumentError OverlayFactor(:a=>f; normalise=:missing)
    @test_throws ArgumentError OverlayFactor(:a=>f; normalise=:k)
    @test_throws DimensionMismatch OverlayFactor(:a=>f,:b=>P36Fixed([0.,0.],Matrix(1.0I,2,2)))
    # No switches and no varying coefficients: separate static path.
    o = OverlayFactor(:a=>prior,:b=>prior; normalise=:y)
    F = cluster_form(o,(a=nt,b=nt))
    @test cluster_switch(o) === nothing
    @test F.C ≈ fill(0.2,1,1) && F.Q ≈ fill(1.3/2,1,1)
    product = OverlayFactor(:a=>prior,:b=>prior)
    F = cluster_form(product,(a=nt,b=nt))
    @test F.C ≈ fill(0.2,2,1) && F.Q ≈ Diagonal([1.3,1.3])
    discrete = OverlayFactor(:a=>TableFactor((k=2,),log.([0.2,0.8])))
    F = cluster_form(discrete,(a=nt,))
    @test isempty(F.A) && size(F.Q) == (0,0)
    @test F.logw ≈ log.([0.2,0.8])
    # Conditional product with a non-normalised channel: coefficients and missing-channel padding.
    cond = OverlayFactor(:affine=>P36Affine{false}(0.4), :prior=>prior; normalise=:y)
    ps = (affine=(a=[1.2],b=[0.3]),prior=nt)
    F = cluster_form(cond,ps)
    P = 1/0.4+1/1.3
    for x in (-0.7,0.4,1.2)
        μ = ((1.2x+0.3)/0.4+0.2/1.3)/P
        @test F.C[:,1]-F.A.x*[x] ≈ [μ]
        @test F.Q ≈ fill(1/P,1,1)
    end
    singular = OverlayFactor(:affine=>P36Affine{false}(0.4);normalise=:x)
    @test_throws ArgumentError cluster_form(singular,(affine=(a=[0.],b=[0.3]),))


end

@testset "P36 cluster rrule: whole-energy derivatives and zero tangents" begin
    c = as_cluster(P36Switched()); nt=(;)
    msgs=(y=Gaussian(0.6,1.2),k=CategoricalBelief([0.4,0.6]))
    p = LuxCore.initialparameters(Xoshiro(36), P36Switched())
    θ = vcat(p.mu,p.q,p.logw)
    ps(v)=(factor=(mu=v[1:2],q=v[3:4],logw=v[5:6]),)
    for ws in ((1.,1.,1.), (0.3,-0.7,1.2))
        energy(v) = (E=first(local_free_energy(c,msgs,ps(v),nt)); sum(w*e for (w,e) in zip(ws,values(E.parts))))
        fd = ForwardDiff.gradient(energy,θ)
        cd = [(energy(θ+1e-6*(1:6 .== j))-energy(θ-1e-6*(1:6 .== j)))/2e-6 for j in 1:6]
        @test fd ≈ cd rtol=2e-5 atol=2e-7
        @test only(Zygote.gradient(energy,θ)) ≈ fd rtol=1e-10 atol=1e-10
    end
    _, pb = P36CRC.rrule(Mycelium._cluster_local_energy,c,msgs,ps(θ),nt)
    for Δ in (P36CRC.ZeroTangent(), (nothing,nothing), (P36CRC.ZeroTangent(),nothing))
        tangent = pb(Δ)
        @test tangent[4] isa P36CRC.ZeroTangent
        @test tangent[3] isa P36CRC.NotImplemented
    end
    # A numeric cotangent has no graded weights, hence zero parameter derivative.
    @test all(iszero,LenticulumCore._flatten(pb((1.0,nothing))[4]))
    @test Base.get_extension(Mycelium,:MyceliumForwardDiffExt)._part(nothing,:fit) == 0
    emptyc=as_cluster(P36Fixed([0.],ones(1,1)))
    y,pb0=P36CRC.rrule(Mycelium._cluster_local_energy,emptyc,(y=Gaussian(0.2,0.8),),(factor=nt,),nt)
    @test pb0((P36CRC.Tangent{typeof(first(y))}(;parts=(fit=1.,complexity=0.,negentropy=0.)),nothing))[4] isa P36CRC.ZeroTangent
end

@testset "P36 flat precision: constant-rank pseudoinverse and pseudodeterminant" begin
    # J(t)=a(t)u(t)u(t)' is rank one, with a moving range and null space.
    J(t) = (1.7+exp(t))*([cos(t),sin(t)]*[cos(t),sin(t)]')
    h=[0.4,-0.9]
    quantity(t) = (F=Mycelium._psd_factor(J(t)); Mycelium._psd_logdet(F)+dot(h,Mycelium._psd_inv(F)*h))
    closed(t) = log(1.7+exp(t))+(dot(h,[cos(t),sin(t)]))^2/(1.7+exp(t))
    for t in (-0.6,0.,0.4)
        @test quantity(t) ≈ closed(t)
        @test ForwardDiff.derivative(quantity,t) ≈ ForwardDiff.derivative(closed,t) atol=1e-10
        @test ForwardDiff.derivative(quantity,t) ≈ (quantity(t+1e-5)-quantity(t-1e-5))/2e-5 atol=1e-8
    end
    @test_throws ArgumentError ForwardDiff.derivative(t -> Mycelium._psd_logdet(Mycelium._psd_factor([1.0 zero(t);zero(t) t])),0.)
    d=ForwardDiff.Dual(2.3,1.)
    @test Mycelium._primal(d) === 2.3
    # All-flat constant family and the direct definite fallback of the extension.
    @test ForwardDiff.derivative(t -> Mycelium._psd_logdet(Mycelium._psd_factor(fill(zero(t),2,2))),0.) == 0
    @test ForwardDiff.derivative(t -> (A=fill(t,1,1); C=cholesky(Symmetric(A)); logdet(Mycelium._flat_factor(A,C))),2.) ≈ 0.5
end

# LP-067 (review of P36): joint outcomes of several switches, summed log-weights, and the message to
# a continuous channel, each against the model written out by hand.
struct P36TwoSwitch <: AbstractLenticulumFactor end       # y - mu[k1,k2], one member with two switches
LenticulumCore.channels(::P36TwoSwitch) = (LenticulumCore.Channel(:y,1), LenticulumCore.Channel(:k1,2), LenticulumCore.Channel(:k2,3))
Mycelium.cluster_switch(::P36TwoSwitch) = (:k1, :k2)
LuxCore.parameterlength(::P36TwoSwitch) = 0
LuxCore.initialparameters(::AbstractRNG, ::P36TwoSwitch) = (;)
const P36_MU2 = [0.1 -0.5 0.9; 0.3 1.4 -1.1]                 # mu[k1, k2]
const P36_W2 = [0.05 0.2 0.1; 0.15 0.3 0.2]
Mycelium.cluster_form(::P36TwoSwitch, ps) =
    ClusterForm((y=ones(1,1),), reshape(vec(P36_MU2),1,6), fill(0.6,1,1,6), log.(vec(P36_W2)), (:k1,:k2))

@testset "P36 overlay: joint outcomes of several switches, summed weights, continuous messages" begin
    nt = (;); f = P36Switched(); p = LuxCore.initialparameters(Xoshiro(36), f)
    two = P36TwoSwitch()
    # The member's own outcomes, column-major in (k1, k2): the overlay of one member keeps them.
    F1 = cluster_form(OverlayFactor(:t=>two), (t=nt,))
    @test cluster_switch(OverlayFactor(:t=>two)) == (:k1,:k2)
    for k2 in 1:3, k1 in 1:2
        j = k1 + 2(k2-1)
        @test F1.C[:,j] ≈ [P36_MU2[k1,k2]] && F1.logw[j] ≈ log(P36_W2[k1,k2])
    end
    # Two members with switches of their own: outcome j = (k1, k2, k) column-major in the order
    # (t's switches, then sw's); log-weights add, residuals stack.
    o = OverlayFactor(:t=>two, :sw=>f)
    @test cluster_switch(o) == (:k1,:k2,:k)
    F = cluster_form(o, (t=nt, sw=p))
    @test size(F.C) == (2, 12)
    for k in 1:2, k2 in 1:3, k1 in 1:2
        j = k1 + 2(k2-1) + 6(k-1)
        @test F.logw[j] ≈ log(P36_W2[k1,k2]) + p.logw[k]
        @test F.C[:,j] ≈ [P36_MU2[k1,k2], p.mu[k]]
        @test F.Q[:,:,j] ≈ Diagonal([0.6, p.q[k]])
    end
    # The message to the switch k of `o` with y observed, k1 and k2 latent: by enumeration.
    y = 0.4
    joint = [P36_W2[k1,k2]*exp(p.logw[k]) * exp(-(y-P36_MU2[k1,k2])^2/1.2)/sqrt(2π*0.6) *
             exp(-(y-p.mu[k])^2/(2p.q[k]))/sqrt(2π*p.q[k]) for k1 in 1:2, k2 in 1:3, k in 1:2]
    pol = Polarity(y=Observed(), k1=Latent(), k2=Latent(), k=Unobserved())
    msgs = (y=DiracBelief([y]), k1=CategoricalBelief([0.5,0.5]), k2=CategoricalBelief(fill(1/3,3)))
    m, _ = factor_message(o, :k, pol, msgs, nothing, (t=nt, sw=p), nt)
    @test probabilities(m) ≈ vec(sum(joint; dims=(1,2))) / sum(joint)
    # The message to the continuous channel y under soft switch messages: a mixture over the joint
    # outcomes, component j the product of the selected Gaussians (precision 1/0.6 + 1/q_k), weight
    # π1[k1] π2[k2] π[k] w[k1,k2] w_k times the product's normaliser N(mu[k1,k2]; mu_k, 0.6 + q_k).
    π1, π2, π3 = [0.3, 0.7], [0.5, 0.2, 0.3], [0.6, 0.4]
    pol = Polarity(y=Unobserved(), k1=Latent(), k2=Latent(), k=Latent())
    m, _ = factor_message(o, :y, pol, (k1=CategoricalBelief(π1), k2=CategoricalBelief(π2), k=CategoricalBelief(π3)),
                          nothing, (t=nt, sw=p), nt)
    @test m isa MixtureBelief && length(m.components) == 12
    w = zeros(12); μ = zeros(12); v = zeros(12)
    for k in 1:2, k2 in 1:3, k1 in 1:2
        j = k1 + 2(k2-1) + 6(k-1); P = 1/0.6 + 1/p.q[k]; s2 = 0.6 + p.q[k]
        w[j] = π1[k1]*π2[k2]*π3[k]*P36_W2[k1,k2]*exp(p.logw[k]) * exp(-(P36_MU2[k1,k2]-p.mu[k])^2/(2s2))/sqrt(2π*s2)
        μ[j] = (P36_MU2[k1,k2]/0.6 + p.mu[k]/p.q[k])/P; v[j] = 1/P
    end
    # The message's components follow the cluster's own outcome order, so compare as a set.
    got = sort([(only(belief_mean(c)), only(belief_cov(c)), wc) for (c, wc) in zip(m.components, mixture_weights(m))])
    ref = sort([(μ[j], v[j], w[j] / sum(w)) for j in 1:12])
    @test all(all(isapprox.(a, b; rtol=1e-8)) for (a, b) in zip(got, ref))
end
