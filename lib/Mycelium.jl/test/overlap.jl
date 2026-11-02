using LinearAlgebra

@testset "Public combine_with_overlap" begin
    for (p,q) in (([0.2,0.3,0.5],[0.6,0.1,0.3]), ([1.,0,0],[0.,0.4,0.6]))
        c,d = CategoricalBelief(p),CategoricalBelief(q)
        pooled,logZ = combine_with_overlap(c,d)
        z = dot(p,q)
        @test logZ ≈ log(z)
        if z>0
            @test probabilities(pooled) ≈ p.*q/z
        else
            @test logZ == -Inf
            @test pooled === c # explicitly documented placeholder for zero mass
        end
    end
    μ,ν = [0.4,-0.7],[-0.3,1.2]
    S,T = [1.3 0.2;0.2 0.7],[0.5 -0.1;-0.1 1.1]
    c,d = Gaussian(μ,S),Gaussian(ν,T)
    pooled,logZ = combine_with_overlap(c,d)
    C = S+T
    expected = -(2log(2π)+logdet(C)+dot(μ-ν,C\(μ-ν)))/2
    @test logZ ≈ expected atol=2e-15
    @test belief_cov(pooled) ≈ inv(inv(S)+inv(T))
    @test belief_mean(pooled) ≈ belief_cov(pooled)*(S\μ+T\ν)
    @test last(combine_with_overlap(d,c)) ≈ logZ
    # Improper likelihood exp(b*x₁-a*x₁²/2), constant along x₂.
    a,b = 1.7,0.3
    improper = GaussianBelief([b,0.],[a 0.;0. 0.])
    cs = (c,Gaussian(ν,T))
    zs = [last(combine_with_overlap(ci,improper)) for ci in cs]
    closed = [-log(2π*(V[1,1]+1/a))/2-(m[1]-b/a)^2/(2*(V[1,1]+1/a))
              for (m,V) in ((μ,S),(ν,T))]
    @test zs[1]-zs[2] ≈ closed[1]-closed[2] atol=2e-15
    @test last(combine_with_overlap(c,uninformative(2))) ≈ 0 atol=2e-15
    @test_throws ArgumentError combine_with_overlap(improper,c)
    @test_throws DimensionMismatch combine_with_overlap(c,Gaussian([0.],ones(1,1)))
end

@testset "A declared-improper Gaussian is improper whatever the rounding" begin
    # rank one in the plane, with rounding that makes it factorise
    v = [1.0, 1e-9]
    Λ = v * v' + 1e-14 * I
    @test isposdef(Symmetric(Λ))
    site = GaussianBelief(Λ * [0.3, 0.0], Λ, true)
    @test !isproper(site) && isproper(GaussianBelief(site.η, site.Λ))
    @test_throws ArgumentError belief_mean(site)
    # pooling keeps the flag only beside zero precision
    @test combine(site, uninformative(2)).improper
    @test !combine(site, GaussianBelief(zeros(2), Matrix(1.0I, 2, 2))).improper
    @test isproper(combine(site, Gaussian([0.0, 0.0], Matrix(1.0I, 2, 2))))
    # the mixture product weights a declared site as a likelihood: the overlap with a proper c
    # differs from ∫ p_c exp(ηᵀx − xᵀΛx/2) by a constant of the site alone, so the difference
    # between two sites with the same Λ is the exact log ratio
    c = Gaussian([0.2, -0.4], [0.8 0.1; 0.1 0.6])
    s1, s2 = GaussianBelief(Λ * [0.3, 0.0], Λ, true), GaussianBelief(Λ * [-0.5, 0.0], Λ, true)
    l(s) = last(combine_with_overlap(c, s))
    μc, Σc = belief_mean(c), belief_cov(c)
    exact(s) = (P = inv(Σc) + Λ; h = Σc \ μc + s.η; dot(h, P \ h) / 2 - logdet(P) / 2)   # up to the c-only constant
    @test l(s1) - l(s2) ≈ exact(s1) - exact(s2) rtol = 1e-8
end
