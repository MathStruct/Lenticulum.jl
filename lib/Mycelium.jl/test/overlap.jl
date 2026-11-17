using LinearAlgebra, Random

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

# P27: a Gaussian component times an extended Gaussian d (hard, flat and Gaussian directions), the
# rule `_mixture_product` uses when a mixture message meets a hard relation's message. Oracle in
# closed form, independent of the reduction the rule uses: d is the law of x = μ_d + Gz + Ft with
# Σ_d = GGᵀ, z standard normal and t Lebesgue along the flat basis F, and
#     log ∫ c(x) dd = ηᵀμ_d − μ_dᵀΛμ_d/2 + bᵀM⁻¹b/2 − log det M/2 + (f/2) log 2π − A(c),
# with M = diag(I, 0) + JᵀΛJ, J = [G F], b = Jᵀ(η − Λμ_d), (η, Λ) the canonical parameters of c and
# A(c) its log-partition (0 for an improper c, an unnormalised exponential). For d flat nowhere the
# value is compared exactly; with flat directions it is defined up to a constant of d alone, and the
# differences across components are compared.
@testset "P27: combine_with_overlap of a Gaussian and an extended Gaussian" begin
    rng = Random.Xoshiro(27)
    spd(n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))
    function oracle(c, d)
        E = eigen(Symmetric(d.Σ)); keep = E.values .> 1e-10
        G = E.vectors[:, keep] * Diagonal(sqrt.(E.values[keep]))
        J = hcat(G, d.D); g, f = size(G, 2), size(d.D, 2)
        Λ = Matrix(Symmetric(c.Λ)); μ = d.μ
        M = Symmetric(Diagonal(vcat(ones(g), zeros(f))) + J' * Λ * J); b = J' * (c.η - Λ * μ)
        A = LenticulumCore.isproper(c) ?
            (dot(c.η, Λ \ c.η) - logdet(Λ) + length(μ) * log(2π)) / 2 : 0.0
        return dot(c.η, μ) - dot(μ, Λ * μ) / 2 + (isempty(b) ? 0.0 : dot(b, M \ b) / 2 - logdet(M) / 2) + f * log(2π) / 2 - A
    end
    worst = 0.0
    for trial in 1:60
        n = rand(rng, 2:4); k = rand(rng, 1:n); f = rand(rng, 0:(n - k))
        Q = Matrix(qr(randn(rng, n, n)).Q)
        Hd, Fd, Gd = Q[:, 1:k], Q[:, (k + 1):(k + f)], Q[:, (k + f + 1):end]
        d = ExtendedGaussian(randn(rng, n), Gd * spd(size(Gd, 2)) * Gd', Fd)
        @test size(LenticulumCore.hard_directions(d), 2) == k
        cs = if isodd(trial)
            [Gaussian(randn(rng, n), spd(n)) for _ in 1:3]
        else                                # likelihoods: rank-deficient, declared improper
            map(1:3) do _
                B = randn(rng, n - 1, n); Λ = B' * B
                GaussianBelief(Λ * randn(rng, n), Λ, true)
            end
        end
        lz = [last(combine_with_overlap(c, d)) for c in cs]
        @test [last(combine_with_overlap(d, c)) for c in cs] == lz
        lo = [oracle(c, d) for c in cs]
        err = f == 0 ? maximum(abs.(lz .- lo)) : maximum(abs.((lz .- lz[1]) .- (lo .- lo[1])))
        @test err < 1e-9
        worst = max(worst, err)
        # the product is the extended product
        for c in cs
            p = first(combine_with_overlap(c, d)); q = combine(c, d)
            @test p.μ ≈ q.μ && p.Σ ≈ q.Σ
        end
    end
    # far from the origin: a translation by 1e6 of both sides leaves logZ unchanged (review of P27:
    # its reference lost digits there, the rule does not)
    for _ in 1:20
        n = 3; Q = Matrix(qr(randn(rng, n, n)).Q); G = Q[:, 2:3]; t = fill(1e6, n)
        M = randn(rng, n, n); Σc = M * M' + 0.5I; μc, μd = randn(rng, n), randn(rng, n); Σd = G * spd(2) * G'
        near = last(combine_with_overlap(Gaussian(μc, Σc), ExtendedGaussian(μd, Σd)))
        @test last(combine_with_overlap(Gaussian(μc + t, Σc), ExtendedGaussian(μd + t, Σd))) ≈ near atol = 1e-7
    end
    # every direction hard: the density of c at the point
    c = Gaussian([0.3, -0.2], [1.0 0.2; 0.2 0.5]); x = [1.1, 0.4]
    @test last(combine_with_overlap(c, ExtendedGaussian(x, zeros(2, 2)))) ≈
          -(2log(2π) + logdet(belief_cov(c)) + dot(x - belief_mean(c), belief_cov(c) \ (x - belief_mean(c)))) / 2
    # two extended Gaussians with hard directions: refused; one without: the Gaussian rule
    h = ExtendedGaussian([1.0, 0.0], [1.0 0.0; 0.0 0.0])
    @test_throws ArgumentError combine_with_overlap(h, h)
    s = ExtendedGaussian([0.5, 0.1], [2.0 0.3; 0.3 1.0])
    @test last(combine_with_overlap(s, h)) ≈ last(combine_with_overlap(GaussianBelief(s), h))
end

# P27, expectation propagation with an extended cavity: the site p / cavity is a Gaussian on the
# cavity's support, possibly indefinite, and its product with the cavity gives p back (through the
# product of potentials, `_potential_conjoin`, when the site has a negative direction). Oracle: p
# itself, and for the product of an indefinite Gaussian with an extended one, the dense solve on the
# support x = x₀ + N t.
@testset "P27: expectation propagation with an extended cavity" begin
    rng = Random.Xoshiro(127)
    spd(n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))
    nindef = 0
    for trial in 1:40
        n = rand(rng, 2:4); k = rand(rng, 1:(n - 1)); f = rand(rng, 0:(n - k - 1))
        Q = Matrix(qr(randn(rng, n, n)).Q)
        Hd, Fd, Gd = Q[:, 1:k], Q[:, (k + 1):(k + f)], Q[:, (k + f + 1):end]
        cav = ExtendedGaussian(randn(rng, n), Gd * spd(size(Gd, 2)) * Gd', Fd)
        m = MixtureBelief([Gaussian(3 .* randn(rng, n), spd(n)) for _ in 1:3], normalize(rand(rng, 3), 1))
        tilted = combine(cav, m)
        @test tilted isa MixtureBelief{<:ExtendedGaussian}
        p = ep_project(tilted)
        # the projection: total moments, the cavity's hard directions kept
        w = LenticulumCore.mixture_weights(tilted); w = w ./ sum(w)
        μ = sum(w[j] .* tilted.components[j].μ for j in eachindex(w))
        @test p.μ ≈ μ
        @test norm(Hd' * p.Σ * Hd) < 1e-10
        site = ep_divide(p, cav)
        @test site isa GaussianBelief
        Mycelium._indefinite(site) && (nindef += 1)
        back = combine(site, cav)
        @test back.μ ≈ p.μ rtol = 1e-7 atol = 1e-9
        @test back.Σ ≈ p.Σ rtol = 1e-7 atol = 1e-9
        @test combine(cav, site).μ ≈ back.μ
    end
    @test nindef > 0                                         # the indefinite path was reached
    # the same hard direction at another value: no site (review of P27, F1; a site came back)
    Wf = reshape([0.0, 1.0], 2, 1)
    @test ep_divide(ExtendedGaussian([0.3, 1.0], 0.5 .* Wf * Wf'), ExtendedGaussian([0.9, 2.0], 0.7 .* Wf * Wf')) === nothing
    # an indefinite Gaussian times a point-like relation: the dense solve on the support
    H = [1.0, 1.0] / sqrt(2); b = ExtendedGaussian([0.5, 0.5], [0.5 -0.5; -0.5 0.5])   # hard along H
    a = GaussianBelief([0.3, -0.1], [2.0 0.0; 0.0 -0.5], true)
    r = combine(a, b)
    N = [1.0, -1.0] / sqrt(2); x0 = H * (H' * b.μ)
    η, Λ = Mycelium.LenticulumCore._canonical_part(b)
    P = N' * (a.Λ + Λ) * N; h = N' * (a.η + η - (a.Λ + Λ) * x0)
    @test r.μ ≈ x0 + N * (h / P)
    @test r.Σ ≈ N * N' / P
    # negative along what the relation leaves free: not normalisable
    @test_throws ArgumentError combine(GaussianBelief([0.0, 0.0], [-3.0 0.0; 0.0 -3.0], true), b)
end
