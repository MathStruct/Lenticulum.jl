# P4 (Feature Backlog): derivatives through extended Gaussians whose subspaces move.
#
# A belief b(t) = pushforward(b₀, M(t)) with M(t) = I + tK invertible keeps the number of flat,
# hard and Gaussian directions and turns all of them with t. The operations of
# `extended_gaussian.jl` (product, preimage, conversion from an improper Gaussian, the support
# entropy, the hard directions, the canonical part) are differentiated by ForwardDiff and Zygote
# and compared with central differences of the float path. Outputs are compared through
# basis-free quantities: μ, Σ and the projector DDᵀ (a basis of a subspace has no unique
# derivative). A tangent that changes a rank at first order is refused under ForwardDiff.

import Zygote
using ForwardDiff, Random, LinearAlgebra, Test, LenticulumCore
using LenticulumCore: _conjoin, _canonical_part, pushforward

module EGAD
using LinearAlgebra, Random, LenticulumCore
spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))
# kf flat, kh hard and n − kf − kh Gaussian directions, in a random basis
function random_eg(rng, n, kf, kh)
    B = Matrix(qr(randn(rng, n, n)).Q)
    V = B[:, (kf + kh + 1):n]
    Σ = V * spd(rng, size(V, 2)) * V'
    return ExtendedGaussian(randn(rng, n), (Σ + Σ') / 2, B[:, 1:kf])
end
# a basis-free summary of an extended Gaussian, as one vector
summary(e) = vcat(vec(e.μ), vec(e.Σ), vec(e.D * e.D'))
# central differences of a vector-valued function of a scalar
fd(f, t; h = 1e-6) = (f(t + h) .- f(t - h)) ./ (2h)
end

@testset "P4: derivatives of extended Gaussians whose subspaces move" begin
    X = EGAD
    rng = Xoshiro(404)
    ops = Dict{Symbol,Int}()
    for trial in 1:30
        n = rand(rng, 2:4)
        kf, kh = rand(rng, 0:(n - 1)), 0
        kh = rand(rng, 0:(n - kf - 1))
        a0 = X.random_eg(rng, n, kf, kh)
        # b: hard only along directions a leaves Gaussian or flat is not needed; keep b proper or
        # with one hard direction, so that the product is consistent
        b0 = X.random_eg(rng, n, 0, rand(rng, 0:1))
        K1, K2 = randn(rng, n, n) ./ 3, randn(rng, n, n) ./ 3
        a(t) = pushforward(a0, I + t * K1)
        b(t) = pushforward(b0, I + t * K2)
        M = randn(rng, n, n)
        t0 = 0.1
        cases = Pair{Symbol,Function}[
            :pushforward => t -> X.summary(a(t)),
            :preimage => t -> X.summary(preimage(a(t), M + t * K2)),
            :hard_projector => t -> (H = hard_directions(a(t)); vec(H * H')),
            :canonical => t -> (ηΛ = _canonical_part(a(t)); vcat(ηΛ[1], vec(ηΛ[2]))),
        ]
        # a product needs consistent hard constraints: b's hard value is moved onto a's support
        # only when b has none, so take the product with a proper b
        if size(hard_directions(b0), 2) == 0
            push!(cases, :product => t -> X.summary(_conjoin(a(t), b(t))))
            # the entropy of a product that is flat nowhere: a without flat directions
            if kf == 0
                push!(cases, :support_entropy => t -> [support_entropy(_conjoin(a(t), b(t)))])
            end
        end
        # an improper Gaussian whose null space turns: Λ(t) = (I + tK)ᵀ Λ₀ (I + tK)
        if kh == 0 && kf > 0
            η0, Λ0 = _canonical_part(a0)
            push!(cases, :extended => t -> X.summary(extended(GaussianBelief(
                (I + t * K1)' * η0, Matrix(Symmetric((I + t * K1)' * Λ0 * (I + t * K1))), true))))
        end
        for (name, f) in cases
            ref = X.fd(f, t0)
            dF = ForwardDiff.derivative(f, t0)
            @test dF ≈ ref atol = 1e-6 rtol = 1e-5
            w = randn(rng, length(ref))
            dZ = Zygote.gradient(t -> w' * f(t), t0)[1]
            @test dZ ≈ w' * ref atol = 1e-6 rtol = 1e-5
            ops[name] = get(ops, name, 0) + 1
        end
    end
    # every operation was exercised several times
    @test all(k -> get(ops, k, 0) >= 3, (:pushforward, :preimage, :hard_projector, :canonical,
                                          :product, :support_entropy, :extended))
end

# Review 2026-11-16 (LP-025): with no Gaussian direction (Σ = 0 on D⊥) the split of `hard_directions`
# has no nonzero eigenvalue, and Zygote's pullback of the broadcasts by the empty λ₀ threw a
# DimensionMismatch. The loop above draws at least one Gaussian direction, so it missed the case.
@testset "P4: no Gaussian direction (all hard or flat)" begin
    rng = Xoshiro(7)
    for (n, kf, kh) in ((2, 0, 2), (3, 1, 2), (3, 0, 3), (2, 2, 0), (3, 3, 0), (1, 0, 1))
        a0 = EGAD.random_eg(rng, n, kf, kh)
        K = randn(rng, n, n) ./ 3
        a(t) = pushforward(a0, I + t * K)
        for f in (t -> (H = hard_directions(a(t)); vec(H * H')), t -> EGAD.summary(a(t)),
                  t -> (ηΛ = _canonical_part(a(t)); vcat(ηΛ[1], vec(ηΛ[2]))))
            ref = EGAD.fd(f, 0.1)
            @test ForwardDiff.derivative(f, 0.1) ≈ ref atol = 1e-6
            w = randn(rng, length(ref))
            @test something(Zygote.gradient(t -> w' * f(t), 0.1)[1], 0.0) ≈ w' * ref atol = 1e-6
        end
    end
end

@testset "P4: floats are unchanged by the constant-rank formulas" begin
    rng = Xoshiro(405)
    for _ in 1:20
        n = rand(rng, 2:4)
        e = EGAD.random_eg(rng, n, rand(rng, 0:1), rand(rng, 0:1))
        @test eltype(e.μ) === Float64 && eltype(hard_directions(e)) === Float64
        H = hard_directions(e)
        @test H' * H ≈ I(size(H, 2))
        @test norm(e.Σ * H) <= 1e-12 && norm(e.D' * H) <= 1e-12
    end
end

@testset "P4: a tangent that changes a rank at first order is refused (ForwardDiff)" begin
    # Λ(t) = diag(1, t²... ) is not first order; diag(1, t) at t = 0 gains rank at first order
    f(t) = extended(GaussianBelief([1.0, 0.0] .+ 0 * t, [1.0 0.0; 0.0 t], true)).μ
    @test_throws ArgumentError ForwardDiff.derivative(f, 0.0)
    # a hard direction that becomes Gaussian: Σ(t) = diag(1, t)
    g(t) = support_entropy(ExtendedGaussian([0.0, 0.0], [1.0 0.0; 0.0 t]))
    @test_throws ArgumentError ForwardDiff.derivative(g, 0.0)
    # a tangent that only turns the null space is accepted: Λ(t) = R(t) diag(1, 0) R(t)ᵀ
    R(t) = [cos(t) -sin(t); sin(t) cos(t)]
    h(t) = (e = extended(GaussianBelief(R(t) * [1.0, 0.0], R(t) * [1.0 0.0; 0.0 0.0] * R(t)', true)); vec(e.D * e.D'))
    @test ForwardDiff.derivative(h, 0.3) ≈ EGAD.fd(h, 0.3) atol = 1e-7
    # nested duals (second derivatives) are refused, not silently wrong
    @test_throws ArgumentError ForwardDiff.derivative(s -> ForwardDiff.derivative(h, s)[1], 0.3)
end
