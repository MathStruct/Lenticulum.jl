# Zygote through the cavity free energies of `LinearConstraintFactor` and `SwitchingConstraintFactor`
# (P23): the derivative in `ps` at fixed messages, on random instances, against ForwardDiff. Both
# failed under Zygote before: the n-ary block diagonal was filled in place, the reduced relation and
# the switching factor's free channels were collected by `push!`, and `inv(::Cholesky)` has no rule.
# The messages cover every kind the reduced relation treats apart: proper Gaussians, points
# (substituted into the constant), extended Gaussians with hard directions (reduced to their other
# directions) and Gaussian mixtures (collapsed to their moments).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random
using ForwardDiff, Zygote
using LenticulumCore: MixtureBelief, CategoricalBelief, DiracBelief, TrivialBelief

zc_spd(rng, n) = (B = randn(rng, n, n); B * B' + n * I)
zc_total(e) = e[:fit] + e[:complexity] + e[:negentropy]
function zc_message(rng, n, kind)
    kind === :gauss && return GaussianBelief(randn(rng, n), zc_spd(rng, n))
    kind === :dirac && return DiracBelief(randn(rng, n))
    kind === :mixture && return MixtureBelief([Gaussian(randn(rng, n), Matrix(zc_spd(rng, n))) for _ in 1:2], [0.3, 0.7])
    # :hard, an extended Gaussian with kh ≥ 1 directions of zero variance and no flat ones
    kh = rand(rng, 1:n)
    B = Matrix(qr(randn(rng, n, n)).Q)
    V = B[:, (kh + 1):n]
    Σ = V * zc_spd(rng, size(V, 2)) * V'
    return ExtendedGaussian(randn(rng, n), (Σ + Σ') / 2, zeros(n, 0))
end
# the free energy of `f` at fixed messages, as a function of the flat parameter vector
zc_energy(f, msgs, ps) = θ -> zc_total(first(Mycelium.local_free_energy(f, msgs, LenticulumCore._unflatten(θ, ps), NamedTuple())))

@testset "Zygote through cavity free energies (P23)" begin
    @testset "_spd_inv: its rrule against ForwardDiff, a non-symmetric cotangent" begin
        rng = Xoshiro(1)
        for n in (1, 2, 4)
            M = zc_spd(rng, n); W = randn(rng, n, n)
            h(θ) = sum(W .* Lenticulum._spd_inv(reshape(θ, n, n), "M"))
            @test Lenticulum._spd_inv(M, "M") ≈ inv(M)
            @test only(Zygote.gradient(h, vec(M))) ≈ ForwardDiff.gradient(h, vec(M)) atol = 1e-10
        end
        @test_throws ArgumentError Lenticulum._spd_inv([1.0 0; 0 -1], "M")
    end

    @testset "LinearConstraintFactor, random messages" begin
        rng = Xoshiro(2)
        kinds = (:gauss, :dirac, :hard, :mixture)
        for trial in 1:20
            dims = (x = rand(rng, 1:3), y = rand(rng, 1:3), z = rand(rng, 1:3))
            m = rand(rng, 1:3)
            f = LinearConstraintFactor(dims, m; noise = zc_spd(rng, m))
            ps0 = LuxCore.initialparameters(rng, f)
            θ0 = LenticulumCore._flatten(ps0) .+ randn(rng, length(LenticulumCore._flatten(ps0)))
            msgs = NamedTuple{keys(dims)}(Tuple(zc_message(rng, d, rand(rng, kinds)) for d in values(dims)))
            F = zc_energy(f, msgs, ps0)
            @test only(Zygote.gradient(F, θ0)) ≈ ForwardDiff.gradient(F, θ0) rtol = 1e-8 atol = 1e-8
        end
    end

    @testset "SwitchingConstraintFactor (EP sites), random messages" begin
        rng = Xoshiro(3)
        kinds = (:gauss, :dirac, :mixture)
        for trial in 1:20
            dims = (x = rand(rng, 1:3), z = rand(rng, 1:3))
            K, m = rand(rng, 2:4), rand(rng, 1:3)
            f = SwitchingConstraintFactor(dims, K, m; noise = zc_spd(rng, m))
            ps0 = LuxCore.initialparameters(rng, f)
            θ0 = LenticulumCore._flatten(ps0) .+ 0.5 .* randn(rng, length(LenticulumCore._flatten(ps0)))
            kmsg = rand(rng, Bool) ? CategoricalBelief(rand(rng, K) .+ 0.1) : TrivialBelief()
            msgs = (x = zc_message(rng, dims.x, rand(rng, kinds)), z = zc_message(rng, dims.z, rand(rng, kinds)), k = kmsg)
            F = zc_energy(f, msgs, ps0)
            @test only(Zygote.gradient(F, θ0)) ≈ ForwardDiff.gradient(F, θ0) rtol = 1e-8 atol = 1e-8
        end
    end
end
