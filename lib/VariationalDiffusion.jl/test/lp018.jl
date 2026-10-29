using VariationalDiffusion, LenticulumCore, Mycelium, LuxCore, Random, LinearAlgebra, Test
using ADTypes: AutoZygote
using DifferentiationInterface
import Zygote

struct LP018Energy <: LuxCore.AbstractLuxLayer
    A::Matrix{Float64}
end
(l::LP018Energy)(inp, ps, st) = (sum(inp[1] .* (l.A * inp[1]); dims = 1) ./ 2, st)
struct LP018Times <: LuxCore.AbstractLuxLayer end
(::LP018Times)(inp, ps, st) = (inp[1] .* reshape(inp[2], size(inp[1])), st)

@testset "LP-018 diffusion regressions" begin
@testset "LP-018.2 calibration under uniform time measure" begin
    # Composite midpoint reference, independent of the trapezoidal implementation;
    # integrate the antithetic noise pair exactly for a unit Gaussian.
    for tmin in (1e-3, 0.5)
        s = VPSDE(; tmin)
        B(t) = 0.1t + 9.95t^2
        ts = tmin .+ (1-tmin) .* ((0:19999) .+ 0.5) ./ 20000
        avg = sum(-expm1(-B(t)) for t in ts) / length(ts)
        λ = calibrate_lambda(s)
        @test λ * avg ≈ 1 rtol = 2e-8
        @test 1 / (1 + λ * avg) ≈ 0.5 atol = 5e-9
    end
end
@testset "LP-018.3 energy block Jacobians weight each column" begin
    s = VPSDE()
    X = [0.2 0.4 -0.1; 0.3 -0.2 0.6]
    T = [0.1 0.2 0.3; 0.4 0.5 0.6]
    w = [0.2, 0.3, 0.5]
    for A in (Matrix{Float64}(I, 2, 2), [1. 0.3; 0.3 2.]), ref in (nothing, GaussianReference([0., 0.], [0.7, 1.3]))
        p = NoisePredictor(EnergyNetwork(LP018Energy(A); reference = ref), s; blocks = [[1], [2]], ad = AutoZygote())
        Js = [Diagonal(sigma.(Ref(s), T[:, b])) *
              (A + (ref === nothing ? zeros(2, 2) : Diagonal(1 ./ (alpha.(Ref(s), T[:, b]).^2 .* ref.v .+ sigma.(Ref(s), T[:, b]).^2)))) for b in 1:3]
        for idx in ([1, 2], [2], [2, 1])
            expected = sum(w[b] .* Js[b][idx, idx] for b in 1:3)
            @test epsilon_jacobian_block(p, X, T, (;), (;), idx, w) ≈ expected atol = 1e-12
        end
        @test epsilon_vjp_input(p, X, T, (;), (;), X) ≈ hcat([Js[b]' * X[:, b] for b in 1:3]...) atol = 1e-12
        @test epsilon_jvp_input(p, X, T, (;), (;), X) ≈ hcat([Js[b] * X[:, b] for b in 1:3]...) atol = 1e-12
    end
end
@testset "LP-018.4 generic batched VJP column times" begin
    p = NoisePredictor(LP018Times(), VPSDE(); blocks = [[1], [2]])
    X = [1. 2. 3.; 4. 5. 6.]
    T = [0.1 0.2 0.3; 0.4 0.5 0.6]
    for times in (T, 0.2)
        @test epsilon_vjp_input(p, X, times, (;), (;), ones(2, 3)) ≈ (times isa Number ? fill(times, 2, 3) : T) atol = 1e-9
    end
end
@testset "LP-018.5 hard clamp pointwise energy" begin
    s = VPSDE()
    pred = NoisePredictor(GaussianMixtureEps(s, zeros(2, 1); s = 1.), s)
    f = DiffusionFactor((x = 1, y = 1), pred; prox = ImplicitProx(noisefree_nodes(2, 0.1)))
    ps, st = LuxCore.setup(Xoshiro(2), f)
    p = Polarity((x = Observed(), y = Unobserved()))
    @test first(LenticulumCore.energy(f, [1., 0.], p, [0., 0.], ps, st))[:clamp] == Inf
    @test first(LenticulumCore.energy(f, [0., 0.], p, [0., 0.], ps, st))[:clamp] == 0.
    for r in (1., 10., 100.)
        soft = Polarity((x = Observed(), y = Unobserved()), (x = r, y = 0.))
        @test first(LenticulumCore.energy(f, [1., 0.], soft, [0., 0.], ps, st))[:clamp] == r^2 / 2
    end
    @test first(Mycelium.local_free_energy(f, (x = DiracBelief([0.]),), ps, st))[:clamp] == 0.
end
@testset "LP-018.6 symmetric mixture prox escapes a maximum" begin
    s = VPSDE()
    l = GaussianMixtureEps(s, [-2. 2.]; s = 0.2)
    u = only(first(proximal(MixtureProx(l), [0.], 0., 1., (μ = [-2. 2.],), (;))))
    obj(x) = x*x/2 - log((exp(-(x+2)^2/0.08)+exp(-(x-2)^2/0.08))/(2sqrt(2π*0.04)))
    @test abs(u) ≈ 2/1.04 atol = 1e-9
    @test obj(u) ≈ obj(2/1.04) atol = 1e-12
    @test (obj(u+1e-4)-2obj(u)+obj(u-1e-4))/1e-8 > 0
end
@testset "LP-018.9 mixture weights avoid finite overflow" begin
    l = GaussianMixtureEps(VPSDE(), [1. 3.]; weights = [1e308, 1e308])
    @test exp.(l.logw) ≈ [0.5, 0.5]
end
@testset "LP-018.10 ELBO epsilon weight from Gaussian KL" begin
    s = VPSDE()
    for t in (0.1, 0.5, 0.9)
        dt = 1e-6
        a = alpha(s,t) / alpha(s,t-dt)
        vt, vu = sigma(s,t)^2, sigma(s,t-dt)^2
        b = 1-a*a
        vq = b*vu/vt
        KL(δ) = sum((-1.,1.)) do sign
            xt = sign*sqrt(vt)
            mq = a*vu/vt*xt
            mp = (xt-b/sqrt(vt)*(sqrt(vt)*xt+δ))/a
            ((mq-mp)^2/b + vq/b - 1 + log(b/vq))/4
        end
        @test (KL(1.)-KL(0.))/dt ≈ beta(s,t)/(2vt) rtol = 3e-5
    end
    note = read(joinpath(@__DIR__, "../src/schedule.md"), String)
    @test occursin(raw"\beta(t)/(2\sigma_t^2)", note)
end
@testset "LP-018.11 terminal RED-Diff weight is finite" begin
    @test reddiff_weight(REDDiff(λ = 1.), VPSDE(), 1.) ≈ sqrt(expm1(10.05)) rtol = 1e-14
    @test occursin("finite", string(@doc reddiff_weight))
    @test !occursin("grows without bound", string(@doc reddiff_weight))
end
@testset "LP-018.12 exact unit-vector logdet uses full depth" begin
    S = [2. 1.; 1. 2.]
    @test VariationalDiffusion._logdet(x -> S*x, 2, KrylovSolve(lanczos = 1, probes = 2)) ≈ log(3.) atol = 1e-14
end
end
