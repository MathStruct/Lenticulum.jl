# P35: MixtureProx where K = I - λ∇²log p_t is indefinite. The solver fell back to steepest
# descent there, which zigzags across a stiff non-convex valley: the full-size example
# `examples/proxdm_backprop_noise.jl` stopped with "exceeded maxiters=100" (case below, from its
# five-training-set part; P15). It now scales the residual by the eigenvalue magnitudes of K.
@testset "P35: MixtureProx converges through indefinite curvature" begin
    μ1 = [1.0, 0.9951847266721969, 0.9807852804032304, 0.9569403357322088, 0.9238795325112867,
        0.881921264348355, 0.8314696123025452, 0.773010453362737, 0.7071067811865476,
        0.6343932841636455, 0.5555702330196023, 0.4713967368259978, 0.38268343236508984,
        0.29028467725446233, 0.19509032201612833, 0.09801714032956077, 6.123233995736766e-17,
        -0.09801714032956065, -0.1950903220161282, -0.29028467725446216, -0.3826834323650897,
        -0.4713967368259977, -0.555570233019602, -0.6343932841636454, -0.7071067811865475,
        -0.773010453362737, -0.8314696123025453, -0.8819212643483549, -0.9238795325112867,
        -0.9569403357322088, -0.9807852804032304, -0.9951847266721968, -1.0, -0.9951847266721969,
        -0.9807852804032304, -0.9569403357322089, -0.9238795325112681, -0.8819212567224769,
        -0.831123883002378, -0.6665248166664733, -0.5425124966897426, -0.48384697597944737,
        -0.4933204082852332, -0.5907277660615, -0.25603834430840083, -0.2828467800203605,
        -0.2406961096272425, -0.004860002570322367, -0.12193367025519297, 0.24459299619737002,
        0.1115596786392982, 0.1830583286165207, 0.42730865803297524, 0.4147282657737697,
        0.5146991685383209, 0.6934890955690206, 0.661576946938285, 0.64968882121049,
        0.6732984240687152, 0.7608161946131431, 0.9198378412697548, 0.9569402410735215,
        0.9807852804030197, 0.9951847266721969]
    μ2 = [-2.4396207769959308e-27, 0.0980171403295606, 0.19509032201612825, 0.29028467725446233,
        0.3826834323650898, 0.47139673682599764, 0.5555702330196022, 0.6343932841636455,
        0.7071067811865475, 0.7730104533627369, 0.8314696123025452, 0.8819212643483549,
        0.9238795325112867, 0.9569403357322089, 0.9807852804032304, 0.9951847266721968, 1.0,
        0.9951847266721969, 0.9807852804032304, 0.9569403357322089, 0.9238795325112867,
        0.881921264348355, 0.8314696123025453, 0.7730104533627371, 0.7071067811865476,
        0.6343932841636455, 0.5555702330196022, 0.47139673682599786, 0.3826834323650899,
        0.2902846772544624, 0.1950903220161286, 0.09801714032956083, 1.2246467991473532e-16,
        -0.09801714032956059, -0.19509032201612836, -0.2902846772544621, -0.38268343236510416,
        -0.47139674169995227, -0.5557473588375416, -0.6734662126216753, -0.6821758263918578,
        -0.6802768140551145, -0.6986215490138783, -0.7365515990946746, -0.7750094457355393,
        -0.7857876270048441, -0.8200813947982554, -0.8130347667004302, -0.8409418512576152,
        -0.8466175151964094, -0.8043902838172108, -0.8057290577312595, -0.7569589279871094,
        -0.8278654867030971, -0.7851020382060523, -0.6388874378592642, -0.6030153746351469,
        -0.5914034496772995, -0.5870277602064395, -0.5769360945245106, -0.38640367854449204,
        -0.2902847817279395, -0.195090322016405, -0.0980171403295605]
    μ = permutedims(hcat(μ1, μ2))
    l = GaussianMixtureEps(VPSDE(), μ; s = 0.05)
    v = [0.03206924750699991, -0.8095174270033064]
    t, λ, ps = 0.005, 0.01, (μ = μ,)
    K(u) = Symmetric(I - λ .* VariationalDiffusion._mixture_grad_hess(l, u, t, μ)[2])
    resid(u) = norm(u .- v .- λ .* VariationalDiffusion._mixture_grad_hess(l, u, t, μ)[1])
    @test !isposdef(K([0.0477, -0.8091]))              # the path crosses an indefinite region

    u, _ = proximal(MixtureProx(l; tol = 1e-12), v, t, λ, ps, NamedTuple())
    @test resid(u) <= 1e-12 * (1 + norm(u))
    @test isposdef(K(u))
    # the minimiser the old solver reached after several hundred steepest-descent iterations
    @test u ≈ [0.07491389921251443, -0.8075966642857769] rtol = 1e-9
    @test first(proximal(MixtureProx(l; tol = 1e-12, maxiters = 20), v, t, λ, ps, NamedTuple())) ≈ u

    # the direction: Newton's on a positive definite K, a descent direction on an indefinite one
    dir = VariationalDiffusion._modified_newton_direction
    A = Symmetric([2.0 0.3; 0.3 0.5]); r = [0.7, -1.1]
    @test dir(A, r) ≈ -(A \ r)
    B = Symmetric([1.0 2.0; 2.0 -3.0])
    for r in ([1.0, 0.0], [0.0, 1.0], [0.3, -2.0]), M in (B, -B)
        @test dot(r, dir(M, r)) < 0
    end
    @test all(isfinite, dir(Symmetric([1.0 0.0; 0.0 0.0]), [1.0, 1.0]))   # a zero eigenvalue is floored

    # review (LP-066): a Newton step within the objective's rounding was refused when the
    # objective was small but λ log p_t was not (λ = 100); the old path avoided that point
    μw1 = [0.3993162328252931, 0.49598666771056593, -2.4039244710772665, -0.861544607140714,
        -0.39229633080244275, -0.7224944307122051, -0.6504220160794558, -0.5413577777156585,
        0.4561519814300526]
    μw2 = [0.5702129659248619, -0.4035738191032367, 1.0400099999650314, 1.4967563414533125,
        0.040768265638907654, -1.4923778311023799, 0.009477596898997426, 0.7811724917948516,
        0.32779082704232165]
    μw = permutedims(hcat(μw1, μw2))
    logw = [-0.46660422282813135, -1.478788561206363, -26.139967117049412, -1.9472924553349316,
        -13.391452820858529, -23.16595303420868, -33.90134619277815, -12.516259723055217,
        -6.077837889459127]
    lw = GaussianMixtureEps(VPSDE(), μw, 0.31622776601683794, logw)
    vw = [0.7704004713295343, 0.14846917942380433]
    uw, _ = proximal(MixtureProx(lw; tol = 1e-10), vw, 0.0, 100.0, (μ = μw,), NamedTuple())
    gw, Hw = VariationalDiffusion._mixture_grad_hess(lw, uw, 0.0, μw)
    @test norm(uw .- vw .- 100.0 .* gw) <= 1e-10 * (1 + norm(uw))
    @test isposdef(Symmetric(I - 100.0 .* Hw))

    # review (LP-066): ForwardDiff through the solve, with a repeated eigenvalue of K on the way
    # (eigen of duals gave NaN partials), against the exact Jacobian
    μ3 = [-1.0 1.0; 0.0 0.0; 0.0 0.0]
    p3 = MixtureProx(GaussianMixtureEps(VPSDE(), μ3; s = 0.5); tol = 1e-10)
    v3 = [0.1, 0.2, 0.3]
    J = ForwardDiff.jacobian(x -> first(proximal(p3, x, 0.0, 1.0, (μ = μ3,), NamedTuple())), v3)
    @test all(isfinite, J)
    @test J ≈ prox_jacobian(p3, v3, 0.0, 1.0, (μ = μ3,), NamedTuple()) rtol = 1e-8

    # random mixtures: a local minimiser with an objective below the start's, indefinite starts included
    rng = Xoshiro(35)
    indefinite = 0
    for _ in 1:200
        d, k = rand(rng, 1:3), rand(rng, 2:20)
        μr = randn(rng, d, k)
        lr = GaussianMixtureEps(VPSDE(), μr; s = rand(rng, (0.02, 0.05, 0.1)))
        tr, λr = rand(rng, (0.0, 0.005, 0.05)), rand(rng, (0.01, 0.1, 1.0))
        vr = 1.3 .* randn(rng, d)
        obj(u) = sum(abs2, u .- vr) / 2 - λr * mixture_logdensity(lr, u, tr, (μ = μr,))
        g0, H0 = VariationalDiffusion._mixture_grad_hess(lr, vr, tr, μr)
        indefinite += !isposdef(Symmetric(I - λr .* H0))
        ur, _ = proximal(MixtureProx(lr; tol = 1e-12), vr, tr, λr, (μ = μr,), NamedTuple())
        g, H = VariationalDiffusion._mixture_grad_hess(lr, ur, tr, μr)
        @test norm(ur .- vr .- λr .* g) <= 1e-12 * (1 + norm(ur))
        @test isposdef(Symmetric(I - λr .* H))
        @test obj(ur) <= obj(vr)
    end
    @test indefinite > 0
end
