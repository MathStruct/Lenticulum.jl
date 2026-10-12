# Backpropagating through proximal inference at a higher noise level than inference used.
#
# `Backpropagation through Implicit Inference.md` §9 asks whether the adjoint should be evaluated
# at a noise level t_b above the forward's t_f: a biased gradient of a more smoothed problem, but
# better conditioned. This runs the question for ProxDM, where `prox_pullback` takes the noise
# level as its own argument, on the §7 task: move the means of a ring mixture so that proximal
# inference "x ↦ y" reproduces y = x² − 0.5, with gradients from the adjoint only.
#
#     julia --project=lib/VariationalDiffusion.jl/test lib/VariationalDiffusion.jl/examples/proxdm_backprop_noise.jl
#
# Prints (1) the cosine between the surrogate gradient (t_b > t_f) and the exact one (t_b = t_f),
# at the start and after training, and (2) training runs for each t_b at two learning rates.
# Results are recorded in `proxdm.md` §8.

using VariationalDiffusion, Random, LinearAlgebra, Statistics, Printf

const S = VPSDE()
const J = 64
const θs = range(0, 2π; length = J + 1)[1:J]
const μ₀ = vcat(cos.(θs)', sin.(θs)')
const P = MixtureProx(GaussianMixtureEps(S, μ₀; s = 0.05); tol = 1e-12)
const tf, λ = 0.005, 0.01                          # the forward's noise level and coupling
const ρ = [Inf, 0.0]                               # x clamped, y free
target(x) = x^2 - 0.5
const XS = Ref(collect(range(-0.8, 0.8; length = 9)))   # training inputs (replaced per training set below)
xs() = XS[]
const xt = collect(range(-0.75, 0.75; length = 31)) # test inputs

solve(x, y0, μ) = first(prox_infer(P, [x, y0], ρ, tf, (μ = μ,), NamedTuple(); λ, maxiters = 2000))

# the gradient of mean ½(y* - y)² over the training set, the adjoint taken at level tb
function gradient_at(μ, yw, tb)
    G, L, dropped = zero(μ), 0.0, 0
    for (i, x) in enumerate(xs())
        sol = solve(x, yw[i], μ)
        sol.converged || (dropped += 1; continue)
        yw[i] = sol.z[2]
        r = sol.z[2] - target(x)
        L += r^2 / 2
        G .+= prox_pullback(P, sol, [x, yw[i]], ρ, tb, [0.0, r], (μ = μ,), NamedTuple(); λ).ps.μ
    end
    return (G ./ length(xs()), L / length(xs()), dropped)
end

cosine(a, b) = dot(a, b) / (norm(a) * norm(b))

function test_rmse(μ; warm)
    errs = map(xt) do x
        y0 = warm ? target(x) + 0.1 : -0.5        # warm: near the answer; cold: one fixed start
        sol = solve(x, y0, μ)
        sol.converged ? (sol.z[2] - target(x))^2 : NaN
    end
    return sqrt(mean(filter(!isnan, errs))), count(isnan, errs)
end

function train(tb; lr, epochs = 150)
    μ = copy(μ₀); yw = fill(-0.5, length(xs())); m = zero(μ); v = zero(μ)
    losses, dropped = Float64[], 0
    for ep in 1:epochs
        G, L, d = gradient_at(μ, yw, tb)
        push!(losses, L); dropped += d
        m = 0.9 .* m .+ 0.1 .* G; v = 0.999 .* v .+ 0.001 .* G .^ 2
        μ .-= lr .* (m ./ (1 - 0.9^ep)) ./ (sqrt.(v ./ (1 - 0.999^ep)) .+ 1e-8)
    end
    push!(losses, gradient_at(μ, yw, tf)[2])
    return (μ = μ, losses = losses, dropped = dropped)
end

const TB = (0.005, 0.01, 0.02, 0.05, 0.1)
@printf("σ(t): %s\n", join([@sprintf("t=%.3g σ=%.3f", t, sigma(S, t)) for t in TB], ", "))

# (1) cosine to the exact gradient, at the start and at the exactly trained means
μ_trained = train(tf; lr = 0.01).μ
for (name, μ) in (("start", μ₀), ("trained", μ_trained))
    Gs = Dict(tb => gradient_at(μ, fill(-0.5, length(xs())), tb)[1] for tb in TB)
    @printf("cosine to exact (%s):  %s\n", name,
            join([@sprintf("t_b=%.3g: %.3f (|g|/|g*| %.2f)", tb, cosine(Gs[tb], Gs[tf]), norm(Gs[tb]) / norm(Gs[tf])) for tb in TB], ",  "))
end

# (2) training with each t_b
for lr in (0.01, 0.03)
    println("\nlr = $lr")
    @printf("%-8s %-10s %-10s %-10s %-12s %-12s %-12s %s\n", "t_b", "loss@0", "loss@50", "loss@150",
            "ep to 1e-3", "ups (>+10%)", "test warm", "test cold")
    for tb in TB
        r = train(tb; lr)
        hit = findfirst(<(1e-3), r.losses)
        ups = count(i -> r.losses[i + 1] > 1.1 * r.losses[i], 1:(length(r.losses) - 1))
        (rw, nw), (rc, nc) = test_rmse(r.μ; warm = true), test_rmse(r.μ; warm = false)
        @printf("%-8.3g %-10.2e %-10.2e %-10.2e %-12s %-12d %-12s %s\n", tb, r.losses[1], r.losses[51], r.losses[end],
                hit === nothing ? "—" : string(hit - 1), ups, @sprintf("%.3f", rw) * (nw > 0 ? " ($nw drop)" : ""),
                @sprintf("%.3f", rc) * (nc > 0 ? " ($nc drop)" : ""))
    end
end

# (3) the lr = 0.01 runs on five training sets: the grid above and four random draws of 9 inputs
println("\nlr = 0.01, five training sets (grid + 4 random): median [min, max]")
sets = [[collect(range(-0.8, 0.8; length = 9))]; [sort(1.6 .* rand(Xoshiro(k), 9) .- 0.8) for k in 1:4]]
med(v) = @sprintf("%.3g [%.3g, %.3g]", median(v), minimum(v), maximum(v))
@printf("%-8s %-28s %-16s %-26s %s\n", "t_b", "loss@150", "ups (>+10%)", "test warm", "test cold")
for tb in TB
    rows = map(sets) do x
        XS[] = x
        r = train(tb; lr = 0.01)
        ups = count(i -> r.losses[i + 1] > 1.1 * r.losses[i], 1:(length(r.losses) - 1))
        (r.losses[end], ups, first(test_rmse(r.μ; warm = true)), first(test_rmse(r.μ; warm = false)))
    end
    @printf("%-8.3g %-28s %-16s %-26s %s\n", tb, med(getindex.(rows, 1)), med(getindex.(rows, 2)),
            med(getindex.(rows, 3)), med(getindex.(rows, 4)))
end
