# Experiment E1's two crossing lines with one noise level per line, as one Mycelium graph of
# RegimeFactors, fitted by the free-energy gradient and checked against closed-form EM.
# Recorded in `examples/regime_lines.md`.
# Run:  julia -t auto --project=docs examples/regime_lines.jl   (an environment with ForwardDiff)
#
# Model: x ~ U(−3, 3), k ~ (w₁, w₂), y = a_k x + c_k + σ_k ε. As a factor on z = (x, y):
# 0 = [−a_k 1] z − c_k + ε_k, ε_k ~ N(0, σ_k²): the coefficients and the noise depend on the line,
# which `SwitchingConstraintFactor` cannot state (`switching.md` §5). Point i is its data, its
# outcome k_i and one RegimeFactor, tied over the points, so the graph is a forest and its Bethe
# free energy is exactly −Σᵢ log p(yᵢ | xᵢ). The coefficient of y is held at 1 (the scale of the
# residual is the noise's). The reference is EM for a mixture of linear regressions (weighted
# least squares per line), from the same start.
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf, Statistics
using ADTypes, DifferentiationInterface, ForwardDiff

const N = 300
const TRUE = (a = [1.0, -1.0], c = [0.0, 0.5], σ = [0.2, 0.5], w = [0.5, 0.5])
function data(rng)
    x = 6 .* rand(rng, N) .- 3
    k = [rand(rng) < TRUE.w[1] ? 1 : 2 for _ in 1:N]
    y = TRUE.a[k] .* x .+ TRUE.c[k] .+ TRUE.σ[k] .* randn(rng, N)
    return x, y
end
nm(s, i) = replica_name(s, i)

function graph(x, y)
    b = GraphBuilder()
    for i in 1:N
        variable!(b, nm(:z, i), 2); variable!(b, nm(:k, i), 2)
        factor!(b, nm(:data, i), DataFactor(:z, [x[i], y[i]])); connect!(b, nm(:data, i), :z, nm(:z, i); direction = Emitting())
        factor!(b, nm(:reg, i), RegimeFactor((z = 2,), 2, 1; learn_noise = true); params = :reg)
        connect!(b, nm(:reg, i), :z, nm(:z, i)); connect!(b, nm(:reg, i), :k, nm(:k, i))
    end
    return build(b)
end
# θ = (a, c, logw, log σ) ↔ the factor's parameters
params(θ) = (A = (z = cat(([-θ.a[k] 1.0] for k in 1:2)...; dims = 3),), b = reshape(θ.c, 1, 2), logw = θ.logw,
             L = reshape(θ.logσ, 1, 1, 2))
fullps(θ) = merge(NamedTuple{Tuple(nm(:data, i) for i in 1:N)}(ntuple(_ -> NamedTuple(), N)), (reg = params(θ),))
# two sequential sweeps solve the forest (data → z → regime → k). The regime factor's message to
# the observed z is left out: its components are improper (a line, not a point) with one precision
# per line, which a cluster refuses (`regime.md` §3), and nothing reads it.
function solve(g, ps)
    keep(t) = !(t.kind === :to_variable && g.variables[g.edges[t.edge].variable].name in zs &&
                g.factors[g.edges[t.edge].factor].factor isa RegimeFactor)
    zs = Set(nm(:z, i) for i in 1:N)
    s = MessageStore(g); sched = SequentialSchedule(filter(keep, all_tasks(g)))
    for _ in 1:2; sweep!(s, g, sched, ps, NamedTuple()); end
    return s
end
lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
function nlp(θ, x, y)              # −Σ log p(y | x), the closed form
    lw = θ.logw .- lse(θ.logw)
    return -sum(lse([lw[k] - (y[i] - θ.a[k] * x[i] - θ.c[k])^2 / (2 * exp(2θ.logσ[k])) - θ.logσ[k] - log(2π) / 2 for k in 1:2])
                for i in 1:N)
end

function fit_graph(g, θ0; steps = 400, η = 0.05)
    θ = deepcopy(θ0)
    keysθ = (:a, :c, :logw, :logσ)
    m = Dict(k => zero(getfield(θ, k)) for k in keysθ); v = Dict(k => zero(getfield(θ, k)) for k in keysθ)
    β1, β2 = 0.9, 0.999
    F = NaN
    for t in 1:steps
        ps = fullps(θ)
        s = solve(g, ps)
        F = first(scalar_free_energy(s, g, ps, NamedTuple()))
        G = free_energy_gradient(s, g, ps, NamedTuple(); ad = AutoForwardDiff()).reg
        grads = (a = -vec(G.A.z[1, 1, :]), c = vec(G.b), logw = G.logw, logσ = vec(G.L))
        θ = NamedTuple{keysθ}(map(keysθ) do k
            gk = getfield(grads, k)
            m[k] = β1 .* m[k] .+ (1 - β1) .* gk; v[k] = β2 .* v[k] .+ (1 - β2) .* gk .^ 2
            getfield(θ, k) .- η .* (m[k] ./ (1 - β1^t)) ./ (sqrt.(v[k] ./ (1 - β2^t)) .+ 1e-8)
        end)
    end
    return θ, F
end

function fit_em(θ0, x, y; tol = 1e-12, maxit = 10_000)
    a, c, σ, w = copy(θ0.a), copy(θ0.c), exp.(θ0.logσ), exp.(θ0.logw) ./ sum(exp.(θ0.logw))
    X = [x ones(N)]
    prev = Inf
    for it in 1:maxit
        lr = [log(w[k]) - (y[i] - a[k] * x[i] - c[k])^2 / (2σ[k]^2) - log(σ[k]) for i in 1:N, k in 1:2]
        R = exp.(lr .- [lse(lr[i, :]) for i in 1:N])
        for k in 1:2
            W = Diagonal(R[:, k])
            a[k], c[k] = (X' * W * X) \ (X' * W * y)
            σ[k] = sqrt(sum(R[:, k] .* (y .- a[k] .* x .- c[k]) .^ 2) / sum(R[:, k]))
            w[k] = mean(R[:, k])
        end
        θ = (a = a, c = c, logw = log.(w), logσ = log.(σ))
        f = nlp(θ, x, y)
        abs(prev - f) < tol && return θ, f, it
        prev = f
    end
    return (a = a, c = c, logw = log.(w), logσ = log.(σ)), prev, maxit
end

rng = Xoshiro(11)
x, y = data(rng)
g = graph(x, y)
θ0 = (a = [0.5, -0.5], c = [0.0, 0.0], logw = [0.0, 0.0], logσ = [0.0, 0.0])
θtrue = (a = TRUE.a, c = TRUE.c, logw = log.(TRUE.w), logσ = log.(TRUE.σ))

# the graph's free energy is the closed form, at the start
ps0 = fullps(θ0)
s0 = solve(g, ps0)
F0 = first(scalar_free_energy(s0, g, ps0, NamedTuple()))
@printf "start: F = %.10f, closed form %.10f, difference %.1e\n" F0 nlp(θ0, x, y) abs(F0 - nlp(θ0, x, y))

t_g = @elapsed θg, Fg = fit_graph(g, θ0)
t_e = @elapsed θe, Fe, ite = fit_em(θ0, x, y)
show(θ) = @sprintf "a = (%.3f, %.3f), c = (%.3f, %.3f), σ = (%.3f, %.3f), w = (%.3f, %.3f)" θ.a... θ.c... exp.(θ.logσ)... (exp.(θ.logw) ./ sum(exp.(θ.logw)))...
println("generating:      ", show(θtrue), @sprintf("   −log p = %.4f", nlp(θtrue, x, y)))
println("graph, 400 Adam: ", show(θg), @sprintf("   −log p = %.4f (%.1f s; F at the last step %.4f)", nlp(θg, x, y), t_g, Fg))
println("EM:              ", show(θe), @sprintf("   −log p = %.4f (%d iterations, %.2f s)", Fe, ite, t_e))
# one noise level for both lines (what SwitchingConstraintFactor can state), by EM with σ tied
function fit_em_tied(θ0, x, y; tol = 1e-12, maxit = 10_000)
    a, c, σ, w = copy(θ0.a), copy(θ0.c), 1.0, [0.5, 0.5]
    X = [x ones(N)]; prev = Inf
    for it in 1:maxit
        lr = [log(w[k]) - (y[i] - a[k] * x[i] - c[k])^2 / (2σ^2) for i in 1:N, k in 1:2]
        R = exp.(lr .- [lse(lr[i, :]) for i in 1:N])
        for k in 1:2
            a[k], c[k] = (X' * Diagonal(R[:, k]) * X) \ (X' * Diagonal(R[:, k]) * y); w[k] = mean(R[:, k])
        end
        σ = sqrt(sum(R[:, k] .* (y .- a[k] .* x .- c[k]) .^ 2 for k in 1:2) |> sum |> v -> v / N)
        θ = (a = a, c = c, logw = log.(w), logσ = log.([σ, σ]))
        f = nlp(θ, x, y)
        abs(prev - f) < tol && return θ, f
        prev = f
    end
end
θt, Ft = fit_em_tied(θ0, x, y)
println("EM, one σ:       ", show(θt), @sprintf("   −log p = %.4f", Ft))
# logw is determined up to a constant: compare the normalised log-weights
norm(θ) = merge(θ, (logw = θ.logw .- lse(θ.logw),))
@printf "graph − EM: %.2e in −log p, max |Δθ| = %.2e (log-weights normalised)\n" nlp(θg, x, y) - Fe maximum(abs, vcat((getfield(norm(θg), k) .- getfield(norm(θe), k) for k in (:a, :c, :logw, :logσ))...))
