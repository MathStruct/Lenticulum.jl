"""
Experiment E1 of `vault/Design/Minimal Stochastic Factors.md`: does the switching constraint
factor fix the spectral measurement's failure on non-orthogonal lines?

Data: two noisy lines through the origin at 0° and 60° (`MeasurementFactors.lines_data`).
Models, all fitted on the same 20 000 points and scored on 5 000 held-out points:

- the spectral measurement of `examples/measurement_factor.jl` (the baseline that fails);
- a single Gaussian;
- an unconstrained two-component Gaussian mixture, by EM (the control);
- the switching constraint factor in the general form, ψ(k, z) = w_k N(a_kᵀz; 0, q_k) with a
  Gaussian reference N(0, s² I), fitted by maximum likelihood (exact log-likelihood, gradient
  ascent with ForwardDiff, several restarts).

The conditional p(y | x) of the fitted switching factor is also computed by message passing in a
Mycelium graph (reference priors on x and y, the factor, a data clamp on x) and checked against the
direct conditional-Gaussian formula. Run with `julia --project=. examples/switching_factor_e1.jl`.
"""
module SwitchingE1

using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, Printf
using LenticulumCore: DiracBelief, MixtureBelief, Observed, Unobserved, Polarity, Channel
import ForwardDiff

isdefined(Main, :MeasurementFactors) ||
    Base.include(Main, joinpath(@__DIR__, "measurement_factor.jl"))
const MF = Main.MeasurementFactors

# --- the switching factor's joint, in closed form ----------------------------------------------

"""
Parameters θ = (φ₁, φ₂, log q₁, log q₂, logit w₁, log s²). Outcome k constrains the residual
a_kᵀz, a_k = (−sin φ_k, cos φ_k) (the normal of the line at angle φ_k), with variance q_k;
the reference is N(0, s² I). Outcome k of the joint is N(0, C_k) with
C_k⁻¹ = I/s² + a_k a_kᵀ/q_k and weight ∝ w_k N(0; 0, q_k + s²) (the constraint's mass under the
reference).
"""
function components(θ)
    s2 = exp(θ[6])
    w1 = 1 / (1 + exp(-θ[5]))
    w = (w1, 1 - w1)
    map(1:2) do k
        a = [-sin(θ[k]), cos(θ[k])]
        q = exp(θ[2+k])
        C = inv(Matrix(I / s2 + a * a' / q))
        mass = w[k] / sqrt(2π * (q + s2))
        (; a, q, s2, w = w[k], mass, C)
    end
end
mixture_weights_of(cs) = (m = [c.mass for c in cs]; m ./ sum(m))

logN2(z, C) = -(dot(z, C \ z) + logdet(C) + 2log(2π)) / 2
function loglik(θ, Z)
    cs = components(θ)
    β = mixture_weights_of(cs)
    tot = zero(eltype(θ))
    for i in axes(Z, 1)
        z = Z[i, :]
        l1, l2 = log(β[1]) + logN2(z, cs[1].C), log(β[2]) + logN2(z, cs[2].C)
        m = max(l1, l2)
        tot += m + log(exp(l1 - m) + exp(l2 - m))
    end
    return tot / size(Z, 1)
end

"Maximum likelihood by Adam on the exact log-likelihood, from a grid of starting angles."
function fit_switching(Z; steps = 600, lr = 0.05, starts = [(0.3, 1.3), (-0.5, 0.8), (1.0, 2.5), (0.1, -1.2)])
    best, bestll, lls = nothing, -Inf, Float64[]
    for (φ1, φ2) in starts
        θ = [φ1, φ2, log(0.1), log(0.1), 0.0, 0.0]
        m, v = zero(θ), zero(θ)
        for t in 1:steps
            g = ForwardDiff.gradient(x -> -loglik(x, Z), θ)
            m = 0.9m + 0.1g; v = 0.999v + 0.001g .^ 2
            θ = θ - lr * (m / (1 - 0.9^t)) ./ (sqrt.(v / (1 - 0.999^t)) .+ 1e-8)
            θ[3:4] = max.(θ[3:4], log(1e-6))             # a noise floor on q_k
        end
        ll = loglik(θ, Z)
        push!(lls, ll)
        ll > bestll && ((best, bestll) = (θ, ll))
    end
    return best, bestll, lls
end

"Conditional p(y | x) of the switching joint, by Gaussian conditioning per outcome."
function conditional(θ, x)
    cs = components(θ)
    β = mixture_weights_of(cs)
    lw = [log(β[k]) - (x^2 / cs[k].C[1, 1] + log(2π * cs[k].C[1, 1])) / 2 for k in 1:2]
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    means = [cs[k].C[2, 1] / cs[k].C[1, 1] * x for k in 1:2]
    vars = [cs[k].C[2, 2] - cs[k].C[2, 1]^2 / cs[k].C[1, 1] for k in 1:2]
    return w, means, vars
end
cond_logpdf(θ, x, y) = (c = conditional(θ, x);
    log(sum(c[1][k] * exp(-(y - c[2][k])^2 / (2c[3][k])) / sqrt(2π * c[3][k]) for k in 1:2)))

# --- the same factor as a Mycelium factor: messages of §3 ---------------------------------------

"""
    SwitchingConstraintFactor(w, A, q)

ψ(k, x, y) = w_k N(A[k,1] x + A[k,2] y; 0, q_k) on two scalar channels, the outcome summed
inside (no `:k` channel). The message to one channel from a Gaussian or point on the other is
the mixture of §3 with per-outcome constant Z_k = N(d_k; 0, R_k); its components are emitted as
normalised Gaussians, so each weight also carries the component's mass √(2π/Λ) e^{η²/2Λ}.
"""
struct SwitchingConstraintFactor <: LenticulumCore.AbstractLenticulumFactor
    w::Vector{Float64}
    A::Matrix{Float64}
    q::Vector{Float64}
end
LenticulumCore.channels(::SwitchingConstraintFactor) = (Channel(:x, 1), Channel(:y, 1))
LenticulumCore.supported_polarities(::SwitchingConstraintFactor) =
    (Polarity(x = Observed(), y = Unobserved()), Polarity(x = Unobserved(), y = Observed()))
LenticulumCore.supports_polarity(::SwitchingConstraintFactor, p::Polarity) = keys(p) == (:x, :y) &&
    ((p[:x] isa Observed && p[:y] isa Unobserved) || (p[:x] isa Unobserved && p[:y] isa Observed))

function Mycelium.factor_message(f::SwitchingConstraintFactor, target::Symbol, pol, inputs, prior, ps, st)
    t, o = target === :y ? (2, 1) : (1, 2)
    src = getfield(inputs, target === :y ? :x : :y)
    m, S = src isa DiracBelief ? (only(src.value), 0.0) : (only(belief_mean(src)), only(belief_cov(src)))
    comps, logw = LenticulumCore.GaussianBelief[], Float64[]
    for k in eachindex(f.w)
        d = -f.A[k, o] * m                    # b_k − Σ_{i≠t} A_{k,i} m_i with b_k = 0
        R = f.q[k] + f.A[k, o]^2 * S
        logZ = -(d^2 / R + log(2π * R)) / 2
        η, Λ = f.A[k, t] * d / R, f.A[k, t]^2 / R
        push!(comps, Gaussian(η / Λ, 1 / Λ))
        push!(logw, log(f.w[k]) + logZ + η^2 / (2Λ) + log(2π / Λ) / 2)   # Z_k times the component mass
    end
    return MixtureBelief(Val(:log), comps, logw .- (maximum(logw) + log(sum(exp.(logw .- maximum(logw)))))), st
end

"The fitted factor as a graph: reference priors on x and y, the factor, x clamped."
function graph_conditional(θ, x)
    cs = components(θ)
    f = SwitchingConstraintFactor([c.w for c in cs], [cs[k].a[j] for k in 1:2, j in 1:2], [c.q for c in cs])
    b = GraphBuilder()
    variable!(b, :x, 1); variable!(b, :y, 1)
    factor!(b, :px, GaussianPrior(:x, [0.0], fill(cs[1].s2, 1, 1)))
    factor!(b, :py, GaussianPrior(:y, [0.0], fill(cs[1].s2, 1, 1)))
    factor!(b, :sw, f)
    factor!(b, :dx, DataFactor(:x, [x]))
    connect!(b, :px, :x, :x); connect!(b, :py, :y, :y)
    connect!(b, :dx, :x, :x; direction = Emitting())
    connect!(b, :sw, :x, :x; direction = Absorbing()); connect!(b, :sw, :y, :y)
    g = build(b)
    ps = LuxCore.initialparameters(Xoshiro(0), g)
    marg, _, _, _ = infer!(g, tree_schedule(g), ps, LuxCore.initialstates(Xoshiro(0), g))
    return marg.y
end
import LuxCore

# --- the control: a free two-component Gaussian mixture by EM -----------------------------------

function fit_gmm(Z; iters = 300, restarts = 4)
    best, bestll = nothing, -Inf
    for r in 1:restarts
        rng = Xoshiro(r)
        μ = [Z[rand(rng, axes(Z, 1)), :] for _ in 1:2]
        C = [Matrix(1.0I, 2, 2) for _ in 1:2]; w = [0.5, 0.5]
        ll = -Inf
        for _ in 1:iters
            L = [log(w[k]) + logN2(Z[i, :] - μ[k], C[k]) for i in axes(Z, 1), k in 1:2]
            M = maximum(L; dims = 2); ll = sum(M .+ log.(sum(exp.(L .- M); dims = 2))) / size(Z, 1)
            Rr = exp.(L .- M); Rr ./= sum(Rr; dims = 2)
            for k in 1:2
                nk = sum(Rr[:, k]); w[k] = nk / size(Z, 1)
                μ[k] = vec(sum(Rr[:, k] .* Z; dims = 1)) / nk
                D = Z .- μ[k]'
                C[k] = (D' * (Rr[:, k] .* D)) / nk + 1e-9I
            end
        end
        ll > bestll && ((best, bestll) = ((; w = copy(w), μ = deepcopy(μ), C = deepcopy(C)), ll))
    end
    return best, bestll
end
function gmm_cond_logpdf(G, x, y)
    lw = [log(G.w[k]) - ((x - G.μ[k][1])^2 / G.C[k][1, 1] + log(2π * G.C[k][1, 1])) / 2 for k in 1:2]
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    log(sum(w[k] * (v = G.C[k][2, 2] - G.C[k][2, 1]^2 / G.C[k][1, 1];
                    exp(-(y - G.μ[k][2] - G.C[k][2, 1] / G.C[k][1, 1] * (x - G.μ[k][1]))^2 / (2v)) / sqrt(2π * v)) for k in 1:2))
end
gmm_logpdf(G, z) = (l = [log(G.w[k]) + logN2(z - G.μ[k], G.C[k]) for k in 1:2]; m = maximum(l); m + log(sum(exp.(l .- m))))

# --- the experiment -------------------------------------------------------------------------------

function run(; θs = [0.0, π / 3], wts = [0.5, 0.5], σ = 0.05, n = 20_000, ntest = 5_000)
    x, y = MF.lines_data(θs, wts, n; σ, rng = Xoshiro(1))
    xt, yt = MF.lines_data(θs, wts, ntest; σ, rng = Xoshiro(1001))
    Z, Zt = hcat(x, y), hcat(xt, yt)
    score(h) = sum(h(xt[i], yt[i]) for i in eachindex(xt)) / ntest
    # the true model's joint
    oracle_joint(z) = (l = [log(wts[k]) + logN2(z, [cos(θs[k]); sin(θs[k])] * [cos(θs[k]) sin(θs[k])] + σ^2 * I) for k in 1:2];
                       m = maximum(l); m + log(sum(exp.(l .- m))))
    θ, _, start_lls = fit_switching(Z)
    G, _ = fit_gmm(Z)
    spec = MF.fit_measurement(x, y)
    gauss = MF.gaussian_baseline(x, y)
    angles = sort(mod.(rad2deg.(θ[1:2]), 180))
    err = [minimum(MF.line_error(a, t) for a in angles) for t in sort(mod.(rad2deg.(θs), 180))]
    # graph messages against the direct conditional at a few inputs
    graph_err = maximum(map([-1.5, -0.4, 0.3, 1.2]) do x0
        w, ms, vs = conditional(θ, x0)
        b = graph_conditional(θ, x0)
        perm = sortperm(only.(belief_mean.(b.components)))
        o = sortperm(ms)
        max(maximum(abs, mixture_weights(b)[perm] .- w[o]),
            maximum(abs, only.(belief_mean.(b.components))[perm] .- ms[o]),
            maximum(abs, only.(belief_cov.(b.components))[perm] .- vs[o]))
    end)
    return (; θ, start_lls, angles, err, q = exp.(θ[3:4]), s2 = exp(θ[6]), w = 1 / (1 + exp(-θ[5])), graph_err,
            cond = (switching = score((a, b) -> cond_logpdf(θ, a, b)), gmm = score((a, b) -> gmm_cond_logpdf(G, a, b)),
                    spectral = score((a, b) -> MF.logpdf(spec, [a], [b])), gauss = score(gauss.logpdf),
                    oracle = score((a, b) -> MF.oracle_logpdf(θs, wts, σ, a, b))),
            joint = (switching = sum(loglik(θ, Zt[i:i, :]) for i in axes(Zt, 1)) / ntest,
                     gmm = sum(gmm_logpdf(G, Zt[i, :]) for i in axes(Zt, 1)) / ntest,
                     oracle = sum(oracle_joint(Zt[i, :]) for i in axes(Zt, 1)) / ntest),
            nparams = (switching = 6, gmm = 11))
end

function main()
    r = run()
    println("E1: lines at 0° and 60°, weights 0.5/0.5, σ = 0.05")
    @printf("  switching factor: lines at %s° (errors %s°), q = %s, s² = %.3f, w₁ = %.3f\n",
            round.(r.angles; digits = 2), round.(r.err; digits = 3), round.(r.q; sigdigits = 3), r.s2, r.w)
    @printf("  held-out mean log p(y|x): switching %.3f, free GMM %.3f, spectral %.3f, Gaussian %.3f, true model %.3f\n",
            r.cond.switching, r.cond.gmm, r.cond.spectral, r.cond.gauss, r.cond.oracle)
    @printf("  held-out mean log p(z):   switching %.3f (6 parameters), free GMM %.3f (11), true model %.3f\n",
            r.joint.switching, r.joint.gmm, r.joint.oracle)
    @printf("  training log-likelihood per restart: %s\n", round.(r.start_lls; digits = 4))
    @printf("  graph message passing vs direct conditional: max difference %.2e\n", r.graph_err)
    return r
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SwitchingE1.main()
end
