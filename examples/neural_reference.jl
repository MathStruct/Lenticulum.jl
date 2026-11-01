"""
A neural reference for E4(b), `vault/Design/Minimal Stochastic Factors.md` §10, asked for by the
second round of reviews (Grok): how much is lost by staying inside the switching-linear family?

The reference is the standard diffusion reverse kernel p(z_{t-1} | z_t) = N(μ_θ(z_t, t), σ_t² I) of
Ho, Jain & Abbeel (2020), with the same forward process, data and scores as E4(b):

- the mean comes from a noise predictor ε̂_θ(z_t, t), μ = (z_t − β_t ε̂ / √(1 − ᾱ_t)) / √(1 − β_t); ε̂ is
  one time-conditioned MLP shared by all steps (inputs z_t and 16 sinusoidal features of t/T, two
  hidden SiLU layers of width H), trained on Ho et al.'s simple loss E‖ε − ε̂‖² by Adam with
  hand-written backpropagation;
- σ_t² is the evidence bound's optimum given the mean, E‖z_{t-1} − μ(z_t)‖² / 2 on fresh pairs
  (Proposition 9 makes each step's term separate);
- the prior at T is the Gaussian of E4(b), and the chain is scored by E4(b)'s importance-weighted
  bound and energy distance.

Run with `julia -t auto --project=. examples/neural_reference.jl`.
"""
module NeuralReference

using LinearAlgebra, Random, Printf

isdefined(Main, :SwitchingChainE4b) || Base.include(Main, joinpath(@__DIR__, "switching_chain_e4b.jl"))
const B = Main.SwitchingChainE4b
const E = Main.SwitchingE4a

# --- the network ---------------------------------------------------------------------------------

const NFREQ = 8
features(t, T) = (s = t / T; vcat([sin(π * 2.0^(i - 1) * s) for i in 1:NFREQ], [cos(π * 2.0^(i - 1) * s) for i in 1:NFREQ]))

mutable struct MLP
    W1::Matrix{Float64}; b1::Vector{Float64}; W2::Matrix{Float64}; b2::Vector{Float64}; W3::Matrix{Float64}; b3::Vector{Float64}
end
MLP(H, rng) = (nin = 2 + 2NFREQ;
    MLP(randn(rng, H, nin) ./ sqrt(nin), zeros(H), randn(rng, H, H) ./ sqrt(H), zeros(H), zeros(2, H), zeros(2)))
nparams(n::MLP) = sum(length, (n.W1, n.b1, n.W2, n.b2, n.W3, n.b3))
params(n::MLP) = (n.W1, n.b1, n.W2, n.b2, n.W3, n.b3)

σ(x) = 1 / (1 + exp(-x))
silu(x) = x * σ(x)
dsilu(x) = (s = σ(x); s * (1 + x * (1 - s)))

forward(n::MLP, X) = (A1 = n.W1 * X .+ n.b1; H1 = silu.(A1); A2 = n.W2 * H1 .+ n.b2; H2 = silu.(A2); (n.W3 * H2 .+ n.b3, (X, A1, H1, A2, H2)))

"Gradients of the mean squared error ‖Y − f(X)‖² / ncols."
function gradients(n::MLP, X, Y)
    out, (X, A1, H1, A2, H2) = forward(n, X)
    m = size(X, 2)
    D3 = 2 .* (out .- Y) ./ m
    gW3 = D3 * H2'; gb3 = vec(sum(D3; dims = 2))
    D2 = (n.W3' * D3) .* dsilu.(A2)
    gW2 = D2 * H1'; gb2 = vec(sum(D2; dims = 2))
    D1 = (n.W2' * D2) .* dsilu.(A1)
    gW1 = D1 * X'; gb1 = vec(sum(D1; dims = 2))
    return (gW1, gb1, gW2, gb2, gW3, gb3), sum(abs2, out .- Y) / m
end

"Train ε̂ on the simple loss: t uniform on 1:T, z_t = α_t z_0 + √(1 − α_t²) ε."
function train(Z0, sch, H; iters = 30_000, batch = 512, lr = 2e-3, seed = 1)
    rng = Xoshiro(seed)
    net = MLP(H, rng)
    ms = [zero(p) for p in params(net)]; vs = [zero(p) for p in params(net)]
    T = sch.T; n = size(Z0, 1)
    loss = 0.0
    for it in 1:iters
        X = zeros(2 + 2NFREQ, batch); Y = randn(rng, 2, batch)
        for b in 1:batch
            t = rand(rng, 1:T); a = sch.α[t+1]
            X[1:2, b] = a .* Z0[rand(rng, 1:n), :] .+ sqrt(1 - a^2) .* Y[:, b]
            X[3:end, b] = features(t, T)
        end
        g, l = gradients(net, X, Y)
        loss = 0.99loss + 0.01l
        η = lr * 0.5 * (1 + cos(π * it / iters))                    # cosine decay
        for (p, gp, m, v) in zip(params(net), g, ms, vs)
            m .= 0.9 .* m .+ 0.1 .* gp; v .= 0.999 .* v .+ 0.001 .* gp .^ 2
            p .-= η .* (m ./ (1 - 0.9^it)) ./ (sqrt.(v ./ (1 - 0.999^it)) .+ 1e-8)
        end
    end
    return net, loss
end

# --- the reverse kernel, in E4(a)'s chain interface ----------------------------------------------

struct NeuralKernel
    net::MLP; t::Int; T::Int; β::Float64; α::Float64; σ2::Float64     # α = α_t = √ᾱ_t
end
function kmean(k::NeuralKernel, zt)
    ε̂, _ = forward(k.net, reshape(vcat(zt, features(k.t, k.T)), :, 1))
    return (zt .- k.β / sqrt(1 - k.α^2) .* vec(ε̂)) ./ sqrt(1 - k.β)
end
E.kernel_logpdf(k::NeuralKernel, zprev, zt) = E.logN(zprev, kmean(k, zt), k.σ2 * Matrix(1.0I, 2, 2))
E.kernel_sample(k::NeuralKernel, zt, rng) = kmean(k, zt) .+ sqrt(k.σ2) .* randn(rng, 2)

"The kernels with the bound-optimal σ_t² given the mean, from one fresh pair per training point and step."
function kernels(net, sch, Z0; seed = 2)
    T = sch.T
    map(1:T) do t
        k0 = NeuralKernel(net, t, T, sch.βs[t], sch.α[t+1], 1.0)
        X = B.SS.step_pairs(sch, t, Z0, Xoshiro(seed + 100t))            # rows (z_{t-1}, z_t)
        r = sum(sum(abs2, X[i, 1:2] - kmean(k0, X[i, 3:4])) for i in axes(X, 1)) / (2size(X, 1))
        NeuralKernel(net, t, T, sch.βs[t], sch.α[t+1], r)
    end
end

# --- the experiment ------------------------------------------------------------------------------

function run(name; ntrain = 4000, ntest = 1000, ngen = 1500, settings = [(20, 64), (100, 64), (100, 128)],
             S = 64, iters = 30_000)
    Ztr = B.sample_data(name, ntrain, Xoshiro(1))
    Zte = B.sample_data(name, ntest, Xoshiro(2))
    truth = E.tmean(i -> B.data_logpdf(name, Zte[i, :]), ntest)
    @printf("%s: held-out log p of the truth %.3f\n", name, truth)
    rows = []
    for (T, H) in settings
        sch = E.schedule(T, 1 - 0.8^(20 / T))
        t1 = @elapsed net, loss = train(Ztr, sch, H; iters)
        ks = kernels(net, sch, Ztr)
        a = sch.α[end]; ZT = a .* Ztr .+ sqrt(1 - a^2) .* randn(Xoshiro(1), size(Ztr)...)
        prior = ([1.0], [vec(sum(ZT; dims = 1)) / size(ZT, 1)], [E.cov(ZT)])
        elbo = E.tmean(i -> B.iw_bound(ks, prior, sch, Zte[i, :], 1, Xoshiro(10i)), ntest)
        iw = E.tmean(i -> B.iw_bound(ks, prior, sch, Zte[i, :], S, Xoshiro(10i)), ntest)
        ed = E.energy_distance(E.generate(ks, prior, sch, ngen, Xoshiro(6)), Zte)
        push!(rows, (; T, H, params = nparams(net) + T + 5, loss, elbo, iw, ed, time = t1))
        @printf("  T = %3d, H = %3d (%5d parameters): ELBO gap %.3f, IW bound (S = %d) gap %.3f, energy distance %.4f; loss %.3f (%.0f s)\n",
                T, H, rows[end].params, truth - elbo, S, truth - iw, ed, loss, t1)
    end
    return (; name, truth, rows)
end

main() = [run(name) for name in (:moons, :ring)]

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    NeuralReference.main()
end
