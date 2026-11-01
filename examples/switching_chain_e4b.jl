"""
Experiment E4(b) of `vault/Design/Minimal Stochastic Factors.md` (§8, §10): does a chain of small
directed switching steps beat a flat Gaussian mixture on data that are *not* a finite mixture?

Data with exact densities (a one-dimensional quadrature over the curve): two moons and a ring,
each a uniform measure on curves convolved with isotropic Gaussian noise. No finite mixture
contains them, so every model here is approximate.

Models, all trained on the same 4 000 points:

- **flat**: a K-component Gaussian mixture by EM (30 restarts), 6K − 1 parameters, exact log-likelihood;
- **chain**: T reverse steps, each a directed switching kernel with K outcomes fitted by
  conditional maximum likelihood (`SwitchingSmallSteps.fit_pairs`) on pairs built from the
  training points with fresh forward noise (the usual diffusion training data); the prior at T is
  a Gaussian fitted to the noised training points. T(15K − 6) + 5 identified parameters. Its marginal is a
  mixture over K^T paths with input-dependent gates, which has no closed form, so it is scored by
  the importance-weighted bound with S forward paths per test point, log (1/S) Σ_s p(z_{0:T}^s) /
  q(z_{1:T}^s | z_0): a lower bound on log p(z_0) in expectation that tightens as S grows.

So a chain that beats a flat mixture's exact log-likelihood beats it for real (up to Monte Carlo
error). Generated samples are also compared by the energy distance. Run with
`julia -t auto --project=. examples/switching_chain_e4b.jl`.
"""
module SwitchingChainE4b

using LinearAlgebra, Random, Printf

isdefined(Main, :SwitchingSmallSteps) || Base.include(Main, joinpath(@__DIR__, "switching_small_steps.jl"))
const SS = Main.SwitchingSmallSteps
const E = Main.SwitchingE4a

# --- data with exact densities -------------------------------------------------------------------

const MOON_NOISE = 0.1
const RING_RADIUS, RING_NOISE = 1.5, 0.15
const MOON_SHIFT = [0.5, 0.25]

moon_point(θ, upper) = upper ? [cos(θ), sin(θ)] - MOON_SHIFT : [1 - cos(θ), 0.5 - sin(θ)] - MOON_SHIFT
ring_point(θ) = RING_RADIUS .* [cos(θ), sin(θ)]

function sample_data(name, n, rng)
    Z = zeros(n, 2)
    for i in 1:n
        if name === :moons
            Z[i, :] = moon_point(π * rand(rng), rand(rng) < 0.5) + MOON_NOISE * randn(rng, 2)
        else
            Z[i, :] = ring_point(2π * rand(rng)) + RING_NOISE * randn(rng, 2)
        end
    end
    return Z
end

"Exact log-density by the midpoint rule on the curve parameter (spacing ≪ the noise)."
function data_logpdf(name, z; m = 4000)
    lse(v) = (mx = maximum(v); mx + log(sum(exp.(v .- mx))))
    lN(c, s) = -log(2π * s^2) - sum(abs2, z - c) / (2s^2)
    if name === :moons
        θs = ((1:m) .- 0.5) .* (π / m)
        return lse([lN(moon_point(θ, u), MOON_NOISE) for θ in θs for u in (true, false)]) - log(2m)
    else
        θs = ((1:m) .- 0.5) .* (2π / m)
        return lse([lN(ring_point(θ), RING_NOISE) for θ in θs]) - log(m)
    end
end

# --- the chain -----------------------------------------------------------------------------------

function fit_chain(Z0, sch, K; iters = 120, seed = 1)
    tasks = [Threads.@spawn SS.fit_pairs(SS.step_pairs(sch, t, Z0, Xoshiro(seed + 100t + K)), K; iters, seed = 7t) for t in 1:sch.T]
    kernels = [r[1] for r in fetch.(tasks)]
    a = sch.α[end]
    ZT = a .* Z0 .+ sqrt(1 - a^2) .* randn(Xoshiro(seed), size(Z0)...)
    prior = ([1.0], [vec(sum(ZT; dims = 1)) / size(ZT, 1)], [E.cov(ZT)])
    return kernels, prior
end

"The importance-weighted bound for one z_0, with S forward paths as the proposal."
function iw_bound(kernels, prior, sch, z0, S, rng)
    ls = [E.elbo_path(kernels, prior, sch, only(E.forward_paths(reshape(z0, 1, 2), sch, rng))) for _ in 1:S]
    return E.logsumexp(ls) - log(S)
end

nchain(T, K) = T * (SS.PER * K - 6) + 5   # identified parameters (`SwitchingSmallSteps.estimation_cost`)
mean(v) = sum(v) / length(v)

function run(name; ntrain = 4000, ntest = 1000, ngen = 1500, Ks_flat = (2, 3, 4, 8, 16, 32, 64),
             chains = [(T, K) for T in (5, 10, 20) for K in (2, 3, 4)], S = 64, iters = 120)
    Ztr = sample_data(name, ntrain, Xoshiro(1))
    Zte = sample_data(name, ntest, Xoshiro(2))
    truth = E.tmean(i -> data_logpdf(name, Zte[i, :]), ntest)
    floor_ed = E.energy_distance(sample_data(name, ngen, Xoshiro(5)), Zte)
    @printf("%s: held-out log p of the truth %.3f; energy distance data vs data %.4f\n", name, truth, floor_ed)
    rows = []
    gm = fetch.([Threads.@spawn E.em_gmm(Ztr, K; iters = 200, restarts = 30, floor = 1e-4, seed = 3) for K in Ks_flat])
    for (K, g) in zip(Ks_flat, gm)
        ll = E.tmean(i -> E.gmm_logpdf(g..., Zte[i, :]), ntest)
        ed = E.energy_distance(E.sample_gmm(g..., ngen, Xoshiro(6)), Zte)
        push!(rows, (; model = "flat, K = $K", params = 6K - 1, ll, elbo = NaN, ed))
        @printf("  %-22s %5d params   held-out log p %.3f (gap %.3f)                    energy distance %.4f\n",
                "flat, K = $K", 6K - 1, ll, truth - ll, ed)
    end
    αT2 = 0.8^20
    for (T, K) in chains
        sch = E.schedule(T, 1 - αT2^(1 / T))
        ks, prior = fit_chain(Ztr, sch, K; iters)
        iw = E.tmean(i -> iw_bound(ks, prior, sch, Zte[i, :], S, Xoshiro(10i)), ntest)
        elbo = E.tmean(i -> iw_bound(ks, prior, sch, Zte[i, :], 1, Xoshiro(10i + 1)), ntest)
        ed = E.energy_distance(E.generate(ks, prior, sch, ngen, Xoshiro(6)), Zte)
        m = "chain, T = $T, K = $K"
        push!(rows, (; model = m, params = nchain(T, K), ll = iw, elbo, ed))
        @printf("  %-22s %5d params   IW bound (S = %d) %.3f (gap %.3f), ELBO %.3f   energy distance %.4f\n",
                m, nchain(T, K), S, iw, truth - iw, elbo, ed)
    end
    return (; name, truth, floor_ed, rows)
end

"""
How loose is the importance-weighted bound? For a few chains, the bound with growing numbers of
forward paths S on the same test points. (With gates that depend on the state, the chain's marginal
is not a finite mixture, so there is no exact path sum to compare with; the bound increases to
log p(z_0) as S → ∞.)
"""
function bound_sweep(name; ntrain = 4000, ntest = 300, chains = [(5, 2), (5, 4), (20, 2), (20, 4)], Ss = (1, 64, 512, 4096), iters = 120)
    Ztr = sample_data(name, ntrain, Xoshiro(1))
    Zte = sample_data(name, ntest, Xoshiro(2))
    truth = E.tmean(i -> data_logpdf(name, Zte[i, :]), ntest)
    @printf("%s, %d test points: truth %.3f\n", name, ntest, truth)
    rows = []
    for (T, K) in chains
        sch = E.schedule(T, 1 - 0.8^(20 / T))
        ks, prior = fit_chain(Ztr, sch, K; iters)
        gaps = [truth - E.tmean(i -> iw_bound(ks, prior, sch, Zte[i, :], S, Xoshiro(10i)), ntest) for S in Ss]
        push!(rows, (; T, K, Ss, gaps))
        @printf("  T = %2d, K = %d: gap of the bound for S = %s: %s\n", T, K, join(Ss, ", "), join([@sprintf("%.3f", g) for g in gaps], ", "))
    end
    return (; name, truth, rows)
end

main() = ([run(name) for name in (:moons, :ring)], [bound_sweep(name) for name in (:moons, :ring)])

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SwitchingChainE4b.main()
end
