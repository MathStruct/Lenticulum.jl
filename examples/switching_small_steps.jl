"""
The fair small-step test of `vault/Design/Minimal Stochastic Factors.md` §8: can T small switching
steps with few outcomes replace a mixture with many components?

Prediction. As β → 0 each true reverse conditional q(z_{t-1} | z_t) becomes unimodal, so a step's
outcomes are needed only to represent the mean, i.e. the score field ∇log q_t, as a softmax-gated
piecewise-linear function. For a K-component mixture that function has K pieces (Proposition 8).
So with as many outcomes as components the bound should approach the truth as T grows, and with
fewer it should level off below it, at any step size.

Setup, chosen to isolate capacity from estimation noise (which E4(a)'s free per-step fits mixed
in): the data are the three-component mixture of `SwitchingE4a`; every step is fitted on fresh
pairs drawn from the true mixture and the forward process; each step's directed switching kernel
is fitted by **conditional** maximum likelihood of z_{t-1} given z_t (initialised from a joint EM
fit on the pairs, refined by Adam with ForwardDiff); steps are fitted on parallel threads. T ranges
over {10, 25, 50, 100} with the same α_T; the prior at T is the exact q(z_T), so only the kernels
differ. Run with `julia -t auto --project=. examples/switching_small_steps.jl`.
"""
module SwitchingSmallSteps

using LinearAlgebra, Random, Printf
import ForwardDiff

isdefined(Main, :SwitchingE4a) || Base.include(Main, joinpath(@__DIR__, "switching_reverse_chain_e4a.jl"))
const E = Main.SwitchingE4a

# --- a kernel's parameters as a flat vector ------------------------------------------------------

const PER = 1 + 2 + 3 + 4 + 2 + 3        # log w, gate mean, gate Cholesky, G, h, expert Cholesky

chol3(L) = [log(L[1, 1]), L[2, 1], log(L[2, 2])]
function from_chol3(v)
    L = [exp(v[1]) 0; v[2] exp(v[3])]
    return L * L' + 1e-8I
end
function pack(k::E.Kernel)
    reduce(vcat, map(eachindex(k.logw)) do j
        [k.logw[j]; k.m[j]; chol3(cholesky(Symmetric(k.S[j])).L); vec(k.G[j]); k.h[j]; chol3(cholesky(Symmetric(k.C[j])).L)]
    end)
end
function unpack(θ, K)
    blk(j) = θ[(j-1)*PER+1:j*PER]
    E.Kernel([blk(j)[1] for j in 1:K], [blk(j)[2:3] for j in 1:K], [from_chol3(blk(j)[4:6]) for j in 1:K],
             [reshape(blk(j)[7:10], 2, 2) for j in 1:K], [blk(j)[11:12] for j in 1:K], [from_chol3(blk(j)[13:15]) for j in 1:K])
end
cond_ll(θ, K, X) = sum(E.kernel_logpdf(unpack(θ, K), X[i, 1:2], X[i, 3:4]) for i in axes(X, 1)) / size(X, 1)

"Pairs (z_{t-1}, z_t) at step t from data rows `Z0`, with fresh forward noise."
function step_pairs(sch, t, Z0::AbstractMatrix, rng)
    n = size(Z0, 1)
    a = sch.α[t]                                     # z_{t-1} = α_{t-1} z_0 + σ_{t-1} ε
    Zp = a .* Z0 .+ sqrt(1 - a^2) .* randn(rng, n, 2)
    Zt = sqrt(1 - sch.βs[t]) .* Zp .+ sqrt(sch.βs[t]) .* randn(rng, n, 2)
    return hcat(Zp, Zt)
end
"Pairs at step t from fresh draws of the true mixture."
step_pairs(sch, t, n::Integer, rng) = step_pairs(sch, t, E.sample_gmm(E.PI, E.MU, E.SIG, n, rng), rng)

"""
A directed switching kernel with `K` outcomes for the pairs `X` (rows z_{t-1}, z_t): joint EM,
then conditional maximum likelihood by Adam. Returns the kernel and the conditional
log-likelihood per pair before and after the refit.
"""
function fit_pairs(X, K; iters = 120, lr = 0.02, seed = 1)
    w, M, V = E.em_gmm(X, K; iters = 100, restarts = 2, seed)
    θ = pack(E.kernel_from_joint(w, M, V))
    joint_ll = cond_ll(θ, K, X)
    m, v = zero(θ), zero(θ)
    for s in 1:iters
        g = ForwardDiff.gradient(x -> -cond_ll(x, K, X), θ)
        m = 0.9m + 0.1g; v = 0.999v + 0.001g .^ 2
        θ = θ - lr * (m / (1 - 0.9^s)) ./ (sqrt.(v / (1 - 0.999^s)) .+ 1e-8)
    end
    return unpack(θ, K), joint_ll, cond_ll(θ, K, X)
end

"One step on fresh pairs from the true mixture."
fit_step(sch, t, K; n = 6000, iters = 120, lr = 0.02) = fit_pairs(step_pairs(sch, t, n, Xoshiro(100t + K)), K; iters, lr, seed = 7t)

function fit_chain(sch, K; kw...)
    tasks = [Threads.@spawn fit_step(sch, t, K; kw...) for t in 1:sch.T]
    res = fetch.(tasks)
    return [r[1] for r in res], sum(r[3] - r[2] for r in res)   # kernels, total gain of the conditional refit
end

"""
The asymptotic estimation cost of the chain: each step's conditional maximum-likelihood fit on `n`
pairs loses about d/(2n) nats of expected log-likelihood, so T steps lose T·d/(2n). Here d = 15K − 6
is the number of parameters the conditional likelihood identifies: 9 per affine expert, and 6 per
gate log-odds against a reference outcome. Adding a common quadratic in z_t (6 coefficients in 2-D)
to every gate's log-numerator leaves the softmax unchanged, so one gate's 6 parameters are not
identified; K = 1 gives 9, an affine Gaussian conditional. (The first version counted 15K − 1; the
second round of reviews corrected it, §13 of the design note.) The formula is the regular
asymptotic one: exact for a well-specified model as n → ∞ (K = 3 here, by Proposition 8), not for
the misspecified K = 2, and it degrades where the gates become nearly flat (large t).
"""
estimation_cost(T, K, n) = T * (PER * K - 6) / (2n)

function run(; Ts = (10, 25, 50, 100), Ks = (2, 3), ntest = 1000, ngen = 1500, n = 6000, iters = 120)
    αT2 = 0.8^20
    Zte = E.sample_gmm(E.PI, E.MU, E.SIG, ntest, Xoshiro(2))
    truth = sum(E.gmm_logpdf(E.PI, E.MU, E.SIG, Zte[i, :]) for i in axes(Zte, 1)) / ntest
    floor_ed = E.energy_distance(E.sample_gmm(E.PI, E.MU, E.SIG, ngen, Xoshiro(5)), Zte)
    rows = []
    for T in Ts
        sch = E.schedule(T, 1 - αT2^(1 / T))
        paths = E.forward_paths(Zte, sch, Xoshiro(4))
        prior = E.prior_T(E.PI, E.MU, E.SIG, sch)
        for K in Ks
            ks, gain = fit_chain(sch, K; n, iters)
            bound = E.tmean(i -> E.elbo_path(ks, prior, sch, paths[i]), length(paths))
            ed = E.energy_distance(E.generate(ks, prior, sch, ngen, Xoshiro(6)), Zte)
            est = estimation_cost(T, K, n)
            push!(rows, (; T, K, bound, ed, gain, est))
            @printf("  T = %3d, K = %d: held-out bound %.3f (truth %.3f; gap %.3f, estimation cost %.3f), energy distance %.4f (floor %.4f), conditional refit gain %.3f\n",
                    T, K, bound, truth, truth - bound, est, ed, floor_ed, gain)
        end
    end
    return (; truth, floor_ed, rows)
end

main() = (println("Small-step test: three-component mixture, same α_T, conditional per-step fits"); run())

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SwitchingSmallSteps.main()
end
