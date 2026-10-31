"""
Experiment E4(a) of `vault/Design/Minimal Stochastic Factors.md`: a diffusion read as a chain of
directed switching kernels, on Gaussian-mixture data, where Proposition 8 gives the exact reverse
kernels.

Data: a three-component Gaussian mixture in two dimensions. Forward process: VP, T steps of
z_t = √(1-β) z_{t-1} + √β ε. Reverse kernels are directed switching kernels
p(z_{t-1} | z_t) = Σ_k g_k(z_t) N(z_{t-1}; G_k z_t + h_k, C_k), g_k ∝ w_k N(z_t; m_k, S_k)
(the normalised, game reading of the note's §8).

1. Oracle check. With the exact kernels of Proposition 8 and the exact q(z_T), the model's joint
   over z_{0:T} is the forward process's, so every single sample of the evidence bound equals
   log q_0(z_0).
2. Learned chains, scored by the held-out evidence bound (a lower bound on the model's log p(z_0),
   same forward draws for every model) and by the energy distance of generated samples:
   - shared, time-conditioned: π_k, μ_k, Σ_k learned once by the per-step denoising objective,
     every step's kernel derived from the schedule (17 parameters);
   - free per step: each step's kernel fitted on its own (a K_t-component Gaussian mixture on the
     pairs (z_{t-1}, z_t) by EM, then conditioned), K_t ∈ {3, 2, 1};
   - one-shot: a 3-component mixture fitted to z_0 by EM (no diffusion).

Run with `julia --project=. examples/switching_reverse_chain_e4a.jl`.
"""
module SwitchingE4a

using LinearAlgebra, Random, Printf
import ForwardDiff

# --- data and schedule ---------------------------------------------------------------------------

const MU = [[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]]
rot(a) = [cos(a) -sin(a); sin(a) cos(a)]
const SIG = [rot(0.4) * Diagonal([0.6, 0.08]) * rot(0.4)', rot(-0.7) * Diagonal([0.3, 0.05]) * rot(-0.7)', Matrix(Diagonal([0.05, 0.4]))]
const PI = [0.5, 0.3, 0.2]

function sample_gmm(π, μ, Σ, n, rng)
    L = [cholesky(Symmetric(S)).L for S in Σ]
    c = cumsum(π)
    Z = zeros(n, 2)
    for i in 1:n
        k = findfirst(>=(rand(rng)), c)
        Z[i, :] = μ[k] + L[k] * randn(rng, 2)
    end
    return Z
end

schedule(T, β) = (βs = fill(β, T); α = [1.0; sqrt.(cumprod(1 .- βs))]; (; T, βs, α))   # α[t+1] = α_t

logN(z, m, S) = (d = z - m; -(dot(d, S \ d) + logdet(S) + length(z) * log(2π)) / 2)
function logsumexp(v)
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

# --- a directed switching kernel -----------------------------------------------------------------

struct Kernel{T}
    logw::Vector{T}; m::Vector{Vector{T}}; S::Vector{Matrix{T}}      # gate on z_t
    G::Vector{Matrix{T}}; h::Vector{Vector{T}}; C::Vector{Matrix{T}}  # expert z_{t-1} | z_t, k
end
function kernel_logpdf(k::Kernel, zprev, zt)
    lg = [k.logw[j] + logN(zt, k.m[j], k.S[j]) for j in eachindex(k.logw)]
    lg = lg .- logsumexp(lg)
    return logsumexp([lg[j] + logN(zprev, k.G[j] * zt + k.h[j], k.C[j]) for j in eachindex(lg)])
end
function kernel_sample(k::Kernel, zt, rng)
    lg = [k.logw[j] + logN(zt, k.m[j], k.S[j]) for j in eachindex(k.logw)]
    p = exp.(lg .- logsumexp(lg))
    j = findfirst(>=(rand(rng)), cumsum(p))
    j === nothing && (j = length(p))
    return k.G[j] * zt + k.h[j] + cholesky(Symmetric(k.C[j])).L * randn(rng, 2)
end

"Kernel of a joint Gaussian mixture on (z_{t-1}, z_t): condition each component on z_t."
function kernel_from_joint(w, M, V)
    K = length(w)
    T = eltype(M[1])
    logw = log.(w); m = [M[j][3:4] for j in 1:K]; S = [V[j][3:4, 3:4] for j in 1:K]
    G = [V[j][1:2, 3:4] / V[j][3:4, 3:4] for j in 1:K]
    h = [M[j][1:2] - G[j] * M[j][3:4] for j in 1:K]
    C = [Matrix{T}(Symmetric(V[j][1:2, 1:2] - G[j] * V[j][3:4, 1:2])) for j in 1:K]
    return Kernel(collect(T, logw), m, S, G, h, C)
end

"Proposition 8: the exact reverse kernel at step t for data GMM (π, μ, Σ)."
function exact_kernel(π, μ, Σ, sch, t)
    a = sqrt(1 - sch.βs[t]); αp, αt = sch.α[t], sch.α[t+1]
    K = length(π)
    M = Vector{Vector{eltype(μ[1])}}(undef, K); V = Vector{Matrix{eltype(Σ[1])}}(undef, K)
    for j in 1:K
        Vp = αp^2 * Σ[j] + (1 - αp^2) * I
        Vt = a^2 * Vp + sch.βs[t] * I
        M[j] = [αp * μ[j]; αt * μ[j]]
        V[j] = [Vp a*Vp; a*Vp Vt]
    end
    return kernel_from_joint(π, M, V)
end
prior_T(π, μ, Σ, sch) = (π, [sch.α[end] * m for m in μ], [sch.α[end]^2 * S + (1 - sch.α[end]^2) * I for S in Σ])
gmm_logpdf(π, μ, Σ, z) = logsumexp([log(π[j]) + logN(z, μ[j], Σ[j]) for j in eachindex(π)])

# --- the evidence bound, per sample --------------------------------------------------------------

"Forward draws: for each z_0 and draw, the path z_0, …, z_T (a T+1 × 2 matrix)."
function forward_paths(Z0, sch, rng)
    map(axes(Z0, 1)) do i
        P = zeros(sch.T + 1, 2); P[1, :] = Z0[i, :]
        for t in 1:sch.T
            P[t+1, :] = sqrt(1 - sch.βs[t]) * P[t, :] + sqrt(sch.βs[t]) * randn(rng, 2)
        end
        P
    end
end
"log p(z_T) + Σ log p(z_{t-1}|z_t) − Σ log q(z_t|z_{t-1}) along one path: a bound on log p(z_0)."
function elbo_path(kernels, prior, sch, P)
    l = gmm_logpdf(prior..., P[end, :])
    for t in 1:sch.T
        l += kernel_logpdf(kernels[t], P[t, :], P[t+1, :])
        l -= logN(P[t+1, :], sqrt(1 - sch.βs[t]) * P[t, :], sch.βs[t] * Matrix(1.0I, 2, 2))
    end
    return l
end

function generate(kernels, prior, sch, n, rng)
    π, μ, Σ = prior
    Z = sample_gmm(π, μ, Σ, n, rng)
    for t in sch.T:-1:1, i in 1:n
        Z[i, :] = kernel_sample(kernels[t], Z[i, :], rng)
    end
    return Z
end

# --- fitting -------------------------------------------------------------------------------------

"EM for a K-component Gaussian mixture on the rows of X (any dimension), with restarts."
function em_gmm(X, K; iters = 150, restarts = 3, floor = 1e-6, seed = 1)
    n, d = size(X)
    best, bestll = nothing, -Inf
    for r in 1:restarts
        rng = Xoshiro(seed + r)
        μ = [X[rand(rng, 1:n), :] for _ in 1:K]; V = [cov(X) for _ in 1:K]; w = fill(1 / K, K)
        ll = -Inf
        for _ in 1:iters
            L = [log(w[j]) + logN(X[i, :], μ[j], V[j]) for i in 1:n, j in 1:K]
            M = maximum(L; dims = 2); ll = sum(M .+ log.(sum(exp.(L .- M); dims = 2))) / n
            R = exp.(L .- M); R ./= sum(R; dims = 2)
            for j in 1:K
                nj = sum(R[:, j]); w[j] = max(nj / n, 1e-12)
                μ[j] = vec(sum(R[:, j] .* X; dims = 1)) / nj
                D = X .- μ[j]'
                V[j] = Matrix(Symmetric((D' * (R[:, j] .* D)) / nj)) + floor * I
            end
        end
        ll > bestll && ((best, bestll) = ((copy(w), deepcopy(μ), deepcopy(V)), ll))
    end
    return best
end
cov(X) = (D = X .- sum(X; dims = 1) / size(X, 1); Matrix(Symmetric(D' * D / size(X, 1))) + 1e-6I)

"Free per-step kernels: a K-component GMM on the pairs (z_{t-1}, z_t) of each step, conditioned.
The steps are independent, so they are fitted on parallel threads (deterministic: seeds per step)."
function fit_free(paths, sch, K)
    tasks = map(1:sch.T) do t
        Threads.@spawn begin
            X = reduce(vcat, [[P[t, :]' P[t+1, :]'] for P in paths])
            w, M, V = em_gmm(X, K; seed = 10t)
            kernel_from_joint(w, M, V)
        end
    end
    return fetch.(tasks)
end
"Mean of f over 1:n, computed on parallel threads."
function tmean(f, n)
    chunks = Iterators.partition(1:n, cld(n, Threads.nthreads()))
    return sum(fetch.([Threads.@spawn sum(f(i) for i in c) for c in chunks])) / n
end
free_prior(paths, sch) = (w = [1.0]; X = reduce(vcat, [P[end, :]' for P in paths]);
    ([1.0], [vec(sum(X; dims = 1)) / size(X, 1)], [cov(X)]))

# shared parameterisation: θ = (π logits (3), μ (3×2), Cholesky of Σ (3×3: l11, l21, l22 with logs))
function unpack(θ)
    π = exp.(θ[1:3] .- logsumexp(θ[1:3]))
    μ = [θ[3+2j-1:3+2j] for j in 1:3]
    Σ = map(1:3) do j
        l11, l21, l22 = exp(θ[9+3j-2]), θ[9+3j-1], exp(θ[9+3j])
        L = [l11 0; l21 l22]
        L * L' + 1e-6I
    end
    return π, μ, Σ
end
function shared_objective(θ, paths, sch)
    π, μ, Σ = unpack(θ)
    ks = [exact_kernel(π, μ, Σ, sch, t) for t in 1:sch.T]
    pr = prior_T(π, μ, Σ, sch)
    return sum(elbo_path(ks, pr, sch, P) for P in paths) / length(paths)
end
function fit_shared(paths, sch; steps = 300, lr = 0.05, batch = 400, seed = 3)
    rng = Xoshiro(seed)
    θ = [0.0, 0.0, 0.0, -1.0, -0.5, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    m, v = zero(θ), zero(θ)
    for s in 1:steps
        B = paths[rand(rng, eachindex(paths), batch)]
        g = ForwardDiff.gradient(x -> -shared_objective(x, B, sch), θ)
        m = 0.9m + 0.1g; v = 0.999v + 0.001g .^ 2
        θ = θ - lr * (m / (1 - 0.9^s)) ./ (sqrt.(v / (1 - 0.999^s)) .+ 1e-8)
    end
    return θ
end

# --- energy distance -----------------------------------------------------------------------------

function energy_distance(X, Y)
    dxy = sum(norm(X[i, :] - Y[j, :]) for i in axes(X, 1), j in axes(Y, 1)) / (size(X, 1) * size(Y, 1))
    dxx = sum(norm(X[i, :] - X[j, :]) for i in axes(X, 1), j in axes(X, 1)) / size(X, 1)^2
    dyy = sum(norm(Y[i, :] - Y[j, :]) for i in axes(Y, 1), j in axes(Y, 1)) / size(Y, 1)^2
    return 2dxy - dxx - dyy
end

# --- the experiment ------------------------------------------------------------------------------

function run(; T = 20, β = 0.2, ntrain = 4000, ntest = 1000, ngen = 1500, Ks = (3, 2, 1), shared = true)
    sch = schedule(T, β)
    Ztr = sample_gmm(PI, MU, SIG, ntrain, Xoshiro(1))
    Zte = sample_gmm(PI, MU, SIG, ntest, Xoshiro(2))
    paths_tr = forward_paths(Ztr, sch, Xoshiro(3))
    paths_te = forward_paths(Zte, sch, Xoshiro(4))          # the same forward draws for every model
    truth = [gmm_logpdf(PI, MU, SIG, Zte[i, :]) for i in axes(Zte, 1)]

    # 1. oracle: exact kernels and exact prior make every path's bound equal log q_0(z_0)
    exact = [exact_kernel(PI, MU, SIG, sch, t) for t in 1:T]
    exact_prior = prior_T(PI, MU, SIG, sch)
    oracle_gap = maximum(abs(elbo_path(exact, exact_prior, sch, paths_te[i]) - truth[i]) for i in eachindex(paths_te))

    models = Dict{String,Any}()
    θ = shared ? fit_shared(paths_tr, sch) : nothing
    if shared
        π̂, μ̂, Σ̂ = unpack(θ)
        models["shared, time-conditioned (17)"] = ([exact_kernel(π̂, μ̂, Σ̂, sch, t) for t in 1:T], prior_T(π̂, μ̂, Σ̂, sch))
    end
    for K in Ks
        np = K * (1 + 4 + 10) - 1                              # weights, means and covariances of a 4-d GMM
        models["free per step, K=$K ($(np * T))"] = (fit_free(paths_tr, sch, K), free_prior(paths_tr, sch))
    end
    w1, μ1, Σ1 = em_gmm(Ztr, 3)
    oneshot = mean_(gmm_logpdf(w1, μ1, Σ1, Zte[i, :]) for i in axes(Zte, 1))

    floor_ed = energy_distance(sample_gmm(PI, MU, SIG, ngen, Xoshiro(5)), Zte)
    rows = map(sort(collect(keys(models)))) do name
        ks, pr = models[name]
        elbo = tmean(i -> elbo_path(ks, pr, sch, paths_te[i]), length(paths_te))
        ed = energy_distance(generate(ks, pr, sch, ngen, Xoshiro(6)), Zte)
        (; name, elbo, ed)
    end
    exact_ed = energy_distance(generate(exact, exact_prior, sch, ngen, Xoshiro(6)), Zte)
    shared || return (; oracle_gap, truth = mean_(truth), oneshot, floor_ed, exact_ed, rows, shared_err = nothing)
    perm = [argmin([norm(m - μt) for m in μ̂]) for μt in MU]
    return (; oracle_gap, truth = mean_(truth), oneshot, floor_ed, exact_ed, rows,
            shared_err = (π = maximum(abs, π̂[perm] - PI), μ = maximum(norm(μ̂[perm][j] - MU[j]) for j in 1:3),
                          Σ = maximum(opnorm(Σ̂[perm][j] - SIG[j]) for j in 1:3)))
end
mean_(itr) = (s = 0.0; n = 0; for x in itr; s += x; n += 1; end; s / n)

function main()
    r = run()
    println("E4(a): three-component Gaussian mixture, T = 20 steps of β = 0.2")
    @printf("  oracle: exact kernels and prior, max |bound − log q0(z0)| over 1000 paths = %.2e\n", r.oracle_gap)
    @printf("  held-out mean log q0(z0) of the truth %.3f; one-shot 3-component GMM by EM %.3f\n", r.truth, r.oneshot)
    @printf("  energy distance of generated samples: data vs data %.4f, exact kernels %.4f\n", r.floor_ed, r.exact_ed)
    for row in r.rows
        @printf("  %-34s held-out bound %.3f   energy distance %.4f\n", row.name, row.elbo, row.ed)
    end
    @printf("  shared fit: max |π̂ − π| %.3f, max ‖μ̂ − μ‖ %.3f, max ‖Σ̂ − Σ‖ %.3f\n", r.shared_err.π, r.shared_err.μ, r.shared_err.Σ)
    # small steps: the same α_T in 100 steps, free per-step kernels only
    β100 = 1 - (0.8^20)^(1 / 100)
    s = run(; T = 100, β = β100, Ks = (2, 1), shared = false)
    @printf("E4(a), small steps: T = 100 steps of β = %.4f (same α_T)\n", β100)
    @printf("  oracle gap %.2e; energy distance data vs data %.4f, exact kernels %.4f\n", s.oracle_gap, s.floor_ed, s.exact_ed)
    for row in s.rows
        @printf("  %-34s held-out bound %.3f   energy distance %.4f\n", row.name, row.elbo, row.ed)
    end
    return (; r, s)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SwitchingE4a.main()
end
