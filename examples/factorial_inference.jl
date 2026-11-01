"""
Inference and learning for E3's additive wiring when the exact posterior is too large:
`vault/Design/Minimal Stochastic Factors.md` §9, §10 (E6). The model is
z = c + Σ_l k_l δ_l + ε, with ε ~ N(0, Σ) and independent bits k_l ~ Bernoulli(p_l), in d
dimensions with L switches. Its exact posterior has 2^L states, so beyond L ≈ 15 learning needs
approximate inference. Three E-steps, each followed by the same closed-form M-step:

- **exact**: enumerate the 2^L configurations (E3's `em_additive`, in d dimensions);
- **graph (collapsed BP)**: the star graph z — sum node — u_l — switch prior, run by Mycelium. The
  sum node is a `LinearConstraintFactor`. Its message to u_l collapses the other switches' mixture
  messages to their first two moments (`_moments` of a `MixtureBelief`, assumed-density
  filtering), so each bit's posterior is exact *given* Gaussian stand-ins for the others. The
  graph is a tree, so one pass is the fixed point. The E-step uses a vectorised copy of these
  messages, checked against Mycelium's `infer!` on the star graph;
- **mean field**: q(k) = Π_l q_l, coordinate ascent on the evidence bound (the factorial
  "cooperative vector quantiser" update), iterated to convergence.

The approximate E-steps give factorised posteriors, so the M-step uses E[k_l k_m] = q_l q_m for
l ≠ m. Every fitted model is scored by its exact held-out log-likelihood (enumeration; L ≤ 12 here).
Run with `julia -t auto --project=. examples/factorial_inference.jl`.
"""
module FactorialInference

using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, Printf
using LenticulumCore: MixtureBelief
import LuxCore

# --- the model -----------------------------------------------------------------------------------

"Parameters: W = [c δ_1 … δ_L] (d × (L+1)), bit probabilities p (L), noise covariance Σ (d × d)."
struct Model
    W::Matrix{Float64}; p::Vector{Float64}; Σ::Matrix{Float64}
end
nparams(d, L) = d * (L + 1) + L + d * (d + 1) ÷ 2

function truth(d, L; seed = 1)
    rng = Xoshiro(seed)
    δ = [2.5 .* normalize(randn(rng, d)) for _ in 1:L]
    W = hcat(zeros(d), δ...)
    p = 0.3 .+ 0.4 .* rand(rng, L)
    return Model(W, p, Matrix(0.25I, d, d))
end

function sample(m::Model, n, rng)
    d, L = size(m.W, 1), length(m.p)
    K = rand(rng, n, L) .< m.p'
    X = (m.W * vcat(ones(1, n), K'))' .+ randn(rng, n, d) * cholesky(Symmetric(m.Σ)).U
    return X
end

configs(L) = reduce(hcat, [digits(i; base = 2, pad = L) for i in 0:2^L-1])      # L × 2^L

"Exact log-likelihood of every row of X, and (optionally) the posterior over configurations."
function exact_loglik_matrix(m::Model, X; Kc = configs(length(m.p)))
    Φ = vcat(ones(1, size(Kc, 2)), Kc)
    C = cholesky(Symmetric(m.Σ)); U = C.L \ Matrix(X'); V = C.L \ (m.W * Φ)
    d = size(X, 2)
    lw = vec(log.(m.p)' * Kc .+ log1p.(-m.p)' * (1 .- Kc))
    Lg = lw' .- (d * log(2π) + logdet(C)) / 2 .- (sum(abs2, U; dims = 1)' .- 2 * (U' * V) .+ sum(abs2, V; dims = 1)) ./ 2
    mx = maximum(Lg; dims = 2); s = sum(exp.(Lg .- mx); dims = 2)
    return vec(mx .+ log.(s)), exp.(Lg .- mx) ./ s
end
exact_loglik(m::Model, X) = sum(first(exact_loglik_matrix(m, X))) / size(X, 1)

# --- E-steps: each returns the bit posteriors' sufficient statistics -------------------------------

"Σ_i E[φ_i], Σ_i E[φ_i φ_iᵀ], Σ_i x_i E[φ_i]ᵀ for φ = [1; k], from the exact posterior."
function estep_exact(m::Model, X)
    Kc = configs(length(m.p)); Φ = vcat(ones(1, size(Kc, 2)), Kc)
    _, R = exact_loglik_matrix(m, X; Kc)
    Rk = vec(sum(R; dims = 1))
    Eφ = Φ * R'                                                     # (L+1) × n
    return (; Sφφ = Φ * Diagonal(Rk) * Φ', Sxφ = X' * Eφ', Eφ)
end

"Statistics of a factorised posterior with marginals Q (n × L): E[k_l k_m] = q_l q_m for l ≠ m."
function factorised_stats(Q, X)
    n, L = size(Q)
    Eφ = vcat(ones(1, n), Q')
    Sφφ = Eφ * Eφ'
    Sφφ[diagind(Sφφ)[2:end]] .= vec(sum(Q; dims = 1))                 # E[k_l²] = q_l
    return (; Sφφ, Sxφ = X' * Eφ', Eφ)
end

"""
The graph's bit posteriors, vectorised: the sum node's message to u_l with the other switches'
mixture messages collapsed to moments, times the switch prior. ε is the switch components' width.
"""
function bp_marginals(m::Model, X; ε = 1e-4)
    d = size(X, 2); L = length(m.p)
    c = m.W[:, 1]; δ = [m.W[:, l+1] for l in 1:L]
    μall = sum(m.p[l] .* δ[l] for l in 1:L)
    Vall = sum(m.p[l] * (1 - m.p[l]) .* δ[l] * δ[l]' for l in 1:L) + L * ε * I
    Q = zeros(size(X, 1), L)
    for l in 1:L
        μo = μall - m.p[l] .* δ[l]
        R = Symmetric(m.Σ + Vall - m.p[l] * (1 - m.p[l]) .* δ[l] * δ[l]')     # includes u_l's own ε
        C = cholesky(R)
        r0 = C.L \ (X' .- (c + μo)); r1 = r0 .- (C.L \ δ[l])
        a = log(m.p[l]) .- vec(sum(abs2, r1; dims = 1)) ./ 2
        b = log1p(-m.p[l]) .- vec(sum(abs2, r0; dims = 1)) ./ 2
        Q[:, l] = 1 ./ (1 .+ exp.(b .- a))
    end
    return Q
end
estep_bp(m::Model, X) = factorised_stats(bp_marginals(m, X), X)

"Mean field: q_l ← σ(logit p_l + δ_lᵀΣ⁻¹(x − c − Σ_{m≠l} q_m δ_m) − δ_lᵀΣ⁻¹δ_l / 2), swept to convergence."
function mf_marginals(m::Model, X; sweeps = 100, tol = 1e-8)
    L = length(m.p); Σi = inv(Symmetric(m.Σ))
    c = m.W[:, 1]; Δ = m.W[:, 2:end]                                # d × L
    G = Δ' * Σi * Δ                                                 # L × L
    B = (X .- c') * (Σi * Δ)                                        # n × L
    Q = repeat(m.p', size(X, 1))
    logitp = log.(m.p ./ (1 .- m.p))
    for _ in 1:sweeps
        change = 0.0
        for l in 1:L
            h = logitp[l] .+ B[:, l] .- Q * G[:, l] .+ Q[:, l] .* G[l, l] .- G[l, l] / 2
            qn = 1 ./ (1 .+ exp.(-h))
            change = max(change, maximum(abs.(qn .- Q[:, l]))); Q[:, l] = qn
        end
        change < tol && break
    end
    return Q
end
estep_mf(m::Model, X) = factorised_stats(mf_marginals(m, X), X)

"""
Gaussian expectation propagation on the star graph, for one observation x. Each switch prior
Σ_k π_k N(u_l; k δ_l, εI) is approximated by a Gaussian site (moment form a_l, B_l). The cavity of
u_l is the sum node's message given the other sites (exact on this tree), the tilted distribution
is the two-component prior times the cavity, and the new site is its moment projection divided by
the cavity, damped in natural parameters; an update whose site precision is not positive definite,
or whose tilted or cavity covariance fails to factorise in floating point, is skipped. Returns the
tilted bit marginals w and the bits' second moments from the joint Gaussian over (u_1, …, u_L),
E[k_l k_m] = δ̂_lᵀ E[u_l u_mᵀ] δ̂_m with δ̂ = δ / ‖δ‖² (l ≠ m).
"""
function ep_point(m::Model, x; sweeps = 30, damp = 0.5, ε = 1e-4, tol = 1e-8)
    d, L = size(m.W, 1), length(m.p)
    c = m.W[:, 1]; δ = [m.W[:, l+1] for l in 1:L]; Id = Matrix(1.0I, d, d)
    a = [m.p[l] .* δ[l] for l in 1:L]
    B = [m.p[l] * (1 - m.p[l]) .* δ[l] * δ[l]' + ε * Id for l in 1:L]
    w = copy(m.p)
    r = x - c
    for _ in 1:sweeps
        change = 0.0
        for l in 1:L
            μc = r - (sum(a) - a[l]); Sc = Symmetric(m.Σ + sum(B) - B[l])
            Cc = cholesky(Sc; check = false); issuccess(Cc) || continue
            Sci = inv(Cc)
            lg = [log1p(-m.p[l]) + E_logN(zeros(d), μc, Sc + ε * Id), log(m.p[l]) + E_logN(δ[l], μc, Sc + ε * Id)]
            wl = 1 / (1 + exp(lg[1] - lg[2]))
            C = inv(Symmetric(Id ./ ε + Sci))
            m0 = C * (Sci * μc); m1 = C * (δ[l] ./ ε + Sci * μc)
            M = (1 - wl) .* m0 + wl .* m1
            V = Symmetric(C + (1 - wl) .* (m0 * m0') + wl .* (m1 * m1') - M * M')
            Cv = cholesky(V; check = false); issuccess(Cv) || continue      # rounding at ε-scale covariances
            Vi = inv(Cv)
            Pn = Vi - Sci; hn = Vi * M - Sci * μc
            Po = inv(Symmetric(B[l])); ho = Po * a[l]
            Pd = Symmetric(damp .* Pn + (1 - damp) .* Po); hd = damp .* hn + (1 - damp) .* ho
            isposdef(Pd) || continue
            Bn = inv(Pd); an = Bn * hd
            change = max(change, maximum(abs.(an - a[l])), abs(wl - w[l]))
            a[l], B[l], w[l] = an, Matrix(Symmetric(Bn)), wl
        end
        change < tol && break
    end
    # the joint Gaussian over (u_1, …, u_L): sites times the sum node's likelihood
    Σi = inv(Symmetric(m.Σ))
    Λ = kron(ones(L, L), Σi)
    η = repeat(Σi * r, L)
    for l in 1:L
        Pl = inv(Symmetric(B[l])); idx = (l-1)*d+1:l*d
        Λ[idx, idx] .+= Pl; η[idx] .+= Pl * a[l]
    end
    S = inv(Symmetric(Λ)); μ = S * η
    Euu = S + μ * μ'
    Kk = zeros(L, L)
    for l in 1:L, k in 1:L
        Kk[l, k] = l == k ? w[l] : (δ[l] / sum(abs2, δ[l]))' * Euu[(l-1)*d+1:l*d, (k-1)*d+1:k*d] * (δ[k] / sum(abs2, δ[k]))
    end
    return w, Kk
end
E_logN(z, μ, S) = (D = z - μ; C = cholesky(Symmetric(S)); -(length(z) * log(2π) + logdet(C) + D' * (C \ D)) / 2)

function ep_marginals(m::Model, X)
    Q = zeros(size(X, 1), length(m.p))
    Threads.@threads for i in axes(X, 1)
        Q[i, :] = first(ep_point(m, X[i, :]))
    end
    return Q
end

"EP's E-step: tilted marginals and pair moments from the joint Gaussian."
function estep_ep(m::Model, X)
    n, L = size(X, 1), length(m.p)
    res = Vector{Any}(undef, n)
    Threads.@threads for i in 1:n
        res[i] = ep_point(m, X[i, :])
    end
    Q = reduce(vcat, [r[1]' for r in res])
    Eφ = vcat(ones(1, n), Q')
    Sφφ = zeros(L + 1, L + 1)
    Sφφ[1, 1] = n; Sφφ[1, 2:end] = Sφφ[2:end, 1] = vec(sum(Q; dims = 1))
    Sφφ[2:end, 2:end] = sum(r[2] for r in res)
    return (; Sφφ, Sxφ = X' * Eφ', Eφ)
end

"""
Mean field with a linear-response correction for the pair moments (Welling & Teh 2004; suggested
by Gemini's review). At the mean-field fixed point q_l = σ(a_l − Σ_{m≠l} G_lm q_m), G = ΔᵀΣ⁻¹Δ,
perturbing the natural parameters a gives Cov(k) ≈ ∂q/∂a = (D⁻¹ + G_off)⁻¹ with D = diag(q(1 − q))
and G_off the off-diagonal part of G. The M-step then uses E[k_l k_m] = q_l q_m + Cov_lm.
"""
function estep_mf_lr(m::Model, X)
    Q = mf_marginals(m, X)
    n, L = size(Q)
    Δ = m.W[:, 2:end]; G = Δ' * (Symmetric(m.Σ) \ Δ); Goff = G - Diagonal(diag(G))
    Eφ = vcat(ones(1, n), Q')
    Skk = zeros(L, L)
    for i in 1:n
        q = Q[i, :]; D = clamp.(q .* (1 .- q), 1e-12, Inf)
        C = inv(Symmetric(Matrix(Diagonal(1 ./ D)) + Goff))
        Skk .+= q * q' + C
        Skk[diagind(Skk)] .+= q .- (q .^ 2 .+ diag(C))      # the diagonal is E[k_l²] = q_l exactly
    end
    Sφφ = zeros(L + 1, L + 1)
    Sφφ[1, 1] = n; Sφφ[1, 2:end] = Sφφ[2:end, 1] = vec(sum(Q; dims = 1)); Sφφ[2:end, 2:end] = Skk
    return (; Sφφ, Sxφ = X' * Eφ', Eφ)
end

"A diagnostic: the exact marginals, but the M-step's pair moments factorised as for the approximations."
estep_exact_marginals(m::Model, X) = factorised_stats(last(exact_loglik_matrix(m, X)) * permutedims(configs(length(m.p))), X)

"The closed-form M-step from sufficient statistics."
function mstep(st, X; floor = 1e-6)
    n = size(X, 1)
    W = st.Sxφ / Symmetric(st.Sφφ + 1e-10I)
    Σ = (X' * X - W * st.Sxφ' - st.Sxφ * W' + W * st.Sφφ * W') / n
    p = clamp.(st.Sφφ[1, 2:end] ./ n, 1e-6, 1 - 1e-6)
    return Model(W, p, Matrix(Symmetric(Σ)) + floor * I)
end

function init(d, L, X, rng)
    pick() = X[rand(rng, axes(X, 1)), :]
    W = hcat(pick(), [pick() - pick() for _ in 1:L]...)
    return Model(W, fill(0.5, L), Matrix(minimum(eigvals(Symmetric(cov(X)))) / 4 * I, d, d))
end
cov(X) = (D = X .- sum(X; dims = 1) / size(X, 1); D' * D / size(X, 1))

"EM with the given E-step until the parameters stop moving (no exact likelihood inside the loop)."
function fit(estep, m::Model, X; iters = 200, tol = 1e-6)
    for _ in 1:iters
        mn = mstep(estep(m, X), X)
        Δ = max(maximum(abs.(mn.W .- m.W)), maximum(abs.(mn.p .- m.p)), maximum(abs.(mn.Σ .- m.Σ)))
        m = mn
        Δ < tol && break
    end
    return m
end

# --- the same graph in Mycelium ------------------------------------------------------------------

"""
The star graph for one observation x: z clamped by a data factor; one `LinearConstraintFactor`
z + Σ_l u′_l = c (u′_l = −u_l, so that every coefficient is I); each u′_l a `PriorFactor` emitting
the two-component switch mixture. Returns each switch's posterior probability of k_l = 1.
"""
function graph_marginals(m::Model, x; ε = 1e-4)
    d, L = size(m.W, 1), length(m.p)
    us = [Symbol(:u, l) for l in 1:L]
    b = GraphBuilder()
    variable!(b, :z, d); foreach(u -> variable!(b, u, d), us)
    dims = NamedTuple{(:z, us...)}(ntuple(_ -> d, L + 1))
    c = m.W[:, 1]
    factor!(b, :sum, LinearConstraintFactor(dims, d; noise = m.Σ, init_coeff = (rng, r, k) -> Matrix(1.0I, r, k),
                                            init_constant = (rng, r) -> copy(c)))
    factor!(b, :data, DataFactor(:z, x))
    connect!(b, :data, :z, :z; direction = Emitting())
    connect!(b, :sum, :z, :z; direction = Absorbing())
    for l in 1:L
        mix = MixtureBelief([Gaussian(zeros(d), ε * Matrix(1.0I, d, d)), Gaussian(-m.W[:, l+1], ε * Matrix(1.0I, d, d))],
                            [1 - m.p[l], m.p[l]])
        factor!(b, Symbol(:sw, l), PriorFactor(us[l], mix, _ -> 0.0))
        connect!(b, Symbol(:sw, l), us[l], us[l]); connect!(b, :sum, us[l], us[l])
    end
    g = build(b)
    ps = LuxCore.initialparameters(Xoshiro(0), g)
    marg, _, _, _ = infer!(g, tree_schedule(g), ps, LuxCore.initialstates(Xoshiro(0), g))
    return [LenticulumCore.mixture_weights(getfield(marg, u))[2] for u in us]
end

# --- the experiment ------------------------------------------------------------------------------

mean(v) = sum(v) / length(v)

const METHODS = (
    ("exact", estep_exact, true),
    ("exact marginals, factorised pairs", estep_exact_marginals, false),
    ("graph (ADF: one pass, prior moments)", estep_bp, true),
    ("mean field", estep_mf, true),
    ("mean field + linear response", estep_mf_lr, true),
    ("EP (joint Gaussian pairs)", estep_ep, false),
)

"""
For each truth seed and L: the graph check, the bit-marginal errors at the true parameters, and
for every E-step its EM fixed point from the true parameters. For the seeds in `random_seeds`, the
methods marked so are also run from `starts` random starts.
"""
function run(; d = 8, Ls = (4, 8, 12), seeds = 1:5, random_seeds = (1,), ntrain = 2000, ntest = 2000,
             starts = 64, iters = 200, ncheck = 20, methods = METHODS)
    rows = []
    for seed in seeds, L in Ls
        tm = truth(d, L; seed)
        X = sample(tm, ntrain, Xoshiro(10 + 100(seed - 1))); Xte = sample(tm, ntest, Xoshiro(11 + 100(seed - 1)))
        t_ll = exact_loglik(tm, Xte)
        Qg = reduce(vcat, [graph_marginals(tm, Xte[i, :])' for i in 1:ncheck])
        graph_err = maximum(abs.(Qg .- bp_marginals(tm, Xte[1:ncheck, :])))
        Qe = last(exact_loglik_matrix(tm, Xte)) * permutedims(configs(L))
        errs = (graph = mean(abs.(bp_marginals(tm, Xte) .- Qe)), mf = mean(abs.(mf_marginals(tm, Xte) .- Qe)),
                ep = mean(abs.(ep_marginals(tm, Xte) .- Qe)))
        Δ = tm.W[:, 2:end]; cosmax = maximum(abs(dot(Δ[:, a], Δ[:, b])) / (norm(Δ[:, a]) * norm(Δ[:, b])) for a in 1:L for b in a+1:L)
        @printf("seed %d, d = %d, L = %d (%d components): held-out log p of the truth %.3f; max |cos| between offsets %.2f\n",
                seed, d, L, 2^L, t_ll, cosmax)
        @printf("  graph (Mycelium infer!) vs its vectorised copy: max |Δq| %.1e; mean |q_l − exact| at the truth: graph %.3f, mean field %.3f, EP %.3f\n",
                graph_err, errs.graph, errs.mf, errs.ep)
        for (name, es, rnd) in methods
            t1 = @elapsed fixed = fit(es, tm, X; iters)
            gap_fixed = t_ll - exact_loglik(fixed, Xte)
            gap, t = NaN, NaN
            if rnd && seed in random_seeds
                t = @elapsed fits = fetch.([Threads.@spawn fit(es, init(d, L, X, Xoshiro(s)), X; iters) for s in 1:starts])
                gap = t_ll - exact_loglik(fits[argmax([exact_loglik(f, X) for f in fits])], Xte)
            end
            push!(rows, (; seed, L, method = name, gap_fixed, gap, time = t, time_fixed = t1, cosmax, errs))
            @printf("  %-38s from the truth: gap %.3f (%.1f s)%s\n", name, gap_fixed, t1,
                    isnan(gap) ? "" : @sprintf("   best of %d random starts: %.3f (%.1f s)", starts, gap, t))
        end
    end
    return rows
end

"The measured run: every method on seeds 1–3, every method but the slow EP on seeds 4 and 5."
main() = vcat(run(; seeds = 1:3), run(; seeds = 4:5, methods = Base.filter(m -> !startswith(m[1], "EP"), METHODS)))

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    FactorialInference.main()
end
