"""
Experiment E3 of `vault/Design/Minimal Stochastic Factors.md` (§8, §10): where does a composition
of binary switching factors beat one flat mixture?

Proposition 7: L binary switches give at most 2^L Gaussian components, so composition buys
parameter efficiency, not a larger class, and only for data whose components have the structure
the composition imposes. Three data sets in two dimensions, each with an exact density:

- `grid3`, `grid4`: 2^L modes at c + Σ_l k_l δ_l (k_l ∈ {0, 1}), independent bits, one shared
  covariance: a product structure, exactly the additive composition's class;
- `ring`: 8 equal modes on a circle, which is not of that form (8 sums of 3 offsets are the
  vertices of a projected cube, at most 6 of them on the convex hull).

Models, all with exact likelihoods:

- **additive**: z = c + Σ_l u_l + ε, u_l ∈ {0, δ_l} a binary switching factor with weight
  σ(±ρ_l), ε ~ N(0, Σ): 3L + 5 parameters, 2^L components;
- **rank-one, shared z** (the graph reading of §4): prior N(z; m, P) times L rank-one switching
  constraints N(a_lᵀz; b_{l,k_l}, q_l). Because the bits are independent the density is a product
  of experts, p(z) = N(z; m, P) Π_l Σ_j π_lj N(a_lᵀz; b_lj, q_l) / Z̄, with the coupling normaliser
  Z̄ = Σ_k π_k N(b_k; A m, A P Aᵀ + Q) over the 2^L configurations: 5L + 5 effective parameters
  (a_l, b_l, q_l are defined up to a common scale);
- **flat**: a K-component Gaussian mixture by EM, 6K − 1 parameters.

The composed models are fitted by exact maximum likelihood (Adam with ForwardDiff) from several
random starts on parallel threads; the best training likelihood wins. Run with
`julia -t auto --project=. examples/composition_e3.jl`.
"""
module CompositionE3

using LinearAlgebra, Random, Printf
import ForwardDiff

isdefined(Main, :SwitchingE4a) || Base.include(Main, joinpath(@__DIR__, "switching_reverse_chain_e4a.jl"))
const E = Main.SwitchingE4a

# --- data ----------------------------------------------------------------------------------------

rot(a) = [cos(a) -sin(a); sin(a) cos(a)]
const SHARED = [0.12 0.04; 0.04 0.09]

"A product grid: modes c + Σ k_l δ_l with independent bits of probabilities `p`."
function grid(δs, p)
    L = length(δs)
    cfgs = [digits(i; base = 2, pad = L) for i in 0:2^L-1]
    μ = [sum(k[l] * δs[l] for l in 1:L; init = zeros(2)) for k in cfgs]
    π = [prod(k[l] == 1 ? p[l] : 1 - p[l] for l in 1:L) for k in cfgs]
    return (; π, μ, Σ = [SHARED for _ in cfgs], L)
end
const DATA = (
    grid3 = grid([[2.0, 0.0], [0.0, 2.0], [4.0, 1.0]], [0.5, 0.35, 0.6]),
    grid4 = grid([[2.0, 0.0], [0.0, 2.0], [4.0, 1.0], [1.0, 4.0]], [0.5, 0.35, 0.6, 0.7]),
    ring = (π = fill(1 / 8, 8), μ = [3 .* [cos(2π * j / 8), sin(2π * j / 8)] for j in 0:7],
            Σ = [Matrix(0.12I, 2, 2) for _ in 1:8], L = 3),
)

# --- small numerics that ForwardDiff can see through ---------------------------------------------

function lse(v)
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end
logsig(x) = -log1p(exp(-x))
"Lower Cholesky factor of a 2×2 covariance from three unconstrained numbers."
chol2(v) = (exp(v[1]), v[2], exp(v[3]))
function logN2(z1, z2, μ1, μ2, (l11, l21, l22))
    u1 = (z1 - μ1) / l11
    u2 = (z2 - μ2 - l21 * u1) / l22
    return -log(2π) - log(l11) - log(l22) - (u1^2 + u2^2) / 2
end
logN1(x, μ, q) = -(log(2π * q) + (x - μ)^2 / q) / 2

# --- additive composition ------------------------------------------------------------------------

nadd(L) = 3L + 5
"Rows of X: the additive model's log-density, θ = [c; δ_1 … δ_L; ρ; chol(Σ)]."
function additive_logpdf(θ, L, X)
    c = θ[1:2]; δ = [θ[2l+1:2l+2] for l in 1:L]; ρ = θ[2L+3:3L+2]; ch = chol2(θ[3L+3:3L+5])
    cfgs = [digits(i; base = 2, pad = L) for i in 0:2^L-1]
    lw = [sum(k[l] == 1 ? logsig(ρ[l]) : logsig(-ρ[l]) for l in 1:L) for k in cfgs]
    μ = [c + sum(k[l] * δ[l] for l in 1:L) for k in cfgs]
    return [lse([lw[j] + logN2(X[i, 1], X[i, 2], μ[j][1], μ[j][2], ch) for j in eachindex(cfgs)]) for i in axes(X, 1)]
end
function additive_init(L, X, rng)
    pick() = X[rand(rng, axes(X, 1)), :]
    s = sqrt(minimum(eigvals(E.cov(X)))) / 4
    return [pick(); reduce(vcat, [pick() - pick() for _ in 1:L]); zeros(L); log(s); 0.0; log(s)]
end

# --- rank-one switching constraints on a shared z (product of experts) ---------------------------

nrank1(L) = 5L + 5                 # effective; the vector below carries 6L + 5 (one scale per factor is free)
function rank1_unpack(θ, L)
    m = θ[1:2]; Pc = chol2(θ[3:5])
    blk(l) = θ[5+6(l-1)+1:5+6l]
    a = [blk(l)[1:2] for l in 1:L]; b = [blk(l)[3:4] for l in 1:L]; q = [exp(blk(l)[5]) for l in 1:L]; ρ = [blk(l)[6] for l in 1:L]
    return m, Pc, a, b, q, ρ
end
function rank1_lognorm(θ, L)
    m, (l11, l21, l22), a, b, q, ρ = rank1_unpack(θ, L)
    Pchol = [l11 0; l21 l22]; P = Pchol * Pchol'
    A = reduce(vcat, [a[l]' for l in 1:L])
    S = A * P * A' + Diagonal(q)
    Sc = cholesky(Symmetric(S))
    ldet = 2sum(log, diag(Sc.U))
    terms = map(0:2^L-1) do i
        k = digits(i; base = 2, pad = L)
        bk = [b[l][k[l]+1] for l in 1:L]
        r = bk - A * m
        sum(k[l] == 1 ? logsig(ρ[l]) : logsig(-ρ[l]) for l in 1:L) - (L * log(2π) + ldet + dot(r, Sc \ r)) / 2
    end
    return lse(terms)
end
function rank1_logpdf(θ, L, X)
    m, Pc, a, b, q, ρ = rank1_unpack(θ, L)
    logZ = rank1_lognorm(θ, L)
    return map(axes(X, 1)) do i
        z = X[i, :]
        lp = logN2(z[1], z[2], m[1], m[2], Pc)
        for l in 1:L
            x = dot(a[l], z)
            lp += lse([logsig(-ρ[l]) + logN1(x, b[l][1], q[l]), logsig(ρ[l]) + logN1(x, b[l][2], q[l])])
        end
        lp - logZ
    end
end
function rank1_init(L, X, rng)
    C = E.cov(X); mX = vec(sum(X; dims = 1)) / size(X, 1)
    θ = [mX; 0.5log(C[1, 1]); C[2, 1] / sqrt(C[1, 1]); 0.5log(max(C[2, 2] - C[2, 1]^2 / C[1, 1], 1e-3))]
    for _ in 1:L
        a = normalize(randn(rng, 2)); x = X * a
        s = sqrt(sum(abs2, x .- sum(x) / length(x)) / length(x))
        append!(θ, [a; sum(x) / length(x) - s; sum(x) / length(x) + s; log(s^2 / 4); 0.0])
    end
    return θ
end

# --- fitting -------------------------------------------------------------------------------------

"Exact maximum likelihood from `starts` random starts on parallel threads (Adam, ForwardDiff)."
function fit(logpdf, init, L, X; starts = 16, iters = 1500, lr = 0.03, seed = 1)
    f(θ) = -sum(logpdf(θ, L, X)) / size(X, 1)
    tasks = map(1:starts) do s
        Threads.@spawn begin
            θ = init(L, X, Xoshiro(seed + 1000s)); m, v = zero(θ), zero(θ)
            for t in 1:iters
                g = ForwardDiff.gradient(f, θ)
                all(isfinite, g) || break
                m = 0.9m + 0.1g; v = 0.999v + 0.001g .^ 2
                θ = θ - lr * (m / (1 - 0.9^t)) ./ (sqrt.(v / (1 - 0.999^t)) .+ 1e-8)
            end
            (θ, -f(θ))
        end
    end
    res = fetch.(tasks)
    lls = [r[2] for r in res]
    best = argmax(x -> isfinite(x) ? x : -Inf, lls)
    return res[findfirst(==(best), lls)][1], best, lls
end

flat_logpdf((w, μ, V), X) = [E.gmm_logpdf(w, μ, V, X[i, :]) for i in axes(X, 1)]
mean(v) = sum(v) / length(v)

function run(name; ntrains = (150, 2000), ntest = 2000, Ks = nothing, starts = 16, iters = 1500)
    d = DATA[name]; L = d.L
    Ks = something(Ks, sort(unique([2^L ÷ 4, 2^L ÷ 2, 2^L])))
    Xte = E.sample_gmm(d.π, d.μ, d.Σ, ntest, Xoshiro(2))
    truth = mean(flat_logpdf((d.π, d.μ, d.Σ), Xte))
    @printf("%s (%d modes): held-out log p of the truth %.3f\n", name, length(d.π), truth)
    rows = []
    for n in ntrains
        X = E.sample_gmm(d.π, d.μ, d.Σ, n, Xoshiro(10 + n))
        θa, _, lla = fit(additive_logpdf, additive_init, L, X; starts, iters)
        θr, _, llr = fit(rank1_logpdf, rank1_init, L, X; starts, iters)
        push!(rows, (; n, model = "additive, L = $L", params = nadd(L), test = mean(additive_logpdf(θa, L, Xte)),
                     at_best = count(>(maximum(lla) - 1e-3), lla)))
        push!(rows, (; n, model = "rank-one, shared z, L = $L", params = nrank1(L), test = mean(rank1_logpdf(θr, L, Xte)),
                     at_best = count(>(maximum(llr) - 1e-3), llr)))
        for K in Ks
            g = E.em_gmm(X, K; iters = 200, restarts = 5, floor = 1e-3, seed = 3)
            push!(rows, (; n, model = "flat, K = $K", params = 6K - 1, test = mean(flat_logpdf(g, Xte)), at_best = 0))
        end
        for r in rows[end-length(Ks)-1:end]
            @printf("  n = %4d  %-28s %3d params   held-out log p %.3f   (gap %.3f)%s\n", r.n, r.model, r.params,
                    r.test, truth - r.test, r.at_best > 0 ? "   $(r.at_best)/$starts starts at the best" : "")
        end
    end
    return (; name, truth, rows)
end

"The rank-one model's density integrates to one: a check of the coupling normaliser Z̄."
function rank1_mass(θ, L; h = 0.05, R = 12.0)
    g = -R:h:R
    X = [x for x in g for _ in g]; Y = [y for _ in g for y in g]
    return sum(exp, rank1_logpdf(θ, L, hcat(X, Y))) * h^2
end

main() = [run(name) for name in keys(DATA)]

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    CompositionE3.main()
end
