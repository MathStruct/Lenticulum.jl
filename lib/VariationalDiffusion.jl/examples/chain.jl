# Chains of three to five diffusion relations: inference across factors that share no variable.
#
#     julia --project=lib/VariationalDiffusion.jl/test lib/VariationalDiffusion.jl/examples/chain.jl
#
# A chain x₀ - x₁ - … - x_K of K relations p_k(x_{k-1}, x_k), with separators p(x_k) on the inner
# variables:
#
#     p(x₀, …, x_K) = Π_k p_k(x_{k-1}, x_k) / Π_{0<k<K} p(x_k).
#
# For K ≥ 3 the outer relations share no variable: what the query clamps at x₀ reaches x_K only
# through the inner ones. Two problems with known answers:
#
#   telephone  a bit passed along K links (a binary Markov chain embedded in the plane). Relation k
#              is a mixture of four blobs at (a, b) ∈ {±1}², weights π_{k-1}(a) T(a, b); the bit
#              flips with probability 0.2 from +1 and 0.3 from -1. The query holds x₀ = +1, and the
#              answers are all 2^K paths of the bit, weighted by their probabilities. The exact
#              P(x_K = +1 | x₀ = +1) is the (+1, +1) entry of T^K.
#   curves     relation k is the curve x_k = f(x_{k-1}), f(u) = 1.3 sin u, as 65 data points. With
#              x₀ held, the answer is x_K ≈ f^K(x₀), and its derivative by the adjoint is checked
#              against central differences of the solve and against the true derivative of f^K, the
#              chain rule Π f'(x_{k-1}). Run at two ranges of noise levels: the smoothing bias.
#
# Results are recorded in `product.md` §4.

using VariationalDiffusion, LenticulumCore, Random, LinearAlgebra, LuxCore, Statistics, Printf

const S = VPSDE()
const s = 0.05

relation(μ, nodes; w = nothing) =
    (p = NoisePredictor(GaussianMixtureEps(S, μ; s, weights = w), S);
     (ImplicitDiffusion(p, nodes; λ = density_lambda(S, nodes)), LuxCore.setup(Xoshiro(0), p)...))

# the chain's product: relation k on coordinates (k, k+1), separators on 2..K with weight -1
function chain(rels, seps)
    K = length(rels)
    parts = [[r[1] => [k, k + 1] for (k, r) in enumerate(rels)]; [r[1] => [k + 1] for (k, r) in enumerate(seps)]]
    rel = ProductRelation(parts...; weights = [ones(K); -ones(length(seps))])
    return rel, Tuple(r[2] for r in [rels; seps]), Tuple(r[3] for r in [rels; seps])
end

nodes(levels) = (field_nodes(Xoshiro(1), 2; samples = 8, levels), field_nodes(Xoshiro(2), 1; samples = 8, levels))
const LEVELS = range(0.002, 0.05; length = 8)           # the default of `field_nodes`
const n2, n1 = nodes(LEVELS)

# --- the telephone chain ------------------------------------------------------------------------
const T = [0.8 0.2; 0.3 0.7]                 # rows: from +1, -1; columns: to +1, -1
const π₀ = [0.5, 0.5]
const BLOBS = [1.0 1.0 -1.0 -1.0; 1.0 -1.0 1.0 -1.0]  # (a, b) = (+,+), (+,-), (-,+), (-,-)
state(v) = v > 0 ? 1 : 2

function telephone(K; separators = true, nstarts = 40 * 2^K)
    πs = [π₀' * T^k for k in 0:K]
    rels = [relation(BLOBS, n2; w = [πs[k][state(a)] * T[state(a), state(b)] for (a, b) in eachcol(BLOBS)]) for k in 1:K]
    seps = separators ? [relation([1.0 -1.0], n1; w = vec(πs[k + 1])) for k in 1:(K - 1)] : []
    rel, ps, st = chain(rels, seps)
    z₀, ρ = [1.0; zeros(K)], [Inf; zeros(K)]
    t = @elapsed mix, _ = implicit_mixture(rel, z₀, ρ, ps, st; nstarts, spread = 1.5)
    w = exp.(mix.logw) ./ sum(exp.(mix.logw))
    # exact weight of each path the answers found, and the bit at the end
    paths = [sign.(belief_mean(c)) for c in mix.components]
    exact = [prod(T[state(a), state(b)] for (a, b) in zip([1.0; p[1:(end - 1)]], p)) for p in paths]
    p₊ = sum(w[i] for i in eachindex(w) if paths[i][end] > 0; init = 0.0)
    return (found = length(paths), all = 2^K, p₊ = p₊, exact = (T^K)[1, 1], maxerr = maximum(abs.(w .- exact)),
            offset = maximum(norm(belief_mean(c) .- p) for (c, p) in zip(mix.components, paths)), seconds = t)
end

println("(1) telephone chain: x₀ = +1 held, all paths of the bit as answers (Laplace mixture)")
@printf("  %-2s %-12s %-14s %-13s %-15s %-14s %s\n", "K", "separators", "paths found", "P(x_K = +1)", "exact (T^K)₁₁",
        "max |w - exact|", "time")
for K in 3:5, sep in (true, false)
    r = telephone(K; separators = sep)
    @printf("  %-2d %-12s %3d of %-7d %-13.4f %-15.4f %-14.5f %.1f s\n", K, sep ? "yes" : "no", r.found, r.all, r.p₊, r.exact,
            r.maxerr, r.seconds)
    sep && @printf("     largest distance of an answer from its path's corner: %.4f\n", r.offset)
end

# --- a chain of curves -------------------------------------------------------------------------
f(u) = 1.3 * sin(u)
f′(u) = 1.3 * cos(u)
const V = collect(range(-1.6, 1.6; length = 65))

function curves(K; x₀ = 0.6, h = 1e-6, levels = LEVELS)
    m2, m1 = nodes(levels)
    rels = [relation(vcat(V', f.(V)'), m2) for _ in 1:K]
    seps = [relation(f.(V)', m1) for _ in 1:(K - 1)]          # the second variable's data of relation k
    rel, ps, st = chain(rels, seps)
    ρ = [Inf; zeros(K)]
    exact = accumulate((u, _) -> f(u), 1:K; init = x₀)
    solve(x; init = [x; exact]) = first(implicit_infer(rel, [x; zeros(K)], ρ, ps, st; z_init = init, tol = 1e-12))
    t = @elapsed sol = solve(x₀)
    b = implicit_pullback(rel, sol, [x₀; zeros(K)], ρ, [zeros(K); 1.0], ps, st)
    fd = (solve(x₀ + h; init = sol.z).z[end] - solve(x₀ - h; init = sol.z).z[end]) / 2h
    rule = prod(f′.([x₀; exact[1:(end - 1)]]))                # the true derivative of f^K
    return (ok = sol.converged && sol.stable, xK = sol.z[end], exact = exact[end], bias = maximum(abs.(sol.z[2:end] .- exact)),
            adjoint = b.z₀[1], fd = fd, rule = rule, seconds = t)
end

println("\n(2) chain of curves x_k = 1.3 sin x_{k-1}: x₀ = 0.6 held")
@printf("  %-2s %-15s %-17s %-9s %-9s %-11s %-11s %-11s %s\n", "K", "levels", "converged, stable", "x_K", "f^K(x₀)", "max bias",
        "dx_K/dx₀", "central FD", "(f^K)′(x₀)")
for (name, levels) in (("0.002–0.05", LEVELS), ("0.0005–0.005", range(0.0005, 0.005; length = 8))), K in 3:5
    r = curves(K; levels)
    @printf("  %-2d %-15s %-17s %-9.4f %-9.4f %-11.4f %-11.5f %-11.5f %.5f\n", K, name, r.ok, r.xK, r.exact, r.bias, r.adjoint,
            r.fd, r.rule)
end
