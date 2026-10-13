# Products of relations: separators, and sampling a product with corrector steps and reweighting.
#
#     julia --project=lib/VariationalDiffusion.jl/test lib/VariationalDiffusion.jl/examples/product_sampling.jl
#
# Two closed-form problems where the exact product is known:
#
#   chain    p(x, y, z) = p_A(x, y) p_B(y, z) / p_S(y), two blobs at (-1, -1, -2) and (1, 1, 2) with
#            weights 0.8 and 0.2; the separator p_S is the shared marginal of y
#   circle   a unit circle (upper half 3× heavier) ∩ the line y = x, on the same variables; the
#            exact product of the two data mixtures puts 0.751 on the upper intersection
#
# Prints (1) the Laplace mixture of the chain with and without the separator, and (2) for each
# sampler setting the mass on the main mode, the share of samples off every mode and the spread,
# averaged over six seeds with 2000 chains each. Results are recorded in `product.md` §5–§6.

using VariationalDiffusion, LenticulumCore, Random, LinearAlgebra, LuxCore, Statistics, Printf

const S = VPSDE()
const s = 0.05

function relation(μ; w = nothing, nodes = noisefree_nodes(size(μ, 1), 0.01))
    p = NoisePredictor(GaussianMixtureEps(S, μ; s, weights = w), S)
    return (ImplicitDiffusion(p, nodes; λ = density_lambda(S, nodes)), LuxCore.setup(Xoshiro(0), p)...)
end

# --- the chain --------------------------------------------------------------------------------
const W = [0.8, 0.2]
n2, n1 = field_nodes(Xoshiro(1), 2; samples = 8), field_nodes(Xoshiro(2), 1; samples = 8)
lA = relation([-1.0 1.0; -1.0 1.0]; w = W, nodes = n2)
lB = relation([-1.0 1.0; -2.0 2.0]; w = W, nodes = n2)
lS = relation([-1.0 1.0]; w = W, nodes = n1)

println("(1) the chain's answers, Laplace mixture (implicit_mixture), exact weights 0.8 / 0.2")
for (name, rel, ps, st) in (("with separator", ProductRelation(lA[1] => [1, 2], lB[1] => [2, 3], lS[1] => [2]; weights = [1, 1, -1]),
                             (lA[2], lB[2], lS[2]), (lA[3], lB[3], lS[3])),
                            ("without", ProductRelation(lA[1] => [1, 2], lB[1] => [2, 3]), (lA[2], lB[2]), (lA[3], lB[3])))
    mix, _ = implicit_mixture(rel, zeros(3), zeros(3), ps, st; nstarts = 30, spread = 1.5)
    w = exp.(mix.logw) ./ sum(exp.(mix.logw))
    @printf("  %-15s", name)
    for (c, wi) in zip(mix.components, w)
        @printf("  %s: %.3f (sd y %.3f)", string(round.(belief_mean(c); digits = 2)), wi, sqrt(belief_cov(c)[2, 2]))
    end
    println()
end

# the samplers use one noise level for the nodes: they read the predictors only
cA, cB, cS = relation([-1.0 1.0; -1.0 1.0]; w = W), relation([-1.0 1.0; -2.0 2.0]; w = W), relation([-1.0 1.0]; w = W)
chain = (ProductRelation(cA[1] => [1, 2], cB[1] => [2, 3], cS[1] => [2]; weights = [1, 1, -1]),
         (cA[2], cB[2], cS[2]), (cA[3], cB[3], cS[3]))
function chainstats(Z)
    b1 = [norm(z .- [-1, -1, -2]) < 0.3 for z in eachcol(Z)]
    b2 = [norm(z .- [1, 1, 2]) < 0.3 for z in eachcol(Z)]
    return (main = mean(b1), off = 1 - mean(b1 .| b2), spread = mean(std(Z[:, b1]; dims = 2)))
end

# --- circle ∩ line ----------------------------------------------------------------------------
θ = range(0, 2π; length = 129)[1:128]
μc, wc = vcat(cos.(θ)', sin.(θ)'), [sin(t) > 0 ? 3.0 : 1.0 for t in θ]
u = range(-1.6, 1.6; length = 65)
μl = vcat(u', u')
lc, ll = relation(μc; w = wc), relation(μl)
circ = (ProductRelation(lc[1], ll[1]), (lc[2], ll[2]), (lc[3], ll[3]))

# exact draws: the product of two isotropic mixtures of width s is a mixture of width s/√2
function exact(n, rng)
    P = [wc[a] * exp(-sum(abs2, μc[:, a] .- μl[:, b]) / (4s^2)) for a in axes(μc, 2), b in axes(μl, 2)]
    c = cumsum(vec(P) ./ sum(P))
    X = zeros(2, n)
    for k in 1:n
        a, b = Tuple(CartesianIndices(P)[min(searchsortedfirst(c, rand(rng)), length(c))])
        X[:, k] = (μc[:, a] .+ μl[:, b]) ./ 2 .+ s / sqrt(2) .* randn(rng, 2)
    end
    return X
end
function circstats(Z)
    up = Z[1, :] .+ Z[2, :] .> 0
    r = vec(sqrt.(sum(abs2, Z; dims = 1)))
    off = mean((abs.(r .- 1) .> 0.15) .| (abs.(Z[1, :] .- Z[2, :]) .> 0.2))
    zu = Z[:, up]
    return (main = mean(up), off = off,
            spread = (std((zu[1, :] .+ zu[2, :]) ./ sqrt(2)) + std((zu[1, :] .- zu[2, :]) ./ sqrt(2))) / 2)
end
let e = circstats(exact(20000, Xoshiro(9)))
    @printf("\n(2) sampling, 100 steps, 2000 chains, mean ± sd over six seeds\n")
    @printf("  exact: chain main 0.800 off 0.000 spread 0.050 | circle main %.3f off %.3f spread %.3f\n", e...)
end

settings = [("plain", (;)),
            ("Langevin ×3, c = 0.1", (corrector = :langevin, corrector_steps = 3, corrector_step = 0.1)),
            ("Langevin ×10, c = 0.5", (corrector = :langevin, corrector_steps = 10, corrector_step = 0.5)),
            ("MALA ×10, c = 0.5", (corrector = :mala, corrector_steps = 10, corrector_step = 0.5)),
            ("SMC", (smc = true,)),
            ("SMC + MALA ×3, c = 0.5", (smc = true, corrector = :mala, corrector_steps = 3, corrector_step = 0.5))]
@printf("  %-24s %-15s %-8s %-8s %-15s %-8s %-8s %s\n", "sampler", "chain main", "off", "spread", "circle main", "off", "spread", "s/run")
pm(v) = @sprintf("%.3f ± %.3f", mean(v), std(v))
for (name, kw) in settings
    rows = map(1:6) do seed
        t = @elapsed begin
            Zc, _ = implicit_sample(chain[1], zeros(3), zeros(3), chain[2], chain[3]; n = 2000, rng = Xoshiro(seed), kw...)
            Zl, _ = implicit_sample(circ[1], zeros(2), zeros(2), circ[2], circ[3]; n = 2000, rng = Xoshiro(seed), kw...)
        end
        (chainstats(Zc), circstats(Zl), t / 2)
    end
    a, b = getindex.(rows, 1), getindex.(rows, 2)
    @printf("  %-24s %-15s %-8.3f %-8.3f %-15s %-8.3f %-8.3f %.1f\n", name,
            pm(getfield.(a, :main)), mean(getfield.(a, :off)), mean(getfield.(a, :spread)),
            pm(getfield.(b, :main)), mean(getfield.(b, :off)), mean(getfield.(b, :spread)), median(getindex.(rows, 3)))
end
