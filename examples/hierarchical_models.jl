"""
Hierarchical models as factor graphs: three linear-Gaussian examples, each checked against a
dense Gaussian computed independently (plain linear algebra, no Lenticulum). Run with
`julia --project=. examples/hierarchical_models.jl`; the test suite includes it. See
`hierarchical_models.md` and the vault note `Hierarchical Models`.

1. The eight schools (Rubin 1981): random intercepts, a tree, exact. Shrinkage as the
   population variance τ varies, the marginal likelihood in τ, a mixture over τ, a new school.
2. Random slopes: each group's regression weights are *variables*, so the hierarchy needs no
   access to any factor's parameters.
3. Crossed effects (rows × columns): the graph has loops; loopy Gaussian BP gets the means
   exactly and the variances not.
"""
module HierarchicalModels

using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, Printf
using Mycelium: connect!

# --- 1. The eight schools ------------------------------------------------------------------

# Rubin (1981): estimated coaching effects and their standard errors, rounded to integers as
# is customary.
const Y8 = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const S8 = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
const V0 = 100.0^2                                # prior variance of the population mean μ

"""
    schools_graph(τ; y, s, newschool) -> (g, ps, st)

μ ~ N(0, V0); θⱼ = μ + τ ηⱼ; yⱼ = θⱼ + sⱼ εⱼ. The population factor μ → θⱼ is a
`GaussianFactor` with noise τ²; each school's measurement is a noise factor θⱼ → zⱼ plus a
`DataFactor`. With `newschool`, a ninth θ hangs off μ with no data: its marginal is the
posterior predictive effect of an unseen school.
"""
function schools_graph(τ; y = Y8, s = S8, newschool = false)
    b = GraphBuilder()
    variable!(b, :mu, 1)
    factor!(b, :prior, GaussianPrior(:mu, [0.0], fill(V0, 1, 1)))
    connect!(b, :prior, :mu, :mu; direction = Bidirectional())
    ps, st = Pair{Symbol,Any}[:prior => NamedTuple()], Pair{Symbol,Any}[:prior => NamedTuple()]
    J = length(y) + newschool
    for j in 1:J
        θ, pop = Symbol(:theta, j), Symbol(:pop, j)
        variable!(b, θ, 1)
        factor!(b, pop, GaussianFactor(1 => 1; noise = fill(τ^2, 1, 1), channels = (:x, :y)))
        connect!(b, pop, :x, :mu; direction = Bidirectional())
        connect!(b, pop, :y, θ; direction = Bidirectional())
        push!(ps, pop => (A = fill(1.0, 1, 1), b = [0.0])); push!(st, pop => NamedTuple())
        j ≤ length(y) || continue
        z, m, d = Symbol(:z, j), Symbol(:meas, j), Symbol(:data, j)
        variable!(b, z, 1)
        factor!(b, m, GaussianFactor(1 => 1; noise = fill(s[j]^2, 1, 1), channels = (:x, :y)))
        factor!(b, d, DataFactor(z, [y[j]]))
        connect!(b, m, :x, θ; direction = Bidirectional())
        connect!(b, m, :y, z; direction = Bidirectional())
        connect!(b, d, z, z; direction = Emitting())
        push!(ps, m => (A = fill(1.0, 1, 1), b = [0.0]), d => NamedTuple())
        push!(st, m => NamedTuple(), d => NamedTuple())
    end
    return validate(build(b)), NamedTuple(ps), NamedTuple(st)
end

"Posterior means and variances of (μ, θ₁, …), and −log p(y | τ), by message passing."
function schools_graph_posterior(τ; newschool = false)
    g, ps, st = schools_graph(τ; newschool)
    istree(g) || error("the schools graph should be a tree")
    marg, _, st, store = infer!(g, tree_schedule(g), ps, st)
    J = length(Y8) + newschool
    names = [:mu; [Symbol(:theta, j) for j in 1:J]]
    mean = [belief_mean(marg[n])[1] for n in names]
    var = [belief_cov(marg[n])[1] for n in names]
    F = first(LenticulumCore.scalar_free_energy(store, g, ps, st))
    return (; mean, var, F)
end

"The same quantities from the joint Gaussian of (μ, θ, y), conditioned on y by linear algebra."
function schools_dense(τ; newschool = false)
    J, Jθ = length(Y8), length(Y8) + newschool
    # z = (μ, θ₁..θ_Jθ) = L ξ with ξ = (μ, η) standard after scaling; y = θ[1:J] + s ε
    Σzz = zeros(1 + Jθ, 1 + Jθ)
    Σzz .= V0
    for j in 1:Jθ
        Σzz[1 + j, 1 + j] += τ^2
    end
    Σzy = Σzz[:, 2:(1 + J)]
    Σyy = Σzz[2:(1 + J), 2:(1 + J)] + Diagonal(S8 .^ 2)
    K = Σzy / Σyy
    mean = K * Y8
    var = diag(Σzz - K * Σzy')
    F = (J * log(2π) + logdet(Σyy) + dot(Y8, Σyy \ Y8)) / 2
    return (; mean, var, F)
end

# --- 2. Random slopes: the weights are variables -------------------------------------------

"""
    slopes_data(; J, n, seed) -> (X, y, βtrue)

J groups, n points each, yᵢ = β₁ + β₂ xᵢ + σε with group weights βⱼ ~ N(β̄, Σβ).
"""
function slopes_data(; J = 5, n = 6, seed = 7, β̄ = [1.0, 2.0], Σβ = Diagonal([0.5, 0.8] .^ 2), σ = 0.3)
    rng = Xoshiro(seed)
    X = [hcat(ones(n), sort(2rand(rng, n) .- 1)) for _ in 1:J]
    β = [β̄ .+ sqrt(Σβ) * randn(rng, 2) for _ in 1:J]
    y = [X[j] * β[j] .+ σ .* randn(rng, n) for j in 1:J]
    return (; X, y, β, Σβ = Matrix(Σβ), σ)
end

"""
The hierarchy β̄ → βⱼ → yⱼ. The design matrix Xⱼ is the *parameter* `A` of the observation
factor and βⱼ its *input channel*: the regression weights live on the graph, so the population
factor reaches them as it reaches any variable.
"""
function slopes_graph(d)
    b = GraphBuilder()
    variable!(b, :betabar, 2)
    factor!(b, :prior, GaussianPrior(:betabar, zeros(2), Matrix(V0 * I, 2, 2)))
    connect!(b, :prior, :betabar, :betabar; direction = Bidirectional())
    ps, st = Pair{Symbol,Any}[:prior => NamedTuple()], Pair{Symbol,Any}[:prior => NamedTuple()]
    for j in eachindex(d.X)
        n = size(d.X[j], 1)
        β, pop, yv, obs, dat = Symbol(:beta, j), Symbol(:pop, j), Symbol(:y, j), Symbol(:obs, j), Symbol(:data, j)
        variable!(b, β, 2); variable!(b, yv, n)
        factor!(b, pop, GaussianFactor(2 => 2; noise = d.Σβ, channels = (:x, :y)))
        factor!(b, obs, GaussianFactor(2 => n; noise = Matrix(d.σ^2 * I, n, n), channels = (:x, :y)))
        factor!(b, dat, DataFactor(yv, d.y[j]))
        connect!(b, pop, :x, :betabar; direction = Bidirectional())
        connect!(b, pop, :y, β; direction = Bidirectional())
        connect!(b, obs, :x, β; direction = Bidirectional())
        connect!(b, obs, :y, yv; direction = Bidirectional())
        connect!(b, dat, yv, yv; direction = Emitting())
        push!(ps, pop => (A = Matrix(1.0I, 2, 2), b = zeros(2)), obs => (A = d.X[j], b = zeros(n)),
              dat => NamedTuple())
        push!(st, pop => NamedTuple(), obs => NamedTuple(), dat => NamedTuple())
    end
    return validate(build(b)), NamedTuple(ps), NamedTuple(st)
end

function slopes_graph_posterior(d)
    g, ps, st = slopes_graph(d)
    istree(g) || error("the random-slopes graph should be a tree")
    marg, _, st, store = infer!(g, tree_schedule(g), ps, st)
    names = [:betabar; [Symbol(:beta, j) for j in eachindex(d.X)]]
    mean = reduce(vcat, [belief_mean(marg[n]) for n in names])
    cov = [belief_cov(marg[n]) for n in names]
    F = first(LenticulumCore.scalar_free_energy(store, g, ps, st))
    return (; mean, cov, F)
end

"Dense reference: (β̄, β₁..β_J) jointly Gaussian with y, conditioned by linear algebra."
function slopes_dense(d)
    J = length(d.X)
    p = 2(1 + J)
    Σzz = zeros(p, p)
    for a in 0:J, c in 0:J            # Cov(βa, βc) with β₀ = β̄: V0 I, plus Σβ on the diagonal blocks
        Σzz[2a+1:2a+2, 2c+1:2c+2] = V0 * I(2) + (a == c && a > 0 ? d.Σβ : zeros(2, 2))
    end
    H = zeros(sum(size.(d.X, 1)), p)  # y = H z + σ ε
    r = 0
    for j in 1:J
        n = size(d.X[j], 1)
        H[r+1:r+n, 2j+1:2j+2] = d.X[j]; r += n
    end
    yall = reduce(vcat, d.y)
    Σyy = H * Σzz * H' + d.σ^2 * I
    K = Σzz * H' / Σyy
    C = Σzz - K * H * Σzz
    mean = K * yall
    cov = [C[2a+1:2a+2, 2a+1:2a+2] for a in 0:J]
    F = (length(yall) * log(2π) + logdet(Σyy) + dot(yall, Σyy \ yall)) / 2
    return (; mean, cov, F)
end

# --- 3. Crossed effects: loops --------------------------------------------------------------

"""
    crossed_data(; R, C, seed) -> NamedTuple

y_rc = a_r + b_c + σε for every row r and column c (students × items, say), with
a_r ~ N(0, σa²) and b_c ~ N(0, σb²). Every observation joins one row effect and one column
effect, so the graph has (R-1)(C-1) independent loops.
"""
function crossed_data(; R = 4, C = 3, seed = 3, σa = 1.0, σb = 0.7, σ = 0.5)
    rng = Xoshiro(seed)
    a, bb = σa .* randn(rng, R), σb .* randn(rng, C)
    y = [a[r] + bb[c] + σ * randn(rng) for r in 1:R, c in 1:C]
    return (; R, C, y, σa, σb, σ)
end

function crossed_graph(d)
    b = GraphBuilder()
    ps, st = Pair{Symbol,Any}[], Pair{Symbol,Any}[]
    for (pre, n, sd) in ((:a, d.R, d.σa), (:b, d.C, d.σb)), k in 1:n
        v, f = Symbol(pre, k), Symbol(:prior_, pre, k)
        variable!(b, v, 1)
        factor!(b, f, GaussianPrior(v, [0.0], fill(sd^2, 1, 1)))
        connect!(b, f, v, v; direction = Bidirectional())
        push!(ps, f => NamedTuple()); push!(st, f => NamedTuple())
    end
    for r in 1:d.R, c in 1:d.C        # 0 = a_r + b_c - y_rc + ε
        f = Symbol(:obs, r, :_, c)
        factor!(b, f, LinearConstraintFactor((a = 1, b = 1), 1; noise = d.σ^2))
        connect!(b, f, :a, Symbol(:a, r); direction = Bidirectional())
        connect!(b, f, :b, Symbol(:b, c); direction = Bidirectional())
        push!(ps, f => (A = (a = fill(1.0, 1, 1), b = fill(1.0, 1, 1)), c = [d.y[r, c]]))
        push!(st, f => NamedTuple())
    end
    return validate(build(b)), NamedTuple(ps), NamedTuple(st)
end

function crossed_graph_posterior(d; maxsweeps = 500, tol = 1e-12)
    g, ps, st = crossed_graph(d)
    marg, report, _, _ = infer!(g, flooding_schedule(g), ps, st; maxsweeps, tol)
    names = [[Symbol(:a, r) for r in 1:d.R]; [Symbol(:b, c) for c in 1:d.C]]
    mean = [belief_mean(marg[n])[1] for n in names]
    var = [belief_cov(marg[n])[1] for n in names]
    return (; mean, var, report, loops = nedges(g) - nvariables(g) - nfactors(g) + 1)
end

function crossed_dense(d)
    R, C = d.R, d.C
    # precision of (a, b) given y: prior + Σ_rc (e_r + e_{R+c})(…)ᵀ/σ²
    Λ = Matrix(Diagonal([fill(1 / d.σa^2, R); fill(1 / d.σb^2, C)]))
    h = zeros(R + C)
    for r in 1:R, c in 1:C
        u = zeros(R + C); u[r] = 1; u[R + c] = 1
        Λ .+= u * u' / d.σ^2
        h .+= u * d.y[r, c] / d.σ^2
    end
    Σ = inv(Symmetric(Λ))
    return (; mean = Σ * h, var = diag(Σ))
end

# --- run everything -------------------------------------------------------------------------

relerr(a, b) = maximum(abs.(a .- b)) / max(1, maximum(abs.(b)))

function main(; io = stdout)
    out = Dict{Symbol,Any}()

    # 1a. one τ: graph vs dense
    for τ in (0.5, 5.0, 10.0, 25.0)
        G, D = schools_graph_posterior(τ), schools_dense(τ)
        out[Symbol(:schools_err_, τ)] = max(relerr(G.mean, D.mean), relerr(G.var, D.var), abs(G.F - D.F))
    end
    println(io, "1. Eight schools — max deviation graph vs dense over τ ∈ {0.5, 5, 10, 25}: ",
            @sprintf("%.1e", maximum(v for (k, v) in out if startswith(String(k), "schools_err"))))

    # 1b. shrinkage table: E[θⱼ | y, τ]
    τs = (0.5, 5.0, 10.0, 25.0, 100.0)
    println(io, "\n   E[θⱼ | y, τ] (shrinkage towards the pooled mean as τ falls)")
    println(io, "   school    y  ", join([@sprintf("τ=%-6g", τ) for τ in τs], " "))
    tab = [schools_graph_posterior(τ).mean for τ in τs]
    for j in 1:8
        println(io, @sprintf("   %-6s %5.0f  ", "  $(Char('A' + j - 1))", Y8[j]),
                join([@sprintf("%-8.2f", t[1 + j]) for t in tab], " "))
    end
    out[:shrinkage] = tab

    # 1c. marginal likelihood in τ, and a mixture over τ (uniform prior on the grid)
    grid = collect(range(0.25, 40.0; length = 160))
    F = [schools_graph_posterior(τ).F for τ in grid]
    Fd = [schools_dense(τ).F for τ in grid]
    out[:F_grid_err] = maximum(abs.(F .- Fd))
    w = exp.(-(F .- minimum(F))); w ./= sum(w)
    posts = [schools_graph_posterior(τ) for τ in grid]
    θmix = sum(w[k] .* posts[k].mean for k in eachindex(grid))
    θvar = sum(w[k] .* (posts[k].var .+ posts[k].mean .^ 2) for k in eachindex(grid)) .- θmix .^ 2
    out[:tau_argmax] = grid[argmin(F)]
    out[:tau_mean] = sum(w .* grid)
    out[:mix_mean], out[:mix_sd] = θmix, sqrt.(θvar)
    println(io, @sprintf("\n   −log p(y | τ): graph vs dense, max |Δ| on the grid = %.1e", out[:F_grid_err]))
    println(io, @sprintf("   maximised at the grid's lower end τ = %.2f (type-II ML pools completely);", out[:tau_argmax]))
    println(io, @sprintf("   with a uniform prior on τ ∈ [0.25, 40]: E[τ | y] = %.2f", out[:tau_mean]))
    println(io, "   E[θⱼ | y] and sd, mixed over τ (a mixture belief, outside the graph):")
    for j in 1:8
        println(io, @sprintf("     %s  %6.2f ± %5.2f", Char('A' + j - 1), θmix[1 + j], sqrt(θvar[1 + j])))
    end
    println(io, @sprintf("     μ  %6.2f ± %5.2f", θmix[1], sqrt(θvar[1])))

    # 1d. a new school: the predictive effect
    G9, D9 = schools_graph_posterior(10.0; newschool = true), schools_dense(10.0; newschool = true)
    out[:new_err] = max(relerr(G9.mean, D9.mean), relerr(G9.var, D9.var))
    out[:new_school] = (G9.mean[end], sqrt(G9.var[end]), G9.mean[1], sqrt(G9.var[1]))
    println(io, @sprintf("\n   new school at τ = 10: θ₉ ~ N(%.2f, %.2f²) vs μ ~ N(%.2f, %.2f²); graph vs dense %.1e",
                         out[:new_school]..., out[:new_err]))

    # 2. random slopes
    d = slopes_data()
    G, D = slopes_graph_posterior(d), slopes_dense(d)
    out[:slopes_err] = max(relerr(G.mean, D.mean), maximum(relerr.(G.cov, D.cov)), abs(G.F - D.F))
    println(io, @sprintf("\n2. Random slopes (%d groups × %d points): graph vs dense %.1e", length(d.X), size(d.X[1], 1), out[:slopes_err]))
    println(io, "   group   true β            posterior mean of β")
    for j in eachindex(d.X)
        m = G.mean[2j+1:2j+2]
        println(io, @sprintf("     %d     (%5.2f, %5.2f)    (%5.2f, %5.2f)", j, d.β[j]..., m...))
    end
    println(io, @sprintf("   β̄: (%5.2f, %5.2f), true (1.00, 2.00)", G.mean[1:2]...))
    out[:betabar] = G.mean[1:2]

    # 3. crossed effects
    c = crossed_data()
    Gc, Dc = crossed_graph_posterior(c), crossed_dense(c)
    out[:crossed_mean_err] = relerr(Gc.mean, Dc.mean)
    out[:crossed_var_ratio] = extrema(Gc.var ./ Dc.var)
    out[:crossed_converged] = Gc.report.converged
    out[:crossed_loops] = Gc.loops
    println(io, @sprintf("\n3. Crossed effects (%d × %d, %d independent loops), flooding schedule, converged = %s",
                         c.R, c.C, Gc.loops, Gc.report.converged))
    println(io, @sprintf("   means: max deviation from the exact posterior %.1e", out[:crossed_mean_err]))
    println(io, @sprintf("   variances: loopy / exact ∈ [%.3f, %.3f]", out[:crossed_var_ratio]...))
    return out
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HierarchicalModels.main()
end
