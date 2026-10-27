"""
LP-010: fixed-mean Gaussian structure search, using Lenticulum pairwise factors and
Mycelium's Bethe normaliser. Run with `julia --project=. examples/gaussian_structure_search.jl`.
No optional solver dependencies. See `gaussian_structure_search.md` and the vault note
`Finding a Factor Graph` for the score, assumptions and limitations.
"""
module GaussianStructureSearch

using Lenticulum
using LenticulumCore
using Mycelium
using LinearAlgebra
using Random

const Edge = Tuple{Int,Int}
all_edges(d) = Edge[(i, j) for i in 1:d for j in (i + 1):d]

function chain_sample(; d = 8, n = 10000, seed = 1010, coupling = 0.12)
    edges = Edge[(i, i + 1) for i in 1:(d - 1)]
    K = Matrix{Float64}(I, d, d)
    for (i, j) in edges
        K[i, j] = K[j, i] = -coupling
    end
    # Known zero mean; S uses divisor n, with no estimated centring parameter.
    X = cholesky(Symmetric(K)).U \ randn(Xoshiro(seed), d, n)
    return (; S = X * X' / n, K, edges, n)
end

exact_nll(K, S, n) = n * (tr(K * S) - logdet(Symmetric(K)) + size(K, 1) * log(2π)) / 2

"""Sparse precision MLE by Newton's method on the allowed entries (small dense oracle)."""
function fit_precision(S, edges; tol = 1e-9, maxiter = 80)
    d = size(S, 1)
    entries = vcat(Edge[(i, i) for i in 1:d], sort(edges))
    basis = map(entries) do (i, j)
        B = zeros(d, d)
        B[i, j] = B[j, i] = 1.0
        B
    end
    K = Matrix(Diagonal(1 ./ diag(S)))
    objective(M) = tr(S * M) - logdet(Symmetric(M))
    for _ in 1:maxiter
        C = inv(Symmetric(K))
        grad = [tr((S - C) * B) for B in basis]
        maximum(abs, grad) < tol && return K
        H = [tr(C * B * C * D) for B in basis, D in basis]
        step = -(Symmetric(H) \ grad)
        Δ = sum(a * B for (a, B) in zip(step, basis))
        α = 1.0
        old = objective(K)
        while α > 1e-12
            next = K + α * Δ
            if isposdef(Symmetric(next)) && objective(next) <= old +
                    1e-4 * α * dot(grad, step) + 10 * eps(abs(old))
                K = next
                break
            end
            α /= 2
        end
        α > 1e-12 || error("precision MLE line search failed")
    end
    error("precision MLE did not converge")
end

"""
Decompose a strictly diagonally dominant K into unary Gaussian priors and pairwise
GaussianFactor potentials. Factor constants cancel between data energy and log Z.
No general precision-matrix factor or new package API is needed.
"""
function factor_graph(K, edges)
    d = size(K, 1)
    remainder = diag(K) - [sum(abs(K[i, j]) for j in 1:d if j != i) for i in 1:d]
    minimum(remainder) > 0 || error("example needs strict diagonal dominance")
    b = GraphBuilder()
    names = Symbol[]
    params = Any[]
    constant = 0.0
    for i in 1:d
        v, f = Symbol(:x, i), Symbol(:prior, i)
        variable!(b, v, 1)
        factor!(b, f, GaussianPrior(:x, 0.0, 1 / remainder[i]))
        connect!(b, f, :x, v; direction = Bidirectional())
        push!(names, f); push!(params, (;))
        constant += log(2π / remainder[i]) / 2
    end
    for (i, j) in sort(edges)
        w = abs(K[i, j])
        w > 1e-14 || continue
        f = Symbol(:pair, i, :_, j)
        factor!(b, f, GaussianFactor(1 => 1; noise = 1 / w))
        connect!(b, f, :x, Symbol(:x, i); direction = Bidirectional())
        connect!(b, f, :y, Symbol(:x, j); direction = Bidirectional())
        push!(names, f)
        push!(params, (A = fill(-sign(K[i, j]), 1, 1), b = [0.0]))
        constant += log(2π / w) / 2
    end
    g = validate(build(b))
    ps = NamedTuple{Tuple(names)}(Tuple(params))
    st = NamedTuple{Tuple(names)}(Tuple((;) for _ in names))
    return g, ps, st, constant
end

function bethe_score(K, edges, S, n; penalty = true)
    g, ps, st, constant = factor_graph(K, edges)
    store = MessageStore(g)
    # Seed the unary messages and the cavities before flooding; pairwise factors
    # then have informative inputs on the very first synchronous sweep.
    for ei in 1:nedges(g)
        if factor_degree(g, g.edges[ei].factor) == 1
            _, _, st = step!(store, g, MessageTask(ei, :to_variable), ps, st)
        end
    end
    for ei in 1:nedges(g)
        _, _, st = step!(store, g, MessageTask(ei, :to_factor), ps, st)
    end
    report, st = propagate!(store, g, flooding_schedule(g), ps, st;
                            tol = 1e-10, maxsweeps = 300)
    report.converged || error("BP failed: $report")
    F, _ = scalar_free_energy(store, g, ps, st)
    # F = -log ∫ product(potentials). Thus normalised data NLL = U_data - F.
    nll = n * (tr(K * S) / 2 + constant - F)
    bic = penalty ? (size(K, 1) + length(edges)) * log(n) / 2 : 0.0
    return (; score = nll + bic, nll, F, exact = exact_nll(K, S, n), sweeps = report.sweeps)
end

"""Refit every proposed single-edge extension, and accept the best strict improvement."""
function greedy_search(S, n; penalty = true)
    edges = Edge[]
    K = fit_precision(S, edges)
    current = bethe_score(K, edges, S, n; penalty)
    trajectory = [(; added = (0, 0), edges = 0, current...)]
    while true
        best, bestK, bestedge = current, K, (0, 0)
        for edge in setdiff(all_edges(size(S, 1)), edges)
            proposed = vcat(edges, [edge])
            fitted = fit_precision(S, proposed)
            candidate = bethe_score(fitted, proposed, S, n; penalty)
            if candidate.score < best.score - 1e-7
                best, bestK, bestedge = candidate, fitted, edge
            end
        end
        bestedge == (0, 0) && break
        push!(edges, bestedge)
        K, current = bestK, best
        push!(trajectory, (; added = bestedge, edges = length(edges), current...))
    end
    return (; edges = sort(edges), K, trajectory)
end

"""
Graphical lasso: minimise tr(SK)-logdet(K)+λ∑_{i≠j}|Kᵢⱼ| by ADMM (ρ=1).
The eigenvalue update solves the logdet proximal problem; the split variable Z
supplies exact zeros. Both residuals must pass; no diagonal shrinkage.
"""
function graphical_lasso(S, λ; tol = 1e-9, maxiter = 20000)
    d = size(S, 1)
    Z = Matrix{Float64}(I, d, d)
    U = zeros(d, d)
    for iteration in 1:maxiter
        E = eigen(Symmetric(Z - U - S))
        K = E.vectors * Diagonal((E.values + sqrt.(E.values .^ 2 .+ 4)) / 2) * E.vectors'
        oldZ = Z
        V = K + U
        Z = sign.(V) .* max.(abs.(V) .- λ, 0.0)
        for i in 1:d
            Z[i, i] = V[i, i]
        end
        U += K - Z
        residual = max(norm(K - Z), norm(Z - oldZ))
        residual < tol && return (; K, Z, residual, iteration)
    end
    error("graphical lasso did not converge")
end

function glasso_search(S, n; lambdas = exp.(range(log(0.3), log(0.003); length = 30)))
    best = nothing
    for λ in lambdas
        fit = graphical_lasso(S, λ)
        edges = filter(e -> abs(fit.Z[e...]) > 1e-6, all_edges(size(S, 1)))
        # Fair comparison: choose λ by exact refitted BIC, not by true-edge labels.
        K = fit_precision(S, edges)
        score = exact_nll(K, S, n) + (size(S, 1) + length(edges)) * log(n) / 2
        if best === nothing || score < best.score
            best = (; edges, K, score, λ, residual = fit.residual)
        end
    end
    return best
end

function edge_metrics(recovered, truth)
    tp = length(intersect(recovered, truth))
    return (; precision = isempty(recovered) ? 1.0 : tp / length(recovered), recall = tp / length(truth))
end

function main()
    data = chain_sample()
    println("seed=1010 d=8 n=$(data.n) coupling=0.12 known mean=0 true_edges=$(data.edges)")
    for penalty in (true, false)
        result = greedy_search(data.S, data.n; penalty)
        println("Bethe penalty=$penalty edges=$(result.edges) metrics=$(edge_metrics(result.edges, data.edges))")
        println("added,edge_count,negative_log_likelihood,penalised_score,unclamped_F,Bethe_minus_exact")
        for row in result.trajectory
            println(join((row.added, row.edges, row.nll, row.score, row.F, row.nll - row.exact), ','))
        end
    end
    result = glasso_search(data.S, data.n)
    println("glasso λ=$(result.λ) edges=$(result.edges) metrics=$(edge_metrics(result.edges, data.edges)) BIC/2=$(result.score) residual=$(result.residual)")
    # Same seed and sample size, stronger true couplings: expose Bethe loop bias.
    stress = chain_sample(; coupling = 0.32)
    result = greedy_search(stress.S, stress.n)
    lastrow = last(result.trajectory)
    println("stress coupling=0.32 Bethe+BIC edges=$(result.edges) metrics=$(edge_metrics(result.edges, stress.edges)) NLL=$(lastrow.nll) score=$(lastrow.score) Bethe_minus_exact=$(lastrow.nll-lastrow.exact)")
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    GaussianStructureSearch.main()
end
