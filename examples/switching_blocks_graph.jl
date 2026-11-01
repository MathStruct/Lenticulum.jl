"""
E7 in the graph: `vault/Design/Minimal Stochastic Factors.md` §10, with the
`SwitchingConstraintFactor` of `src/switching.jl`. E6 (`examples/factorial_inference.jl`) ran
the additive wiring z = c + Σ_l k_l δ_l + ε as Mycelium's star graph, one `PriorFactor` per
switch, and found its factorised posterior biased through the pair moments. E7
(`examples/hierarchy_inference.jl`) removed most of the bias with exact blocks of overlapping
switches, inside a structured mean-field iteration. Here a block of B switches is one
`SwitchingConstraintFactor` with 2^B outcomes,

    0 = u′_b + Σ_{l∈b} k_l δ_l + ε,   P(k_b) = Π_{l∈b} p_l^{k_l} (1 − p_l)^{1 − k_l},

feeding the same sum node z + Σ_b u′_b = c, with the block's outcome as a variable of its own.
The graph is a tree and one pass is its fixed point; the sum node still collapses the other
blocks' mixture messages to their moments (ADF across blocks), but within a block the
posterior and its pair moments are exact. Blocks of one switch are E6's graph, and the
vectorised E-step below then agrees with E6's to rounding.

The question: does changing the factor alone (exact blocks, one pass of messages) remove the
bias, or does it need E7's iteration between blocks? Same truths, data, M-step and scores as E6;
E7's structured mean field (iterated, the same blocks) is run on the same data as the control.

Run with `julia -t auto --project=. examples/switching_blocks_graph.jl`.
"""
module SwitchingBlocksGraph

using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random, Printf
import LuxCore

isdefined(Main, :FactorialInference) || Base.include(Main, joinpath(@__DIR__, "factorial_inference.jl"))
isdefined(Main, :HierarchyInference) || Base.include(Main, joinpath(@__DIR__, "hierarchy_inference.jl"))
const F = Main.FactorialInference
const overlap_blocks = Main.HierarchyInference.overlap_blocks
const fixed_blocks = Main.HierarchyInference.fixed_blocks
const HI = Main.HierarchyInference

"E7's structured mean field (iterated between blocks) on E6's model: one cluster, the same M-step."
function fit_structured(m::F.Model, X, blocking; iters = 200)
    h = HI.fit(HI.Hier([1.0], [m.W], [m.p], [m.Σ]), X, blocking; iters)
    return F.Model(h.W[1], h.p[1], h.Σ[1])
end

# --- the graph -----------------------------------------------------------------------------------

"The block factor's outcome log-weights: the prior of each configuration of the block's bits."
block_logw(p, b, Kc) = vec(log.(p[b])' * Kc .+ log1p.(-p[b])' * (1 .- Kc))

"""
The block graph for one observation x, run by Mycelium. Returns, per block, the posterior over
its 2^B configurations (the marginal of its outcome variable).
"""
function graph_posteriors(m::F.Model, x, blocks; ε = 1e-4)
    d = size(m.W, 1)
    us = [Symbol(:u, i) for i in eachindex(blocks)]; ks = [Symbol(:k, i) for i in eachindex(blocks)]
    b = GraphBuilder()
    variable!(b, :z, d); foreach(u -> variable!(b, u, d), us)
    dims = NamedTuple{(:z, us...)}(ntuple(_ -> d, length(blocks) + 1))
    c = m.W[:, 1]
    factor!(b, :sum, LinearConstraintFactor(dims, d; noise = m.Σ, init_coeff = (rng, r, k) -> Matrix(1.0I, r, k),
                                            init_constant = (rng, r) -> copy(c)))
    factor!(b, :data, DataFactor(:z, x))
    connect!(b, :data, :z, :z; direction = Emitting())
    connect!(b, :sum, :z, :z; direction = Absorbing())
    psb = Dict{Symbol,Any}()
    for (i, bl) in enumerate(blocks)
        Kc = F.configs(length(bl))
        variable!(b, ks[i], size(Kc, 2))
        sw = Symbol(:sw, i)
        factor!(b, sw, SwitchingConstraintFactor((u = d,), size(Kc, 2), d; noise = ε))
        psb[sw] = (A = (u = Matrix(1.0I, d, d),), b = -m.W[:, bl .+ 1] * Kc, logw = block_logw(m.p, bl, Kc))
        connect!(b, sw, :u, us[i]); connect!(b, :sum, us[i], us[i]); connect!(b, sw, :k, ks[i])
    end
    g = build(b)
    ps = merge(LuxCore.initialparameters(Xoshiro(0), g), NamedTuple(psb))
    marg, _, _, _ = infer!(g, tree_schedule(g), ps, LuxCore.initialstates(Xoshiro(0), g))
    return [LenticulumCore.probabilities(getfield(marg, k)) for k in ks]
end

"""
The same messages, vectorised: block b's posterior is its prior times
N(x − c − μ_o − Δ_b k; 0, Σ + V_o + n_b ε I), where μ_o and V_o are the other blocks' prior
moments and n_b the number of blocks (each switch mixture component has width ε).
"""
function block_posteriors(m::F.Model, X, blocks; ε = 1e-4)
    d = size(X, 2); L = length(m.p); nb = length(blocks)
    c = m.W[:, 1]; Δ = m.W[:, 2:end]
    μall = Δ * m.p
    map(blocks) do bl
        o = setdiff(1:L, bl)
        μo = Δ[:, o] * m.p[o]
        Vo = Δ[:, o] * Diagonal(m.p[o] .* (1 .- m.p[o])) * Δ[:, o]'
        C = cholesky(Symmetric(m.Σ + Vo + nb * ε * I))
        Kc = F.configs(length(bl))
        R0 = C.L \ (X' .- (c + μo)); M = C.L \ (Δ[:, bl] * Kc)
        Lg = block_logw(m.p, bl, Kc)' .- (sum(abs2, R0; dims = 1)' .- 2 * (R0' * M) .+ sum(abs2, M; dims = 1)) ./ 2
        mx = maximum(Lg; dims = 2)
        P = exp.(Lg .- mx); P ./ sum(P; dims = 2)                  # n × 2^B
    end
end

"Sufficient statistics: E[k k_ᵀ] exact within a block, factorised across blocks."
function block_stats(Ps, blocks, X)
    n = size(X, 1); L = sum(length, blocks)
    Q = zeros(n, L)
    for (P, bl) in zip(Ps, blocks)
        Q[:, bl] = P * permutedims(F.configs(length(bl)))
    end
    Eφ = vcat(ones(1, n), Q')
    Sφφ = Eφ * Eφ'
    for (P, bl) in zip(Ps, blocks)
        Kc = F.configs(length(bl))
        Sφφ[bl .+ 1, bl .+ 1] = Kc * Diagonal(vec(sum(P; dims = 1))) * Kc'
    end
    return (; Sφφ, Sxφ = X' * Eφ', Eφ)
end

"An E-step for `F.fit`: blocks chosen by `blocking(G)`, G = ΔᵀΣ⁻¹Δ, at every E-step."
estep_blocks(blocking) = (m, X) -> begin
    Δ = m.W[:, 2:end]
    blocks = blocking(Δ' * (Symmetric(m.Σ) \ Δ))
    block_stats(block_posteriors(m, X, blocks), blocks, X)
end

methods(L) = [
    ("exact", F.estep_exact),
    ("graph, blocks of 1 (E6's ADF)", estep_blocks(G -> [[l] for l in 1:L])),
    ("graph, fixed blocks of 4", estep_blocks(G -> fixed_blocks(L, 4))),
    ("graph, overlap blocks of ≤ 4", estep_blocks(G -> overlap_blocks(G, 4))),
    (L > 8 ? [("graph, overlap blocks of ≤ $(L ÷ 2)", estep_blocks(G -> overlap_blocks(G, L ÷ 2)))] : [])...,
    ("mean field (E6)", F.estep_mf),
    ("structured MF, overlap blocks of ≤ 4 (E7)", (:structured, G -> overlap_blocks(G, 4))),
    (L > 8 ? [("structured MF, overlap blocks of ≤ $(L ÷ 2) (E7)", (:structured, G -> overlap_blocks(G, L ÷ 2)))] : [])...,
]

"""
For each truth seed and L: the graph check, then every E-step's EM fixed point from the true
parameters, scored by the exact held-out log-likelihood. E6's truths and data.
"""
function run(; d = 8, Ls = (8, 12), seeds = 1:5, ntrain = 2000, ntest = 2000, iters = 200, ncheck = 10)
    rows = []
    for seed in seeds, L in Ls
        tm = F.truth(d, L; seed)
        X = F.sample(tm, ntrain, Xoshiro(10 + 100(seed - 1))); Xte = F.sample(tm, ntest, Xoshiro(11 + 100(seed - 1)))
        t_ll = F.exact_loglik(tm, Xte)
        Δ = tm.W[:, 2:end]; blocks = overlap_blocks(Δ' * (Symmetric(tm.Σ) \ Δ), 4)
        Pv = block_posteriors(tm, Xte[1:ncheck, :], blocks)
        err = maximum(maximum(abs.(reduce(hcat, graph_posteriors(tm, Xte[i, :], blocks)[j] for i in 1:ncheck)' .- Pv[j]))
                      for j in eachindex(blocks))
        e6 = maximum(abs.(block_stats(block_posteriors(tm, X, [[l] for l in 1:L]), [[l] for l in 1:L], X).Eφ .- F.estep_bp(tm, X).Eφ))
        @printf("seed %d, L = %d: truth %.3f; graph vs vectorised (overlap blocks of ≤ 4) %.1e; blocks of 1 vs E6's ADF %.1e\n",
                seed, L, t_ll, err, e6)
        for (name, es) in methods(L)
            t = @elapsed fitted = es isa Tuple ? fit_structured(tm, X, es[2]; iters) : F.fit(es, tm, X; iters)
            gap = t_ll - F.exact_loglik(fitted, Xte)
            push!(rows, (; seed, L, method = name, gap, time = t, graph_err = err, e6_err = e6))
            @printf("  %-42s from the truth: gap %.3f (%.1f s)\n", name, gap, t)
        end
    end
    return rows
end

main() = run()

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    SwitchingBlocksGraph.main()
end
