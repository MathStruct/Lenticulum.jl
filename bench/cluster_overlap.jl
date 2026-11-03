# Does grouping switches by overlap help? E7's additive wiring through Mycelium: z = Σₗ uₗ + ε in
# d = 8 dimensions, L = 8 binary switches uₗ ∈ {0, Wₗ}, the columns of W in overlapping pairs. For
# each of N points drawn from the model, the switch posteriors P(kₗ = 2 | z) of loopy graphs are
# compared with the exact ones (every factor in one cluster node):
#
#   - one switching factor per switch, all feeding the sum;
#   - E7's block wiring (`partial_sums`): each block's switches feed a partial sum s_b = Σ_{l ∈ b} uₗ, the
#     blocks' s_b the sum, and each block (its switches and its partial sum) is one cluster node; blocks of 4 grouped
#     by overlap (`switch_overlaps`, `overlap_groups`) or in index order;
#   - clusters of the same 4 switches without the partial sum (they meet only at the sum node).
#
# Recorded in `lib/Mycelium.jl/src/cluster.md` and L-SPEC-009 §6.
# Run:  julia --project=. bench/cluster_overlap.jl
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random, Printf, Statistics

const d, L, B, N = 8, 8, 4, 100
const E = Matrix(1.0I, d, d)
sw_ps(W, l) = (A = (u = E,), b = hcat(zeros(d), W[:, l]), logw = zeros(2))

function add_switch!(b, ps, W, l, into)
    u, k, sw = Symbol(:u, l), Symbol(:k, l), Symbol(:sw, l)
    variable!(b, u, d); variable!(b, k, 2)
    factor!(b, sw, SwitchingConstraintFactor((u = d,), 2, d; noise = 1e-4))
    connect!(b, into, u, u); connect!(b, sw, :u, u); connect!(b, sw, :k, k)
    ps[sw] = sw_ps(W, l)
end
function graph_of(b, ps)
    return build(b), NamedTuple{Tuple(keys(ps))}(Tuple(values(ps)))
end

# every switch feeds the sum
function direct(W, Σ, x)
    b = GraphBuilder(); variable!(b, :z, d)
    factor!(b, :data, DataFactor(:z, x)); connect!(b, :data, :z, :z; direction = Emitting())
    dims = NamedTuple{(:z, (Symbol(:u, l) for l in 1:L)...)}(ntuple(_ -> d, L + 1))
    factor!(b, :sum, LinearConstraintFactor(dims, d; noise = Σ)); connect!(b, :sum, :z, :z)
    ps = Dict{Symbol,Any}(:data => NamedTuple(), :sum => (A = NamedTuple{keys(dims)}((E, ntuple(_ -> -E, L)...)), c = zeros(d)))
    for l in 1:L; add_switch!(b, ps, W, l, :sum); end
    return graph_of(b, ps)
end

# E7's block wiring: partial sums per block, as an edit of the direct graph (`partial_sums`)
function blocked(W, Σ, x, blocks)
    g, ps = direct(W, Σ, x)
    g2, ps2, groups = partial_sums(g, ps, :sum, [Symbol(:sw, l) for l in 1:L], blocks; noise = 1e-4)
    gc = cluster(g2, groups)
    return gc, cluster_parameters(gc, ps2)
end

# P(kₗ = 2) of every switch, read from the graph or the cluster holding it
function posteriors(g, s, ps)
    map(1:L) do l
        k = Symbol(:k, l)
        haskey(g.variable_index, k) && return exp(marginal(s, g, g.variable_index[k]).logp[2])
        node = only(f for f in g.factors if f.factor isa ClusterFactor && k in f.factor.variables)
        return exp(cluster_marginal(s, g, node.name, k, ps).logp[2])
    end
end
function run(g, ps; sweeps = 50)
    s = MessageStore(g)
    for _ in 1:sweeps; sweep!(s, g, SequentialSchedule(all_tasks(g)), ps, NamedTuple()); end
    return posteriors(g, s, ps)
end

rng = Xoshiro(7)
W = randn(rng, d, L) ./ sqrt(d)
for l in 2:2:L          # overlapping pairs, then interleaved so that index order splits them
    W[:, l] .= 0.9 .* W[:, l - 1] .+ 0.3 .* randn(rng, d) ./ sqrt(d)
end
W = W[:, [1, 3, 5, 7, 2, 4, 6, 8]]
Σ = Matrix(0.05I, d, d)
sws = [Symbol(:sw, l) for l in 1:L]
g0, ps0 = direct(W, Σ, zeros(d))
groups = overlap_groups(switch_overlaps(g0, ps0, :sum, sws), B)
fixed = [collect(1:4), collect(5:8)]
println("overlap groups: ", groups, "   index order: ", fixed)
rows = [:plain, :switches_only, :fixed, :overlap]
errs = Dict(r => Float64[] for r in rows)
for n in 1:N
    k = rand(rng, Bool, L)
    x = W * k .+ sqrt(0.05) .* randn(rng, d)
    g, ps = direct(W, Σ, x)
    ge = cluster(g, [:all => [:sum; sws]]); pe = run(ge, cluster_parameters(ge, ps); sweeps = 1)
    push!(errs[:plain], mean(abs, run(g, ps) .- pe))
    gs = cluster(g, [Symbol(:blk, i) => sws[gi] for (i, gi) in enumerate(groups)])
    push!(errs[:switches_only], mean(abs, run(gs, cluster_parameters(gs, ps)) .- pe))
    for (key, gr) in ((:fixed, fixed), (:overlap, groups))
        gb, pb = blocked(W, Σ, x, gr)
        push!(errs[key], mean(abs, run(gb, pb) .- pe))
    end
end
labels = Dict(:plain => "one factor per switch", :switches_only => "overlap clusters, no partial sum",
              :fixed => "blocks of 4, index order", :overlap => "blocks of 4, by overlap")
@printf("%-34s %12s %12s\n", "E-step (loopy, 50 sweeps)", "mean |ΔP|", "max |ΔP|")
for key in rows
    @printf("%-34s %12.4f %12.4f\n", labels[key], mean(errs[key]), maximum(errs[key]))
end
