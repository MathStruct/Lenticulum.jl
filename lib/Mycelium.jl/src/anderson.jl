# ---------------------------------------------------------------------------
# Anderson acceleration of a schedule: a sweep is a map G on the message store, and
# AndersonSchedule replaces G(x_k) by a combination of the last m + 1 images that best cancels
# the residuals G(x) − x. It keeps G's fixed points. See `anderson.md`.
# ---------------------------------------------------------------------------

mutable struct _AndersonMemory
    layout::Vector{Tuple{Symbol,Int,Int}}   # (kind, edge, dimension) of each mixed message
    w::Vector{Float64}                      # coordinate scales, fixed when the memory starts
    X::Vector{Vector{Float64}}              # scaled iterates x_k
    G::Vector{Vector{Float64}}              # scaled images G(x_k)
    restarts::Int
end

"""
    AndersonSchedule(inner; m = 3, ridge = 1e-14)

Anderson acceleration (Walker & Ni, 2011; type II, undamped) of the schedule `inner`. One
sweep runs one sweep of `inner`, a map ``G`` on the stored messages, and then replaces its
result by

```math
x^+ = G(x_k) - \\Delta G\\,\\gamma, \\qquad \\gamma = \\arg\\min_\\gamma \\lVert f_k - \\Delta F\\,\\gamma\\rVert_2,
```

where ``f = G(x) - x`` and ``\\Delta F``, ``\\Delta G`` hold the differences of the last `m + 1`
residuals and images. A fixed point of ``G`` has ``f = 0``, so the fixed points are `inner`'s.
The cost is one sweep of `inner` per sweep, plus a least-squares problem of `m` columns.

Only Gaussian messages in `Float64` arrays are mixed, in natural parameters (``\\eta`` and the
lower triangle of ``\\Lambda``), with the means and the precisions each scaled by their largest
entry when the memory starts. Where any stored message is of another kind the sweep is `inner`'s,
unchanged. A mixed precision that loses positive semidefiniteness, or a non-finite step, is
replaced by `inner`'s result and the memory cleared (a restart). The memory is kept per
`MessageStore` and cleared when the set of stored messages changes, so one schedule can
serve several stores.

There is no convergence guarantee. Measured on the resistive divider
(`bench/bp_anderson.jl`; `vault/Control/Composing Optimisers.md` §4.3), around
[`spanning_tree_schedule`](@ref) with `m = 2` or 3 it needs 0.09 to 0.19 of flooding's message
evaluations, and about 700 times fewer than the plain spanning-tree order when the loop gain is
near 1; around a flooding schedule it was slower in three of four settings, or failed. On Gaussian
grids (`bench/bp_grid.jl`, §4.4) it needs 5 to 19 times fewer sweeps than the plain spanning-tree
order in the slow regime, and there it also speeds up flooding. Parameter
gradients through an `AndersonSchedule` (schedule replay) are not supported.
"""
struct AndersonSchedule{S<:AbstractSchedule} <: AbstractSchedule
    inner::S
    m::Int
    ridge::Float64
    memory::WeakKeyDict{MessageStore,_AndersonMemory}
    function AndersonSchedule(inner::S; m::Int = 3, ridge::Real = 1e-14) where {S<:AbstractSchedule}
        m >= 1 || throw(ArgumentError("AndersonSchedule needs a memory m ≥ 1; got $m"))
        ridge >= 0 || throw(ArgumentError("ridge must be ≥ 0; got $ridge"))
        inner isa ResidualSchedule && throw(ArgumentError(
            "AndersonSchedule wraps a static schedule; a ResidualSchedule's sweep is not a fixed map"))
        return new{S}(inner, m, float(ridge), WeakKeyDict{MessageStore,_AndersonMemory}())
    end
end

_default_criterion(a::AndersonSchedule) = _default_criterion(a.inner)

const _HostMatrix = Union{Matrix{Float64},LinearAlgebra.Symmetric{Float64,Matrix{Float64}},LinearAlgebra.Diagonal{Float64,Vector{Float64}}}

# the stored Gaussian messages as one vector, and their layout; `nothing` if a stored message
# cannot be mixed (another belief type, another element type, a device array)
function _anderson_state(s::MessageStore)
    x = Float64[]; layout = Tuple{Symbol,Int,Int}[]
    for (kind, msgs) in ((:to_factor, s.to_factor), (:to_variable, s.to_variable))
        for e in eachindex(msgs)
            m = msgs[e]
            m === nothing && continue
            b = m.belief
            b isa LenticulumCore.GaussianBelief || return nothing
            (b.η isa Vector{Float64} && b.Λ isa _HostMatrix) || return nothing
            d = length(b.η)
            append!(x, b.η)
            for j in 1:d, i in j:d
                push!(x, b.Λ[i, j])
            end
            push!(layout, (kind, e, d))
        end
    end
    return x, layout
end

# a precision-like block must not lose positive semidefiniteness
function _anderson_admissible(x::Vector{Float64}, gx::Vector{Float64}, layout)
    all(isfinite, x) || return false
    k = 0
    for (_, _, d) in layout
        k += d
        Λ = zeros(d, d); Λg = zeros(d, d)
        for j in 1:d, i in j:d
            k += 1
            Λ[i, j] = Λ[j, i] = x[k]; Λg[i, j] = Λg[j, i] = gx[k]
        end
        λ = d == 1 ? Λ[1, 1] : LinearAlgebra.eigmin(LinearAlgebra.Symmetric(Λ))
        λg = d == 1 ? Λg[1, 1] : LinearAlgebra.eigmin(LinearAlgebra.Symmetric(Λg))
        λ >= min(0.0, λg) - 1e-12 * max(1.0, maximum(abs, Λg)) || return false
    end
    return true
end

function _anderson_write!(s::MessageStore, x::Vector{Float64}, layout)
    k = 0
    for (kind, e, d) in layout
        msgs = kind === :to_factor ? s.to_factor : s.to_variable
        old = msgs[e]
        η = x[k+1:k+d]; k += d
        Λ = zeros(d, d)
        for j in 1:d, i in j:d
            k += 1
            Λ[i, j] = Λ[j, i] = x[k]
        end
        msgs[e] = Message(LenticulumCore.GaussianBelief(η, Λ, old.belief.improper), old.iteration)
    end
    return s
end

# the scale of each coordinate: means by the largest |η|, precisions by the largest |Λ|
function _anderson_scales(x::Vector{Float64}, layout)
    isη = falses(length(x)); k = 0
    for (_, _, d) in layout
        isη[k+1:k+d] .= true
        k += d + d * (d + 1) ÷ 2
    end
    sη = maximum(abs, x[isη]; init = 0.0); sΛ = maximum(abs, x[.!isη]; init = 0.0)
    return [isη[i] ? 1 / max(sη, eps()) : 1 / max(sΛ, eps()) for i in eachindex(x)]
end

function sweep!(s::MessageStore, g::FactorGraph, sched::AndersonSchedule, @nospecialize(ps), @nospecialize(st);
                damping::Real = 0.0)
    before = _anderson_state(s)
    res, st = sweep!(s, g, sched.inner, ps, st; damping)
    after = _anderson_state(s)
    mem = get(sched.memory, s, nothing)
    if before === nothing || after === nothing || before[2] != after[2]
        mem === nothing || (empty!(mem.X); empty!(mem.G); mem.layout = Tuple{Symbol,Int,Int}[])
        return res, st
    end
    x, layout = before; gx = after[1]
    if mem === nothing || mem.layout != layout
        mem = _AndersonMemory(layout, _anderson_scales(gx, layout), Vector{Float64}[], Vector{Float64}[],
                              mem === nothing ? 0 : mem.restarts)
        sched.memory[s] = mem
    end
    w = mem.w
    push!(mem.X, w .* x); push!(mem.G, w .* gx)
    length(mem.X) > sched.m + 1 && (popfirst!(mem.X); popfirst!(mem.G))
    length(mem.X) < 2 && return res, st
    n = length(mem.X)
    F = [mem.G[i] .- mem.X[i] for i in 1:n]
    ΔF = reduce(hcat, [F[i+1] .- F[i] for i in 1:n-1])
    ΔG = reduce(hcat, [mem.G[i+1] .- mem.G[i] for i in 1:n-1])
    A = ΔF' * ΔF
    γ = (A + sched.ridge * max(LinearAlgebra.tr(A), eps()) * LinearAlgebra.I) \ (ΔF' * F[end])
    xn = (mem.G[end] .- ΔG * γ) ./ w
    if _anderson_admissible(xn, gx, layout)
        _anderson_write!(s, xn, layout)
    else
        mem.restarts += 1
        empty!(mem.X); empty!(mem.G)
    end
    return res, st
end

"""
    anderson_restarts(sched::AndersonSchedule, store) -> Int

How many times the acceleration fell back to the plain sweep on `store` (a step that was not
finite or would have made a precision indefinite).
"""
anderson_restarts(sched::AndersonSchedule, s::MessageStore) =
    (mem = get(sched.memory, s, nothing); mem === nothing ? 0 : mem.restarts)
