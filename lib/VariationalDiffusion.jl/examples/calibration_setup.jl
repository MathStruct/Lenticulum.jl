# The shared part of the calibration benchmarks (`calibration.jl`, `nodes_calibration.jl`): the three
# relations, the training of their noise predictors, the query grid with the remedies' verdicts and
# the metrics. Included by those scripts; not run on its own.
@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true" || get(ENV, "CALIBRATION_SMOKE", "") == "1"

using VariationalDiffusion, Lux, LuxCore, Random, LinearAlgebra, Printf, Optimisers, Statistics
using DifferentiationInterface, Zygote

const sched = VPSDE()
const FREQ = [0.5, 1.0, 2.0, 4.0, 8.0]
embed(t) = vcat(sin.((2π .* t) .* FREQ), cos.((2π .* t) .* FREQ))
input(x, t) = vcat(x, embed(t) .* ones(eltype(x), 1, size(x, 2)))
const mlp = Chain(Dense(12 => 64, swish), Dense(64 => 64, swish), Dense(64 => 2))
const τ = 0.1

# --- the relations: a sampler and the branches y(x) --------------------------------------------
circle_sample(rng, B) = (θ = 2π .* rand(rng, B); vcat(cos.(θ)', sin.(θ)'))
circle_branches(x) = abs(x) <= 1 ? [sqrt(1 - x^2), -sqrt(1 - x^2)] : Float64[]
gerono_sample(rng, B) = (θ = 2π .* rand(rng, B); vcat(cos.(θ)', (sin.(θ) .* cos.(θ))'))
gerono_branches(x) = abs(x) <= 1 ? unique([x * sqrt(1 - x^2), -x * sqrt(1 - x^2)]) : Float64[]
function rings_sample(rng, B)
    r = [rand(rng) < 0.9 ? 1.0 : 2.0 for _ in 1:B]; θ = 2π .* rand(rng, B)
    return vcat((r .* cos.(θ))', (r .* sin.(θ))')
end
rings_branches(x) = reduce(vcat, [abs(x) <= r ? [sqrt(r^2 - x^2), -sqrt(r^2 - x^2)] : Float64[] for r in (1.0, 2.0)])
const RELATIONS = (
    (name = "circle", sample = circle_sample, branches = circle_branches, xs = -1.4:0.1:1.4, radii = (1.0,)),
    (name = "lemniscate", sample = gerono_sample, branches = gerono_branches, xs = -1.4:0.1:1.4, radii = ()),
    (name = "two circles (0.9 / 0.1)", sample = rings_sample, branches = rings_branches, xs = -2.4:0.1:2.4, radii = (1.0, 2.0)),
)

function batch(rng, sample, B)
    x₀ = sample(rng, B) .+ 0.02 .* randn(rng, 2, B)
    t = 1e-3 .+ 0.2 .* rand(rng, 1, B)
    ε = randn(rng, 2, B)
    xₜ = alpha.(Ref(sched), t) .* x₀ .+ sigma.(Ref(sched), t) .* ε
    return (Float32.(vcat(xₜ, embed(t))), Float32.(ε))
end
function train(rng, sample; steps = SMOKE ? 1 : 6000)
    ps, st = Lux.setup(rng, mlp)
    ts = Training.TrainState(mlp, ps, st, Optimisers.Adam(1.0f-3))
    loss = NaN
    for _ in 1:steps
        _, loss, _, ts = Training.single_train_step!(AutoZygote(), MSELoss(), batch(rng, sample, SMOKE ? 2 : 256), ts)
    end
    return ts.parameters, ts.states, loss
end
f64(x::AbstractArray) = Float64.(x); f64(x::NamedTuple) = map(f64, x); f64(x) = x

# --- the roots of one query grid, with the remedies' verdicts -----------------------------------
# Each root carries three verdicts of F3 (the calibration remedies): `round_trip` (clamp the root's y,
# solve back for x, keep it if x moves by at most τ), `ridge_residual` (keep it if the normal Newton
# step to the ridge, codimension 1, is at most τ) and `coverage` (keep it unless flagged at
# p < 0.05 against held-out data). With `remedies = false` only the ridge verdict is computed; the
# other two are `missing`, so a filter on them fails loudly. The queries run on all threads.
function query_grid(m, ps, st, rel, ref; spread = 1.5, continuation = (), remedies::Bool = true)
    out = Vector{Any}(undef, length(rel.xs))
    Threads.@threads for q in eachindex(rel.xs)
        x = rel.xs[q]
        SMOKE && !(q in (1, cld(length(rel.xs), 2))) && (out[q] = []; continue)
        z₀, ρ = [x, 0.0], [Inf, 0.0]
        sols, _ = implicit_roots(m, z₀, ρ, ps, st; nstarts = SMOKE ? 1 : 32, spread, rng = Xoshiro(q), continuation)
        out[q] = [(y = s.z[2], trip = !remedies ? missing : first(round_trip(m, s, z₀, ρ, ps, st)).distance ≤ τ,
                   ridge = first(ridge_residual(m, s, ρ, ps, st)).distance ≤ τ,
                   cov = !remedies ? missing : !coverage(ref, s).flagged) for s in sols]
    end
    return out
end

# --- the metrics over one query grid, for the roots a filter keeps ------------------------------
function calibrate(rel, grid, keep)
    found, total, roots, spurious, offdomain, offqueries, dists, radial = 0, 0, 0, 0, 0, 0, Float64[], Float64[]
    minority_found, minority_total = 0, 0
    for (q, x) in enumerate(rel.xs)
        ys = [r.y for r in grid[q] if keep(r)]
        br = rel.branches(x)
        roots += length(ys)
        if isempty(br)
            offqueries += 1
            offdomain += !isempty(ys)
            spurious += length(ys)
            continue
        end
        for y in ys
            d = minimum(abs.(br .- y))
            if d > τ
                spurious += 1
            else
                push!(dists, d)
                isempty(rel.radii) || push!(radial, norm([x, y]) - rel.radii[argmin(abs.(collect(rel.radii) .- norm([x, y])))])
            end
        end
        total += length(br)
        found += count(b -> any(y -> abs(y - b) <= τ, ys), br)
        if rel.name == "two circles (0.9 / 0.1)"                       # the radius-2 branches alone
            outer = abs(x) <= 2 ? [sqrt(4 - x^2), -sqrt(4 - x^2)] : Float64[]
            minority_total += length(outer)
            minority_found += count(b -> any(y -> abs(y - b) <= τ, ys), outer)
        end
    end
    return (recall = found / max(total, 1), spurious = spurious / max(roots, 1), roots = roots,
            offdomain = offdomain, offqueries = offqueries, dist = isempty(dists) ? NaN : mean(dists),
            radial = isempty(radial) ? NaN : mean(radial),
            minority = minority_total == 0 ? NaN : minority_found / minority_total)
end

