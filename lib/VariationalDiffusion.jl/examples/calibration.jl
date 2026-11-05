# Calibration of implicit relations: how well the stable roots of a trained noise predictor's
# field recover a known relation. Recorded in `examples/calibration.md`.
#
#     cd lib/VariationalDiffusion.jl
#     julia -t auto --project=test examples/calibration.jl        (about 70 minutes on a CPU, mostly the queries)
#
# Three relations in R², each learned by the 3-layer MLP of `circle_mlp.jl` (denoising score
# matching, 6000 Adam steps, data noise 0.02):
#   - the unit circle;
#   - the lemniscate of Gerono, y² = x² − x⁴ (two branches meeting at the origin);
#   - two concentric circles of radius 1 and 2 with weights 0.9 and 0.1 (a minority branch).
# Queries clamp x on a grid and ask for every stable y (`implicit_roots`, 33 starts). Against the
# relation's branches y(x), with tolerance τ = 0.1 (five data-noise widths):
#   - branch recall: the fraction of true branches matched by a root within τ;
#   - spurious roots: the fraction of roots farther than τ from every branch at their x;
#   - off-domain queries: queries with no branch at their x that still return a root;
#   - precision distance: mean |y_root − nearest branch| over the roots matched to a branch;
#   - for circles, the radial bias of matched roots, mean (|z| − r).
# Each model is queried with three choices of the field's noise levels (no retraining): low
# (t ≤ 0.01), the default (`field_nodes`, t ≤ 0.05) and high (t ≤ 0.2).
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
function train(rng, sample; steps = 6000)
    ps, st = Lux.setup(rng, mlp)
    ts = Training.TrainState(mlp, ps, st, Optimisers.Adam(1.0f-3))
    loss = NaN
    for _ in 1:steps
        _, loss, _, ts = Training.single_train_step!(AutoZygote(), MSELoss(), batch(rng, sample, 256), ts)
    end
    return ts.parameters, ts.states, loss
end
f64(x::AbstractArray) = Float64.(x); f64(x::NamedTuple) = map(f64, x); f64(x) = x

# --- the metrics over one query grid ------------------------------------------------------------
function calibrate(m, ps, st, rel; spread = 1.5)
    found, total, roots, spurious, offdomain, offqueries, dists, radial = 0, 0, 0, 0, 0, 0, Float64[], Float64[]
    minority_found, minority_total = 0, 0
    for (q, x) in enumerate(rel.xs)
        sols, _ = implicit_roots(m, [x, 0.0], [Inf, 0.0], ps, st; nstarts = 32, spread, rng = Xoshiro(q))
        ys = [s.z[2] for s in sols]
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

const LEVELS = (("low, t ≤ 0.01", range(0.001, 0.01; length = 8)), ("default, t ≤ 0.05", range(0.002, 0.05; length = 8)),
                ("high, t ≤ 0.2", range(0.01, 0.2; length = 8)))

for (k, rel) in enumerate(RELATIONS)
    t0 = time()
    ps, st, loss = train(Xoshiro(100 + k), rel.sample)
    ps = f64(ps)
    @printf "\n%s: trained in %.0f s, final denoising loss %.4f\n" rel.name time() - t0 loss
    pred = NoisePredictor(mlp, sched; input, ad = AutoZygote())
    for (label, levels) in LEVELS
        m = ImplicitDiffusion(pred, field_nodes(Xoshiro(1), 2; samples = 8, levels))
        r = calibrate(m, ps, st, rel; spread = rel.name == "two circles (0.9 / 0.1)" ? 2.5 : 1.5)
        @printf "  %-18s recall %.3f  spurious %.3f (%d roots)  off-domain %d/%d  dist %.4f  radial %+.4f%s\n" label r.recall r.spurious r.roots r.offdomain r.offqueries r.dist r.radial (isnan(r.minority) ? "" : @sprintf("  minority recall %.3f", r.minority))
    end
end
