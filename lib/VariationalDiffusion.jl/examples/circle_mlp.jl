# A small diffusion model, trained and then used as a relation.
#
# A 3-layer MLP (≈ 5k parameters, no U-Net, no attention) learns the noise predictor of points
# on the unit circle in R². The same network then answers "given x, which y?" in either
# branch, and the answer is differentiated by the implicit-function-theorem adjoint, checked
# against finite differences. Runs on the CPU in about a minute.
#
#     cd lib/VariationalDiffusion.jl
#     julia --project=test examples/circle_mlp.jl
#
# Swap `AutoZygote()` for `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`,
# `AutoMooncake()`, or `AutoReactant()` (with Reactant loaded) — nothing else changes.
# See `src/backends.md`.
using VariationalDiffusion, Lux, LuxCore, Random, LinearAlgebra, Printf, Optimisers
using DifferentiationInterface, Zygote

const sched = VPSDE()
const FREQ = [0.5, 1.0, 2.0, 4.0, 8.0]
embed(t) = vcat(sin.((2π .* t) .* FREQ), cos.((2π .* t) .* FREQ))   # t scalar or a 1×B row
input(x, t) = vcat(x, embed(t) .* ones(eltype(x), 1, size(x, 2)))

const mlp = Chain(Dense(12 => 64, swish), Dense(64 => 64, swish), Dense(64 => 2))

# denoising score matching on the circle, on the noise levels inference uses (t ≤ 0.2)
function batch(rng, B)
    θ = 2π .* rand(rng, B)
    x₀ = vcat(cos.(θ)', sin.(θ)') .+ 0.02 .* randn(rng, 2, B)
    t = 1e-3 .+ 0.2 .* rand(rng, 1, B)
    ε = randn(rng, 2, B)
    xₜ = alpha.(Ref(sched), t) .* x₀ .+ sigma.(Ref(sched), t) .* ε
    return (Float32.(vcat(xₜ, embed(t))), Float32.(ε))
end

function train(rng; steps = 6000)
    ps, st = Lux.setup(rng, mlp)
    ts = Training.TrainState(mlp, ps, st, Optimisers.Adam(1.0f-3))
    for k in 1:steps
        _, loss, _, ts = Training.single_train_step!(AutoZygote(), MSELoss(), batch(rng, 256), ts)
        k % 2000 == 0 && @printf "step %5d   denoising loss %.4f\n" k loss
    end
    return ts.parameters, ts.states
end

f64(x::AbstractArray) = Float64.(x); f64(x::NamedTuple) = map(f64, x); f64(x) = x

rng = Xoshiro(0)
@printf "parameters: %d\n" Lux.parameterlength(mlp)
ps, st = train(rng)
ps = f64(ps)

# the trained network as a relation; ad = … is used only by the backward pass
pred = NoisePredictor(mlp, sched; input, ad = AutoZygote())
m = ImplicitDiffusion(pred, field_nodes(Xoshiro(1), 2; samples = 8))
for (x, y₀) in ((0.6, 0.5), (0.6, -0.5), (0.0, 0.7))
    sol, _ = implicit_infer(m, [x, y₀], [Inf, 0.0], ps, st)      # x clamped, y free
    @printf "x = %.1f, start y = %+.1f  →  y = %+.4f   radius %.4f   converged %s  stable %s\n" x y₀ sol.z[2] norm(sol.z) sol.converged sol.stable
end

# differentiate ℓ = (y - 0.7)²/2 through inference, w.r.t. the input and one weight
ℓ(z) = (z[2] - 0.7)^2 / 2
solve(x, ps) = first(implicit_infer(m, [x, 0.5], [Inf, 0.0], ps, st; tol = 1e-12)).z
sol, _ = implicit_infer(m, [0.6, 0.5], [Inf, 0.0], ps, st; tol = 1e-12)
b = implicit_pullback(m, sol, [0.6, 0.5], [Inf, 0.0], [0.0, sol.z[2] - 0.7], ps, st)
h = 1e-6
@printf "dℓ/dx     adjoint %+.8f   finite differences %+.8f\n" b.z₀[1] (ℓ(solve(0.6 + h, ps)) - ℓ(solve(0.6 - h, ps))) / 2h
bump(δ) = (W = copy(ps.layer_2.weight); W[5] += δ; merge(ps, (layer_2 = (weight = W, bias = ps.layer_2.bias),)))
@printf "dℓ/dW[5]  adjoint %+.8f   finite differences %+.8f\n" b.ps.layer_2.weight[5] (ℓ(solve(0.6, bump(h))) - ℓ(solve(0.6, bump(-h)))) / 2h
