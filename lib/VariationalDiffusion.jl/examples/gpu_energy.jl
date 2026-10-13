# An energy network at MNIST size on a GPU: correctness against the CPU, and the costs.
#
# Needs CUDA and an environment with LuxCUDA (not a dependency of anything here):
#
#     julia --project=<env with VariationalDiffusion (dev), Lux, LuxCUDA, Zygote, ForwardDiff,
#                      DifferentiationInterface, ADTypes, MLDataDevices> lib/VariationalDiffusion.jl/examples/gpu_energy.jl
#
# The network is E(x, t) on x ∈ ℝ⁷⁹⁴ (an image and a label), 802 → 512 → 512 → 1, with
# ad = SecondOrder(AutoForwardDiff(), AutoZygote()) and device = (to = gpu_device(), back = cpu_device()):
# the solver's state stays on the CPU, every network call and derivative runs on the GPU in
# Float32. Prints (1) the agreement with the same network on the CPU in Float64, (2) the cost of
# a training step, of ε and of a Hessian-vector product, and (3) the cost of classification-shaped
# solves (784 coordinates clamped, 10 free) for many queries at once. Recorded in
# `energy_network.md` §8.

using VariationalDiffusion, Lux, LuxCUDA, LuxCore, Zygote, ForwardDiff, Random, LinearAlgebra, Printf, MLDataDevices
using DifferentiationInterface, ADTypes
gdev, cdev = gpu_device(), cpu_device()
n = 794
net = Chain(Dense(n + 8 => 512, swish), Dense(512 => 512, swish), Dense(512 => 1))
const FREQ = [0.5, 1.0, 2.0, 4.0]
# the time features: 8 numbers per column, made on the CPU and placed next to x
function inp(x, t)
    te = Zygote.@ignore begin
        tt = t isa Number ? fill(Float64(t), 1, size(x, 2)) : Array(t)
        f = vcat(sin.((2π .* tt) .* FREQ), cos.((2π .* tt) .* FREQ))
        x isa Array ? f : copyto!(similar(x, Float32, size(f)), f)
    end
    return vcat(x, te)
end
S = VPSDE(); ad = SecondOrder(AutoForwardDiff(), AutoZygote())
ps0, st0 = Lux.setup(Xoshiro(0), net)
f64(p) = Lux.fmap(a -> a isa AbstractArray ? Float64.(a) : a, p)
cpu = NoisePredictor(EnergyNetwork(net), S; input = inp, ad)
gpu = NoisePredictor(EnergyNetwork(net), S; input = inp, ad, device = (to = gdev, back = cdev))
psc, psg = f64(ps0), gdev(ps0)
rel(a, b) = norm(a - b) / norm(b)
B = 128
x₀ = randn(Xoshiro(1), n, B); tb = rand(Xoshiro(2), 1, B) .* 0.9 .+ 0.05; εb = randn(Xoshiro(3), n, B)
Lc, gc, _ = denoising_gradient(cpu, x₀, tb, εb, psc, st0)
Lg, gg, _ = denoising_gradient(gpu, x₀, tb, εb, psg, st0)
@printf("training step: loss rel %.1e, grad rel %.1e (layer 1), cotangent on %s\n", abs(Lg - Lc) / Lc,
        rel(Array(gg.layer_1.weight), gc.layer_1.weight), typeof(gg.layer_1.weight).name.name)
for _ in 1:2
    tg = @elapsed (denoising_gradient(gpu, x₀, tb, εb, psg, st0); CUDA.synchronize())
    tc = @elapsed denoising_gradient(cpu, x₀, tb, εb, psc, st0)
    @printf("  per step (B = %d): GPU %.4f s, CPU %.4f s\n", B, tg, tc)
end
X = randn(Xoshiro(5), n, 400); W = randn(Xoshiro(6), n, 400)
@printf("eps rel %.1e, hvp rel %.1e, jac block rel %.1e\n",
        rel(first(epsilon(gpu, X, 0.01, psg, st0)), first(epsilon(cpu, X, 0.01, psc, st0))),
        rel(epsilon_vjp_input(gpu, X, 0.01, psg, st0, W), epsilon_vjp_input(cpu, X, 0.01, psc, st0, W)),
        rel(epsilon_jacobian_block(gpu, X, 0.01, psg, st0, 785:794, fill(1 / 400, 400)), epsilon_jacobian_block(cpu, X, 0.01, psc, st0, 785:794, fill(1 / 400, 400))))
X4 = randn(n, 4000); V4 = zeros(n, 4000); V4[790, :] .= 1
for _ in 1:2
    a = @elapsed (epsilon(gpu, X4, 0.01, psg, st0); CUDA.synchronize())
    b = @elapsed (epsilon_vjp_input(gpu, X4, 0.01, psg, st0, V4); CUDA.synchronize())
    @printf("4000 columns: eps %.4f s, hvp %.4f s\n", a, b)
end
nodes = field_nodes(Xoshiro(1), n; samples = 2, levels = range(0.002, 0.05; length = 4))
ρ = [fill(Inf, 784); zeros(10)]
m = ImplicitDiffusion(gpu, nodes; batched = true)
for Bq in (1, 100, 1000)    # (3)
    Z = randn(Xoshiro(4), n, Bq)
    implicit_infer_many(m, Z[:, 1:1], ρ, psg, st0; maxiters = 1, tol = 0.0)
    t = @elapsed implicit_infer_many(m, Z, ρ, psg, st0; maxiters = 3, tol = 0.0)
    @printf("%4d queries, 3 iterations: %.2f s (%.2f ms per query-iteration)\n", Bq, t, 1000t / (3Bq))
end
mc = ImplicitDiffusion(cpu, nodes; batched = true)
Z = randn(Xoshiro(4), n, 3)
sc, _ = implicit_infer_many(mc, Z, ρ, psc, st0; maxiters = 3, tol = 0.0)
sg, _ = implicit_infer_many(m, Z, ρ, psg, st0; maxiters = 3, tol = 0.0)
@printf("iterates GPU vs CPU: max rel diff of the free block %.1e\n", maximum(rel(a.z[785:end], b.z[785:end]) for (a, b) in zip(sg, sc)))
