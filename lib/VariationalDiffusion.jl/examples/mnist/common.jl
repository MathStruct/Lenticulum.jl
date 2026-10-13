# MNIST in both directions: the shared definitions of `train.jl` and `evaluate.jl`.
#
# One energy-parametrised diffusion model of the joint state z = (x, y): the image x ∈ [-1, 1]^784
# and the label y = c·onehot ∈ ℝ^10. The direction of a query is chosen by its precisions only.
# See `MNIST in Both Directions.md` in the vault.

using VariationalDiffusion, LenticulumCore
using Lux, LuxCUDA, LuxCore, MLDataDevices, Zygote, ForwardDiff, Optimisers
using DifferentiationInterface, ADTypes
using MLDatasets, JLD2, Random, LinearAlgebra, Statistics, Printf

const NX, NY = 784, 10
const N = NX + NY
const LABEL = (NX + 1):N
const IMAGE = 1:NX
const C = 1.0                                   # the label code: C · one-hot
const OUT = joinpath(homedir(), ".cache", "lenticulum", "mnist")
mkpath(OUT)

const GDEV, CDEV = gpu_device(), cpu_device()
const SCHED = VPSDE()
const AD = SecondOrder(AutoForwardDiff(), AutoZygote())

# --- data -------------------------------------------------------------------------------------
codes() = C .* Matrix(1.0I, NY, NY)
function joint(split)
    d = MNIST(split)
    X = 2 .* reshape(Float64.(d.features), NX, :) .- 1
    Y = codes()[:, d.targets .+ 1]
    return vcat(X, Y), d.targets
end

# --- the model --------------------------------------------------------------------------------
# BLOCKS=1: image and label are two noise blocks with their own levels (UniDiffuser), and the
# network sees both times; otherwise one level for all coordinates
const BLOCKS = get(ENV, "BLOCKS", "0") == "1"
const NBLOCKS = BLOCKS ? 2 : 1

# time features: sin and cos of 2π·f·t for 8 frequencies and each block's time, made on the CPU
# (16 numbers per block and column) and placed next to x on its device
const FREQ = 2.0 .^ (-1:6)
function time_input(x, t)
    te = Zygote.@ignore begin
        tt = t isa Number ? fill(Float64(t), 1, size(x, 2)) : Array(t)        # 1 × B, or blocks × B
        f = vcat(sin.((2π .* kron(tt, FREQ))), cos.((2π .* kron(tt, FREQ))))
        x isa Array ? f : copyto!(similar(x, Float32, size(f)), f)
    end
    return vcat(x, te)
end

# KIND=energy (default): a scalar energy, ε = σ∇ₓE; KIND=eps: a network that outputs ε directly
# (no energy, so no Laplace weights, but no second derivatives in training either)
const KIND = get(ENV, "KIND", "energy")
const WIDTH = 1024
network() = Chain(Dense(N + 2NBLOCKS * length(FREQ) => WIDTH, swish), Dense(WIDTH => WIDTH, swish),
                  Dense(WIDTH => WIDTH, swish), Dense(WIDTH => KIND == "eps" ? N : 1))
const BLOCKSPEC = BLOCKS ? [IMAGE, LABEL] : nothing
predictor(net; device = (to = GDEV, back = CDEV)) = KIND == "eps" ?
    NoisePredictor(net, SCHED; input = time_input, ad = AutoZygote(), device, blocks = BLOCKSPEC) :
    NoisePredictor(EnergyNetwork(net), SCHED; input = time_input, ad = AD, device, blocks = BLOCKSPEC)

# with blocks, the levels of a query: its fully clamped blocks held clean (HELD=1, default), or
# every block annealed at the nodes' level (HELD=0, the joint field)
query_levels(pred, ρ) = BLOCKS && get(ENV, "HELD", "1") == "1" ? conditional_levels(pred, ρ) : nothing

# the trained model; MODEL=<name> selects ~/.cache/lenticulum/mnist/<name>.jld2 (default "energy")
modelfile() = joinpath(OUT, get(ENV, "MODEL", "energy") * ".jld2")
load_model() = (d = JLD2.load(modelfile()); (d["ps"], d["st"]))

# the field's nodes for a query: `samples` antithetic draws at each level
query_nodes(levels; samples = 2, rng = Xoshiro(1)) = field_nodes(rng, N; levels, samples)
