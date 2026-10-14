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
# the floor of the reverse diffusion when sampling (TMIN, env): the sampler stops there and returns
# Tweedie's denoised estimate. Training and the field keep `SCHED`; only `predictor(...; sampling = true)`
# uses this schedule, which has the same α and σ.
const SAMPLE_SCHED = VPSDE(; tmin = parse(Float64, get(ENV, "TMIN", "1e-3")))
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

# KIND=energy (default): a scalar energy, ε = σ∇ₓE, from an MLP; KIND=conv: the same from a
# convolutional network (below); KIND=eps: an MLP that outputs ε directly
# (no energy, so no Laplace weights, but no second derivatives in training either)
const KIND = get(ENV, "KIND", "energy")
const WIDTH = 1024
mlp() = Chain(Dense(N + 2NBLOCKS * length(FREQ) => WIDTH, swish), Dense(WIDTH => WIDTH, swish),
              Dense(WIDTH => WIDTH, swish), Dense(WIDTH => KIND == "eps" ? N : 1))
network() = KIND == "conv" ? conv_energy() : mlp()

# KIND=conv: a convolutional energy. The image goes through three 3×3 convolutions (28 → 14 → 7
# pixels, 32 → 64 → 64 channels); the time features and the label are the conditioning vector,
# added as a per-channel bias after every convolution and passed to the dense head beside the
# image features. The head returns the scalar energy.
const NCOND = 2NBLOCKS * length(FREQ) + NY
function conv_energy(; c1 = 32, c2 = 64, head = 256)
    return @compact(conv1 = Conv((3, 3), 1 => c1; pad = 1), conv2 = Conv((3, 3), c1 => c2; pad = 1, stride = 2),
                    conv3 = Conv((3, 3), c2 => c2; pad = 1, stride = 2),
                    film1 = Dense(NCOND => c1), film2 = Dense(NCOND => c2), film3 = Dense(NCOND => c2),
                    head1 = Dense(7 * 7 * c2 + NCOND => head, swish), head2 = Dense(head => 1)) do inp
        B = size(inp, 2)
        x = reshape(inp[IMAGE, :], 28, 28, 1, B)
        c = vcat(inp[(N + 1):end, :], inp[LABEL, :])           # time features and label
        h = swish.(conv1(x) .+ reshape(film1(c), 1, 1, :, B))
        h = swish.(conv2(h) .+ reshape(film2(c), 1, 1, :, B))
        h = swish.(conv3(h) .+ reshape(film3(c), 1, 1, :, B))
        @return head2(head1(vcat(reshape(h, :, B), c)))
    end
end

const BLOCKSPEC = BLOCKS ? [IMAGE, LABEL] : nothing
predictor(net; device = (to = GDEV, back = CDEV), sampling = false) = KIND == "eps" ?
    NoisePredictor(net, sampling ? SAMPLE_SCHED : SCHED; input = time_input, ad = AutoZygote(), device, blocks = BLOCKSPEC) :
    NoisePredictor(EnergyNetwork(net), sampling ? SAMPLE_SCHED : SCHED; input = time_input, ad = AD, device, blocks = BLOCKSPEC)

# with blocks, the levels of a query: its fully clamped blocks held clean (HELD=1, default), or
# every block annealed at the nodes' level (HELD=0, the joint field)
query_levels(pred, ρ) = BLOCKS && get(ENV, "HELD", "1") == "1" ? conditional_levels(pred, ρ; level = SCHED.tmin) : nothing

# the trained model; MODEL=<name> selects ~/.cache/lenticulum/mnist/<name>.jld2 (default "energy")
modelfile() = joinpath(OUT, get(ENV, "MODEL", "energy") * ".jld2")
load_model() = (d = JLD2.load(modelfile()); (d["ps"], d["st"]))

# queries solved together: the batched Jacobian block stacks |F| = 10 copies of every node input,
# 40000 columns at 1000 queries, which a convolutional network's activations do not fit in 10 GB
const CHUNK = parse(Int, get(ENV, "CHUNK", KIND == "conv" ? "100" : "1000"))

# the field's nodes for a query: `samples` antithetic draws at each level
query_nodes(levels; samples = 2, rng = Xoshiro(1)) = field_nodes(rng, N; levels, samples)
