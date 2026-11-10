# The smoothing bias of the implicit learner on a ring, for noise-free nodes and for nodes with
# noise at one level (vault: Implicit Diffusion Learners §5). Nodes with noise average the
# logarithm of the noised density over their shifts, which smooths a second time:
#   noise-free   r* ≈ R − (s² + σ²/α²) / 2R
#   with noise   r* ≈ R − (s² + 2σ²/α²) / 2R
# Run: julia --project=lib/VariationalDiffusion.jl lib/VariationalDiffusion.jl/examples/ring_bias.jl
using VariationalDiffusion, LuxCore, Random
sched = VPSDE()
θ = range(0, 2π; length = 401)[1:400]
ring = NoisePredictor(GaussianMixtureEps(sched, vcat(cos.(θ)', sin.(θ)'); s = 0.05), sched)
ps, st = LuxCore.setup(Xoshiro(0), ring)
for t in (0.03, 0.06)
    s2 = sigma(sched, t)^2 / alpha(sched, t)^2
    m0 = ImplicitDiffusion(ring, noisefree_nodes(2, t))
    r0 = first(implicit_infer(m0, [0.0, 0.9], [Inf, 0.0], ps, st)).z[2]
    rs = Float64[]
    for seed in 1:5
        m1 = ImplicitDiffusion(ring, field_nodes(Xoshiro(seed), 2; levels = [t], samples = 2000))
        push!(rs, first(implicit_infer(m1, [0.0, 0.9], [Inf, 0.0], ps, st)).z[2])
    end
    println("t=$t  σ²/α²=$(round(s2; digits=5))  noise-free: $(round(r0; digits=5)) (note's formula $(round(1 - (0.0025 + s2)/2; digits=5)))   noisy nodes: $(round(sum(rs)/5; digits=5)) ± $(round(maximum(rs)-minimum(rs); digits=5))  (single smoothing $(round(1 - (0.0025 + s2)/2; digits=5)), double $(round(1 - (0.0025 + 2s2)/2; digits=5)))")
end
