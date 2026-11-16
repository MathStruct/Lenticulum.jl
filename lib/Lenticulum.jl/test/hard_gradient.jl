# P4 (Feature Backlog), AD half: the coupled replay (`free_energy_gradient(...; schedule, ad)`,
# `ExactCoupling`) through a `HardConstraintFactor`, under ForwardDiff and Zygote.
#
# Oracle: on a tree the Bethe free energy is −log Z (`hard_free_energy.jl`), so its gradient by
# the parameters is that of the dense closed form, Z = N(c; Cμ, CΣCᵀ) for z ~ N(μ, Σ) and the
# relation Cz = c imposed exactly. The dense −log Z is differentiated by ForwardDiff directly (no
# message passing), and the replay's gradients are compared with it and with reference
# differences. The relation fixes some of u and v, or all of them, so hard messages reach the
# GaussianFactors, the variable entropies and the hard factor's own term, and the derivative by
# A₁ moves the hard factor's inputs.

using Test, Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, LuxCore, Random
using ADTypes, DifferentiationInterface, ForwardDiff
import Zygote

module HardGradientOracle
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra

spd(rng, n) = (M = randn(rng, n, n); Matrix(Symmetric(M * M' + 0.3I)))

# x ~ N(μ₀, Σ₀); u = A₁x + b₁ + ε₁; w = A₂u + b₂ + ε₂; v ~ N(μᵥ, Σᵥ); Cᵤu + Cᵥv = c exactly;
# every edge with an exact coupling
function model(rng, n, nu, nv, nw, m)
    gb = GraphBuilder()
    c = ExactCoupling()
    variable!(gb, :x, n); variable!(gb, :u, nu); variable!(gb, :v, nv); variable!(gb, :w, nw)
    μ0, Σ0, μv, Σv = randn(rng, n), spd(rng, n), randn(rng, nv), spd(rng, nv)
    Q1, Q2 = spd(rng, nu), spd(rng, nw)
    factor!(gb, :px, GaussianPrior(:x, μ0, Σ0)); connect!(gb, :px, :x, :x; coupling = c)
    factor!(gb, :pv, GaussianPrior(:x, μv, Σv)); connect!(gb, :pv, :x, :v; coupling = c)
    factor!(gb, :g1, GaussianFactor(n => nu; noise = Q1))
    connect!(gb, :g1, :x, :x; coupling = c); connect!(gb, :g1, :y, :u; coupling = c)
    factor!(gb, :g2, GaussianFactor(nu => nw; noise = Q2))
    connect!(gb, :g2, :x, :u; coupling = c); connect!(gb, :g2, :y, :w; coupling = c)
    factor!(gb, :rel, HardConstraintFactor((u = nu, v = nv), m))
    connect!(gb, :rel, :u, :u; coupling = c); connect!(gb, :rel, :v, :v; coupling = c)
    return validate(build(gb)), (; μ0, Σ0, μv, Σv, Q1, Q2)
end

# −log Z of the dense model as a function of (A₁, b₁, A₂, b₂), for any element type
function nlogZ(p, rel, A1, b1, A2, b2)
    n, nu, nv, nw = size(A1, 2), size(A1, 1), length(p.μv), size(A2, 1)
    # z = (x, u, v, w) = L ξ + o with independent ξ = (x, ε₁, v, ε₂) ~ N(μξ, S)
    Z(r, c) = zeros(r, c)
    L = [I(n) Z(n, nu) Z(n, nv) Z(n, nw)
         A1 I(nu) Z(nu, nv) Z(nu, nw)
         Z(nv, n) Z(nv, nu) I(nv) Z(nv, nw)
         A2 * A1 A2 Z(nw, nv) I(nw)]
    o = vcat(zeros(n), b1, zeros(nv), A2 * b1 + b2)
    S = cat(p.Σ0, p.Q1, p.Σv, p.Q2; dims = (1, 2))
    μ = L * vcat(p.μ0, zeros(nu), p.μv, zeros(nw)) + o
    C = hcat(zeros(size(rel.c, 1), n), rel.A.u, rel.A.v, zeros(size(rel.c, 1), nw))
    Sc = C * L * S * L' * C'
    Sc = (Sc + Sc') / 2
    d = rel.c - C * μ
    return (d' * (Sc \ d) + logdet(2π * Sc)) / 2
end

# its gradient by ForwardDiff, as a NamedTuple shaped like the parameters of g1 and g2
function dense_gradient(p, ps)
    θ = vcat(vec(ps.g1.A), ps.g1.b, vec(ps.g2.A), ps.g2.b)
    sz = (size(ps.g1.A), length(ps.g1.b), size(ps.g2.A), length(ps.g2.b))
    function f(θ)
        i = 0
        take(k) = (r = θ[(i + 1):(i + k)]; i += k; r)
        A1 = reshape(take(prod(sz[1])), sz[1]); b1 = take(sz[2])
        A2 = reshape(take(prod(sz[3])), sz[3]); b2 = take(sz[4])
        return nlogZ(p, ps.rel, A1, b1, A2, b2)
    end
    return f, θ
end
end # module

@testset "P4: the coupled replay through a hard relation, against d(−log Z)" begin
    HG = HardGradientOracle
    rng = Xoshiro(97)
    nfull = 0
    for trial in 1:12
        n, nu, nv, nw = rand(rng, 1:3), rand(rng, 1:3), rand(rng, 1:2), rand(rng, 1:2)
        m = rand(rng, 1:(nu + nv))
        nfull += m == nu + nv
        g, p = HG.model(rng, n, nu, nv, nw, m)
        ps = LuxCore.initialparameters(Xoshiro(trial), g); st = LuxCore.initialstates(Xoshiro(trial), g)
        f, θ = HG.dense_gradient(p, ps)
        dθ = ForwardDiff.gradient(f, θ)
        _, _, st, s = infer!(g, tree_schedule(g), ps, st)
        flat(G) = vcat(vec(G.g1.A), G.g1.b, vec(G.g2.A), G.g2.b)
        Gref = LenticulumCore.free_energy_gradient(s, g, ps, st; schedule = tree_schedule(g), nsweeps = 1)
        @test flat(Gref) ≈ dθ atol = 1e-5 rtol = 1e-5
        for ad in (AutoForwardDiff(), AutoZygote())
            G = LenticulumCore.free_energy_gradient(s, g, ps, st; ad, schedule = tree_schedule(g), nsweeps = 1)
            @test flat(G) ≈ dθ atol = 1e-8 rtol = 1e-7
        end
    end
    @test nfull > 0      # some trials fix u and v entirely
end
