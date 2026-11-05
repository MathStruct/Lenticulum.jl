# RQ-008: a switching factor with its outcome summed out is one energy on z, the linear constraint's
# quadratic minus a log-sum-exp of affine functions of y = Az; as the noise goes to zero (Q = sQ₀,
# s → 0), s·E tends to the quadratic minus a maximum of affine functions, a tropical polynomial.
# Oracle: the graph's exact free energy (the outcome has degree one, so the graph is a forest).
using Test, Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, LuxCore, Random

rq8_lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))

@testset "RQ-008: the summed-out switch is a soft-min, and tropical in the low-noise limit" begin
    A = [1.0 0.4; -0.3 1.0]
    B = [0.0 2.0 -1.5; 1.0 -1.0 2.5]                                   # three outcomes' offsets
    w = [0.5, 0.3, 0.2]
    Q0 = [0.3 0.05; 0.05 0.2]
    function F(z, s)
        Q = s .* Q0
        b = GraphBuilder()
        variable!(b, :z, 2); variable!(b, :k, 3)
        factor!(b, :d, DataFactor(:z, z)); connect!(b, :d, :z, :z; direction = Emitting())
        factor!(b, :sw, SwitchingConstraintFactor((z = 2,), 3, 2; noise = Q)); connect!(b, :sw, :z, :z); connect!(b, :sw, :k, :k)
        g = build(b)
        ps = (d = NamedTuple(), sw = (A = (z = A,), b = B, logw = log.(w)))
        _, _, _, st = infer!(g, tree_schedule(g), ps, NamedTuple())
        return first(scalar_free_energy(st, g, ps, NamedTuple()))
    end
    # E(z) = ½ yᵀQ⁻¹y − logsumexp_k(b_kᵀQ⁻¹y − ½ b_kᵀQ⁻¹b_k + log w_k) + ½ log det 2πQ, y = Az
    softmin(z, s) = (Q = s .* Q0; y = A * z;
        dot(y, Q \ y) / 2 - rq8_lse([dot(B[:, k], Q \ y) - dot(B[:, k], Q \ B[:, k]) / 2 + log(w[k]) for k in 1:3]) + logdet(2π * Q) / 2)
    tropical(z) = (y = A * z; dot(y, Q0 \ y) / 2 - maximum(dot(B[:, k], Q0 \ y) - dot(B[:, k], Q0 \ B[:, k]) / 2 for k in 1:3))
    rng = Xoshiro(8)
    zs = [randn(rng, 2) .* 2 for _ in 1:6]
    for z in zs
        @test F(z, 1.0) ≈ softmin(z, 1.0) rtol = 1e-12
        @test F(z, 0.05) ≈ softmin(z, 0.05) rtol = 1e-12
    end
    # s·E_s → the tropical form, with an error O(s log s): s·½ log det 2πsQ₀ dominates, beside s·log w and
    # the other outcomes' share. Measured: 0.035, 8.1e-4, 1.3e-5 at s = 1e-2, 1e-4, 1e-6, i.e. 0.76–0.92·s|log s|
    err(s) = maximum(abs(s * F(z, s) - tropical(z)) for z in zs)
    for s in (1e-2, 1e-4, 1e-6)
        @test err(s) <= s * abs(log(s))
    end
end
