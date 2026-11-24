# P29: how far the rank of a NonlinearFactor's difference Jacobian is from the exact one
# (lib/Lenticulum.jl/src/nonlinear.md, "The rank of a difference Jacobian").
#
#     julia --project=bench bench/p29_difference_rank.jl
#
# 20 000 random f(x) = g(W x + c), W of random rank, g one of tanh, affine, sin + quadratic, scales
# 10^±2. Prints (1) the error of the raw central-difference Jacobian against a rounding-only bound
# (the Frobenius norm of the per-entry bounds of `_difference_jacobian`), as quantiles, and (2) how the rank of `Lenticulum._jacobian` (cut at the
# difference error) compares with the rank of the exact Jacobian, with the eps-relative test the
# messages use; for the draws cut below the exact rank, the largest exact singular value that was
# cut, absolute and in units of the rounding bound.

# Smoke: LENTICULUM_SMOKE=true (bench/smoke.jl) runs it with 100 seeded random Jacobians.

@isdefined(provenance) || include(joinpath(@__DIR__, "provenance.jl"))
abspath(PROGRAM_FILE) == (@__FILE__) && provenance()

@isdefined(SMOKE) || const SMOKE = get(ENV, "LENTICULUM_SMOKE", "false") == "true"
using Lenticulum, LinearAlgebra, Random

rk(A) = (s = svdvals(A); isempty(s) || s[1] == 0 ? 0 : count(>(maximum(size(A)) * eps() * s[1]), s))
quantiles(v, ps) = (w = sort(v); [w[clamp(ceil(Int, p * length(w)), 1, length(w))] for p in ps])

function main(; n = (SMOKE ? 100 : 20_000), seed = 29)
    rng = Xoshiro(seed)
    ratios = Float64[]; same = 0; low = 0; high = 0; cut_abs = 0.0; cut_rel = 0.0
    for _ in 1:n
        nin, nout = rand(rng, 1:4), rand(rng, 1:4)
        r = rand(rng, 1:min(nin, nout))
        W = (r == min(nin, nout) ? randn(rng, nout, nin) : randn(rng, nout, r) * randn(rng, r, nin)) .* 10.0^rand(rng, -2:2)
        c = randn(rng, nout); x0 = randn(rng, nin) .* 10.0^rand(rng, -1:1)
        kind = rand(rng, 1:3)
        g = kind == 1 ? (x -> tanh.(W * x .+ c)) : kind == 2 ? (x -> 5 .* (W * x .+ c)) :
            (x -> sin.(W * x .+ c) .+ 0.1 .* (W * x .+ c) .^ 2)
        A = kind == 1 ? Diagonal(1 .- tanh.(W * x0 .+ c) .^ 2) * W : kind == 2 ? 5 .* W :
            Diagonal(cos.(W * x0 .+ c) .+ 0.2 .* (W * x0 .+ c)) * W
        f = NonlinearFactor(g, nin => nout; noise = 0.1)
        Jraw, R = Lenticulum._difference_jacobian(f, x0, 1)
        ρ = norm(R)                                  # the per-entry rounding bounds, Frobenius
        push!(ratios, opnorm(Jraw - A) / ρ)
        J = Lenticulum._jacobian(f, x0)
        re, rg = rk(A), rk(J)
        same += rg == re; high += rg > re
        if rg < re
            low += 1; σ = svdvals(A)[rg + 1]
            cut_abs = max(cut_abs, σ); cut_rel = max(cut_rel, σ / ρ)
        end
    end
    println("raw error / rounding bound, quantiles 50/90/99/99.9/100%: ",
            round.(quantiles(ratios, (0.5, 0.9, 0.99, 0.999, 1.0)); sigdigits = 3))
    println("cleaned rank vs exact: equal $same, below $low, above $high, of $n")
    println("largest exact singular value cut: $(round(cut_abs; sigdigits = 3)), $(round(cut_rel; sigdigits = 3)) rounding bounds")
end

main()
