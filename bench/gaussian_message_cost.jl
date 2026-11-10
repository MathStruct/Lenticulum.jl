# The cost of the two checks a GaussianFactor message runs since the review of 2026-11-09 (B1):
# `_partly_flat` (an `isproper` of the input) and `_rank_deficient` (an SVD of A), against the whole
# message. Recorded in lib/Lenticulum.jl/src/gaussian.md, "What the checks cost".
# Run:  julia --project=bench bench/gaussian_message_cost.jl
using Lenticulum, LenticulumCore, Mycelium, LinearAlgebra, Random
const L = Lenticulum
function tmin(f, n=20000); f(); best=Inf; for _ in 1:5; t=@elapsed (for _ in 1:n; f(); end); best=min(best,t/n); end; best*1e6; end
for d in (2, 6)
    rng = Xoshiro(1); M = randn(rng,d,d); Λ = M*M' + I; b = GaussianBelief(randn(rng,d), Λ); A = randn(rng,d,d)
    f = GaussianFactor(d => d; noise = 0.5)
    ps = (A = A, b = zeros(d))
    full(t, inp) = Mycelium.factor_message(f, t, nothing, inp, nothing, ps, NamedTuple())
    tf = tmin(() -> full(:y, (x = b,))); tb = tmin(() -> full(:x, (y = b,)))
    tp = tmin(() -> L._partly_flat(b)); tr = tmin(() -> L._rank_deficient(A))
    println("d=$d: forward message $(round(tf;digits=2)) μs, backward $(round(tb;digits=2)) μs; _partly_flat $(round(tp;digits=2)) μs, _rank_deficient $(round(tr;digits=2)) μs")
end
