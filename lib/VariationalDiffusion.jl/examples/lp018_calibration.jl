# Reproduce the calibration numbers in reddiff.md and RED-Diff as a Statistical Game.
# Run: julia --project=lib/VariationalDiffusion.jl lib/VariationalDiffusion.jl/examples/lp018_calibration.jl
using VariationalDiffusion, LenticulumCore, LuxCore, Random
s = VPSDE()
function average_shrinkage(s, v; nodes = 20000)
    # Midpoint quadrature, independently of calibrate_lambda's trapezoidal rule.
    sum(begin
        t = s.tmin + (1-s.tmin)*(i-0.5)/nodes
        b = 0.1t + 9.95t*t
        -expm1(-b) / (exp(-b)*v-expm1(-b))
    end for i in 1:nodes) / nodes
end
λ = calibrate_lambda(s)
println((calibrated_lambda = λ, old_unnormalised_lambda = λ/(1-s.tmin), terminal_weight = reddiff_weight(REDDiff(λ=1.), s, 1.)))
for v in (0.25, 0.5, 1., 2., 4.)
    a = average_shrinkage(s,v)
    println((variance = v, normalised_sI = v*a, old_sI = v*a*(1-s.tmin), relative_prior_precision = λ*v*a))
end
# Same six seeds and inference settings as the existing Gaussian posterior testset.
pred = NoisePredictor(GaussianMixtureEps(s, zeros(1,1); s = 1.), s)
ps, st = LuxCore.setup(Xoshiro(0), pred)
for r in (1., 2.)
    want = 2r*r/(1+r*r)
    answers = [begin
        cfg = REDDiff(λ=λ, steps=1500, samples=64, lr=0.1, adam=false, rng=MersenneTwister(seed))
        only(first(reddiff_solve(pred, cfg, [2.], x -> r*r .* (x .- 2.), (falses(1), [2.]), ps, st)))
    end for seed in 1:6]
    println((clamp=r, answers, maximum_relative_error=maximum(abs.(answers./want.-1)), mean_relative_error=abs(sum(answers)/length(answers)/want-1)))
end
