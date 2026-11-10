#implementation

> The parameter gradient averaged over the samples of a `Mycelium.SampleBuffer`: Monte Carlo EM
> when the sampled variables are the discrete ones and the rest is integrated out exactly.

> Sources: code: `sample_gradient.jl`; the sampler, the buffer and the theory are in Mycelium's [[sampling]]; tests: `lib/Lenticulum.jl/test/switching_constraint.jl` ("the gradient: with exact posterior weights …")

`sample_gradient(buffer, sampler, ps, st)` pins each stored sample, reruns inference at the
current parameters, takes the graph gradient at the resulting beliefs, and returns the weighted
mean. With `through_inference = true` it differentiates the pinned free energy as a function of
the parameters instead, inference included.

It moved here from `sampling.jl` in November 2026 because it is the one function of that file
that needs the gradient.
