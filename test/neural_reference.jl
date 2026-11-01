using Test
isdefined(@__MODULE__, :NeuralReference) ||
    include(joinpath(@__DIR__, "..", "examples", "neural_reference.jl"))

@testset "Neural reverse kernel as a reference for E4(b) (examples/neural_reference.jl)" begin
    using Random
    N = NeuralReference; B = N.B; E = N.E
    # hand-written backpropagation against central finite differences
    rng = Xoshiro(3); net = N.MLP(8, rng); net.W3 .= randn(rng, size(net.W3)); net.b1 .= randn(rng, 8)
    X = randn(rng, 2 + 2N.NFREQ, 5); Y = randn(rng, 2, 5)
    g, _ = N.gradients(net, X, Y)
    f() = sum(abs2, first(N.forward(net, X)) .- Y) / 5
    err = 0.0
    for (p, gp) in zip(N.params(net), g), i in eachindex(p)
        old = p[i]; p[i] = old + 1e-6; fp = f(); p[i] = old - 1e-6; fm = f(); p[i] = old
        err = max(err, abs((fp - fm) / 2e-6 - gp[i]))
    end
    @test err < 1e-6
    # σ_t² is the bound's optimum given the mean, on the pairs it was computed from
    Z0 = B.sample_data(:moons, 500, Xoshiro(1)); sch = E.schedule(5, 1 - 0.8^4)
    net, _ = N.train(Z0, sch, 16; iters = 200, batch = 64)
    ks = N.kernels(net, sch, Z0)
    P = B.SS.step_pairs(sch, 3, Z0, Xoshiro(2 + 300))
    ll(s2) = (k = N.NeuralKernel(net, 3, 5, sch.βs[3], sch.α[4], s2);
              sum(E.kernel_logpdf(k, P[i, 1:2], P[i, 3:4]) for i in axes(P, 1)))
    @test ll(ks[3].σ2) > ll(0.9ks[3].σ2) && ll(ks[3].σ2) > ll(1.1ks[3].σ2)
    # the chain runs in E4(b)'s scoring
    r = N.run(:moons; ntrain = 500, ntest = 50, ngen = 100, settings = [(5, 16)], S = 4, iters = 200)
    @test all(isfinite, (r.rows[1].elbo, r.rows[1].iw, r.rows[1].ed))
end
