using BenchmarkTools
using Lenticulum: GaussianFactor, HardConstraintFactor, NonlinearFactor
using LenticulumCore: Gaussian, Observed, Polarity, TrivialBelief, Unobserved
using Mycelium: factor_message
using MyceliumLearning: fit_em

include(joinpath(@__DIR__, "../../../benchmark/fixtures.jl"))
include(joinpath(@__DIR__, "../../../benchmark/regime.jl"))

const SUITE::BenchmarkGroup = BenchmarkGroup()
lenticulum = SUITE["Lenticulum"] = BenchmarkGroup()
messages = lenticulum["message"] = BenchmarkGroup()

forward_polarity() = Polarity(; x = Observed(), y = Unobserved())
cavity() = (x = Gaussian([0.3, -0.4], [0.5 0.1; 0.1 0.4]),)

messages["gaussian"] = @benchmarkable factor_message(f, :y, pol, inputs, TrivialBelief(), ps, (;)) setup=(
    f = GaussianFactor(2 => 2; noise=0.25); pol = forward_polarity(); inputs = cavity();
    ps = (A = [1.0 0.5; 0.0 1.0], b = [0.2, -0.1])
)
messages["hard_constraint"] = @benchmarkable factor_message(f, :y, nothing, inputs, nothing, ps, (;)) setup=(
    f = HardConstraintFactor((x = 2, y = 2), 2); inputs = cavity();
    ps = (A = (x = [1.0 0.5; 0.0 1.0], y = [-1.0 0.0; 0.0 -1.0]), c = [0.2, -0.1])
)
messages["nonlinear"] = @benchmarkable factor_message(f, :y, pol, inputs, TrivialBelief(), (;), (;)) setup=(
    f = NonlinearFactor(x -> [x[1]^2 + x[2], sin(x[2])], 2 => 2; noise=0.25);
    pol = forward_polarity(); inputs = cavity()
)

fit_regime(g, sched, ps, st) = fit_em(g, ps, st; schedule=sched, iters=2, tol=1e-10)

const regime = cached_fixture(regime_plate)
lenticulum["regime"] = BenchmarkGroup()
lenticulum["regime"]["plate_fit_em"] = @benchmarkable fit_regime(g, sched, ps, st) setup=((g, sched, ps, st) = $regime()) evals=1
