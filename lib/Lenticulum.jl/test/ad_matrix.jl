module LenticulumADMatrix

using Lenticulum, LenticulumCore, Mycelium, MyceliumLearning, LinearAlgebra, Test
include("../../LenticulumCore.jl/test/support/ad_matrix.jl")
using .Q5ADMatrix

const EMPTY = (;)
const UNIT = [1.0 0.0; 0.0 1.0]

function energy_loss(f, x, y, ps)
    value, _ = energy(f, x, nothing, y, ps, EMPTY)
    return value isa AbstractVector ? sum(abs2, value) :
           value isa GradedEnergy ? scalarise(scalarisation(f), value) : value
end

message(f, target, polarity, inputs, ps; prior=TrivialBelief()) =
    first(factor_message(f, target, polarity, inputs, prior, ps, EMPTY))

function local_energy(f, inputs, ps)
    value, _ = local_free_energy(f, inputs, ps, EMPTY)
    return scalarise(scalarisation(f), value)
end

const BROKEN = Dict{String,Tuple}(
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/GaussianFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _improper_check
    "local_free_energy/GaussianFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeNoDerivativeError
    # ejlstr$dpotri_64_$libblastrampoline.so.5
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/GaussianPrior" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "local_free_energy/GaussianPrior" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Zygote ErrorException. Mutating arrays is not supported -- called setindex!(Matrix{Float64}, ...)
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/NonlinearFactor" => ("Zygote", "Enzyme", "Mooncake"),
    # Q5 baseline: Zygote ErrorException. Mutating arrays is not supported -- called setindex!(Matrix{Float64}, ...)
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "local_free_energy/NonlinearFactor" => ("Zygote", "Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "energy/LinearConstraintFactor" => ("Enzyme",),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/LinearConstraintFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "local_free_energy/LinearConstraintFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    # Q5 baseline: Mooncake MissingForeigncallRuleError. No rrule!! available for foreigncall with primal argument types
    # with primal argument types
    "factor_message/HardConstraintFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    # Q5 baseline: Mooncake MissingForeigncallRuleError. No rrule!! available for foreigncall with primal argument types
    # with primal argument types
    "local_free_energy/HardConstraintFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Zygote ErrorException. Mutating arrays is not supported -- called setindex!(Matrix{Float64}, ...)
    # Q5 baseline: Enzyme AssertionError. Number of non-ghost elements of julia type Lenticulum.var"#_split_on_whole##2#_split_on_whol...
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/SplitFactor" => ("Zygote", "Enzyme", "Mooncake"),
    # Q5 baseline: Zygote ErrorException. Mutating arrays is not supported -- called copyto!(Vector{Int64}, ...)
    # Q5 baseline: Enzyme AssertionError. Number of non-ghost elements of julia type Lenticulum.var"#_split_on_whole##2#_split_on_whol...
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "local_free_energy/SplitFactor" => ("Zygote", "Enzyme", "Mooncake"),
    # Q5 baseline: Mooncake MissingForeigncallRuleError. No rrule!! available for foreigncall with primal argument types
    # with primal argument types
    "factor_message/ScaleShiftFactor" => ("Mooncake",),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MissingForeigncallRuleError. No rrule!! available for foreigncall with primal argument types
    # with primal argument types
    "local_free_energy/ScaleShiftFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "energy/SwitchingConstraintFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/SwitchingConstraintFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "local_free_energy/SwitchingConstraintFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Zygote ErrorException. Mutating arrays is not supported -- called copyto!(SubArray{Float64, 1, Vector{Float64}, Tup...
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "factor_message/RegimeFactor" => ("Zygote", "Enzyme"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "local_free_energy/RegimeFactor" => ("Enzyme",),
    # Q5 baseline: Zygote MethodError. no method matching rrule(::typeof(maximum), ::Vector{Float64}; init::Bool)
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/LikelihoodFactor" => ("Zygote", "Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme IllegalTypeAnalysisException. Enzyme compilation failed due to illegal type analysis.
    "local_free_energy/LikelihoodFactor" => ("Enzyme",),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _emit_plan
    "free_energy after infer!/Kalman" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Zygote MethodError. no method matching rrule(::typeof(sum), ::Vector{Int64}; init::Int64)
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    # Q5 baseline: Mooncake MissingForeigncallRuleError. No rrule!! available for foreigncall with primal argument types
    # with primal argument types
    "moment_statistics/regime" => ("Zygote", "Enzyme", "Mooncake"),
)
# The first default matrix took 4m52s. Keep pointwise energy and the Gaussian
# transition message in the default tier, and compile the other paths in the slow tier.
is_slow_row(row) = !startswith(row, "energy/") && row != "factor_message/GaussianFactor"
function check_gradient(row, f, p; kwargs...)
    slow = is_slow_row(row) ? ("Zygote", "ForwardDiff", "Enzyme", "Mooncake") : ()
    # Q5 baseline: Enzyme alone did not finish compiling ScaleShiftFactor's
    # message within the isolated 185-second limit.
    skipped = row == "factor_message/ScaleShiftFactor" ? ("Enzyme",) : ()
    return gradient_row(row, f, p; broken=get(BROKEN, row, ()), slow, skipped, kwargs...)
end

@testset verbose=true "Lenticulum AD backend matrix" begin
    p = [0.2, 0.4]
    x, y = [0.8, -0.2], [1.8, 0.3]
    xy = Polarity(x=Observed(), y=Unobserved())
    points = (x=DiracBelief(x), y=DiracBelief(y))
    beliefs = (x=Gaussian(x, 0.7UNIT), y=Gaussian(y, 1.3UNIT))

    f = GaussianFactor(2 => 2; noise=0.7)
    gaussian_ps(v) = (A=[0.8 0.1; -0.3 0.6], b=v)
    check_gradient("energy/GaussianFactor", v -> energy_loss(f, x, y, gaussian_ps(v)), p)
    check_gradient("factor_message/GaussianFactor", v -> sum(message(f, :y, xy, (x=points.x,), gaussian_ps(v)).η), p)
    check_gradient("local_free_energy/GaussianFactor", v -> local_energy(f, beliefs, gaussian_ps(v)), p)

    gp = GaussianPrior(:x, x, 0.7UNIT; learn_noise=:diagonal)
    prior_ps(v) = (noise=(log_precision=v,),)
    check_gradient("energy/GaussianPrior", v -> energy_loss(gp, nothing, y, prior_ps(v)), p)
    check_gradient("factor_message/GaussianPrior", v -> sum(message(gp, :x, Polarity(x=Unobserved()), EMPTY, prior_ps(v)).η), p)
    check_gradient("local_free_energy/GaussianPrior", v -> local_energy(gp, (x=beliefs.y,), prior_ps(v)), p)

    # NonlinearFactor has an empty Lux parameter tree. Vary the input of its fixed map.
    nl = NonlinearFactor(v -> v .+ v.^2 / 4, 2 => 2; noise=0.7)
    check_gradient("energy/NonlinearFactor", v -> energy_loss(nl, v, y, EMPTY), p)
    check_gradient("factor_message/NonlinearFactor", v -> sum(message(nl, :y, xy, (x=DiracBelief(v),), EMPTY).η), p)
    check_gradient("local_free_energy/NonlinearFactor", v -> local_energy(nl, (x=DiracBelief(v), y=points.y), EMPTY), p)

    linear = LinearConstraintFactor((x=2, y=2), 2; noise=0.7)
    constraint_ps(v) = (A=(x=-0.8UNIT, y=UNIT), c=v)
    check_gradient("energy/LinearConstraintFactor", v -> energy_loss(linear, (; x, y), nothing, constraint_ps(v)), p)
    check_gradient("factor_message/LinearConstraintFactor", v -> sum(message(linear, :y, xy, (x=points.x,), constraint_ps(v)).η), p)
    check_gradient("local_free_energy/LinearConstraintFactor", v -> local_energy(linear, beliefs, constraint_ps(v)), p)

    hard = HardConstraintFactor((x=2, y=2), 2)
    hard_ps(v) = (A=(x=-Diagonal(v), y=UNIT), c=[0.2, 0.4])
    check_gradient("factor_message/HardConstraintFactor", v -> sum(message(hard, :y, xy, (x=beliefs.x,), hard_ps(v)).μ), p)
    check_gradient("local_free_energy/HardConstraintFactor", v -> local_energy(hard, beliefs, hard_ps(v)), p)

    # SplitFactor has discrete wiring and no Lux parameters. Perturb incoming
    # covariance and mean so both the message and entropy can change.
    split_factor = SplitFactor(:whole, (left=1:1, right=2:2))
    split_belief(v) = Gaussian(v, Diagonal(exp.(v)))
    check_gradient("factor_message/SplitFactor", v -> sum(message(split_factor, :left, Polarity(whole=Observed(), left=Unobserved(), right=Latent()), (whole=split_belief(v),), EMPTY).μ), p)
    check_gradient("local_free_energy/SplitFactor", v -> local_energy(split_factor, (whole=split_belief(v),), EMPTY), p)

    scale_shift = ScaleShiftFactor(:x, :z, 2)
    scale_ps(v) = (b=[0.1, -0.2], logs=v)
    check_gradient("factor_message/ScaleShiftFactor", v -> sum(message(scale_shift, :z, Polarity(x=Observed(), z=Unobserved()), (x=beliefs.x,), scale_ps(v)).μ), p)
    check_gradient("local_free_energy/ScaleShiftFactor", v -> local_energy(scale_shift, (x=beliefs.x, z=beliefs.y), scale_ps(v)), p)

    switch = SwitchingConstraintFactor((x=2, y=2), 2, 2; noise=0.7, output=:y)
    switch_ps(v) = (A=(x=-0.8UNIT,), b=[0.1 0.4; -0.2 0.6], logw=v)
    prob(b) = probabilities(b)[1]
    check_gradient("energy/SwitchingConstraintFactor", v -> energy_loss(switch, (; x, y, k=1), nothing, switch_ps(v)), p)
    check_gradient("factor_message/SwitchingConstraintFactor", v -> prob(message(switch, :k, Polarity(x=Observed(), y=Observed(), k=Unobserved()), points, switch_ps(v))), p)
    check_gradient("local_free_energy/SwitchingConstraintFactor", v -> local_energy(switch, points, switch_ps(v)), p)

    regime = RegimeFactor((y=2, x=2), 2, 2; noise=0.7, output=:y)
    regime_ps(v) = (A=(x=cat(-0.8UNIT, 0.4UNIT; dims=3),), b=[0.1 0.4; -0.2 0.6], logw=v)
    check_gradient("factor_message/RegimeFactor", v -> prob(message(regime, :k, Polarity(y=Observed(), x=Observed(), k=Unobserved()), points, regime_ps(v))), p)
    check_gradient("local_free_energy/RegimeFactor", v -> local_energy(regime, points, regime_ps(v)), p)

    # LikelihoodFactor also has an empty Lux parameter tree. Vary the predictor
    # input and use a fixed expansion point for the Laplace message.
    lik = LikelihoodFactor(PoissonLikelihood(2), 2; x0=x)
    counts = [2.0, 3.0]
    observed = DiracBelief(counts)
    check_gradient("energy/LikelihoodFactor", v -> energy_loss(lik, v, counts, EMPTY), p)
    check_gradient("factor_message/LikelihoodFactor", v -> sum(message(lik, :x, Polarity(x=Unobserved(), y=Observed()), (y=observed,), EMPTY; prior=Gaussian(v, UNIT)).η), p)
    check_gradient("local_free_energy/LikelihoodFactor", v -> local_energy(lik, (x=DiracBelief(v), y=observed), EMPTY), p)

    # These factors implement the graph free-energy interface, with no
    # pointwise energy method.
    for T in (HardConstraintFactor, SplitFactor, ScaleShiftFactor, RegimeFactor)
        skip_row("energy/$(nameof(T))")
    end
end

slow_tier() && @testset "Lenticulum path A AD matrix" begin
    # Path A. One two-dimensional Kalman prediction and observation.
    b = GraphBuilder()
    for v in (:state0, :state1, :observed); variable!(b, v, 2); end
    factor!(b, :prior, GaussianPrior(:x, [0.7, -0.2], UNIT))
    factor!(b, :transition, GaussianFactor(2 => 2; noise=0.5))
    factor!(b, :sensor, GaussianFactor(2 => 2; noise=0.3))
    factor!(b, :data, DataFactor(:y, [0.4, 0.8]))
    connect!(b, :prior, :x, :state0)
    connect!(b, :transition, :x, :state0); connect!(b, :transition, :y, :state1)
    connect!(b, :sensor, :x, :state1); connect!(b, :sensor, :y, :observed)
    connect!(b, :data, :y, :observed; direction=Emitting())
    g = build(b)
    schedule = tree_schedule(g)
    function kalman_loss(v)
        ps = (transition=(A=[0.9 0.1; -0.2 0.8], b=v), sensor=(A=UNIT, b=zeros(2)))
        _, _, st, store = infer!(g, schedule, ps, EMPTY)
        return first(scalar_free_energy(store, g, ps, st))
    end
    check_gradient("free_energy after infer!/Kalman", kalman_loss, [0.1, 0.2]; check_fixture=false)
end

slow_tier() && @testset "Lenticulum path C AD matrix" begin
    # Path C. Two regression regimes and a latent categorical label.
    b = GraphBuilder()
    variable!(b, :x, 2); variable!(b, :y, 2); variable!(b, :k, 2)
    factor!(b, :input, DataFactor(:x, [0.8, -0.2]))
    factor!(b, :output, DataFactor(:y, [1.8, 0.3]))
    factor!(b, :regime, RegimeFactor((y=2, x=2), 2, 2; noise=0.7, output=:y))
    connect!(b, :input, :x, :x; direction=Emitting())
    connect!(b, :output, :y, :y; direction=Emitting())
    for v in (:x, :y, :k); connect!(b, :regime, v, v); end
    g = build(b)
    p = [0.2, 0.4]
    graph_ps(v) = (regime=(A=(x=cat(-0.8UNIT, 0.4UNIT; dims=3),),
                           b=[0.1 0.4; -0.2 0.6], logw=v),)
    _, _, st, store = infer!(g, tree_schedule(g), graph_ps(p), EMPTY)
    function responsibility(v)
        stats = moment_statistics(store, g, graph_ps(v), st)
        return stats[(:regime, nothing)][2].n[1]
    end
    check_gradient("moment_statistics/regime", responsibility, p; check_fixture=false)
end

end
