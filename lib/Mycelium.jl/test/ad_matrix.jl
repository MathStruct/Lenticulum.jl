module MyceliumADMatrix

using Mycelium, LenticulumCore, LinearAlgebra, Random, LuxCore, Test
include("../../LenticulumCore.jl/test/support/ad_matrix.jl")
using .Q5ADMatrix
include("support/affine.jl")
include("support/switched_normal.jl")

const BROKEN = Dict{String,Tuple}(
    # Q5 baseline: Enzyme EnzymeNoDerivativeError
    # ejlstr$dpotri_64_$libblastrampoline.so.5
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/P36Affine" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeNoDerivativeError
    # ejlstr$dpotri_64_$libblastrampoline.so.5
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/ExposedFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Zygote BoundsError. attempt to access Tuple{} at index [0]
    # Q5 baseline: Enzyme EnzymeNoDerivativeError
    # ejlstr$dpotri_64_$libblastrampoline.so.5
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "local_free_energy/RelayFactor" => ("Zygote", "Enzyme", "Mooncake"),
    # Q5 baseline: Enzyme EnzymeNoDerivativeError
    # ejlstr$dpotri_64_$libblastrampoline.so.5
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _chol
    "factor_message/PriorFactor" => ("Enzyme", "Mooncake"),
    # Q5 baseline: Zygote ErrorException. Mutating arrays is not supported -- called copyto!(SubArray{Float64, 1, Vector{Float64}, Tup...
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "factor_message/ClusterFactor" => ("Zygote", "Enzyme"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "local_free_energy/ClusterFactor" => ("Enzyme",),
    # Q5 baseline: Zygote MethodError. no method matching iterate(::Nothing)
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "factor_message/OverlayFactor" => ("Zygote", "Enzyme"),
    # Q5 baseline: Zygote MethodError. no method matching iterate(::Nothing)
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "local_free_energy/OverlayFactor" => ("Zygote", "Enzyme"),
    # Q5 baseline: Enzyme EnzymeRuntimeActivityError. Detected potential need for runtime activity.
    "free_energy" => ("Enzyme",),
    # Q5 baseline: Zygote ErrorException. Mutating arrays is not supported -- called push!(Vector{String}, ...)
    # Q5 baseline: Enzyme MethodError. no method matching EnzymeCore.MixedDuplicated(::Mycelium._EmitPlan{(), Mycelium.DataFactor{V...
    # Q5 baseline: Mooncake MooncakeRuleCompilationError. Mooncake failed to differentiate the following method:
    # try/catch in _emit_plan
    "infer!" => ("Zygote", "Enzyme", "Mooncake"),
)
# Q5 baseline: Enzyme UndefVarError, `vaEltypes` not defined in `Enzyme`, when the row
# runs alone; a correct value after the rest of the suite has run (LP-089). Enzyme
# also changed captured point data before throwing; the helper gives each backend its
# own fixture.
const UNSTABLE = Dict{String,Tuple}(
    "local_free_energy/ExposedFactor" => ("Enzyme",),
    "local_free_energy/LossFactor" => ("Enzyme",),
)

# These rows each compile an inference, cluster or higher-level factor path.
is_slow_row(row) = row in ("infer!", "free_energy") ||
    any(suffix -> endswith(row, suffix), ("/ClusterFactor", "/OverlayFactor", "/ExposedFactor"))
function check_gradient(row, f, p; kwargs...)
    slow = is_slow_row(row) ? ("Zygote", "ForwardDiff", "Enzyme", "Mooncake") : ()
    return gradient_row(row, f, p; broken=get(BROKEN, row, ()), unstable=get(UNSTABLE, row, ()), slow, kwargs...)
end

@testset verbose=true "Mycelium AD backend matrix" begin
    nt = (;)
    p = [1.2, 0.3]
    parameters(v) = (a=v[1:1], b=v[2:2])
    f = P36Affine{true}(0.7)
    x, y = DiracBelief([0.8]), DiracBelief([1.8])
    pol = Polarity(x=Observed(), y=Unobserved())
    message(v) = sum(first(factor_message(f, :y, pol, (; x), nothing, parameters(v), nt)).η)
    check_gradient("factor_message/P36Affine", message, p)
    local_energy(v) = first(local_free_energy(f, (; x, y), parameters(v), nt))
    check_gradient("local_free_energy/P36Affine", local_energy, p)

    e = expose(f, :a => :weight)
    expol = Polarity(x=Observed(), y=Unobserved(), weight=Observed())
    msgs = (; x, y, weight=DiracBelief([1.1]))
    exp_params(v) = (b=[sum(v)],)
    exp_message(v) = sum(first(factor_message(e, :y, expol, msgs, nothing, exp_params(v), nt)).η)
    check_gradient("factor_message/ExposedFactor", exp_message, p)
    exp_energy(v) = first(local_free_energy(e, msgs, exp_params(v), nt))
    check_gradient("local_free_energy/ExposedFactor", exp_energy, p)

    # These factors are parameterless. Vary the continuous data or beliefs
    # passed to the same methods used by inference.
    relay = RelayFactor(:x, :y)
    relay_message(v) = sum(first(factor_message(relay, :y, pol, (x=DiracBelief(v),), nothing, nt, nt)).value)
    check_gradient("factor_message/RelayFactor", relay_message, p)
    relay_energy(v) = first(local_free_energy(relay, (x=Gaussian(v[1], exp(v[2])),), nt, nt))
    check_gradient("local_free_energy/RelayFactor", relay_energy, p)

    loss = LossFactor((:x, :y), (x, y) -> sum(abs2, x - y)/2)
    loss_energy(v) = first(local_free_energy(loss, (x=DiracBelief(v), y=DiracBelief([0.1, -0.2])), nt, nt))
    check_gradient("local_free_energy/LossFactor", loss_energy, p)

    prior = PriorFactor(:x, Gaussian(0.0, 1.0), v -> sum(abs2, v)/2)
    prior_energy(v) = first(local_free_energy(prior, (x=DiracBelief([sum(v)]),), nt, nt))
    check_gradient("local_free_energy/PriorFactor", prior_energy, p)

    prior_message(v) = sum(first(factor_message(
        PriorFactor(:x, Gaussian(v, [1.0 0.0; 0.0 1.0]), prior.nlogp),
        :x, nothing, nt, nothing, nt, nt)).η)
    check_gradient("factor_message/PriorFactor", prior_message, p)

    # A data factor's observation and an optimiser's state are their active inputs.
    data_message(v) = sum(first(factor_message(DataFactor(:x, v), :x, nothing, nt, nothing, nt, nt)).value)
    check_gradient("factor_message/DataFactor", data_message, p)
    optimiser = OptimiserFactor(:x, Nesterov(0.1, 0.9))
    optimiser_message(v) = first(factor_message(optimiser, :x, nothing, nt, nothing, nt,
                                                 (p=v[1], state=v[2]))).value
    check_gradient("factor_message/OptimiserFactor", optimiser_message, p)

    table = TableFactor((x=2, y=2), [0.1 -0.5; 0.2 0.6])
    table_message(v) = probabilities(first(factor_message(
        table, :y, pol, (x=CategoricalBelief(exp.(v)),), nothing, nt, nt)))[1]
    check_gradient("factor_message/TableFactor", table_message, p)

    switched = P36Switched()
    overlay = OverlayFactor(:factor => switched, :prior => P36Fixed([0.2], fill(1.3,1,1)))
    c = as_cluster(switched)
    sw = Polarity(y=Observed(), k=Unobserved())
    inputs = (y=DiracBelief([0.6]),)
    switched_ps(v) = (factor=(mu=[-0.7, 1.1], q=[0.4, 0.8], logw=v),)
    for (factor, ps) in ((c, switched_ps), (overlay, v -> merge(switched_ps(v), (prior=nt,))))
        name = nameof(typeof(factor))
        switched_message(v) = probabilities(first(factor_message(factor, :k, sw, inputs, nothing, ps(v), nt)))[1]
        check_gradient("factor_message/$name", switched_message, p)
        switched_energy(v) = scalarise(scalarisation(factor), first(local_free_energy(factor, inputs, ps(v), nt)))
        check_gradient("local_free_energy/$name", switched_energy, p)
    end

    # LossFactor only absorbs. The data and optimiser factors contribute zero
    # local free energy, and TableFactor has no standalone local energy.
    skip_row("factor_message/LossFactor")
    skip_row("local_free_energy/DataFactor")
    skip_row("local_free_energy/OptimiserFactor")
    skip_row("local_free_energy/TableFactor")

    # Evaluate graph energy at a fixed store. For infer!, leave the output
    # unclamped so the resulting belief depends on both coordinates of ps.
    function graph(observed)
        b = GraphBuilder()
        variable!(b, :x, 1); variable!(b, :y, 1)
        factor!(b, :input, DataFactor(:x, x.value))
        factor!(b, :model, f)
        connect!(b, :input, :x, :x; direction=Emitting())
        connect!(b, :model, :x, :x); connect!(b, :model, :y, :y)
        if observed
            factor!(b, :output, DataFactor(:y, y.value))
            connect!(b, :output, :y, :y; direction=Emitting())
        end
        return build(b)
    end
    graph_params(v) = (model=parameters(v),)
    g = graph(true)
    _, _, st, store = infer!(g, tree_schedule(g), graph_params(p), nt)
    graph_loss(v) = first(scalar_free_energy(store, g, graph_params(v), st))
    check_gradient("free_energy", graph_loss, p)

    infer_graph = graph(false)
    schedule = tree_schedule(infer_graph)
    function belief_loss(v)
        beliefs, _, _, _ = infer!(infer_graph, schedule, graph_params(v), nt)
        return sum(belief_mean(beliefs.y))
    end
    check_gradient("infer!", belief_loss, p)
end

end
