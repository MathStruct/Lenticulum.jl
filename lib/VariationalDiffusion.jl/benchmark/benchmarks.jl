using BenchmarkTools
using ADTypes: AutoZygote
using DifferentiationInterface, Lux, VariationalDiffusion, Zygote
using Random: Xoshiro

include(joinpath(@__DIR__, "../../../benchmark/fixtures.jl"))

const SUITE::BenchmarkGroup = BenchmarkGroup()
diffusion = SUITE["VariationalDiffusion"] = BenchmarkGroup()
queries = diffusion["relation"] = BenchmarkGroup()

function time_input(x, t)
    phases = (2π .* t) .* [1.0, 2.0]
    features = vcat(sin.(phases), cos.(phases))
    return vcat(x, features .* ones(eltype(x), 1, size(x, 2)))
end

function relation_inputs()
    net = Chain(Dense(6 => 8, swish), Dense(8 => 1))
    ps, st = Lux.setup(Xoshiro(15), net)
    predictor = NoisePredictor(EnergyNetwork(net), VPSDE(); input=time_input, ad=AutoZygote())
    model = ImplicitDiffusion(predictor, noisefree_nodes(2); batched=true)
    return model, [0.6, 0.5], [Inf, 4.0], Lux.f64(ps), st
end

function relation_query(model, z, precision, ps, st)
    return implicit_infer(model, z, precision, ps, st)
end

function relation_gradient(model, z, precision, ps, st)
    loss(p) = sum(abs2, first(relation_query(model, z, precision, p, st)).z)
    return only(Zygote.gradient(loss, ps))
end

const relation = cached_fixture(relation_inputs)
queries["implicit_infer"] = @benchmarkable relation_query(m, z, precision, ps, st) setup=((m, z, precision, ps, st) = $relation()) evals=1
queries["implicit_infer_many"] = @benchmarkable implicit_infer_many(m, Z, precision, ps, st) setup=(
    (m, z, precision, ps, st) = $relation(); Z = z .+ [0.0 0.03 -0.02 0.01]
) evals=1
queries["zygote_gradient"] = @benchmarkable relation_gradient(m, z, precision, ps, st) setup=((m, z, precision, ps, st) = $relation()) evals=1
