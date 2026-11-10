# Run with --project=lib/ImplicitLayers.jl/test. Opt-in diagnostics: failures are
# caught and printed, never treated as evidence of a successful derivative.
using ImplicitLayers, LenticulumCore, LuxCore, DifferentiationInterface, ADTypes
using ForwardDiff, Enzyme, SciMLSensitivity, OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
struct EndpointProbe <: LuxCore.AbstractLuxLayer end
(::EndpointProbe)(inp,p,s) = (p.a .* inp[1],s)
endpoint(p,f) = sum(first(flow_forward(f,[1.2],(a=p[1],),(;))))
function sciml_probe(mode)
    f = NeuralODEFactor(EndpointProbe(),1;
        integrator=SciMLIntegrator(Tsit5();abstol=1e-12,reltol=1e-10))
    ad = AutoEnzyme(;mode,function_annotation=Enzyme.Const)
    # Direct DI bypasses the package's guard to expose the upstream failure.
    return DifferentiationInterface.gradient(endpoint,ad,[-0.35],Constant(f))
end
function sciml_split_probe()
    f=NeuralODEFactor(EndpointProbe(),1;integrator=SciMLIntegrator(Tsit5();abstol=1e-12,reltol=1e-10))
    ad=AutoEnzyme(;mode=Enzyme.set_runtime_activity(Enzyme.Reverse),function_annotation=Enzyme.Const)
    vector_endpoint(x,f)=first(flow_forward(f,x,(a=-0.35,),(;)))
    return DifferentiationInterface.jacobian(vector_endpoint,ad,[1.2],Constant(f))
end
struct ClosureCell <: LuxCore.AbstractLuxLayer end
(::ClosureCell)(inp,p,s) = (p.a .* inp[1] .+ inp[2],s)
function closure_probe(ad)
    f=DEQFactor(ClosureCell(),(x=1,z=1)); x=[1.2]; ps=(a=[0.2],); st=(;)
    θ=LenticulumCore._flatten(ps); off=LenticulumCore.offset_tree(ps)
    loss=t -> sum(first(solve_state(f,x,LenticulumCore._unflatten(t,off,ps),st;ad)))
    return DifferentiationInterface.gradient(loss,ad,θ)
end
for (label,probe) in (
    ("SciML runtime Forward",() -> sciml_probe(Enzyme.set_runtime_activity(Enzyme.Forward))),
    ("SciML runtime Reverse",() -> sciml_probe(Enzyme.set_runtime_activity(Enzyme.Reverse))),
    ("SciML vector-return split reverse",sciml_split_probe),
    ("automatic outer closure",() -> closure_probe(AutoEnzyme())),
    ("explicit constant closure",() -> closure_probe(AutoEnzyme(;function_annotation=Enzyme.Const))))
    println(label)
    try
        println(probe())
    catch e
        println(typeof(e))
        showerror(stdout,e)
        println()
    end
end
