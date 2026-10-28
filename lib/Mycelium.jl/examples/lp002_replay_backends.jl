# Run with --project=lib/ImplicitLayers.jl/test, which supplies native backends.
# Deliberately bypasses the public backend guard to investigate replay itself.
using Mycelium, LenticulumCore, ADTypes, DifferentiationInterface, ForwardDiff
using Enzyme, Mooncake

# CLI: ... lp002_replay_backends.jl mooncake [gaussian]
# The Dirac probe can succeed while Gaussian Bethe entropy still fails.

struct ReplayProbe <: AbstractLenticulumFactor end
LenticulumCore.channels(::ReplayProbe) = (LenticulumCore.Channel(:x,nothing),)
LenticulumCore.supported_polarities(::ReplayProbe) = (Polarity(x=Unobserved()),)
LenticulumCore.supports_polarity(::ReplayProbe,p::Polarity) = p == Polarity(x=Unobserved())
LenticulumCore.islearnable(::ReplayProbe) = true
Mycelium.factor_message(::ReplayProbe,target::Symbol,pol,inputs,prior,ps,st) = (DiracBelief(ps.a),st)
Mycelium.local_free_energy(::ReplayProbe,beliefs,ps,st) = (0.0,st)

function replay_probe(ad; schedule_kind=:sequential)
    b=GraphBuilder(); variable!(b,:x)
    factor!(b,:source,ReplayProbe()); factor!(b,:loss,LossFactor((:x,),x -> x*x/2))
    connect!(b,:source,:x,:x;direction=Emitting(),coupling=ExactCoupling())
    connect!(b,:loss,:x,:x;direction=Absorbing(),coupling=ExactCoupling())
    g=build(b); p=(source=(a=0.7,),loss=(;)); st=(source=(;),loss=(;))
    sched=schedule_kind === :flooding ? flooding_schedule(g) :
        schedule_kind === :residual ? ResidualSchedule(maxtasks=2) : forward_backward_schedule(g)
    damping=0.3
    ops=Mycelium._realised_tasks(g,sched,2,p,st;damping)
    tape,_,_=Mycelium._reference_tape(g,ops,p,st,nothing,damping)
    return LenticulumCore._parameter_gradient(ad,Mycelium._coupled_loss,p,p,g,ops,tape,st,ad,damping)
end

struct ReplayGaussianProbe <: AbstractLenticulumFactor end
struct ReplayQuadraticProbe <: AbstractLenticulumFactor end
LenticulumCore.channels(::Union{ReplayGaussianProbe,ReplayQuadraticProbe}) = channels(ReplayProbe())
LenticulumCore.supported_polarities(::ReplayGaussianProbe) = supported_polarities(ReplayProbe())
LenticulumCore.supported_polarities(::ReplayQuadraticProbe) = ()
LenticulumCore.supports_polarity(::ReplayGaussianProbe,p::Polarity) = p == Polarity(x=Unobserved())
LenticulumCore.supports_polarity(::ReplayQuadraticProbe,p::Polarity) = false
LenticulumCore.islearnable(::ReplayGaussianProbe) = true
LenticulumCore.islearnable(::ReplayQuadraticProbe) = false
Mycelium.uses_cavity(::ReplayQuadraticProbe) = true
Mycelium.factor_message(::ReplayGaussianProbe,t::Symbol,pol,inputs,prior,p,st) =
    (GaussianBelief([p.a],fill(p.q,1,1)),st)
Mycelium.local_free_energy(::ReplayGaussianProbe,bs,p,st) = (0.0,st)
function Mycelium.local_free_energy(::ReplayQuadraticProbe,bs,p,st)
    q=only(bs.x.Λ); h=only(bs.x.η)
    return ((h/q)^2+inv(q))/2,st
end
function replay_gaussian_probe(ad; schedule_kind=:sequential)
    b=GraphBuilder(); variable!(b,:x)
    factor!(b,:source,ReplayGaussianProbe()); factor!(b,:loss,ReplayQuadraticProbe())
    connect!(b,:source,:x,:x;direction=Emitting(),coupling=ExactCoupling())
    connect!(b,:loss,:x,:x;direction=Absorbing(),coupling=ExactCoupling())
    g=build(b); p=(source=(a=0.7,q=1.2),loss=(;)); st=(source=(;),loss=(;))
    sched=schedule_kind === :flooding ? flooding_schedule(g) :
        schedule_kind === :residual ? ResidualSchedule(maxtasks=2) : forward_backward_schedule(g)
    damping=0.3
    ops=Mycelium._realised_tasks(g,sched,2,p,st;damping)
    tape,_,_=Mycelium._reference_tape(g,ops,p,st,nothing,damping)
    # Objective = ((a/q)^2 + 1/q)/2 + log(2π*e/q)/2.
    return LenticulumCore._parameter_gradient(ad,Mycelium._coupled_loss,p,p,g,ops,tape,st,ad,damping)
end

function replay_backend_probe(backend; gaussian=false)
    @show VERSION Base.pkgversion(Enzyme) Base.pkgversion(Mooncake)
    ad=backend == "mooncake" ? AutoMooncake() : AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(backend == "forward" ? Enzyme.Forward : Enzyme.Reverse),
        function_annotation=Enzyme.Const)
    for schedule_kind in (:sequential,:flooding,:residual)
        try
            @show backend schedule_kind gaussian
            result=gaussian ? replay_gaussian_probe(ad;schedule_kind) : replay_probe(ad;schedule_kind)
            @show result # Dirac expected source.a == 0.7
        catch e
            @show backend schedule_kind typeof(e)
            println(first(sprint(showerror,e),2000))
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    replay_backend_probe(get(ARGS,1,"mooncake");gaussian=get(ARGS,2,"")=="gaussian")
end
