# Run with --project=lib/ImplicitLayers.jl/test. This probes the original
# finite-difference algorithms directly; it adds no derivative rule.
using ImplicitLayers, LenticulumCore, LinearAlgebra
using ADTypes, DifferentiationInterface, ForwardDiff, Zygote

struct LocalisationProbe <: AbstractLenticulumFactor end
LenticulumCore.energy(::LocalisationProbe,x,a,y,p,st) = ([p.a*y[1]^2-p.b],st)
metric_probe(p) = first(scalar_energy(LocalisationProbe(),nothing,nothing,[1.3],p,(;);σ=SampsonNorm()))
metric_exact(p) = (p.a*1.3^2-p.b)^2/(8*p.a^2*1.3^2)
divergence_probe(p) = ImplicitLayers._divergence((u,t) -> p.a .* u.^2,[0.7,-0.2],0.0)
divergence_exact(p) = 2*p.a*(0.7-0.2)
logdet_probe(p) = last(integrate_with_divergence((u,t) -> p.a .* u,[0.7,-0.2],0.0,0.6,RK4Integrator(4)))
logdet_exact(p) = 2*0.6*p.a

function metric_divergence_probe()
    @show VERSION Base.pkgversion(ForwardDiff) Base.pkgversion(Zygote)
    p=(a=0.7,b=0.2)
    for (name,fun,exact) in ((:localisation,metric_probe,metric_exact),
            (:divergence,divergence_probe,divergence_exact),(:logdet,logdet_probe,logdet_exact))
        reference=parameter_gradient(exact,p;ad=AutoForwardDiff())
        @show name fun(p) exact(p) reference
        for ad in (nothing,AutoForwardDiff(),AutoZygote())
            try
                result=parameter_gradient(fun,p;ad)
                @show ad result norm(LenticulumCore._flatten(result)-LenticulumCore._flatten(reference))
            catch e
                @show ad typeof(e)
                println(first(sprint(showerror,e),300))
            end
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    metric_divergence_probe()
end
