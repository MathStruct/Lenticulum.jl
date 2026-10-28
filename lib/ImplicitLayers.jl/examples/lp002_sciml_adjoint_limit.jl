# Run with --project=lib/ImplicitLayers.jl/test. Deliberately bypass the factor
# validation by differentiating an arbitrary outer callable, to expose SciML's
# measured Kvaerno3/InterpolatingAdjoint error on the installed dependency stack.
using ImplicitLayers,LenticulumCore,LuxCore,DifferentiationInterface,ADTypes,Zygote
using OrdinaryDiffEqCore,OrdinaryDiffEqSDIRK,SciMLSensitivity,LinearAlgebra
struct AffineAdjointProbe <: LuxCore.AbstractLuxLayer end
(::AffineAdjointProbe)(inp,p,s)=(p.a .* inp[1] .+ s.k*p.c,(k=s.k+1,))
p,s,T=(a=-0.35,c=0.12),(k=2,),0.7
for (alg,sa) in ((Kvaerno3(),InterpolatingAdjoint(;autojacvec=ZygoteVJP())),
                 (Kvaerno3(),GaussAdjoint(;autojacvec=ZygoteVJP())),
                 (KenCarp4(),InterpolatingAdjoint(;autojacvec=ZygoteVJP()))), reverse in (false,true)
    h=reverse ? -T : T
    reference=[h*1.2exp(p.a*h)+s.k*p.c*(h*p.a*exp(p.a*h)-expm1(p.a*h))/p.a^2,
               s.k*expm1(p.a*h)/p.a]
    f=NeuralODEFactor(AffineAdjointProbe(),1;tspan=(0.,T),state_policy=:fixed,
        integrator=SciMLIntegrator(alg;sensealg=sa,abstol=1e-12,reltol=1e-11))
    g=parameter_gradient(p -> only(first(reverse ? flow_reverse(f,[1.2],p,s) : flow_forward(f,[1.2],p,s))),p;ad=AutoZygote())
    println((algorithm=nameof(typeof(alg)),sensealg=nameof(typeof(sa)),reverse,gradient=g,
             closed_form=reference,error=norm([g.a,g.c]-reference)))
end
