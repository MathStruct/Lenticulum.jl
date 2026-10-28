module ImplicitLayersSDIRKSensitivityExt
using ImplicitLayers: ImplicitLayers as IL
using DifferentiationInterface: DifferentiationInterface as DI
using OrdinaryDiffEqSDIRK: Kvaerno3
using SciMLSensitivity: InterpolatingAdjoint

# A measured accuracy failure on the supported dependency stack, rather than an
# assertion that all SDIRK adjoints are invalid. Primal evaluation stays available.
function IL._check_sciml_ad(ad, ::IL.SciMLIntegrator{A,K,S}) where {A<:Kvaerno3,K,S<:InterpolatingAdjoint}
    ad isa DI.ADTypes.AutoForwardDiff && return nothing # direct dual solve ignores sensealg
    throw(ArgumentError("Kvaerno3 with InterpolatingAdjoint fails closed-form reverse " *
        "sensitivities on the tested stack. Use GaussAdjoint, KenCarp4, or AutoForwardDiff. " *
        "See examples/lp002_sciml_adjoint_limit.jl for a reproducer."))
end
end
