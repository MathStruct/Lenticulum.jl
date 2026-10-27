module ImplicitLayersEnzymeCoreActivityExt
using ImplicitLayers: ImplicitLayers as IL
using LenticulumCore: LenticulumCore as LC
using DifferentiationInterface: DifferentiationInterface as DI
using EnzymeCore: EnzymeCore as EC

# Validate the ADTypes mode outside differentiation; leave _flow to Enzyme.
function LC._check_factor_ad(ad::DI.ADTypes.AutoEnzyme, f::IL.NeuralODEFactor)
    f.integrator isa IL.SciMLIntegrator && throw(ArgumentError(
        "SciML NeuralODE sensitivities are tested only with AutoForwardDiff, AutoZygote and AutoMooncake; AutoEnzyme is unsupported"))
    f.integrator isa Union{IL.EulerIntegrator,IL.RK4Integrator} || return nothing
    ad.mode !== nothing && EC.runtime_activity(ad.mode) && return nothing
    throw(ArgumentError(
        "Enzyme differentiation of built-in NeuralODE flows requires runtime activity. " *
        "Pass AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)) " *
        "or AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))."))
end

end
