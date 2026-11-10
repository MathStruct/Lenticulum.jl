module ImplicitLayersEnzymeCoreActivityExt
using ImplicitLayers: ImplicitLayers as IL
using LenticulumCore: LenticulumCore as LC
using DifferentiationInterface: DifferentiationInterface as DI
using EnzymeCore: EnzymeCore as EC

EC.EnzymeRules.inactive(::typeof(IL._endpoint_state), args...) = true

# Scalar losses use Enzyme's combined reverse mode. DI's vector-return split
# reverse fails to generate an ABI wrapper for this SciML path on the tested stack.
function LC._jacobian(ad::DI.ADTypes.AutoEnzyme, fun::typeof(LC._energy_at), u,
                      f::IL.NeuralODEFactor, args...; ε=1e-6)
    f.integrator isa IL.SciMLIntegrator && throw(ArgumentError(
        "SciML AutoEnzyme supports scalar endpoint loss gradients only; vector-return " *
        "split-reverse Jacobians fail in Enzyme's ABI wrapper. Use energy_gradient " *
        "or AutoForwardDiff/AutoZygote/AutoMooncake for Jacobians."))
    return invoke(LC._jacobian, Tuple{DI.ADTypes.AbstractADType,Any,Any,Vararg{Any}},
        ad, fun, u, f, args...; ε)
end

# Validate the ADTypes mode outside differentiation; leave _flow to Enzyme.
function LC._check_factor_ad(ad::DI.ADTypes.AutoEnzyme, f::IL.NeuralODEFactor)
    if f.integrator isa IL.SciMLIntegrator
        IL._check_sciml_ad(ad, f.integrator)
        Base.get_extension(IL, :ImplicitLayersSciMLSensitivityExt) === nothing &&
            throw(ArgumentError("SciML endpoint AD requires loading SciMLSensitivity"))
        ad.mode isa EC.ReverseMode && EC.runtime_activity(ad.mode) && return nothing
        throw(ArgumentError("SciML endpoint AutoEnzyme requires mode=Enzyme.set_runtime_activity(Enzyme.Reverse). " *
            "Forward mode fails in OrdinaryDiffEq callback initialization on the tested stack."))
    end
    f.integrator isa Union{IL.EulerIntegrator,IL.RK4Integrator} || return nothing
    ad.mode !== nothing && EC.runtime_activity(ad.mode) && return nothing
    throw(ArgumentError(
        "Enzyme differentiation of built-in NeuralODE flows requires runtime activity. " *
        "Pass AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)) " *
        "or AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))."))
end

end
