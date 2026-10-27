module ImplicitLayersDifferentiationInterfaceExt
using ImplicitLayers: ImplicitLayers as IL
using LenticulumCore: LenticulumCore as LC
using DifferentiationInterface: DifferentiationInterface as DI
function LC._check_factor_ad(ad::DI.ADTypes.AbstractADType, f::IL.NeuralODEFactor, st)
    if f.integrator isa IL.SciMLIntegrator && !IL._empty_flow_state(st)
        throw(ArgumentError(
            "SciML NeuralODE differentiation requires recursively empty Lux state; " *
            "nonempty state uses legacy evaluation-state threading for primal evaluation, " *
            "but adaptive/rejected stages do not define supported ODE sensitivities."))
    end
    return LC._check_factor_ad(ad, f)
end
function LC._check_factor_ad(ad::DI.ADTypes.AbstractADType, f::IL.NeuralODEFactor)
    f.integrator isa IL.SciMLIntegrator || return nothing
    Base.get_extension(IL, :ImplicitLayersSciMLSensitivityExt) === nothing &&
        throw(ArgumentError("NeuralODE SciML parameter AD requires loading SciMLSensitivity"))
    ad isa Union{DI.ADTypes.AutoForwardDiff,DI.ADTypes.AutoZygote,DI.ADTypes.AutoMooncake} ||
        throw(ArgumentError("SciML NeuralODE sensitivities are tested only with AutoForwardDiff, AutoZygote and AutoMooncake"))
    return nothing
end
function IL._solve_state(ad::DI.ADTypes.AbstractADType, f, x, ps, st, z0)
    ad isa Union{DI.ADTypes.AutoForwardDiff,DI.ADTypes.AutoZygote,
                    DI.ADTypes.AutoEnzyme,DI.ADTypes.AutoMooncake} || throw(ArgumentError(
        "DEQ solve IFT rules currently support AutoForwardDiff, AutoZygote, AutoEnzyme and AutoMooncake. " *
        "Other DI backends can still differentiate pointwise DEQ energies; " *
        "an unsupported solve backend is rejected to prevent differentiating root iterations."))
    ext = ad isa DI.ADTypes.AutoEnzyme ? :ImplicitLayersEnzymeCoreExt :
          ad isa DI.ADTypes.AutoMooncake ? :ImplicitLayersMooncakeExt : nothing
    if ext !== nothing && Base.get_extension(IL, ext) === nothing
        throw(ArgumentError("DEQ solves with Enzyme/Mooncake requires loading " *
            "DifferentiationInterface, the backend package and ForwardDiff for its IFT rule."))
    end
    u = IL._joint_vector(x, LC._flatten_active(ps))
    return IL._implicit_state(IL._solve_metadata(ad), IL._solve_metadata(f), u,
        IL._solve_metadata(ps), LC.offset_tree(ps), IL._solve_metadata(st), IL._solve_metadata(z0))
end
end
