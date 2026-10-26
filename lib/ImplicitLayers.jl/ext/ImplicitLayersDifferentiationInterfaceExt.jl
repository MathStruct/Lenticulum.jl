module ImplicitLayersDifferentiationInterfaceExt
using ImplicitLayers: ImplicitLayers as IL
using LenticulumCore: LenticulumCore as LC
using DifferentiationInterface: DifferentiationInterface as DI
function IL._solve_state(ad::DI.ADTypes.AbstractADType, f, x, ps, st, z0)
    ad isa Union{DI.ADTypes.AutoForwardDiff,DI.ADTypes.AutoZygote,
                    DI.ADTypes.AutoEnzyme,DI.ADTypes.AutoMooncake} || throw(ArgumentError(
        "solve_state IFT rules currently support AutoForwardDiff, AutoZygote, AutoEnzyme and AutoMooncake. " *
        "Other DI backends can still differentiate pointwise DEQ energies; " *
        "an unsupported solve backend is rejected to prevent differentiating root iterations."))
    ext = ad isa DI.ADTypes.AutoEnzyme ? :ImplicitLayersEnzymeCoreExt :
          ad isa DI.ADTypes.AutoMooncake ? :ImplicitLayersMooncakeExt : nothing
    if ext !== nothing && Base.get_extension(IL, ext) === nothing
        throw(ArgumentError("solve_state with Enzyme/Mooncake requires loading " *
            "DifferentiationInterface, the backend package and ForwardDiff for its IFT rule."))
    end
    u = IL._joint_vector(x, LC._flatten_active(ps))
    return IL._implicit_state(IL._solve_metadata(ad), IL._solve_metadata(f), u,
        IL._solve_metadata(ps), LC.offset_tree(ps), IL._solve_metadata(st), IL._solve_metadata(z0))
end
end
