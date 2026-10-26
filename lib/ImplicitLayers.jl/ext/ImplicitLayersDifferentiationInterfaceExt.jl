module ImplicitLayersDifferentiationInterfaceExt
using ImplicitLayers: ImplicitLayers as IL
using LenticulumCore: LenticulumCore as LC
using DifferentiationInterface: DifferentiationInterface as DI
function IL._solve_state(ad::DI.ADTypes.AbstractADType, f, x, ps, st, z0)
    ad isa Union{DI.ADTypes.AutoForwardDiff,DI.ADTypes.AutoZygote} || throw(ArgumentError(
        "solve_state IFT rules currently support AutoForwardDiff and AutoZygote. " *
        "Other DI backends can still differentiate pointwise DEQ energies; " *
        "an unsupported solve backend is rejected to prevent differentiating root iterations."))
    u = vcat(x, LC._flatten_active(ps))
    return IL._implicit_state(ad, f, u, ps, LC.offset_tree(ps), st, z0)
end
end
