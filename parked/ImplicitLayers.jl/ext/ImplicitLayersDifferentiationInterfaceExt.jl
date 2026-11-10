module ImplicitLayersDifferentiationInterfaceExt
using ImplicitLayers: ImplicitLayers as IL
using LenticulumCore: LenticulumCore as LC
using DifferentiationInterface: DifferentiationInterface as DI
using Mycelium: Mycelium as MY
function LC._check_factor_ad(ad::DI.ADTypes.AbstractADType, f::IL.NeuralODEFactor, st)
    if f.integrator isa IL.SciMLIntegrator && f.state_policy !== :fixed && !IL._empty_flow_state(st)
        throw(ArgumentError(
            "SciML NeuralODE differentiation requires recursively empty Lux state; " *
            "use state_policy=:fixed for nonempty state; legacy evaluation-state threading is available for primal evaluation, " *
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
    IL._check_sciml_ad(ad, f.integrator)
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
    IL._check_solve_device(ad, u)
    z, rep, state = IL._implicit_state(IL._solve_metadata(ad), IL._solve_metadata(f), u,
        IL._solve_metadata(ps), LC.offset_tree(ps), IL._solve_metadata(st), IL._solve_metadata(z0))
    return z, IL._public_report(rep), state
end

# Coupled graph replay must select the same implicit primitive as direct solves,
# rather than tracing a derivative-free root iteration in factor_message.
function MY._gradient_message(ad::DI.ADTypes.AbstractADType, f::IL.DEQFactor,
        target, polarity, inputs, prior, ps, st)
    target === IL.inchannel(f) && !IL._empty_flow_state(st) && throw(ArgumentError(
        "DEQ input-message AD replay requires empty Lux state: primal input inversion threads state, while its IFT fixes state"))
    return IL._solve_for(f, target, inputs, prior, ps, st; ad)
end
function MY._gradient_message(::Val{:chainrules}, f::IL.DEQFactor,
        target, polarity, inputs, prior, ps, st)
    return MY._gradient_message(DI.ADTypes.AutoZygote(), f, target, polarity, inputs, prior, ps, st)
end
end
