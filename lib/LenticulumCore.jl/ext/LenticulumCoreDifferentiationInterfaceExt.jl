module LenticulumCoreDifferentiationInterfaceExt
using LenticulumCore: LenticulumCore
using DifferentiationInterface: DifferentiationInterface as DI
const ADTypes = DI.ADTypes
function LenticulumCore._jacobian(ad::ADTypes.AbstractADType, fun, u, args...; ε = 1e-6)
    return DI.jacobian(fun, ad, u, map(DI.Constant, args)...)
end
function LenticulumCore._gradient(ad::ADTypes.AbstractADType, fun, u, args...; ε = 1e-6)
    return DI.gradient(fun, ad, u, map(DI.Constant, args)...)
end
end
