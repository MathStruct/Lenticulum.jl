module LenticulumCoreGPUArraysCoreDIExt
using LenticulumCore: LenticulumCore as LC
using GPUArraysCore: AbstractGPUArray
using DifferentiationInterface: DifferentiationInterface as DI
const AD = DI.ADTypes

# ForwardDiff's native vector API scalar-indexes during seeding/extraction. DI's
# directional pushforward broadcasts dual seeds; assemble columns without indexing.
function LC._jacobian(ad::AD.AutoForwardDiff, fun, u::AbstractGPUArray, args...; ε=1e-6)
    ctx = map(DI.Constant, args)
    columns = map(eachindex(u)) do j
        e = similar(u)
        e .= ifelse.(eachindex(u) .== j, one(eltype(u)), zero(eltype(u)))
        only(DI.pushforward(fun, ad, u, (e,), ctx...))
    end
    return hcat(columns...)
end
function LC._gradient(ad::AD.AutoForwardDiff, fun, u::AbstractGPUArray, args...; ε=1e-6)
    ctx = map(DI.Constant, args)
    out = similar(u)
    for j in eachindex(u)
        e = similar(u)
        e .= ifelse.(eachindex(u) .== j, one(eltype(u)), zero(eltype(u)))
        fill!(view(out, j:j), only(DI.pushforward(fun, ad, u, (e,), ctx...)))
    end
    return out
end
for backend in (:AutoEnzyme, :AutoMooncake)
    @eval begin
        LC._gradient(ad::AD.$backend, fun, u::AbstractGPUArray, args...; ε=1e-6) =
            throw(ArgumentError("Enzyme/Mooncake device parameter AD is unsupported; use AutoForwardDiff or AutoZygote"))
        LC._jacobian(ad::AD.$backend, fun, u::AbstractGPUArray, args...; ε=1e-6) =
            throw(ArgumentError("Enzyme/Mooncake device parameter AD is unsupported; use AutoForwardDiff or AutoZygote"))
    end
end
end
