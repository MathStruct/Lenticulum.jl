module ImplicitLayersGPUArraysCoreExt
using ImplicitLayers: ImplicitLayers as IL
using GPUArraysCore: AbstractGPUArray
using LinearAlgebra: lu

IL._initial_guess(x::AbstractGPUArray, n) = fill!(similar(x, n), 0)
# Times and coefficients must have the primal state precision, including under AD.
IL._flow_times(u::AbstractGPUArray, t0, t1) =
    (convert(typeof(real(zero(eltype(u)))), t0), convert(typeof(real(zero(eltype(u)))), t1))
function IL._dense_ift_system(J::AbstractGPUArray, transposed)
    try
        return lu(transposed ? J' : J)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("DenseIFT requires a device LU factorisation, unavailable for $(typeof(J)); use KrylovIFT(). Underlying error: $(sprint(showerror, err))"))
    end
end
IL._divergence(vf, u::AbstractGPUArray, t; ε=1e-7) = throw(ArgumentError("Device finite-difference flow divergence is unsupported"))
IL._check_hvp_device(x::AbstractGPUArray) = throw(ArgumentError("DEQ second-order adjoints require CPU arrays; device HVPs are unsupported"))
function IL._check_solve_device(ad, u::AbstractGPUArray)
    nameof(typeof(ad)) in (:AutoForwardDiff, :AutoZygote) || throw(ArgumentError(
        "Device DEQ IFT is tested with AutoForwardDiff and AutoZygote only; Enzyme/Mooncake device rules are unsupported"))
end
end
