module ImplicitLayersGPUArraysCoreExt
using ImplicitLayers: ImplicitLayers as IL
using GPUArraysCore: AbstractGPUArray

IL._solver_scalar(u::AbstractGPUArray, x) = convert(eltype(u), x)
function IL._broyden_storage(u::AbstractGPUArray, r, scale)
    V = typeof(similar(u, promote_type(eltype(u), eltype(r))))
    return V[], V[], convert(eltype(u), scale)
end
function IL._broyden_inverse(u::AbstractGPUArray, scale, n)
    H = similar(u, n, n)
    H .= (1:n) .== (1:n)'
    H .*= convert(eltype(u), scale)
    return H
end
IL._initial_guess(x::AbstractGPUArray, n) = fill!(similar(x, n), 0)
# Times and coefficients must have the primal state precision, including under AD.
IL._flow_times(u::AbstractGPUArray, t0, t1) =
    (convert(typeof(real(zero(eltype(u)))), t0), convert(typeof(real(zero(eltype(u)))), t1))
function IL._ift_factorize(J::AbstractGPUArray, transposed)
    try
        return invoke(IL._ift_factorize, Tuple{Any,Any}, J, transposed)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("DenseIFT requires a device LU factorisation, unavailable for $(typeof(J)); use KrylovIFT(). Underlying error: $(sprint(showerror, err))"))
    end
end
IL.fd_jacobian(F, u::AbstractGPUArray; ε=1e-7) = throw(ArgumentError("Device finite-difference Jacobians are unsupported; use automatic residual derivatives"))
IL._divergence(vf, u::AbstractGPUArray, t; ε=1e-7) = throw(ArgumentError("Device finite-difference flow divergence is unsupported"))
IL._check_hvp_device(x::AbstractGPUArray) = throw(ArgumentError("DEQ second-order adjoints require CPU arrays; device HVPs are unsupported"))
function IL._check_solve_device(ad, u::AbstractGPUArray)
    nameof(typeof(ad)) in (:AutoForwardDiff, :AutoZygote) || throw(ArgumentError(
        "Device DEQ IFT is tested with AutoForwardDiff and AutoZygote only; Enzyme/Mooncake device rules are unsupported"))
end
end
