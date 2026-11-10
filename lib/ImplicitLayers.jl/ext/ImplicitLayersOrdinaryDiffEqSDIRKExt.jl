module ImplicitLayersOrdinaryDiffEqSDIRKExt

using ImplicitLayers: ImplicitLayers
using OrdinaryDiffEqCore: OrdinaryDiffEqCore
using OrdinaryDiffEqSDIRK: ImplicitMidpoint, Trapezoid

# A user-supplied step limiter can destroy symmetry even with fixed dt.
ImplicitLayers._symmetric_algorithm(alg::Union{ImplicitMidpoint,Trapezoid}) =
    alg.step_limiter! === OrdinaryDiffEqCore.trivial_limiter!

end
