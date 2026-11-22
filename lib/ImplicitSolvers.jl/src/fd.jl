"""
    fd_jacobian(F, u::AbstractVector; step=nothing)

Return the dense Jacobian of `F` by central differences on host vectors. The probe
displacement in coordinate `j` is `step * max(1, abs(u[j]))`. The default multiplier
is `cbrt(eps(T))` in the primal coordinate type. Divide by the actual separation
of the two rounded probes.
"""
function fd_jacobian(F, u::AbstractVector; step=nothing)
    x = float.(u)
    T = _primal_type(eltype(x))
    h = step === nothing ? cbrt(eps(T)) : T(primal(step))
    isfinite(h) && h > 0 || throw(ArgumentError("step must be finite and positive"))
    if isempty(x)
        y = F(x)
        return Matrix{promote_type(eltype(x), float(eltype(y)))}(undef, length(y), 0)
    end
    return stack(eachindex(x)) do j
        xp, xm = copy(x), copy(x)
        value = primal(x[j])
        δ = h * max(one(T), abs(value))
        xp[j], xm[j] = x[j] + δ, x[j] - δ
        separation = primal(xp[j]) - primal(xm[j])
        isfinite(separation) && separation > 0 ||
            throw(ArgumentError("finite-difference probes must be finite and distinct at index $j"))
        return (F(xp) .- F(xm)) ./ separation
    end
end
