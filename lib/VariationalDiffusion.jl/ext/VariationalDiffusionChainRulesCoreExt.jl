# ---------------------------------------------------------------------------
# `implicit_infer` as a differentiable function for AD packages: a `ChainRulesCore.rrule` whose
# pullback is `implicit_pullback`, closed over the forward solution. Zygote (and anything else
# that reads ChainRules) then differentiates through inference without unrolling the solver:
#
#     Zygote.gradient(ps -> loss(first(implicit_infer(m, z₀, ρ, ps, st)).z), ps)
#
# Conventions (`Backward Signatures.md` §5): no cotangent for `st` or the model, none for the
# solver's report (residual, iterations, flags), and the pullback refuses a non-converged
# solution, as `implicit_pullback` does.
# ---------------------------------------------------------------------------
module VariationalDiffusionChainRulesCoreExt

using VariationalDiffusion: VariationalDiffusion, AbstractImplicitRelation, ImplicitSolution, DenseSolve,
                            implicit_infer, implicit_pullback, prox_infer, prox_pullback
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, AbstractZero, Tangent, unthunk

# the cotangent of `sol.z`, from whatever form the AD package hands back
_zbar(::Union{Nothing,AbstractZero}, z) = nothing
_zbar(s̄::Tangent, z) = _zbar(s̄.z, z)
_zbar(s̄::NamedTuple, z) = haskey(s̄, :z) ? _zbar(s̄.z, z) : nothing
_zbar(z̄::AbstractArray, z) = z̄

function ChainRulesCore.rrule(::typeof(implicit_infer), m::AbstractImplicitRelation, z₀, ρ, ps, st; kwargs...)
    sol, st′ = implicit_infer(m, z₀, ρ, ps, st; kwargs...)
    linsolve = get(kwargs, :linsolve, DenseSolve())
    function implicit_infer_pullback(ȳ)
        ȳ = unthunk(ȳ)
        s̄ = ȳ isa Union{Tuple,Tangent} ? unthunk(ȳ[1]) : ȳ
        z̄ = _zbar(s̄, sol.z)
        z̄ === nothing && return (NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent())
        b = implicit_pullback(m, sol, z₀, ρ, collect(float.(unthunk(z̄))), ps, st; linsolve)
        return (NoTangent(), NoTangent(), b.z₀, b.ρ, b.ps, NoTangent())
    end
    return (sol, st′), implicit_infer_pullback
end

# the same for proximal inference; `t` (the prox's noise level) gets no cotangent
function ChainRulesCore.rrule(::typeof(prox_infer), p, z₀, ρ, t, ps, st; kwargs...)
    sol, st′ = prox_infer(p, z₀, ρ, t, ps, st; kwargs...)
    λ = get(kwargs, :λ, 0.01)
    linsolve = get(kwargs, :linsolve, DenseSolve())
    function prox_infer_pullback(ȳ)
        ȳ = unthunk(ȳ)
        s̄ = ȳ isa Union{Tuple,Tangent} ? unthunk(ȳ[1]) : ȳ
        z̄ = _zbar(s̄, sol.z)
        z̄ === nothing && return (NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), ZeroTangent(), NoTangent())
        b = prox_pullback(p, sol, z₀, ρ, t, collect(float.(unthunk(z̄))), ps, st; λ, linsolve)
        return (NoTangent(), NoTangent(), b.z₀, b.ρ, NoTangent(), b.ps, NoTangent())
    end
    return (sol, st′), prox_infer_pullback
end

end # module
