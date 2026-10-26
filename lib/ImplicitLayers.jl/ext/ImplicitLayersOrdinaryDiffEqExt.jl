module ImplicitLayersOrdinaryDiffEqExt

using ImplicitLayers: ImplicitLayers, SciMLIntegrator
using SciMLBase: ODEProblem, solve, successful_retcode

function ImplicitLayers._sciml_integrate(vf, u₀, t₀, t₁, itg::SciMLIntegrator)
    t₀ == t₁ && return copy(u₀)
    opts = itg.kwargs
    if haskey(opts, :dt)
        opts = merge(opts, (; dt = sign(t₁ - t₀) * opts.dt))
    end
    prob = ODEProblem((u, p, t) -> vf(u, t), copy(u₀), (t₀, t₁))
    # These storage settings are invariants of this final-state-only interface.
    opts = merge(opts, (; save_everystep = false, save_start = false, save_end = true,
                         saveat = (), save_idxs = nothing, dense = false))
    sol = solve(prob, itg.alg; opts...)
    successful_retcode(sol) && last(sol.t) == t₁ || error(
        "SciMLIntegrator did not reach the final time $t₁ (retcode=$(sol.retcode))")
    return last(sol.u)
end

function ImplicitLayers._sciml_divergence(vf, u₀, t₀, t₁, itg::SciMLIntegrator)
    n = length(u₀)
    augmented = vcat(u₀, zero(eltype(u₀)))
    rhs = function (v, t)
        u = v[1:n]
        du = vf(u, t)
        div = ImplicitLayers._divergence(vf, u, t)
        return vcat(du, div)
    end
    final = ImplicitLayers.integrate(rhs, augmented, t₀, t₁, itg)
    return (final[1:n], final[n + 1])
end

end
