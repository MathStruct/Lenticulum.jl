module ImplicitLayersSciMLSensitivityExt
using ImplicitLayers: ImplicitLayers as IL
using LuxCore: LuxCore
using LenticulumCore: LenticulumCore as LC
using SciMLBase: ODEProblem, solve, successful_retcode
using SciMLSensitivity: ForwardDiffSensitivity

function IL._flow(f, u, ps, st, t0, t1, itg::IL.SciMLIntegrator)
    # Loading sensitivities must preserve primal evaluation-state threading.
    IL._empty_flow_state(st) || return invoke(IL._flow,
        Tuple{Any,Any,Any,Any,Any,Any,IL.AbstractIntegrator}, f, u, ps, st, t0, t1, itg)
    t0 == t1 && return copy(u), st
    θ = LC._flatten_active(ps)
    offs = LC.offset_tree(ps)
    # Capture only a zero-valued shape template, never the learnable parameters.
    template = LC._unflatten(zero(θ), offs, ps)
    dynamics, input = f.dynamics, f.input
    rhs = function (z, p, t)
        du, state = LuxCore.apply(dynamics, input(z, t), LC._unflatten(p, offs, template), st)
        IL._empty_flow_state(state) || throw(ArgumentError(
            "SciMLSensitivity NeuralODE dynamics must return empty Lux state"))
        return du
    end
    prob = ODEProblem(rhs, copy(u), (t0, t1), θ)
    opts = itg.kwargs
    if haskey(opts, :dt)
        opts = merge(opts, (; dt = sign(t1 - t0) * opts.dt))
    end
    opts = merge(opts, (; save_everystep=false, save_start=false, save_end=true,
                         saveat=(), save_idxs=nothing, dense=false))
    sensealg = itg.sensealg === nothing ? ForwardDiffSensitivity() : itg.sensealg
    sol = solve(prob, itg.alg; sensealg, opts...)
    successful_retcode(sol) && last(sol.t) == t1 || error(
        "SciMLIntegrator did not reach the final time $t1 (retcode=$(sol.retcode))")
    return last(sol.u), st
end
end
