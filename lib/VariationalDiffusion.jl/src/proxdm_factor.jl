# ---------------------------------------------------------------------------
# A DiffusionFactor whose inversion is proximal inference (`ProxDM`).
#
# `invert`, `factor_message` and `local_free_energy` in `factor.jl` route through `_run_prox`;
# `implicit_solution` and `implicit_factor_pullback` in `implicit_factor.jl` through
# `_factor_solve` and `_factor_pullback`. This file adds the ProxDM methods. The factor carries
# two parameter trees, `(predictor = …, prox = …)`; the inversion reads `prox` only. See
# `proxdm.md` §6.
# ---------------------------------------------------------------------------

function _factor_solve(prox::ProxDM, f::DiffusionFactor, z₀, ρ, ps, st)
    sol, stp = prox_infer(prox.prox, z₀, ρ, prox.t, ps.prox, st.prox; λ = prox.λ, z_init = z₀,
                          tol = prox.tol, maxiters = prox.maxiters)
    return (sol, merge(st, (prox = stp,)))
end

function _run_prox(::ProxDM, f::DiffusionFactor, p::LenticulumCore.Polarity, inputs, π, ps, st)
    sol, st = implicit_solution(f, p, inputs, π, ps, st)
    return (sol.z, st)
end

function _factor_pullback(prox::ProxDM, f::DiffusionFactor, sol, z₀, ρ, z̄, ps, st)
    b = prox_pullback(prox.prox, sol, z₀, ρ, prox.t, z̄, ps.prox, st.prox; λ = prox.λ, linsolve = prox.linsolve)
    return (z₀ = b.z₀, ρ = b.ρ, ps = (predictor = _zero(ps.predictor), prox = b.ps))
end
