"""
    SolverReport(converged, iterations, residual, nfe, reason)

Record the status of a solve. `residual` is the primal Euclidean norm at the returned
iterate. `nfe` counts calls to the residual or linear operator. `reason` is one of
`:converged`, `:maxiters`, `:nonfinite`, and `:stalled`.
"""
struct SolverReport{R<:Real}
    converged::Bool
    iterations::Int
    residual::R
    nfe::Int
    reason::Symbol

    function SolverReport(converged::Bool, iterations::Int, residual::Real, nfe::Int, reason::Symbol)
        value = primal(residual)
        return new{typeof(value)}(converged, iterations, value, nfe, reason)
    end
end

_report(reason, iterations, residual, nfe) =
    SolverReport(reason === :converged, iterations, residual, nfe, reason)

Base.show(io::IO, r::SolverReport) = print(io,
    "SolverReport(", r.reason, ", iters=", r.iterations,
    ", residual=", round(r.residual; sigdigits=3), ", nfe=", r.nfe, ")")
