# ---------------------------------------------------------------------------
# The query report (P11): every query returns a root; this says how far it can be trusted.
#
# It gathers the checks the package already has into one record:
#   the solve       converged, stable (ImplicitSolution)
#   other starts    the stable roots `implicit_roots` reaches, and where the answer is among them
#   conditioning    cond₁ J_FF (the pullback's test) and the smallest eigenvalue of sym J_FF
#   on the relation `round_trip`, and `coverage` when a reference is given
# and lists what failed as flags. See `report.md`.
# ---------------------------------------------------------------------------

"""
    QueryReport

What [`query_report`](@ref) found about one answer to a query. Fields:

| field | meaning |
|---|---|
| `solution` | the answer, an [`ImplicitSolution`](@ref) |
| `converged`, `stable` | from the solve |
| `roots` | the distinct stable roots reached from other starts ([`implicit_roots`](@ref)), lowest energy first when the relation has one |
| `rank` | the answer's position in `roots`, `0` if it is not among them |
| `conditioning` | `(cond, lambda_min)`: ``\\operatorname{cond}_1 J_{FF}`` (`nothing` with [`KrylovSolve`](@ref)) and the smallest eigenvalue of ``\\operatorname{sym} J_{FF}`` |
| `round_trip` | the result of [`round_trip`](@ref), `nothing` for an answer that did not converge to a stable root |
| `coverage` | the result of [`coverage`](@ref), `nothing` without a reference |
| `flags` | what failed, a vector of symbols (below) |

Flags: `:not_converged`, `:unstable`, `:other_roots` (more than one stable root was found),
`:not_lowest` (a root found has a query energy below the answer's by more than ``\\sqrt\\epsilon``
relative; only for a relation with an energy), `:ill_conditioned`
(``\\operatorname{cond}_1 J_{FF}\\,\\epsilon \\ge 10^{-2}`` with ``\\epsilon`` of ``J_{FF}``'s element type:
[`implicit_pullback`](@ref) refuses such an answer), `:round_trip` (the reversed query found no
stable point, or moved the clamped coordinates by more than `round_trip_tol`), `:atypical`
(`coverage` flagged it).
`isempty(r.flags)` (or `trusted(r)`) when none applies.
"""
struct QueryReport{S<:ImplicitSolution,R<:AbstractVector,C<:NamedTuple,RT,CV}
    solution::S
    converged::Bool
    stable::Bool
    roots::R
    rank::Int
    conditioning::C
    round_trip::RT
    coverage::CV
    flags::Vector{Symbol}
end

"""
    trusted(r::QueryReport) -> Bool

True when the report has no flag.
"""
trusted(r::QueryReport) = isempty(r.flags)

"""
    query_report(m, z₀, ρ, ps, st; solution = nothing, z_init = z₀, nstarts = 16, spread = 1.0,
                 starts = nothing, rng = Xoshiro(0), unique_tol = 1e-3, reference = nothing,
                 alpha = 0.05, round_trip_tol = nothing, linsolve = DenseSolve(), kwargs...)
        -> (QueryReport, st)

Answer the query `(z₀, ρ)` and say how far the answer can be trusted. The answer is `solution`
when given, otherwise [`implicit_infer`](@ref) from `z_init`. Then:

1. **Other starts.** [`implicit_roots`](@ref) with `nstarts`, `spread`, `starts`, `rng` and
   `unique_tol`: every distinct stable root found, and the answer's place among them. A
   multivalued relation has several; with an energy, they are ordered lowest first.
2. **Conditioning** at the answer: ``\\operatorname{cond}_1 J_{FF}``, the test
   [`implicit_pullback`](@ref) applies (dense only), and the smallest eigenvalue of
   ``\\operatorname{sym}J_{FF}``, the curvature of the flattest direction (by Lanczos with
   [`KrylovSolve`](@ref)). A small eigenvalue means a wide Laplace covariance along it.
3. **On the relation.** [`round_trip`](@ref) for a converged, stable answer. It is flagged when
   the reversed query finds no stable point (distance `Inf`), and otherwise only against
   `round_trip_tol`, since the distance is in data units (none by default: the value is reported,
   not judged). It also fails where the relation is parallel to the clamped coordinates, so a
   flag there is not proof of an answer off the relation (`calibration.md`, "Remedies").
4. **Coverage**, when `reference` (a [`CoverageReference`](@ref)) is given: [`coverage`](@ref)
   of the answer at level `alpha`.

`linsolve` and `kwargs` (`tol`, `maxiters`, `method`, …) go to every solve. The cost is about
`nstarts + 3` solves.
"""
function query_report(m::AbstractImplicitRelation, z₀, ρ, ps, st; solution = nothing, z_init = z₀,
                      nstarts::Integer = 16, spread = 1.0, starts = nothing, rng::AbstractRNG = Random.Xoshiro(0),
                      unique_tol = 1e-3, reference = nothing, alpha::Real = 0.05, round_trip_tol = nothing,
                      linsolve::AbstractLinearSolve = DenseSolve(), kwargs...)
    sol = solution
    sol === nothing && ((sol, st) = implicit_infer(m, z₀, ρ, ps, st; z_init, linsolve, kwargs...))
    roots, st = implicit_roots(m, z₀, ρ, ps, st; nstarts, spread, starts, rng, unique_tol, linsolve, kwargs...)
    rank = something(findfirst(r -> norm(r.z .- sol.z) ≤ unique_tol * (1 + norm(sol.z)), roots), 0)
    cnd, Tj = _conditioning(linsolve, m, sol.z, ρ, ps, st)
    rt = nothing
    if sol.converged && sol.stable
        rt, st = round_trip(m, sol, z₀, ρ, ps, st; linsolve, kwargs...)
    end
    cov = reference === nothing ? nothing : coverage(reference, sol; alpha)

    flags = Symbol[]
    sol.converged || push!(flags, :not_converged)
    sol.stable || push!(flags, :unstable)
    length(roots) > 1 && push!(flags, :other_roots)
    _has_energy(m) && _lower_root(m, sol, roots, z₀, ρ, ps, st) && push!(flags, :not_lowest)
    cnd.cond !== nothing && _near_singular(cnd.cond, Tj) && push!(flags, :ill_conditioned)
    rt !== nothing && (rt.distance == Inf || (round_trip_tol !== nothing && !(rt.distance ≤ round_trip_tol))) &&
        push!(flags, :round_trip)
    cov !== nothing && cov.flagged && push!(flags, :atypical)
    return (QueryReport(sol, sol.converged, sol.stable, roots, rank, cnd, rt, cov, flags), st)
end

# the pullback's test: cond₁ J_FF so large that a solve with J_FF loses all accuracy
_near_singular(κ, T) = !(isfinite(κ) && κ * eps(T) < 1e-2)

# Is there a root whose query energy is below the answer's? Compared by energy, not by rank: the
# answer may be missing from `roots`, and tied roots are in no particular order (LP-031 F2, F3).
# The margin √ε (relative) lies above the energy error of a root solved to the tolerance.
function _lower_root(m, sol, roots, z₀, ρ, ps, st)
    isempty(roots) && return false
    E = _query_energy(m, sol.z, z₀, ρ, ps, st)
    Emin = minimum(_query_energy(m, r.z, z₀, ρ, ps, st) for r in roots)
    return Emin < E - sqrt(eps(float(eltype(sol.z)))) * (1 + abs(E))
end

function _conditioning(::DenseSolve, m, z, ρ, ps, st)
    free = .!isinf.(ρ)
    any(free) || return ((cond = 1.0, lambda_min = Inf), Float64)
    J = prior_jacobian(m, z, ps, st; coords = free) .+ Diagonal(ρ[free] .^ 2)
    Jh = Matrix(J)
    # the element type of J_FF is the arithmetic the pullback's test uses (LP-031 F1)
    return ((cond = float(LinearAlgebra.cond(Jh, 1)), lambda_min = float(LinearAlgebra.eigmin(Symmetric((Jh .+ Jh') ./ 2)))),
            eltype(Jh))
end

function _conditioning(ls::KrylovSolve, m, z, ρ, ps, st)
    op = _free_operators(m, z, ρ, ps, st)
    return ((cond = nothing, lambda_min = float(_lambda_min(op.S, op.n, ls))), Float64)
end

function Base.show(io::IO, ::MIME"text/plain", r::QueryReport)
    println(io, "QueryReport: ", trusted(r) ? "no flags" : join(r.flags, ", "))
    println(io, "  converged = ", r.converged, ", stable = ", r.stable, ", residual = ", r.solution.residual,
            ", iters = ", r.solution.iters)
    println(io, "  roots from other starts: ", length(r.roots), ", the answer's rank: ", r.rank)
    println(io, "  cond₁ J_FF = ", r.conditioning.cond, ", λ_min(sym J_FF) = ", r.conditioning.lambda_min)
    r.round_trip === nothing || println(io, "  round trip distance = ", r.round_trip.distance)
    r.coverage === nothing || print(io, "  coverage p = ", r.coverage.p, r.coverage.flagged ? " (flagged)" : "")
end
