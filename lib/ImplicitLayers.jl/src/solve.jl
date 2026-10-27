# ---------------------------------------------------------------------------
# Root finding and finite-difference reference derivatives.
#
# Every factor in this package is a RELATION defined by a residual F(u) = 0, and inference is
# solving that residual for whichever channel the polarity left unobserved. So one root-finder
# serves both directions of a DEQ, which is the entire reason a DEQ is worth expressing as a
# factor rather than as a Lux layer.
#
# Two solvers, and the difference between them is the whole caveat of `Implicit Learners.md`
# §"Equilibrium":
#
#   Picard   u ← u - βF(u)      converges iff the iteration is a contraction
#   Broyden  quasi-Newton       converges on plenty of problems where Picard diverges
#
# Neither needs a derivative of the network. `fd_jacobian` does, and pays O(n) forward passes
# for it; see `solve.md` §4 for why that is a test-scale tool and not the real answer.
#
# See `solve.md`.
# ---------------------------------------------------------------------------

"""
    SolveReport(converged, iterations, residual, nfe)

What a root solve is willing to say about itself.

`nfe` (number of function evaluations) is reported because it is the honest cost measure for
an implicit layer — `DeepEquilibriumNetworks.jl` reports it too, in its
`DeepEquilibriumSolution`, for the same reason: iteration count means nothing when one
Broyden step and one Picard step cost differently.

> [!important] `converged == false` is not an error
> `Bayesian Lens.md`: a solver that stopped early is an inexact inversion, and inexact
> inversions are legal — the free energy records the cost. So the report is threaded out
> rather than thrown, and it is the caller's business what to do about it.
"""
struct SolveReport
    converged::Bool
    iterations::Int
    residual::Float64
    nfe::Int
end

Base.show(io::IO, r::SolveReport) = print(io,
    "SolveReport(", r.converged ? "converged" : "NOT converged",
    ", iters=", r.iterations, ", ‖F‖=", round(r.residual; sigdigits = 3), ", nfe=", r.nfe, ")")

"""
    abstract type AbstractRootSolver

A method for solving ``F(u) = 0`` given only the ability to *evaluate* ``F``.

Derivative-freeness is a requirement, not a convenience: the residual of a DEQ contains a
neural network, and this package has no automatic-differentiation dependency (see
`ImplicitLayers.md`). It is also what `DeepEquilibriumNetworks.jl` does in practice —
Broyden and limited-memory Broyden are its workhorses.
"""
abstract type AbstractRootSolver end

"""
    PicardSolver(; maxiters = 100, tol = 1e-10, damping = 1.0)

Damped fixed-point iteration ``u \\leftarrow u - \\beta F(u)``.

For the DEQ residual ``F(z) = z - g(z,x)`` this is exactly ``z \\leftarrow (1-\\beta)z +
\\beta g(z,x)``: the naive "just keep applying the layer" loop.

**A contraction is sufficient for convergence**, and nothing checks that in
advance; it is not a necessary condition for a particular initial guess. That is the caveat `Implicit Learners.md` records — *"this only works if the
iteration converges, and unconstrained DEQs need not"* — and the test suite exhibits a linear
cell with spectral radius > 1 on which this solver diverges and [`BroydenSolver`](@ref) does
not.
"""
struct PicardSolver <: AbstractRootSolver
    maxiters::Int
    tol::Float64
    damping::Float64
end
PicardSolver(; maxiters = 100, tol = 1e-10, damping = 1.0) =
    PicardSolver(maxiters, tol, damping)

"""
    BroydenSolver(; maxiters = 100, tol = 1e-10, init_scale = 1.0)

"Good" Broyden: a quasi-Newton method maintaining an approximate **inverse** Jacobian updated
by the Sherman–Morrison formula from secant pairs.

Derivative-free, and it is what `DeepEquilibriumNetworks.jl` reaches for. Note the
initialisation: with ``H_0 = I`` the first step is

```math
u_1 = u_0 - H_0F(u_0) = u_0 - (u_0 - g(u_0)) = g(u_0)
```

— **the first Broyden step is exactly a Picard step**, and everything after it is the
correction Picard never makes. Asserted in the test suite.

Stores a dense ``n\\times n`` matrix, so it is ``O(n^2)`` in memory. Real DEQs use
limited-memory Broyden for exactly this reason; see `solve.md` §4.2.
"""
struct BroydenSolver <: AbstractRootSolver
    maxiters::Int
    tol::Float64
    init_scale::Float64
end
BroydenSolver(; maxiters = 100, tol = 1e-10, init_scale = 1.0) =
    BroydenSolver(maxiters, tol, init_scale)

"""
    LimitedBroydenSolver(; memory=20, maxiters=100, tol=1e-10, init_scale=1.0)

Good inverse Broyden with at most `memory` rank-one secant corrections to
`init_scale*I`. Stores two vectors per correction, never a dense inverse: O(memory*n)
storage and work per step. Evicts the oldest correction without rebuilding the others.
With no eviction, reproduces dense Broyden up to roundoff. The first step is damped
Picard with damping `init_scale` (ordinary Picard when it is one).
"""
struct LimitedBroydenSolver <: AbstractRootSolver
    memory::Int
    maxiters::Int
    tol::Float64
    init_scale::Float64
    function LimitedBroydenSolver(; memory=20, maxiters=100, tol=1e-10, init_scale=1.0)
        memory > 0 || throw(ArgumentError("memory must be positive"))
        maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
        isfinite(tol) && tol >= 0 || throw(ArgumentError("tol must be finite and nonnegative"))
        isfinite(init_scale) && init_scale != 0 || throw(ArgumentError("init_scale must be finite and nonzero"))
        new(memory, maxiters, tol, init_scale)
    end
end

# H = scale*I + sum(a*b'). The stored vectors are transformed secant pairs.
function _broyden_product(a, b, scale, v)
    out = scale .* v
    for (ai, bi) in zip(a, b)
        out .+= ai .* dot(bi, v)
    end
    return out
end

function solve_root(F, u₀, s::LimitedBroydenSolver)
    u = copy(u₀)
    r = F(u)
    n = _norm(r)
    n <= s.tol && return (u, SolveReport(true, 0, n, 1))
    T = promote_type(Float64, eltype(u), eltype(r))
    a, b = Vector{T}[], Vector{T}[]
    for k in 1:s.maxiters
        du = -_broyden_product(a, b, s.init_scale, r)
        unew = u .+ du
        rnew = F(unew)
        nn = _norm(rnew)
        isfinite(nn) || return (unew, SolveReport(false, k, nn, k+1))
        Hdr = _broyden_product(a, b, s.init_scale, rnew .- r)
        denom = dot(du, Hdr)
        if abs(denom) > 1e-14
            right = _broyden_product(b, a, s.init_scale, du)
            left = (du .- Hdr) ./ denom
            if length(a) == s.memory
                popfirst!(a)
                popfirst!(b)
            end
            push!(a, left)
            push!(b, right)
        end
        u, r, n = unew, rnew, nn
        n <= s.tol && return (u, SolveReport(true, k, n, k+1))
    end
    return (u, SolveReport(false, s.maxiters, n, s.maxiters+1))
end

_norm(v) = sqrt(sum(abs2, v))

"""
    solve_root(F, u₀, solver) -> (u, SolveReport)

Solve ``F(u) = 0`` from the initial guess `u₀`.

`F` is called as `F(u)` and must return something the same shape as `u`. The residual norm
that decides convergence is the Euclidean norm; `tol` is absolute, which is a simplification
worth knowing about (`solve.md` §4.3).
"""
function solve_root(F, u₀, s::PicardSolver)
    u = copy(u₀)
    r = F(u)
    nfe = 1
    n = _norm(r)
    n <= s.tol && return (u, SolveReport(true, 0, n, nfe))
    for k in 1:(s.maxiters)
        u = u .- s.damping .* r
        r = F(u)
        nfe += 1
        n = _norm(r)
        # A diverging Picard iteration produces Inf/NaN rather than a large number; report
        # it as non-convergence instead of letting it poison everything downstream.
        (isfinite(n) && n <= s.tol) && return (u, SolveReport(true, k, n, nfe))
        isfinite(n) || return (u, SolveReport(false, k, n, nfe))
    end
    return (u, SolveReport(false, s.maxiters, n, nfe))
end

function solve_root(F, u₀, s::BroydenSolver)
    u = copy(u₀)
    r = F(u)
    nfe = 1
    n = _norm(r)
    n <= s.tol && return (u, SolveReport(true, 0, n, nfe))
    m = length(u)
    H = Matrix{Float64}(s.init_scale * I, m, m)   # approximate inverse Jacobian
    for k in 1:(s.maxiters)
        du = -(H * r)
        unew = u .+ du
        rnew = F(unew)
        nfe += 1
        nn = _norm(rnew)
        isfinite(nn) || return (unew, SolveReport(false, k, nn, nfe))
        dr = rnew .- r
        # Sherman–Morrison update of H ≈ J⁻¹; skipped when the denominator is degenerate,
        # which happens when the step produced no new information.
        denom = dot(du, H * dr)
        if abs(denom) > 1e-14
            H = H .+ ((du .- H * dr) * (du' * H)) ./ denom
        end
        u, r, n = unew, rnew, nn
        n <= s.tol && return (u, SolveReport(true, k, n, nfe))
    end
    return (u, SolveReport(false, s.maxiters, n, nfe))
end

# --- Derivatives, by finite differences ------------------------------------

"""
    fd_jacobian(F, u; ε = 1e-7) -> Matrix

The dense Jacobian ``\\partial F/\\partial u`` by forward differences: ``n+1`` evaluations of
`F`.

This remains a reference for small systems. Optional DI derivatives and solved-state
IFT rules use automatic differentiation instead; see `deq_sensitivity` and
`solve_state(...; ad)`. The root iterations themselves do not need derivatives.

"""
function fd_jacobian(F, u; ε = 1e-7)
    f0 = F(u)
    m, n = length(f0), length(u)
    J = zeros(Float64, m, n)
    for j in 1:n
        h = ε * max(one(eltype(u)), abs(u[j]))
        up = copy(u)
        up[j] += h
        J[:, j] .= (F(up) .- f0) ./ h
    end
    return J
end

"""
    ift_sensitivity(Jz, Jx) -> Matrix

The implicit function theorem, applied to a fixed point ``z^\\ast = g(z^\\ast, x)``:

```math
(I - \\partial_z g)\\,\\frac{\\partial z^\\ast}{\\partial x} = \\partial_x g
\\qquad\\Longrightarrow\\qquad
\\frac{\\partial z^\\ast}{\\partial x} = (I - \\partial_z g)^{-1}\\,\\partial_x g
```

**One linear solve, no unrolling** — storage independent of the root iteration count
(the dense state Jacobian still takes quadratic storage), and the reason `Implicit Learners.md` lists the IFT as the equilibrium family's
backward pass. See [[Backpropagation by the Implicit Function Theorem]].

Throws if ``I - \\partial_z g`` is singular: the IFT does not establish a locally
differentiable inverse there. This alone does not imply nonuniqueness (a cubic
residual can have a unique root with zero derivative). Dense matrix reference only.

"""
function ift_sensitivity(Jz::AbstractMatrix, Jx::AbstractMatrix)
    n = size(Jz, 1)
    M = Matrix{Float64}(I, n, n) .- Jz
    F = lu(M; check = false)
    issuccess(F) || throw(SingularException(0))
    return F \ Jx
end

# The differentiation linear solve is independent of the primal root solver.
abstract type AbstractIFT end
"Dense residual Jacobian and LU (the default first-order IFT rule)."
struct DenseIFT <: AbstractIFT end
"""
    KrylovIFT(; rtol=1e-10, atol=1e-12, itmax=0, memory=20,
               precond=nothing, ldiv=false, on_solve=nothing)

Opt-in matrix-free GMRES IFT solve. Load Krylov and DifferentiationInterface.
`itmax=0` uses Krylov's default (twice the state dimension). Restart after `memory`
iterations to bound basis storage. Failure to meet the residual tolerance throws.

`precond` accepts a left operator or a factory `ctx -> (; left=M, right=N)`;
omitted sides are identity. `ctx` contains `operator`, `transposed`, `factor`, `solved` (channel name),
`joint` (observed channel followed by parameters), `root`, `template`, `offsets`
and `state`. Build preconditioners for the indicated orientation. Krylov applies
`mul!` by default (operators approximate the inverse); `ldiv=true` applies `ldiv!`
instead. `JacobiIFT()` probes the operator diagonal. `on_solve`, when supplied,
receives `(iterations, residual, solved)` as a named tuple after each linear solve,
including failures; residual is measured in the original, unpreconditioned system.
"""
struct KrylovIFT{P,C} <: AbstractIFT
    rtol::Float64
    atol::Float64
    itmax::Int
    memory::Int
    precond::P
    ldiv::Bool
    on_solve::C
    function KrylovIFT(; rtol=1e-10, atol=1e-12, itmax=0, memory=20,
                       precond=nothing, ldiv=false, on_solve=nothing)
        isfinite(rtol) && rtol >= 0 || throw(ArgumentError("rtol must be finite and nonnegative"))
        isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative"))
        itmax >= 0 || throw(ArgumentError("itmax must be nonnegative"))
        memory > 0 || throw(ArgumentError("memory must be positive"))
        new{typeof(precond),typeof(on_solve)}(rtol, atol, itmax, memory, precond, ldiv, on_solve)
    end
end

"""
    JacobiIFT(; min_diagonal=1e-14)

Opt-in `KrylovIFT(precond=JacobiIFT())` inverse diagonal preconditioner. Uses n
operator probes and O(n) storage, without forming the state Jacobian. Rejects
nonfinite or small diagonal entries. Uses `mul!` (`ldiv=false`).
"""
struct JacobiIFT
    min_diagonal::Float64
    function JacobiIFT(; min_diagonal=1e-14)
        isfinite(min_diagonal) && min_diagonal >= 0 ||
            throw(ArgumentError("min_diagonal must be finite and nonnegative"))
        new(min_diagonal)
    end
end
_ift_prepare(ad, f, u, z, like, offs, st, transposed) =
    _ift_prepare(f.ift, ad, f, u, z, like, offs, st, transposed)
function _ift_prepare(::DenseIFT, ad, f, u, z, like, offs, st, transposed)
    J = _root_jacobian(ad, f, u, z, like, offs, st)
    return lu(transposed ? J' : J)
end
_ift_prepare(::KrylovIFT, args...) = throw(ArgumentError(
    "KrylovIFT requires loading Krylov and DifferentiationInterface"))
_ift_solve(A, b) = A \ b
