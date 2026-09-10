# ---------------------------------------------------------------------------
# Root finding, and the derivative it does not have.
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

**It converges if and only if the iteration is a contraction**, and nothing checks that in
advance. That is the caveat `Implicit Learners.md` records — *"this only works if the
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

This is the package's **only** source of derivative information, and it is a test-scale tool.
For a real DEQ the answer is a vector–Jacobian product from automatic differentiation, at
``O(1)`` cost; here it is ``O(n)`` forward passes and it loses roughly half the significant
digits to the step size. It exists so that [`ift_sensitivity`](@ref) can be *implemented and
checked* against a closed form, not so that it can be used at scale. See `solve.md` §4.1.
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

**One linear solve, no unrolling** — the property that makes implicit layers ``O(1)`` in
memory, and the reason `Implicit Learners.md` lists the IFT as the equilibrium family's
backward pass. See [[Backpropagation by the Implicit Function Theorem]].

Throws if ``I - \\partial_z g`` is singular. That is not a numerical accident: it is the
statement that the fixed point is **not locally unique**, i.e. that the relation branches
there. The algebraic family calls that locus the discriminant
([[Branches and the Discriminant]]); here it is the same phenomenon and the same failure.
"""
function ift_sensitivity(Jz::AbstractMatrix, Jx::AbstractMatrix)
    n = size(Jz, 1)
    M = Matrix{Float64}(I, n, n) .- Jz
    F = lu(M; check = false)
    issuccess(F) || throw(SingularException(0))
    return F \ Jx
end
