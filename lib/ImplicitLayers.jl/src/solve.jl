const SolveReport = ImplicitSolvers.SolverReport

# The shared algorithms take the tolerance and iteration budget at solve time.
# These settings keep the ImplicitLayers constructor and call conventions.
struct _ConfiguredRootSolver{A<:AbstractRootSolver,T<:Tolerance} <: AbstractRootSolver
    algorithm::A
    maxiters::Int
    tol::T
end

const PicardSolver = _ConfiguredRootSolver{A} where {A<:Picard}
const BroydenSolver = _ConfiguredRootSolver{A} where {A<:Broyden}
const LimitedBroydenSolver = _ConfiguredRootSolver{A} where {A<:LimitedBroyden}

_root_tolerance(tol::Tolerance) = tol
_root_tolerance(tol::Real) = Tolerance(atol=tol, rtol=0)

"""
    PicardSolver(; maxiters=100, tol=Tolerance(), damping=1)

Damped fixed-point iteration using `ImplicitSolvers.Picard`. A real `tol`
specifies an absolute residual bound.
"""
PicardSolver(; maxiters=100, tol=Tolerance(), damping=1) =
    _ConfiguredRootSolver(Picard(; damping), Int(maxiters), _root_tolerance(tol))
PicardSolver(maxiters, tol, damping) = PicardSolver(; maxiters, tol, damping)

"""
    BroydenSolver(; maxiters=100, tol=Tolerance(), init_scale=1)

Dense inverse Broyden using `ImplicitSolvers.Broyden`. A real `tol` specifies
an absolute residual bound.
"""
BroydenSolver(; maxiters=100, tol=Tolerance(), init_scale=1) =
    _ConfiguredRootSolver(Broyden(; init_scale), Int(maxiters), _root_tolerance(tol))
BroydenSolver(maxiters, tol, init_scale) = BroydenSolver(; maxiters, tol, init_scale)

"""
    LimitedBroydenSolver(; memory=20, maxiters=100, tol=Tolerance(), init_scale=1)

Limited-memory inverse Broyden using `ImplicitSolvers.LimitedBroyden`.
A real `tol` specifies an absolute residual bound. Pass `history` to
`solve_root` to retain the inverse approximation for `BroydenIFT`.
"""
function LimitedBroydenSolver(; memory=20, maxiters=100, tol=Tolerance(), init_scale=1)
    maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
    tol isa Real && !isfinite(tol) && throw(ArgumentError("tol must be finite"))
    isfinite(init_scale) && !iszero(init_scale) ||
        throw(ArgumentError("init_scale must be finite and nonzero"))
    return _ConfiguredRootSolver(LimitedBroyden(; memory=Int(memory), init_scale),
        Int(maxiters), _root_tolerance(tol))
end

function solve_root(F, u₀::AbstractVector, solver::_ConfiguredRootSolver;
                    tol=solver.tol, maxiters=solver.maxiters, history=nothing)
    return solve_root(F, u₀, solver.algorithm; tol=_root_tolerance(tol),
        maxiters=Int(maxiters), history)
end

# Compatibility for the scaled norm used by the LP-019 regression tests.
const _norm = LinearAlgebra.norm

# A forward-difference probe for the NeuralODE flow divergence, in the precision of x.
# The requested h is rounded when it is added, so the quotient must use the rounded one; and
# 1e-7 is below the resolution of Float16 and close to Float32's, so the default relative step
# is max(1e-7, √eps) for the element type (LP-019 item 4). Float64 keeps 1e-7.
# `eps` of a ForwardDiff dual is its primal's, so Float16 duals inside a stiff SciML solver get
# the Float16 step too; types without `eps` keep 1e-7. A step too small to change x is an error,
# not a 0/0 (LP-019 review).
function _fd_step(x::Real, ε)
    T = typeof(float(x))
    e = ε !== nothing ? ε : applicable(eps, T) ? max(1e-7, sqrt(Float64(eps(T)))) : 1e-7
    xp = x + T(e * max(one(T), abs(x)))
    h = xp - x                             # the increment the rounded probe actually makes
    iszero(h) && throw(ArgumentError("finite-difference step ε = $e does not change x = $x in $T; use a larger ε"))
    return xp, h
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
    return lu(I - Jz) \ Jx
end

# The differentiation linear solve is independent of the primal root solver.
abstract type AbstractIFT end
"Dense residual Jacobian and LU (the default first-order IFT rule)."
struct DenseIFT <: AbstractIFT end
"""
    KrylovIFT(; rtol=1e-10, atol=1e-12, itmax=0, memory=20,
               precond=nothing, ldiv=false, on_solve=nothing)

Opt-in matrix-free GMRES IFT solve. Load DifferentiationInterface.
`itmax=0` uses twice the dimension of the linear system. Restart after `memory`
iterations to bound basis storage. Failure to meet the residual tolerance throws.

`precond` accepts a left operator or a factory `ctx -> M` or `ctx -> (; left=M, right=N)`;
omitted sides are identity. `ctx` contains `operator`, `transposed`, `factor`, `solved` (channel name),
`joint` (observed channel followed by parameters), `root`, `template`, `offsets`
and `state`. Build preconditioners for the indicated orientation. The solver applies
`mul!` by default (operators approximate the inverse); `ldiv=true` applies `ldiv!`
instead. `JacobiIFT()` probes the operator diagonal. `on_solve`, when supplied,
receives `(iterations, residual, solved)` as a named tuple after each linear solve,
including failures; residual is measured in the original, unpreconditioned system.
"""
struct KrylovIFT{P,C,R<:Real,A<:Real} <: AbstractIFT
    rtol::R
    atol::A
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
        new{typeof(precond),typeof(on_solve),typeof(rtol),typeof(atol)}(
            rtol, atol, itmax, memory, precond, ldiv, on_solve)
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

"""
    BroydenIFT()

Opt-in `KrylovIFT(precond=BroydenIFT())` left inverse preconditioner using the
actual forward `LimitedBroydenSolver` history. Uses its transpose for adjoint
solves. Requires `ldiv=false`. History is owned by one AD solve, never global;
no extra root solve or dense Jacobian is used. Warm starts may leave no secants.
If restarted GMRES stalls, up to two correction solves use the true residual.
Iteration diagnostics include corrections within the same budget.
"""
struct BroydenIFT end
# The history's array types depend on the first residual. Keeping them out of
# the report type makes the AD primitive's return type concrete with Ref{Any}.
struct _HistoryReport{R}
    report::R
    history::Any
end
Base.getproperty(r::_HistoryReport, s::Symbol) = s in (:report, :history) ?
    getfield(r, s) : getproperty(getfield(r, :report), s)
_public_report(r) = r
_public_report(r::_HistoryReport) = r.report
_solve_history(r) = nothing
_solve_history(r::_HistoryReport) = r.history
function _solve_root_ift(F, z0, f)
    if f.ift isa KrylovIFT && f.ift.precond isa BroydenIFT
        f.solver isa Union{LimitedBroydenSolver,LimitedBroyden} || throw(ArgumentError(
            "BroydenIFT requires LimitedBroydenSolver to retain forward history"))
        f.ift.ldiv && throw(ArgumentError("BroydenIFT supplies an inverse; use ldiv=false"))
        history = Ref{Any}()
        z, rep = solve_root(F, z0, f.solver; history)
        return z, _HistoryReport(rep, history[])
    end
    return solve_root(F, z0, f.solver)
end
_ift_prepare(ad, f, u, z, like, offs, st, transposed, history=nothing) =
    _ift_prepare(f.ift, ad, f, u, z, like, offs, st, transposed, history)
function _ift_prepare(::DenseIFT, ad, f, u, z, like, offs, st, transposed, history=nothing)
    J = _root_jacobian(ad, f, u, z, like, offs, st)
    return _dense_ift_system(J, transposed)
end
_ift_prepare(::KrylovIFT, args...) = throw(ArgumentError(
    "KrylovIFT requires loading DifferentiationInterface"))
_ift_solve(A, b) = A \ b

_dense_ift_system(J, transposed) = transposed ? J' : J

function _ift_solve(J::AbstractMatrix, b::AbstractVector)
    x, report = adjoint_solve(J, b, DenseSolve())
    if report.iterations == 0 && !isempty(b) && report.reason !== :nonfinite
        throw(SingularException(0))
    end
    report.converged || throw(ArgumentError(
        "DEQ dense IFT did not converge: $(report.reason); residual=$(report.residual)"))
    return x
end

# Device specialisations live in the optional GPUArraysCore extension.
function _initial_guess(x, n)
    T = eltype(x)
    return zeros(T <: Number && isconcretetype(T) ? float(T) : Float64, n)
end
_flow_times(u, t0, t1) = (t0, t1)
_check_solve_device(ad, u) = nothing

_check_hvp_device(x) = nothing
