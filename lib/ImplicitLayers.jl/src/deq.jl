# ---------------------------------------------------------------------------
# The DEQ factor:   r(x, z) = z - g_θ(z, x),   and R_θ = { (x,z) : r(x,z) = 0 }
#
# `DeepEquilibriumNetworks.jl` gives you a Lux layer  x ↦ z*(x): the relation is solved in
# ONE direction and the solve is sealed inside a function. This factor keeps the residual and
# lets the polarity decide which channel to solve for — which is the difference between an
# implicit layer and an implicit *learner*.
#
#   x observed, z unobserved  →  solve z = g(z,x) for z     (what SciML's DEQ does)
#   z observed, x unobserved  →  solve z = g(z,x) for x     (what it cannot do)
#
# Same residual, same solver, different polarity. See `deq.md` and `DEQ as a Relation.md`.
# ---------------------------------------------------------------------------

"""
    DEQFactor(cell, dims::NamedTuple; solver = BroydenSolver(), channels = (:x, :z),
              input = default_cell_input)

A deep equilibrium model as a **relation**: the fixed-point condition
``z = g_\\theta(z, x)`` presented as a residual, solvable for either channel.

`cell` is any `LuxCore.AbstractLuxLayer` computing ``g_\\theta(z,x)`` — including a
`DeepEquilibriumNetworks.jl` cell, which is exactly such a layer. `dims` gives the two
channel dimensions. Parameters and state are the cell's, untouched.

```julia
f = DEQFactor(my_cell, (x = 4, z = 4); solver = BroydenSolver(maxiters = 200))
```

!!! warning "The reverse direction needs a square system"
    Solving ``z = g(z,x)`` for `x` is ``\\dim z`` equations in ``\\dim x`` unknowns. Unless
    ``\\dim x = \\dim z`` it is over- or under-determined and the root solve is not asking a
    well-posed question. [`LenticulumCore.supports_polarity`](@ref) refuses that polarity
    rather than solving it badly; see `deq.md` §3.

Contrast [`LuxFactor`](@ref), which wraps an *already assembled* `DeepEquilibriumNetwork` and
gets one polarity — a Lux layer with extra steps.
"""
struct DEQFactor{names,D<:Tuple,C,S<:AbstractRootSolver,F} <:
       LenticulumCore.AbstractLenticulumFactor
    dims::NamedTuple{names,D}
    cell::C
    solver::S
    input::F
end

"""
    default_cell_input(z, x) = (z, x)

How a DEQ cell is called: `LuxCore.apply(cell, input(z, x), ps, st)`. Override via the
`input` keyword for cells expecting a different arrangement — the factor deliberately knows
nothing about the cell's internals.
"""
default_cell_input(z, x) = (z, x)

function DEQFactor(
    cell, dims::NamedTuple;
    solver::AbstractRootSolver = BroydenSolver(),
    channels::Tuple{Symbol,Symbol} = (:x, :z),
    input = default_cell_input,
)
    length(dims) == 2 || throw(ArgumentError(
        "a DEQFactor has exactly two channels (input and state); got $(keys(dims))"))
    all(d -> d isa Int && d > 0, values(dims)) ||
        throw(ArgumentError("channel dimensions must be positive Ints; got $dims"))
    keys(dims) == channels || throw(ArgumentError(
        "dims keys $(keys(dims)) must match channels $channels, in order"))
    return DEQFactor(dims, cell, solver, input)
end

inchannel(f::DEQFactor{names}) where {names} = names[1]
statechannel(f::DEQFactor{names}) where {names} = names[2]
indim(f::DEQFactor) = f.dims[inchannel(f)]
statedim(f::DEQFactor) = f.dims[statechannel(f)]

LuxCore.initialparameters(rng::AbstractRNG, f::DEQFactor) =
    LuxCore.initialparameters(rng, f.cell)
LuxCore.initialstates(rng::AbstractRNG, f::DEQFactor) = LuxCore.initialstates(rng, f.cell)
LuxCore.parameterlength(f::DEQFactor) = LuxCore.parameterlength(f.cell)
LuxCore.statelength(f::DEQFactor) = LuxCore.statelength(f.cell)

LenticulumCore.channels(f::DEQFactor{names}) where {names} =
    map((n, d) -> LenticulumCore.Channel(n, d), names, values(f.dims))

LenticulumCore.islearnable(::DEQFactor) = true

"""
    LenticulumCore.supported_polarities(f::DEQFactor)

Both directions when ``\\dim x = \\dim z``; only the forward one otherwise.

**The count is the point.** A `DeepEquilibriumNetwork` has one direction, fixed when you
construct it. This has two whenever the residual is square, and the second one is a genuinely
new capability rather than a re-labelling — it solves the same equation for the other
variable.
"""
function LenticulumCore.supported_polarities(f::DEQFactor{names}) where {names}
    O, U = LenticulumCore.Observed(), LenticulumCore.Unobserved()
    xc, zc = inchannel(f), statechannel(f)
    fwd = LenticulumCore.Polarity(NamedTuple{names}(map(n -> n === zc ? U : O, names)))
    indim(f) == statedim(f) || return (fwd,)
    bwd = LenticulumCore.Polarity(NamedTuple{names}(map(n -> n === xc ? U : O, names)))
    return (fwd, bwd)
end

function LenticulumCore.supports_polarity(
    f::DEQFactor{names}, p::LenticulumCore.Polarity
) where {names}
    keys(p) == names || return false
    u = LenticulumCore.unobserved_channels(p)
    length(u) == 1 || return false
    only(u) === statechannel(f) && return true
    # the reverse solve is only well posed on a square residual
    return only(u) === inchannel(f) && indim(f) == statedim(f)
end

"""
    LenticulumCore.energyspace(::DEQFactor)

``E_c = \\mathbb{R}^{\\dim z}``: the residual itself, as a **vector**.

This is what [[Scalar and Multivariate Energy]] asks for and what
`VariationalDiffusion`'s factor could not provide — here the residual is a genuine
``\\mathbb{R}^n``-valued object and the natural scalarisation is ``\\tfrac12\\|\\cdot\\|^2``.
Keeping it vector-valued is exactly what the implicit function theorem needs, per that note's
§6.3.
"""
LenticulumCore.energyspace(f::DEQFactor) =
    LenticulumCore.EuclideanEnergySpace(statedim(f))
LenticulumCore.scalarisation(::DEQFactor) = LenticulumCore.SquaredNorm()

"""
    cell_value(f, z, x, ps, st) -> (g, st)

One forward pass of the cell: ``g_\\theta(z, x)``.
"""
cell_value(f::DEQFactor, z, x, ps, st) =
    LuxCore.apply(f.cell, f.input(z, x), ps, st)

"""
    residual(f::DEQFactor, x, z, ps, st) -> (r, st)

``r = z - g_\\theta(z,x)``, the vector energy. Membership of the relation is ``r \\approx 0``,
which is [[README]]'s definition of an implicit learner instantiated exactly.
"""
function residual(f::DEQFactor, x, z, ps, st)
    g, st = cell_value(f, z, x, ps, st)
    return (z .- g, st)
end

LenticulumCore.energy(f::DEQFactor, x, a, y, ps, st) = residual(f, x, y, ps, st)

# --- Solving, in whichever direction --------------------------------------

"""
    solve_state(f, x, ps, st; z₀) -> (z★, SolveReport, st)

Solve ``z = g_\\theta(z, x)`` for `z`. The forward DEQ pass — what
`DeepEquilibriumNetworks.jl` does.

`z₀` defaults to zeros, which is `SkipDeepEquilibriumNetwork`'s point of departure: a learned
initial guess is strictly better and is not implemented (`deq.md` §4.4).
"""
function solve_state(f::DEQFactor, x, ps, st; z₀ = zeros(Float64, statedim(f)))
    stref = Ref(st)
    F = function (z)
        r, s = residual(f, x, z, ps, stref[])
        stref[] = s
        return r
    end
    z, rep = solve_root(F, z₀, f.solver)
    return (z, rep, stref[])
end

"""
    solve_input(f, z, ps, st; x₀) -> (x★, SolveReport, st)

Solve ``z = g_\\theta(z, x)`` for **`x`**, holding `z` fixed — the direction a
`DeepEquilibriumNetwork` cannot be asked for.

Requires ``\\dim x = \\dim z``. Even then it is a general nonlinear root-find in `x` with no
contraction structure to lean on, so `Broyden` is the only sensible solver and convergence is
a genuinely open question per call. That is why the `SolveReport` is returned rather than
discarded.
"""
function solve_input(f::DEQFactor, z, ps, st; x₀ = zeros(Float64, indim(f)))
    indim(f) == statedim(f) || throw(ArgumentError(
        "solve_input needs dim($(inchannel(f))) == dim($(statechannel(f))); got \
         $(indim(f)) and $(statedim(f)). The residual is not square, so solving for \
         :$(inchannel(f)) is not a well-posed root-find."))
    stref = Ref(st)
    F = function (x)
        r, s = residual(f, x, z, ps, stref[])
        stref[] = s
        return r
    end
    x, rep = solve_root(F, x₀, f.solver)
    return (x, rep, stref[])
end

"""
    deq_sensitivity(f, x, z★, ps, st) -> ∂z★/∂x

The implicit function theorem at the solved fixed point:
``\\partial z^\\ast/\\partial x = (I - \\partial_z g)^{-1}\\partial_x g``.

Both Jacobians come from [`fd_jacobian`](@ref), so this costs ``O(\\dim x + \\dim z)`` forward
passes and is a **test-scale tool** — the real answer is a VJP from automatic
differentiation. It is here because it makes the IFT *checkable*: for a linear cell
``g = Wz + Ux + b`` the exact answer is ``(I-W)^{-1}U``, and the test suite compares against
it.

This is also the quantity a Gaussian message would need in order to push a covariance through
the layer, which `deq.md` §4.1 explains cannot currently be returned.
"""
function deq_sensitivity(f::DEQFactor, x, z★, ps, st)
    # ∂r/∂z = I - ∂_z g  and  ∂r/∂x = -∂_x g, so both cell Jacobians come from the residual
    Jr_z = fd_jacobian(z -> first(residual(f, x, z, ps, st)), z★)
    Jr_x = fd_jacobian(xx -> first(residual(f, xx, z★, ps, st)), x)
    n = size(Jr_z, 1)
    Idn = Matrix{Float64}(I, n, n)
    ∂zg = Idn .- Jr_z
    ∂xg = .-Jr_x
    return ift_sensitivity(∂zg, ∂xg)
end

# --- The open model and the solver inversion -------------------------------

"""
    DEQModel(factor, polarity)

The open model the polarity selects: "solve this fixed-point equation for the unobserved
channel". Pure — a deterministic relation with no internal randomness.
"""
struct DEQModel{F<:DEQFactor,P} <: LenticulumCore.AbstractOpenModel
    factor::F
    polarity::P
end
LenticulumCore.ispure(::DEQModel) = true
LenticulumCore.latentspace(::DEQModel) = nothing

LenticulumCore.assemble(f::DEQFactor, p::LenticulumCore.Polarity, ps, st) =
    (LenticulumCore.BayesianLens(DEQModel(f, p), LenticulumCore.SolverInversion(f.solver)), st)

"""
    LenticulumCore.invert(lens, π, inputs, ps, st) -> (DiracBelief, st)

Run the root solve in the direction the polarity chose.

Returns a `DiracBelief`: a root-find produces a point, not a distribution. Propagating
uncertainty would need the linearisation of [`deq_sensitivity`](@ref) *and* a
`GaussianBelief` to put it in — and `GaussianBelief` lives in the top-level `Lenticulum`
package, which no `lib/` package may depend on. See `deq.md` §4.1; the same wall is recorded
in `VariationalDiffusion`'s `factor.md`.
"""
function LenticulumCore.invert(
    lens::LenticulumCore.BayesianLens{<:DEQModel,<:LenticulumCore.SolverInversion},
    π, inputs, ps, st,
)
    f, p = lens.model.factor, lens.model.polarity
    target = only(LenticulumCore.unobserved_channels(p))
    b, st = _solve_for(f, target, inputs, π, ps, st)
    return (b, st)
end

function _solve_for(f::DEQFactor, target::Symbol, inputs, π, ps, st)
    xc, zc = inchannel(f), statechannel(f)
    if target === zc
        x = _point(_get(inputs, xc))
        x === nothing && return (LenticulumCore.TrivialBelief(), st)
        z₀ = something(_point(π), zeros(Float64, statedim(f)))
        z, _, st = solve_state(f, x, ps, st; z₀ = z₀)
        return (LenticulumCore.DiracBelief(z), st)
    elseif target === xc
        z = _point(_get(inputs, zc))
        z === nothing && return (LenticulumCore.TrivialBelief(), st)
        x₀ = something(_point(π), zeros(Float64, indim(f)))
        x, _, st = solve_input(f, z, ps, st; x₀ = x₀)
        return (LenticulumCore.DiracBelief(x), st)
    end
    throw(ArgumentError("channel :$target is not a channel of this DEQFactor"))
end

"""
    Mycelium.factor_message(f::DEQFactor, target, polarity, inputs, prior, ps, st)

The factor → variable message: a `DiracBelief` on `target`, from a root solve.

The incoming `prior` is used as the solver's **initial guess** and nothing else. That is a
genuinely good use for it — warm-starting from the current belief is what makes iterated
message passing over implicit layers cheap — and it is *not* the same as conditioning on it.
See `deq.md` §4.2 for why this message is a posterior rather than a likelihood.
"""
function Mycelium.factor_message(
    f::DEQFactor{names}, target::Symbol, polarity, inputs, prior, ps, st
) where {names}
    target in names || throw(ArgumentError(
        "channel :$target is not a channel of this DEQFactor (has $names)"))
    return _solve_for(f, target, inputs, prior, ps, st)
end

"""
    Mycelium.local_free_energy(f::DEQFactor, msgs, ps, st)

``\\tfrac12\\|r\\|^2`` at the incoming messages' points, or `0.0` when a channel carries no
point.

Note what this is *not*: there is no entropy term, because the inversion returns a Dirac and
a Dirac has none. So this factor contributes energy only, and the Bethe counting correction
of [[Bethe Free Energy]] has nothing to correct here. `deq.md` §4.3.
"""
function Mycelium.local_free_energy(f::DEQFactor, msgs, ps, st)
    x = _point(_get(msgs, inchannel(f)))
    z = _point(_get(msgs, statechannel(f)))
    (x === nothing || z === nothing) && return (0.0, st)
    r, st = residual(f, x, z, ps, st)
    return (sum(abs2, r) / 2, st)
end
