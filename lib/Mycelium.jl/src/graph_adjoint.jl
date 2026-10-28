# Fixed points live in message coordinates, not factor solver coordinates. No
# dependency on ImplicitLayers is needed. All reference contexts are outside AD.
_empty_graph_state(x::Union{Tuple,NamedTuple}) = all(_empty_graph_state, x)
_empty_graph_state(x) = false

_message_coordinates(::LenticulumCore.TrivialBelief) = Float64[]
function _message_coordinates(b::GaussianBelief)
    b.η isa Vector{<:AbstractFloat} && (b.Λ isa Matrix{<:AbstractFloat} ||
        b.Λ isa LinearAlgebra.Diagonal{<:AbstractFloat,<:Vector}) ||
        throw(ArgumentError("implicit graph adjoints require CPU real floating Gaussian messages"))
    # Full precision coordinates avoid triangular packing conventions. Antisymmetric
    # directions are redundant but do not make I-DT singular by themselves.
    return _pack_message(b)
end
function _message_coordinates(b::LenticulumCore.DiracBelief)
    x = b.value
    x isa AbstractFloat && return [x]
    x isa Array{<:AbstractFloat} && return vec(x)
    throw(ArgumentError("implicit graph adjoints require numeric CPU real floating Dirac messages"))
end
_message_coordinates(b) = throw(ArgumentError("implicit graph adjoint has no coordinates for $(typeof(b)); use Gaussian/Dirac messages"))

_pack_message(::LenticulumCore.TrivialBelief) = Float64[]
_precision_coordinates(Λ) = vec(Λ)
_precision_coordinates(Λ::LinearAlgebra.Diagonal) = Λ.diag
_pack_message(b::GaussianBelief) = vcat(b.η, _precision_coordinates(b.Λ))
_pack_message(b::LenticulumCore.DiracBelief) = b.value isa Number ? [b.value] : vec(b.value)
_restore_precision(x, Λ) = reshape(x,size(Λ))
_restore_precision(x, Λ::LinearAlgebra.Diagonal) = LinearAlgebra.Diagonal(x)
_restore_message(x, b::GaussianBelief) = GaussianBelief(x[1:length(b.η)], _restore_precision(x[length(b.η)+1:end], b.Λ))
_restore_message(x, b::LenticulumCore.DiracBelief) = LenticulumCore.DiracBelief(b.value isa Number ? only(x) : reshape(x, size(b.value)))
_restore_message(x, b::LenticulumCore.TrivialBelief) = b
_stored_belief(s, t) = (t.kind === :to_factor ? s.to_factor[t.edge] : s.to_variable[t.edge]).belief

_same_message_layout(a, b) = false
_same_message_layout(::LenticulumCore.TrivialBelief, ::LenticulumCore.TrivialBelief) = true
_same_message_layout(a::GaussianBelief, b::GaussianBelief) =
    size(a.η) == size(b.η) && size(a.Λ) == size(b.Λ) &&
    (a.Λ isa LinearAlgebra.Diagonal) == (b.Λ isa LinearAlgebra.Diagonal)
_same_message_layout(a::LenticulumCore.DiracBelief, b::LenticulumCore.DiracBelief) =
    a.value isa Union{Real,AbstractArray{<:Real}} && size(a.value) == size(b.value)

function _coordinate_store(x, ctx)
    return foldl(eachindex(ctx.ts); init=_ReplayStore(ctx.g)) do s, i
        _with_message(s, ctx.ts[i], _restore_message(x[ctx.ranges[i]], ctx.beliefs[i]))
    end
end

# Synchronous BP defines the fixed-point equation even when another schedule was
# used to find its root. Damping follows step!: only factor -> variable messages.
function _graph_update(x, ps, ctx)
    s = _coordinate_store(x, ctx)
    pinned = _pinned_parameters(ps, ctx.ps, ctx.g)
    parts = map(eachindex(ctx.ts)) do i
        t = ctx.ts[i]
        e = ctx.g.edges[t.edge]
        if e.detach || e.coupling isa DiagonalCoupling
            return ctx.x[ctx.ranges[i]]
        end
        b, _ = _damped_candidate(s, ctx.g, t, pinned, ctx.st, ctx.damping, ctx.ad)
        e.coupling isa LenticulumCore.PathwiseCoupling && _check_pathwise(b)
        _same_message_layout(b, ctx.beliefs[i]) || throw(ArgumentError(
            "implicit BP update must preserve message family, shape and precision structure"))
        return _pack_message(b)
    end
    return reduce(vcat, parts)
end
_graph_state_map(x, ps, ctx) = _graph_update(x, ps, ctx)
_graph_loss(x, ps, ctx) = _graph_scalar(_coordinate_store(x, ctx), ctx.g,
    _pinned_parameters(ps, ctx.ps, ctx.g), ctx.st)
_graph_map_dot(x, ps, ctx, v) = LinearAlgebra.dot(v, _graph_update(x, ps, ctx))
_graph_adjoint_loss(ps, ctx, λ) = _graph_loss(ctx.x, ps, ctx) +
    LinearAlgebra.dot(λ, _graph_update(ctx.x, ps, ctx))

struct _GraphAdjointOperator{T,C} <: AbstractMatrix{T}
    ctx::C
end
Base.size(A::_GraphAdjointOperator) = (length(A.ctx.x), length(A.ctx.x))
Base.size(A::_GraphAdjointOperator, d::Int) = d <= 2 ? length(A.ctx.x) : 1
function Base.:*(A::_GraphAdjointOperator, v::AbstractVector)
    c = A.ctx
    product = LenticulumCore._gradient(c.ad, _graph_map_dot, c.x, c.ps, c, v; ε=c.ε)
    return product === nothing ? copy(v) : v - product
end
function LinearAlgebra.mul!(y::AbstractVector, A::_GraphAdjointOperator, v::AbstractVector)
    copyto!(y, A * v)
    return y
end

function _graph_linear_solve(::Val{:dense}, A, rhs, ctx, tol, maxiter)
    J = LenticulumCore._jacobian(ctx.ad, _graph_state_map, ctx.x, ctx.ps, ctx; ε=ctx.ε)
    J === nothing && return copy(rhs)
    M = Matrix{eltype(J)}(LinearAlgebra.I, size(J)...) - J'
    λ = try
        M \ rhs
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        throw(ArgumentError("implicit graph adjoint: I-DT is singular"))
    end
    all(isfinite, λ) || throw(ArgumentError("implicit graph adjoint produced nonfinite values"))
    return λ
end
_graph_linear_solve(::Val{:krylov}, A, rhs, ctx, tol, maxiter) = throw(ArgumentError(
    "linear_solver=:krylov requires loading Krylov.jl"))

function _implicit_graph_gradient(g, ps, st; ad, ε, schedule, damping, maxsweeps, tol,
        linear_solver, linear_tol, linear_maxiter, rng, nsamples, return_state)
    _check_schedule_ad(ad)
    _empty_graph_state(st) || throw(ArgumentError("implicit graph adjoints require empty factor state; evolving Lux state is not part of this fixed-point equation"))
    LenticulumCore._flatten(ps) isa Vector{<:AbstractFloat} || throw(ArgumentError(
        "implicit graph adjoints require CPU real floating parameters; device coordinate packing is unsupported"))
    schedule isa Union{SequentialSchedule,TreeSchedule,ForwardBackwardSchedule,FloodingSchedule,ResidualSchedule} ||
        throw(ArgumentError("implicit graph adjoint requires a supported schedule=... for reference convergence"))
    all(e -> e.detach || e.coupling isa Union{DiagonalCoupling,LenticulumCore.ExactCoupling,LenticulumCore.PathwiseCoupling}, g.edges) ||
        throw(ArgumentError("implicit graph adjoints support Exact/Pathwise couplings and diagonal/detach cuts; score estimators are unsupported"))
    rng === nothing && nsamples === nothing || throw(ArgumentError("implicit graph adjoints do not sample; omit rng/nsamples"))
    linear_solver in (:dense, :krylov) || throw(ArgumentError("linear_solver must be :dense or :krylov"))
    maxsweeps isa Int && maxsweeps > 0 && isfinite(tol) && tol > 0 || throw(ArgumentError("maxsweeps and tol must be positive"))
    isfinite(linear_tol) && linear_tol > 0 && linear_maxiter isa Int && linear_maxiter > 0 || throw(ArgumentError("linear_tol and linear_maxiter must be positive"))
    0 <= damping < 1 || throw(ArgumentError("damping must be in [0, 1)"))
    for f in g.factors
        LenticulumCore._check_factor_ad(ad, f.factor, _subtree(st, f.name))
    end
    ts = all_tasks(g)
    isempty(ts) && throw(ArgumentError("implicit graph adjoints need message tasks"))
    if !(schedule isa ResidualSchedule)
        chosen = tasks(schedule)
        all(t -> 1 <= t.edge <= nedges(g) && t.kind in (:to_factor, :to_variable) && islegal(g,t), chosen) ||
            throw(ArgumentError("implicit graph reference schedule contains an illegal message task"))
        all(t -> any(u -> u.edge == t.edge && u.kind == t.kind, chosen), ts) ||
            throw(ArgumentError("implicit graph reference schedule must cover every legal message task"))
    else
        schedule.maxtasks > 0 || throw(ArgumentError("ResidualSchedule.maxtasks must be positive"))
    end
    # Do not trust TreeSchedule's one-sweep report when damping or a hand-built
    # task list is involved. Iterate and verify all message coordinates below.
    store = MessageStore(g)
    converged = false
    for _ in 1:maxsweeps
        res, nextstate = sweep!(store, g, schedule, ps, st; damping)
        _empty_graph_state(nextstate) || throw(ArgumentError(
            "implicit graph messages must preserve empty factor state"))
        if res <= tol
            converged = true
            break
        end
    end
    converged || throw(ArgumentError("implicit graph reference inference did not converge within maxsweeps"))
    all(t -> (t.kind === :to_factor ? store.to_factor[t.edge] : store.to_variable[t.edge]) !== nothing, ts) ||
        throw(ArgumentError("implicit graph reference did not populate every legal message"))
    beliefs = map(t -> _stored_belief(store,t), ts)
    parts = map(_message_coordinates, beliefs)
    lengths = length.(parts)
    starts = cumsum(vcat(1, lengths[1:end-1]))
    ranges = map((i,n) -> i:i+n-1, starts, lengths)
    x = reduce(vcat, parts)
    isempty(x) && throw(ArgumentError("implicit graph message state has no numeric coordinates"))
    ctx = (; g, ps, st, ts, beliefs, ranges, x, ad, damping, ε)
    # Check the full, uncut update too: a detached row must still be at a BP root.
    s = _coordinate_store(x, ctx)
    residual = maximum(ts) do t
        b, _ = _damped_candidate(s, g, t, ps, st, damping)
        belief_distance(_stored_belief(s,t), b)
    end
    isfinite(residual) && residual <= 10tol || throw(ArgumentError("implicit graph reference fails the full BP fixed-point check (residual=$residual)"))
    rhs = LenticulumCore._gradient(ad, _graph_loss, x, ps, ctx; ε)
    rhs = rhs === nothing ? zero(x) : rhs
    # The scalar objective can promote its message cotangent (e.g. CPU Gaussian
    # entropy constants with Float32 messages). GMRES vectors follow that RHS.
    A = _GraphAdjointOperator{eltype(rhs),typeof(ctx)}(ctx)
    λ = _graph_linear_solve(Val(linear_solver), A, rhs, ctx, linear_tol, linear_maxiter)
    residual_linear = LinearAlgebra.norm(A * λ - rhs)
    residual_linear <= linear_tol * max(one(residual_linear), LinearAlgebra.norm(rhs)) ||
        throw(ArgumentError("implicit graph adjoint linear solve failed residual check ($residual_linear)"))
    gradient = LenticulumCore._parameter_gradient(ad, _graph_adjoint_loss, ps, ctx, λ; ε)
    state = return_state ? last(factor_free_energies(store,g,ps,st)) : nothing
    return return_state ? (gradient, state) : gradient
end
