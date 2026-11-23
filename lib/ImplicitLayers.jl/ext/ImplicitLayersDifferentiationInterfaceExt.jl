module ImplicitLayersDifferentiationInterfaceExt
using ImplicitLayers: ImplicitLayers as IL
using LenticulumCore: LenticulumCore as LC
using DifferentiationInterface: DifferentiationInterface as DI
using MyceliumLearning: MyceliumLearning as ML
using ImplicitSolvers: GMRES, Tolerance, adjoint_solve, _broyden_product, _primal_norm, _status, _report
using LinearAlgebra: LinearAlgebra, I, Diagonal
function LC._check_factor_ad(ad::DI.ADTypes.AbstractADType, f::IL.NeuralODEFactor, st)
    if f.integrator isa IL.SciMLIntegrator && f.state_policy !== :fixed && !IL._empty_flow_state(st)
        throw(ArgumentError(
            "SciML NeuralODE differentiation requires recursively empty Lux state; " *
            "use state_policy=:fixed for nonempty state; legacy evaluation-state threading is available for primal evaluation, " *
            "but adaptive/rejected stages do not define supported ODE sensitivities."))
    end
    return LC._check_factor_ad(ad, f)
end
function LC._check_factor_ad(ad::DI.ADTypes.AbstractADType, f::IL.NeuralODEFactor)
    f.integrator isa IL.SciMLIntegrator || return nothing
    Base.get_extension(IL, :ImplicitLayersSciMLSensitivityExt) === nothing &&
        throw(ArgumentError("NeuralODE SciML parameter AD requires loading SciMLSensitivity"))
    ad isa Union{DI.ADTypes.AutoForwardDiff,DI.ADTypes.AutoZygote,DI.ADTypes.AutoMooncake} ||
        throw(ArgumentError("SciML NeuralODE sensitivities are tested only with AutoForwardDiff, AutoZygote and AutoMooncake"))
    IL._check_sciml_ad(ad, f.integrator)
    return nothing
end
function IL._solve_state(ad::DI.ADTypes.AbstractADType, f, x, ps, st, z0)
    ad isa Union{DI.ADTypes.AutoForwardDiff,DI.ADTypes.AutoZygote,
                    DI.ADTypes.AutoEnzyme,DI.ADTypes.AutoMooncake} || throw(ArgumentError(
        "DEQ solve IFT rules currently support AutoForwardDiff, AutoZygote, AutoEnzyme and AutoMooncake. " *
        "Other DI backends can still differentiate pointwise DEQ energies; " *
        "an unsupported solve backend is rejected to prevent differentiating root iterations."))
    ext = ad isa DI.ADTypes.AutoEnzyme ? :ImplicitLayersEnzymeCoreExt :
          ad isa DI.ADTypes.AutoMooncake ? :ImplicitLayersMooncakeExt : nothing
    if ext !== nothing && Base.get_extension(IL, ext) === nothing
        throw(ArgumentError("DEQ solves with Enzyme/Mooncake requires loading " *
            "DifferentiationInterface, the backend package and ForwardDiff for its IFT rule."))
    end
    u = IL._joint_vector(x, LC._flatten_active(ps))
    IL._check_solve_device(ad, u)
    z, rep, state = IL._implicit_state(IL._solve_metadata(ad), IL._solve_metadata(f), u,
        IL._solve_metadata(ps), LC.offset_tree(ps), IL._solve_metadata(st), IL._solve_metadata(z0))
    return z, IL._public_report(rep), state
end

# Since the split of Mycelium (2026-11) the coupled replay, and with it `_gradient_message`,
# is MyceliumLearning's.
# Coupled graph replay must select the same implicit primitive as direct solves,
# rather than tracing a derivative-free root iteration in factor_message.
function ML._gradient_message(ad::DI.ADTypes.AbstractADType, f::IL.DEQFactor,
        target, polarity, inputs, prior, ps, st)
    target === IL.inchannel(f) && !IL._empty_flow_state(st) && throw(ArgumentError(
        "DEQ input-message AD replay requires empty Lux state: primal input inversion threads state, while its IFT fixes state"))
    return IL._solve_for(f, target, inputs, prior, ps, st; ad)
end
function ML._gradient_message(::Val{:chainrules}, f::IL.DEQFactor,
        target, polarity, inputs, prior, ps, st)
    return ML._gradient_message(DI.ADTypes.AutoZygote(), f, target, polarity, inputs, prior, ps, st)
end

# Minimal operator protocol: no indexing and no matrix conversion.
struct ResidualOperator{T,F}
    n::Int
    product::F
end
Base.size(A::ResidualOperator) = (A.n, A.n)
Base.size(A::ResidualOperator, k::Integer) = k <= 2 ? A.n : 1
Base.eltype(::ResidualOperator{T}) where {T} = T
LinearAlgebra.mul!(y, A::ResidualOperator, x) = copyto!(y, A.product(x))
struct KrylovSystem{A,K,M,N}
    operator::A
    options::K
    left::M
    right::N
end
_preconditioners(::Nothing, ctx, k) = (nothing, nothing)
function _preconditioners(::IL.BroydenIFT, ctx, k)
    k.ldiv && throw(ArgumentError("BroydenIFT supplies an inverse; use ldiv=false"))
    h = ctx.history
    h === nothing && throw(ArgumentError("BroydenIFT requires retained forward solve history"))
    a, b = ctx.transposed ? (h.b, h.a) : (h.a, h.b)
    product = v -> _broyden_product(a, b, h.scale, v)
    return ResidualOperator{eltype(ctx.root),typeof(product)}(length(ctx.root), product), nothing
end
function _preconditioners(p::IL.JacobiIFT, ctx, k)
    k.ldiv && throw(ArgumentError("JacobiIFT supplies an inverse; use ldiv=false"))
    z, A = ctx.root, ctx.operator
    diagonal, e, out = similar(z), zero(z), similar(z)
    for j in eachindex(z)
        e .= ifelse.(eachindex(z) .== j, one(eltype(z)), zero(eltype(z)))
        LinearAlgebra.mul!(out, A, e)
        d = view(out, j:j)
        all(v -> isfinite(v) && abs(v) > p.min_diagonal, d) || throw(ArgumentError(
            "JacobiIFT requires finite nonzero diagonal entries; invalid entry $j"))
        view(diagonal, j:j) .= inv.(d)
    end
    return Diagonal(diagonal), nothing
end
function _preconditioners(p, ctx, k)
    # A callable is a factory receiving this solve's problem and orientation.
    pair = applicable(p, ctx) ? p(ctx) : p
    if pair isa NamedTuple
        all(key -> key in (:left, :right), keys(pair)) ||
            throw(ArgumentError("precond factory must return an operator or (; left, right)"))
        return get(pair, :left, nothing), get(pair, :right, nothing)
    end
    return pair, nothing
end
_state_residual(z, u, f, like, offs, st) = IL._joint_residual(u, f, z, like, offs, st)
function IL._ift_prepare(k::IL.KrylovIFT, ad, f, u, z, like, offs, st, transposed, history=nothing)
    ctx = map(DI.Constant, (u, f, like, offs, st))
    product = if transposed
        prep = DI.prepare_pullback(_state_residual, ad, z, (zero(z),), ctx...)
        v -> only(DI.pullback(_state_residual, prep, ad, z, (v,), ctx...))
    else
        prep = DI.prepare_pushforward(_state_residual, ad, z, (zero(z),), ctx...)
        v -> only(DI.pushforward(_state_residual, prep, ad, z, (v,), ctx...))
    end
    A = ResidualOperator{eltype(z),typeof(product)}(length(z), product)
    factor = f isa IL._InputProblem ? f.factor : f
    solved = f isa IL._InputProblem ? IL.inchannel(factor) : IL.statechannel(factor)
    problem = (; operator=A, transposed, factor, solved, joint=u, root=z, template=like, offsets=offs, state=st, history)
    M, N = _preconditioners(k.precond, problem, k)
    return KrylovSystem(A, k, M, N)
end
function _preconditioner_action(M, b, ldiv)
    (M === nothing || M === I) && return nothing
    buffer = similar(b)
    apply! = ldiv ? LinearAlgebra.ldiv! : LinearAlgebra.mul!
    return function (v)
        apply!(buffer, M, v)
        return buffer
    end
end

# GMRES already restarts from the true residual. If a retained Broyden inverse
# causes a stall, try up to two correction solves within the same iteration budget.
function _refine_broyden(apply, x, b, report, alg, tol, left, right)
    scale = _primal_norm(b)
    for _ in 1:2
        remaining = alg.maxiters - report.iterations
        report.reason === :stalled && remaining > 0 || break
        r = b .- apply(x)
        dx, correction = adjoint_solve(apply, r, GMRES(restart=alg.restart, maxiters=remaining);
            tol, left, right)
        x .+= dx
        residual = _primal_norm(b .- apply(x))
        status = _status(residual, scale, tol)
        iterations = report.iterations + correction.iterations
        if status === :continue
            status = correction.reason === :nonfinite ? :nonfinite :
                     iterations == alg.maxiters ? :maxiters : :stalled
        end
        report = _report(status, iterations, residual, report.nfe + correction.nfe + 2)
    end
    return x, report
end

function IL._ift_solve(s::KrylovSystem, b)
    k = s.options
    maxiters = iszero(k.itmax) ? 2length(b) : k.itmax
    alg = GMRES(restart=k.memory; maxiters)
    tol = Tolerance(atol=k.atol, rtol=k.rtol)
    apply = s.operator.product
    left, right = _preconditioner_action(s.left, b, k.ldiv), _preconditioner_action(s.right, b, k.ldiv)
    x, report = adjoint_solve(apply, b, alg; tol, left, right)
    if k.precond isa IL.BroydenIFT && report.reason === :stalled
        x, report = _refine_broyden(apply, x, b, report, alg, tol, left, right)
    end
    solved = report.converged
    k.on_solve === nothing || k.on_solve((; report.iterations, report.residual, solved))
    solved || throw(ArgumentError(
        "DEQ Krylov IFT did not converge: $(report.reason); residual=$(report.residual)"))
    return x
end
end
