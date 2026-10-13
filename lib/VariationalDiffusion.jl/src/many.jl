# ---------------------------------------------------------------------------
# Many queries at once: B independent queries with the same polarity, solved together so that
# every network call evaluates all of them (one call per noise level per iteration, with
# `batched = true`), instead of B solves that each call the network on their own.
#
# Each query keeps its own Newton iteration, step size, stability test and stopping rule; the
# answers are the ones `implicit_infer` gives query by query. What is shared is the network
# call. The free-coordinate Jacobian blocks come from |F| input JVPs (or VJPs) of the whole
# batch per noise level, one per free coordinate, so the cost grows with the number of free
# coordinates, not with the dimension: the shape of classification (ten free label coordinates,
# 784 clamped pixels).
#
# See `implicit.md` §7.
# ---------------------------------------------------------------------------

# the node inputs of one noise level for every column of Z: n × (B·|grp|), query-major blocks
function _many_inputs(m::ImplicitDiffusion, Z, grp)
    α, σ, _ = _node_coefs(m, m.nodes.t[first(grp)])
    n, B = size(Z)
    X = similar(float.(Z), n, B * length(grp))
    for (a, k) in enumerate(grp)
        X[:, ((a - 1) * B + 1):(a * B)] .= α .* Z .+ σ .* view(m.nodes.ε, :, k)
    end
    return X
end

_levels(m::ImplicitDiffusion) = (ts = m.nodes.t; [findall(==(u), ts) for u in unique(ts)])

# the field of every column of Z
function _field_many(m::ImplicitDiffusion, Z, ps, st)
    G = zero(float.(Z))
    B = size(Z, 2)
    for grp in _levels(m)
        t = m.nodes.t[first(grp)]
        _, _, λc = _node_coefs(m, t)
        E, st = _eps_columns(m, _many_inputs(m, Z, grp), _node_times(m, t), ps, st)
        for (a, k) in enumerate(grp)
            G .+= (m.nodes.w[k] .* λc) .* (view(E, :, ((a - 1) * B + 1):(a * B)) .- view(m.nodes.ε, :, k))
        end
    end
    return (G, st)
end

# directional derivatives of ε for every column: one network derivative call when batched
function _eps_dir_columns(m::ImplicitDiffusion, X, t, ps, st, V; transpose::Bool)
    p = m.predictor
    f(x, v) = transpose ? epsilon_vjp_input(p, x, t, ps, st, v) : epsilon_jvp_input(p, x, t, ps, st, v)
    m.batched && return reshape(f(X, V), size(X))
    out = similar(float.(X))
    for b in axes(X, 2)
        out[:, b] .= vec(f(X[:, b], V[:, b]))
    end
    return out
end

# the free-coordinate Jacobian block of every column: |F| × |F| × B. Column j of each block is a
# JVP along e_j; with an AD backend the VJP along e_j gives row j instead, exactly. With
# `batched` the |F| directions go into one derivative call per noise level, as |F| copies of the
# inputs side by side: one call of |F| times the width instead of |F| calls, which is what an
# accelerator wants (a call costs little more than its transfer).
function _jacobian_many(m::ImplicitDiffusion, Z, free, ps, st)
    idx = findall(free)
    nF, B = length(idx), size(Z, 2)
    J = zeros(eltype(float(Z)), nF, nF, B)
    nF == 0 && return J
    rows = m.predictor.ad !== nothing                 # VJPs give rows
    for grp in _levels(m)
        t = m.nodes.t[first(grp)]
        α, _, λc = _node_coefs(m, t)
        c = _rowcol(λc, α, idx)                       # λ_i α_j on the free block
        τ = _node_times(m, t)
        X = _many_inputs(m, Z, grp)
        N = size(X, 2)                                # B·|grp| columns per direction
        if m.batched
            V = zeros(eltype(X), size(X, 1), N * nF)
            for (a, j) in enumerate(idx)
                V[j, ((a - 1) * N + 1):(a * N)] .= 1
            end
            Dall = _eps_dir_columns(m, repeat(X, 1, nF), τ, ps, st, V; transpose = rows)
        end
        for (a, j) in enumerate(idx)
            D = if m.batched
                view(Dall, :, ((a - 1) * N + 1):(a * N))
            else
                V = zero(X); V[j, :] .= 1
                _eps_dir_columns(m, X, τ, ps, st, V; transpose = rows)
            end
            for (g, k) in enumerate(grp), b in 1:B
                col = view(D, idx, (g - 1) * B + b)
                rows ? (J[a, :, b] .+= m.nodes.w[k] .* _crow(c, a) .* col) : (J[:, a, b] .+= m.nodes.w[k] .* _ccol(c, a) .* col)
            end
        end
    end
    return J
end
# the factor λ_i α_j of row a (from a VJP) or of column a (from a JVP) of the block
_crow(c::Number, a) = c
_crow(c, a) = view(c, a, :)
_ccol(c::Number, a) = c
_ccol(c, a) = view(c, :, a)

"""
    implicit_infer_many(m::ImplicitDiffusion, Z₀::AbstractMatrix, ρ, ps, st; Z_init = Z₀,
                        tol = default_tolerance(m, Z₀, ps), maxiters = 200, step = 0.05, method = :auto,
                        radius = nothing) -> (Vector{ImplicitSolution}, st)

[`implicit_infer`](@ref) for many queries at once: column `b` of `Z₀` is a query (its clamps
and soft targets), all with the same precisions `ρ`, warm-started from column `b` of `Z_init`.
Returns one solution per column, the same ones `implicit_infer` returns query by query (to
the solver's tolerance).

Every iteration evaluates the field of all unfinished queries in one network call per noise
level (with `m.batched`; otherwise one call per node input), and their free-coordinate
Jacobian blocks with one input JVP per free coordinate and noise level (a VJP when the
predictor has an AD backend). Each query takes its own Newton or descent step, backtracks on its
own residual, and leaves the batch when it converges. The Jacobian blocks are dense, so this is
for queries with few free coordinates, such as classification (the image clamped, ten label
coordinates free); for many free coordinates solve the queries one by one with
[`KrylovSolve`](@ref).

`method` is that of [`implicit_infer`](@ref): by default (`:auto`) the trust-region method for a
relation with an energy, each query with its own radius and acceptance test, and
Newton-or-descent otherwise. The trial points of all queries are evaluated together as well
(residual and energy, one call per noise level each).
"""
function implicit_infer_many(m::ImplicitDiffusion, Z₀::AbstractMatrix, ρ, ps, st; Z_init = Z₀,
                             tol = nothing, maxiters::Integer = 200, step = 0.05, method::Symbol = :auto,
                             radius = nothing)
    method in (:auto, :newton, :trust_region) ||
        throw(ArgumentError("method must be :auto, :newton or :trust_region, got :$method"))
    method === :auto && (method = _has_energy(m) ? :trust_region : :newton)
    tol === nothing && (tol = default_tolerance(m, Z₀, ps))
    method === :trust_region && return _infer_many_trust_region(m, Z₀, ρ, ps, st; Z_init, tol, maxiters, radius)
    size(Z_init) == size(Z₀) || throw(DimensionMismatch("Z_init is $(size(Z_init)), Z₀ is $(size(Z₀))"))
    length(ρ) == size(Z₀, 1) || throw(DimensionMismatch("ρ has $(length(ρ)) entries for $(size(Z₀, 1)) coordinates"))
    free = .!isinf.(ρ)
    P = Diagonal(ρ[free] .^ 2)
    Z = float.(copy(Z_init))
    Z[.!free, :] .= Z₀[.!free, :]
    B = size(Z, 2)
    resid(Zs, cols) = (G = first(_field_many(m, Zs, ps, st)); G[free, :] .+ ρ[free] .^ 2 .* (Zs[free, :] .- Z₀[free, cols]))
    sols = Vector{ImplicitSolution{Vector{eltype(Z)},eltype(Z)}}(undef, B)
    active = collect(1:B)
    for it in 0:maxiters
        isempty(active) && break
        Za = Z[:, active]
        R = resid(Za, active)
        nr = vec(sqrt.(sum(abs2, R; dims = 1)))
        J = _jacobian_many(m, Za, free, ps, st)
        stable = [_stable(J[:, :, b] .+ P) for b in eachindex(active)]
        done = [nr[b] ≤ tol * (1 + norm(Za[:, b])) for b in eachindex(active)]
        last = it == maxiters
        for (b, q) in enumerate(active)
            (done[b] || last) && (sols[q] = ImplicitSolution(Z[:, q], nr[b], it, done[b], stable[b]))
        end
        keep = findall(b -> !done[b], eachindex(active))
        (last || isempty(keep)) && break
        active, Za, R, nr, J, stable = active[keep], Za[:, keep], R[:, keep], nr[keep], J[:, :, keep], stable[keep]
        # Newton where stable, descent elsewhere; backtracking on each query's own residual
        D = similar(R)
        for b in eachindex(active)
            D[:, b] .= stable[b] ? -((J[:, :, b] .+ P) \ R[:, b]) : -step .* R[:, b]
        end
        η = ones(length(active))
        pending = findall(stable)
        while !isempty(pending) && any(>(1e-8), η[pending])
            Zt = Za[:, pending]
            Zt[free, :] .+= η[pending]' .* D[:, pending]
            nt = vec(sqrt.(sum(abs2, resid(Zt, active[pending]); dims = 1)))
            ok = nt .< (1 .- 1e-4 .* η[pending]) .* nr[pending]
            η[pending[.!ok]] ./= 2
            pending = pending[.!ok .& (η[pending] .> 1e-8)]
        end
        for (b, q) in enumerate(active)
            Z[free, q] .+= η[b] .* D[:, b]
        end
    end
    return (sols, st)
end

# The trust-region method of `_infer_trust_region` (DenseSolve) for every column at once: the
# same steps, radii and acceptance per query, with the residuals, Jacobian blocks and merits of
# all unfinished queries (and of their trial points) evaluated together.
function _infer_many_trust_region(m::ImplicitDiffusion, Z₀, ρ, ps, st; Z_init, tol, maxiters, radius)
    size(Z_init) == size(Z₀) || throw(DimensionMismatch("Z_init is $(size(Z_init)), Z₀ is $(size(Z₀))"))
    length(ρ) == size(Z₀, 1) || throw(DimensionMismatch("ρ has $(length(ρ)) entries for $(size(Z₀, 1)) coordinates"))
    free = .!isinf.(ρ)
    nF = count(free)
    P = Diagonal(ρ[free] .^ 2)
    Z = float.(copy(Z_init))
    Z[.!free, :] .= Z₀[.!free, :]
    T = eltype(Z)
    B = size(Z, 2)
    energetic = _has_energy(m)
    εa = _arith_eps(m, Z₀, ps)
    resid(Zs, cols) = (G = first(_field_many(m, Zs, ps, st)); G[free, :] .+ ρ[free] .^ 2 .* (Zs[free, :] .- Z₀[free, cols]))
    function merit(Zs, Rs, cols)
        energetic || return vec(sum(abs2, Rs; dims = 1)) ./ 2
        U = first(_energy_many(m, Zs, ps, st))
        return U .+ vec(sum(abs2, ρ[free] .* (Zs[free, :] .- Z₀[free, cols]); dims = 1)) ./ 2
    end
    Δ = fill(T(radius === nothing ? 0.1 * sqrt(max(nF, 1)) : radius), B)
    R = resid(Z, 1:B)
    f = merit(Z, R, 1:B)
    sols = Vector{ImplicitSolution{Vector{T},T}}(undef, B)
    active = collect(1:B)
    for it in 0:maxiters
        isempty(active) && break
        Za = Z[:, active]
        Ra = R[:, active]
        nr = vec(sqrt.(sum(abs2, Ra; dims = 1)))
        J = _jacobian_many(m, Za, free, ps, st)
        stable = [_stable(J[:, :, b] .+ P) for b in eachindex(active)]
        stop = [nr[b] ≤ tol * (1 + norm(Za[:, b])) || it == maxiters || Δ[q] < 10εa * (1 + norm(Za[:, b]))
                for (b, q) in enumerate(active)]
        for (b, q) in enumerate(active)
            stop[b] && (sols[q] = ImplicitSolution(Z[:, q], nr[b], it, nr[b] ≤ tol * (1 + norm(Za[:, b])), stable[b]))
        end
        keep = findall(!, stop)
        isempty(keep) && break
        active, Za, Ra, nr, J = active[keep], Za[:, keep], Ra[:, keep], nr[keep], J[:, :, keep]
        # each query's step within its own radius, from its own model
        Pst = similar(Ra)
        pred = zeros(T, length(active))
        for (b, q) in enumerate(active)
            Jb = J[:, :, b] .+ P
            g, H = energetic ? (Ra[:, b], Symmetric((Jb .+ Jb') ./ 2)) : (Jb' * Ra[:, b], Symmetric(Jb' * Jb))
            p = _tr_step(DenseSolve(), g, H, Δ[q])
            Pst[:, b] .= p
            pred[b] = -(dot(g, p) + dot(p, H * p) / 2)
        end
        Zt = copy(Za); Zt[free, :] .+= Pst
        Rt = resid(Zt, active)
        ft = merit(Zt, Rt, active)
        for (b, q) in enumerate(active)
            noisy = pred[b] ≤ 100 * εa * max(abs(f[q]), one(T))
            ratio = noisy ? (norm(view(Rt, :, b)) < nr[b] ? one(T) : zero(T)) : (f[q] - ft[b]) / pred[b]
            np = norm(view(Pst, :, b))
            if ratio < 0.25
                Δ[q] = np / 4
            elseif ratio > 0.75 && np ≥ 0.99Δ[q]
                Δ[q] = 2Δ[q]
            end
            if ratio > 1e-4
                Z[:, q] .= view(Zt, :, b); R[:, q] .= view(Rt, :, b); f[q] = ft[b]
            end
        end
    end
    return (sols, st)
end

# the relation's energy U of every column of Z, one network call per noise level when batched
function _energy_many(m::ImplicitDiffusion, Z, ps, st)
    s = m.predictor.schedule
    B = size(Z, 2)
    U = zeros(eltype(float(Z)), B)
    for grp in _levels(m)
        t = m.nodes.t[first(grp)]
        _, _, λc = _node_coefs(m, t)
        E, st = _energy_columns(m, _many_inputs(m, Z, grp), t, ps, st)
        for (a, k) in enumerate(grp)
            Ek = view(E, ((a - 1) * B + 1):(a * B))
            U .+= m.nodes.w[k] .* ((m.λ * sigma(s, t)^2 / alpha(s, t)^2) .* Ek .- vec((λc .* view(m.nodes.ε, :, k))' * Z))
        end
    end
    return (U, st)
end

_energy_columns(m::ImplicitDiffusion{<:NoisePredictor{<:GaussianMixtureEps}}, X, t, ps, st) =
    ([-mixture_logdensity(m.predictor.model, x, _mixture_times(m, t), ps) for x in eachcol(X)], st)
function _energy_columns(m::ImplicitDiffusion{<:NoisePredictor{<:EnergyNetwork}}, X, t, ps, st)
    p = m.predictor
    t = _node_times(m, t)
    m.batched && return energy_columns(p, X, t, ps, st)
    E = zeros(eltype(float(X)), size(X, 2))
    for b in axes(X, 2)
        E[b], st = energy(p, X[:, b], t, ps, st)
    end
    return (E, st)
end

"""
    implicit_classify(m::ImplicitDiffusion, Z₀::AbstractMatrix, ρ, ps, st; label, classes,
                      labels = nothing, radius = nothing, unique_tol = 1e-3, kwargs...)
        -> (Vector{answer}, st)

[`implicit_classify`](@ref) for many queries at once, the columns of `Z₀` (all with the same
`ρ`). Every query is started from every class code, and all starts of all queries are solved
together by [`implicit_infer_many`](@ref) (`kwargs` go to it); the Laplace covariances and the
energies of the roots are computed in batches as well. Each answer is what the single-query
method returns: the same roots, weights and class probabilities.
"""
function implicit_classify(m::ImplicitDiffusion, Z₀::AbstractMatrix, ρ, ps, st; label, classes::AbstractMatrix,
                           labels = nothing, radius = nothing, temperature = nothing, cov_scale = 1, unique_tol = 1e-3,
                           kwargs...)
    idx = _indices(label)
    length(idx) == size(classes, 1) || throw(DimensionMismatch(
        "$(length(idx)) label coordinates, class codes of length $(size(classes, 1))"))
    all(!isinf, ρ[idx]) || throw(ArgumentError("the label coordinates must not be hard-clamped"))
    free = .!isinf.(ρ)
    B, K = size(Z₀, 2), size(classes, 2)
    starts = repeat(float.(Z₀), 1, K)                                  # start (b, k) is column (k - 1)B + b
    for k in 1:K
        starts[idx, ((k - 1) * B + 1):(k * B)] .= classes[:, k]
    end
    sols, st = implicit_infer_many(m, repeat(Z₀, 1, K), ρ, ps, st; Z_init = starts, kwargs...)
    Zr = reduce(hcat, [r.z for r in sols])
    J = _jacobian_many(m, Zr, free, ps, st)
    energetic = _has_energy(m)
    U = energetic ? first(_energy_many(m, Zr, ps, st)) : zeros(size(Zr, 2))
    soft = free
    pos = cumsum(free)[idx]                                            # the label among the free coordinates
    answers = map(1:B) do b
        keep = Int[]
        for k in 1:K
            c = (k - 1) * B + b
            (sols[c].converged && sols[c].stable) || continue
            any(j -> norm(Zr[:, j] .- Zr[:, c]) ≤ unique_tol * (1 + norm(Zr[:, c])), keep) && continue
            push!(keep, c)
        end
        isempty(keep) && throw(ArgumentError("query $b: no stable answer found"))
        comps = map(keep) do c
            Σ = inv(Symmetric((J[:, :, c] .+ J[:, :, c]') ./ 2 .+ Diagonal(ρ[free] .^ 2)))
            hld = energetic ? LinearAlgebra.logdet(Symmetric(Σ)) / 2 : 0.0
            E = U[c] + sum(abs2, ρ[soft] .* (Zr[soft, c] .- Z₀[soft, b])) / 2
            (LenticulumCore.Gaussian(Zr[idx, c], Matrix(Symmetric(Σ[pos, pos]))), energetic ? -E + hld : 0.0)
        end
        logw = [c[2] for c in comps]
        mix = LenticulumCore.MixtureBelief([c[1] for c in comps], exp.(logw .- maximum(logw)))
        (categorical_answer(mix, classes; labels, radius, temperature, cov_scale)..., mixture = mix)
    end
    return (answers, st)
end
