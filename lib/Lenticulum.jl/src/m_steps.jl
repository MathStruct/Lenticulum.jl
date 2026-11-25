# ---------------------------------------------------------------------------
# Closed-form M-steps for the constraint factors (`MyceliumLearning.m_step`, `lib/Mycelium.jl/src/em.jl`).
#
# A constraint factor's residual is r = W φ with φ = [x; e_k] (x its continuous channels stacked,
# e_k the outcome's indicator) and W = [A, −B]: E_b[−log f] = ½ tr(Q⁻¹ W S_φ Wᵀ) + const, with
# S_φ = Σ_k [S_k m_k e_kᵀ; e_k m_kᵀ n_k e_k e_kᵀ] from the moment statistics. Minimising over the
# learned columns L of W with the others F fixed (a declared output, frozen paths) is weighted
# least squares, W_L = −W_F S_FL S_LL⁺, the same for every Q; directions S_LL does not see keep
# their old values. With nothing fixed the minimiser is W = 0 (LP-022), which is refused. The
# outcome weights are the masses n_k; `RegimeFactor` solves per outcome and, with learned noise,
# sets Q_k = W_k S_φ,k W_kᵀ / n_k.
#
# See `m_steps.md`.
# ---------------------------------------------------------------------------

# whether `path` (or a prefix of it) is frozen
_frozen_path(frozen, path::Symbol...) = frozen === :all || any(p -> length(p) <= length(path) && Tuple(p) == path[1:length(p)], frozen)

# W_L = −W_F S_FL S_LL⁺ + W_L (I − S_LL S_LL⁺), in place
function _solve_blocks!(W::AbstractMatrix, S::AbstractMatrix, learned::AbstractVector{Bool}, name)
    any(learned) || return W
    fixed = .!learned
    any(fixed) || throw(ArgumentError(
        "m_step: every coefficient of $name is learned, and the M-step of a relation normalised in its residual is A = 0 " *
        "(LP-022); declare an `output` channel or freeze a coefficient"))
    if _device_leaf(W) !== nothing
        # The partitions are host indices. Decompose only the small learned normal block.
        li, fi = findall(learned), findall(fixed)
        SLL = S[li, li]
        P = _host_spectral(LinearAlgebra.pinv, SLL)
        W[:, li] .= -W[:, fi] * S[fi, li] * P + W[:, li] * (_eye(W, eltype(W), length(li)) - SLL * P)
        return W
    end
    SLL = Symmetric(Matrix(S[learned, learned]))
    P = LinearAlgebra.pinv(Matrix(SLL))
    W[:, learned] = -W[:, fixed] * S[fixed, learned] * P + W[:, learned] * (I - SLL * P)
    return W
end

function _normalised_logw(n)
    t = sum(n)
    floor = oftype(float(t), floatmin(typeof(float(LenticulumCore.stop_gradient(t)))))
    return log.(max.(n, floor) ./ max(t, floor))
end

# the column ranges of each channel in x
_channel_cols(dims) = (o = cumsum([0; dims]); [(o[j] + 1):o[j + 1] for j in eachindex(dims)])

function _learned_columns(f, st, frozen, K, outcome_name)
    d = sum(st.dims; init = 0)
    learned = falses(d + K)
    for (j, r) in enumerate(_channel_cols(st.dims))
        c = st.channels[j]
        (c === f.output || _frozen_path(frozen, :A, c)) || (learned[r] .= true)
    end
    _frozen_path(frozen, outcome_name) || (learned[(d + 1):end] .= true)
    return learned
end

function MyceliumLearning.m_step(f::Union{LinearConstraintFactor,SwitchingConstraintFactor}, st::MyceliumLearning.MomentStats, ps, frozen)
    A = _coeffs(f, ps)
    sw = f isa SwitchingConstraintFactor
    device = _first_device((st.S, ps))
    C = sw ? _matrix(_on_device(device, ps.b)) : reshape(_on_device(device, ps.c), :, 1)
    name = sw ? :b : :c
    K = size(C, 2)
    length(st.n) == K || throw(DimensionMismatch("m_step: $(length(st.n)) outcomes in the statistics, $K in the factor"))
    Sxx = dropdims(sum(st.S; dims = 3); dims = 3)
    Sφ = if device === nothing
        [Sxx st.m; st.m' Diagonal(st.n)]
    else
        m = _on_device(device, st.m)
        n = _on_device(device, st.n)
        vcat(hcat(_on_device(device, Sxx), m), hcat(_matrix(m'), _matrix(Diagonal(n))))
    end
    W = hcat((_matrix(_on_device(device, A[c])) for c in st.channels)..., -C) .* one(eltype(Sφ))
    _solve_blocks!(W, Sφ, _learned_columns(f, st, frozen, K, name), nameof(typeof(f)))
    cols = _channel_cols(st.dims)
    d = sum(st.dims; init = 0)
    newA = NamedTuple{keys(ps.A)}(map(c -> W[:, cols[findfirst(==(c), st.channels)]], keys(ps.A)))
    Cn = -W[:, (d + 1):end]
    out = merge(ps, (A = newA,), sw ? (b = Cn,) : (c = vec(Cn),))
    sw && !_frozen_path(frozen, :logw) && (out = merge(out, (logw = _normalised_logw(_on_device(device, st.n)),)))
    return out
end

function MyceliumLearning.m_step(f::RegimeFactor, st::MyceliumLearning.MomentStats, ps, frozen)
    A = _coeffs(f, ps)
    K = f.K
    length(st.n) == K || throw(DimensionMismatch("m_step: $(length(st.n)) outcomes in the statistics, $K in the factor"))
    learned = _learned_columns(f, st, frozen, 1, :b)
    cols = _channel_cols(st.dims)
    d = sum(st.dims; init = 0)
    T = promote_type(eltype(ps.b), eltype(st.S), eltype(st.m), eltype(st.n), f.learn_noise ? eltype(ps.L) : Bool, map(eltype, values(A))...)
    device = _first_device((st.S, ps))
    cast(x) = device === nothing ? Array{T}(x) : _on_device(device, T.(x))
    newA = map(c -> cast(A[c]), NamedTuple{keys(ps.A)}(keys(ps.A)))
    b = cast(ps.b)
    L = f.learn_noise ? cast(ps.L) : nothing
    learn_noise = f.learn_noise && !_frozen_path(frozen, :L)
    for k in 1:K
        n = _entry(st.n, k)
        n > 0 || continue
        Sφ = _on_device(device, _moment_gram(st, k, n))
        W = hcat((_matrix(_on_device(device, A[c][:, :, k])) for c in st.channels)..., -b[:, k])
        _solve_blocks!(W, Sφ, learned, :RegimeFactor)
        for c in keys(ps.A)
            newA[c][:, :, k] .= W[:, cols[findfirst(==(c), st.channels)]]
        end
        b[:, k] .= -W[:, d + 1]
        if learn_noise
            Q = W * Sφ * W' ./ n
            Cq = _symmetric_cholesky(Q; check = false)
            LinearAlgebra.issuccess(Cq) || continue          # a degenerate outcome keeps its noise
            Lk = LenticulumCore._lower_matrix(Cq)
            L[:, :, k] .= LinearAlgebra.tril(Lk, -1) + _matrix(Diagonal(log.(_device_diag(Lk))))
        end
    end
    out = merge(ps, (A = newA, b = b))
    _frozen_path(frozen, :logw) || (out = merge(out, (logw = _normalised_logw(_on_device(device, st.n)),)))
    learn_noise && (out = merge(out, (L = L,)))
    return out
end

# What of a factor besides its parameters enters its m_step's minimiser, so that tied factors pool
# their statistics only when these agree: the fixed output coefficient enters the minimiser (W_F),
# and so does the fixed noise — a common noise precision cancels within one factor's regression but
# not across tied factors with different ones, where the pooled objective is a different weighted
# least squares (GH-16). A learned noise is a parameter, tied through `ps` already, so only the fixed
# one is listed.
MyceliumLearning.m_step_signature(f::Union{LinearConstraintFactor,SwitchingConstraintFactor}) = (f.output, f.output_coeff, f.Q)
MyceliumLearning.m_step_signature(f::RegimeFactor) =
    f.learn_noise ? (f.output, f.output_coeff) : (f.output, f.output_coeff, f.Q)
MyceliumLearning.m_step_signature(f::GaussianFactor) = f.noise_learning === nothing ? f.Q : nothing

# --- The Gaussian factors ---------------------------------------------------------------------

# Learned noise from a covariance estimate: the eigenvalue floor applied as `_noise_metrics` does
# (covariance eigenvalues at least `floor`), then the factor's own coordinates.
function _noise_from_cov(c, Q)
    Q = _matrix(Q)
    c.diagonal && return (noise = (log_precision = -log.(max.(_device_diag(Q), c.floor)),),)
    E = _eigen(Symmetric(Q))
    return _noise_parameters(c, _matrix(Symmetric(E.vectors * Diagonal(max.(E.values, c.floor)) * E.vectors')))
end
_learns_noise(f, frozen) = f.noise_learning !== nothing && f.noise_learning.prior === nothing && !_frozen_path(frozen, :noise)

# y = Ax + b + ε: weighted least squares in A and b with the coefficient of y fixed (identity), and
# with learned noise (no prior) the residual covariance. A noise prior makes the noise's M-step a
# MAP problem that is not done here; the noise then keeps its value.
function MyceliumLearning.m_step(f::GaussianFactor, st::MyceliumLearning.MomentStats, ps, frozen)
    cols = _channel_cols(st.dims)
    ix = findfirst(==(f.in_channel), st.channels); iy = findfirst(==(f.out_channel), st.channels)
    d = sum(st.dims)
    n = _only_entry(st.n)
    Sφ = _moment_gram(st, 1, n)
    device = _first_device((Sφ, ps))
    Sφ = _on_device(device, Sφ)
    T = promote_type(eltype(ps.A), eltype(ps.b), eltype(Sφ))
    W = _zeros_like(Sφ, T, f.nout, d + 1)
    W[:, cols[ix]] .= -_on_device(device, ps.A)
    W[:, cols[iy]] .= _eye(W, T, f.nout)
    W[:, d + 1] .= -_on_device(device, ps.b)
    learned = falses(d + 1)
    _frozen_path(frozen, :A) || (learned[cols[ix]] .= true)
    _frozen_path(frozen, :b) || (learned[d + 1] = true)
    _solve_blocks!(W, Sφ, learned, :GaussianFactor)
    out = merge(ps, (A = -W[:, cols[ix]], b = -W[:, d + 1]))
    _learns_noise(f, frozen) && n > 0 && (out = merge(out, _noise_from_cov(f.noise_learning, W * Sφ * W' ./ n)))
    return out
end

# A Gaussian prior's mean is fixed; with learned noise (no prior) its covariance is E[(x − μ)(x − μ)ᵀ].
function MyceliumLearning.m_step(f::GaussianPrior, st::MyceliumLearning.MomentStats, ps, frozen)
    _learns_noise(f, frozen) || return nothing
    n = _only_entry(st.n)
    n > 0 || return nothing
    device = _first_device((st.S, ps, f.μ))
    μ = _on_device(device, float.(f.μ)); m = _on_device(device, st.m[:, 1])
    Q = (_on_device(device, st.S[:, :, 1]) .- m * μ' .- μ * m' .+ n .* (μ * μ')) ./ n
    return merge(ps, _noise_from_cov(f.noise_learning, Q))
end


function _moment_gram(st, k, n)
    m = st.m[:, k]
    v = _on_device(_device_leaf(m), fill(n, 1, 1))
    return _device_leaf(m) === nothing ? [st.S[:, :, k] m; m' n] :
           vcat(hcat(st.S[:, :, k], m), hcat(_matrix(m'), v))
end
