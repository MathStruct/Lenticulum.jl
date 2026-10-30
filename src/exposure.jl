# ---------------------------------------------------------------------------
# The linear-Gaussian factor with exposed weight and/or bias (L-SPEC-003 §1.2).
#
# With a point input x0, the residual r = y - A x0 - b is linear in (vec A, b, y) jointly:
# A x0 = (x0ᵀ ⊗ I) vec A. So the factor is linear-Gaussian in whichever of those blocks carry
# beliefs, and its messages and Bethe term are exact closed forms. A belief on x as well
# would make it bilinear (x times A), which is not implemented.
#
# The generic wrapper is Mycelium's `expose`; see `exposure.md` beside this file.
# ---------------------------------------------------------------------------

function Mycelium.supports_parameter_polarity(f::GaussianFactor, p::LenticulumCore.Polarity)
    _shadows_blocks(f) && return false
    ks = keys(p)
    all(k -> k in (f.in_channel, f.out_channel, :A, :b), ks) || return false
    any(k -> k in (:A, :b), ks) || return false
    p[f.in_channel] isa LenticulumCore.Observed || return false   # a point input is required
    us = Tuple(k for k in ks if p[k] isa LenticulumCore.Unobserved)
    return length(us) == 1 && only(us) !== f.in_channel
end

# The blocks of r = y - A x0 - b as (key, coefficient in r, belief). A block that is not
# exposed is the point `ps.A` or `ps.b`; an exposed block without a message is `nothing`.
# The belief path keys blocks :A and :b beside the channels, so channels with those names
# would be ambiguous (a channel named :noise, say, is fine).
_shadows_blocks(f::GaussianFactor) = f.in_channel in (:A, :b) || f.out_channel in (:A, :b)

function _exposed_blocks(f::GaussianFactor, msgs, ps)
    _shadows_blocks(f) && throw(ArgumentError(
        "a GaussianFactor with a channel named :A or :b can expose parameters only as points; rename the channel"))
    xb = _get(msgs, f.in_channel)
    xb isa LenticulumCore.DiracBelief || throw(ArgumentError(
        "a GaussianFactor with exposed parameters needs a point input on :$(f.in_channel); \
         with beliefs on both x and A the factor is bilinear (L-SPEC-003 §1.2), which is not implemented"))
    x0 = _vec(xb.value)
    Im = Matrix{Float64}(I, f.nout, f.nout)
    K = LinearAlgebra.kron(x0', Im)
    A = haskey(ps, :A) ? LenticulumCore.DiracBelief(vec(ps.A)) : _get(msgs, :A)
    b = haskey(ps, :b) ? LenticulumCore.DiracBelief(ps.b) : _get(msgs, :b)
    return ((f.out_channel, Im, _get(msgs, f.out_channel)), (:A, -K, A), (:b, -Im, b))
end

# Mean and covariance of C z under a block's belief: only the directions C sees need to be
# proper. A canonical belief with singular Λ is fine when the rows of C lie in its range
# (C Λ⁺ Λ = C); `nothing` when the belief carries no information at all.
function _projected_moments(bel, C, key)
    bel isa LenticulumCore.DiracBelief && return C * _vec(bel.value), zeros(size(C, 1), size(C, 1))
    (bel === nothing || bel isa LenticulumCore.TrivialBelief) && return nothing, nothing
    bel isa GaussianBelief || throw(ArgumentError("unsupported belief $(typeof(bel)) on :$key"))
    # η ≠ 0 with Λ = 0 is a linear site exp(ηᵀz), not "no information": it falls through to
    # the flat-direction test below and is refused where C reads it (LP-019 third review).
    LenticulumCore.isuninformative(bel) && return nothing, nothing
    if isproper(bel)
        m, S = belief_mean(bel), belief_cov(bel)
        return C * m, C * S * C'
    end
    # In Float64 whatever the input precision: eigenvectors with eigenvalue ≤ n·eps·λmax are
    # the flat directions, and C must not see them beyond rounding (64·eps·‖C‖). Anything C
    # sees at all in a flat direction makes Cz improper, so the test is deliberately tight.
    # Equilibrate first, z = D z̃ with D = diag(Λᵢᵢ^{-1/2}) (1 where Λᵢᵢ = 0), so that the
    # flatness cut-off does not depend on the units of other coordinates (LP-019 fifth review).
    Λ = Matrix{Float64}(LenticulumCore._symmetric_part(bel.Λ))
    d = [λ > 0 ? 1 / sqrt(λ) : 1.0 for λ in LinearAlgebra.diag(Λ)]
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(d .* Λ .* d'))
    λmax = maximum(abs, E.values)
    flat = E.values .<= size(C, 2) * eps(Float64) * λmax
    Cf = Matrix{Float64}(C) .* d'                                  # C̃ = C D
    maximum(abs, Cf * E.vectors[:, flat]; init = 0.0) <= 64 * eps(Float64) * maximum(abs, Cf) ||
        throw(ArgumentError("the belief on :$key is improper in directions this factor sees; put a prior on it"))
    V = E.vectors[:, .!flat]
    P = V * LinearAlgebra.Diagonal(1 ./ E.values[.!flat]) * V'   # Λ̃⁺ on the proper directions
    return Cf * (P * (d .* bel.η)), Cf * P * Cf'
end

"""
    Mycelium.parameter_message(f::GaussianFactor, target, polarity, msgs, ps, st)

The exact likelihood message to `target` ∈ (`:A`, `:b`, the output channel) when `A` and/or
`b` are exposed, for a point input ``x_0``. Writing ``r = \\sum_k C_k z_k`` over the blocks
``z = (y, \\operatorname{vec}A, b)`` with ``C = (I, -x_0^\\top \\otimes I, -I)``, the other
blocks' means and covariances give ``s = \\sum_{k \\ne t} C_k m_k`` and
``V = Q + \\sum_{k \\ne t} C_k S_k C_k^\\top``, and the message is
``\\mathcal N^{-1}(-C_t^\\top V^{-1} s,\\ C_t^\\top V^{-1} C_t)``. To `:A` it has rank
`nout` when ``x_0 \\ne 0`` (improper unless `nin = 1`) and rank 0 when ``x_0 = 0``; to the
output it is the predictive
``\\mathcal N(\\bar A x_0 + \\bar b,\\ K S_A K^\\top + S_b + Q)``.

A block without information, whose coefficient has full row rank, leaves the residual
unconstrained, so the message is `TrivialBelief()`; this is exact, not a fallback. An improper
belief on a block is used where the factor sees only its proper directions; otherwise the
message throws.
"""
function Mycelium.parameter_message(f::GaussianFactor, target::Symbol, polarity, msgs, ps, st)
    target === f.in_channel && throw(ArgumentError(
        "a GaussianFactor with exposed parameters cannot send to its input :$target while a \
         parameter carries a belief: the factor is bilinear there (L-SPEC-003 §1.2)"))
    Q = noise_covariance(f, ps)
    s = zeros(f.nout)
    V = Matrix{Float64}(Q)
    Ct = nothing
    for (k, C, bel) in _exposed_blocks(f, msgs, ps)
        if k === target
            Ct = C
            continue
        end
        Cm, CSC = _projected_moments(bel, C, k)
        if Cm === nothing
            iszero(C) && continue                    # x0 = 0: A does not enter
            return (LenticulumCore.TrivialBelief(), st)
        end
        s += Cm
        V += CSC
    end
    Ct === nothing && throw(ArgumentError("channel :$target is not exposed on this GaussianFactor"))
    Vi = inv(_chol(V, "Q plus the propagated covariances"))
    return (GaussianBelief(-(Ct' * (Vi * s)), Ct' * Vi * Ct), st)
end

"""
    Mycelium.parameter_free_energy(f::GaussianFactor, msgs, ps, st)

The Bethe term ``U_c - H(b_c)`` with the factor belief over every block that carries a
belief: ``b_c(z) \\propto \\mathcal N(r(z); 0, Q) \\prod_k \\mu_{k \\to c}(z_k)``, Gaussian in
``z``. Same grading and entropy convention as `local_free_energy` (points carry no
entropy; Gaussians relative to Lebesgue measure).
"""
function Mycelium.parameter_free_energy(f::GaussianFactor, msgs, ps, st)
    Qi, logdetQ = _noise_metrics(f.noise_learning, f.Q, ps)
    c = zeros(f.nout)
    Cs, ηs, Λs = Matrix{Float64}[], Vector{Float64}[], Matrix{Float64}[]
    for (_, C, bel) in _exposed_blocks(f, msgs, ps)
        if bel isa LenticulumCore.DiracBelief
            c += C * _vec(bel.value)
        else
            η, Λ = _canon(bel, size(C, 2))
            push!(Cs, C); push!(ηs, η); push!(Λs, Λ)
        end
    end
    isempty(Cs) && return _graded_free_energy(f, c, zeros(f.nout, f.nout), 0.0, Qi, logdetQ), st
    G = reduce(hcat, Cs)
    Λ = G' * Qi * G + cat(Λs...; dims = (1, 2))
    η = reduce(vcat, ηs) - G' * (Qi * c)
    m, Σ, H = _posterior_statistics(f.noise_learning, η, Λ)
    return _graded_free_energy(f, G * m + c, G * Σ * G', H, Qi, logdetQ), st
end
