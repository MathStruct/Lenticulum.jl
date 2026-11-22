module LenticulumCoreChainRulesCoreExt
using LenticulumCore
using ChainRulesCore
ChainRulesCore.@non_differentiable LenticulumCore._trainable_indices(ps, freeze)
ChainRulesCore.@non_differentiable LenticulumCore.offset_tree(ps)
ChainRulesCore.@non_differentiable LenticulumCore._flatlen(ps)
ChainRulesCore.@non_differentiable LenticulumCore._nearly_symmetric(M)
ChainRulesCore.@non_differentiable LenticulumCore.isproper(b)
# The decompositions of an extended Gaussian are of values: their outputs are constants of the
# constant-rank formulas, which carry the derivative (`extended_gaussian.jl`, P4; before P4 the
# basis `_complement` was held fixed, P23).
ChainRulesCore.@non_differentiable LenticulumCore._values(X)
ChainRulesCore.@non_differentiable LenticulumCore._value_eigen(S)
ChainRulesCore.@non_differentiable LenticulumCore._value_svd(M)
ChainRulesCore.@non_differentiable LenticulumCore._check_psd(S, what)
ChainRulesCore.@non_differentiable LenticulumCore._scale(λ)
ChainRulesCore.@non_differentiable LenticulumCore._logsum(λ)
ChainRulesCore.@non_differentiable LenticulumCore._check_constant_rank(X, Δ, scale, what)
# d(M⁻¹) = -M⁻¹ dM M⁻¹ (P23: `inv(::Cholesky)` has no Zygote rule). `_chol` factors (M + Mᵀ)/2, so
# only the symmetric part of the tangent reaches the result.
function ChainRulesCore.rrule(::typeof(LenticulumCore._spd_inv), M, name)
    P = LenticulumCore._spd_inv(M, name)
    function spd_inv_pullback(dP)
        D = unthunk(dP)
        D isa AbstractZero && return NoTangent(), ZeroTangent(), NoTangent()
        G = -(P' * D * P')
        return NoTangent(), (G + G') / 2, NoTangent()
    end
    return P, spd_inv_pullback
end
function ChainRulesCore.rrule(::typeof(LenticulumCore._normalised_logweights), weights)
    y = LenticulumCore._normalised_logweights(weights)
    function back(dy)
        D = unthunk(dy)
        D isa AbstractZero && return NoTangent(), ZeroTangent()
        all(>(0),weights) || throw(ArgumentError(
            "normalised log-weight derivatives require strictly positive weights; support-boundary derivatives are undefined"))
        # ∂ log(wᵢ/Σw)/∂wⱼ = δᵢⱼ/wᵢ − 1/Σw, so the pullback is (Dⱼ − pⱼ ΣD)/wⱼ with p = w/Σw.
        # Form the cancelling numerator first: ΣD/Σw overflows for tiny w (LP-019 item 7).
        # Where pⱼ underflows (wⱼ ≪ max w) that loses 1/Σw, so use Dⱼ/wⱼ − ΣD/Σw there, which
        # is then safe because Σw ≥ max w is not small (LP-019 review).
        # The rule is linear in D: scale D by its largest entry so that ΣD cannot overflow,
        # and multiply back at the end (LP-019 third review).
        Dm = maximum(abs, D)
        iszero(Dm) && return NoTangent(), zero(D) ./ weights
        Ds = D ./ Dm
        m = maximum(weights)
        s = sum(weights ./ m)
        p = (weights ./ m) ./ s
        SD = sum(Ds)
        tiny = p .< floatmin(float(eltype(weights)))
        return NoTangent(), Dm .* ifelse.(tiny, Ds ./ weights .- (SD / m) / s, (Ds .- p .* SD) ./ weights)
    end
    return y, back
end
# θ rebuilt into the parameter tree (I1): the cotangent of each leaf is added into its slice of
# one vector. Zygote through `_rebuild` mapped over the tree's tuples with its generic adjoints,
# compiled for every tree type (a graph-wide tree changes type with every structure: 3.7 s of the
# 7.4 s that a Zygote replay compiled on a new 31-factor chain). On the top-level call only: a rule
# on `_rebuild` would apply at every level and give a cotangent as long as θ per subtree.
function ChainRulesCore.rrule(::typeof(LenticulumCore._unflatten), θ, offs, like)
    y = LenticulumCore._unflatten(θ, offs, like)
    function unflatten_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero && return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        # The element type the plain pullbacks give: θ's promoted with the cotangents' (review of
        # I1, Codex finding 3).
        T = _cotangent_type(eltype(θ), offs, Δ)
        dθ = fill!(similar(θ, T), 0)
        _scatter!(dθ, offs, Δ)
        return NoTangent(), dθ, NoTangent(), NoTangent()
    end
    return y, unflatten_pullback
end
# I30, step 2: the same rebuild, boxed. The pullback unboxes the cotangent of the box (a
# `Tangent{_Erased}`, a `NamedTuple` `(tree = Δ,)`, or a zero) and calls `_unflatten`'s rule above.
# The call goes through a barrier so that the pullback closure's captures are inferred as `Any`,
# not as the tree's type: the rule is compiled once for all structures.
function ChainRulesCore.rrule(::typeof(LenticulumCore._unflatten_erased), θ, offs::LenticulumCore._Erased, like::LenticulumCore._Erased)
    y, back = _unflatten_barrier(θ, offs.tree, like.tree)
    function unflatten_erased_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero && return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        d = _erased_cotangent(Δ)
        return NoTangent(), back(d)[2], NoTangent(), NoTangent()
    end
    return LenticulumCore._Erased(y), LenticulumCore._ErasedPullback(unflatten_erased_pullback)
end
@noinline function _unflatten_barrier(θ, @nospecialize(offs), @nospecialize(like))
    return ChainRulesCore.rrule(LenticulumCore._unflatten, θ, offs, like)
end
# The cotangent of a box: `Tangent{_Erased}`, a `NamedTuple` `(tree = Δ,)`, or the tree's own
# cotangent when the caller handed it over directly.
function _erased_cotangent(@nospecialize(Δ))
    Δ = unthunk(Δ)
    Δ isa AbstractZero && return Δ
    Δ isa Tangent && (Δ = ChainRulesCore.backing(Δ))
    if Δ isa NamedTuple && length(Δ) == 1 && haskey(Δ, :tree)
        return unthunk(Δ.tree)
    end
    return Δ
end
# `@nospecialize`: the walks are compiled once, not per tree type.
function _cotangent_type(T, @nospecialize(o), @nospecialize(Δ))
    Δ = unthunk(Δ)
    (Δ === nothing || Δ isa AbstractZero) && return T
    o isa Int && return promote_type(T, eltype(Δ))
    d = _backing(Δ)
    for i in 1:length(o)
        T = _cotangent_type(T, o[i], _component(d, o, i))
    end
    return T
end
function _scatter!(dθ, @nospecialize(o), @nospecialize(Δ))
    Δ = unthunk(Δ)
    (Δ === nothing || Δ isa AbstractZero) && return nothing
    if o isa Int
        view(dθ, o:(o + length(Δ) - 1)) .+= (Δ isa Number ? Δ : vec(Δ))
        return nothing
    end
    d = _backing(Δ)
    for i in 1:length(o)
        _scatter!(dθ, o[i], _component(d, o, i))
    end
    return nothing
end
_backing(@nospecialize(Δ)) = Δ isa Tangent ? ChainRulesCore.backing(Δ) : Δ
# The cotangent of field `i`: by position when the cotangent has the tree's fields in order (the
# usual case, so that a wide tree is not searched by name per field: review of I1, Codex finding 5),
# by name when it has some fields only (a `Tangent` built with keywords).
function _component(@nospecialize(d), @nospecialize(o), i)
    o isa NamedTuple || return d[i]
    d isa NamedTuple && length(d) == length(o) && keys(d)[i] === keys(o)[i] && return d[i]
    k = keys(o)[i]
    return hasproperty(d, k) ? getproperty(d, k) : nothing
end
ChainRulesCore.rrule(::typeof(stop_gradient), x) =
    (x, _ -> (NoTangent(), ZeroTangent()))
function ChainRulesCore.rrule(::typeof(belief_cov), b::GaussianBelief)
    C = belief_cov(b)
    function back(dC)
        dC = unthunk(dC)
        dC isa AbstractZero && return NoTangent(), ZeroTangent()
        # Forward evaluation inverts the symmetric part of Λ, so its cotangent is symmetric too.
        G = -(C' * dC * C')
        return NoTangent(), Tangent{typeof(b)}(;η=ZeroTangent(),Λ=(G + G') / 2)
    end
    return C, back
end
# The fused `_mean_cov` differentiates as its two parts would (I18).
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode},
                             ::typeof(LenticulumCore._mean_cov), b::GaussianBelief)
    μ, back_μ = rrule_via_ad(config, belief_mean, b)
    Σ, back_Σ = rrule_via_ad(config, belief_cov, b)
    function back(dy)
        dy = unthunk(dy)
        dy isa AbstractZero && return NoTangent(), ZeroTangent()
        dμ, dΣ = unthunk(dy[1]), unthunk(dy[2])
        gμ = dμ isa AbstractZero ? ZeroTangent() : back_μ(dμ)[2]
        gΣ = dΣ isa AbstractZero ? ZeroTangent() : back_Σ(dΣ)[2]
        return NoTangent(), gμ + gΣ
    end
    return (μ, Σ), back
end
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode},
                             ::typeof(LenticulumCore.straight_through), f, s, x)
    _, back = rrule_via_ad(config, s, x)
    return f(x), dy -> (NoTangent(), NoTangent(), NoTangent(), back(dy)[2])
end
end
