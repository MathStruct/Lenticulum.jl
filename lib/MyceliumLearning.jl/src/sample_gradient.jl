# The gradient averaged over the samples of a `Mycelium.SampleBuffer` (Monte Carlo EM).
# See `sample_gradient.md` and Mycelium's `sampling.md`.

"""
    sample_gradient(buffer, sampler, ps, st; ad = nothing, ε = 1e-6, through_inference = false)
        -> gradient

The weighted average, over the stored samples, of the gradient of each sample's pinned total
``-\\log p(\\text{data}, \\text{pins})`` at the **current** parameters `ps`. With samples from the
posterior and an exact pinned graph (a tree), this is the Monte Carlo EM gradient of
``-\\log p(\\text{data})``. Samples drawn under older parameters bias it, which is what the buffer's
`horizon` limits.

Each sample's points are pinned and belief propagation runs the sampler's schedule at `ps`. Then:

- by default, [`free_energy_gradient`](@ref LenticulumCore.free_energy_gradient) is taken with the
  factor beliefs held fixed (LP-020): one inference per sample, exact when that inference has
  converged, as on a tree. A factor without `factor_belief` gets the fixed-message derivative.
- with `through_inference = true`, the scalar free energy is differentiated *through* the
  inference instead: central differences of step `ε` for `ad = nothing`, or the given AD backend.
  That costs an inference per sample and, with differences, per parameter coordinate, and needs no
  factor hook. It is the reference the default is checked against.

Factors that are not learnable get zero gradients. The result has `ps`'s structure.
"""
function sample_gradient(b::SampleBuffer, sampler::GibbsSampler, ps, st; ad = nothing, ε::Real = 1e-6,
                         through_inference::Bool = false)
    g = sampler.graph
    w = sample_weights(b)
    acc = nothing
    like = nothing
    for (e, wi) in zip(b.entries, w)
        wi == 0 && continue
        grad = through_inference ?
            LenticulumCore._parameter_gradient(ad, _pinned_total, ps, sampler, e.pins, st; ε) :
            _pinned_gradient(ps, sampler, e.pins, st; ad, ε)
        θ = LenticulumCore._flatten(_mask_unlearnable(grad, g))
        acc = acc === nothing ? wi .* θ : acc .+ wi .* θ
        like = grad
    end
    return LenticulumCore._unflatten(acc, like)
end

# Inference with the pins at `ps`, then the graph gradient at the fixed factor beliefs.
function _pinned_gradient(ps, sampler::GibbsSampler, pins, st; ad, ε)
    g = sampler.graph
    s = MessageStore(g)
    merge!(s.pins, pins)
    _, st = propagate!(s, g, sampler.schedule, ps, st)
    return LenticulumCore.free_energy_gradient(s, g, ps, st; ad, ε, schedule = sampler.schedule)
end

# −log p(data, pins): inference with the pins at `ps`, then the scalar free energy.
function _pinned_total(ps, sampler::GibbsSampler, pins, st)
    g = sampler.graph
    s = MessageStore(g)
    merge!(s.pins, pins)
    _, st = propagate!(s, g, sampler.schedule, ps, st)
    return first(LenticulumCore.scalar_free_energy(s, g, ps, st))
end

function _mask_unlearnable(grad, g::FactorGraph)
    names = Tuple(k for k in parameter_keys(g) if haskey(grad, k))
    sub = map(names) do n
        f = _key_node(g, n)
        p = getfield(grad, n)
        θ = LenticulumCore._flatten(p)
        LenticulumCore._unflatten(_mask_gradient(θ, _active_indices(f, p)), p)
    end
    return merge(grad, NamedTuple{names}(sub))
end
