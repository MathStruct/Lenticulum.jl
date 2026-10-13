# ---------------------------------------------------------------------------
# What a query costs: root finding against the diffusion classifier.
#
# Both read the same network. Root finding pays per iteration, per field node, one ε evaluation
# and one input derivative per free coordinate (the Jacobian block), plus the trust region's
# energy at the trial point; the diffusion classifier pays one ε evaluation per class and noise
# draw. Counting the passes and weighting them by their measured cost per column predicts the
# time per query of each, and so which is cheaper:
#
#     root finding   S [(I+1) N (|F| c_d + c_ε) + (I+1) N c_E]      (trust region)
#                    S [(I+1) N (|F| c_d + c_ε) + I N b c_ε]         (Newton-or-descent, b backtracks)
#     classifier     K M c_ε
#
# with S starts, I iterations, N field nodes, |F| free coordinates, K classes and M draws. The
# first grows with the free coordinates, the second with the classes. See `implicit.md` §12.
# ---------------------------------------------------------------------------

"""
    pass_costs(pred::NoisePredictor, X, t, ps, st; directions = 1, reps = 3)
        -> (epsilon, energy, derivative)

The measured cost, in seconds per column, of the three passes a query makes through the
network, at the batch width of `X` (one column per input, as the batched solvers call it):

- `epsilon`: ``\\varepsilon_\\theta`` ([`epsilon`](@ref));
- `energy`: the energy of an [`EnergyNetwork`](@ref) (`nothing` for other predictors);
- `derivative`: one input derivative per column, measured on `directions` copies of `X` side
  by side as [`implicit_infer_many`](@ref) stacks the free coordinates: a VJP with an AD backend
  (rows of the Jacobian), a JVP without.

Each pass runs once to compile, then `reps` times; the fastest is kept. With a device the
transfer of the result back is included, which is also where the device synchronises.
"""
function pass_costs(pred::NoisePredictor, X::AbstractMatrix, t, ps, st; directions::Integer = 1, reps::Integer = 3)
    W = size(X, 2)
    timed(f) = (f(); minimum(@elapsed(f()) for _ in 1:reps))
    tε = timed(() -> epsilon(pred, X, t, ps, st))
    tE = pred.model isa EnergyNetwork ? timed(() -> energy_columns(pred, X, t, ps, st)) : nothing
    Xd = repeat(X, 1, directions)
    V = zero(float.(Xd)); V[1, :] .= 1
    der = pred.ad === nothing ? epsilon_jvp_input : epsilon_vjp_input
    td = timed(() -> der(pred, Xd, t, ps, st, V))
    return (epsilon = tε / W, energy = tE === nothing ? nothing : tE / W, derivative = td / (W * directions))
end

"""
    root_cost(m::ImplicitDiffusion, nfree, units; iterations, starts = 1, method = :auto, backtracks = 1)
        -> seconds per query

The predicted time per query of root finding with [`implicit_infer_many`](@ref): `starts` solves
of `iterations` iterations each, with `nfree` free coordinates and the relation's field nodes,
from the per-column `units` of [`pass_costs`](@ref). Per iteration and node, one ε evaluation and
`nfree` input derivatives (the Jacobian block, once more at the end for the stability test);
the trust region (`method`, as in [`implicit_infer`](@ref)) adds the energy at each trial point,
Newton-or-descent `backtracks` trial residuals per iteration. The iterations are an input: measure
them on a few queries, they vary little across queries of one kind.
"""
function root_cost(m::ImplicitDiffusion, nfree::Integer, units; iterations::Real, starts::Integer = 1,
                   method::Symbol = :auto, backtracks::Real = 1)
    method === :auto && (method = _has_energy(m) ? :trust_region : :newton)
    method in (:newton, :trust_region) || throw(ArgumentError("method must be :auto, :newton or :trust_region"))
    N = length(m.nodes.t)
    I = iterations
    c = (I + 1) * N * (nfree * units.derivative + units.epsilon)
    if method === :trust_region
        units.energy === nothing && throw(ArgumentError("the trust region needs the energy's cost (units.energy)"))
        c += (I + 1) * N * units.energy
    else
        c += I * N * backtracks * units.epsilon
    end
    return starts * c
end

"""
    classifier_cost(nclasses, draws, units) -> seconds per query

The predicted time per query of the diffusion classifier (Li et al. 2023) on the same network:
one ε evaluation per class and noise draw, ``K M c_\\varepsilon``, from the `units` of [`pass_costs`](@ref).
"""
classifier_cost(nclasses::Integer, draws::Integer, units) = nclasses * draws * units.epsilon

"""
    break_even_classes(m::ImplicitDiffusion, nfree, units; draws, kwargs...) -> K

The number of classes above which root finding with `nfree` free coordinates is predicted to be
cheaper than the diffusion classifier with `draws` noise draws per class: `root_cost(m, nfree,
units; kwargs...) / (draws · units.epsilon)`. Root finding's cost grows with the free
coordinates, the classifier's with the classes. With one-hot labels (`nfree` = K) the crossing
may never come; with a label code of fixed dimension it always does, at some K.
"""
break_even_classes(m::ImplicitDiffusion, nfree::Integer, units; draws::Integer, kwargs...) =
    root_cost(m, nfree, units; kwargs...) / (draws * units.epsilon)
