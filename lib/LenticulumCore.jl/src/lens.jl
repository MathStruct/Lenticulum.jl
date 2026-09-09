# ---------------------------------------------------------------------------
# Bayesian lenses (AutoBayes Defs. 9, 10, 12).
#
# A Bayesian lens is a lens whose `put` is conditioned on a PRIOR rather than on a cached
# forward value. The prior plays the role the linearisation point plays in reverse-mode AD:
#
#     autodiff:  (g ∘ f)*(a, c̄) = f*(a, g*(f(a), c̄))
#     Bayes:     (d ∘ c)'_π      = c'_π ∘ d'_{c_*π}
#
# See `lens.md`.
# ---------------------------------------------------------------------------

"""
    BayesianLens(model, inversion)

The pair ``(c, c')`` of AutoBayes Definition 9.

Nothing constrains `inversion` to be exact. Exact posteriors, amortised encoders,
mean-field families, root-finding solvers and diffusion denoisers are all legal, and their
relative quality is precisely what the free energy measures.
"""
struct BayesianLens{M<:AbstractOpenModel,I<:AbstractInversion} <: AbstractBayesianLens
    model::M
    inversion::I
end

"""
    invert(lens, π::AbstractBelief, y, ps, st) -> (AbstractBelief, st)

Apply ``c'_\\pi`` to the observation `y`, returning a belief over
``X \\times \\llbracket c \\rrbracket``.

Note the return type reconstructs the **latent space too**, not just ``X``. That is what the
next factor upstream will consume, and dropping it breaks the chain rule.
"""
function invert end

"""
    ExactInversion()

``c^\\dagger`` of AutoBayes Definition 10: Bayes' law applied to the model's own kernel.
Available only when the model declares a tractable posterior.

!!! note "Almost-sure caveat"
    Footnote 3 of the paper: inversions may not be fully supported and are defined only up
    to almost-sure equality, so ``(-)^\\dagger`` is only a.s. a pseudofunctor. Numerically
    this means guarding against conditioning on a null set.
"""
struct ExactInversion <: AbstractInversion end
isexact(::ExactInversion) = true

"""
    AmortisedInversion(net)

``c'`` realised by a learned network — a VAE encoder. `net` is an `AbstractLuxLayer`, so the
inversion carries its own parameters, independent of the forward kernel's.

That a factor has *two* independently parametrised halves is the structural reason a factor
cannot be a Lux layer.
"""
struct AmortisedInversion{L} <: AbstractInversion
    net::L
end

"""
    SolverInversion(solver)

``c'`` realised by root-finding on a residual: `Implicit Learners.md`'s algebraic and
equilibrium families. The backward pass is the implicit function theorem, which needs the
**Jacobian of the vector energy** — see `Scalar and Multivariate Energy.md` §6.

A solver that stopped early is simply an inexact inversion, and the loss records the cost.
That is a far better failure mode than a divergent unroll.
"""
struct SolverInversion{S} <: AbstractInversion
    solver::S
end

"""
    ProximalInversion(prox)

``c'`` realised by a proximal operator on an energy — the diffusion family (RED-Diff,
ProxDM). See `lib/VariationalDiffusion.jl` and `ImplicitREDDiff.md`.
"""
struct ProximalInversion{P} <: AbstractInversion
    prox::P
end

"""
    TrivialInversion()

The inversion of a prior ``\\pi : 1 \\nrightarrow X``: there is nothing to infer.

AutoBayes Remark 24 uses exactly this to turn a prior into a statistical game with
``l^\\pi = -\\log p_\\pi`` and ``H^\\pi \\equiv 0``, whose composite with `c` has the true
variational free energy as its loss. The prior is a **factor**, not part of the model.
"""
struct TrivialInversion <: AbstractInversion end
isexact(::TrivialInversion) = true

"""
    ComposedLens(first, second)

``(d, d') \\diamond (c, c')`` of AutoBayes Definition 12. Priors propagate forward by
pushforward; corrections propagate backward by sampling the inversions in reverse order.
"""
struct ComposedLens{A<:AbstractBayesianLens,B<:AbstractBayesianLens} <: AbstractBayesianLens
    first::A   # c : X ↦ Y
    second::B  # d : Y ↦ Z
end

"""
    compose(c::AbstractBayesianLens, d::AbstractBayesianLens)

Build ``d \\diamond c``. Argument order follows the wiring (`c` then `d`), not the
mathematical notation ``d \\diamond c``.
"""
compose(c::AbstractBayesianLens, d::AbstractBayesianLens) = ComposedLens(c, d)

"""
    TensorLens(parts::Tuple)

``(c,c') \\otimes (d,d')`` of AutoBayes Definition 15.

!!! warning "Lossy — this is Remark 16"
    Parallel composition of inversions is **lax**: each branch only ever sees the *marginal*
    of a joint prior, so correlations between branches are discarded and the composite
    inversion is mean-field. The discrepancy is the mutual information between the branches
    (Remark 26). Track and report it; do not pretend it is zero.
"""
struct TensorLens{T<:Tuple} <: AbstractBayesianLens
    parts::T
end
