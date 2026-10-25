# [AlgebraicFactors](@id algebraicfactors)

```@meta
CurrentModule = AlgebraicFactors
```

Finite toric potentials over named discrete channels, implementing L-SPEC-004 §0 and
§1.1–§1.7. Messages, energies and entropies use exact enumeration; fitting uses
Newton or binary-feature IPF. Later algebraic families are not implemented.

```julia
using AlgebraicFactors, LenticulumCore, Mycelium, Random

names, alphabets = (:a, :b), (2, 3)
A = independence(names, alphabets)   # a design matrix, not a factor
f = ToricFactor(names, alphabets, A)
ps, st = setup(Random.Xoshiro(1), f)
p = Polarity(a = Unobserved(), b = Observed())
site, st = factor_message(f, :a, p, (b = DiracBelief(2),), nothing, ps, st)
```

Joint states are ordered with the first variable fastest, as in `CartesianIndices`.
Default labels are `1:alphabet`; custom labels are a tuple of unique-label vectors.
Diracs match actual labels. Unlabelled categoricals are positional; labelled categoricals
are matched and reordered by label. Shared graph variables need consistent canonical label
order because Mycelium combines outgoing categorical messages positionally.

`hierarchical` builds a marginal-cell indicator block for each generator, named by tuples
of channels or indices. Both matrix builders retain redundant statistics. Parameters are
`(η = …,)`, initially zero; the factor retains redundant statistics. Fitting returns
the minimum Euclidean norm representative modulo constant logits and null directions.

`factor_message` excludes the target prior and target input. Latent channels enter with
unit weight, and the factor supports every observed/latent combination around one target.
This is a direct-message factor: it provides no assembled lens, joint latent posterior
or forward model. The common conformance checker uses independent site and pooled-posterior
fixtures through its direct-message path.

`local_free_energy` computes the joint factor expectation minus joint entropy from the
incoming cavity messages. Mycelium supplies the variable entropy counting correction.
Set `normalised = true` for standalone game quantities: `free_energy(f, π, y, ps, st)`
and `LenticulumCore.entropy(f, π, y, ps, st)`. Use disjoint NamedTuples of prior beliefs
and observations, or one prior belief and observations naming every other channel.

`fit_mle(f, counts; method = :newton, tol = 1e-10, maxiter = 1000)` returns
`(η, p, converged, iterations, residual, existence)`. Counts can be a joint-state vector
or an array of size `alphabets`. `p` is a vector; the parameter can be used with
`ps = (η = fit.η,)`.
Newton removes the gauge by an SVD of centred logits; `:ipf` requires a 0/1 design and
cycles over individual rows. At target moments zero or one, IPF uses finite log-odds
steps of -32 or +32 to approximate the infinite endpoint update.

`converged` means the infinity norm of the moment residual is at most `tol`.
`existence == :exists` certifies an ordinary MLE only when every count is positive.
Otherwise it is `:suspected_boundary` when the centred logit norm has grown from zero
to at least 10 and `length(p)*minimum(p) <= 1e-8`, or `:unknown`. The boundary status
is a heuristic, not proof of non-existence; it can accompany moment convergence.

`inmodel(f, p; atol = 1e-8)` requires strictly positive probabilities and tests the
Euclidean least-squares residual of log probabilities in the augmented row space
`[1'; A]`. `binomial_residuals(f, p, moves)` permits zero probabilities and checks
supplied integer moves in the exact kernel of `[1'; A]` before evaluating their
binomials. No Markov basis is computed. Both check probability normalisation with
`atol=1e-12, rtol=1e-8`, separately from the membership tolerance.

Positive independence tables and a decomposable chain are checked against their
closed forms by both methods; the tests also cover an arbitrary integer design,
likelihood finite differences, the gauge, empty-row extended MLEs and independence
minors. The implementation note `lib/AlgebraicFactors.jl/src/fit.md` gives the
numerical thresholds and limitations.

Enumeration costs grow with the product of alphabet sizes. This implementation uses CPU
arrays and makes no GPU or automatic differentiation guarantee.

```@index
Modules = [AlgebraicFactors]
```

```@autodocs
Modules = [AlgebraicFactors]
```
