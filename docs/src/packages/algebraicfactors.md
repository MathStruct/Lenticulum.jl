# [AlgebraicFactors](@id algebraicfactors)

```@meta
CurrentModule = AlgebraicFactors
```

Finite toric potentials over named discrete channels, implementing L-SPEC-004 §0 and
§1.1–§1.5. Messages, energies and entropies use exact enumeration. Fitting and the later
algebraic families are not implemented.

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
`(η = …,)`, initially zero; no gauge reduction is performed.

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

Enumeration costs grow with the product of alphabet sizes. This implementation uses CPU
arrays and makes no GPU or automatic differentiation guarantee.

```@index
Modules = [AlgebraicFactors]
```

```@autodocs
Modules = [AlgebraicFactors]
```
