#definition

> `CategoricalBelief(p)` is a distribution over a **finite set** of states: a vector of
> probabilities, stored as log-probabilities. Pooling two of them is elementwise multiplication
> and renormalisation: exact, associative, commutative, and always defined unless their
> supports are disjoint. `bernoulli(p)` is the two-state case.

> Sources: code: `open_model.jl` (the type), `messages.jl` and `free_energy.jl` (its operations)
>
> Theory (CT-ML wiki): [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

## Where it comes from

- **Discrete variables**: "is this enzyme active?" ([[Metabolomics and Proteomics]]), a
  match outcome (win, draw, loss), a regime switch, a token at one position of a text
  ([[Language Models]] §3).
- **The switch of a mixture**: the categorical variable that says which component of a
  [[Mixture Belief]] a value came from.
- **Logic factors**: implication, AND, OR between Bernoulli variables are factors whose
  messages are categorical beliefs ([[Belief Algebra]] §2; the factors themselves are not built).

## How it behaves

| operation | result |
|---|---|
| `combine` with a categorical over the same states | elementwise product, renormalised; disjoint supports are an error |
| `combine` with a `TrivialBelief` / `DiracBelief` | the other / the Dirac |
| `combine` with a `SampleBelief` of states | the samples, reweighted by the probabilities |
| `belief_logdensity(b, k)` | ``\log p_k``; a label instead of an index if the belief has labels |
| `variable_entropy` | Shannon entropy ``-\sum_k p_k\log p_k`` (with ``0\log 0 = 0``) |
| `belief_distance` | total variation ``\tfrac12\sum_k\lvert p_k - q_k\rvert`` |
| `damp` | the convex combination of the probability vectors |

It is the **easy** belief type: the blocker that pooling general beliefs needs densities does
not arise, because on a finite set the density is a table. The cost is size: one number per
state, which is fine for a switch and large for a vocabulary.

Related: [[Beliefs]], [[Belief Algebra]], [[Mixture Belief]], [[Dirac Belief]], [[Sample Belief]]
