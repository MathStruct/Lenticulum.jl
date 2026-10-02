#definition

> `TrivialBelief()` is the belief that carries **no information**: the unit of
> [[Beliefs|`combine`]]. It is also, literally, the unique belief on the one-point space $1$.

> Sources: code: `open_model.jl`, `messages.jl`
>
> Theory (CT-ML wiki): [Terminal Object](https://mathstruct.org/CategoryTheory-ML-Wiki/Terminal-Object) · [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) · [Monoid](https://mathstruct.org/CategoryTheory-ML-Wiki/Monoid)

## Two readings, one object

1. **The unit of pooling.** `combine(TrivialBelief(), b) == b` for every belief `b`. An edge
   whose message has not been computed yet carries it, so that a variable's marginal is the
   product of the messages that do exist.
2. **The belief on the one-point space.** $\mathcal P 1 = \{\ast\}$: there is exactly one
   distribution on a point. A factor with no input ($X \cong 1$, a prior) receives this as its
   prior argument; in AutoBayes' terms, a prior is an open model $1 \to X$.

In canonical Gaussian form it is $(\eta, \Lambda) = (0, 0)$, `uninformative(n)`, the improper
flat density, which is why the two readings agree: adding $(0, 0)$ changes nothing.

`belief_distance` of two Trivial beliefs is $0$, and `isexact(TrivialBelief()) == true`.

Related: [[Beliefs]], [[Gaussian Belief]], [[Dirac Belief]], [[Open Models and Latent Channels]]
