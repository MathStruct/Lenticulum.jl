# Probabilistic Types

> "Probabilistic type" names at least three different things. Only one of them is what this
> project is doing, and it is not the one people usually mean.
>
> The short answer: [[README]]'s central definition
> ``(x_1,\ldots,x_n) \in R_\theta :\Longleftrightarrow r_\theta(x_1,\ldots,x_n) \approx 0``
> **is a graded type judgment**, and the energy is the grade.

## 1. Three readings

| reading | the type is | the tradition |
|---|---|---|
| **(1) types of probabilistic things** | ``\mathrm{Dist}[X]`` — a distribution over ``X`` | the Giry monad; probabilistic programming |
| **(2) probabilistic judgments** | ``p(a : T)`` — membership is *graded*, not boolean | Cooper, Dobnik, Lappin & Larsson (TTR) |
| **(3) graded / quantitative types** | ``x :_r T`` — a semiring element annotates the judgment | Atkey's quantitative type theory; graded monads |

These are genuinely different. (1) keeps the judgment boolean and makes the *thing* probabilistic.
(2) keeps the thing ordinary and makes the *judgment* probabilistic. (3) generalises (2) to an
arbitrary semiring of grades.

## 2. Reading (1): the probability monad, and why it does not fit

The standard semantics of a probabilistic program is the **Giry monad** ``\mathcal{P}`` on
measurable spaces: `return` is the Dirac, `bind` is integration, and the Kleisli category is
the category of Markov kernels. A sampling program is a Kleisli morphism; `Dist[X]` is
``\mathcal{P}X``.

The synthetic version — the axioms without the measure theory — is a **Markov category**
(Fritz), which [[Acausal Composition is a Hypergraph Category]] §5 already uses. And that note
already contains the reason this reading fails here:

> A Markov category has **copy** and **delete**, and delete is *natural* because every kernel
> is normalised. What it does not have is **merge**: you cannot ask that two probability wires
> be equal.

`Mycelium.combine` is exactly that missing merge. It is the Frobenius multiplication, it is
unnormalised, and it is the operation the whole framework is built on. So:

> [!important] `AbstractBelief` is not the probability monad
> It looks like ``\mathcal{P}X`` and it is not. `combine : \mathcal{P}X \times \mathcal{P}X \to
> \mathcal{P}X` is not a monad operation and cannot be one — normalisation is precisely what
> obstructs it.
>
> The type `AbstractBelief` is closer to *unnormalised measure*, or to *linear relation*, than
> to *distribution*. `GaussianBelief` makes this literal: with singular ``\Lambda`` it is not a
> distribution at all, and `isproper` is the runtime check for which of the two you have.

This is why Lenticulum is not a probabilistic programming language and why comparing it to one
misleads. A PPL types programs in the Kleisli category of a monad; this types *relations*, and
relations do not form one.

## 3. Reading (2): the one that fits

Cooper, Dobnik, Lappin and Larsson give a probabilistic formulation of **Type Theory with
Records** in which the judgment itself is graded: instead of ``a : T`` holding or not, one has
``p(a : T) \in [0,1]``. Types become classifiers with confidence rather than sets with
membership, and the paper frames this as an interface between *classifying situations according
to types* and compositional semantics.

That framing transfers directly. A factor **is** a classifier of configurations — it says how
well a tuple of values satisfies it — and the energy is the confidence.

Now read [[README]]'s definition again:

$$(x_1,\ldots,x_n) \in R_\theta \quad:\Longleftrightarrow\quad r_\theta(x_1,\ldots,x_n) \approx 0$$

The ``\approx`` is doing all the work. Membership in the relation is not decided; it is
**scored**, by the residual. And the score is exactly the energy:

$$x : R_\theta \quad\text{to degree}\quad \exp\bigl(-E_\theta(x)\bigr)$$

That is a graded type judgment, written in this project's own notation before anyone called it
one. [[Energy-Based Learning]] says the same thing from the modelling side — an energy is a
*score* on configurations, never a boolean — and the two statements are the same statement.

Three consequences worth drawing out.

**The framework has two kinds of judgment, and only one is graded.**

| judgment | about | graded? | in code |
|---|---|---|---|
| ``x : R_\theta`` — this configuration satisfies this relation | a **value** | **yes**, by the energy | `energy(f, …)` |
| this factor can be run this way round | a **type** | no, boolean | `supports_polarity(f, p)` |

The second is a well-formedness condition on the type, checked once. The first is the model.
Keeping them apart is what stops "the relation is approximately satisfied" from collapsing into
"the model is approximately well-formed", which are different problems with different remedies.

**Composition of grades is addition of energies.** [[Energy-Based Factor Graphs]] §1:
``E = \sum_c E_c``. A conjunction of graded judgments has, as its grade, the sum of the grades —
which is what makes a factor graph a *proof* of a graded judgment about the whole
configuration, assembled from local ones.

**The free energy is the grade of the whole graph.** And the fact that it coincides with
``-\log p(\text{data})`` on the tractable fragment ([[The Linear Gaussian Chain]] §4) says the
grade is not arbitrary: on that fragment it is exactly the log-evidence.

## 4. Reading (3): the grade lives in a semiring

Quantitative type theory (Atkey) annotates each variable in a context with an element of a
**semiring**, and the semiring decides what the annotation means — ``\{0,1,\omega\}`` for usage
counting, ``\mathbb{N}`` for multiplicity, and so on. Graded monads (Katsumata and others) do
the same for effects: a computation is indexed by an element of a monoid recording what it did.

The semiring is exactly the right structure for this project, and the vault has already been
using one without saying so:

| | operation | in the vault |
|---|---|---|
| combine grades along a conjunction | ``+`` on energies | ``E = \sum_c E_c`` |
| combine grades along an alternative | ``\min`` or ``\log\!\sum\exp`` | the semiring choice |

And [[Energy-Based Factor Graphs]] §3 shows the two choices are the endpoints of a temperature:
sum-product at ``T = 1``, min-sum at ``T = 0``, with
``-T\log\sum e^{-E_i/T} \to \min_i E_i``.

> [!important] The temperature is a grade, and it is currently untracked
> That note records a real defect: point-valued factors operate at ``T = 0`` inside a form that
> is implicitly ``T = 1``, and a graph mixing them adds incommensurable quantities with nothing
> checking.
>
> In type-theoretic terms this is an **ungraded composition**. If beliefs carried their semiring
> as an index — ``\mathrm{Belief}_S X`` — then mixing would be a type error rather than a silent
> one, and the fix would be forced rather than remembered. See
> [[The Type Discipline of a Factor Graph]] §3.

## 5. What normalisation costs, said as a type discipline

Worth stating once, because it is the same fact from a third direction.

In a Markov category, `delete` is natural — every morphism is *total*, which is the
type-theoretic statement that every program returns something. Normalisation is what buys that.
Dropping it (as [[Energy-Based Learning]] §2 argues one should) means morphisms are
sub-probabilistic or unnormalised: the "computation" may have mass less than one, and the
missing mass is evidence.

That is the setting of **partial Markov categories** and of the Gaussian-relations work already
cited in [[Acausal Composition is a Hypergraph Category]] §4. It is also, informally, why
`combine` returns something you must renormalise later and why `logpartition` is carried
separately: the type says "unnormalised", and the normaliser is a second component that
composition must track by hand.

## 6. Summary

- **Not** a probabilistic programming language: beliefs are not the probability monad, because
  the merge operation the whole framework rests on is not a monad operation.
- **Yes** a graded type theory in reading (2): membership in a relation is scored by an energy,
  which is what ``r_\theta \approx 0`` has meant all along.
- **The grade lives in a semiring**, and which semiring is the temperature — currently a real,
  recorded, untracked source of error.

What follows from this for the code — which errors the type domain has already caught, and
which it is currently catching at runtime instead — is [[The Type Discipline of a Factor Graph]].

## Sources

- Giry, *A categorical approach to probability theory*, 1982 — the probability monad.
- Fritz, *A synthetic approach to Markov kernels, conditional independence and theorems on
  sufficient statistics*, 2020 — Markov categories, and why delete's naturality forces
  normalisation.
- Cooper, Dobnik, Lappin & Larsson, *Probabilistic Type Theory and Natural Language
  Semantics*, Linguistic Issues in Language Technology **10**(4), CSLI, 2015 —
  [ACL Anthology](https://aclanthology.org/2015.lilt-10.4/). A probabilistic formulation of
  Type Theory with Records (TTR), in which judgments ``a : T`` carry probabilities rather than
  truth values.
- Atkey, *Syntax and Semantics of Quantitative Type Theory*, LICS 2018 — semiring-annotated
  judgments.
- Katsumata, *Parametric effect monads and semantics of effect systems*, POPL 2014,
  pp. 633–646 — graded monads.
- Stein & Samuelson, *A Category for Unifying Gaussian Probability and Nondeterminism*, CALCO
  2023 — the completion in which improper beliefs are first-class.

Related: [[The Type Discipline of a Factor Graph]], [[Energy-Based Learning]],
[[Energy-Based Factor Graphs]], [[Acausal Composition is a Hypergraph Category]],
[[Implicit Learners]], [[Three Senses of Implicit]], [[Scalar and Multivariate Energy]]
