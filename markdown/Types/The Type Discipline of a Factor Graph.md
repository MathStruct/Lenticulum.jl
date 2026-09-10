# The Type Discipline of a Factor Graph

> The companion to [[Probabilistic Types]]. That note asks what kind of type theory this
> project is doing; this one asks what its **types are actually doing in the code** — which
> errors the type domain has already caught, and which it is catching at runtime instead
> because Julia cannot express them.
>
> The finding: several of the vault's recorded gaps are the same gap. **They are typing
> problems, stated in other vocabulary.**

## 1. Neither linear nor cartesian

Standard type disciplines are classified by what you may do with a variable:

| discipline | duplicate? | discard? | merge? |
|---|---|---|---|
| linear | no | no | no |
| affine | no | yes | no |
| cartesian (ordinary functional) | yes | yes | no |
| **Frobenius / hypergraph** | **yes** | **yes** | **yes** |

[[Acausal Composition is a Hypergraph Category]] establishes that a variable node of degree
``d`` is a Frobenius spider: the wire splits to ``d`` factors and their claims are merged back.
So a factor-graph variable is duplicated *and* merged, and **no standard type discipline
covers that**. Linear logic forbids the duplication; the cartesian setting has no merge to
forbid.

The spider theorem also says something pleasingly deflationary about what the "type" of a
junction is:

> Any two connected wirings with the same legs are equal. So a junction's type is nothing but
> its **connectivity** — a partition of the legs, with no internal structure to distinguish.

Which is the formal content of "a variable is a wire and has no content of its own"
([[Everything is a Factor]]), stated as a typing rule rather than as a design principle.

## 2. What the types already do — and the bug they caught

Julia offers parametric types and multiple dispatch, and the project uses them to put *names*
in types:

| type | what is in the type parameter |
|---|---|
| `Polarity{names}` | the channel names of a factor |
| `Channel{name,S}` | the channel's name and space |
| `GradedEnergySpace{names,T}` | the factor names indexing an energy's summands |
| `AbstractLenticulumContainerFactor{factors}` | the sub-factor names a parameter tree is built from |

The `Polarity` case is worth dwelling on, because [[Messages are Inversions]] records that it
caught a real bug:

> [!important] The type domain caught what the value domain would have swallowed
> When computing a message, the target channel must not also be counted among the observed
> ones. Because `Polarity{names}` carries the channel names in its *type*, "this channel has
> two polarities" is a **construction error**. Had the polarity been a `Dict`, one assignment
> would have silently overwritten the other and the factor would have conditioned on its own
> prediction — producing a plausible wrong answer rather than an error.
>
> That is the argument for static typing, arrived at empirically in this repository. The note
> even concedes the performance justification for the type-domain representation "turned out to
> be the lesser reason for it."

## 3. Where the type system is doing the work at runtime instead

This is the payoff. Five recorded gaps, all of them typing problems.

### 3.1 `combine` is partial — that is a type error deferred to runtime

`Mycelium.combine` is exact addition for Gaussians, dominance for Diracs, and **throws for
everything else**. `messages.md` §1 calls this "the package's main gap" and it is right, but the
description is incomplete: the set of combinable pairs is a *relation on types*, and it is
currently checked dynamically by method lookup and an error branch.

A discipline that indexed beliefs by what they support would make
`combine(SampleBelief, SampleBelief)` a compile-time error, and would make the *conditions* for
a new belief type explicit rather than discovered by the first crash.

There is a sharper version. The very first defect fixed in this repository was a **multiple
dispatch ambiguity** in `combine`: two methods, neither more specific than the other, on a pair
of beliefs. Symmetric multiple dispatch is being used as a poor man's type system for a partial
binary operation on a type lattice, and that is exactly the situation it handles worst —
ambiguity rather than a decision. It failed loudly only because the catch-all threw; had it
returned a value, the resolution would have been silent and order-dependent.

### 3.2 `isproper` is a refinement type, checked at runtime

`GaussianBelief` with singular ``\Lambda`` is not a distribution — it is a linear relation
([[Probabilistic Types]] §2). One type is carrying two things, and `isproper` sorts them out
dynamically. `belief_mean` then throws on the wrong one, which is the right behaviour and is
also a runtime check standing in for a refinement.

The honest reading: `GaussianBelief` should be indexed by properness, and `belief_mean` should
only accept the proper index. Julia cannot say that.

### 3.3 The semiring mismatch is an ungraded composition

[[Energy-Based Factor Graphs]] §3.2 records that point-valued factors run at ``T = 0`` inside a
free energy that is implicitly ``T = 1``, and a graph mixing them sums incommensurable
quantities **with nothing checking**.

That is a missing type index, exactly. If a belief carried its semiring — ``\mathrm{Belief}_S X``
— mixing would not typecheck, and the choice would be forced at graph construction rather than
remembered by the author. It is the cleanest available example of a typing discipline that
would have prevented a recorded defect rather than merely described it.

### 3.4 `supports_polarity` is a declared predicate that is sometimes decidable

Whether a factor may be run in a given direction is asserted by its author. `channels.jl` is
explicit that the default is conservative and the predicate must be *declared*.

But [[Differential Algebra and DAE Factors]] establishes that for differential-algebraic
families this is **structural identifiability**, which is *decidable* by differential
elimination. So for a real class of factors the predicate is not a matter of the author's
judgement but a computation — and a decidable predicate is a candidate for a genuine
refinement type rather than a convention.

This is the strand the PhD proposal calls certification, and it is the one place where the type
discipline would be doing something a comment cannot.

And it has a precedent worth reading rather than re-deriving: **Mercury** is Prolog with modes
and determinism *declared and statically checked*, which is this ambition, solved, for logic
programs — [[Prolog and Logic Programming]] §2.

### 3.5 Latent channels and the exclusion principle are structural, not semantic

`resolve_polarity` marks one channel `Unobserved()`, those with messages `Observed()`, and the
rest `Latent()`. Doing that consistently across every factor at once is a **matching** problem
in the incidence graph ([[ModelingToolkit as an Acausal Relation]] §4), and Mycelium picks the
matching one message at a time without ever checking global consistency.

A graph-level well-formedness condition, checkable before any message is passed, is the natural
home for that — and `validate(g)` is the function that ought to hold it.

## 4. Laxness is an effect system

A separate thread, and the vault has all the pieces without naming the discipline.

An **effect system** annotates a computation with what it does beyond returning a value, and a
**graded monad** makes those annotations compose. Now list what this project already tracks:

| in the code | is an annotation saying |
|---|---|
| `isexact(inversion)` | whether this backward pass is exact or approximate |
| `AbstractGradientCoupling` — `Diagonal`, `Pathwise`, `ScoreFunction`, `Exact` | **which terms of the true Jacobian this edge drops** |
| `islinear(scalarisation)` | whether composing energies is strict or lax |
| `jensen_gap`, `chain_rule_defect` | *how much* laxness |

`AbstractGradientCoupling` is the striking one. Its docstring says Definition 29 composes
gradients block-diagonally while the true Jacobian has two extra blocks, that the paper calls
the assignment "lax" and says it "can be accounted for mechanistically by an implementation" —
and that this type *is* that mechanism, with each edge declaring which correction it applies.

> [!important] That is an effect annotation, per edge
> A per-edge declaration of what a computation drops, intended to compose along the graph, with
> a numeric measure of the discrepancy. The framework has an effect system; it is spelled as a
> collection of traits and there is nothing that composes them.
>
> Today the grades are read *after* the fact. A graded discipline would make "this composite is
> exact" a **type**, and would refuse to combine an exact edge with a lax one without recording
> the result as lax. That is precisely the bookkeeping [[Composition of Gradients]] says an
> implementation must do.

## 5. What Julia can and cannot express

Stated plainly, because it bounds everything above.

**Can:** parametric types (names and shapes in the type domain), multiple dispatch, traits by
dispatch, compile-time specialisation. That is enough for §2, and §2 caught a real bug.

**Cannot:** dependent types, refinement types, effect tracking, linearity or any substructural
discipline, and — critically — it cannot make a *partial* operation total by restricting its
domain at the type level.

So everything in §3 and §4 is currently documentation plus runtime checks. That is not a
failure of the design; it is the ceiling of the language. It is also why the certification
strand points at a proof assistant rather than at more Julia types: the conditions worth
proving (§3.4, §3.5) are decidable properties of a graph, and a language that cannot express
refinement will not check them however carefully the traits are written.

## 6. The consolidated list

Five recorded gaps, one heading:

| gap, as recorded | as a typing problem |
|---|---|
| `combine` is deliberately partial | a partial operation with a dynamically-checked domain |
| improper beliefs need `isproper` | a missing refinement index |
| mixed-temperature graphs are silently wrong | a missing semiring index |
| polarity legality is declared | a decidable refinement left as a convention |
| laxness is tracked but not composed | an effect system with no composition rule |

None of these is news individually — each is written up somewhere in the vault. What is new is
that they are one thing, and that a single question ("what should the index on a belief be?")
touches four of the five.

Related: [[Probabilistic Types]], [[Acausal Composition is a Hypergraph Category]],
[[Messages are Inversions]], [[Energy-Based Factor Graphs]], [[Composition of Gradients]],
[[Scalar and Multivariate Energy]], [[Differential Algebra and DAE Factors]],
[[Everything is a Factor]], [[ModelingToolkit as an Acausal Relation]]
