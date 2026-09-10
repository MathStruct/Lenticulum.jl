# Prolog and Logic Programming

> **"The Prolog of machine learning"** — how apt is that, and where does it fall short?
>
> Apter than it sounds in three places, one of them exact. And the shortfall is a single
> specific thing: **Prolog programs are schemas; factor graphs are ground instances.**
>
> The slogan is also, in its most literal reading, already taken — see §7.

## 1. The scorecard

| Prolog has | here |
|---|---|
| relations, not functions | ✓ the whole design |
| **modes** — which arguments are in/out, per call | ✓ **`Polarity`**, and this is exact |
| unification as the merge operation | ≈ `combine`, closely |
| a declarative/procedural split | ✓ and it fails in the same way |
| execution in the **boolean** semiring | ✓ same framework, different semiring |
| **quantification, recursion, schemas** | ✗ — the real gap |
| enumeration of all solutions by backtracking | ✗ |
| negation as failure | ✗ |

## 2. Modes are polarities, exactly

This is the sharpest correspondence and it is not an analogy.

A Prolog predicate has no fixed direction: `append(X, Y, Z)` can be called as
`append([1,2], [3], Z)` to concatenate, or as `append(X, Y, [1,2,3])` to enumerate every way of
splitting a list. Which arguments are bound at the call site is the **mode**, and asking whether
a predicate can be run in a given mode is **mode analysis**.

That is `supports_polarity(f, p)`, with the same meaning and the same role. The vault's
[[Channels and Polarity]] could be rewritten in Prolog's vocabulary almost without loss.

> [!important] Mercury is the precedent for what `supports_polarity` should become
> [[The Type Discipline of a Factor Graph]] §3.4 records that polarity legality is currently
> *declared by the factor author* and ought to be *decided*. The logic-programming community
> hit that exactly and answered it: **Mercury** is Prolog with a **statically checked mode
> system** and **determinism declarations** (`det`, `semidet`, `nondet`, …), verified at compile
> time rather than trusted.
>
> Mercury's contribution over Prolog is precisely the ambition of that §3.4 — so if the
> question is "what would a well-typed version of this look like", the answer has existed since
> the 1990s and it is worth reading rather than re-deriving.

Determinism is the second half and it maps too: Mercury's `det` vs `nondet` is
`isunidirectional` vs a relation that branches — and [[Branches and the Discriminant]] is where
the vault studies the branch locus.

## 3. Unification is close to `combine`

Both operations pool constraints from two sources about the same object, and the
correspondence runs surprisingly deep:

| Prolog | here |
|---|---|
| unbound variable | `TrivialBelief` — the unit of `combine` |
| ground term | `DiracBelief` |
| partially instantiated term | a proper belief with finite precision |
| unification | `combine` |
| **unification failure** | `combine` **throwing** on contradictory `DiracBelief`s |
| most general unifier — least commitment | the pooled belief |

That fifth row is the pleasing one. `messages.jl` throws on two disagreeing Diracs with the
message *"two hard clamps in contradiction is a wiring error"* — which is unification failure,
described in different words and treated the same way.

Where it differs: unification is **exact and idempotent**; `combine` is graded and, for
Gaussians, *sharpening* (precisions add, so combining a belief with itself is not that belief —
see [[Acausal Composition is a Hypergraph Category]] §7's note that the special Frobenius axiom
fails). Prolog's idempotence is a consequence of its boolean semiring, which is §4.

## 4. Prolog is the boolean semiring of the same framework

This is the technical statement that makes the comparison precise rather than suggestive.

[[Energy-Based Factor Graphs]] §3 sets up two semirings and shows they are endpoints of a
temperature: sum-product at ``T=1``, min-sum at ``T=0``. There is a **third**, and it is
Prolog's:

| semiring | combine / eliminate | is |
|---|---|---|
| ``(\vee, \wedge)`` | disjunction / conjunction | **constraint satisfaction — Prolog** |
| ``(\min, +)`` | minimise / add | energy minimisation — min-sum |
| ``(+, \times)`` | marginalise / multiply | probability — sum-product |

Dechter's **bucket elimination** is the unifying framework: one algorithm parameterised by the
operators, covering constraint satisfaction, constraint optimisation and probabilistic
inference. Bistarelli, Montanari and Rossi's **semiring-based constraint solving** makes the
same point from the constraint side — hard CSP, weighted CSP, fuzzy and probabilistic
constraints are one framework with different semirings.

> So logic programming and this project are not neighbouring ideas. They are **the same
> algorithm at different semirings**, and the vault's own temperature story ([[Energy-Based
> Factor Graphs]] §3) already contains the machinery to say so — it simply stopped at two.

That also explains §3's asymmetry. Unification is idempotent because ``\wedge`` is idempotent;
`combine` is not because ``+`` is not. Nothing is wrong with either — it is the semiring
showing through.

## 5. Declarative versus procedural — and both break the same way

Prolog's oldest lesson is that a clause has two readings: a **declarative** one (a logical
implication, order-independent) and a **procedural** one (what SLD resolution will actually do,
order-dependent). Left-recursive clauses are declaratively fine and procedurally
non-terminating, and half of learning Prolog is learning where the two come apart.

[[Depth in Implicit Learning]] §5 draws exactly this distinction here without naming the
precedent:

| the graph is | reading | order-dependent? |
|---|---|---|
| a hypothesis class — which relations hold | **declarative** | no |
| an elimination order — how to compute | **procedural** | yes |

And the failure modes rhyme. Prolog's left recursion loops forever; a loopy factor graph fails
to converge, or converges to exact means with wrong variances ([[Loopy Message Passing]]). In
both cases the *declarative* content is fine and the *procedural* strategy is what broke.

## 6. The real shortfall: schemas versus ground instances

Here is where the slogan overreaches, and it is one thing rather than several.

A Prolog program is a **schema over an unbounded domain**:

```prolog
append([],     Y, Y).
append([H|T],  Y, [H|Z]) :- append(T, Y, Z).
```

Two clauses define a relation on lists of *any* length. Variables range over an unbounded
universe; recursion generates an unbounded family of ground relations; and the *structure* of
the answer is produced by unification rather than fixed in advance.

A factor graph has a **fixed, finite** set of variables and factors, built by hand before
inference starts. There is no quantifier, no recursion, no way to write "for every adjacent pair
of poses, this relation holds" as a schema that instantiates itself against the data.

> [!important] This is the same gap as three others the vault has recorded
> - [[The Structural Gap to ModelingToolkit]] §4: subsystems are not first-class; MTK's
>   `@component` instantiates a resistor twenty times and Mycelium cannot.
> - [[Acausal Composition is a Hypergraph Category]] §6: the decorated-cospan formalisation of
>   subgraph-as-factor, unimplemented — with `oapply` in Catlab as the existing answer
>   ([[Related Julia Projects]] §9).
> - [[GANs as Two Factors]] §2: two factor nodes cannot share one parameter set, so a schema
>   instantiated twice cannot tie its weights.
>
> All four are **"there is no language for generating graphs, only for writing them down"**,
> and Prolog is the most ambitious version of what such a language could be.

Two levels of ambition are worth separating. MTK-style components are *macro* instantiation —
finitely many copies, known before you start. Prolog's recursion is *unbounded and
data-dependent* — the graph depends on the input. The second is much harder, and if it were
ever wanted, the relevant technology is **lifted inference**: doing message passing on the
first-order structure without grounding it out.

## 7. Also missing, and worth being clear about

**Enumeration.** Prolog returns *all* solutions by backtracking. Message passing returns one
belief or one argmin — sum-product gives marginals, min-sum gives the best configuration,
neither enumerates. So a relation with several branches is handled natively by Prolog and badly
here: multi-modality is exactly the [[messages]] §1 gap, and [[Branches and the Discriminant]]
is where the vault records that a branch point is where the machinery fails.

**Negation.** Prolog has negation as failure under a closed-world assumption. This project has
no negation at all, and it is awkward to add: ``-E`` is not a valid energy, since energies are
bounded below. The nearest thing is a negative weight in a sum of energies — which is what
contrastive decoding does ([[Language Models]] §6) — and that is a weighting, not a negation.

## 8. The slogan is taken, in its most literal reading

"Prolog plus weights" exists and is mature. **Markov Logic Networks** (Richardson & Domingos)
are exactly that: first-order formulas with weights, grounded into a Markov random field —
*i.e.* into a factor graph — with the weights as energies and inference by message passing or
MCMC. **ProbLog** and **PRISM** do the probabilistic-Prolog version directly, and the field is
**Statistical Relational Learning**.

So the honest position:

> **MLN and SRL already occupy "the Prolog of machine learning"** — for *discrete, symbolic*
> relations over logical atoms, with the schema-and-grounding story this project lacks entirely
> (§6).

What is left that is genuinely different is the other fragment. MLN factors are weighted
formulas over discrete atoms; here a factor may be a **continuous residual, a solver, an ODE, a
diffusion prior or a neural network**, over ``\mathbb{R}^n``. The logic side has the
quantification and lacks the continuous learned relations; this side has the continuous learned
relations and lacks the quantification.

## 9. A more accurate slogan

Two candidates, both more defensible than the original:

- **"The Mercury of differentiable modelling."** Mercury's advance over Prolog is the
  statically-checked mode and determinism system, which is precisely what `supports_polarity`
  and `isunidirectional` are reaching for and precisely what
  [[The Type Discipline of a Factor Graph]] §3.4 wants decided rather than declared.
- **"Statistical relational learning over continuous, learned relations."** Longer, unlovely,
  and it names the gap it fills relative to the existing field.

The original slogan is not wrong so much as *ambitious in the direction the project is weakest*.
It claims the schema-and-quantification half, which is absent, rather than the modes-and-graded-
truth half, which is the actual contribution.

## Sources

- Somogyi, Henderson & Conway, *The execution algorithm of Mercury* — the mode and determinism
  system.
- Dechter, *Bucket Elimination: A Unifying Framework for Reasoning*, Artificial Intelligence
  113, 1999 (and Constraints 2(1), 1997) — one algorithm across CSP, optimisation and
  probabilistic inference.
- Bistarelli, Montanari & Rossi, *Semiring-Based Constraint Satisfaction and Optimization*,
  JACM 44(2):201–236, 1997; with Fargier, Schiex & Verfaillie, *Semiring-based CSPs and Valued
  CSPs*, Constraints 4:199–240, 1999.
- Richardson & Domingos, *Markov Logic Networks*, Machine Learning 62, 2006.
- De Raedt, Kimmig & Toivonen, *ProbLog*, IJCAI 2007; Sato & Kameya, *PRISM*.
- Poole, *First-order probabilistic inference*, IJCAI 2003 — lifted inference.

Related: [[Channels and Polarity]], [[The Type Discipline of a Factor Graph]],
[[Energy-Based Factor Graphs]], [[Depth in Implicit Learning]],
[[Branches and the Discriminant]], [[The Structural Gap to ModelingToolkit]],
[[Acausal Composition is a Hypergraph Category]], [[Related Julia Projects]],
[[Loopy Message Passing]], [[messages]]
