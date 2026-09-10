# Acausal Composition is a Hypergraph Category

> The categorical foundation under [[ModelingToolkit as an Acausal Relation]], and the answer
> to a question the vault has been circling since [[Copiers Cups and Caps]]: *how much
> structure does it take to wire factors into an arbitrary graph rather than a DAG?*
>
> **Answer: a special commutative Frobenius algebra on every object.** Compact closure — what
> [[Copiers Cups and Caps]] buys — is strictly less than that, and the difference is exactly
> the difference between bending one wire and joining $d$ of them.
>
> The payoff is a justification of `beliefs.jl` that is stronger than the numerical one:
> **improper Gaussian beliefs are not a convenience, they are what makes acausal composition
> possible at all.**

## 1. What a lens cannot do

$\mathbf{Lens}(\mathcal{C})$ composes by function composition ([[Lens]] §"Reading the
composition formula"). To compose $(f,f^*)$ and $(g,g^*)$ you need $f$'s codomain to be $g$'s
domain: an *output* meeting an *input*. Two consequences, both recorded in
[[Lux as a Parametric Lens]]:

1. the wiring must be a DAG, because function composition around a cycle does not terminate;
2. each morphism knows which side is input, at construction time.

An acausal `connect` violates both. It joins two wires with no direction, and it does not care
how many wires meet — a Kirchhoff node with five branches is one interconnection, not four
nested binary ones. Whatever `connect` is, it is not composition in a lens category.

## 2. The structure that does it: Frobenius

A **hypergraph category** (Fong–Spivak) is a symmetric monoidal category in which every object
$X$ carries a *special commutative Frobenius algebra*:

$$
\mu : X\otimes X \to X,\quad
\eta : I \to X,\quad
\delta : X \to X\otimes X,\quad
\epsilon : X \to I
$$

with $(\mu,\eta)$ a commutative monoid, $(\delta,\epsilon)$ a cocommutative comonoid, plus

$$
\underbrace{(\mu\otimes 1)(1\otimes\delta) = \delta\mu = (1\otimes\mu)(\delta\otimes 1)}_{\text{Frobenius}},
\qquad
\underbrace{\mu\,\delta = 1_X}_{\text{special}}
$$

and these are required to be compatible with $\otimes$. In pictures, $\delta$ is a wire
splitting and $\mu$ is two wires merging.

### The spider theorem is the whole point

The Frobenius axioms have a striking consequence: **any connected diagram built from
$\mu,\eta,\delta,\epsilon$ with $m$ inputs and $n$ outputs equals the canonical
$(m,n)$-spider.** The internal structure collapses; only the number of legs survives.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[every node/.style={font=\small}, dot/.style={circle,fill=black,inner sep=1.6pt}]
\node[dot] (s) at (0,0) {};
\foreach \a in {150,180,210} \draw (s) -- ++(\a:1.1);
\foreach \a in {-30,0,30} \draw (s) -- ++(\a:1.1);
\node at (0,-1.35) {\footnotesize the $(3,3)$-spider};
\node at (3.1,0) {$=$};
\node[dot] (a) at (4.4,0.35) {};
\node[dot] (b) at (5.4,-0.35) {};
\draw (a)--(b);
\foreach \a in {150,180,210} \draw (a) -- ++(\a:0.9);
\foreach \a in {-30,0,30} \draw (b) -- ++(\a:0.9);
\node at (4.9,-1.35) {\footnotesize any connected wiring of it};
\end{tikzpicture}
\end{document}
```

> [!important] A variable node in a factor graph is a Frobenius spider
> A variable of degree $d$ in a `FactorGraph` is a $d$-legged spider. That is the formal
> content of "a variable is a wire with no content of its own"
> ([[Everything is a Factor]]): the spider theorem says a $d$-way junction has no internal
> structure to have content *with*.
>
> And it is the formal reason the wiring may be an arbitrary graph. Composition in a lens
> category is a binary operation with a direction; a spider is a $d$-ary operation with none.

This is not decoration. It makes two pieces of `Mycelium` into named categorical operations:

| Mycelium | Frobenius |
|---|---|
| `combine(a, b)` | the multiplication $\mu$ |
| `TrivialBelief()`, the unit of `combine` | the unit $\eta$ |
| `marginal(store, g, v)` — fold `combine` over all $d$ incident edges | the $d$-legged spider at $v$ |
| `excluded_marginal(store, g, v, e)` — fold over $d-1$ of them | the $(d{-}1)$-spider, one leg left open |

`messages.md`'s claim that `combine` is "the hardest operation in the package" is, in this
light, the statement that **the Frobenius multiplication is the hard part of being a
hypergraph category** — and for most belief representations it is genuinely unavailable, which
is why `combine` throws for everything but Gaussians and Diracs.

## 3. Where the vault already was: compact closed $\subsetneq$ hypergraph

[[Copiers Cups and Caps]] lifts the DAG restriction using cups and caps — compact closure. A
cup $I \to X\otimes X$ lets you bend an input wire into an output wire, so a cycle becomes a
straight line with a bent end. That is real and it is enough for [[Bayesian Inversion]].

But compact closed is *weaker* than hypergraph. Every hypergraph category is compact closed
(take the cup to be $\delta\,\eta$ and the cap $\epsilon\,\mu$), and the converse fails: a
cup is a binary operation, and it does not give you a $d$-way merge for $d > 2$.

| structure | what you can wire | vault note |
|---|---|---|
| monoidal | a DAG | [[Lens]], [[Para]] |
| compact closed | a DAG with feedback loops (bend a wire) | [[Copiers Cups and Caps]] |
| **hypergraph** | an arbitrary undirected (hyper)graph; $d$-way junctions | this note |

So the honest reading of the vault's own history: [[Copiers Cups and Caps]] got Lenticulum out
of the DAG, and this note says that what it actually needs — because a factor graph variable
has arbitrary degree — is one rung further up.

## 4. Gaussians: the punchline for `beliefs.jl`

Here the theory says something concrete and slightly surprising about code already in this
repository.

**Gaussian *maps* do not form a hypergraph category.** A Gaussian conditional $x\mapsto
\mathcal{N}(Ax+b,\Sigma)$ is a morphism in a *Markov* category: it has copy and delete, and
delete is natural because every kernel is normalised. But there is no $\mu$ — **you cannot
merge two probability wires.** Merging means "these two are equal", and conditioning two
independent Gaussians to be equal produces an *unnormalised* density. Normalisation is exactly
what obstructs the Frobenius multiplication.

The fix, worked out by Stein and Samuelson, is to enlarge the category:

- **Gaussian relations** — Gaussian distributions together with *linear relations* — do form a
  hypergraph category. The extra objects needed are precisely the **improper / uninformative
  priors** and the totally-undetermined relations.
- **Graphical Quadratic Algebra** axiomatises this diagrammatically, and it is complete: the
  string-diagram equations characterise the category of quadratic relations exactly.
- The papers name Willems' theory of open systems and uninformative priors in Bayesian
  statistics as the two phenomena this unifies — which are §3 and §4 of
  [[ModelingToolkit as an Acausal Relation]] respectively.

Now compare `beliefs.jl`:

```julia
struct GaussianBelief{V,M} <: LenticulumCore.AbstractBelief
    η::V      # information vector
    Λ::M      # precision — MAY BE SINGULAR OR ZERO, and that is not an error
end
```

and its docstring: *"`Λ` may be **singular or zero**, and that is not an error… The moment
form cannot represent this at all."* The note argues for canonical form on two numerical
grounds — pooling becomes addition, and rank-deficient likelihoods become representable. The
categorical statement is stronger and subsumes both:

> [!important] The canonical form is what makes Gaussians a hypergraph category
> - $\Lambda$ singular is an **improper belief**, i.e. a linear relation rather than a
>   distribution. Admitting these is the completion that makes $\mu$ exist.
> - `combine` being *addition of canonical parameters* is $\mu$ being **unnormalised**. In
>   moment form there is no such operation; in canonical form it is $+$, total and
>   associative.
> - `uninformative(n) = GaussianBelief(zeros(n), zeros(n,n))` is the Frobenius **unit**
>   $\eta$: the relation that constrains nothing.
> - `logpartition(b)` is the normaliser that $\mu$ discards, carried separately — which is
>   why the free-energy bookkeeping of [[Bethe Free Energy]] is a distinct concern from
>   message passing rather than a detail of it.
>
> `beliefs.jl` was written for numerical reasons and landed on the right object for
> structural ones. The improper beliefs are not a tolerated degeneracy; they are half the
> category.

This also explains, after the fact, why `LinearConstraintFactor` cost so little to add
(`constraint.md`): the belief type was already the acausal one. Nothing new was needed on the
belief side, because canonical-form Gaussians *are* Gaussian relations.

## 5. Markov vs hypergraph: what acausality costs

The trade is worth stating as a table, because it is the reason directed graphical models and
acausal models are different subjects.

| | Markov category | hypergraph category |
|---|---|---|
| copy $\delta$ | yes | yes |
| delete $\epsilon$ | yes, **natural** (everything normalised) | yes, not natural |
| merge $\mu$ | **no** | yes |
| morphisms are | normalised kernels | relations / unnormalised |
| models | directed generative models, Bayes nets | acausal equations, circuits, behaviors |
| composition needs | a direction | nothing |

You cannot have both naturally: naturality of delete is what forces normalisation, and
normalisation is what forbids merge. Lenticulum sits on the right-hand column and pays the
price in bookkeeping — every message is unnormalised, and the partition function is tracked by
hand through the free energy. **That is not an implementation shortcut; it is the cost of
letting a variable have degree three.**

Work reconciling the two — *partial Markov categories*, and the conditioning-as-comb
constructions — is the current research answer, and it is the right place to look when
[[Bethe Free Energy]]'s normaliser bookkeeping starts to hurt.

## 6. Decorated cospans: where the graph itself lives

One more layer, mentioned because it is the standard construction and because
[[Composition is Elimination]] is secretly about it.

An **open system** is a system with a boundary: a cospan $L \to S \leftarrow R$ in a category
of "interfaces", *decorated* by the system's actual content. Composition is a pushout — glue
along the shared boundary. Fong's decorated cospans and Baez–Courser's structured cospans are
the two standard ways to make this precise, and the theorem is that the result is a hypergraph
category, which closes the circle with §2.

The dictionary for this repository:

| decorated cospans | Lenticulum |
|---|---|
| the apex $S$ | a subgraph of the factor graph |
| the legs $L \to S \leftarrow R$ | its boundary variables — the channels it exposes |
| the decoration | the factors and their parameters |
| pushout along a shared leg | joining two subgraphs at a variable |
| the resulting hypergraph category | the algebra of factor graphs |

And the operation this makes precise is **collapsing a subgraph into a single factor**: an
open system with a boundary *is* a factor whose channels are the boundary variables. That is
[[Composition is Elimination]] read categorically, and it is what an `MTKFactor`
([[ModelingToolkit as an Acausal Relation]] §8) would be — a hard subsystem, eliminated down
to its interface, presented as one factor.

## 7. What this note does not claim

> [!warning] None of this is implemented as category theory
> `Mycelium` has no `Frobenius` type, no cospans, no pushouts. §2's table is an
> *identification* of existing code with existing mathematics, not a description of an
> abstraction layer. The value is diagnostic: it says which operations are load-bearing
> (`combine`, and the improper beliefs it needs) and which absences are structural rather than
> accidental.

Specific gaps, stated honestly:

1. **`combine` is only a $\mu$ for Gaussians and Diracs.** For every other belief type it
   throws. So Lenticulum is a hypergraph category on the Gaussian fragment and a partial
   mess elsewhere — which is the same conclusion `messages.md` §1 reaches from the other
   direction.
2. **The `special` axiom $\mu\delta = 1$ is not checked and probably fails.** $\mu\delta$ on a
   Gaussian belief squares the density: $\Lambda \mapsto 2\Lambda$. So `combine(b, b) != b` —
   copying a belief and immediately merging it double-counts. This is the *same bug* the
   exclusion principle of [[Messages are Inversions]] exists to prevent, seen from the
   categorical side. Belief propagation's exclusion is what restores speciality by hand.
   **That is a genuinely useful reframing: the exclusion principle is the special-Frobenius
   axiom, enforced by the scheduler because the beliefs do not satisfy it themselves.**
3. **No hyperedges.** A `FactorGraph` variable has arbitrary degree, so the *spider* part is
   there; but factors connect to variables one channel at a time and there is no notion of a
   factor sharing one channel with several variables.
4. **Nothing here addresses laxness.** The Frobenius story is about wiring, not about the
   inexactness of inversions — that is [[Composition of Bayesian Lenses]] Remark 16 and
   [[Scalar and Multivariate Energy]], and the two concerns are orthogonal.

## Sources

- Fong & Spivak, *Hypergraph Categories*, [arXiv:1806.08304](https://arxiv.org/abs/1806.08304)
  — definition, the spider theorem, and the relationship to compact closure.
- Fong, *The Algebra of Open and Interconnected Systems*,
  [arXiv:1609.05382](https://arxiv.org/abs/1609.05382) — decorated cospans; open circuits and
  dynamical systems as the motivating examples.
- Baez & Courser, *Structured Cospans* — the variant that avoids decorated cospans' technical
  hypotheses.
- Stein & Samuelson, *A Category for Unifying Gaussian Probability and Nondeterminism*, CALCO
  2023, [arXiv:2204.14024](https://arxiv.org/abs/2204.14024) — Gaussian relations; improper
  priors as the completion.
- Stein & Samuelson, *Graphical Quadratic Algebra*,
  [arXiv:2403.02284](https://arxiv.org/abs/2403.02284) — a complete diagrammatic axiomatisation
  of quadratic relations, with Willems' open systems and uninformative priors named as the
  unified phenomena.
- Bonchi, Sobociński & Zanasi, *Interacting Hopf Algebras* — the prop of linear relations, the
  non-probabilistic ancestor of the above.
- Fritz, *A synthetic approach to Markov kernels, conditional independence and theorems on
  sufficient statistics* — Markov categories, and why delete's naturality matters.
- Di Lavore, Román & Sobociński, *Partial Markov Categories*,
  [arXiv:2502.03477](https://arxiv.org/abs/2502.03477) — reconciling normalisation with
  conditioning.

Related: [[Copiers Cups and Caps]], [[ModelingToolkit as an Acausal Relation]], [[Lens]],
[[Factor Graphs]], [[Everything is a Factor]], [[Messages are Inversions]],
[[Composition is Elimination]], [[Bethe Free Energy]]
