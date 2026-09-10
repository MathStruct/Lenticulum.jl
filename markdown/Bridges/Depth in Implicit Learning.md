# Depth in Implicit Learning

> **Is there a "no deep learning theorem" — can a flat graph always do what a deep one does?**
>
> Short answer: the vault already contains the answer twice, written as a complexity argument
> rather than an expressivity one. **The theorem holds exactly on the linear-Gaussian fragment
> and fails everywhere else**, and the boundary is a single number — the degree $d$ in
> [[Composition is Elimination]]'s bound $d^L$.
>
> The longer answer is more interesting: in implicit learning, *depth-as-computation* and
> *depth-as-expressivity* come apart. They are the same thing in explicit deep learning, which
> is why the question feels strange.

## 1. What depth is, in a factor graph

A "deep" graph is one with **intermediate variables**. Flattening it means eliminating them:

$$R_2 \circ R_1 \;=\; \bigl\{(x,z) : \exists\, y,\ (x,y)\in R_1 \wedge (y,z)\in R_2\bigr\}$$

So the question "can a flat graph do what a deep one does?" is exactly the question **"can
latent variables always be eliminated, and at what cost?"** — and that has a literature, three
theorems, and three different answers.

## 2. Three elimination theorems, three costs

| family | eliminable? | cost | in the vault |
|---|---|---|---|
| linear / Gaussian relations | **yes, exactly** | **none** | [[The Linear Gaussian Chain]] |
| linear differential (Willems) | yes | the differential **order** rises | [[ModelingToolkit as an Acausal Relation]] §3 |
| polynomial / semialgebraic | only up to **closure** | degree $d^L$; information is lost | [[Composition is Elimination]] |

Willems' elimination theorem is the cleanest statement of the shape: *the latent variables of a
linear differential system can always be eliminated — at the cost of raising the differential
order.* Eliminable, but not free.

The bottom row is worse than "not free", and it is worth being precise about why.

## 3. Where the theorem is true

**On the linear-Gaussian fragment, the claim is correct.** A chain of `GaussianFactor`s or
`LinearConstraintFactor`s eliminates to a single linear-Gaussian relation. Marginalising a
Gaussian gives a Gaussian; composing linear relations gives a linear relation; the class is
closed and nothing grows.

And that fragment is most of what is implemented, so the observation bites. The odometry chain
of [[The Linear Gaussian Chain]] is *expressively* a single joint Gaussian over three poses —
its depth buys no model class that a flat one lacks.

What the depth buys instead is **sparsity**:

> The chain's joint precision is block-tridiagonal. Flattening it means forming the dense
> inverse. Same model, different representation — and the sparse one is the reason inference is
> $O(n)$ rather than $O(n^3)$.

So on this fragment, depth is a *computational* property, not a modelling one. That is the
precise, true version of the intuition, and it is the same fact as **deep linear networks being
no more expressive than shallow ones**.

> [!note] One caveat, and it is the same one as in the linear case
> A deep linear network with *narrow* hidden layers expresses only low-rank maps. Identically,
> a chain of linear relations through a narrow intermediate variable constrains the rank of the
> composite: $\dim x_1 = k$ forces $\operatorname{rank} \le k$.
>
> So bottlenecked depth is not free even here — but it **restricts** the class rather than
> enriching it. That is the opposite of what depth does in explicit deep learning.

## 4. Where it fails, in two independent ways

Take $d \ge 2$ and both failures appear at once. [[Composition is Elimination]] has them
already; the point here is only that they are *expressivity* statements.

**Failure 1 — the class is not closed.** By Chevalley, the projection of a variety is
constructible, not a variety; over $\mathbb{R}$, Tarski–Seidenberg gives semialgebraic — sets
defined by equations *and inequalities*. Elimination computes the Zariski **closure**, and the
closure is strictly bigger:

$$V(I_{12}) \;=\; \overline{R_2 \circ R_1} \;\supsetneq\; R_2 \circ R_1$$

So flattening a two-factor algebraic graph does not merely cost more — **it produces a
different relation**, with spurious solutions along the boundary. The flat model cannot express
what the deep one expressed, at any size.

**Failure 2 — degree is exponential in depth.** A chain of $L$ factors of degree $d$
eliminates to degree $d^L$, and the parameter count of a degree-$D$ relation in $N$ variables is
$\binom{N+D}{D}$. So the flat model needs $\binom{N + d^L}{d^L}$ parameters to say what the deep
one says with $L\cdot\binom{N+d}{d}$.

> [!important] That bound *is* a depth-separation theorem
> $d^L$ is exponential in depth and equals $1$ when $d = 1$. So the single expression
> $$\text{degree of the flattened chain} = d^L$$
> contains both answers: at $d = 1$ it says depth buys nothing — which is the proposed theorem
> — and at $d \ge 2$ it says a flat model needs exponentially higher degree, which is the
> opposite.
>
> The vault derived this as a *cost* argument for keeping factors separate. It is simultaneously
> an *expressivity* argument, and nobody had read it that way.

The same shape appears in explicit deep learning: shallow networks are universal approximators,
so depth adds no *representable functions*, but the depth-separation results show it saves
exponentially in width. Implicit learning has the identical trade with degree in place of width.

## 5. The real distinction: two things called depth

Here is why the question is genuinely subtle rather than merely answerable.

In explicit deep learning the computation graph **is** the function. The forward pass, the
hypothesis class and the architecture are one object, so "depth" means all three at once.

In implicit learning they come apart:

| the graph is | which means depth is | is it collapsible? |
|---|---|---|
| a **hypothesis class** — which relations are representable | an expressivity property | **no** (§4) |
| an **elimination order** — how to compute the answer | a computational property | **yes, freely** |

The second row is a theorem the vault already states: on a tree, message passing *is*
elimination, reorganised — *"BP is not an approximation to elimination; it **is** elimination"*
([[The Linear Gaussian Chain]] §3). Any valid elimination order gives the same answer, so for
**inference** the arrangement of the graph carries no semantic content at all. It is a schedule.

So the proposed theorem is:

- **true of the graph as a computation** — flat and deep compute the same thing, and this is
  exactly what makes [[Schedules]] a free choice;
- **false of the graph as a hypothesis class** — flattening changes what is representable, or
  its size, or both.

Conflating them is natural because explicit deep learning gives no reason to separate them.

## 6. Depth does not vanish — it converts

The observation about wrapping a Lux model is the right one and generalises further than it
looks.

If a factor's residual is a deep network, its depth is invisible to the graph: the graph is
flat, the model is deep, and the depth has moved from the *wiring* into a *node*. Since
relational composition collapses (up to §4's costs) while functional composition does not, this
is not a coincidence — **in the implicit setting, depth is naturally a factor-level phenomenon
rather than a graph-level one.**

The sharpest illustration is in the repository. A `DEQFactor` wraps the fixed point
$z = g_\theta(z, x)$, and Bai, Kolter and Koltun's founding observation is that this *is* an
infinitely deep weight-tied network. So:

> A DEQ is a **flat graph node that is infinitely deep inside**, and the mechanism converting
> one into the other is exactly the move from a function to a relation.

Depth and implicitness are two currencies for the same thing, and the DEQ is the exchange rate
made explicit. Which suggests the honest slogan is not "implicit learning has no depth" but:

> **Implicit learning trades depth for latency of a different kind.** Explicit models pay in
> layers; implicit models pay in solver iterations, in eliminated latents, or in degree.

## 7. Summary

| question | answer |
|---|---|
| Can a flat graph express what a deep one does? | On the linear-Gaussian fragment, yes exactly. Otherwise no — the class is not even closed. |
| Then what does depth buy on that fragment? | **Sparsity**, hence cheap inference — not expressivity. |
| And off it? | An exponential saving in degree: $d^L$ flat versus $L$ factors of degree $d$. |
| Is the graph's shape ever semantically free? | Yes — for **inference**. Any elimination order computes the same answer. |
| Where does depth go if the graph is flat? | Into the factors. A DEQ is the extreme case. |

The one-line version: **the theorem is the $d = 1$ case of a degree bound the vault already
derived for another purpose**, and the reason it feels like it might be true in general is that
the implemented core is entirely $d = 1$.

Related: [[Composition is Elimination]], [[The Linear Gaussian Chain]],
[[Implicit Learners]], [[Three Senses of Implicit]], [[DEQ as a Relation]],
[[The Veronese Parametrisation]], [[Branches and the Discriminant]],
[[ModelingToolkit as an Acausal Relation]], [[Schedules]], [[Lux as a Parametric Lens]]
