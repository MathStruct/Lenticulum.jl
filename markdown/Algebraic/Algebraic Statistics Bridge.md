# The Algebraic Statistics Bridge

> The observation that makes this whole family more than a curiosity for Lenticulum:
> **conditional independence is a polynomial constraint**, so a Bayesian network *is* an
> algebraic variety, and **marginalisation is elimination**. The algebraic learner and the
> AutoBayes factor graph are the same object seen twice.

## Conditional independence is determinantal

Let $X, Y$ be discrete with $|X| = a$, $|Y| = b$, and let
$p \in \Delta^{ab-1}$ be the joint probability table, $p_{ij} = P(X=i, Y=j)$. Then

$$X \perp\!\!\!\perp Y \iff p_{ij} = p_{i+}p_{+j} \iff \operatorname{rank}(p) = 1
\iff \text{all } 2\times 2 \text{ minors vanish}$$

$$\boxed{\;p_{ij}p_{kl} - p_{il}p_{kj} = 0 \quad \text{for all } i<k,\ j<l\;}$$

Independence is the vanishing of a set of quadrics. Conditional independence
$X \perp\!\!\!\perp Y \mid Z$ is the same statement slice-by-slice in $z$: the $2\times2$
minors of each conditional slice.

So the model "all distributions satisfying this CI statement" is a **determinantal variety**
intersected with the probability simplex. This is the founding observation of
**algebraic statistics** (Drton–Sturmfels–Sullivant, *Lectures on Algebraic Statistics*;
Pistone–Riccomagno–Wynn).

## Therefore a graphical model is a variety

A Bayesian network's model — the set of joint distributions that factorise according to the
DAG — is cut out by the CI statements implied by the graph (its global Markov property), each
of which is a set of minors. Discrete graphical models are **toric or determinantal
varieties** in the simplex.

This is exactly the object an [[Algebraic Implicit Learners|algebraic implicit factor]]
learns. **Fitting a variety to data and fitting a graphical model to data are, in the
discrete case, the same problem in different notation.** In particular:

- the [[The Veronese Parametrisation|Veronese parametrisation]] $\Theta v_d(p)$ with $d=2$
  spans all quadrics, hence all pairwise CI constraints;
- [[Fitting is a Nullspace Problem|the nullspace fit]] is then *structure learning*: which
  quadrics vanish on the empirical distribution is which CI statements hold;
- the [[The Parameter is a Grassmannian|Grassmannian]] non-identifiability is the statement
  that many generating sets encode the same CI structure.

## Marginalisation is elimination

Here the correspondence becomes sharp and useful. Consider a graphical model with a hidden
variable $H$. The **marginal model** — the set of distributions over the observed variables
obtained by summing out $H$ — is the image of a polynomial map

$$\phi \;:\; \Theta \longrightarrow \Delta, \qquad \phi(\theta)_{\text{obs}} = \sum_h p_\theta(\text{obs}, h)$$

Its **Zariski closure** is computed by **elimination**: eliminate the parameters and the
hidden coordinates from the ideal. The resulting equations are the constraints the marginal
model satisfies — classically, the **tetrad constraints** of factor analysis
($3\times3$ minors vanishing) are exactly the elimination ideal of a one-hidden-factor model.

Now line this up with [[Composition is Elimination]]:

| algebraic geometry | probability | AutoBayes |
|---|---|---|
| projection $\pi_{XZ}$ | marginalisation $\sum_y$ | pushforward $c_*\pi$ |
| elimination ideal $I\cap k[x,z]$ | the marginal model's equations | the composite model |
| image is only **constructible** (Chevalley) | the marginal model is only **semialgebraic** | composition is **lax** |
| Zariski closure adds spurious points | inequality constraints are lost | the 2-cell witnessing laxness |
| doubly exponential (Mayr–Meyer) | marginalisation is #P-hard | "computing $c_*\pi$ is expensive" |

**Every row is the same fact stated in a different language.** The paper's closing warning —
that "priors are propagated by pushforward (i.e. marginalization), and computing these is
similarly expensive to computing exact inversions", so belief propagation is warranted — is
the probabilistic form of "do not run Gröbner, keep the factors separate".

> [!note] Why this matters for Lenticulum
> [[Composition is Elimination]] argues from complexity that you must not compose algebraic
> factors. This note shows the argument is not special to the algebraic family: it is the
> *same* argument as the one AutoBayes already makes for Bayesian factors. The two
> justifications for `Mycelium.jl` — algebraic and probabilistic — are one justification.

## The semialgebraic gap, made concrete

Chevalley says the image of a variety is constructible; over $\mathbb{R}$,
Tarski–Seidenberg says semialgebraic. In statistics this is not a technicality: hidden-variable
models are famously **not** varieties. The set of distributions realisable by a latent-class
model with $k$ classes is cut out by equations **and inequalities**, and the inequalities are
essential — the "expected" variety contains distributions that are not achievable by any
non-negative parameter values.

So the honest type of a marginalised graphical model is *semialgebraic set*, and:

- the correct tool is **real** quantifier elimination (Collins' CAD, doubly exponential) or
  Positivstellensatz certificates, not Gröbner bases;
- an implementation that models the marginal as a variety is systematically **too
  permissive**, admitting parameter values that no distribution realises.

This is the precise algebraic content of the [[Composition of Bayesian Lenses|laxness]]
theme, and it says the laxness is not merely quantitative (a KL gap) but **type-level**: the
composite is a different kind of object than the parts.

## What is worth borrowing

Algebraic statistics has fifteen years of results on exactly the objects Lenticulum needs,
and they are directly transferable:

- **Model invariants** — the elimination ideal of a graphical model is a *checkable
  signature* of its structure. Two graphs with the same invariants are indistinguishable from
  data; this is model identifiability, and it is decidable.
- **Maximum likelihood degree** — the number of complex critical points of the likelihood on
  the model, the exact analogue of the Euclidean Distance Degree in
  [[Algebraic versus Geometric Distance]]. It counts how many local maxima EM can get stuck
  in. For many standard models it is known exactly.
- **Toric structure** — decomposable/hierarchical models are toric varieties, for which
  everything (Gröbner bases, degree, ML estimation) is far better behaved. **If a factor's
  relation can be made toric, do so**; it is the single most valuable structural property in
  the family.

Related: [[Composition is Elimination]], [[Composition of Statistical Games]], [[The Algebraic Factor as a Statistical Game]], [[Statistical Game]]
