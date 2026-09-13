# Geometric Deep Learning and Physical Laws

> Geometric deep learning builds the **symmetry** of a physical problem into a network.
> That is necessary — and it is not sufficient, because a physical law is not a symmetry.
> It is a **relation**, and a network can respect every symmetry a problem has while
> violating every law it has.
>
> There is a clean example where this is not philosophy but a measured, published failure
> (§4), and a one-line counterexample that makes the distinction precise (§2).

## 1. What geometric deep learning is

Bronstein, Bruna, Cohen and Veličković's programme (the "5G" proto-book — grids, groups,
graphs, geodesics, gauges) derives most successful architectures from one principle:
**respect the symmetry group $G$ of the domain**. A layer $f$ should be *equivariant*,

$$
f(g \cdot x) \;=\; g \cdot f(x) \qquad \text{for all } g \in G,
$$

so that translating an image translates its feature map, rotating a molecule rotates its
predicted forces, permuting a graph's nodes permutes its outputs. CNNs, GNNs and transformers
fall out as the equivariant maps for translations, permutations and set symmetries
respectively. As an inductive bias it is enormously effective, and nothing in this note
disputes that.

## 2. The category error: a symmetry constrains a map, a law constrains a configuration

Equivariance is a property of the **function** $f$: it says how the *output transforms* when
the *input transforms*. A physical law is a property of **configurations**: it says which
states are *admissible*.

| | is a statement about | says |
|---|---|---|
| a symmetry | the map $f: X \to Y$ | $f$ commutes with $G$ |
| a law | the state $x$ | $r(x) = 0$ |

These are orthogonal. The set of $G$-equivariant maps is huge, and almost none of them
produce outputs satisfying any given law. One line makes it concrete:

> **$\dot x = -x$ is rotation-equivariant and conserves nothing.** It is a perfectly good
> $SO(n)$-equivariant vector field — rotate the input, the output rotates — and every
> trajectory it generates loses energy, shrinks phase-space volume and decays to the origin.
> Equivariance placed no constraint whatsoever on conservation.

Or, for forces: **$F(x) = x^\perp$** (the 90° rotation field in the plane) is
rotation-equivariant and has curl $2$. It is not the gradient of any potential, and it does
net work around every closed loop. An equivariant force field need not be a conservative one.

Being the gradient of a scalar — $F = -\nabla E$, equivalently $\nabla \times F = 0$ — is a
**relation on the output**, not a symmetry of the map. That is the whole of the argument, and
the rest of this note is one instance of it, measured.

## 3. Why one might have expected otherwise: Noether

The intuition that symmetry *should* give you conservation is Noether's theorem, and it is
worth saying precisely why it does not apply here.

Noether: every continuous symmetry of a **Lagrangian** yields a conserved quantity. The
hypothesis is the variational structure — the dynamics are Euler–Lagrange equations of an
action. An equivariant neural network is not that. Equivariance of $f$ under $G$ says nothing
about whether trajectories of $\dot x = f(x)$ conserve anything, because there is no action
for them to be stationary points of.

Hamiltonian and Lagrangian neural networks (Greydanus et al.; Cranmer et al.) are exactly the
fix: they impose the variational structure — $\dot x = J\nabla H(x)$ — and *then* symmetry
of $H$ gives conservation by Noether. But "the dynamics are Hamiltonian" is a **relation**
between $\dot x$ and $x$, not a symmetry. So HNNs are a case of adding a relation on top of
geometry, which is the point.

## 4. The example: machine-learned force fields

This is the cleanest real case, because the community ran the experiment and published the
failure.

**The setting.** Machine-learned interatomic potentials (NequIP, MACE, and successors) are
$E(3)$-equivariant by construction — a triumph of geometric deep learning. Rotate the
molecule, the predicted forces rotate. That symmetry is necessary: a force field that got it
wrong would be unusable.

**The temptation.** Predicting forces *directly* as an equivariant vector output is faster
than predicting a scalar energy and differentiating it. Several recent models do exactly that,
on the argument that conservation "can be learned from the data".

**The result.** Bigi, Langer and Ceriotti ([arXiv:2412.11569](https://arxiv.org/abs/2412.11569),
ICML 2025) tested it:

> direct-force models exhibit *"fundamental issues, from ill-defined convergence of geometry
> optimization to instability in various types of molecular dynamics"*, and *"energy
> conservation is hard to learn, monitor, and correct for."*

The models are equivariant. They are also non-conservative, and in a long simulation the
non-conservation compounds into drift and instability. The recommended remedy is to keep a
*conservative* model — one whose forces are $-\nabla E$ by construction — in the loop.

**What this is, in the vault's terms.** $F = -\nabla_x E$ is a residual:

$$
r(E, F, x) \;=\; F + \nabla_x E \;\approx\; 0,
$$

a relation among three quantities with no privileged one. Predicting $E$ and differentiating
is *one polarity* of it — solve for $F$ given $E$. The direct-force models dropped the
relation and kept the symmetry, and that is exactly the failure the opening sentence
describes.

> [!important] "Hard to learn" is the key phrase
> A relation that must hold *exactly* cannot be acquired from finite data by a function
> approximator. It can be approximated to any accuracy on the training distribution and still
> be violated by an amount that compounds over $10^6$ integration steps. Laws are imposed,
> not learned — which is the case for putting them in the graph as factors rather than in the
> network as hoped-for behaviour.

## 5. A second example, in a domain the vault already has

A permutation-equivariant GNN over a power grid respects the graph's symmetry — relabel the
buses and the predictions relabel with them. Nothing in that symmetry makes the predicted
flows satisfy Kirchhoff's current law at each node. KCL is a relation among the *outputs at
neighbouring nodes*, $\sum_i i_i = 0$, and an equivariant map is free to violate it at every
node simultaneously.

[[Energy Markets and Power Grids]] and the `LinearConstraintFactor` example in this project
are that law, as a factor. The GNN would be a *learned* factor beside it — and the graph, not
the network, is what enforces the law.

## 6. Where relations outperform, honestly

Not on single-shot accuracy. An equivariant network trained on enough data is very hard to
beat at predicting one output from one input, and nothing here claims otherwise. The
advantage is elsewhere, and it is specific.

**Long horizons.** Any quantity that is *approximately* conserved per step and *exactly*
conserved by the physics diverges over enough steps. Molecular dynamics, climate integration,
orbital mechanics: the error is not in any single prediction but in the accumulation. A
relation enforced exactly has zero drift by construction. §4 is the measured version.

**Partial observation.** A symmetry constrains a map from what you observe to what you
predict. A relation lets you solve for *whichever quantity is missing*. Given a conservation
law and all but one of the quantities it relates, the law **determines** the last one; an
equivariant map has no way to even pose that question. This is [[Channels and Polarity]], and
it is why the acausal factor in this project reconciles three inconsistent flow measurements
against one conservation law in its test suite — a query that has no equivariant-network
formulation at all.

**A residual that means something.** When the relation is a factor, its violation is a
number with domain meaning — energy drift, a failed sensor, a mispricing
([[Motivating Examples]] §6). An equivariant network has no such quantity: it produces an
output, and how badly that output violates the physics is not something it computes.

**Model comparison.** The free energy scores a model against data with the law's cost
included. There is no equivariant-network analogue of "how well does this respect
conservation, as a number".

## 7. The converse, which is also true

Relations without geometry are insufficient too, and this project is on the wrong side of
that one.

**Nothing in Lenticulum is equivariant.** A `GaussianFactor` is not rotation-equivariant; a
`LinearConstraintFactor` has no notion of a group acting on its channels; the factor libraries
wrap Lux models without asking whether those models respect anything. A factor graph that
enforces $F = -\nabla E$ exactly but whose learned energy $E$ is not rotation-invariant would
conserve energy while predicting different physics in different orientations — a different,
equally fatal failure.

So the honest position is not "relations beat symmetry" but:

> [!important] Both, and in a specific arrangement
> The **symmetry** belongs *inside* the learned factors — an equivariant network as the
> residual's parametrisation, so that the hypothesis class respects $G$. The **law** belongs
> *on the graph* — a constraint factor, so that the admissible set respects it.
>
> Geometric deep learning supplies the first. This framework supplies the second. Neither
> alone is a physical model, and the case for factor graphs of learned relations is precisely
> that they are the natural place to hold both.

Making that concrete — an equivariant Lux model as a `DEQFactor` cell or a `DiffusionFactor`
prior, beside a `LinearConstraintFactor` for the law — needs no new machinery on the graph
side. What it needs on the factor side, an equivariance-aware `Channel` and a group action
on beliefs, is entirely absent and is a real gap.

## 8. The statement

- Equivariance constrains the **map**; a law constrains the **configuration**. Orthogonal.
- $\dot x = -x$ is equivariant and conserves nothing; $F = x^\perp$ is equivariant and does
  work. One line each.
- Noether needs a Lagrangian; an equivariant network is not one. Symmetry gives conservation
  only *through* a relation.
- Direct-force models are the measured failure: equivariant, non-conservative, unstable over
  long horizons. Energy conservation *"is hard to learn"* because it is not the kind of thing
  learning does.
- Relations win on long horizons, partial observation, attribution and model comparison —
  not on single-shot accuracy.
- And this project has no equivariance at all, so the right architecture is equivariant
  factors on a relational graph, and only half of it exists here.

## Sources

- Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning: Grids, Groups, Graphs,
  Geodesics, and Gauges*, [arXiv:2104.13478](https://arxiv.org/abs/2104.13478), 2021.
- Bigi, Langer & Ceriotti, *The dark side of the forces: assessing non-conservative force
  models for atomistic machine learning*, ICML 2025,
  [arXiv:2412.11569](https://arxiv.org/abs/2412.11569). **The measured example of §4.**
- Greydanus, Dzamba & Yosinski, *Hamiltonian Neural Networks*, NeurIPS 2019; Cranmer et al.,
  *Lagrangian Neural Networks*, 2020 — adding the variational relation on top of symmetry.
- Batzner et al., *E(3)-equivariant graph neural networks for data-efficient and accurate
  interatomic potentials* (NequIP), Nature Communications 2022; Batatia et al., *MACE*,
  NeurIPS 2022 — the equivariant force fields in question.
- Noether, *Invariante Variationsprobleme*, 1918.

Related: [[Motivating Examples]], [[Molecular Dynamics]], [[Energy Markets and Power Grids]],
[[Channels and Polarity]], [[Implicit Learners]], [[Energy-Based Learning]],
[[Lux as a Parametric Lens]], [[Three Senses of Implicit]]
