#definition #theorem

> A **switching constraint factor** is a linear constraint whose right-hand side is chosen by a
> discrete outcome: $0 = \sum_i A_i x_i - b_k + \varepsilon$, with $P(k) = w_k$. Linear-Gaussian factors never
> produce a second mode, however they are wired. This is the smallest addition that does while
> keeping every message and every local free energy in closed form, and composed it represents
> exactly the finite Gaussian mixtures. In Lenticulum it is `SwitchingConstraintFactor`.

> Sources: original to this vault (the propositions and proofs; derived on the branch `experiment-qm`, see [[Composing Stochastic Factors]]). Conditional-Gaussian distributions: Lauritzen & Wermuth, *Graphical Models for Associations between Variables, some of which are Qualitative and some Quantitative*, Ann. Statist. 17(1), 1989; Lauritzen, *Propagation of Probabilities, Means, and Variances in Mixed Graphical Association Models*, JASA 87(420), 1992. Gaussian sums in filtering: Sorenson & Alspach, *Recursive Bayesian Estimation Using Gaussian Sums*, Automatica 7(4), 1971. Switching models: Ghahramani & Hinton, *Variational Learning for Switching State-Space Models*, Neural Comput. 12(4), 2000. Factorial models and their mean-field E-step: Ghahramani & Jordan, *Factorial Hidden Markov Models*, Machine Learning 29, 1997. Hybrid belief propagation and mean field: Riegler, Kirkelund, Manchon, Badiu & Fleury, *Merging Belief Propagation and the Mean Field Approximation: A Free Energy Approach*, IEEE Trans. Inf. Theory 59(1), 2013. code: `lib/Lenticulum.jl/src/switching.jl`, `lib/Lenticulum.jl/src/constraint.jl`, `lib/Lenticulum.jl/src/gaussian.jl`, `lib/Mycelium.jl/src/passing.jl`
>
> Bibliography: [[Bibliography#^lauritzen1989graphical|Lauritzen & Wermuth 1989]] · [[Bibliography#^lauritzen1996graphical|Lauritzen 1996]] · [[Bibliography#^lauritzen1992propagation|Lauritzen 1992]] · [[Bibliography#^sorenson1971recursive|Sorenson & Alspach 1971]] · [[Bibliography#^ghahramani2000variational|Ghahramani & Hinton 2000]] · [[Bibliography#^ghahramani1997factorial|Ghahramani & Jordan 1997]] · [[Bibliography#^riegler2013merging|Riegler et al. 2013]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Gaussian Relations](https://mathstruct.org/CategoryTheory-ML-Wiki/Gaussian-Relations) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## 1. Why something has to be added

`LinearConstraintFactor` is a soft linear relation, $\mathcal N(\sum_i A_i x_i; c, Q)$, and with Gaussian
priors it is the whole linear-Gaussian fragment of the library ([[The Linear Gaussian Chain]]).

> **Proposition 1.** Any factor graph of linear-Gaussian potentials and Gaussian priors, in any
> polarity and any wiring, has a Gaussian joint (possibly degenerate), provided the constraints are
> consistent and the product is normalisable. Every marginal and every conditional is Gaussian, so
> no wiring, depth or choice of query creates a second mode.
>
> *Proof.* The joint density is a product of exponentials of quadratics, hence an exponential of a
> quadratic. Marginals and conditionals of a Gaussian are Gaussian. $\square$

So a generative model built from simple factors needs one non-Gaussian ingredient. Several
candidates exist:

- a discrete outcome selecting the constraint;
- a non-Gaussian law on the constraint's null space (Student-$t$, Laplace);
- half-space constraints (truncated Gaussians);
- a Born-rule measurement with a POVM.

The first is the one that keeps every belief, message and exact marginal in the finite
**conditional-Gaussian** family (Lauritzen & Wermuth 1989) with no special functions. The others
need Gamma-type beliefs, the normal CDF $\Phi$, or a new belief type. The next lemma makes "the
smallest" precise.

> **Lemma 2 (closure).** Let a factor with a discrete channel $k$ and continuous channels $z$ have
> potentials $\psi(k, \cdot)$. Its messages to the continuous channels, for every categorical message
> on $k$ and every Gaussian message on the others, are finite mixtures of (possibly improper,
> canonical) Gaussians if and only if each $\psi(k, \cdot)$ is a finite sum of exponentials of
> (possibly degenerate) quadratics in $z$.
>
> *Proof.* If so, every product with Gaussians and every Gaussian integral stays in that form, and
> the sum over $k$ is finite. Conversely, with $z$ the only continuous channel and the message on $k$
> a point mass at one outcome, the message is $\psi(k, \cdot)$ itself. $\square$

A factor with one Gaussian site per outcome is therefore the general closed case, up to splitting
outcomes. A normalised softmax gate $p(k \mid z)$ is outside the family (its normaliser is not a sum
of Gaussian sites); an unnormalised log-linear gate $\exp(a_k^\top z)$ is inside it.

## 2. The factor

> **Definition (switching constraint factor).** Continuous channels $x_1, \dots, x_n$, a discrete
> channel $k \in \{1, \dots, K\}$, and the potential
>
> $$
> f(x, k) = w_k\, \mathcal N\Bigl(\sum_i A_{k,i} x_i - b_k;\ 0,\ Q_k\Bigr), \qquad \textstyle\sum_k w_k = 1,\ Q_k \succ 0.
> $$
>
> The **shared form** has one $A_i$ per channel and one $Q$ for all outcomes: "the observable
> $\sum_i A_i x_i$ is measured, and the outcome is one of $b_1, \dots, b_K$". The **general form** has
> per-outcome $A_{k,i}$ and $Q_k$.

`SwitchingConstraintFactor(dims, K, m; switch = :k, noise)` is the shared form, with parameters
`(A, b, logw)` and `Q` fixed, as in `LinearConstraintFactor`. With $K = 1$ it is exactly a
`LinearConstraintFactor` with $c = b_1$ (asserted in the tests, message for message). Its solution
sets are $K$ parallel copies of the null space of $A$ with one covariance, so it represents
location-shift mixtures; non-parallel modes (lines through the origin at different angles) need
the general form, which is not implemented.

Like `LinearConstraintFactor`, $f$ is a **potential**, not a normalised density: flat along the null
space, with the normalisation coming from the rest of the graph.

## 3. Messages

With incoming $\mathcal N(m_i, S_i)$ on the continuous channels $i \ne t$ and $\pi_k$ on the outcome, let

$$
d_k = b_k - \sum_{i \ne t} A_i m_i, \qquad R = Q + \sum_{i \ne t} A_i S_i A_i^\top .
$$

- **To the outcome** ("the measurement"): $\mu(k) \propto w_k\, \mathcal N(d_k; 0, R)$, the evidence of each
  outcome. A `CategoricalBelief`, and a likelihood, not a posterior
  ([[L-ADR-0006 Separate likelihood_message from invert]]).
- **To a continuous channel $t$:**

$$
\mu(x_t) \;\propto\; \sum_k \pi_k w_k\, \mathcal N(A_t x_t;\ d_k,\ R),
$$

  a `MixtureBelief` of the canonical Gaussians $\mathcal N^{-1}(A_t^\top R^{-1} d_k,\ A_t^\top R^{-1} A_t)$. All components
  share one precision. **The per-outcome constants are part of the message**: they are the relative
  masses of the components. If $A_t$ has full column rank the components are normalised Gaussians and
  outcome $k$ is weighted by $\pi_k w_k \exp(-\tfrac12 d_k^\top R^{-1} d_k + \tfrac12 \eta_k^\top \Lambda^{-1} \eta_k)$. If not, they are
  improper sites, weighted as the unnormalised exponentials that Mycelium's mixture product expects.

A Gaussian-mixture *input* is first collapsed to its mean and covariance (`lib/Lenticulum.jl/src/gaussian.md`,
"Mixture inputs"). That is assumed-density filtering, and §6 says what it costs.

## 4. The local free energy

The factor's belief $b(x, k) \propto f(x, k)\, \pi(k) \prod_i \mu_i(x_i)$ is, for each outcome, a Gaussian over the
free channels, and in the shared form all of them have one precision. Because $k$ is a channel of
the factor, its entropy is exact:

$$
H(b) = H(q) + H(x \mid k), \qquad q = b(k).
$$

> **Proposition 3.** The local functional $F = \mathbb E_b[-\log f] - H(b)$ is
>
> $$
> F = \underbrace{\sum_k q_k\, \tfrac12 \bigl(\bar r_k^\top Q^{-1} \bar r_k + \operatorname{tr} Q^{-1}\operatorname{Cov} r\bigr)}_{\text{fit}}
> \;+\; \underbrace{\tfrac12 \log\det 2\pi Q - \sum_k q_k \log w_k}_{\text{complexity}}
> \;\underbrace{-\,H(q) - H(x \mid k)}_{\text{negentropy}},
> $$
>
> every term in closed form. It is a KL divergence only when $f$ is normalisable (all $A$ invertible):
> then $F = \mathrm{KL}(b \,\|\, f / Z_f) - \log Z_f$.
>
> *Proof.* $-\log f = \tfrac12 r_k^\top Q^{-1} r_k + \tfrac12\log\det 2\pi Q - \log w_k$, its expectation under the
> Gaussian $b(x \mid k)$ is the mean term plus the trace, and the entropy of $b$ is the chain rule of
> entropy. With $Z_f < \infty$, Gibbs' inequality gives the KL form. $\square$

On a tree with exact messages the graph's Bethe free energy is $-\log p(\text{data})$. The tests check
this for a categorical prior, the factor and clamped data, against the mixture's log-density. The
obstacle is at the **variables**: a continuous variable whose marginal is a mixture has no
closed-form entropy, so the Bethe sum is available only where such variables have degree one or
are clamped.

## 5. Composition gives exactly the finite mixtures

> **Proposition 4.** A graph of switching constraint factors with outcomes $k_1, \dots, k_L$ ($K_l$ states
> each), any linear-Gaussian factors and Gaussian priors, has a joint over its continuous variables
> that is a finite Gaussian mixture with at most $\prod_l K_l$ components. Conversely every
> $K$-component Gaussian mixture is one switching factor (with per-outcome covariances in the general
> form).
>
> *Proof.* For each configuration of the outcomes the joint is a product of Gaussian potentials,
> Gaussian by Proposition 1, weighted by $\prod_l w_{l,k_l}$ times its normaliser. Summing over
> configurations gives the mixture. $\square$

So **composition buys parameter efficiency, not a larger class.** $L$ binary switches give up to $2^L$
components from $O(L)$ parameters, but arranged: with switches feeding a sum node,
$z = c + \sum_l k_l \delta_l + \varepsilon$, the component means form a product grid. Measured on the branch
(E3, [[Composing Stochastic Factors]]):

- this pays exactly when the data have that arrangement;
- the wiring matters: switches feeding a **sum node** add, while several constraints on one shared
  variable **intersect**, and with more switches than dimensions some configurations cannot meet.

## 6. Inference in compositions, and the message rules

Exact inference over $L$ switches enumerates $2^L$ configurations. Message passing is the scalable
route, and the experiments on the branch found where it goes wrong and what fixes it (E6, E7).

- **Belief propagation with collapsed inputs is biased for learning.** At a sum node fed by
  switches, BP collapses the other switches' mixtures to their *prior* moments (one pass of
  assumed-density filtering). Mean field iterates, but with a factorised posterior. With
  overlapping switches both learn parameters 0.4–1.1 nats from the truth, against 0.01–0.05 for
  exact EM. The cause is the **pair moments** $\mathbb E[k_l k_m]$: with exact marginals but factorised pair
  moments the bias is nearly as large. At the true parameters $\mathbb E_z[q_l q_m]$ misses the
  explaining-away covariance, so the truth is not a fixed point of the M-step.
- **Exact blocks plus iteration remove most of it.** A block of $B$ switches is one switching factor
  with $2^B$ outcomes (its configurations), exact inside. Blocks that keep overlapping switches
  together, updated in turn with the other blocks' current means (structured mean field), come within
  0.14–0.16 nats on average, against 0.59–0.61 for the per-switch graph. The blocks alone, in one
  pass of belief propagation, reach only 0.30–0.44.

The iteration is a **mean-field node** in Mycelium (`factor!(b, name, f; rule = MeanField())`,
[[L-ADR-0009 Factor nodes carry a message rule]]). Such a node sends $\exp \mathbb E_q[\log f]$ over its
neighbours' current marginals instead of their cavities. For the switching factor the expectation
needs only means. With $\bar s = \sum_{i \ne t} A_i \mathbb E_q[x_i]$ and $q$ the outcome's marginal:

$$
\mu(x_t) = \mathcal N^{-1}\Bigl(A_t^\top Q^{-1}\bigl(\textstyle\sum_k q_k b_k - \bar s\bigr),\ A_t^\top Q^{-1} A_t\Bigr), \qquad
\mu(k) \propto w_k \exp\bigl(-\tfrac12 (b_k - \bar s)^\top Q^{-1} (b_k - \bar s)\bigr).
$$

`LinearConstraintFactor` has the same rule with one outcome: its belief-propagation message with
each input replaced by a point mass at its mean. A graph with the sum node on `MeanField()` and the
block factors on belief propagation is the hybrid scheme of Riegler et al. (2013), and its
fixed point is block structured mean field. The test builds one with two blocks and checks it
against an independent coordinate-ascent implementation to $10^{-6}$. On the branch, EM through such a
graph reproduces the structured-mean-field fits to $3 \cdot 10^{-8}$.

## 7. Using it

```julia
using Lenticulum, Mycelium, LenticulumCore, LinearAlgebra, LuxCore, Random

# a three-component mixture in the plane, z = b_k + ε, with a prior on the outcome
f  = SwitchingConstraintFactor((z = 2,), 3, 2; noise = 0.1)
ps = (A = (z = Matrix(1.0I, 2, 2),), b = [-1.0 1.0 0.0; 0.0 0.0 2.0], logw = zeros(3))

b = GraphBuilder()
variable!(b, :z, 2); variable!(b, :k, 3)
factor!(b, :data, DataFactor(:z, [0.9, 0.2])); factor!(b, :sw, f)
factor!(b, :pk, PriorFactor(:k, CategoricalBelief([0.5, 0.3, 0.2]), k -> -log([0.5, 0.3, 0.2][k])))
connect!(b, :data, :z, :z; direction = Emitting()); connect!(b, :sw, :z, :z; direction = Absorbing())
connect!(b, :sw, :k, :k); connect!(b, :pk, :k, :k)
g = build(b)
marg, _, st, store = infer!(g, tree_schedule(g), merge(LuxCore.initialparameters(Xoshiro(0), g), (sw = ps,)), NamedTuple())
probabilities(marg.k)        # P(k | z)
```

The tutorial *Mixtures from switching factors* in the API documentation works through one factor,
a grid of switches feeding a sum node, and the three message rules side by side.

## 8. What is not done

- The general form (per-outcome $A_k$, $Q_k$). Its components have different precisions, and improper
  components with different precisions cannot be weighted consistently, so the improper case needs
  a decision first.
- The directed variant, a switching kernel normalised by $Z(x)$, which a reverse diffusion chain
  needs (its gate depends on the state, so a chain of them is not a finite mixture).
- Mixture entropies at variables, hence the Bethe free energy of graphs with mixture marginals of
  degree two or more, and the free energy of graphs with mean-field nodes. Both are on the plan in
  [[Extending Mycelium]].

Related: [[Composing Stochastic Factors]], [[Mixture Belief]], [[Categorical Belief]], [[Bethe Free Energy]], [[Hierarchical Models]], [[Extending Mycelium]]
