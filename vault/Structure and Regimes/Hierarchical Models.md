#model #example

> A hierarchical model is an ordinary factor graph as long as the shared quantities are
> **variables**: the hyperprior is one more factor, attached to a shared variable, and it never
> touches a local factor's parameters. Polarity gives both halves of partial pooling (shrinkage
> and pooling) from one factor. Only a hierarchy over the **parameters** of learned factors needs
> the wiring of [[Unusual Wirings]]. Three examples, each checked against a dense Gaussian.

> Sources: Rubin, *Estimation in Parallel Randomized Experiments*, Journal of Educational Statistics 6(4), 1981 (the eight-schools data); Weiss & Freeman, *Correctness of Belief Propagation in Gaussian Graphical Models of Arbitrary Topology*, Neural Computation 13(10), 2001 (converged loopy Gaussian BP has exact means); the analysis is original to this vault. code: `examples/hierarchical_models.jl`, `test/hierarchical_models.jl`
>
> Bibliography: [[Bibliography#^rubin1981parallel|Rubin 1981]] · [[Bibliography#^weiss2001gaussian|Weiss & Freeman 2001]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## 1. Where the hierarchy lives

A two-level model, groups $j$ with observations $i$:

$$
y_{ij} \sim p(y \mid \theta_j), \qquad \theta_j \sim p(\theta \mid \phi), \qquad \phi \sim p(\phi).
$$

```text
               ┌─────┐
               │p(φ) │                        φ : the shared (population) variable
               └──┬──┘
                  φ
       ┌──────────┼──────────┐
   ┌───┴───┐  ┌───┴───┐  ┌───┴───┐
   │p(θ₁|φ)│  │p(θ₂|φ)│  │p(θ₃|φ)│            one population factor per group
   └───┬───┘  └───┬───┘  └───┬───┘
       θ₁         θ₂         θ₃               group variables
     ┌─┴─┐      ┌─┴─┐      ┌─┴─┐
     y₁ y₁      y₂ y₂      y₃ y₃               local factors, with their data
```

Nothing here is a new kind of edge. The factor $p(\theta_j \mid \phi)$ is a relation between two
variables like any other, and "long range" has no meaning in a factor graph: what makes it a
hierarchy is only that many factors share the variable $\phi$. The local factors are untouched;
the hyperprior sees their *channels* $\theta_j$, never their parameters.

Which quantities are variables and which are parameters is the modelling choice that decides
everything below:

| the hierarchy runs over | example | wiring | today |
|---|---|---|---|
| **variables** (means, effects, latent states) | random intercepts; random slopes written with the weights as channels | ordinary factor graph | works; exact on trees in the linear-Gaussian fragment (§2, §3) |
| **variances of variables** | $\theta_j \sim \mathcal N(\mu, \tau^2)$ with $\tau$ unknown | ordinary, but the factor $p(\theta_j \mid \mu, \tau)$ is not jointly Gaussian | scan $\tau$ as a parameter (§2), or a non-Gaussian belief on $\tau$ (not in LenticulumCore) |
| **parameters `ps` of learned factors** | one network per group with a shared prior on the weights; a shared hyperprior on several factors' noise | output into a weight, beliefs on weights ([[Unusual Wirings]] §2, §4) | `expose` ([[L-SPEC-003 Exposed Parameters and Energy Channels]] §1): exact for linear-Gaussian weights (§3); networks and noise not yet |

The third row turns into the first once the parameter is exposed: an exposed weight is a
variable, and the hyperprior attaches to it as in the picture. Random slopes (§3) do that
exposure by hand for a linear model.

## 2. Partial pooling is two polarities of one factor

The population factor $p(\theta_j \mid \mu)$ sends two messages, and they are the two halves of
partial pooling:

| message | polarity | what it does |
|---|---|---|
| $\mu \to \theta_j$ | $\mu$ observed, $\theta_j$ unobserved | **shrinkage**: the population acts as a prior on group $j$ |
| $\theta_j \to \mu$ | $\theta_j$ observed, $\mu$ unobserved | **pooling**: each group's evidence informs the population |

On a tree (any two-level hierarchy whose groups meet only through $\mu$) two sweeps give the
exact posterior, and the [[Bethe Free Energy]] at the fixed point is $-\log p(y)$, the marginal
likelihood used to choose the population variance.

**The eight schools.** Rubin's (1981) coaching experiments: eight schools, an estimated effect
$y_j$ with known standard error $s_j$ each. The model is $y_j \sim \mathcal N(\theta_j, s_j^2)$,
$\theta_j \sim \mathcal N(\mu, \tau^2)$, $\mu \sim \mathcal N(0, 100^2)$. For each $\tau$ the graph is
linear-Gaussian, and the posterior means (from `examples/hierarchical_models.jl`) move from the
raw estimates towards the pooled mean as $\tau$ falls:

| school | $y_j$ | $s_j$ | $\tau = 100$ | $\tau = 25$ | $\tau = 10$ | $\tau = 5$ | $\tau = 0.5$ |
|---|---|---|---|---|---|---|---|
| A | 28 | 15 | 27.55 | 22.83 | 14.22 | 9.85 | 7.70 |
| B | 8 | 10 | 8.00 | 8.06 | 8.05 | 7.87 | 7.68 |
| C | −3 | 16 | −2.73 | 0.33 | 4.98 | 6.87 | 7.66 |
| D | 7 | 11 | 7.01 | 7.24 | 7.60 | 7.69 | 7.67 |
| E | −1 | 9 | −0.93 | 0.09 | 3.07 | 5.75 | 7.65 |
| F | 1 | 11 | 1.08 | 2.21 | 4.89 | 6.66 | 7.66 |
| G | 18 | 10 | 17.90 | 16.68 | 13.05 | 9.87 | 7.70 |
| H | 12 | 18 | 11.87 | 10.79 | 9.02 | 8.13 | 7.68 |

Schools with large standard errors (A, C, H) move furthest: their own evidence is weak, so the
shrinkage message dominates. At every $\tau$ the means, variances and free energy agree with the
dense Gaussian to $1.2\cdot10^{-13}$.

**Choosing $\tau$.** $\tau$ is the noise of the population factor, a parameter, so it is not
inferred by message passing. Two ways to handle it, both shown in the example:

- *Type-II maximum likelihood* minimises $-\log p(y \mid \tau)$, which the graph returns as its free
  energy (agreeing with the dense value to $6.4\cdot10^{-14}$ over 160 values of $\tau$). For these
  data the minimum is at the lower end of the scan: maximum likelihood pools the schools
  completely, a well-known weakness of plugging in a point estimate of $\tau$.
- *A mixture over $\tau$*: with a uniform prior on a grid in $[0.25, 40]$, the weights
  $\propto e^{-F(\tau)}$ give $\mathbb E[\tau \mid y] = 6.61$, and each $\theta_j$'s posterior is a
  mixture of the per-$\tau$ Gaussians. School A comes out at $11.41 \pm 8.34$, between complete
  pooling and its raw 28. The mixture is assembled outside the graph; inside it would be a
  belief on $\tau$ that is not Gaussian, the second row of the table in §1.

**A new school.** Add a ninth $\theta$ with a population factor and no data. Its marginal is the
predictive effect of an unseen school: at $\tau = 10$, $\theta_9 \sim \mathcal N(8.10, 11.42^2)$, with
$\mu \sim \mathcal N(8.10, 5.51^2)$, so $\operatorname{Var}\theta_9 = \operatorname{Var}\mu + \tau^2$. No
new machinery: the same factor, queried with nothing on one side.

## 3. Random slopes: exposing the weights by hand

Group regressions $y_{ij} = \beta_{j1} + \beta_{j2}x_{ij} + \varepsilon$ with
$\beta_j \sim \mathcal N(\bar\beta, \Sigma_\beta)$. Written the usual way, $\beta_j$ would be the
*parameters* of a regression factor, and a prior shared across groups would be the third row of
§1. Written as a factor graph, the roles swap: the design matrix $X_j$ is the parameter `A` of a
`GaussianFactor`, and $\beta_j$ is its input **channel**, so the population factor
$\bar\beta \to \beta_j$ attaches to it like to any variable. The graph is again a tree, and with
five groups of six points the posterior (means, $2\times2$ covariances and $-\log p(y)$) agrees
with the dense Gaussian to $2.2\cdot10^{-10}$.

For a linear model this rewrite is free, and since 2026-10-30 it is also unnecessary: the
usual way round works too. `expose(GaussianFactor(2 => 1), (:A => :w,))` turns the weight of
an ordinary one-observation regression factor into a channel; connecting every observation of
group $j$ to one variable $\beta_j$ ties the weight, and the population factor attaches to
$\beta_j$ as before. With the same data this graph (thirty observation factors) agrees with the
dense reference to rtol $10^{-9}$ in the means and $10^{-10}$ in $-\log p(y)$
(`test/exposure.jl`). For a neural network the message to the weights would no longer be exact
(a network is not linear in its weights; [[Unusual Wirings]] §2), and no approximate one is
implemented yet.

## 4. Crossed effects: hierarchy with loops

Not every hierarchy is a tree. Students $r$ answering items $c$, with
$y_{rc} = a_r + b_c + \varepsilon$ and priors on both, put every observation on a row effect *and*
a column effect: a full $R \times C$ design has $(R-1)(C-1)$ independent loops. Each observation is
one `LinearConstraintFactor` on $(a_r, b_c)$ with the reading as its constant.

With $R = 4$, $C = 3$ (six loops) the flooding schedule converges, and:

- the **means** agree with the exact posterior to $1.3\cdot10^{-13}$, as Weiss & Freeman (2001) prove
  for any Gaussian graph on which belief propagation converges;
- the **variances** do not: they are 0.60 to 0.65 of the exact ones, so the loopy posterior is
  overconfident here.

So crossed and multiple-membership hierarchies are expressible, with point estimates you can
trust and uncertainties you cannot; [[Loopy Message Passing]] discusses remedies.

## 5. Summary

- A hierarchy over variables is contained in implicit learning: the hyperprior is a factor,
  and partial pooling is two polarities of it.
- Nested hierarchies are trees and exact in the linear-Gaussian fragment; crossed ones have
  loops, with exact means and approximate variances.
- Unknown variances of the hierarchy are parameters today: scan them with the free energy, or
  mix over them outside the graph.
- A hierarchy over the parameters of learned factors needs those parameters exposed as
  channels: done exactly for linear-Gaussian weights (`expose`), not yet for networks.

Related: [[Unusual Wirings]], [[Everything is a Factor]], [[Bethe Free Energy]], [[Loopy Message Passing]], [[Channels and Polarity]], [[LP-005 Learning the noise covariance]]
