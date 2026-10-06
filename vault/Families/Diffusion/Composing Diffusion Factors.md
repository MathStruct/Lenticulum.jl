#design #theory

> Several learned relations on the same variables can be **pooled**: add their fields, and the
> answers of the sum are the configurations that satisfy all of them, e.g. the intersection points
> of a learned circle and a learned line. This is a product of experts, read as a relation. It is
> exact at zero noise and slightly biased at the noise levels the learner works at, the same bias a
> single relation already has. Implemented as `ProductRelation` ([[product]]).

> Sources: original to this vault (design and analysis); Hinton, *Training Products of Experts by Minimizing Contrastive Divergence*, Neural Computation 2002; Du et al., *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*, ICML 2023; code: `product.jl`, `answers.jl`
>
> Bibliography: [[Bibliography#^hinton2002poe|Hinton 2002]] · [[Bibliography#^du2023reduce|Du et al. 2023]]
>
> Theory (CT-ML wiki): [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category)

## 1. The idea

An implicit learner turns a dataset of joint configurations into a relation: the set of stable
roots of its field $g$, which is (for an ideal model) the gradient of a smoothed negative
log-density ([[Implicit Diffusion Learners]]). Two such learners $g_1, g_2$ trained on
**different** data over the same variables define two relations $R_1, R_2$. Their
**intersection** $R_1 \cap R_2$ is the set of configurations both datasets allow, and it is found
by solving

$$\beta_1 g_1(z) + \beta_2 g_2(z) = 0,$$

the field of the product density $p_1^{\beta_1}p_2^{\beta_2}$. Nothing is retrained: the
intersection of two relations is a query on two models that already exist.

Examples: a circle and a line (two points), a circle and an ellipse (four points), a learned
dynamics constraint and a learned measurement model (the states consistent with both), or two
experts trained on different sensors.

## 2. Conditioning is a special case

The learner's soft clamp, the residual $g(z) + \rho^2\odot(z - z_0)$, is already a product: the
second term is the field of an axis-aligned Gaussian centred at the anchor $z_0$, with
precision $\rho^2$. A hard clamp ($\rho = \infty$) is the limit, a Dirac on the clamped
coordinates. So **every query was always an intersection**, of the learned relation with the
relation "these coordinates have these values". `ProductRelation` lets the second factor be
learned as well.

## 3. What the weights mean

$\beta_i$ is a tempering exponent: how strongly factor $i$ pulls. On an exact intersection both
fields vanish, so the weights barely move the answer, only its conditioning and the Laplace
covariance (the Jacobian is $\sum_i\beta_iJ_i$). They matter when the relations **do not**
meet: the stable roots are then compromises, nearer the factor with the larger weight.
Nothing flags such a compromise yet. The residual norms of the separate factors at the root
would be the natural diagnostic ([[Open Problems in Implicit Diffusion Learning]] T4).

## 4. Exactness: the obstruction at $t > 0$

Noising does not commute with products, $q_t * (p_1p_2) \ne (q_t*p_1)(q_t*p_2)$, so the sum of
fields at a noise level $t$ is not the field of the smoothed product (T9; [[Language Models]]
§7). For **sampling** this is serious, and Du et al. correct it with MCMC steps at each level.
For **root finding** it is much milder:

- the field's noise levels are small, and the learner's answers are already biased by the
  smoothing for a single relation;
- the product inherits a bias of the same size. Measured with closed-form models, all
  intersections are found, within 0.022 (circle ∩ line) and 0.029 (circle ∩ ellipse) of the
  exact points;
- the bias should shrink with the noise levels, as it does for a single relation (expected,
  not yet measured for products).

The adjoint is exact for the field that is solved, so training through an intersection is
unaffected (checked against finite differences, [[product]] §3).

## 5. Geometry: transversal and tangent intersections

Where the relations cross transversally, $\sum_i\beta_iJ_i$ is well conditioned and the answer
is sharp. Where they touch, it is nearly singular along the common tangent. The Laplace
covariance then becomes long in that direction, which is the honest answer: the data barely
determine the point. This is the composite version of the branch points of a single relation
(T3).

## 6. What is not covered

- **Different variable sets.** A product here requires every factor to see the whole joint
  space. Relations on overlapping subsets ($R_A$ on $(x,y)$, $R_B$ on $(y,z)$) compose by
  sharing $y$; as densities this is $p_Ap_B/p_S$, with a separator term to avoid double
  counting the shared marginal. That is the factor-graph composition ([[Messages are Inversions]],
  [[The Diffusion Factor]] §4) and a separate project.
- **Logic beyond "and".** Union is a mixture of the densities, not a sum of fields; negation
  ($p_1/p_2^\beta$) is a difference of fields and has no stable roots where $p_2$ is flat. See
  [[Belief Algebra]] for the corresponding operations on beliefs.

Related: [[product]], [[Implicit Diffusion Learners]], [[Inference Signatures]],
[[Open Problems in Implicit Diffusion Learning]], [[Belief Algebra]], [[Kernel Methods for Implicit Learning]]
