#implementation

> Products of relations. Several implicit learners on the **same** variables, pooled: the field
> of the product is the weighted sum of their fields, so its stable roots are where all the
> relations hold at once, e.g. the two points where a learned circle meets a learned line.
> `ProductRelation` is one more `AbstractImplicitRelation`, so every query function works on it
> unchanged, and gradients flow back into each factor.

> Sources: code: `product.jl`, `implicit.jl` (`AbstractImplicitRelation`, `_params_cotangent`), `answers.jl`; Hinton, *Training Products of Experts by Minimizing Contrastive Divergence*, Neural Computation 2002; Du et al., *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*, ICML 2023 (adding scores, and why it is not exact at $t > 0$)
>
> Bibliography: [[Bibliography#^hinton2002poe|Hinton 2002]] · [[Bibliography#^du2023reduce|Du et al. 2023]]
>
> Theory: [[Composing Diffusion Factors]] · [[Implicit Diffusion Learners]] · [[Open Problems in Implicit Diffusion Learning]]

## 1. The construction

An implicit learner's field $g(z) = \sum_k w_k\lambda_k\,(\varepsilon_\theta(\alpha_k z + \sigma_k\varepsilon_k, t_k) - \varepsilon_k)$
is, for an ideal predictor, the gradient of a smoothed negative log-density. A weighted sum of
such fields is therefore the field of the product of the densities:

$$g_\Pi(z) = \sum_i \beta_i\, g_i(z) \quad\longleftrightarrow\quad p_\Pi \propto \prod_i p_i^{\beta_i}.$$

`ProductRelation(m₁, m₂, …; weights = β)` implements exactly the three functions the solvers
use:

| function | product |
|---|---|
| `prior_field` | $\sum_i \beta_i g_i$ |
| `prior_jacobian` | $\sum_i \beta_i J_i$ |
| `_params_cotangent` | a tuple: factor $i$ gets the adjoint $\beta_i\lambda$ |

Parameters and states are tuples, one entry per factor: `ps = (ps₁, ps₂)`. Passing a single
parameter set raises an `ArgumentError`. Products nest, because a product is itself an
`AbstractImplicitRelation`.

## 2. What works on it unchanged

- `implicit_infer`: one answer, with clamps (`ρ = Inf`) and soft anchors as usual;
- `implicit_roots`: all answers by multi-start, e.g. the intersection points;
- `implicit_laplace`: a covariance per answer;
- `implicit_pullback`: the IFT adjoint. The cotangent of the parameters is a tuple, so a loss
  on the intersection trains every factor;
- `implicit_energy`, and ordering roots by energy, when every factor is an `EnergyNetwork`
  (the energy of the product is $\sum_i\beta_i U_i$).

The refactor that made this possible is small. `implicit.jl` now types its functions with
the abstract supertype, and moves the parameter loop of the pullback into `_params_cotangent`,
the one place where a product differs.

## 3. Accuracy: the smoothing bias

Pooling is exact only at zero noise. At the field's noise levels the product of the smoothed
densities is not the smoothed product, $q_t * (p_1p_2) \ne (q_t * p_1)(q_t * p_2)$
([[Open Problems in Implicit Diffusion Learning]] T9). For root finding this produces the same kind of
bias that a single relation already has (its ridge sits slightly on the inside of a bend), not a
failure. Closed-form mixtures (`GaussianMixtureEps`, $s = 0.05$), the default `field_nodes`:

| product | answers found | exact | largest distance to exact |
|---|---|---|---|
| circle ∩ line $y = x$ | 2 | 2 | 0.022 |
| circle ∩ ellipse (1.5, 0.6) | 4 | 4 | 0.029 |

The adjoint is exact for the field that is actually solved. Clamping $x = 0.95$ on circle ∩
ellipse, $\partial y/\partial x$ is −0.65669290204 by the adjoint and −0.65669290206 by finite
differences. A parameter of the ellipse gives 0.0220369733 by both.

## 4. Limits

- **Same variables only.** All factors see the whole joint space. Factors on *different* but
  overlapping variable sets (a product with a separator, $p_Ap_B/p_S$) are not covered.
- **Tangent intersections** are ill-conditioned: the Jacobian of the sum is nearly singular,
  so the Laplace covariance blows up along the common tangent. This is the right answer, but
  slow to converge.
- **Empty intersections.** If the relations do not meet, the stable roots are compromises
  between them, weighted by β. They are still returned; nothing flags them yet.
- **No correction steps.** The MCMC corrections of Du et al. are for sampling and are not built
  (I8).

## 5. Tests

`test/runtests.jl`, "products of relations: intersections":

- the field equals the weighted sum;
- a non-tuple `ps` is rejected;
- two and four roots within 0.05 of the exact intersections;
- the adjoint for the clamp and for one ellipse parameter matches central differences
  (rtol 1e-6);
- the parameter cotangent is a tuple.
