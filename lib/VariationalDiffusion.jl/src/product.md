#implementation

> Products of relations. Several implicit learners pooled: the field of the product is the
> weighted sum of their fields, so its stable roots are where all the relations hold at once,
> e.g. the two points where a learned circle meets a learned line. Factors may see different
> variables, with a separator term against double counting. `ProductRelation` is one more
> `AbstractImplicitRelation`, so every query function works on it unchanged, gradients flow
> back into each factor, and it can be sampled, with corrector steps and reweighting.

> Sources: code: `product.jl`, `sampler.jl`, `implicit.jl` (`AbstractImplicitRelation`, `_params_cotangent`), `answers.jl`, `examples/product_sampling.jl`; Hinton, *Training Products of Experts by Minimizing Contrastive Divergence*, Neural Computation 2002; Du et al., *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*, ICML 2023 (adding scores, and why it is not exact at $t > 0$); Roberts & Tweedie, *Exponential Convergence of Langevin Distributions and Their Discrete Approximations*, Bernoulli 1996 (MALA); Del Moral, Doucet & Jasra, *Sequential Monte Carlo Samplers*, JRSS B 2006; Skreta et al., *Feynman-Kac Correctors in Diffusion*, ICML 2025 (reweighting a product of experts)
>
> Bibliography: [[Bibliography#^hinton2002poe|Hinton 2002]] · [[Bibliography#^du2023reduce|Du et al. 2023]] · [[Bibliography#^roberts1996mala|Roberts & Tweedie 1996]] · [[Bibliography#^delmoral2006smc|Del Moral et al. 2006]] · [[Bibliography#^skreta2025fkc|Skreta et al. 2025]]
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
  or a closed-form mixture (the energy of the product is $\sum_i\beta_i U_i$), and so
  `implicit_mixture` with Laplace mass weights;
- `implicit_sample`, when all factors share a noise schedule (§5).

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

## 4. Different variables: separators

A factor may see a subset of the joint coordinates: `relation => coords`. Its field is
evaluated on `z[coords]` and added back at those positions, and the same holds for the
Jacobian block, the JVP and VJP, the parameter cotangent and the energy. Relations on
overlapping variables compose as in a junction tree,

$$p(x, y, z) = \frac{p_A(x, y)\,p_B(y, z)}{p_S(y)},$$

exact when $x \perp z \mid y$ and both factors have the marginal $p_S$ on $y$. The separator is a
relation on the shared coordinates with weight $-1$:

```julia
ProductRelation(mA => [1, 2], mB => [2, 3], mS => [2]; weights = [1, 1, -1])
```

Without it the shared marginal is counted once per factor, $p_S^2$ instead of $p_S$: it is
sharpened, and its modes are reweighted. The constructor requires every coordinate to be seen
by some relation, and the weights of the relations that see a coordinate to sum to a positive
number, so a separator cannot outweigh its factors.

**Measured.** A chain with two blobs at $(-1,-1,-2)$ and $(1,1,2)$, weights 0.8 and 0.2 (closed-form
mixtures of width 0.05, `density_lambda`), `implicit_mixture` with no coordinate clamped:

| | main blob | second blob | sd of $y$ | other stable roots |
|---|---|---|---|---|
| with separator | 0.800 | 0.200 | 0.133 | one, weight $< 10^{-3}$ |
| without | 0.941 | 0.059 | 0.094 | two, weight $< 10^{-3}$ |

$0.941 = 0.8^2/(0.8^2+0.2^2)$: without the separator the weights are those of $p_S^2$, and $y$
is narrower by $\sqrt2$. The extra roots are compromises such as $(-1, 0, 2)$, where $A$ and $B$
disagree about $y$. They are stable but carry no mass. Division by the separator's small
density in the gap is what makes them stable, and it matters for sampling (§5).

The adjoint through such a chain (curves $y = x/2$ and $z = y^2$ with a separator, $x$
clamped) matches central differences to $10^{-5}$ for the clamp and for a parameter of $B$.

## 5. Sampling a product

`implicit_sample` runs the reverse diffusion with the summed noise prediction
$\hat\varepsilon_\Pi = \sum_i\beta_i\hat\varepsilon_i$, the summed score. At $t > 0$ that is the score of
$\prod_i p_{i,t}^{\beta_i}$, not of the noised product (T9). For root finding the error is a
small bias (§3). For sampling it is a wrong distribution: the reverse process does not end at
the product. Two remedies, both options of `implicit_sample`:

- **Corrector steps** (`corrector = :langevin` or `:mala`, Du et al.): after every reverse
  step, `corrector_steps` Langevin steps $x \leftarrow x + \eta s(x) + \sqrt{2\eta}\xi$ with
  $\eta = c\,\sigma_u^2$, which move the chains towards $\pi_u \propto \prod_i p_{i,u}^{\beta_i}$ at the
  current level. `:mala` adds the Metropolis–Hastings test, which needs log-densities.
- **Reweighting** (`smc = true`): a sequential Monte Carlo sampler for the sequence $\pi_u$. The
  ancestral step is the proposal $r(x_u\mid x_t)$, the forward noising $q(x_t\mid x_u)$ the
  backward kernel, and each chain's weight is multiplied by
  $\pi_u(x_u)\,q(x_t\mid x_u)/(\pi_t(x_t)\,r(x_u\mid x_t))$. Chains are resampled
  (systematically) when the effective sample size drops below half, and at the end. Moves
  between reweightings must leave $\pi_u$ invariant, so it combines with `:mala`, not with
  `:langevin`. This is the discrete-time form of the Feynman–Kac correctors of Skreta et al.

**Measured** (`examples/product_sampling.jl`). Two products with exact answers: the chain of §4,
and circle (upper half three times heavier) ∩ line $y = x$ on shared variables, whose exact
product puts 0.751 on the upper intersection. 100 steps, 2000 chains, mean ± sd over six
seeds; "off" is the share of samples near no mode, "spread" the standard deviation within the
main mode (exact 0.050), time per run in seconds:

| sampler | chain: main (0.800) | off | circle: main (0.751) | off (0.008) | spread | time |
|---|---|---|---|---|---|---|
| plain | 0.736 ± 0.014 | 0.146 | 0.655 ± 0.011 | 0.081 | 0.087 | 0.5 |
| Langevin ×3, $c = 0.1$ | 0.766 ± 0.009 | 0.095 | 0.662 ± 0.007 | 0.006 | 0.050 | 2.2 |
| Langevin ×10, $c = 0.5$ | 0.788 ± 0.008 | 0.003 | 0.694 ± 0.005 | 0.006 | 0.049 | 5.3 |
| MALA ×10, $c = 0.5$ | 0.780 ± 0.003 | 0.041 | 0.695 ± 0.012 | 0.007 | 0.049 | 10.7 |
| SMC | 0.813 ± 0.035 | 0.000 | 0.760 ± 0.019 | 0.005 | 0.049 | 0.9 |
| SMC + MALA ×3, $c = 0.5$ | 0.798 ± 0.013 | 0.000 | 0.744 ± 0.019 | 0.006 | 0.050 | 4.0 |

What this shows:

- **Summed scores alone are wrong in both ways.** On the circle, 8% of samples lie off both
  intersections and the spread is 0.087. On the chain, 15% land in the compromise states of §4.
- **Corrector steps fix the shape, not the mass.** A few Langevin steps bring the spread to
  0.050 and the strays to the exact level. The split between modes stays wrong (0.69 against
  0.75): by the time the modes are separated, local moves cannot carry mass across.
- **Reweighting fixes the mass.** SMC alone gets both splits right within its seed-to-seed
  spread, at twice the cost of the plain sampler, and removes the compromise states. MALA moves
  between reweightings reduce the variance on the chain (0.013 against 0.035) by refreshing
  the resampled duplicates.
- Langevin and MALA perform about the same as correctors. MALA's test removes the
  discretisation bias inside a mode, which is small here.

Both remedies need what plain sampling does not. `:mala` and `smc` need log-densities: mixtures,
kernel estimates and energy networks have them, a direct ε-network does not. SMC's weights
degenerate when the proposal is poor; there are only 2000 chains and two modes here, and its
cost in higher dimensions is not measured.

## 6. Limits

- **The separator is supplied, not derived.** It is a relation the user trains on the shared
  marginal (or a kernel estimate of it). Deriving it from a factor would need the factor's
  marginal score, which a joint diffusion model does not give in closed form.
- **Tangent intersections** are ill-conditioned: the Jacobian of the sum is nearly singular,
  so the Laplace covariance blows up along the common tangent. This is the right answer, but
  slow to converge.
- **Empty intersections.** If the relations do not meet, the stable roots are compromises
  between them, weighted by β. They are still returned; nothing flags them yet.
- **Sampling with clamps under `smc`** holds the clamped coordinates at $\alpha_u z_{0,C}$ without
  noise, so the targets are the conditionals at each level; it does not combine with RePaint's
  `resample`.

## 7. Tests

`test/runtests.jl`, "products of relations: intersections":

- the field equals the weighted sum;
- a non-tuple `ps` is rejected;
- two and four roots within 0.05 of the exact intersections;
- the adjoint for the clamp and for one ellipse parameter matches central differences
  (rtol 1e-6);
- the parameter cotangent is a tuple.

"products over different variables: separators": the field, Jacobian blocks, JVP and VJP of a
chain against central differences; the Laplace weights with and without the separator; the
adjoint through a chain of curves; the energy; the constructor's checks.

"sampling a product: corrector steps and reweighting": circle ∩ line, plain (strays above 5%),
Langevin (strays below 1%, spread 0.05), SMC + MALA (upper share within 0.04 of 0.75); SMC on a
single clamped model; the refused combinations.
