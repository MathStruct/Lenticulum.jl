#design #derivation

> A factor learned by the free energy can satisfy its data trivially: a relation that holds
> everywhere (a constant or zero residual), a noise that shrinks with its coefficients, an
> unnormalised potential that grows without bound. What prevents it is always the same thing:
> the factor is −log of a density **normalised in its data channels**, or it is trained by
> comparison. This note goes through every learnable factor family, says which of them are safe,
> which are guarded, and lists the options. Since 2026-11-05 every collapse found is at least warned.

> Sources: LeCun, Chopra, Hadsell, Ranzato & Huang, *A Tutorial on Energy-Based Learning*, 2006, §2 (loss functionals) and §5 (the energy loss and flat energy surfaces); original analysis of this repository's factors. code: `lib/Lenticulum.jl/src/constraint.jl` (`_coeffs`, `output`), `lib/Lenticulum.jl/src/regime.jl`, `lib/Lenticulum.jl/src/m_steps.jl`, `lib/Mycelium.jl/src/free_energy.jl` (`learns_unnormalised`), `lib/AlgebraicFactors.jl/src/toric.jl`, `lib/AlgebraicFactors.jl/src/polynomial.jl`, `lib/ImplicitLayers.jl/src/deq.jl`; tests: `test/lp022.jl`, `lib/AlgebraicFactors.jl/test/lp022.jl`, `lib/Lenticulum.jl/test/learnable_noise.jl`
>
> Bibliography: [[Bibliography#^lecun2006tutorial|LeCun et al. 2006]]
>
> Theory (CT-ML wiki): [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## 1. The problem

A Mycelium graph scores its parameters by the free energy $F(\theta)$. Data enter as point messages
(`DataFactor`), latent variables are integrated out, and

$$
F(\theta) = \sum_f \mathbb E_b\bigl[E_f(\theta)\bigr] - H(b), \qquad E_f = -\log f .
$$

Nothing normalises over the **observed** variables: there is no partition function
$\int e^{-E}\,dy$ over the data anywhere in the graph. So $F$ is $-\log p(\text{data})$ only if every factor that
touches data is itself normalised in those channels. Otherwise $F$ is what LeCun et al. call the
*energy loss* (§2): it pushes the energy of the observed configuration down and nothing else up,
and a model can satisfy it by making the energy small everywhere. [[Energy-Based Learning]] §3 has
the general argument; this note is about the concrete factors.

**Three ways to collapse**, each met in the code:

1. **A constant relation.** A residual $r_\theta(x)$ that is zero for every $x$ (all coefficients 0,
   or a map that is the identity on its own output) puts every point on the relation.
2. **Scale.** If the coefficients and the noise are both learned, $A \to cA$ and $Q \to c^2 Q$ multiply the
   factor by $c^{-m}$: the free energy falls without bound as $c \to 0$.
3. **An unnormalised potential.** $E = -\eta^\top a(x)$ without a log partition falls without bound along
   the observed state's features.

## 2. When a factor is safe

A learned factor is safe on data if, for every value of its parameters, $e^{-E_f}$ is a normalised
density (or probability) in its **data channels**, given the others. Then $F$ is $-\log$ of a
likelihood, which is bounded below for any fixed data set that is not fitted exactly (a perfect fit
of finitely many points is overfitting, not collapse; see the Gaussian row below). There are five
ways to get there, and the factors below use each of them:

| mechanism | what it fixes | example |
|---|---|---|
| a fixed coefficient on the data channel | the change of variables $y \mapsto r$ | `GaussianFactor` ($A_y = I$), `output = :y` |
| a normalised noise model | the volume $\tfrac12 \log\det 2\pi Q$ is paid | learned noise in `GaussianFactor`, `RegimeFactor` |
| a log partition | $\log \sum_x e^{\eta^\top a(x)}$ is subtracted | `ToricFactor(...; normalised = true)` |
| a constrained parameter set | the trivial parameter is excluded | `PolynomialFactor` on the Grassmannian |
| comparison against something | a contrastive term pushes energy up | `RatioFactor` (classification against $q$), denoising |

## 3. The factors

| factor | learned | normalised in its data? | can it collapse? | guard |
|---|---|---|---|---|
| `GaussianFactor` | $A$, $b$, optionally $Q$ | yes: $p(y \mid x)$, $A_y = I$ | no; with learned $Q$, an exact fit drives $Q \to 0$ | noise prior or floor (`lib/Lenticulum.jl/test/learnable_noise.jl`) |
| `GaussianPrior` | the noise, optionally | yes | as above | as above |
| `NonlinearFactor`, `LikelihoodFactor` | nothing (`islearnable = false`) | yes | no | — |
| `LinearConstraintFactor` | $A$, $c$ | only in the residual | **yes**: $A = 0$ (1) | `output = :y`; a warning; `m_step` refuses |
| `SwitchingConstraintFactor` | $A$, $b$, $\log w$ | only in the residual | **yes** (1) | as above |
| `RegimeFactor` | $A_k$, $b_k$, $\log w$, optionally $Q_k$ | only in the residual | **yes** (1), and (2) with learned noise | as above |
| cluster nodes | the clustered factors' | as their factors | as their factors | the factors' guards; tied parameterisations must keep their normalisation (below) |
| `TableFactor` | nothing | — | no | — |
| `ToricFactor` | $\eta$ | only with `normalised = true` | **yes** (3) when unnormalised (the default) | a warning (since 2026-11-04) |
| `PolynomialFactor` | $\Theta$ | no (Sampson energy) | not to $\Theta = 0$; can interpolate | Grassmann parameter; no graph free energy |
| `DEQFactor` | the cell $g$ | no: $\tfrac12\lVert z - g(z, x)\rVert^2$ | **yes** (1) with $z$ observed: $g(z, x) = z$ | a warning when $z$ is observed (since 2026-11-05) |
| `NeuralODEFactor` | the field | no, but $z_1$'s coefficient is fixed | no: $\tfrac12\lVert z_1 - \Phi(z_0)\rVert^2$ is a regression | — |
| `LuxFactor` | the layer | `residual = true`: $\tfrac12\lVert y - \mathrm{layer}(x)\rVert^2$ | no (a fixed-noise regression); default energy 0 | — |
| `GeneratorFactor` | $G$ | energy 0 | nothing to collapse; no signal of its own | learned by comparison |
| `RatioFactor` | the discriminator | by comparison | no flat solution; it estimates $\log p/q$ | contrastive by construction |
| `DiffusionFactor` | the noise predictor | trained offline by the denoising loss | no: $\varepsilon_\theta = 0$ costs $\mathbb E\lVert\varepsilon\rVert^2$ | the denoising objective; its graph free energy is not a training objective |

### Constraint factors (`LinearConstraintFactor`, `SwitchingConstraintFactor`, `RegimeFactor`)

Their energy is $\tfrac12 r^\top Q^{-1} r + \tfrac12\log\det 2\pi Q - \log w_k$ with $r = \sum_i A_i x_i - b_k$: normalised over $r \in \mathbb R^m$,
not over any channel. They are relations, and every coefficient is a parameter by default. Measured
in `test/lp022.jl` (200 points of $y = 2x + 0.5 + 0.3\varepsilon$, $Q = 0.09$): least squares has $F = 32.34$, the
constant relation $A = 0$ has $F = -57.01$, and at least squares the gradient in $A_y$ is $\sum r^2/Q = 178.69$, so
descent shrinks the relation. With learned noise (`RegimeFactor(...; learn_noise = true)`), halving $A$
and the noise's scale lowers $F$ by $N m \log 2$ exactly (2).

**Guards** ([[LP-022 Constraint factors are normalised in their residual, not their channels]]):
- `output = :y` fixes the coefficient of the data channel (identity, or an invertible
  `output_coeff`) and removes it from the parameters. The factor is then the normalised
  conditional density of $y$ given the others, up to $|\det A_y|$, and fits by gradient or EM reach
  least squares (tested).
- Freezing a coefficient (`freeze = (:A,)` or `(:A, :y)`) does the same by hand.
- `free_energy_gradient` warns once when such a factor, learnable and unfrozen and with no output,
  touches a `DataFactor` variable (`Mycelium.learns_unnormalised`).
- The closed-form `m_step` refuses a factor whose every coefficient is learned, because its
  minimiser is $A = 0$.

**Cluster nodes** inherit this. A cluster's free energy is $-\log p(\text{data})$ only when its factors
are normalised. A parameterisation that ties several factors must keep the normalisation itself. In
`lib/Lenticulum.jl/examples/structure_clusters.jl` the switches' weights are normalised per cluster so that the
tables' $\log J$ makes the joint a mixture. An unconstrained per-factor M-step would break that,
and its free energy would no longer be $-\log p$. The example's EM therefore uses its own
constrained M-step.

### Toric factors

`ToricFactor` is the log-linear potential $\exp(\eta^\top a(x))$, unnormalised by default. On observed states
its energy $-\eta^\top a(x)$ has no lower bound. Measured in `lib/AlgebraicFactors.jl/test/lp022.jl` (two
binary variables observed, $\eta = t\, a(x_{\text{obs}})$): $F = -3.34, -30.0, -300.0$ for $t = 1, 10, 100$. With
`normalised = true` the log partition is included and $F = -\log p(x_{\text{obs}}) \ge 0$; it is 0.0 there, the
single observation fitted perfectly. The gradient shows the same thing (`toric.md`):
$\partial_\eta F = -\mathbb E_b[a] + [\text{normalised}]\,\mathbb E_{p_\eta}[a]$, and the contrastive second term is exactly the partition's.
Unnormalised toric factors are right as potentials between latent variables, and wrong as
learned models of data. Since 2026-11-04 `learns_unnormalised(::ToricFactor)` makes
`free_energy_gradient` warn for them.

### Polynomial factors

`PolynomialFactor` scores $\Theta v(x)$ by the Sampson distance $\tfrac12 r^\top (J\Sigma J^\top)^{-1} r$ with $J = \Theta\, Dv$. That is scale
invariant in $\Theta$, and $\Theta$ lives on the Grassmannian (orthonormal rows), so the zero polynomial is not a
parameter (`polynomial.md`). It is fitted by `fit_sampson`, and no graph free energy is supplied, so
it is not trained by $F$. What remains is the ordinary algebraic overfitting: with more monomials than
points, a nonzero polynomial can vanish on all of them. That is a choice of degree, which
`fit_nullspace`'s nullity diagnoses. It is not a collapse of the parameterisation.

### Implicit layers

- **`DEQFactor`**: $\tfrac12\lVert r\rVert^2$ with $r = z - g_\theta(z, x)$ and no normalising term (`deq.md` §4.3). With $z$ and $x$
  both observed and the energy used as a training objective (`free_energy_gradient` scores fixed
  incoming endpoints, `deq.md` §4.5), a cell with $g(z, x) = z$ puts every pair on the relation. It is a
  collapse of kind (1), because the coefficient of $z$ is not fixed: $z$ appears inside $g$. Measured in
  `lib/ImplicitLayers.jl/test/lp022.jl` (50 pairs of $z = 2x + 0.3 + 0.1\varepsilon$, a linear cell
  $g = Wz + Ux + b$): the generating relation has $F = 0.237$, the identity cell ($W = 1$, $U = b = 0$) $F = 0$, and
  at the generating relation the gradient in $W$ is $-\sum_i r_i z_i = 0.62$, so descent moves $W$ towards 1. Since
  2026-11-05 `learns_unnormalised(::DEQFactor, observed)` makes `free_energy_gradient` warn when the
  state channel is observed (the input alone is the usual case and does not warn).
- **`NeuralODEFactor`**: $r = z_1 - \Phi_\theta(z_0)$ (`neuralode.md` §4.4). The coefficient of $z_1$ is fixed, so this is a
  regression with unit noise, safe in the sense of §2.
- **`LuxFactor`**: energy zero by default, so it delegates to other factors and has nothing to
  collapse. With `residual = true` it is $\tfrac12\lVert y - \mathrm{layer}_\theta(x)\rVert^2$, a fixed-noise regression.

### Comparison-trained families

- **`GeneratorFactor`**: energy zero (`generator.md` §4.2). It cannot learn from its own free energy
  and is trained by comparison, as Mohamed & Lakshminarayanan argue implicit models must be
  ([[Implicit Generative Models]]).
- **`RatioFactor`**: a classifier between $p$ and $q$. The contrastive term is the other class, so its
  logit converges to $\log p/q$ (`ratio.md` §1), which is noise-contrastive estimation
  ([[Training Energy-Based Models]]). The generator side can still fail in the usual GAN ways
  (mode collapse); nothing here measures that.
- **`DiffusionFactor`** and the implicit relations: the predictor is trained offline by the
  denoising loss $\mathbb E\lVert\varepsilon_\theta(x_t, t) - \varepsilon\rVert^2$, whose target is the injected noise. A constant predictor costs
  $\mathbb E\lVert\varepsilon\rVert^2$, so there is no flat solution. The graph free energy of a diffusion factor is Monte Carlo
  and not a training objective (`factor.md` §5.3). The relation it defines (the stable roots of the
  smoothed score) has a different defect: those roots are modes of $p_t$, not points of the data
  manifold ([[LP-023 Findings of the external reviews of 2026-11-04]], "implicit-relation calibration").

## 4. Options for a factor that is not safe

In order of preference:

1. **Make it a conditional density.** Normalise it in its data channel
   (`OverlayFactor(:f => f; normalise = :y)`, since 2026-11-05: every coefficient stays learnable, and the
   relation becomes the conditional Gaussian of $y$; `lib/Mycelium.jl/src/overlay.md` §2), fix the
   coefficient of the data channel (`output`), or write
   the factor as $y = h_\theta(x) + \varepsilon$ with $y$ outside $h$. For a DEQ that means training through a
   downstream loss on the solved state, or a residual $z - h(x)$ with $z$ not inside $h$.
2. **Pay the volume.** Learn the noise together with its $\tfrac12 \log\det 2\pi Q$, and guard against an exact
   fit with a noise prior or floor. For an invertible map (a flow) the log-Jacobian is the volume
   term: $\log\lvert\det(I - \partial_z g)\rvert$ would make a DEQ a density in $z$. That isn't implemented.
3. **Normalise over the states.** Use a log partition where it is computable (`ToricFactor(...;
   normalised = true)`, categorical factors).
4. **Constrain the parameters** away from the trivial set: a manifold (the Grassmannian), a fixed
   norm, a frozen block (`freeze`).
5. **Train by comparison**: classification against noise or a reference (`RatioFactor`, NCE),
   denoising, or a margin loss with negatives (LeCun et al.'s generalised margin losses). The
   framework has no slot for a loss *functional* ([[Energy-Based Learning]] §3), so this happens
   outside the free energy.

## 5. What the code enforces today

| check | where | covers |
|---|---|---|
| warning in `free_energy_gradient` | `Mycelium.learns_unnormalised`, `_unnormalised_nodes` | the three constraint factors without `output`; `ToricFactor` without `normalised` (any observed channel) |
| refusal in the closed-form M-step | `lib/Lenticulum.jl/src/m_steps.jl` | constraint factors with every coefficient learned |
| `output` | the three constraint factors | makes them conditional densities |
| parameter validation | `validate_parameters` | an `output` coefficient passed as a parameter |
| warning in `free_energy_gradient`, by channel | `learns_unnormalised(f, observed)` | `DEQFactor` with its state observed |

The warning fires only for data that reach the factor through a `DataFactor`. Pinned variables and
point messages from other factors are not detected.

## Related

[[Energy-Based Learning]] · [[Training Energy-Based Models]] · [[Bethe Free Energy]] ·
[[Switching Constraint Factors]] · [[LP-022 Constraint factors are normalised in their residual, not their channels]]
