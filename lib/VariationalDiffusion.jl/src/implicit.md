#implementation

> Implicit inference and its backward pass for a diffusion model: a deterministic residual field,
> a solver that only reports roots it trusts, and an adjoint that turns one linear solve into
> gradients for parameters, inputs, anchors and precisions. The theory is in
> [[Implicit Diffusion Learners]] and [[Backpropagation through Implicit Inference]].

> Sources: code: `implicit.jl`; the trust-region solver (§11): Nocedal & Wright, *Numerical Optimization*, 2nd ed., ch. 4 and 7; Steihaug 1983
>
> Bibliography: [[Bibliography#^nocedal2006numopt|Nocedal & Wright 2006]] · [[Bibliography#^steihaug1983cg|Steihaug 1983]]
>
> Theory (CT-ML wiki): [Least Fixed Point](https://mathstruct.org/CategoryTheory-ML-Wiki/Least-Fixed-Point) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category)

## 1. What is in the file

| name | is |
|---|---|
| `FieldNodes`, `field_nodes`, `noisefree_nodes` | the fixed $(t_k, \varepsilon_k, w_k)$ that make the field deterministic |
| `ImplicitDiffusion` | the relation: a `NoisePredictor` plus a node set plus λ; `batched = true` evaluates one noise level per network call |
| `prior_field`, `prior_jacobian`, `prior_vjp`, `prior_jvp` | $g_\theta(z)$, the block of $\partial g/\partial z$ on chosen coordinates (`coords`), $(\partial g/\partial z)^\top w$ and $(\partial g/\partial z)\,v$ |
| `implicit_residual` | $r = g + \rho^2\odot(z - z_0)$ on the free coordinates |
| `implicit_infer` → `ImplicitSolution` | a root, with `residual`, `iters`, `converged`, `stable`; `linsolve = DenseSolve()` or `KrylovSolve()` ([[krylov]]) |
| `implicit_pullback` | the adjoint: `(z₀ = …, ρ = …, ps = …)` |

## 2. Design decisions

- **Deterministic by construction.** RED-Diff redraws $(t, \varepsilon)$ every step; here the nodes are fixed, so inference is a function and the implicit function theorem applies. `field_nodes` uses antithetic pairs so that the $-\varepsilon$ control variate cancels exactly.
- **Polarity as a precision vector**, the same encoding as `precision_vector(f, polarity)`: `Inf` is a hard clamp, imposed by projection, never as a penalty.
- **The solver trusts Newton only where it should.** A Newton step is taken when the symmetric part of $J_{FF}$ is positive definite, with backtracking on $\lVert r_F\rVert$; otherwise a fixed descent step along the field. Newton is attracted to every root, saddles and maxima included; restricting it to locally convex regions keeps it on stable roots.
- **Two flags, not one.** `converged` (the residual is small) and `stable` ($\operatorname{sym}J_{FF}\succ0$) are separate, because a converged unstable root is on the zero set but not in the relation one wants.
- **The pullback refuses non-converged states.** The IFT says nothing about a non-root, and silently returning a gradient there is what made the first prototype diverge.
- **No AD dependency.** The input Jacobian has a finite-difference fallback (`epsilon_jacobian`); the parameter VJP is an interface method (`epsilon_vjp_params`) that closed-form predictors implement exactly and a Lux network gets from the AD backend in its `ad` field ([[backends]]). A model with no parameters skips it.

## 3. Implementation difficulties

1. **The tolerance floor.** The field of a narrow mixture at small $t$ has a floating-point noise floor around $10^{-11}$ (it scales like $1/\sigma_t$). An absolute tolerance of $10^{-11}$ made a third of the solves "fail"; the tolerance is relative, `tol * (1 + ‖z‖)`.
2. **The prototype's solver.** A version with a fixed descent step in non-convex regions and an energy line search dropped about 60% of solves during training, and its training stalled at RMSE 0.15. The package's solver (Newton where convex, residual backtracking) dropped none and reached 0.012.
3. **Smoothing levels.** With RED-Diff's training range of noise levels, the deterministic field of a circle has its only root at the centre. The default `levels` stop at $t = 0.05$; the trade-off is measured in [[Implicit Diffusion Learners]] §5.
4. **Dense Jacobians.** The block $J_{FF}$ is formed densely by default, which is fine while few coordinates are free (§6) but not when a whole image is free. `linsolve = KrylovSolve()` replaces it by GMRES on Jacobian-vector products in the Newton step and on vector-Jacobian products in the adjoint, and by a Lanczos test of stability ([[krylov]]).

## 4. How it is tested

`test/runtests.jl`, against the closed-form mixture of [[analytic]] and the Gaussian oracle:
- the field equals $\nabla\Phi$ and its Jacobian is symmetric;
- both branches of the circle are found, and they merge near the branch point;
- the smoothing bias is monotone in the levels and matches $R - (s^2+\sigma^2/\alpha^2)/(2R)$;
- noise-free roots are Tweedie fixed points;
- the Gaussian root and its derivative match the closed form;
- every adjoint cotangent matches finite differences of the full solve;
- training through inference reduces the loss by more than 50×;
- the Krylov solver, adjoint, Laplace marginals and mixtures agree with the dense ones, and `prior_jvp` equals $Jv$ on every backend;
- Zygote through `implicit_infer` (its `rrule`) equals `implicit_pullback` and finite differences;
- the free-coordinate block equals the slice of the full Jacobian and `prior_vjp` equals $J^\top w$, for the closed form, finite differences, Zygote, ForwardDiff, Enzyme, energy networks, products and (separate suite) Reactant; batched and unbatched nodes give the same field, Jacobian, answer and gradient;
- `implicit_roots` finds both branches of the circle, and the Laplace covariance of a Gaussian
  query equals the closed form $1/(\kappa + \rho^2)$.

## 5. All answers, and how sure (`answers.jl`)

Two functions build on `implicit_infer` without changing it.

**`implicit_roots`** solves from `z₀` and from perturbed starts (normal on the free coordinates,
the clamped ones fixed), keeps converged and stable answers, and removes duplicates. For an
energy-parametrised predictor the answers are ordered by the query's energy. On the circle,
"$y$ given $x = 0.6$" returns both branches; near $x = \pm 1$ they merge into one. It finds
what its starts reach: no guarantee of completeness
([[Open Problems in Implicit Diffusion Learning]] T1).

**`implicit_laplace`** inverts the symmetric part of $J_{FF}$, the residual's Jacobian on the
free coordinates at the answer, which is the curvature of the query's energy. The subtle part
is the units. With $x_k = \alpha_k z + \sigma_k\varepsilon_k$ and $\varepsilon_\theta = -\sigma_t\nabla_x\log p_t$,

$$
g(z) = \nabla_z\Bigl[\sum_k w_k\,\lambda\,\frac{\sigma_k^2}{\alpha_k^2}\bigl(-\log p_{t_k}(x_k)\bigr)\Bigr] + (\text{terms linear in } z),
$$

so the prior term is a sum of smoothed negative log-densities with weights
$\lambda\sigma_k^2/\alpha_k^2$. They sum to one exactly when $\lambda = 1/\sum_k w_k\sigma_k^2/\alpha_k^2$,
`density_lambda(schedule, nodes)`, and then the energy is a proper negative log-density in
the units of the clamp's Gaussian likelihood, and the covariance is in data units. The
linear terms come from the noise draws and cancel for antithetic nodes. For queries with only
hard inputs and free outputs, λ scales the curvature and leaves the answers unchanged.

On the circle with this λ, the standard deviation of $y$ is 0.135 at $x = 0$ (the smoothed
ring's width: data width 0.05 combined with the noise levels gives about 0.12), 0.17 at
$x = 0.6$ (the vertical line crosses the ring obliquely), and it grows towards the branch
points (0.31, then 0.41) and off the circle (0.53). Calibration is to the density **smoothed
at the field's noise levels**, so it is wider than the data's own spread, the same smoothing
that biases the answers inwards ([[Implicit Diffusion Learners]] §5).

**`laplace_belief`** returns that covariance as a `GaussianBelief` (marginalised onto chosen
free coordinates), and **`implicit_mixture`** combines all answers into a `MixtureBelief`. Its
weights are Laplace estimates of each answer's probability mass,
$w_k \propto e^{-E(z_k)}\,\det(\Sigma_k)^{1/2}$, with $E$ the query's energy. That needs an
energy: energy networks have one, and so does the closed-form mixture, whose energy is the
same formula with $E_\theta = -\log p_t$ in closed form (so ordering in `implicit_roots` uses it
too). With $x = 0.3$ on a circle whose upper half carries three times the data, the upper
answer gets weight 0.78. The unweighted circle gives 0.54 rather than 0.5, because the field's
random noise draws are not symmetric; the ratio of the two odds is 3.0. For plain ε-networks
the weights are equal.

**`coverage`** (`coverage.jl`) says whether an answer lies where the model has seen data: the
distance to the nearest data point, the denoiser's reconstruction error and the energy, each as
a p-value against held-out points. A clamp off the relation still converges, and is flagged
there ([[coverage]]).

What is still missing: λ chosen for calibration is not always the λ one wants for balancing soft
evidence.


## 6. Cost: free-coordinate blocks and batched nodes

Every consumer of the Jacobian needs only a block of it: the solver and its stability check need
$J_{FF}$, the Laplace covariance needs $J_{FF}$, and the adjoint needs $J_{FF}$ and
$J_{FC}^\top\lambda$. So `prior_jacobian(m, z, ps, st; coords = free)` computes only $J_{FF}$, and
`prior_vjp` computes $J^\top\lambda$ (hence $J_{FC}^\top\lambda$) without forming $J$. The cost
then grows with the number of **free** coordinates, not with the dimension:

| backend | the block $J_{FF}$, per noise level | $J^\top w$ |
|---|---|---|
| none (finite differences) | $2\lvert F\rvert$ forward passes along the free coordinates | from the per-node Jacobian (slow; small $n$ only) |
| `GaussianMixtureEps` | closed form, restricted | closed form |
| DifferentiationInterface | one Jacobian of $u \mapsto (\sum_b w_b\,\varepsilon(x_b + Pu))_F$, a map $\mathbb R^{\lvert F\rvert}\to\mathbb R^{\lvert F\rvert}$: $\lvert F\rvert$ JVPs or VJPs | one input VJP |
| energy networks | the Hessian of $\sum_b w_b E(x_b + Pu)$: $\lvert F\rvert$ Hessian-vector products | one HVP |
| Reactant | $\lvert F\rvert$ compiled input VJPs | one compiled VJP |

All columns move together with $z$ ($x_b = \alpha z + \sigma\varepsilon_b$), so one derivative of the
weighted sum over a noise level's nodes covers all of them.

**Batching.** With `batched = true` the nodes of one noise level go through the network as the
columns of one matrix, for the field, the block, the VJPs and the parameter cotangent: 8 calls
instead of 64 with the default nodes. It is opt-in because the predictor must accept a matrix at
a scalar time; the closed-form mixture does not.

**Measured** (an MLP $802 \to 256 \to 256 \to 794$, 64 nodes, Zygote, CPU), a query with 10 of 794
coordinates free, as in the MNIST classification of [[MNIST in Both Directions]]:

| | time |
|---|---|
| full $794\times794$ Jacobian, as every solver iteration used to compute it | ≈ 26 s (batched; unbatched is slower) |
| the $10\times10$ block, unbatched | 0.74 s |
| the $10\times10$ block, batched | 0.04 s |
| the field, unbatched / batched | 3.1 ms / 1.0 ms |

With ForwardDiff the block costs 0.04 s either way: forward mode already pays per free coordinate.

**Many free coordinates** (the image generated, the label clamped: 784 of 794 free) make the block
itself the cost: one dense $784\times784$ block takes 33 s. With `KrylovSolve` a whole
`implicit_infer` takes 1.1 s (two Newton iterations), the adjoint 0.24 s and a Laplace marginal on
ten coordinates 0.8 s; one Jacobian-vector product costs 3.2 ms and one vector-Jacobian product
5.7 ms ([[krylov]] §4).

## 7. Many queries at once (`many.jl`)

`implicit_infer_many(m, Z₀, ρ, ps, st)` solves the columns of `Z₀` as separate queries with
one polarity. Each query keeps its own Newton or descent step, backtracking, stability test and
stopping rule, and leaves the batch when it converges. What is shared is the network: per
iteration, one call per noise level evaluates the field of all unfinished queries. The
free-coordinate blocks of all of them come from one more call per level, which stacks the
$\lvert F\rvert$ directions as copies of the inputs side by side: JVPs in general, VJPs (rows)
when the predictor has an AD backend. The blocks are dense, so this suits few free coordinates,
which is classification's shape.

The answers are those of `implicit_infer`, query by query: equal to $10^{-10}$ on the circle,
with the same iteration counts, and to $10^{-6}$ for a Lux network with every backend, batched or
not (tests). On the CPU the gain is small: 1.9 s against 2.6 s one by one, for 100 queries with finite
differences, where the network's cost dominates. On a GPU it is the difference between 24 ms and
1.8 ms per query and iteration ([[energy_network]] §8).

## 8. Classification: categorical answers

`implicit_classify(m, z₀, ρ, ps, st; label, classes)` makes the label coordinates free, starts
the solver from every class code (the columns of `classes`, e.g. one-hot vectors), and reads the
`implicit_mixture` on the label coordinates as a class with `categorical_answer`. Each root goes
to its nearest code, and the Laplace masses of the roots of a class are summed. Roots at least
half the smallest distance between two codes away from every code are reported as `unassigned`
mass, not dropped and not renormalised away. The answer keeps the mixture it was read from. With
a matrix of queries, `implicit_classify` solves all queries from all codes in one
`implicit_infer_many`, and batches the Jacobian blocks and energies of the roots as well.

**Measured** on three classes in the plane with one-hot labels (closed-form mixtures, width
0.15, `density_lambda`), against the exact posterior $p(\text{class}\mid x)$ of the data:

| $x$ | answer | exact |
|---|---|---|
| a class centre | 1.000 | 1.000 |
| on the boundary between two classes | 0.500 / 0.500 | 0.500 / 0.500 |
| 0.02 off the boundary | 0.713 / 0.287 | 0.744 / 0.256 |
| 0.04 off | 0.860 / 0.140 | 0.895 / 0.105 |
| 0.06 off | 0.939 / 0.061 | 0.961 / 0.039 |

Within 0.04 everywhere, and slightly less confident than the data. This is the smoothing: the
answer is the posterior of the density smoothed at the field's noise levels, which is wider
than the data's. A query far from every class ($x = (3, 3)$) is still answered with certainty
(class 3, the nearest). Classification inherits the extrapolation problem, and `coverage` is the
flag for it ([[coverage]]).

## 9. Conditional fields: block levels

With a predictor with noise blocks ([[predictor]] §6), `ImplicitDiffusion(pred, nodes; levels)`
says, per block, whether it is annealed with the nodes (`nothing`) or held at a fixed level.
`conditional_levels(pred, ρ)` holds every fully clamped block at `tmin`: the network sees the
clamp clean at every node, and the field on the free coordinates is the smoothed conditional
score instead of the joint score at a common level. The node input is
$\alpha_k\odot z + \sigma_k\odot\varepsilon_k$ with $\alpha$, $\sigma$ per coordinate. The weight of coordinate $i$ at a
node of level $t$ is $\lambda\,(\sigma_t^2/\alpha_t^2)\,\alpha_i/\sigma_i$: RED-Diff's $\lambda\sigma_t/\alpha_t$ on the annealed
blocks, and chosen so that the field of an energy network stays the gradient of

$$
U(z) = \sum_k w_k\Bigl[\lambda\tfrac{\sigma_k^2}{\alpha_k^2}\,E(x_k, \tau_k) - \varepsilon_k^\top(\lambda_\cdot\odot z)\Bigr],
$$

which `implicit_energy` returns. The Jacobian block on the free coordinates gains the factor
$\lambda_i\alpha_j$; the VJP scales the cotangent by $\lambda$ before the network and by $\alpha$ after.
Without `levels` every formula reduces to the scalar one, and the tests check that a blocked
predictor with every block annealed gives the unblocked relation.

**What it changes: the bias against conditioning (T2).** On the unit circle (48 narrow
components), query $y$ given $x$, 19 values of $x$ in $[-0.9, 0.9]$, the radius of the stable
answers (exact: 1):

| field levels | joint field (x smoothed too) | x held clean |
|---|---|---|
| 0.002–0.05 | 0.978 ± 0.008 (18/19 stable) | 0.993 ± 0.011 (19/19) |
| 0.05–0.2 | 0.637 ± 0.135 (18/19) | 0.775 ± 0.048 (19/19) |

Smoothing $x$ as well pulls the joint answer inwards; holding the clamp clean leaves only the
smoothing in $y$. It does not remove the bias, which now comes from the free coordinates alone.
Tested: field = $\nabla U$, Jacobian against finite differences, adjoint against finite
differences in the clamp and in a mixture mean, batched against unbatched, many queries
against single ones, a Lux energy network on JLArrays.

## 10. Single roots, tolerances, continuation

Found on MNIST ([[MNIST in Both Directions]] §6; Open Problems I13):

- **Probabilities from a single root.** When a classification query has one stable root, the
  Laplace mixture gives the class probabilities 0 and 1. `categorical_answer(mix, classes;
  temperature = τ, cov_scale = κ)` instead reads each root's Gaussian as evidence about the code:
  $p(k) = \sum_j w_j\,\mathcal N(c_k;\ \mu_j, \kappa\Sigma_j + \tau^2 I)/\sum_l \mathcal N(c_l;\ \mu_j, \kappa\Sigma_j + \tau^2 I)$. A root
  between two codes splits its mass between them. `calibrate_temperature(answers, classes,
  truth)` fits τ and κ by maximum likelihood on held-out answers (alternating golden sections
  over $\log\tau$ and $\log\kappa$). Tests: for answers drawn as $\mathcal N(c_k, 0.7^2 I)$ it recovers
  $\tau = 0.7$, and for a covariance twice too wide in standard deviation $\kappa = 1/4$. On MNIST the
  Laplace covariance is too wide (κ = 0.45), and with both parameters the probabilities have a
  calibration error of 0.009 ([[MNIST in Both Directions]] §7). `implicit_classify` passes both on.
- **Unassigned mass as a coverage flag.** `coverage(ref, z; unassigned)` returns the mass and
  sets `flagged` when $p < \alpha$ or at least half the answer lies between the codes.
- **Tolerances that follow the arithmetic.** `tol` defaults to `default_tolerance(m, z, ps)`,
  $\max(10^{-9}, 30\,\epsilon)$ for the type the field is computed in: the parameters' type on a
  device (Float32 on a GPU gives about $3.6\cdot10^{-6}$), and on the CPU the promotion of parameters
  and state (Float32 weights against a Float64 state compute in Float64, so $10^{-9}$). A Float32
  field's rounding floor is 10 to 15 $\epsilon$ relative, with 10 free coordinates and with 784 (§11);
  the first version, $10\,\epsilon$, sat on it, and solves stalled just above.
- **Continuation in the noise level.** `implicit_infer(…; continuation = (8, 3))` solves first
  with every node's level multiplied by 8 (capped at 1), then by 3, each from the previous
  answer, then the actual relation. A smoother field has fewer, wider basins. With
  Newton-or-descent it is not a general cure: on the circle a high level smooths both branches
  into one blob centred on the origin, and the continuation stops at the symmetric point between
  them (from $x = 0.6$, $y = 3$ the plain solve finds the branch $y = 0.78$ in 5 iterations; the
  continuation ends at $y = 0.06$, unstable). With the trust region (§11) it walks off that point
  onto a branch. It is for starts far from every basin with a single plausible answer,
  such as a clamped label and a blank image: on MNIST, 3–4 of 10 modes with 784 free pixels
  converge with it and none without, at five to nine times the cost
  ([[MNIST in Both Directions]] §7).

## 11. A trust-region solver

Newton-or-descent takes a Newton step where $\operatorname{sym}J_{FF}$ is positive definite and a fixed
descent step along the residual elsewhere. Far from a basin, and in non-convex regions, the
descent steps are too short and the Newton steps are not taken, so the iteration stalls.
`implicit_infer(…; method = :trust_region)` is a trust-region method on a merit function
(Nocedal & Wright 2006, ch. 4):

- **with an energy**, the query energy $Q(z) = U(z) + \tfrac12\lVert\rho_F(z_F - z_{0,F})\rVert^2$, whose gradient is
  the residual and whose Hessian is $\operatorname{sym}J_{FF}$. Every accepted step lowers $Q$, so the
  iteration cannot settle on a saddle or a maximum; negative curvature is a direction to move in,
  not a reason to fall back;
- **without one**, $\tfrac12\lVert r_F\rVert^2$ with the Gauss–Newton model (Levenberg–Marquardt), which
  can stop at a minimum of the residual that is not a root; `converged` then says so.

Each step minimises the quadratic model in a ball of radius $\Delta$: exactly with `DenseSolve`
(the eigendecomposition and a one-dimensional search for the shift, with the hard case), by
Steihaug's truncated conjugate gradients (Steihaug 1983) with `KrylovSolve` (Hessian-vector products only).
$\Delta$ starts at $0.1\sqrt{|F|}$ and is halved or doubled by the ratio of actual to predicted decrease.
Near the answer that ratio drowns in the rounding of $Q$, computed in the network's precision;
there a smaller residual decides instead.

`method = :auto`, the default, uses it for relations with an energy (energy networks, mixtures,
products of them) and Newton-or-descent otherwise. `implicit_infer_many` keeps Newton-or-descent.

**Measured.** On the circle (y given x, 20 values of $x$ and 8 starts each, 50 iterations at
most):

| | converged and stable | mean iterations |
|---|---|---|
| Newton-or-descent | 122/160 | 18.3 |
| trust region (dense or Krylov, identical) | **160/160** | **6.6** |

On MNIST, modes with the label held and 784 free pixels ([[MNIST in Both Directions]] §7, at most
50 iterations):

| start | method | converged and stable | median residual | iterations | time per mode |
|---|---|---|---|---|---|
| blank image | Newton-or-descent | 0/10 | 2.9 | 50 | 22 s |
| blank image | trust region | **10/10** | $6\cdot10^{-5}$ | 16 | **6 s** |
| a sample | Newton-or-descent | 0/10 | 16 | 50 | 19 s |
| a sample | trust region | **9/10** | $6\cdot10^{-5}$ | 37 | 6 s |

Continuation in the noise level with Newton-or-descent reached 3–4 of 10 at 120–220 s per mode;
the trust region needs neither it nor the time. Getting there needed the tolerance fix of §10:
the solves first stalled at residuals of $5\cdot10^{-5}$, just above a tolerance of $10\,\epsilon$.

Related: [[analytic]], [[reddiff]], [[factor]], [[energy_network]], [[krylov]], [[sampler]], [[Deterministic Relaxation]], [[Inference Signatures]],
[[The Implicit Diffusion Factor as a Statistical Game]]
