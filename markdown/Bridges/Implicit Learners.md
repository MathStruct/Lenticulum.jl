# Implicit Learners — the three families

> Relating [[README]]'s explicit/implicit table to the AutoBayes machinery, and to the
> three model families you can actually build.

## The claim

An explicit learner approximates a **function** $f_\theta : X \to Y$. An implicit learner
approximates a **relation** $R_\theta \subseteq X_1 \times \cdots \times X_n$, represented by
a residual

$$r_\theta : X_1 \times \cdots \times X_n \longrightarrow E,
\qquad (x_1,\ldots,x_n) \in R_\theta :\Longleftrightarrow r_\theta(x_1,\ldots,x_n) \approx 0$$

## Where this sits in AutoBayes

$r_\theta$ **is a multivariate energy** in the sense of
[[Scalar and Multivariate Energy]]: $\mathbf{l}^c = r_\theta$, with $E$ the residual space
and $\sigma = \tfrac12\|\cdot\|^2$ the scalarisation. Membership of the relation is
"$\mathbf{l}^c \approx 0$", i.e. zero energy.

More precisely, an implicit factor is a statistical game in which:

- the **forward kernel** $c$ is not given directly — only implicitly, as the solution set
  of $r_\theta = 0$ after a [[Channels and Polarity|polarity]] is chosen;
- the **inversion** $c'$ is the solver;
- the **energy** is $r_\theta$;
- the **entropy** is whatever regularises the solver's output (a proximal term, a diffusion
  prior, an entropy over the solution set when it is not a singleton).

The energy-based reading also explains the well-posedness row of the README's table: a
relation may be multi-valued or empty, and then "inference" returns the *closest point to
the variety* rather than a point on it. That is exactly $\arg\min_x \sigma(r_\theta(x))$ —
minimising the energy rather than zeroing it. **Energy minimisation is the total version of
root-finding**, and it is total precisely because $\sigma \ge 0$ always has an infimum.

## The three families

| | approximator | inference | backward pass | regime |
|---|---|---|---|---|
| **algebraic** | algebraic varieties; differential algebra | Gröbner / homotopy continuation / Newton | implicit function theorem; differential elimination | low dimension, exact structure |
| **equilibrium** | DEQ, NeuralODE, fixed points | fixed-point iteration, ODE solve | implicit function theorem at the fixed point; adjoint | medium; **needs convergence guarantees** |
| **diffusion** | score / denoiser networks in a prox operator | annealed proximal steps (RED-Diff, ProxDM) | variational, via the score | high dimension |

All three are *the same statistical game* with a different realisation of $c'$. That is what
makes them interchangeable behind the factor interface, and it is the reason the interface
is worth having.

### Algebraic

> Worked out in full in [[Algebraic Implicit Learners]] and the twelve notes it indexes:
> the Veronese parametrisation, the closed-form fit, the Grassmannian parameter, root-finding
> inference, the branch/discriminant structure, and an honest gap list.

$r_\theta$ a vector of polynomials; $R_\theta$ its variety. Universal approximation via
Nash–Tognoli (compact smooth manifolds are approximable by real algebraic varieties), the
implicit counterpart of Weierstraß. Backward pass: [[Backpropagation by the Implicit Function Theorem|the implicit function
theorem]] where the Jacobian block is invertible; the failure locus is
[[Branches and the Discriminant|the discriminant]]. Differential algebra enters elsewhere
than the backward pass — see [[Differential Algebra and DAE Factors]]. **Needs the vector
residual and its Jacobian** — see [[Scalar and Multivariate Energy]] §6.3.

### Equilibrium

$r_\theta(x, z) = z - g_\theta(z, x)$; inference solves for the fixed point $z^*$.
Differentiating: $\frac{\partial z^*}{\partial \theta} = (I - \partial_z g)^{-1}\partial_\theta g$ —
one linear solve, no unrolling. The caveat in [[Prompt1|the prompt]] is the right one:
*this only works if the iteration converges*, and unconstrained DEQs need not. Remedies are
architectural (contractivity via spectral normalisation, monotone operator parametrisation)
or a damped/regularised solve.

In game terms: an equilibrium factor's inversion is a solver whose *entropy* $\mathbf{H}$
should charge for non-convergence. A solver that stopped early is an inexact inversion, and
[[Bayesian Lens]] says inexact inversions are legal — the loss just gets worse. That is a
much more graceful failure mode than a divergent unroll.

### Diffusion — see [[ImplicitREDDiff]]

Energy
$$E(x_0, x) = \mathbb{E}_{t,\epsilon}\bigl[\omega(t)\|\epsilon_\theta(\alpha_t x + \sigma_t\epsilon, t) - \epsilon\|_2^2\bigr] + \tfrac12\|P(x_0 - x)\|^2$$

with $P = \rho_{in}P_{in} + \rho_{out}P_{out} + \rho_{latent}P_{latent}$.

Read as a statistical game:

- the first term is the **entropy/regulariser** $\mathbf{H}$ — it depends on the *learned
  distribution*, not on the data point, and it is the score-matching prior;
- the second term is the **energy** $\mathbf{l}$ — pointwise, quadratic, and it is where
  the [[Channels and Polarity|polarity]] enters, as a precision-weighted clamp;
- $\rho_{in} \to \infty$ recovers a hard [[Copiers Cups and Caps|cup]] on the observed
  channels.

This is `lib/VariationalDiffusion.jl`, the `LenticulumFactor` whose prox operator is a
diffusion model. RED-Diff supplies the gradient; ProxDM is the alternative.

## The cost

| | explicit | implicit |
|---|---|---|
| inference | forward evaluation, one pass | root-finding: Newton, fixed point, or annealing |
| wiring | DAG | arbitrary weakly-connected digraph |
| direction | fixed | chosen per call ([[Channels and Polarity]]) |

The middle row is why `Mycelium.jl` exists: with a general digraph there is no topological
order, so "run the network" is replaced by "schedule messages until convergence". The
bottom row is why a factor cannot be a Lux layer.

Related: [[Algebraic Implicit Learners]], [[Channels and Polarity]], [[Scalar and Multivariate Energy]], [[ImplicitREDDiff]], [[Statistical Game]]
