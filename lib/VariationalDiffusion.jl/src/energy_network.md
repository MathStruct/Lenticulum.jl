#implementation

> Energy-parametrised diffusion models: the network outputs a **scalar energy** and the noise
> prediction is its gradient, ``\varepsilon_\theta = \sigma_t\nabla_x E_\theta``. The score is then
> conservative by construction, the implicit learner's field is the exact gradient of a scalar,
> and the learned relation has a real energy. The cost is second derivatives, which the AD
> extension provides.

> Sources: Salimans & Ho, *Should EBMs model the energy or the score?*, ICLR 2021 EBM workshop; Du et al., *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC*, ICML 2023; Bigi, Langer & Ceriotti, ICML 2025, [arXiv:2412.11569](https://arxiv.org/abs/2412.11569) (the same distinction for force fields); code: `energy_network.jl`, `ext/VariationalDiffusionDifferentiationInterfaceExt.jl`
>
> Bibliography: [[Bibliography#^salimans2021ebm|Salimans & Ho 2021]] · [[Bibliography#^du2023reduce|Du et al. 2023]] · [[Bibliography#^bigi2025dark|Bigi et al. 2025]]
>
> Theory: [[Geometric Deep Learning and Physical Laws]] · [[Implicit Diffusion Learners]] · [[The Implicit Diffusion Factor as a Statistical Game]] · [[backends]]

**Notation.** As in the rest of the vault: $z \in Z$ is the joint state; the network's argument
is written $x$ where it is the noised point $x = \alpha_t z + \sigma_t\varepsilon$.

## 1. Why: a direct score model is a direct force model

A noise predictor ``\varepsilon_\theta(x, t)`` is a learned vector field, and nothing makes it the
gradient of anything. Its Jacobian is not symmetric (about 5% off for a trained circle model,
[[backends]] §6), so the implicit learner's field ``g`` is not the gradient of a scalar, the
learned relation has no energy, and [[The Implicit Diffusion Factor as a Statistical Game]] §2
had to list "a scalar loss for learned networks" as missing.

This is the force-field situation of [[Geometric Deep Learning and Physical Laws]] §4: a model
that predicts forces directly versus one that predicts an energy and differentiates. The
tutorial *Symmetry is not a law* measures what the direct version costs there (circulation,
energy drift, spurious resting points). `EnergyNetwork` is the energy version for diffusion.

## 2. The parametrisation

```julia
pred = NoisePredictor(EnergyNetwork(net), VPSDE(); input, ad = AutoZygote())
```

`net` returns one value per column of `input(x, t)`. Then

$$
\varepsilon_\theta(x, t) = \sigma_t\,\nabla_x E_\theta(x, t),\qquad
s_\theta(x, t) = -\nabla_x E_\theta(x, t),\qquad
\partial_x\varepsilon_\theta = \sigma_t\,\nabla_x^2 E_\theta \ \text{(symmetric)}.
$$

`EnergyNetwork` is a marker wrapped *inside* `NoisePredictor`, the same pattern as
`NoisePredictor{<:GaussianMixtureEps}`: `epsilon`, `epsilon_jacobian` and `epsilon_vjp_params`
specialise on it, so `ImplicitDiffusion`, `DiffusionFactor`, RED-Diff and everything else
accept an energy model unchanged.

## 3. What the implicit learner gains: an energy

With ``x_k = \alpha_k z + \sigma_k\varepsilon_k`` and ``\nabla_z E(x_k) = \alpha_k\nabla_x E(x_k)``,

$$
g(z) = \sum_k w_k\lambda_k\bigl(\varepsilon_\theta(x_k, t_k) - \varepsilon_k\bigr) = \nabla_z U(z),\qquad
U(z) = \sum_k w_k\lambda_k\Bigl[\tfrac{\sigma_k}{\alpha_k}E_\theta(x_k, t_k) - \varepsilon_k^\top z\Bigr].
$$

`implicit_energy(m, z, ps, st)` returns ``U``. Consequences:

- **A query has a loss**, ``U(z) + \tfrac12\lVert P(z - z_0)\rVert^2``. Two answers to the same
  query (the two branches of the circle) can be compared by it; a solver can do a line search
  on it.
- **`stable` means "local minimum"** exactly: the field Jacobian is a Hessian, so its positive
  definiteness is the second-order condition, not a heuristic.
- **The adjoint uses ``J^\top = J``**, so the backward pass solves the same linear system as
  the forward Newton step.
- **The statistical-game reading is complete** for learned networks: the prior game's energy
  is ``U``, not only its gradient ([[The Implicit Diffusion Factor as a Statistical Game]] §2).

## 4. Derivatives: one Hessian-vector product does the hard part

| quantity | needed for | computed as |
|---|---|---|
| ``\nabla_x E`` | ``\varepsilon`` itself | `DI.gradient` with the first-order backend |
| ``\nabla_x^2 E`` | Newton steps, adjoint solve | `DI.hessian` |
| ``\partial_\theta\langle\nabla_x E, w\rangle`` | backward pass, training | θ-block of `DI.hvp` on ``u = (x, \theta)`` in direction ``(w, 0)`` |

The mixed derivative is the one that looks hard: a derivative with respect to the parameters
of a derivative with respect to the input. Concatenating input and parameters into one vector
turns it into an ordinary Hessian-vector product, which DifferentiationInterface computes
forward-over-reverse.

**Training needs no nested AD in the loop.** The denoising loss
``\lVert\sigma_t\nabla_x E - \varepsilon\rVert^2`` contains a derivative, so differentiating it
naively is AD over AD. But its gradient is ``\partial_\theta\langle\nabla_x E, 2\sigma_t r\rangle`` with
``r`` the residual, the same mixed derivative. `denoising_gradient(pred, x₀, t, ε, ps, st)`
returns the loss and that gradient for a batch; apply it with any optimiser.

Backends tested: `AutoZygote()` and `SecondOrder(AutoForwardDiff(), AutoZygote())` agree with
the closed form to 1e-16. Enzyme's forward-over-reverse fails on Lux layers on CPU
(`jl_eqtable_get` not implemented in forward mode); Reactant is not wired for energy models.

## 5. Validation

- **Exact.** A Lux layer returning ``-\log p_t`` of a Gaussian mixture, wrapped as an
  `EnergyNetwork`, reproduces `GaussianMixtureEps` (whose ``\varepsilon`` is ``-\sigma_t\nabla\log p_t``
  in closed form): ``\varepsilon``, its Jacobian and the parameter VJP agree to 1e-16 with both
  backends.
- **Conservative.** For an untrained Lux energy network the Jacobian is symmetric to 1e-17;
  ``\nabla U`` equals `prior_field` to 2e-13.
- **Backward pass.** `implicit_pullback` through inference agrees with finite differences
  (relative 1e-8 for a weight), and `denoising_gradient` on a batch with a different ``t`` per
  column agrees with finite differences of the loss.
- **Training** (§6).

## 6. Training on the circle, against an ε-network

The documentation's tutorial *A conservative score* runs this comparison live. Same data (noisy points on the unit circle, ``t\in[0.001, 0.2]``), same width (two hidden layers
of 64), 6000 Adam steps at batch 256. The ε-network trains with Lux's `Training` API; the
energy network with `denoising_gradient` and Optimisers.

| | ε-network | energy network |
|---|---|---|
| denoising loss per coordinate, last 500 steps | ≈ 0.54 | 0.547 |
| stable answers to "y given x", 38 queries | 37 | 38 |
| radius of the stable answers | 0.983 ± 0.009 | 0.978 ± 0.008 |
| asymmetry of the field Jacobian on the circle | 2.6% | 5·10⁻¹⁷ |
| training time on a laptop CPU | 15 s | 41 s (`SecondOrder(ForwardDiff, Zygote)`), 52 s (`AutoZygote`) |

The two fit equally well; the energy network costs about three times as much per step, the
price of a second derivative. In return the field is a gradient to machine precision, and the
relation has an energy that means something: along ``x = 0.6``,

| ``y`` | ``+0.78`` (upper branch) | ``0`` | ``-0.78`` (lower branch) |
|---|---|---|---|
| ``U(0.6, y)`` | −0.1850 | −0.1468 | −0.1839 |

Both branches sit at almost the same energy (the data are symmetric), with a barrier between
them. A point method returns one branch; the energy says the other is equally good.

## 7. Implementation difficulties

- **The mixture code is not AD-friendly.** `_mixture_parts` normalises the responsibilities in
  place (`γ ./= sum(γ)`), which Zygote cannot differentiate, so the exact test uses its own
  non-mutating mixture energy.
- **Float32 parameters pollute second derivatives.** Lux initialises in Float32; a
  finite-difference check at step ``10^{-5}`` against Float32 parameters disagrees at 1e-3. All
  energy computations here convert parameters to Float64 first.
- **The input must accept a batch.** `input(x, t)` is called with ``x`` a matrix and ``t`` a
  scalar or a ``1\times B`` row during training, and with a vector during inference.
- **Reactant** would need its own compiled Hessian and mixed derivative; not done.

Related: [[predictor]], [[implicit]], [[backends]], [[analytic]],
[[Geometric Deep Learning and Physical Laws]], [[The Implicit Diffusion Factor as a Statistical Game]]
