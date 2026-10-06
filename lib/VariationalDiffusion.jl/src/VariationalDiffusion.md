#implementation

> A diffusion model as a **statistical game**. The third of the three [[Implicit Learners]]
> families, and the only one whose Bayesian inversion is neither exact nor a root-find.

> Sources: code: `VariationalDiffusion.jl`, `factor.jl`, `lens.jl`, `predictor.jl`, `reddiff.jl`, `schedule.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## The chain, in ten files

| file | note | supplies |
|---|---|---|
| `schedule.jl` | [[schedule]] | the VP-SDE: $\alpha_t,\sigma_t$ with $\alpha_t^2+\sigma_t^2=1$ |
| `predictor.jl` | [[predictor]] | $\varepsilon_\theta$ wrapping a Lux model; score and Tweedie |
| `reddiff.jl` | [[reddiff]] | the proximal operator; Proposition 2; λ calibration |
| `factor.jl` | [[factor]] | the `LenticulumFactor`; $P$ from the polarity |
| `analytic.jl` | [[analytic]] | closed-form ε* of a Gaussian mixture: an exact-score relation (oracle) |
| `kernel.jl` | [[kernel]] | kernel density estimates as relations: batch KDE with bandwidth selection, an online forgetting KDE |
| `implicit.jl` | [[implicit]] | deterministic implicit inference and its adjoint backward pass |
| `energy.jl` | [[energy]] | energy-parametrised predictors: $\varepsilon = \sigma_t\nabla_x E_\theta$, a conservative score, an energy for the relation |
| `implicit_factor.jl` | [[implicit_factor]] | `DiffusionFactor` with `ImplicitProx`: report and per-channel pullback |
| `proxdm.jl` | [[proxdm]] | proximal diffusion models: prox interface, exact oracle, sampler, proximal inference |
| `ext/` | [[backends]] | the network's derivatives through any AD backend (DifferentiationInterface, Reactant) |

Concept notes are in `vault/Families/Diffusion/`, entry point [[The Diffusion Family]].

## What `LenticulumCore` already had

Nothing in the core needed changing, which is the strongest evidence so far that its
abstractions were the right ones:

- `ProximalInversion(prox)` exists in `lens.jl` and its docstring already names *this
  package* and *RED-Diff*. It was written before there was an implementation.
- `Polarity` with per-channel precisions is exactly $P = \rho_{in}P_{in} +
  \rho_{out}P_{out} + \rho_{latent}P_{latent}$, including `default_precision(Observed()) = Inf`.
- `GradedEnergySpace` expresses the `(clamp, score)` split without a new type.
- `DiracBelief` is what RED-Diff's variational family actually produces.

One thing it did *not* have, and the gap is now confirmed from two directions: the
`energy(factor, x, a, y, ps, st)` signature presumes a causal $X/Y$ split. `constraint.md` §3
hit it from the acausal side; `factor.md` §4 hits it here. Two independent factor families
means it is an interface bug rather than a quirk.

## Two properties worth stating up front

**No automatic-differentiation dependency.** RED-Diff's stop-gradient means the denoiser
Jacobian is never formed, and the clamp's gradient is $P^2(z-z_0)$ in closed form. So the
dependency list is `LuxCore`, `Random`, `LinearAlgebra` — and **not `Lux`**, since any Lux
model is an `AbstractLuxLayer`. Where a learned network *is* differentiated (the implicit
learner's Newton steps and its backward pass), the user chooses the backend with
`NoisePredictor(…; ad = AutoZygote())` or any other ADTypes object, and a package extension
supplies it ([[backends]]). Training the denoiser itself is ordinary Lux training.

**The inversion returns a `DiracBelief`.** That is the paper's own variational family
($\sigma\to0$), not a shortcut, and it is the source of most of what is awkward about putting
this factor in a graph — see [[factor]] §5.

## How it is tested

There is exactly one closed-form diffusion model, and the test suite is built on it: for
Gaussian data $x_0\sim\mathcal{N}(0,\Sigma)$,

$$
p_t = \mathcal{N}(0,\ \alpha_t^2\Sigma+\sigma_t^2 I),
\qquad
\varepsilon_\theta(x,t) = \sigma_t(\alpha_t^2\Sigma+\sigma_t^2I)^{-1}x
$$

is a *perfectly trained* noise predictor. Against it:

- Tweedie's `denoise` reproduces the exact linear-Gaussian posterior mean to floating point;
- the RED-Diff regulariser collapses to a linear shrinkage $\kappa x$ with $\kappa$ known in
  closed form;
- with the calibrated $\lambda$, the RED-Diff fixed point **is** the exact Gaussian posterior
  mean, to within Monte-Carlo noise that averages away over seeds;
- and no single $\lambda$ can do the same for a correlated prior — [[reddiff]] §4.2.

This is the same move `GaussianFactor` makes in the parent package: build the one case where
everything is computable, and check the framework against arithmetic instead of against
itself.

Related: [[Implicit Learners]], [[The Diffusion Family]], [[ImplicitREDDiff]],
[[Factors are Parameterized Statistical Games]], [[Channels and Polarity]]
