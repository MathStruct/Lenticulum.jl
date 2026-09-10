# VariationalDiffusion.jl — package note

> A diffusion model as a **statistical game**. The third of the three [[Implicit Learners]]
> families, and the only one whose Bayesian inversion is neither exact nor a root-find.

## The chain, in four files

| file | note | supplies |
|---|---|---|
| `schedule.jl` | [[schedule]] | the VP-SDE: $\alpha_t,\sigma_t$ with $\alpha_t^2+\sigma_t^2=1$ |
| `predictor.jl` | [[predictor]] | $\varepsilon_\theta$ wrapping a Lux model; score and Tweedie |
| `reddiff.jl` | [[reddiff]] | the proximal operator; Proposition 2; λ calibration |
| `factor.jl` | [[factor]] | the `LenticulumFactor`; $P$ from the polarity |

Concept notes are in `markdown/Diffusion/`, entry point [[The Diffusion Family]].

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

**No automatic differentiation anywhere.** RED-Diff's stop-gradient means the denoiser
Jacobian is never formed, and the clamp's gradient is $P^2(x-x_0)$ in closed form. So the
dependency list is `LuxCore`, `Random`, `LinearAlgebra` — and **not `Lux`**, since any Lux
model is an `AbstractLuxLayer`. The consequence is that this package can *use* a trained
$\varepsilon_\theta$ and cannot *train* one.

**The inversion returns a `DiracBelief`.** That is the paper's own variational family
($\sigma\to0$), not a shortcut, and it is the source of most of what is awkward about putting
this factor in a graph — see [[factor]] §5.

## How it is tested

There is exactly one closed-form diffusion model, and the test suite is built on it: for
Gaussian data $x_0\sim\mathcal{N}(0,\Sigma)$,

$$p_t = \mathcal{N}(0,\ \alpha_t^2\Sigma+\sigma_t^2 I),
\qquad
\varepsilon_\theta(x,t) = \sigma_t(\alpha_t^2\Sigma+\sigma_t^2I)^{-1}x$$

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
[[Statistical Game]], [[Channels and Polarity]]
