# The Diffusion Family

> Entry point for the third of the three [[Implicit Learners]] families, worked out the way
> [[Algebraic Implicit Learners]] works out the first.
>
> The claim being tested: **a diffusion model is a statistical game whose inversion is a
> proximal operator.** Not an analogy — `lib/VariationalDiffusion.jl` implements it, and the
> pieces `LenticulumCore` needed for it were already there.

## The one-paragraph version

Train $\varepsilon_\theta(x,t)$ to denoise samples corrupted by a known Gaussian process
([[The VP-SDE]]). Tweedie's formula turns it into an MMSE denoiser, which makes it a *prior*
rather than merely a sampler. Then, to condition on data, do not sample the reverse SDE —
**optimise**: minimise data misfit plus a score-matching regulariser whose gradient is one
forward pass of $\varepsilon_\theta$ ([[RED-Diff as a Statistical Game]]). Wrap the result as
a factor whose channels are blocks of one state space and whose polarity supplies the
selection matrices ([[The Diffusion Factor]]).

## The notes

1. [[The VP-SDE]] — the forward process, its perturbation kernel, and why $\alpha_t^2+\sigma_t^2=1$
   is the only property that matters downstream. Song et al. 2021.
2. [[RED-Diff as a Statistical Game]] — the variational reformulation; the stop-gradient; the
   energy/entropy split; and **λ is derivable, not merely tunable**, with the calculation.
3. [[The Diffusion Factor]] — $P_{in}+P_{out}+P_{latent}=\mathrm{Id}$ becomes a `Polarity`;
   what breaks when a Dirac-valued message meets a factor graph.
4. [[ProxDM and Proximal Alternatives]] — the other way to build the prox, and how the family
   relates to DPS, ΠGDM and plain plug-and-play.

Implementation notes sit next to the code: [[VariationalDiffusion]], [[schedule]],
[[predictor]], [[reddiff]], [[factor]].

## Where it sits among the three families

[[Implicit Learners]]'s table, with the diffusion column filled in from experience rather
than expectation:

| | algebraic | equilibrium | **diffusion** |
|---|---|---|---|
| approximator | varieties | fixed points | **score / denoiser network** |
| inference | Gröbner, homotopy | fixed-point iteration | **proximal descent** |
| backward pass | implicit function theorem | IFT at the fixed point | **stop-gradient; no Jacobian at all** |
| inversion type | `SolverInversion` | `SolverInversion` | `ProximalInversion` |
| posterior | a point (or a branch) | a point | **a point** — $q$ is a Dirac |
| exactness | exact where the Jacobian is invertible | exact at convergence | **biased, by a computable amount** |

The last row is the interesting one and is the subject of
[[RED-Diff as a Statistical Game]] §4. The other two families are *approximate because they
stop early*; this one is approximate **at its fixed point**, because the method deliberately
discards the denoiser Jacobian. A solver that stopped early is an inexact inversion and
[[Bayesian Lens]] says that is legal — the loss just gets worse. A method whose fixed point is
the wrong point is a different situation, and the vault had not previously distinguished them.

## What this family cost the framework

Nothing structural, which is the headline. `ProximalInversion` was already in `lens.jl`,
named after this package, before this package existed. Per-channel precisions in `Polarity`
were already exactly $P$. `GradedEnergySpace` already expressed the split.

Two things it *did* expose:

- **`energy`'s signature presumes a causal factor.** The core's
  `energy(factor, x, a, y, ps, st)` wants AutoBayes' $X\times\llbracket c\rrbracket\times Y$
  split, which a diffusion factor does not have (it has one state space and a polarity).
  [[ModelingToolkit as an Acausal Relation]]'s `LinearConstraintFactor` hit the same wall
  from the acausal side. Two families, same complaint.
- **The Bethe machinery assumes every factor's $-H$ is an entropy.** For this factor it is a
  score-matching term standing in for one, because the actual posterior is a Dirac whose
  differential entropy is $-\infty$. Mixing a `DiffusionFactor` and a `GaussianFactor` in one
  graph gives a total free energy that is not $-\log p(y)$ for anything. See
  [[The Diffusion Factor]] §5.

## Sources

- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, *Score-Based Generative Modeling through
  Stochastic Differential Equations*, ICLR 2021,
  [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).
- Mardani, Song, Kautz, Vahdat, *A Variational Perspective on Solving Inverse Problems with
  Diffusion Models*, [arXiv:2305.04391](https://arxiv.org/abs/2305.04391).
- Efron, *Tweedie's Formula and Selection Bias*, JASA 2011 — the denoiser identity.
- Romano, Elad, Milanfar, *The Little Engine that Could: Regularization by Denoising (RED)*,
  2017 — the "RED" that RED-Diff is named after.

Related: [[Implicit Learners]], [[ImplicitREDDiff]], [[Statistical Game]],
[[Algebraic Implicit Learners]], [[Channels and Polarity]]
