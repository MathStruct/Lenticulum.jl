#model

> [[ImplicitREDDiff]] ends: *"Alternatively use [ProxDM](https://arxiv.org/pdf/2507.08956) to
> achieve a prox."* This note is what that alternative is, and how the whole family of
> diffusion-based inverse solvers sorts itself.

> Sources: original to this vault (design and analysis); Fang, Díaz, Buchanan & Sulam, *Beyond Scores: Proximal Diffusion Models*, [arXiv:2507.08956](https://arxiv.org/abs/2507.08956); code: `proxdm.jl` ([[proxdm]])
>
> Bibliography: [[Bibliography#^fang2025proxdm|Fang et al. 2025]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## 1. The family, sorted by what they do with the reverse process

Every method here has the same ingredients — a trained $\varepsilon_\theta$, a measurement
$y$, a forward model $f$ — and differs in one decision: **do you sample the reverse SDE, or
do you optimise?**

| method | reverse process | prior enters as | posterior | needs $\partial\varepsilon_\theta$? |
|---|---|---|---|---|
| **DPS** (Chung et al.) | sampled | guidance term $\nabla\log p(y\mid\hat x_0)$ | samples | **yes** — through Tweedie |
| **ΠGDM** (Song et al.) | sampled | pseudo-inverse guidance | samples | **yes** |
| **RED-Diff** | *replaced by optimisation* | score-matching regulariser, stop-gradient | a point | **no** |
| **ProxDM** | discretised *backwards*: each step a prox | a learned prox operator | samples (sampler); a point (proximal inference) | no |
| **PnP-ADMM / RED** | replaced by splitting | an off-the-shelf denoiser | a point | no |

The top two are *samplers with a correction*; the bottom three are *optimisers with a
denoiser*. [[RED-Diff as a Statistical Game]] is about the third row; this note is about how
the others would land in the same framework.

> [!note] All five are the same `AbstractInversion`
> `LenticulumCore.ProximalInversion(prox)` covers rows 3–5 directly. Rows 1–2 are
> `SolverInversion` with an SDE integrator — they *are* legal Bayesian lenses, they just need
> AD through the denoiser, which is exactly what [[reddiff]] §2 says this package avoids.
> The framework does not prefer one; the free energy is supposed to measure which is better,
> and [[The Diffusion Factor]] §4.3 explains why it currently cannot.

## 2. What a prox actually is, and why RED-Diff is not quite one

The proximal operator of a function $g$ is

$$\mathrm{prox}_{\tau g}(v) \;=\; \arg\min_x\ \Bigl\{\,g(x) + \tfrac{1}{2\tau}\|x-v\|^2\,\Bigr\}$$

and the classical splitting story is: if your objective is $g(x) + h(x)$ with $h$ smooth,
alternate a gradient step on $h$ with a prox step on $g$. Plug-and-play replaces
$\mathrm{prox}_{\tau g}$ — the "denoising" half — with an off-the-shelf denoiser, on the
observation that a good denoiser *behaves like* the prox of a good image prior.

RED-Diff is a member of this family in spirit but not in form. It does **not** compute a prox;
it computes a *gradient* of a regulariser and takes an ordinary descent step. The
`ProximalInversion` wrapper is therefore slightly generous in its naming — what is shared is
"an inner optimisation using a denoiser as the prior", not "a proximal operator" in the
convex-analysis sense.

**ProxDM** closes that gap: it trains or derives an actual proximal operator for the diffusion
prior, so the inner loop becomes genuine proximal splitting rather than gradient descent on a
surrogate. The practical differences that matter here:

- **Step sizes.** Prox steps are unconditionally stable in $\tau$ where gradient steps are
  not; [[reddiff]] §5.1's "hovers in a neighbourhood, no convergence check" is a
  gradient-descent problem that a true prox largely removes.
- **The Jacobian.** A prox step implicitly uses curvature information that RED-Diff's
  stop-gradient throws away — which is where [[RED-Diff as a Statistical Game]] §4.2's
  over-smoothing bias comes from. A correct prox for the diffusion prior would not have that
  bias.
- **The cost.** A prox is itself an inner solve, so each outer step is more expensive. It buys
  stability with compute.

## 3. Why RED-Diff was implemented first anyway

Three reasons, in order of weight:

1. **No AD.** The stop-gradient makes the whole inversion forward-only, which is why
   `lib/VariationalDiffusion.jl` depends on `LuxCore`, `Random` and `LinearAlgebra` and
   nothing else. A prox needs either an inner solve, which needs gradients, or a second
   network trained to *be* the prox. (AD is now available as a user-chosen backend,
   [[backends]], so this reason has lapsed.)
2. **A closed-form check exists.** For Gaussian data the RED-Diff fixed point can be computed
   by hand and compared to the exact posterior — the oracle of
   [[RED-Diff as a Statistical Game]] §4.1. That check is what makes the implementation
   trustworthy, and it is available because the method is simple enough to analyse.
3. **It is the prompt's first suggestion.** [[ImplicitREDDiff]] names RED-Diff as the primary
   route and ProxDM as the alternative.

## 4. What is implemented

ProxDM is now in `proxdm.jl` ([[proxdm]] has the details and the numbers):

- **The interface** `proximal(p, v, t, λ, ps, st)`, with `ProxNetwork` (a Lux model in the
  paper's parametrisation $v - \sqrt\lambda\,\varepsilon_\theta(v; t, \lambda)$) and `MixtureProx`, the
  exact prox of a Gaussian mixture, the oracle a perfect `ProxNetwork` would match.
- **The sampler**, the paper's Algorithm 1 in both variants, checked against the moments of a
  Gaussian: the spread converges at first order in the step size, as backward Euler should.
- **Proximal inference for the implicit learner**, `prox_infer`: half-quadratic splitting
  between the prior's prox and the clamp's prox. Its $\lambda\to0$ limit is the implicit
  learner's problem ([[Deterministic Relaxation]]); hard clamps are exact.
- **The training loss**, proximal matching, whose $\zeta\to0$ minimiser is the MAP denoiser
  (the prox) where the squared loss gives the MMSE denoiser.

Still open: an adjoint for `prox_infer` (the implicit function theorem applies to its fixed
point as it does to `implicit_infer`), and a `ProxDM` inversion for `DiffusionFactor`, next to
`REDDiff` and `ImplicitProx`. That factor would carry **two parameter trees**, the prior's
$\varepsilon_\theta$ and the prox network. It would be the first factor in this project with the
structure the vault uses to argue that a factor cannot be a Lux layer
([[Inversions and Bayesian Lenses]], the `AmortisedInversion` case).

## 5. The honest ranking

For the problems this repository can currently test — small, Gaussian, closed-form —
RED-Diff's bias is measurable and its convergence is adequate, and ProxDM's sampler and
proximal inference agree with the exact answers to the precision stated in [[proxdm]]. For real inverse problems the
literature's ranking is roughly: DPS and ΠGDM sample better, RED-Diff optimises faster and
smoother, ProxDM is more stable than RED-Diff at more cost per step. None of that is checkable
here, and this note does not claim otherwise.

Related: [[The Diffusion Family]], [[RED-Diff as a Statistical Game]],
[[The Diffusion Factor]], [[reddiff]], [[Inversions and Bayesian Lenses]], [[Implicit Learners]],
[[ImplicitREDDiff]], [[proxdm]]
