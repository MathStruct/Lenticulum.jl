#implementation

> Kernel models as relations. A Gaussian kernel density estimate is a Gaussian mixture with one
> centre per sample, so the closed-form mixture of [[analytic]] already is one, with exact
> derivatives: `kde_predictor` builds it from samples, `kde_bandwidth` picks its bandwidth, and
> `OnlineKDE` learns from a stream, merging nearby samples and forgetting old ones.

> Sources: code: `kernel.jl`, `analytic.jl` (weights); Kristan, Leonardis & Skočaj, *Multivariate online kernel density estimation with Gaussian kernels*, Pattern Recognition 2011 (online KDE with compression, the idea behind `OnlineKDE`; not checked against its algorithm in detail)
>
> Theory: [[Kernel Methods for Implicit Learning]] · [[Implicit Diffusion Learners]]

## 1. The batch KDE

`kde_predictor(schedule, data; bandwidth)` is `GaussianMixtureEps` with the samples as centres.
Every query function (`implicit_infer`, `implicit_roots`, `implicit_laplace`) works on it
unchanged; the relation is the ridge of the KDE smoothed at the field's noise levels.

`kde_bandwidth(data)` maximises the held-out log-likelihood over a log grid (0.5 % to 100 % of
the data's scale). It picks the best *density*, using no query information. On the robot arm
this is not the bandwidth with the best answers, but no bandwidth rescues a small dataset
(the documentation's kernel-baseline tutorial: 500 samples, median miss 0.080 against 0.010 for
a network trained on the same data; tuned on the query errors themselves, still about 0.07).

## 2. Weights

`GaussianMixtureEps(…; weights)` stores normalised log-weights and adds them to the
log-responsibilities. Every closed form (the noise prediction, its Jacobian, the parameter
derivative, the log-density, `MixtureProx`) is written in terms of the responsibilities, so all
of them become weighted without further change. Equal weights remain the default, so nothing
existing changed.

## 3. The online, forgetting KDE

`OnlineKDE(dim; bandwidth, forget, merge_radius, budget)` and `observe!(kde, samples)`; per
sample:

| step | what | why |
|---|---|---|
| forget | all weights × `forget` | a sample seen ``k`` steps ago weighs ``\text{forget}^k``; memory ≈ ``1/(1-\text{forget})`` samples |
| merge or insert | into the nearest centre within `merge_radius` (weighted mean), else a new centre | bounds the number of centres for dense streams |
| prune | drop centres below `prune` × the largest weight | forgotten mass |
| budget | while over `budget`, merge the **lightest** centre into its nearest neighbour | mass and mean kept |

`kde_predictor(schedule, kde)` returns the current state as a weighted mixture, so queries see
the model as it is now.

**Measured on a drifting relation** (a circle whose radius grows from 1.0 to 1.5 over 3000
samples; query "y given x = 0"):

| after sample | true radius | remember all (`forget = 1`) | forget, memory ≈ 200 |
|---|---|---|---|
| 750 | 1.125 | 1.053 | 1.085 |
| 1500 | 1.250 | 1.117 | 1.208 |
| 2250 | 1.375 | 1.162 | 1.329 |
| 3000 | 1.500 | 1.213 | 1.445 |

The forgetting model tracks the current relation with a lag of about its memory; the
remembering one answers with the average of the whole history (the mean radius seen so far,
minus the smoothing bias). Neither is wrong: which one is right depends on whether the drift is
a change of the system or noise to be averaged. Queries stay at about 0.07 s because the budget
caps the centres at 400.

## 4. Could this become an extension?

The kernel models themselves need nothing beyond the package: they are the closed-form mixture.
Package extensions are the right home for connections to the Julia kernel ecosystem, following
the pattern of the AD backends ([[backends]]):

- **KernelFunctions.jl**: kernels other than the isotropic Gaussian (Matérn, anisotropic,
  periodic). The closed-form derivatives are Gaussian-specific, so other kernels would take their
  Jacobians from an AD backend.
- **AbstractGPs.jl**: a Gaussian-process implicit factor, whose answer is a belief over the
  relation itself ([[Kernel Methods for Implicit Learning]] §4).
- **KernelDensity.jl**: fast FFT-based estimates in one or two dimensions.

None is built.

## 5. Implementation difficulties

- **The first budget rule dropped the lightest centres**, which under `forget = 1` are the
  *newest* samples (older centres had accumulated weight by merging). The model silently stopped
  learning: its answer froze at the first radius it saw. Merging the lightest centre into its
  neighbour fixed it; a test now checks that compression conserves the total weight.
- **The bandwidth is fixed** after construction. Re-selecting it online (e.g. from the recent
  window) is the obvious extension; the merge radius would have to follow.
- **Merging is greedy and isotropic.** It moves a centre by at most the merge radius and never
  widens it, so a merged centre represents its samples by their mean only. Moment-preserving
  merges (a wider component for the merged pair) would be more faithful and need per-component
  widths.

Related: [[analytic]], [[implicit]], [[Kernel Methods for Implicit Learning]], [[backends]]
