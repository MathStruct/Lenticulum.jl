#open-problem

> **Status: resolved (2026-10-26).** Related: `src/gaussian.jl` (`GaussianFactor`, `GaussianPrior`), [[Scalar and Multivariate Energy]] §8 ("Normalised or not").

**Problem.** In `GaussianFactor` the noise covariance $Q$ is a fixed hyperparameter, and `GaussianPrior`'s $\Sigma$ is not learnable either: learning them needs a positive-definite parametrisation, which the code defers. It matters beyond convenience. The pointwise energy $\tfrac12 r^\top Q^{-1} r$ leaves out the normaliser $\tfrac12 \log\det(2\pi Q)$, a constant only while $Q$ is fixed; once $Q$ is learned, only the free energy, which keeps it as the complexity summand, is a valid objective.

**Options.**
1. A Cholesky factor with a log-diagonal, $Q = L L^\top$, as parameters.
2. A precision parametrisation, $Q^{-1} = L L^\top$, which is what the scalarisation uses.
3. Diagonal $Q$ only, through log-variances.

**Leaning.** Not chosen. Whichever it is, training must use the free energy, and a test should show what goes wrong without it: the pointwise energy $\tfrac12 r^\top Q^{-1} r$ alone *decreases* as $Q$ grows, so it drives $Q \to \infty$ (variance inflation); the normaliser $\tfrac12 \log\det(2\pi Q)$ in the free energy penalises that. At an exact fit ($r = 0$) the likelihood is unbounded as $Q \to 0$, so even the free energy needs a prior on $Q$ or a floor. (Direction corrected 2026-10-25; it said $Q \to 0$.)

## Decision and implementation — 2026-10-26

The maintainer chose **precision Cholesky coordinates**, $Q^{-1}=LL^\top$, with a
log diagonal; diagonal mode stores log-precisions. `GaussianFactor` and `GaussianPrior`
opt in with `learn_noise=true` / `:full` or `:diagonal`, initialised from their supplied
covariance. Fixed noise remains the default; the prior mean stays fixed.

`GaussianFactor.energy` stays the residual, and `scalar_energy` is still quadratic fit
without a normaliser. The full Gaussian normaliser belongs in `free_energy` and
`local_free_energy`, as in [[L-ADR-0003 Pointwise energy and free energy are separate levels]].
`GaussianPrior` keeps its existing normalised pointwise energy. Optional noise-prior
penalties belong only in free energy. The earlier options and leaning above record
the original open question, not the current API.

A user-supplied `WishartPrecisionPrior(df, scale)` (full precision) or
`LogNormalVariancePrior(logmean, logstd)` (diagonal variances) adds its normalised
negative log density to `complexity`. MAP is in the prior's declared density
coordinates; no Jacobian for trainable Cholesky/log coordinates is added. Without a
prior, a configurable covariance eigenvalue floor (default `1e-6`) applies. The hard
floor has zero derivative in saturated directions and can trap them. The prior
replaces the floor. See [[noise]] and [[gaussian]] for formulas and limitations.

The LP-002 parameter-AD APIs and Mycelium's `parameter_step` consume the new nested
`ps.noise` tree. Optional ForwardDiff and ChainRulesCore extensions differentiate the
spectral floor, including repeated eigenspaces on a smooth branch. A noise prior is
charged once per local energy: shared-parameter dataset objectives must count it once
per dataset, rather than accidentally once per sample.

## Verification results — 2026-10-26

`test/learnable_noise.jl` contributes **115 passing checks**, with ForwardDiff and
Zygote compared to central finite differences and densities compared to
Distributions.jl. It checks:

- **MLE:** eight balanced linear-Gaussian observations with known $A,b$ have residual
  second moment $S=\begin{pmatrix}0.8&0.3\\0.3&1.4\end{pmatrix}$. Averaging
  `parameter_step` gradient-descent candidates from the same parameters is a batch
  step on the mean sample free energy. After 220 steps, full learning recovers $S$
  and diagonal learning recovers $\operatorname{diag}(S)$, with `atol=2e-6` under
  both backends. The affine parameters remain at their known values.
- **Variance inflation:** pointwise fit decreases along increasingly inflated noise
  covariances; noise-only gradient descent grows scalar variance beyond 100 after
  1000 steps. The normalised free energy grows along the inflation sequence instead.
- **Exact fit:** full and diagonal noise reach the configured covariance floor `0.05`.
  With one zero-residual observation and a Wishart prior, the precision converges to
  $(\nu-d)V$ (`atol=2e-6`). The diagonal log-normal case converges to
  $v_i=\exp(\mu_i-1.5s_i^2)$ (`atol=1e-9`).
- **AD and inference:** new terms agree with finite differences, including rotated
  eigenspaces with partial spectral clipping and repeated eigenvalues away from the
  kink. Learned covariance enters forward/backward messages, all four Dirac/Gaussian
  endpoint combinations, and `GaussianPrior` graph parameter updates.

The package runs used `julia --project=<package> -e 'using Pkg; Pkg.test()'` (root
project `.` for Lenticulum). Wall times include package setup and compilation; jobs
ran concurrently, so these are verification timings, not performance benchmarks.

| suite | passing assertions | wall time |
|---|---:|---:|
| Lenticulum (including LP-005) | 894 | 232.41 s |
| ImplicitLayers (test dependant of Lenticulum) | 1195 | 505.06 s |
| LenticulumCore | 316 | 53.75 s |
| Mycelium | 494 | 12.52 s |

`julia --project=docs docs/make.jl` passed in 86.85 s, producing 26 API HTML pages
and building the vault. The standalone `docs/site/build.sh` passed in 11.18 s with
248 HTML pages and no KaTeX errors; after recording these results it was rebuilt
successfully in 7.78 s. Both used `QUARTZ_DIR=/tmp/lp005-quartz`, a
disposable copy with its checkout/clean restore hook removed, and a temporary npm
cache. No checkout/clean/reset/stash/commit command was run. Generated bibliography
and release-metadata files retained their pre-existing contents.

**Remaining scope:** device execution, higher-order AD and other backends are not
established. A hard spectral floor has a kink and can trap saturated directions;
the prior alternatives avoid that floor. This resolves LP-005 for the tested CPU
first-derivative API, not posterior inference over noise parameters or exposed
parameter channels (see [[L-SPEC-003 Exposed Parameters and Energy Channels]]).
