#open-problem

> **Status: open, later (2026-10-22).** Related: `src/gaussian.jl` (`GaussianFactor`, `GaussianPrior`), [[Scalar and Multivariate Energy]] §8 ("Normalised or not").

**Problem.** In `GaussianFactor` the noise covariance $Q$ is a fixed hyperparameter, and `GaussianPrior`'s $\Sigma$ is not learnable either: learning them needs a positive-definite parametrisation, which the code defers. It matters beyond convenience. The pointwise energy $\tfrac12 r^\top Q^{-1} r$ leaves out the normaliser $\tfrac12 \log\det(2\pi Q)$, a constant only while $Q$ is fixed; once $Q$ is learned, only the free energy, which keeps it as the complexity summand, is a valid objective.

**Options.**
1. A Cholesky factor with a log-diagonal, $Q = L L^\top$, as parameters.
2. A precision parametrisation, $Q^{-1} = L L^\top$, which is what the scalarisation uses.
3. Diagonal $Q$ only, through log-variances.

**Leaning.** Not chosen. Whichever it is, training must use the free energy, and a test should show what goes wrong without it: the pointwise energy $\tfrac12 r^\top Q^{-1} r$ alone *decreases* as $Q$ grows, so it drives $Q \to \infty$ (variance inflation); the normaliser $\tfrac12 \log\det(2\pi Q)$ in the free energy penalises that. At an exact fit ($r = 0$) the likelihood is unbounded as $Q \to 0$, so even the free energy needs a prior on $Q$ or a floor. (Direction corrected 2026-10-25; it said $Q \to 0$.)
