# A linear factor read as a quantum measurement

Run `julia --project=. examples/measurement_factor.jl` from the repository root; the test is
`test/measurement_factor.jl`. A toy, built to see whether the idea works, not a package factor.

It is the spectral baseline of the design note
[`Minimal Stochastic Factors`](../vault/Design/Minimal%20Stochastic%20Factors.md) (§7), which explains why it is
not polarity-compatible and proposes the switching constraint factor and the Born joint instead.

## The idea

A quantum measurement does not denoise: it picks an outcome with a probability given by the
state and returns the corresponding eigenvector, in one step. Applied to the simplest linear
factor:

- the factor's state is a density matrix over $z = (x, y)$,
  $\rho = Z^\top Z / \operatorname{tr}(Z^\top Z) = \sum_k \lambda_k u_k u_k^\top$, the normalised
  second moment of the data (closed form, no training loop);
- with $x$ clamped, outcome $k$ has the Born weight
  $p(k \mid x) \propto \lambda_k\, (u_k^{x} \cdot x)^2$, and returns the point on eigen-line $k$
  whose $x$-part best matches $x$, $y_k = u_k^{y}\, (u_k^{x} \cdot x)/\lVert u_k^{x}\rVert^2$.

As a Mycelium factor the message to $y$ is a `MixtureBelief` with one Gaussian per
eigenvector (width fitted from the residuals to the nearest component), so it is multimodal
and needs no iterative sampler. Statistically it is a mixture of rank-one linear experts
whose gate comes from the spectrum.

## Measured (2026-10-31; 20 000 training and 5 000 held-out points, noise $\sigma = 0.05$)

| data | eigen-lines | held-out mean $\log p(y \mid x)$: measurement | one linear-Gaussian | true model |
|---|---|---|---|---|
| $y = x$ (0.3) and $y = -x$ (0.7) | 45.00°, 135.00° (exact) | 0.688 | −0.977 | 0.691 |
| lines at 0° and 60° (0.5 each) | 30.03°, 120.03° (30° off) | −0.939 | −0.823 | 0.702 |

In the first case the Born weights at $x = 1$ are `[0.295, 0.705]` and the means `[1.0, -1.0]`;
of 10 000 one-shot measurements 28.7 % land on $y = +1$. The measurement is as good as the
true model and far better than a single Gaussian, which puts one broad blob between the
lines. In the second case it is worse than the single Gaussian.

## Why the second case fails

Eigenvectors of a symmetric matrix are orthogonal, so a measurement in the eigenbasis can
only return orthogonal directions. Two lines 60° apart are not orthogonal, so neither can be
an eigenvector; the eigen-lines settle symmetrically between them. This is the classical
face of a quantum fact: non-orthogonal states cannot be distinguished by a projective
measurement.

It is also a limit of the data the factor keeps. A mixture $\sum_k w_k v_k v_k^\top$ of
non-orthogonal directions has the same second moment as infinitely many other mixtures, so
no rule that sees only $\rho$ can recover the lines. Expectations, not tested here: a
non-orthogonal measurement (a POVM, $\sum_k w_k v_k v_k^\top$ with the $v_k$ the lines) can
return the right lines if they are known, and finding them from data needs more than second
moments, e.g. a third-moment tensor decomposition after whitening by $\rho$.

## What the toy does not do

- Lines through the origin only: the factor is linear, without the homogeneous coordinate
  $[1; x; y]$ that would give it offsets.
- With a scalar $x$ the Born weights do not depend on $x$, whereas the true conditional
  weights do whenever the lines' $x$-variances differ; here they are equal (both $\cos^2 45°$).
- No free energy, no learning beyond the closed-form fit, no feature lift.
