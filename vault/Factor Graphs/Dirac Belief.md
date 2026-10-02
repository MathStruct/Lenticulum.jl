#definition

> `DiracBelief(x)` is a **point mass** $\delta_x$: certainty about a value. It is what an
> observed (hard-clamped) channel carries and what every point inference returns.

> Sources: code: `open_model.jl`, `messages.jl`; AutoBayes [arXiv:2503.18608](https://arxiv.org/abs/2503.18608), Appendix A, Example 4 (a cup collapses the posterior)
>
> Theory (CT-ML wiki): [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Copy-Discard Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Copy-Discard-Category)

## Where it comes from

- **Observations.** An `Observed()` channel has precision $\rho = \infty$ by default
  ([[Channels and Polarity]]); its value enters as a Dirac and is imposed by projection, never as a
  penalty. Categorically this is the *cup* of [[Open Models and Latent Channels]]: data clamped
  onto a wire.
- **Point inference.** Solvers that return a single configuration — RED-Diff, the implicit
  diffusion solver ([[Inference Signatures]] #1–#4), DEQ root-finding — return Diracs. Here
  the Dirac is an *approximation* of a posterior, not certainty.
- **Deterministic maps.** In a [Markov category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) the deterministic
  morphisms are exactly those that send Diracs to Diracs.

## How it behaves

- **It dominates `combine`.** Combined with a Gaussian, a sample set or the trivial belief, the
  Dirac wins: it is the $\Lambda \to \infty$ limit of a Gaussian, and infinite precision outweighs
  any finite one. Two *different* Diracs on one variable are an error ("contradictory hard
  clamps").
- **It has no entropy**: $H(\delta) = -\infty$. A factor whose inference returns a Dirac cannot
  contribute a meaningful entropy to the [[Bethe Free Energy]]
  ([[The Implicit Diffusion Factor as a Statistical Game]] §2).
- **It cannot negotiate.** Because it dominates, a factor that *outputs* a Dirac overrides its
  neighbours instead of pooling with them ([[The Diffusion Factor]] §4.2). The fix is a finite
  precision: a [[Gaussian Belief]].

`belief_distance` between two Diracs is the Euclidean distance of their values, and Diracs over
numbers or arrays can be damped (mixed) during message passing.

Related: [[Beliefs]], [[Gaussian Belief]], [[Trivial Belief]], [[Channels and Polarity]],
[[Messages are Inversions]]
