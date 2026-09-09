# Composition of Statistical Games — Definition 22, Theorem 23

> AutoBayes, Definition 22, Theorem 23, Remarks 24 and 26, Definition 25.
> **This is the heart of the paper.**

## Definition 22

Given games $c : X \multimap Y$ and $d : Y \multimap Z$, the composite $d \diamond c : X \multimap Z$:

- **Lens**: the composite Bayesian lens $(d,d') \diamond (c,c')$ ([[Composition of Bayesian Lenses|Definition 12]]).
- **Energy** $l^{dc} : X \times \llbracket c \rrbracket \times Y \times \llbracket d \rrbracket \times Z \to [0,\infty]$:
  $$\boxed{\;l^{dc}(x,a,y,b,z) \;:=\; l^c(x,a,y) \;+\; l^d(y,b,z)\;}$$
- **Entropy** $H^{dc} : \mathcal{P}X \times Z \to [0,\infty]$:
  $$\boxed{\;H^{dc}(\pi, z) \;:=\; \mathop{\mathbb{E}}_{(y,b)\sim d'_{c_*\pi}(z)}\bigl[H^c(\pi,y)\bigr] \;+\; H^d(c_*\pi, z)\;}$$

Stare at the asymmetry. **Energies just add.** No expectation, no pushforward — they are
pointwise functions and the composite's argument tuple simply concatenates. **Entropies
chain**: the upstream entropy is averaged under the downstream inversion, and the
downstream entropy is evaluated at the pushforward prior.

## Theorem 23 — the chain rule for free energy

$$F^{dc}(\pi, z) \;=\; \mathop{\mathbb{E}}_{(y,b)\sim d'_{c_*\pi}(z)}\bigl[F^c(\pi, y)\bigr] \;+\; F^d(c_*\pi, z)$$

Derivation (worth doing once):

$$F^{dc}(\pi,z) = \mathop{\mathbb{E}}_{(x,a,y,b)}\bigl[l^{dc}\bigr] - H^{dc}(\pi,z)$$
$$= \mathop{\mathbb{E}}_{(y,b)}\Bigl[\mathop{\mathbb{E}}_{(x,a)}[l^c(x,a,y)] - H^c(\pi,y) + l^d(y,b,z)\Bigr] - H^d(c_*\pi,z)$$
$$= \mathop{\mathbb{E}}_{(y,b)}\bigl[F^c(\pi,y)\bigr] + F^d(c_*\pi,z)$$

The additive energy and the chained entropy conspire to give a **single clean recursion for
the total loss**. Neither half alone has this property; that is why they must be tracked
separately.

## What this buys you

> It is unnecessary to derive complex loss functions monolithically by hand, as is often
> done in statistical machine learning: they may be composed mechanistically and locally
> instead, just like gradients in differentiable programming.

Concretely: you never again write down an ELBO for a hierarchical model by hand. You
declare each factor's $(c, c', l, H)$ and the framework accumulates $F$ by the recursion
above. This is *the* reason to build Lenticulum.

## Definition 25 — parallel composition

$$l^{c \otimes d}(x,x',a,a',y,y') = l^c(x,a,y) + l^d(x',a',y')$$
$$H^{c \otimes d}(\omega, y, y') = H^c(\omega_X, y) + H^d(\omega_{X'}, y')$$

Both halves add, because in parallel there is no upstream/downstream.

## Remark 26 — the laxness, quantified

> In the case that the entropies are Shannon, the laxness of the tensor is measured by the
> **mutual information**.

That is: $H^{c \otimes d}$ built from marginals differs from the true joint entropy by
$I(X; X')$. See [[Composition of Bayesian Lenses]] Remark 16. This is a *number you can
compute and report*, not an unquantified approximation. Lenticulum should surface it.

## Implementation shape

The recursion is a fold over the graph. In pseudo-Julia:

```julia
function free_energy(chain, π, z)
    # backward sweep: sample the inversions from the observed end
    F = 0.0
    priors = accumulate_pushforward(chain, π)      # π, c₁*π, c₂*c₁*π, ...
    samples = sample_inversions(chain, priors, z)   # z ↦ (y_n,b_n) ↦ ... ↦ (x,a)
    for (factor, prior, obs) in zip(chain, priors, samples)
        F += free_energy_local(factor, prior, obs)
    end
    return F
end
```

Two sweeps: **priors forward, samples backward**. Identical in shape to forward/backward
autodiff. The differences from autodiff are (i) the backward sweep is stochastic, and
(ii) the forward sweep carries distributions, not points.

> [!warning] The two expensive steps
> The paper's own closing discussion names them: computing the **pushforward priors**
> $c_*\pi$ (marginalisation) and taking the **expectations under the inversions**. Both are
> approximated in practice — by belief propagation / variational message passing for the
> former (this is Mycelium.jl's job) and by sampling or conjugacy for the latter.

Related: [[Statistical Game]], [[Parameterized Statistical Game]], [[Scalar and Multivariate Energy]]
