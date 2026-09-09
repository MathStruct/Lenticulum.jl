# Variational Free Energy — Definition 17, Proposition 18

> AutoBayes, Definition 17, Proposition 18, Remark 19. **The energy/entropy split lives here.**

## The question being answered

[[Bayesian Lens]] lets you attach *any* approximate inversion $c'$. How good is it? The
first answer is the KL divergence to the exact posterior:

$$\mathrm{KL}(c,c')(\pi, y) \;=\; D_{KL}\bigl(c'_\pi(y),\; c^\dagger_\pi(y)\bigr)$$

And it, too, has a chain rule:

$$\mathrm{KL}\bigl[(d,d') \diamond (c,c')\bigr](\pi, z)
\;=\; \mathbb{E}_{(y,b) \sim d'_{c_*\pi}(z)}\bigl[\mathrm{KL}(c,c')(\pi,y)\bigr] \;+\; \mathrm{KL}(d,d')(c_*\pi, z)$$

Beautifully compositional. **But useless**, because evaluating it requires $c^\dagger_\pi$,
the intractable exact posterior. So you bound it.

## Definition 17

$$\mathrm{VFE}(c,c')(\pi, y) \;=\; \mathrm{KL}(c,c')(\pi,y) \;-\; \log p_{c_Y \bullet \pi}(y)$$

where $c_Y$ is the $Y$-marginal of $c$, so
$(c_Y \bullet \pi)(dy) = \int_{a : \llbracket c \rrbracket} (c_*\pi)(da, dy)$.

VFE = relative entropy + negative marginal log-likelihood. Elsewhere: the **evidence upper
bound (EUBO)**, or negative ELBO.

## Proposition 18 — three forms

$$\mathrm{VFE}(c,c')(\pi,y) = \mathop{\mathbb{E}}_{(x,a)\sim c'_\pi(y)}\Bigl[\log p_{c'_\pi}(x,a\mid y) - \log p_{c^\dagger_\pi}(x,a\mid y)\Bigr] - \log p_{c_Y\bullet\pi}(y) \tag{1}$$

$$= \mathop{\mathbb{E}}_{(x,a)\sim c'_\pi(y)}\Bigl[\log p_{c'_\pi}(x,a\mid y) - \log p_c(a,y\mid x) - \log p_\pi(x)\Bigr] \tag{2}$$

$$= \underbrace{\mathop{\mathbb{E}}_{(x,a)\sim c'_\pi(y)}\bigl[-\log p_c(a,y\mid x) - \log p_\pi(x)\bigr]}_{\text{expected \textbf{energy}}} \;-\; \underbrace{H\bigl(c'_\pi(y)\bigr)}_{\textbf{entropy}} \tag{3}$$

**Form (2) is the point: $c^\dagger_\pi$ has vanished.** The log-likelihood term interacted
with the KL to cancel the intractable posterior. That cancellation is the entire reason
variational inference works, and it is worth staring at (1) → (2) until it is obvious.

## Form (3) is the seed of the framework

The Helmholtz free energy is (expected energy) − (entropy). Form (3) has exactly that
shape, and the paper's central observation is:

> The energy and entropy parts **behave differently** [under composition]. Energies compose
> by simple addition, but entropies (and thus losses built from them) compose like the
> chain rule.

So: do not carry $F$ around as one number. Carry $(l, H)$ as a pair, because they have
different composition laws. That pair is a [[Statistical Game]].

## Why Lenticulum splits it *again*

Lenticulum keeps a third distinction the paper does not: the energy in the paper is
$[0,\infty]$-valued, and its composition law "energies add" is exactly the statement that
$([0,\infty], +, 0)$ is a commutative monoid. But addition is a *lossy* projection of the
structure you actually have — which factor contributed which residual, and in which
coordinates. Keeping the vector is what makes Gauss–Newton, precision weighting, and the
implicit function theorem available. See [[Scalar and Multivariate Energy]].

## Remark 19 — prior art

Knoblauch et al. (generalized VI) and Khan & Rue (the Bayesian learning rule) both observe
that learning objectives split into (loss in expectation under a posterior) + (a
divergence/entropy regulariser). Neither notices the *compositional* consequences. That is
AutoBayes' contribution.

Related: [[Statistical Game]], [[Bayesian Lens]], [[Scalar and Multivariate Energy]]
