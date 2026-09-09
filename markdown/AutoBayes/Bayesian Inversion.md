# Bayesian Inversion and the Bayesian Chain Rule

> AutoBayes, §3. The observation the whole paper is built on.

## Inversion of a kernel

For $c : X \rightsquigarrow Y$, the **Bayesian inversion** is a *function*

$$c^\dagger \;:\; \mathcal{P}X \longrightarrow \{Y \rightsquigarrow X\}$$

from priors to kernels going the other way, given by Bayes' law:

$$c^\dagger_\pi(dx \mid y) \;=\; \frac{c(dy \mid x)\, \pi(dx)}{(c_*\pi)(dy)}$$

Note the type. It is **not** a kernel $Y \rightsquigarrow X$; it is a *family* of them
indexed by the prior. The prior is an extra input that has to be supplied. Keeping that in
the type is the difference between a lens and an ordinary arrow.

## The chain rule

> Given $c : X \rightsquigarrow Y$ and $d : Y \rightsquigarrow Z$,
> $$(d \bullet c)^\dagger_\pi \;=\; c^\dagger_\pi \bullet d^\dagger_{c_*\pi}$$

Compare the reverse-mode chain rule of calculus:

$$\mathrm{d}_x(g \circ f) \;=\; \mathrm{d}_x f \circ \mathrm{d}_{f(x)}\,g$$

Line them up:

| autodiff | Bayes |
|---|---|
| $f, g$ | $c, d$ |
| point $x$ | prior $\pi$ |
| pushforward point $f(x)$ | pushforward prior $c_*\pi$ |
| $\mathrm{d}_x f$ | $c^\dagger_\pi$ |
| composite reverses order | composite reverses order |

**The prior plays the role of the linearisation point.** That single correspondence is the
seed of the entire AutoBayes framework, and it is the reason a Lenticulum factor's backward
pass takes a prior the way a Lux layer's backward pass takes cached activations.

Its proof is one line (substitute Bayes' law twice). The paper notes, correctly, that it is
"surprisingly ill-known".

## Why open models make it usable

The chain rule as stated involves $d \bullet c$, whose computation requires integrating out
$y$ — expensive. But the same statement holds for *open* models, where composition involves
no integral (see [[Composition of Open Models]]):

$$(d \circ\!\!\!\bullet\; c)^\dagger_\pi \;=\; c^\dagger_\pi \circ\!\!\!\bullet\; d^\dagger_{c_*\pi}$$

So: **open models are the setting in which the Bayesian chain rule is both true and cheap.**
That is the answer to "why did they bother inventing open models".

## The payoff: local inversions

Because the chain rule holds, you can attach an *approximate* inversion to each factor
locally, compose them, and the composite is automatically **structured correctly** as a
posterior for the composite model — even though it is not the exact posterior. You get to
be wrong locally and still be coherent globally.

This is the Bayesian analogue of "define `rrule` per primitive and let the AD system
compose them". It is what makes a modular library possible at all. Without it, every new
model architecture needs its own hand-derived ELBO.

> [!note] Footnote 3 of the paper
> Strictly, $(-)^\dagger$ is only *almost surely* a pseudofunctor: inversions may not be
> fully supported and are defined only up to a.s. equality. In practice this means
> numerically you must guard against zero-measure conditioning — an implementation concern
> for `bayes_lens.jl`.

Related: [[Bayesian Lens]], [[Composition of Bayesian Lenses]], [[Cartesian Reverse Differential Category]]
