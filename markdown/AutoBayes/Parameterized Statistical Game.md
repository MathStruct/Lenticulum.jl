# Parameterized Statistical Game — Definitions 27, 28

> AutoBayes, §5, Definitions 27 and 28. **The definition [[README]] singles out.**

## The gap it fills

$F^c(\pi, y)$ exposes no parameter. There is nothing to optimise. §5 fixes that in the
simplest possible way.

## Definition 27

> A **parameterized statistical game** $X \multimap Y$ is a pair $(\Theta, c)$ of a space
> $\Theta$ and a function
> $$c : \Theta \longrightarrow \{X \multimap Y\}$$
> from $\Theta$ to the set of statistical games $X \multimap Y$. We may write each of the
> components of the game as $c(dy \mid x; \theta)$, $c'_\pi(dx \mid y; \theta)$,
> $l^c(x,y;\theta)$ and $H^c(\pi, y; \theta)$.

The deliberate design choice: **any part of the game may depend on the parameter.** Not
just the forward kernel. That covers, uniformly:

| what $\theta$ parametrises | the model this gives you |
|---|---|
| $c$ only | a plain generative model / decoder |
| $c'$ only | an amortised encoder (VAE), or a learned posterior's natural parameters |
| $l^c$ only | a learned loss / critic / temperature |
| $H^c$ only | a learned regulariser, $\beta$-annealing |
| all of them | end-to-end deep Bayesian model |

The Bayesian learning rule of Khan & Rue is the case where $\Theta$ picks the *natural
parameter of the posterior* — and the paper's footnote 6 points out that for them $c'$ is
merely $\Theta \to \mathcal{P}X$, not $\Theta \times \mathcal{P}X \times Y \to \mathcal{P}X$, so
the compositional structure is invisible in their treatment. Lenticulum keeps the full type.

## This is Para, again

$(\Theta, c)$ with $c : \Theta \to \{X \multimap Y\}$ is exactly a
[[Para|$\mathbf{Para}$-morphism]] in the category of statistical games. Compare Definition 2.1:
a $\mathbf{Para}(\mathcal{C})$-map is $(P, f)$ with $f : P \otimes A \to B$. Same shape,
one level up: instead of parametrising a *map*, you parametrise a whole *lens-with-losses*.

So:

$$\underbrace{\mathbf{Para}(\mathbf{Lens}(\mathcal{C}))}_{\text{Lux.jl}}
\qquad\text{vs.}\qquad
\underbrace{\mathbf{Para}(\mathbf{StatGame})}_{\text{Lenticulum.jl}}$$

and $\mathbf{StatGame}$ is $\mathbf{BayesLens}$ decorated with $(l, H)$. That is the
one-line summary of the difference between the two libraries. See
[[AutoBayes to Lenticulum]].

## Definition 28 — composition

> Given $(\Theta, c) : X \multimap Y$ and $(\Phi, d) : Y \multimap Z$, their composite is
> $(\Phi \times \Theta,\; d \diamond c)$ where $d \diamond c$ maps $(\varphi, \theta)$ to
> $d(\varphi) \diamond c(\theta)$.

Parameter spaces multiply, exactly as in $\mathbf{Para}$. In Julia this is a nested
`NamedTuple`, and it is *literally the same data structure* Lux uses for
`Chain(Dense(...), Dense(...))`.

## What you optimise

Assuming $c$ differentiable, form $F^c(\pi, y; \theta)$ and compute $\nabla_\theta F^c$.

> In many applications, $\Theta$ will parameterize a **statistical manifold**, and thus be
> equipped with the **Fisher information metric**. Gradient descent of generalized free
> energy with respect to this metric is what Khan and Rue call the **Bayesian learning
> rule**, and it is this kind of gradient descent that we take to be the standard semantics
> for AutoBayes.

Note "with respect to this metric" — the default semantics is **natural gradient**, not
plain gradient. This matters for Lenticulum: the parameter-update space $P'$ of
[[Parametric Lens]] is a *cotangent* space, and turning a covector into a parameter update
requires a metric. Keeping the [[Scalar and Multivariate Energy|multivariate energy]] is
what makes a Gauss–Newton approximation of that metric available for free.

Related: [[Composition of Gradients]], [[Statistical Game]], [[Para]], [[AutoBayes to Lenticulum]]
