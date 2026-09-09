# Composition of Bayesian Lenses — Definitions 12, 15; Theorem 13

> AutoBayes, Definition 12, Theorem 13, Remarks 14, 16, Definition 15.

## Definition 12 — sequential composition

Given $(c, c') : X \mapsto Y$ and $(d, d') : Y \mapsto Z$, the composite is

$$(d, d') \diamond (c, c') \;=\; \bigl(d \circ\!\!\!\bullet\; c,\; c' \mathbin{\hat{\circ}} d'_c\bigr)$$

where the backward part maps $\pi \in \mathcal{P}X$ to the kernel

$$\bigl(c'_\pi \mathbin{\hat{\circ}} d'_{c_*\pi}\bigr)(dx, da, dy, db \mid z)
\;=\; c'_\pi(dx, da \mid y)\; d'_{c_*\pi}(dy, db \mid z)$$

Read the right-hand side right-to-left: $d'$ conditioned on the **pushforward prior**
$c_*\pi$ turns $z$ into $(y, b)$; then $c'$ conditioned on the **original prior** $\pi$
turns that $y$ into $(x,a)$.

Two things to notice:

1. **The prior propagates forward, the correction propagates backward.** $\pi \to c_*\pi$
   going right; $z \to y \to x$ coming back. Same shape as [[Lens]]'s
   $f^*(a, g^*(f(a), c'))$, with the prior in the role of the cached activation.
2. Again **no integral**. The composite inversion is a product of the local inversions.

## Theorem 13 — the chain rule for open models

> $(d \circ\!\!\!\bullet\; c)^\dagger = (d, d^\dagger) \diamond (c, c^\dagger) = (d)^\dagger \diamond (c)^\dagger$.
> That is, $(-)^\dagger$ is functorial.

$(-)^\dagger$ is a pseudofunctor from the bicategory of open models to the bicategory of
Bayesian lenses. **This is the exact analogue of [[Cartesian Reverse Differential Category|Proposition 2.7]]**
($R : \mathcal{C} \to \mathbf{Lens}(\mathcal{C})$ is a functor). One says "autodiff is
correct"; the other says "compositional inference is correct".

## Definition 15 and Remark 16 — parallel composition is lossy

$$\bigl((c' \otimes d')_\omega\bigr)(dx, dx', da, da' \mid y, y')
= c'_{\omega_X}(dx, da \mid y)\; d'_{\omega_{X'}}(dx', da' \mid y')$$

where $\omega_X, \omega_{X'}$ are the marginals of a joint prior $\omega$.

> Composing inversions in parallel is **lossy**, because the inversions being composed can
> only accept the *marginals* of a joint prior. Consequently $(-)^\dagger$ is only a **lax
> monoidal** functor: $(c \otimes d)^\dagger \neq c^\dagger \otimes d^\dagger$.

### What laxness means concretely

If your prior over $(X, X')$ has correlations, the parallel composite throws them away —
each branch sees only its own marginal. The composite inversion is therefore a *mean-field*
approximation of the true joint inversion, and the discrepancy is the **mutual information**
$I(X; X')$ under $\omega$ (Remark 26 makes this precise once entropies are Shannon).

> [!warning] This is not a bug you can fix
> Laxness is intrinsic. It is the formal statement of *why* mean-field VI is wrong, and by
> exactly how much. An implementation has two honest options: (a) accept the laxness and
> track the 2-cell (the KL gap) as a diagnostic; (b) refuse to factor across correlated
> branches and keep a joint inversion. Lenticulum should expose this as a choice at the
> graph level, and *report* the gap rather than hiding it.

## Cups and caps for lenses

$\mathrm{cup}_A : 1 \mapsto A \otimes A$ is the pair $(\mathrm{cup}_A, \mathrm{cap}_A)$ —
the cup forward, the (constant function on the) cap as its inversion. The cap is dual.
See [[Copiers Cups and Caps]].

Related: [[Bayesian Lens]], [[Composition of Statistical Games]], [[Composition of Open Models]]
