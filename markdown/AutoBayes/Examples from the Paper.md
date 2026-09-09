# Examples from the Paper — Appendix A and B

> AutoBayes, Appendix A (Examples 1–5) and Appendix B.
> These are the sanity checks: every example is a factor-graph wiring, not a new derivation.

## Example 1 — Gaussian mixture / maximum likelihood

Unobserved finite set $M$, Gaussian $c$ over observed $Y$ given $M$, prior $\pi$ over $M$.
The marginal $c_*\pi$ on $Y$ is a mixture of Gaussians.

- $c : M \multimap Y$ with exact inversion $c^\dagger$, energy $= $ NLL, entropy $= $ Shannon.
- $\pi : 1 \multimap M$ with trivial inversion (so $H^\pi = 0$) and $l^\pi = -\log p_\alpha$.
- Under these choices $F^{c\pi}(\ast, y) = -\log p_{c_*\pi}(y)$.
- **No parameter on $c$**; $\pi$ is parameterized by the mixing probabilities
  $\alpha$: the function $\mathcal{P}M \to \{1 \multimap M\}$ maps $\alpha$ to
  $(\alpha,\; \cdot,\; -\log p_\alpha,\; 0)$.

Descending the $\alpha$-gradient of $F^{c\pi}(\ast, y; \alpha)$ **is** maximum likelihood in
the mixing probabilities. Note the moral: MLE is what you get when all entropies are zero.

## Example 2 — Expectation–maximization

Lens $c : X \mapsto Y$ and prior lens $\pi : 1 \mapsto X$ (trivial inversion), both with NLL
energies and **zero entropies**. Then

$$F^{c\pi}(\ast, y) = \mathop{\mathbb{E}}_{x\sim c'_\pi(y)}\bigl[-\log p(x,y)\bigr],
\qquad p(x,y) = p_c(y\mid x)\,p_\pi(x)$$

Computing this is the **E-step**. Parametrising the composite in $\Theta$ and maximising
over $\theta$ is the **M-step**. So EM is: *evaluate the composite loss, then descend it* —
the two halves of the algorithm are the two halves of the framework, not two separate ideas.

## Example 3 — Variational Bayesian EM

Extend so $\Theta$ is *part of the model*: games $c : \Theta \otimes X \multimap Y$ and
$\pi : \Theta \multimap X$, composed to $c\pi : \Theta \multimap Y$ by
$\theta \mapsto c(\theta, -) \diamond \pi(\theta)$. Add a hyperprior $\psi$ on $\Theta$
parameterized in $\Psi$, giving $(\Psi, \psi) : 1 \multimap \Theta$, and compose with
$(1, c\pi) : \Theta \multimap Y$. VBEM = gradient descent on the composite loss w.r.t. $\Psi$.

The move to notice: **$\Theta$ migrated from the parameter space into the wire.** A
parameter you want a posterior over is not a parameter — it is a variable. In a factor
graph that distinction is a property of the node, not of the type, which is exactly why
Lenticulum's factors have *channels* rather than a fixed input/output split. See
[[Channels and Polarity]].

## Example 4 — Supervised learning (uses the cup)

You *know* the "unobserved" labels in $X$ corresponding to observed data in $Y$. Compose
$X \otimes c : X \otimes X \multimap X \otimes Y$ after the cup $1 \multimap X \otimes X$.
Now both $X$ and $Y$ are observed.

- The inversion $c'$ typically **trivializes** — the prior is a deterministic sample, so
  there is nothing left to infer.
- The regularizer $H^c$ may not trivialize.
- The resulting loss depends only on the parameter and the paired data.

**Ordinary supervised learning is the degenerate case of the framework where the cup has
collapsed the posterior.** That is a satisfying place to land: Lenticulum reduces to Lux
when every wire is clamped.

## Example 5 — Bayesian deep learning

Take $c : \Theta \otimes X \multimap Y$ and apply the cup to $X$ *only*, yielding
$\Theta \multimap X \otimes Y$.

- The forward part of $c$ is a neural network with weights in $\Theta$ making a stochastic
  prediction of $Y$.
- $l^c(\theta, x, y)$ is a complex ML loss.
- A prior on the weights is a game $1 \multimap \Theta$.
- If $c'$ is **mean-field factorized** over $\Theta$ and $X$ independently, the cup
  trivializes the $X$ factor, leaving only a posterior over $\Theta$ — the classic Bayesian
  deep learning setup.

And "both $c$ and the prior may themselves be complex models constructed compositionally".

## Appendix B — dependent types

Ordinarily a joint $p(dx,dy)$ lives on $X \times Y$ — "$X$-many copies of $Y$", the same $Y$
for every $x$. In a **dependently typed** model the space $Y_x$ varies with $x$, and the
joint lives on the dependent sum $\sum_{x \in X} Y_x$. A conditional becomes a **stochastic
section** of the projection $\sum_x Y_x \to X$: a kernel $X \rightsquigarrow \sum_x Y_x$ that
preserves $X$.

The paper's example: a weather model whose report *type* differs at sea (tides, wind) from
on land. Enforcing this with types saves the model from inferring it from data.

> [!note] Relevance to Lenticulum
> Julia's type system plus multiple dispatch is unusually well-suited to this: a factor's
> channel type can genuinely depend on a value via a type parameter, and dispatch will
> select the right kernel. This is a real advantage of Julia over a Python implementation,
> and worth keeping in view even though it is not needed for v0.

Related: [[Statistical Game]], [[Copiers Cups and Caps]], [[Channels and Polarity]]
