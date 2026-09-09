# Statistical Game — Definition 20

> AutoBayes, Definition 20, Remark 21. **This is the object a Lenticulum factor is.**

## Definition

> A **statistical game** $c : X \multimap Y$ consists of a quadruple
> $(c,\; c',\; l^c,\; H^c)$ where
> - $(c, c')$ is a [[Bayesian Lens]] $X \mapsto Y$,
> - $l^c : X \times \llbracket c \rrbracket \times Y \to [0,\infty]$ is the **energy** or
>   **likelihood**,
> - $H^c : \mathcal{P}X \times Y \to [0,\infty]$ is the **entropy** or **regularizer**.
>
> These combine into a **loss** (a generalized free energy)
> $F^c : \mathcal{P}X \times Y \to [0,\infty]$:
> $$F^c(\pi, y) \;=\; \mathop{\mathbb{E}}_{(x,a) \sim c'_\pi(y)}\bigl[\,l^c(x,a,y)\,\bigr] \;-\; H^c(\pi, y)$$

Four pieces. Read them as four independent choices you get to make per factor:

| piece | what it is | what varies |
|---|---|---|
| $c$ | the generative kernel | your architecture |
| $c'$ | the inversion | exact / amortised / mean-field / solver / diffusion |
| $l^c$ | pointwise energy | NLL, reconstruction error, a robust loss, a residual norm |
| $H^c$ | regulariser | Shannon entropy, a KL to a prior, $\beta$-weighted, zero |

## Note the argument types carefully

- $l^c$ eats **points**: $(x, a, y)$. It is evaluated *inside* the expectation, at samples
  drawn from the inversion. Cheap, pointwise, no normalisation needed.
- $H^c$ eats a **distribution**: $(\pi, y) \in \mathcal{P}X \times Y$. It is a functional of
  the prior, not a function of a sample.

That type difference is the whole reason they compose differently. Energies are pointwise,
so they add. Entropies are functionals of distributions that get pushed forward, so they
chain. See [[Composition of Statistical Games]].

## The "game" in statistical game

Remark 21: the name references game theory — losses are utility/fitness functions, and
compositional game theory (Ghani et al. 2018) is also built on lenses (different ones). The
concept appeared in St Clere Smithe's earlier work, but **the novelty here is the correct
energy/entropy decomposition** and the recognition that the two halves compose differently.
If you have read the older "statistical games" papers, this definition supersedes them.

## Identity and the bicategory (Remark 24)

The identity game $X \multimap X$ is the identity lens with **constantly zero** energy and
entropy. Games form a bicategory. Zero is the unit of the energy monoid — worth noting,
because it means "a factor that does nothing contributes nothing to the loss", which is the
sanity property you want when you insert an identity into a graph.

## The subtlety about open free energy

Remark 24 flags a trap. For a pure game $c$ with $l^c = -\log p_c(y|x)$ and
$H^c = H(c'_\pi(y))$, the induced loss is

$$F^c(\pi, y) = \mathbb{E}_{x \sim c'_\pi(y)}[-\log p_c(y \mid x)] - H(c'_\pi(y))$$

which is **not** the VFE of [[Variational Free Energy|Proposition 18 (3)]] — it is missing
$-\log p_\pi(x)$. This is not an error; it is because $c$ is an *open* model, so this is an
"open free energy".

The fix is instructive: turn the prior $\pi : 1 \nrightarrow\!\!\!\bullet\; X$ into its own game
$\pi : 1 \multimap X$ by giving it a trivial inversion, setting $l^\pi(x) = -\log p_\pi(x)$,
and noting $H^\pi \equiv 0$ (trivial inversion ⇒ no entropy). Then

$$F^{c\pi}(\ast, y) \;=\; \mathrm{VFE}(c,c')(\pi, y)$$

**The prior term is not part of the model; it is a separate factor you compose with.**
This is exactly the [[README]]'s claim that "optimizers and losses become nonparametric
factors in our computation graph" — priors join that list. The free energy is genuinely a
compositional object, assembled from factors, not derived monolithically.

Related: [[Variational Free Energy]], [[Composition of Statistical Games]], [[Parameterized Statistical Game]], [[Scalar and Multivariate Energy]]
