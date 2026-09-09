# Open Model — Definition 1

> AutoBayes, Definitions 1–3, Remark 2.

## Definition

> If $X$ and $Y$ are measurable spaces, then an **open model** $p : X \nrightarrow\!\!\!\bullet\; Y$
> consists of a pair of a measurable space $\llbracket p \rrbracket$ and a measure kernel
> $$p : X \rightsquigarrow \llbracket p \rrbracket \times Y$$

Three spaces, three names:

| space | name | intuition |
|---|---|---|
| $X$ (domain) | **unobserved** | what we want to infer; "the parameter / latent variable" in ordinary usage |
| $Y$ (codomain) | **observed** | where the data lives |
| $\llbracket p \rrbracket$ | **latent** | scratch space; things that were observable in the parts but got hidden by composition |

Naming special cases:

- $\llbracket p \rrbracket \cong 1$: **pure model** (an ordinary kernel $X \rightsquigarrow Y$)
- $X \cong \llbracket p \rrbracket \cong 1$: **pure distribution** (a prior)
- $X \cong 1$, $\llbracket p \rrbracket \not\cong 1$: **joint distribution**

## Why "open"

Because they are open *to composition*, hence not complete. An open model corresponds to a
**conditional** distribution. Only $1 \nrightarrow\!\!\!\bullet\; 1$ is closed.

## Why a third space at all

This is the single most important design move in the paper, and it is easy to skim past.

Compose two pure models $p : X \rightsquigarrow Y$ and $q : Y \rightsquigarrow Z$. The
ordinary Chapman–Kolmogorov composite $\int_y q(dz|y)\,p(dy|x)$ *integrates $y$ out* — the
intermediate value is destroyed. But for inference you need it back: the posterior over $X$
given $z$ runs through $y$. So AutoBayes refuses to integrate, and instead *files $y$ away*
in the latent space:

$$\llbracket q \circ\!\!\!\bullet\; p \rrbracket = \llbracket p \rrbracket \times Y \times \llbracket q \rrbracket$$

$$(q \circ\!\!\!\bullet\; p)(ds, dy, dt, dz \mid x) = q(dt, dz \mid y)\, p(ds, dy \mid x)$$

**Composition of open models involves no integration.** That is stated almost in passing in
§3 of the paper and it is the reason the whole framework is computable: the expensive
marginalisation is deferred, not performed.

## The trade you are making

You have traded *space* for *tractability*. The latent space of a deep composite is the
product of every intermediate space — it grows linearly with depth. In implementation terms
$\llbracket p \rrbracket$ **is the activation cache**. Exactly like reverse-mode autodiff,
which stores every intermediate activation so the backward pass can use it. The analogy is
not loose; it is the same phenomenon under [[Bayesian Inversion|the same chain rule]].

> [!note] Implementation consequence for Lenticulum
> `⟦c⟧` is a first-class field of a factor's *state*, not of its output. A factor's forward
> pass returns `(y, latent, st)`, and `latent` is what the inversion consumes. See
> [[statistical_game]].

## Example

Model: cloud cover $X \in \{\text{clear}, \text{cloudy}\}$ → rainfall
$Y \in \mathbb{R}_{\ge 0}$, via an intermediate humidity $H$.

- $p : X \rightsquigarrow H$, pure. $\llbracket p \rrbracket = 1$.
- $q : H \rightsquigarrow Y$, pure. $\llbracket q \rrbracket = 1$.
- $q \circ\!\!\!\bullet\; p : X \nrightarrow\!\!\!\bullet\; Y$ has $\llbracket q \circ\!\!\!\bullet\; p \rrbracket = 1 \times H \times 1 = H$.

The composite is *not* pure: humidity is still in there, as latent. If you want it back as
an output you apply `reveal` (a 2-cell) to move it from $\llbracket - \rrbracket$ to $Y$.
If you want it truly gone you marginalise — and pay for it.

## Remark 3 — relation to probabilistic programming languages

AutoBayes does not replace Pyro/Turing/Gen. Probabilistic programs *define* kernels; those
kernels define open models; AutoBayes composes and optimises the models. AutoBayes sits one
level above a PPL. For Lenticulum this means: a factor's forward kernel may perfectly well
be a `Turing.jl` model or a Lux network with a sampling head.

Related: [[Composition of Open Models]], [[Copiers Cups and Caps]], [[Bayesian Lens]]
