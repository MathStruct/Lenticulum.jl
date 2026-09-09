# Bayesian Lens — Definitions 9–11

> AutoBayes, Definitions 9, 10, Remark 11.

## Definition 9

> A **Bayesian lens** $X \mapsto Y$ consists of a pair $(c, c')$ of an open model
> $c : X \nrightarrow\!\!\!\bullet\; Y$ and a function
> $$c' : \mathcal{P}X \longrightarrow \{Y \rightsquigarrow X \times \llbracket c \rrbracket\}$$
> mapping priors on $X$ to kernels inverse to $c$.

Compare [[Lens|Definition 2.4]] side by side:

| | ordinary lens | Bayesian lens |
|---|---|---|
| forward (`get`) | $f : A \to B$ | $c : X \rightsquigarrow \llbracket c \rrbracket \times Y$ |
| backward (`put`) | $f^* : A \times B' \to A'$ | $c' : \mathcal{P}X \times Y \rightsquigarrow X \times \llbracket c \rrbracket$ |
| backward's extra input | the forward input $A$ | the **prior** $\pi \in \mathcal{P}X$ |

The shapes match. The `put` of a lens needs the point it was linearised at; the `put` of a
Bayesian lens needs the prior it was conditioned on. And note the backward pass returns
$X \times \llbracket c \rrbracket$ — **not just $X$**. It reconstructs the latent scratch
space too, which is exactly what the *next* factor down the chain will need.

## Definition 10 — the exact Bayesian lens

Every open model has a canonical inversion:

$$c^\dagger_\pi(dx, da \mid y) \;=\; \frac{c(da, dy \mid x)\,\pi(dx)}{\int_{a' : \llbracket c \rrbracket} (c_*\pi)(da', dy)}$$

giving the exact lens $(c, c^\dagger)$.

## Remark 11 — the notation, decoded

The paper's notation is unusual, deliberately. Here is the translation table, which is
worth internalising because the standard notation is genuinely ambiguous:

| paper | usual ML notation |
|---|---|
| $c(dy\mid x)$ | $p(y \mid x)$ — the likelihood / decoder |
| $\pi$ | $p(x)$ — the prior |
| $c^\dagger_\pi(dx\mid y)$ | $p(x \mid y)$ — the **exact** posterior |
| $c'_\pi(dx\mid y)$ | $q(x \mid y)$ or just $q(x)$ — the **approximate** posterior / encoder |

The usual notation writes both the exact and approximate posterior with the same letter $p$
or drops the conditioning entirely. The paper's $c$ vs $c'$ vs $c^\dagger$ makes it
impossible to confuse "the thing Bayes' law says" with "the thing your encoder network
computes", which is the single most common source of confusion when reading VI papers.

## The key liberty: $c'$ need not equal $c^\dagger$

$c'$ is *any* function of the right type. It can be:

- the exact posterior $c^\dagger$ (conjugate models);
- an amortised encoder network $q_\phi(x \mid y)$ (VAE);
- a Gaussian with learned mean and covariance (Laplace, mean-field VI);
- a particle set (SMC);
- a fixed point of a solver (implicit / DEQ);
- a diffusion model's denoiser (see [[ImplicitREDDiff]]).

**All of these are the same kind of object.** The quality of the choice is what
[[Statistical Game]] measures.

> [!note] Design consequence for Lenticulum
> A `LenticulumFactor` therefore has *two* Lux-style sub-models: the forward kernel and
> the inversion. Both have their own `ps`/`st`. This is the deep reason Lenticulum cannot
> be a Lux layer — a Lux layer has one direction, not two independently parametrised ones.

## Granularity is free

Footnote 4 makes a point worth pinning: nothing forces a level of granularity. You may

- build a complex model compositionally and attach *one* inversion to the whole thing
  (a monolithic amortised encoder), or
- attach an inversion to every factor and compose them as lenses (structured VI), or
- anything in between.

All three are the same formalism, differing only in where you place the $(-)'$ annotations.
In Lenticulum that decision becomes a *graph annotation*, not a rewrite.

Related: [[Bayesian Inversion]], [[Composition of Bayesian Lenses]], [[Lens]]
