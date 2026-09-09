# Channels and Polarity

> Reconciling AutoBayes' $(X, \llbracket c \rrbracket, Y)$ with [[ImplicitREDDiff]]'s
> $P_{in} + P_{out} + P_{latent} = \mathrm{Id}$.

## The two trichotomies

AutoBayes ([[Open Model|Definition 1]]) gives every model three spaces:

$$c : X \rightsquigarrow \llbracket c \rrbracket \times Y$$

| | AutoBayes name | role |
|---|---|---|
| $X$ | **unobserved** | inferred; the posterior is over this |
| $Y$ | **observed** | clamped to data |
| $\llbracket c \rrbracket$ | **latent** | internal scratch, revealed or marginalised |

[[ImplicitREDDiff]] gives every factor three diagonal selection matrices with
$P_{in} + P_{out} + P_{latent} = \mathrm{Id}$ and pairwise-zero products, assembling a
precision-weighted projector

$$P \;=\; \rho_{in}P_{in} + \rho_{out}P_{out} + \rho_{latent}P_{latent}$$

> [!warning] The names cross over — read this twice
> AutoBayes' **unobserved** $X$ is the thing you *solve for*, i.e. the note's **output**
> $P_{out}$. AutoBayes' **observed** $Y$ is the thing you *clamp*, i.e. the **input**
> $P_{in}$.
>
> $$X \;\longleftrightarrow\; P_{out}, \qquad Y \;\longleftrightarrow\; P_{in}, \qquad \llbracket c \rrbracket \;\longleftrightarrow\; P_{latent}$$
>
> The inversion $c'_\pi : Y \rightsquigarrow X \times \llbracket c \rrbracket$ runs
> *input → output* in the note's sense. The forward kernel $c$ runs the other way. The
> mismatch is real, not a typo in either source: "observed" is a *statistical* word about
> data availability, "input" is an *operational* word about evaluation order, and in
> Bayesian inference those point in opposite directions.

Once that is said, the correspondence is exact, and $\rho$ has a clean reading: the
precision with which each block is clamped. $\rho_{in} \to \infty$ is a hard clamp
(a [[Copiers Cups and Caps|cup]]); $\rho_{out} = 0$ leaves the block free.

## Why polarity must be dynamic

A Lux layer has a fixed direction. A Lenticulum factor is a **relation**
$R_\theta \subseteq X_1 \times \cdots \times X_n$ and has none: the same factor
$x^2 + y^2 = 1$ can be asked for $y$ given $x$, or $x$ given $y$, or neither given nothing.

Categorically this is [[Copiers Cups and Caps|compact closure]]: the cup and cap let you
bend any leg from unobserved to observed and back. Operationally it means a factor's
`get`/`put` pair *does not exist until you choose*. Hence:

```
Factor  +  Polarity  ──►  parametric lens  ──►  message
```

## The data type

A **channel** is a named port of a factor with a space attached. A **polarity** is an
assignment of one of three states to each channel:

```julia
abstract type ChannelPolarity end
struct Observed  <: ChannelPolarity end   # Y   / P_in    / clamped to data
struct Unobserved<: ChannelPolarity end   # X   / P_out   / inferred, posterior over it
struct Latent    <: ChannelPolarity end   # ⟦c⟧ / P_latent/ internal, marginal or revealed
```

with `Polarity` a `NamedTuple{names}` of these. The type-level `names` means a polarised
factor's lens type is known at compile time and Julia can specialise the assembled kernel —
this is the payoff for putting the polarity in the type domain rather than a runtime `Dict`.

## Legality

Not every polarity is legal for every factor. A `Gaussian(μ, σ)` factor can be inverted
for `μ` given `x` and `σ`, but a factor whose forward kernel is a one-way hash cannot be
inverted at all. So a factor declares which polarities it supports:

```julia
supports_polarity(factor, pol)::Bool
```

and a graph compiles only if every scheduled message uses a supported polarity. **The
DAG restriction of Lux is replaced by a polarity-legality check.** That is the concrete
form the [[README]]'s "arbitrary connected graph" claim takes in the type system.

## The three ways a polarity can be realised

| how the factor answers a polarity | model family | cost |
|---|---|---|
| closed form (conjugate, invertible map) | Gaussian/linear, normalising flows | cheap |
| root-finding on a residual $r_\theta(x) = 0$ | algebraic varieties, DEQ, NeuralODE | Newton/fixed point |
| proximal step on an energy | diffusion / RED-Diff | iterative denoising |

These are exactly the three families of [[Implicit Learners]]. Which one is available is a
property of the factor; the *graph* does not need to know, which is the point of the
abstraction.

Related: [[Open Model]], [[Copiers Cups and Caps]], [[Implicit Learners]], [[ImplicitREDDiff]], [[channels]]
