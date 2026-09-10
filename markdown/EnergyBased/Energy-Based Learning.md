# Energy-Based Learning

> LeCun, Chopra, Hadsell, Ranzato & Huang, *A Tutorial on Energy-Based Learning* (2006)
> — [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf).
>
> **This is the paper [[README]]'s table was written from**, whether or not anyone had read it
> at the time. Energy, no normalisation, inference by minimisation, a model that scores
> configurations rather than mapping inputs to outputs — every row of that table is in this
> tutorial, and the AutoBayes machinery was layered on top afterwards.
>
> The vault has been calling this the *third* backbone since this note was written; it is
> arguably the *first*.

## 1. The framework

An energy-based model is a function

$$E(W, Y, X) \;\in\; \mathbb{R}$$

scoring the compatibility of an answer $Y$ with an observation $X$, given parameters $W$.
Inference is minimisation:

$$Y^\ast \;=\; \operatorname*{arg\,min}_{Y\in\mathcal{Y}} E(W, Y, X)$$

and **that is the whole model**. There is no requirement that $\exp(-E)$ integrate to
anything. The partition function is not approximated, not bounded, not estimated — it is
simply never mentioned.

Compare [[README]]'s own table, row by row:

| [[README]] | LeCun |
|---|---|
| approximator: relations $R_\theta \subseteq X_1\times\cdots\times X_n$ | an energy over configurations |
| $r_\theta(x_1,\ldots,x_n) \approx 0$ | low energy |
| inference: root-finding | inference: $\arg\min_Y E$ |
| loss $\|r_\theta\|^2$ | the energy is $\|r\|^2$ |
| symmetric, no distinguished input/output | $E(Y,X)$ is a scalar on a joint configuration |
| may be multi-valued or have no solution | the energy surface has whatever minima it has |

The last row is the one that matters most, and [[Implicit Learners]] already made the
connection without the citation:

> *Energy minimisation is the total version of root-finding*, and it is total precisely
> because $\sigma \ge 0$ always has an infimum.

That sentence is LeCun's framework in one line. A relation may branch or be empty; an energy
always has an argmin.

## 2. What normalisation costs, and why dropping it is the point

A probabilistic model needs $Z(W) = \int \exp(-E(W,y,X))\,dy$. That integral is the reason
most tractable model families are tractable — conjugacy, exponential families, normalising
flows all exist to keep $Z$ computable.

An EBM refuses to pay. The consequence is a trade the tutorial states plainly:

| | probabilistic | energy-based |
|---|---|---|
| architecture | constrained by tractability of $Z$ | **unconstrained** |
| what you get out | a distribution | a **ranking** |
| composition | products of normalised kernels; renormalise | **add energies**, no renormalisation |
| inference | integrate | minimise |

The third row is the one this project lives on. `Mycelium.combine` adds canonical parameters,
which in the log domain is *adding energies*, and the result is deliberately unnormalised —
[[Acausal Composition is a Hypergraph Category]] §4 shows that this unnormalisedness is
exactly what makes the whole thing a hypergraph category rather than a Markov category.

So the vault has already worked out, from the categorical side, that **Lenticulum is not a
Markov category because it does not normalise.** LeCun's tutorial is the same observation from
the modelling side, twenty years earlier and without the string diagrams.

## 3. Loss functionals — the part Lenticulum does not have

Here is where the tutorial says something the vault does not, and it is the sharpest thing in
this note.

An energy is not a loss. Training an EBM means *shaping the energy surface* so that correct
answers sit lower than incorrect ones, and the tutorial's central contribution is a taxonomy
of **loss functionals** — functions that take the whole energy surface $E(W,\cdot,X^i)$ as an
argument, not merely its value at the correct answer:

| loss | contrastive term? |
|---|---|
| energy loss — $E(W,Y^i,X^i)$ | **none** |
| perceptron loss | $\min_Y E$ |
| generalised margin losses (hinge, log, LVQ2, MCE) | the most offending incorrect answer |
| negative log-likelihood | $-\tfrac1\beta\log\int_Y e^{-\beta E}$ — *all* answers |

And the condition a loss must satisfy: pushing down on the correct answer's energy must be
accompanied by pushing **up** somewhere else, or the model can satisfy the objective by making
the energy surface *flat*.

> [!important] Lenticulum's free energy is the "energy loss" — the one the tutorial warns
> about
> The Bethe free energy ([[Bethe Free Energy]]) is evaluated at the inferred configuration and
> summed. It is $E(W, Y^\ast, X)$ with extra bookkeeping. It contains no term that raises the
> energy of anything else, so it is the **first row** of that table: the loss LeCun singles
> out as collapsing for most architectures.
>
> `LenticulumCore.scalarisation` is `E_c → ℝ` — it takes an energy *value*. A loss functional
> takes an energy *function*. The framework has no slot for one.

## 4. Why it has not collapsed yet

Because every energy in the project is quadratic in its unobserved channel with a **fixed**
metric, and that is precisely LeCun's safe case.

The tutorial notes that the energy loss is safe for architectures that cannot flatten — a
regressor with $E = \|G_W(X)-Y\|^2$ being the canonical example, because the energy is a fixed
quadratic in $Y$ and no choice of $W$ makes it constant. Now look at what is implemented:

| factor | energy | can it flatten? |
|---|---|---|
| `LinearConstraintFactor` | $\tfrac12\|r\|^2_{Q^{-1}}$, `Q` a fixed hyperparameter | no |
| `DEQFactor`, `NeuralODEFactor` | $\tfrac12\|r\|^2$, `SquaredNorm` | no |
| `GaussianFactor` | $\tfrac12\|r\|^2_{Q^{-1}} + \tfrac12\log\det 2\pi Q$ | no — **and see below** |

Every one is LeCun's safe regressor. The safety is an accident of the fixed metric, not a
property of the framework.

### The repo already has a contrastive term and calls it something else

`gaussian.jl` documents its `complexity` summand as:

> `complexity` — $\tfrac12\log\det(2\pi Q)$ — *the log-normaliser; **what stops $Q \to 0$***

That is a contrastive term. Without it, learning $Q$ would drive it to zero, the fit term to
zero, and the energy surface flat — **LeCun's collapse, exactly**. The repo identified the
mechanism and named it after its probabilistic origin (the log-normaliser) rather than its
energy-based function (the term that prevents collapse). Both names are right; only one
generalises.

> [!warning] The deferred `Q`-learning note is only half the reason
> `gaussian.md` §3 records that learning $Q$ is deferred because it "needs a
> positive-definite parametrisation". True, and incomplete. The other half is that
> `DEQFactor` and `LinearConstraintFactor` have **no** $\log\det$ term, so giving either a
> learnable noise scale would collapse it immediately — and nothing in the framework would
> notice, because the free energy would be dutifully decreasing the whole time.
>
> The positive-definiteness problem is a parametrisation detail. The collapse problem is
> structural, and the tutorial is where it is explained.

## 5. Latent variables: minimise or marginalise

The tutorial handles latents two ways:

$$E(W,Y,X) \;=\; \min_Z E(W,Z,Y,X)
\qquad\text{or}\qquad
E(W,Y,X) \;=\; -\tfrac1\beta\log\!\int_Z e^{-\beta E(W,Z,Y,X)}$$

Minimise, or marginalise — and the first is the $\beta\to\infty$ limit of the second.

[[Channels and Polarity]]'s `Latent()` has only ever been read as *marginalise*, and
[[Copiers Cups and Caps]] calls marginalisation "the expensive one". The energy-based option
is to **minimise the latent out**, which is cheap, and it is the right thing whenever the
downstream consumer is going to take an argmin anyway.

`constraint.md` §4.1 records that latent-channel marginalisation is "the single most valuable
missing piece" in that file. Minimising it out is a legitimate alternative that nobody has
considered, and for a Dirac-valued graph it is not even an approximation — see
[[Energy-Based Factor Graphs]] §3.

## 6. What this reframes

The honest summary of where the project sits relative to this paper:

- **The architecture is LeCun's.** Relations, energies, factor graphs, no normalisation,
  inference by minimisation. [[README]]'s table is his table.
- **The theory on top is AutoBayes's.** Beliefs, inversions, free energies, the chain rule.
  That layer is what makes an energy into a *posterior*.
- **The implementation keeps falling back to LeCun's layer**, and has been treating that as a
  defect. It is not. See [[Energy-Based Factor Graphs]] §2, which is the point of this whole
  section of the vault.
- **The training theory is missing entirely.** No loss functionals, no contrastive terms, no
  collapse analysis. [[Training Energy-Based Models]] is about what filling that in would
  look like.

## Sources

- LeCun, Chopra, Hadsell, Ranzato & Huang, *A Tutorial on Energy-Based Learning*, in
  *Predicting Structured Data*, MIT Press 2006 —
  [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf). The framework, the loss
  taxonomy, the collapse conditions, and §6's non-probabilistic factor graphs.

Related: [[Energy-Based Factor Graphs]], [[Training Energy-Based Models]],
[[Implicit Learners]], [[Scalar and Multivariate Energy]], [[Bethe Free Energy]],
[[Channels and Polarity]], [[Acausal Composition is a Hypergraph Category]],
[[Three Senses of Implicit]]
