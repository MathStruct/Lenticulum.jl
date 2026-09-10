# Everything is a Factor

> Data, priors, losses and optimisers are all nodes in the graph. This is not a stylistic
> choice — it is forced by AutoBayes Remark 24.

## The argument from Remark 24

[[Statistical Game|Definition 20]]'s loss for a pure game $c$ with energy $-\log p_c(y|x)$
and Shannon entropy is

$$F^c(\pi,y) = \mathbb{E}_{x\sim c'_\pi(y)}\bigl[-\log p_c(y\mid x)\bigr] - H\bigl(c'_\pi(y)\bigr)$$

which is **not** the variational free energy of
[[Variational Free Energy|Proposition 18]] — it is missing $-\log p_\pi(x)$. The paper's own
fix is to make the prior a factor:

> Given a prior distribution $\pi : 1 \nrightarrow\!\!\!\bullet\; X$, we can turn it into a statistical game
> by first equipping it with a trivial inversion, then setting $l^\pi(x) = -\log p_\pi(x)$,
> and finally noting that $H^\pi(\cdot, x)$ must be $0$. Composing $c : X \multimap Y$ after
> the resulting $\pi : 1 \multimap X$ yields a composite with
> $F^{c\pi}(\ast, y) = \mathrm{VFE}(c,c')(\pi,y)$.

**The prior is not part of the model. It is a separate node you compose with.** Once that is
true of priors it is true of everything else that contributes a term to the objective.

## The four structural factors

| node | arity | edges | learnable? | energy | entropy |
|---|---|---|---|---|---|
| `DataFactor(ch, v)` | 1 | `Emitting` | no | $0$ | $0$ |
| `PriorFactor(ch, b, nlogp)` | 1 | `Emitting` | optional | $-\log p_\pi(x)$ | $0$ |
| `LossFactor(chs, f)` | $n$ | `Absorbing` | no | $f(\ldots)$ | $0$ |
| `OptimiserFactor(ch, rule)` | 1 | `Bidirectional` | no (holds *state*) | $0$ | $0$ |
| `RelayFactor(a, b)` | 2 | `Bidirectional` | no | $0$ | $0$ |

### Data is a factor, not a variable

[[README]] says "our training data are fixed variables in our factor graph". The refinement
that falls out of Remark 24 is: **data is a `DataFactor` of arity one attached to a variable.**

A variable is a wire and has no content; data is *evidence*, and evidence contributes to the
objective. A `DataFactor` is precisely the categorical [[Copiers Cups and Caps|cup]] and the
$\rho_{in} = \infty$ hard clamp of [[Channels and Polarity]] — and because it is a factor, it
can be softened (a finite precision, a noisy observation) without changing the graph's
structure, only the node's type.

### Losses are sinks

A `LossFactor` supports **zero** polarities: it absorbs on every channel and emits nothing. It
is a third case beyond unidirectional (one polarity) and bidirectional (two or more), and it
corresponds exactly to Cruttwell et al.'s **learning-rate cap** — the lens $(L,L')\to(1,1)$
that terminates a wire ([[Learning Components as Parametric Lenses]] §3.3).

Note also that Definition 3.3 makes the *label* the loss map's parameter. In a graph the label
is just another absorbed channel, and the graph does the bookkeeping. Same statement, less
machinery.

> A loss factor never tells its input variable anything **in belief terms**. What a loss
> "tells" its input is a *cotangent*, and cotangents travel by the
> `AbstractGradientCoupling` on each edge, not by the message scheduler. This is why the tree
> schedule prunes messages at loss nodes — see [[Schedules]] §"Pruning".

### Optimisers are reparametrisations, and a wire is a variable

Cruttwell et al. present an optimiser as a **reparametrisation** — a box sitting *above* the
parameter wire, a lens $(S\times P, S\times P) \to (P, P')$ ([[Learning Components as Parametric Lenses]] §3.4).

In a factor graph a wire is a variable, so a box above it is a **factor**. The optimiser
becomes an ordinary node, bidirectional by nature: it *emits* the current parameter
(`rule_get` $= U$) and *absorbs* the update (`rule_put` $= U^*$).

This only works because a factor may **expose its parameters as channels** instead of hiding
them in `ps`. That is exactly the move AutoBayes Example 3 (VBEM) makes when $\Theta$
"migrates from the parameter space into the wire" ([[Examples from the Paper]]).

> [!note] Parameter exposure is a graph-level decision, not a factor-level one
> The same factor can be used with hidden parameters (Lux style, `ps` threaded through) or
> exposed ones (a variable, with an optimiser or a hyperprior attached). Hidden gives you
> maximum-likelihood; exposed gives you a posterior over weights — i.e. Bayesian deep learning
> ([[Examples from the Paper|Example 5]]). **The difference between the two is one edge.**

The rules implemented are Cruttwell's table verbatim:

| rule | `get` $U(s,p)$ | `put` $U^*(s,p,\bar p)$ |
|---|---|---|
| `GradientDescent(η)` | $p$ | $(s,\; p - \eta\bar p)$ |
| `Momentum(η,γ)` | $p$ | $s' = -\gamma s - \eta\bar p$; $(s',\; p + s')$ |
| `Nesterov(η,γ)` | $p + \gamma s$ | as momentum |

Nesterov is the one with a **non-trivial `get`**, and it is the reason optimisers must be
lenses rather than functions. "Evaluate the gradient at the look-ahead point" *is* "the
forward part of the optimiser lens is not the identity". Asserted in the test suite, because
it is the claim most worth checking.

## What this buys

The entire Cruttwell et al. pipeline — model, loss, learning rate, optimiser
([[Learning Components as Parametric Lenses]] Figure 4) — becomes **one graph** with no
privileged machinery. A training loop is a schedule over it. There is no separate "optimiser
API", no separate "loss API", no separate "data loader": there are factors, and there is a
message schedule.

That is also the answer to [[README]]'s note that Lenticulum "cannot rely on traditional data
handling like done with MLUtils.jl". Batching is not a data-loader concern; it is a question of
which `DataFactor`s are attached to the graph this step.

> [!note] The optimiser-as-factor is half of a larger idea
> Making the optimiser a node in the graph is the per-factor case of attaching a *graph* to the
> graph — one whose variables are the base graph's parameters. Its counterpart, a graph over
> the base graph's **messages**, does not exist, and `LenticulumCore.AmortisedInversion` is its
> per-factor case. See [[The Inferencer and the Optimizer]].

Related: [[The Inferencer and the Optimizer]], [[Factor Graphs]], [[Learning Components as Parametric Lenses]], [[Examples from the Paper]], [[factors]]
