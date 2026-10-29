#definition #algorithm

> The identity `Mycelium` is built on: a factor → variable message is a Bayesian inversion
> $c'_\pi$ **with the cavity $\pi$ divided out**, i.e. the likelihood site the inversion
> multiplies into $\pi$; and the polarity is determined by which channel is the target.

> Sources: original to this vault (design and analysis; no single paper).
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

## The identity

To compute the message from factor $f$ along the edge attached to channel $c$:

$$
\begin{aligned}
\text{target channel } c &\;\longmapsto\; \texttt{Unobserved()} \\
\text{channels with an incoming message} &\;\longmapsto\; \texttt{Observed()} \\
\text{everything else} &\;\longmapsto\; \texttt{Latent()}
\end{aligned}
$$

then `assemble(factor, polarity)` gives a [[Inversions and Bayesian Lenses]] $(c, c')$, and

$$
c'_\pi(y) \;=\; \pi \odot \mu_{f \to v},
\qquad\text{i.e.}\qquad
\mu_{f \to v}(x) \;\propto\; \frac{c'_\pi(y)(x)}{\pi(x)} \;=\; \ell(y \mid x),
$$

where $\pi$ is the belief at $v$ **excluding** this edge (the cavity), $y$ is the tuple of
incoming beliefs on the observed channels, and $\odot$ is pooling (`combine`). The message is
the likelihood site, not the posterior: `LenticulumCore.likelihood_message` returns it, and
`invert` returns the posterior ([[L-ADR-0006 Separate likelihood_message from invert]]). The
variable pools its sites with its prior once; sending the posterior instead would count $\pi$
once per edge.

A worked case (`GaussianFactor` with $A = 1$, unit noise): cavity $\pi = \mathcal N(0, 1)$,
observation $y = 2$. The outgoing message is $\mathcal N(2, 1)$ and `invert` gives
$\mathcal N(1, \tfrac12)$; pooling the message with $\pi$ gives that posterior back, while
pooling the posterior with $\pi$ would give $\mathcal N(\tfrac23, \tfrac13)$ (LP-019 item 11).

For an approximate inversion $\pi$ is used only as a linearisation, tilting or warm-start point.
A lens that does not override `likelihood_message` sends whatever `invert` returns, so a lens
whose `invert` is a posterior counts $\pi$ again at every edge beyond the first.

Every piece has a name in the paper:

| message-passing | AutoBayes |
|---|---|
| the outgoing message | the inversion $c'_\pi$ with $\pi$ divided out (the likelihood site) |
| the variable's belief after this message | the inversion $c'_\pi(y) = \pi \odot \mu_{f\to v}$ |
| the prior it is conditioned on | $\pi \in \mathcal{P}X$ |
| the observation it is conditioned on | $y \in Y$ |
| the channels not involved | $\llbracket c \rrbracket$ |
| variable → factor message | the pushforward-and-pool of upstream beliefs |

**This is why `Mycelium` needs nothing from `LenticulumCore` except `assemble` and
`likelihood_message`** (which defaults to `invert` for lenses whose inversion is already
extrinsic).
The scheduler decides *which* inversion to compute and *when*; the factor decides *how*.

## Two exclusion principles, and both matter

Belief propagation's correctness rests on not feeding a claim back to its own source. In a
bipartite graph that has to be enforced twice, and **both were caught by the test suite rather
than anticipated** — worth recording, because each produces a plausible-looking wrong answer
rather than an error.

### 1. Variable side

$$\mu_{v\to f} \;=\; \textstyle\bigodot_{g \ne f}\; \mu_{g \to v}$$

The message a variable sends to a factor pools all the *other* factors' claims. Without this,
$f$'s own previous message is returned to it as if it were independent evidence, and the belief
becomes exponentially over-confident with each sweep.

Implemented by recomputing the product over the other edges rather than dividing the full
marginal by the skipped message. **Division requires densities and is numerically fragile;
recomputation costs $O(d_v)$ and is always defined.**

### 2. Factor side

When computing $\mu_{f\to v}$, the incoming message *on that same edge* must not be counted
among the observed channels. Otherwise the target channel resolves as both `Observed()` and
`Unobserved()` — which, in this implementation, throws a `PolarityError`, and that is a lucky
accident: had the polarity type been a `Dict` rather than a `NamedTuple`, one assignment would
have silently overwritten the other and the factor would have been conditioning on its own
prediction.

> [!note] Putting the polarity in the type domain caught a bug
> `Polarity{names}` carries the channel names in its type, so "this channel has two
> polarities" is a construction error rather than a silent overwrite. The performance argument
> for the type-domain representation ([[Channels and Polarity]]) turned out to be the lesser
> reason for it.

## What flows: beliefs, not cotangents

Every message in `Mycelium` is an `AbstractBelief`. The **two directions of belief flow** are

- **forward**: priors, propagated by pushforward $c_*\pi$;
- **backward**: likelihood sites, the inversions $c'_\pi$ with the cavity divided out.

Both are beliefs; that is what makes AutoBayes' framework a message-passing framework at all.

**Cotangents are not messages.** A graph gradient pass governed by each edge's
[`AbstractGradientCoupling`](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients)
would run once per training step, separately from belief sweeps. That pass is not
implemented; non-default coupling annotations fail at construction (LP-011 option 2).
`optimiser_step` is separate from `step!`, but this separation does not implement the
planned gradient pass of [[L-SPEC-002 Stop Gradient]].

This also explains a result that looks wrong at first: on a strictly unidirectional DAG the
**backward belief sweep is empty** ([[Schedules]]). Correct — a Lux graph has no backward
*belief* flow. Its backward pass carries cotangents.

## `combine` is the hard part, and it is deliberately partial

Pooling two beliefs about the same variable is the product of densities. What can be done
honestly with the belief types that currently exist:

| case | result |
|---|---|
| `TrivialBelief` with anything | the other one (it is the unit) |
| two agreeing `DiracBelief`s | that Dirac |
| two **disagreeing** `DiracBelief`s | **throws** — two hard clamps in contradiction is a wiring error |
| `DiracBelief` with anything | the Dirac ($\rho_{in} = \infty$ dominates) |
| two `SampleBelief`s | **throws**, informatively |

Two clouds alone still have no product rule. A particle belief can be pooled with a density
or a likelihood potential by importance weighting through `belief_logdensity`.
`RatioFactor` emits a `LogPotentialBelief`; two such messages add log-potentials, and a
particle prior is reweighted by their product once. Gaussian, categorical, and mixture
products with an arbitrary potential are refused. See [[messages]] and [[ratio]].

> [!note] The polarity-in-the-type observation is a typing result
> [[The Type Discipline of a Factor Graph]] §2 takes the callout above as this vault's own
> empirical argument for static typing: the type domain caught what the value domain would have
> swallowed, and the swallowed version would have been a *plausible wrong answer* rather than an
> error.

Related: [[Polarity Resolution]], [[L-ADR-0006 Separate likelihood_message from invert]], [[LP-019 Defects found in the review of the core packages]], [[Inversions and Bayesian Lenses]], [[Schedules]], [[messages]], [[passing]],
[[The Type Discipline of a Factor Graph]]
