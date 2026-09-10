# Messages are Inversions

> The identity `Mycelium` is built on: a factor → variable message **is** a Bayesian
> inversion $c'_\pi$, and the polarity is determined by which channel is the target.

## The identity

To compute the message from factor $f$ along the edge attached to channel $c$:

$$
\begin{aligned}
\text{target channel } c &\;\longmapsto\; \texttt{Unobserved()} \\
\text{channels with an incoming message} &\;\longmapsto\; \texttt{Observed()} \\
\text{everything else} &\;\longmapsto\; \texttt{Latent()}
\end{aligned}
$$

then `assemble(factor, polarity)` gives a [[Bayesian Lens]] $(c, c')$, and

$$\mu_{f \to v} \;=\; c'_\pi(y)$$

where $\pi$ is the belief at $v$ **excluding** this edge, and $y$ is the tuple of incoming
beliefs on the observed channels.

Every piece has a name in the paper:

| message-passing | AutoBayes |
|---|---|
| the outgoing message | the inversion $c'_\pi$ |
| the prior it is conditioned on | $\pi \in \mathcal{P}X$ |
| the observation it is conditioned on | $y \in Y$ |
| the channels not involved | $\llbracket c \rrbracket$ |
| variable → factor message | the pushforward-and-pool of upstream beliefs |

**This is why `Mycelium` needs nothing from `LenticulumCore` except `assemble` and `invert`.**
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
- **backward**: posteriors, produced by inversions $c'_\pi$.

Both are beliefs; that is what makes AutoBayes' framework a message-passing framework at all.

**Cotangents are not messages.** The gradient of the free energy with respect to a factor's
parameters is accumulated per factor, governed by that edge's
[[Composition of Gradients|`AbstractGradientCoupling`]], on a *different schedule* — once per
training step, not once per inference sweep. Conflating the two is a classic source of silent
bugs, which is why `optimiser_step` is a separate function from `step!`.

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

The last row is the gap. Pooling particle sets needs importance reweighting, which needs a
`belief_logdensity` that no belief type implements yet. This is the same open question
[[open_model]] §4 records: the belief representation is the one abstraction that cannot be
designed before there is a working factor to design it against.

> [!note] The polarity-in-the-type observation is a typing result
> [[The Type Discipline of a Factor Graph]] §2 takes the callout above as this vault's own
> empirical argument for static typing: the type domain caught what the value domain would have
> swallowed, and the swallowed version would have been a *plausible wrong answer* rather than an
> error.

Related: [[Polarity Resolution]], [[Bayesian Lens]], [[Schedules]], [[messages]], [[passing]],
[[The Type Discipline of a Factor Graph]]
