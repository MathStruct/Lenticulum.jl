#implementation

> The discriminator, read as what it is: a **density-ratio estimator**. And the reason this
> file matters to Lenticulum has nothing to do with generating images.

> Sources: code: `ratio.jl`
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## Graph free-energy edge directions

`RatioFactor` declares `Mycelium.uses_cavity == true`: its local free energy reads the
incoming point or particles, forms the factor belief $b_f \propto r\mu$, and averages
the negative log ratio under that belief.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. The identity

Train a classifier to separate $p$ (label 1) from $q$ (label 0) with equal class priors. At
the optimum

$$
D^\ast(x) = \frac{p(x)}{p(x)+q(x)}
\qquad\Longrightarrow\qquad
\operatorname{logit} D^\ast(x) = \log p(x) - \log q(x)
$$

**The logit of the optimal discriminator is the log density ratio.** That is Mohamed &
Lakshminarayanan §3, it is why GANs work, and `logratio` / `discriminator` are the two readings
of the same number — the test suite checks they are logistic transforms of one another against
a closed-form Gaussian pair.

## 2. Why the energy is $-\log r$

$\mathbb{E}_{q}[-\log r] = \mathrm{KL}(q\,\|\,p)$. So the factor's energy is the integrand of a
KL divergence, and minimising it over whatever produced $q$ pushes $q$ toward $p$. That is the
generator's objective, expressed as a factor energy rather than as a training loop.

## 3. `AmortisedInversion` is literally true here

`lens.md` describes `AmortisedInversion(net)` as *"$c'$ realised by a learned network — a VAE
encoder… That a factor has **two independently parametrised halves** is the structural reason
a factor cannot be a Lux layer."*

This is the first factor in the project where that description is exact. The discriminator is
trained separately from whatever produced the samples it scores; the inversion genuinely is a
network with its own parameters. Every other factor's "inversion" is arithmetic on the forward
model.

## 4. The payoff: a route around the `combine` gap

`RatioFactor` emits a `LogPotentialBelief` with log-potential $\log r(x)$.
`Mycelium.combine` pools it with a particle cavity by

$$
w_i \propto w_i^{\text{cavity}} r(x_i).
$$

`reweight` is the explicit posterior operation, also used by `invert`. The likelihood
message consumes neither the cavity nor target-channel inputs. Two ratio messages pool by
adding log-potentials; pooling their product with a particle prior applies each ratio once.
`test/conformance.jl` checks both factor insertion orders on a one-variable graph.

Weights are normalised in log space with a max-shift. Mycelium warns on severe ESS collapse;
callers of `reweight` can inspect `effective_sample_size` themselves. Applying the same
ratio twice squares it, as multiplying the same evidence twice should. This does not
violate associativity or commutativity of products. The previous note confused those laws
with idempotence and incorrectly used that as a reason to keep ratios out of `combine`.

The Gaussian oracle in `test/runtests.jl` draws 20 000 samples from
$q = \mathcal{N}(0,4)$, reweights by the exact ratio against
$p = \mathcal{N}(0.5,1)$, and checks the resulting mean and variance.

## 5. The energy is estimated, and nothing accounts for that

Every other factor in this project computes its energy. This one *fits* it: $\log r$ comes from
a network trained on a finite sample, so the energy carries an estimation error of unknown
sign and size.

`Bayesian Lens.md` licenses inexact **inversions** — *"the quality of the choice is what the
free energy measures."* This is a different animal: an inexact **energy**, and the free energy
has no term for it because the free energy *is* the thing being estimated. You cannot measure
the error of your ruler with the ruler.

Recorded as a genuine gap in the framework's accounting, not merely in this file. A candidate
fix is the discriminator's own validation loss as a confidence term, which nothing computes.

## 6. Implementation difficulties

Particle matrices use one column per sample, as in Mycelium's belief algebra. Reweighting,
moments, ESS and generator pushforward share that convention with vector-of-vectors input;
the tests compare both representations. Zero prior weights stay zero under importance
reweighting: their log weights are `-Inf`, not a positive epsilon floor.

### 6.1 Likelihood messages and posterior inversion

The former particle message included its cavity (LP-016). The outgoing message now carries
only a log-potential; `likelihood_message` and `factor_message` do not read the cavity or
any target-channel input. `invert` reweights its prior and preserves Dirac clamps.

The callable captures the supplied parameters and network state. Message construction does
not evaluate the network. Each density evaluation starts with that captured state and drops
the returned state; use inference-mode state for deterministic scoring. Captured mutable
parameters and state must not be mutated while a message is in use. Callable identity is
only a conservative convergence check, not a numerical comparison of potentials.

### 6.2 Parametric products need more structure

`invert` supports `SampleBelief`, Dirac clamps, and absent/trivial priors (the latter retain
an uninformative posterior placeholder). Parametric beliefs throw. Mycelium likewise refuses
Gaussian, categorical, and mixture products with an arbitrary log-potential rather than
silently sampling or projecting. Even categorical labels need an explicit interpretation as
points; this implementation does not assume one.

`AbstractPotentialBelief` leaves room for richer likelihood types carrying gradients,
parametric structure, or batched evaluation and their own exact product rules.

### 6.3 No training, so `logratio` is only as good as what you hand it

This package scores; it does not fit. An untrained `net` gives a meaningless ratio and
everything downstream is confidently wrong, with no diagnostic. The Gaussian oracle in the
tests exists precisely because a *perfectly trained* discriminator is the only one whose
output can be checked.

Related: [[generator]], [[Adversarial]], [[Implicit Generative Models]],
[[GANs as Two Factors]], [[messages]]

## Graph energy and game quantities (LP-016)

`messages_into` in Mycelium supplies incoming variable→factor messages $\mu$, not variable
marginals. For the unary ratio factor the factor belief is $b_f \propto r\mu$, so
`local_free_energy` returns $\mathbb{E}_{b_f}[-\log r]$. It reweights the incoming particles
before taking that expectation. This is the Bethe **energy contribution** only: particle
clouds have no Lebesgue density, and no differential entropy is subtracted (LP-006).
It is not a complete continuous Bethe free energy or generally $\mathrm{KL}(q\|p)$.
An absent/trivial cavity retains the zero contribution; unsupported cavity types throw.

For a Dirac prior the point remains fixed. The game methods retain energy `-logratio` and
the zero point-entropy convention. Particle game methods still throw `ArgumentError`:
no reference measure for their entropy has been chosen.

`test/conformance.jl` checks $p=N(0,1)$, $q=N(1,1)$, hence $\log r=1/2-x$.
On equally weighted particles $\{0,1\}$ the posterior weights are
$\{e/(1+e),1/(1+e)\}$ and the energy contribution is $1/(1+e)-1/2$.
The outgoing potential, target exclusion, posterior inversion, and graph marginal with two
ratio factors are tested separately. The three former LP-016 broken assertions now pass.
