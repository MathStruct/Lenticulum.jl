#design #open-problem

> Which operations can be done with beliefs, and where each one lives. The organising
> principle: **an operation between variables is a factor, and the operation on beliefs is
> that factor's message**, one per polarity. Today only pooling (`combine`) exists, and only
> partially. This note lists the rest, what each needs, the order in which to build them, and
> where they should live (§6).

> Sources: Loeliger, *An introduction to factor graphs*, IEEE Signal Processing Magazine 2004; Loeliger, Dauwels, Hu, Korl, Ping & Kschischang, *The factor graph approach to model-based signal processing*, Proc. IEEE 2007 (message tables for equality, addition and matrix nodes); Minka, *Expectation Propagation for approximate Bayesian inference*, UAI 2001; Fritz, *A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics*, Adv. Math. 2020 (Markov categories); code: `Mycelium/messages.jl` (`combine`), `LenticulumCore/open_model.jl` (`pushforward`)
>
> Bibliography: [[Bibliography#^loeliger2004intro|Loeliger 2004]] · [[Bibliography#^loeliger2007factor|Loeliger et al. 2007]] · [[Bibliography#^minka2001ep|Minka 2001]] · [[Bibliography#^fritz2020synthetic|Fritz 2020]]
>
> Theory (CT-ML wiki): [Markov Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Markov-Category) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)

## 1. Operations are factors

In a Forney-style factor graph a variable is a wire and everything else is a factor
([[Everything is a Factor]]). So "add two beliefs", "mix two beliefs" or "x implies y" is not
an operation on the graph's variables; it is a factor connecting them, and what happens to the
*beliefs* is the message that factor sends.

A factor sends a different message for each choice of output, which is Lenticulum's polarity
([[Channels and Polarity]]). Take the addition factor $z = x + y$:

| polarity | message | operation on beliefs |
|---|---|---|
| $x, y \to z$ | $p_z = p_x * p_y$ | convolution |
| $z, y \to x$ | $p_x(x) = \int p_z(x + y)\,p_y(y)\,dy$ | deconvolution (correlation) |
| $z, x \to y$ | the same, with $x$ and $y$ swapped | |

The "operation" is one relation; the three operations on beliefs are its three readings.
That is the same move the implicit learners make for learned relations, applied to the
building blocks.

## 2. The catalogue

| operation | as a factor | on beliefs | Gaussian | Dirac | Sample | status |
|---|---|---|---|---|---|---|
| **identification** $x = y$ | equality node; the copy map of a Markov category | product of the incoming beliefs | exact | absorbing | needs densities | implicit: a variable node *is* an equality node |
| **fusion** (pooling) | the variable node | product of densities, normalised | add $(\eta, \Lambda)$ | absorbing | importance reweighting | `combine`: Gaussian, Dirac, Trivial, categorical; samples reweighted by any density; mixtures componentwise ([[Beliefs]]) |
| **addition**, linear maps $z = Ax + By$ | linear factor | convolution / deconvolution | exact | shift | pairwise sums (forward) | Gaussian through the linear factors only |
| **deterministic function** $z = f(x)$ | function factor | forward: pushforward; backward: inversion | linearised or unscented | exact | exact forward | forward `pushforward`; backward is the implicit learners' job |
| **mixture** | factor with a categorical switch $s$, $p(x \mid s)$ | weighted sum of components | a Gaussian mixture | a weighted point set | union with weights | `MixtureBelief` ([[Mixture Belief]]): density, products, `reduce_mixture`; the switch factor itself is not built |
| **marginalisation** | delete a wire (the delete map) | integrate out | drop blocks | drop coordinates | drop coordinates | Gaussian only, inside the linear-Gaussian code |
| **conditioning** | clamp | restrict and renormalise | Schur complement | | reweight | Gaussian only |
| **logic**: implication, AND, OR, XOR | factor on Bernoulli variables | discrete sum-product | | | | beliefs exist ([[Categorical Belief]]); the logic factors are not built |
| **projection onto a family** | moment matching | the nearest Gaussian in $\mathrm{KL}(p \Vert q)$ | identity | degenerate | sample moments | `moment_match` for samples and Gaussian mixtures; an EP outcome factor is not built |
| **tempering** $p^\alpha$, **division** $p/q$ | power EP, cavity distributions, counting numbers | scale / subtract natural parameters | exact | | | missing as operations; `counting_number` in `Mycelium` already assumes them |
| **composition with a kernel** | transition factor (a random walk in time) | Chapman–Kolmogorov | exact for linear-Gaussian | | propagate particles | Gaussian through the linear factors |
| **max instead of sum** | the semiring | max-product: MAP rather than marginals | same algebra, different meaning | natural | | not a separate mode; point inference already behaves this way |

## 3. The categorical reading

Copy, delete, tensor and composition are the structure of a **Markov category**: they exist
for every belief type that is a probability measure, and they compose freely. Mixture is the
convex structure of the distribution monad. **Fusion is different**: the product of two
densities is not a morphism of the Markov category but **conditioning**, a Bayesian inversion
([[Inversions and Bayesian Lenses]]). That is why it is the one operation that is partial,
needs densities, and can fail (two contradictory Diracs). Everything else in the catalogue is
easy in principle; fusion and projection are where the approximations live.

## 4. What to build, by what it unlocks

Status: items 1–3 are **built** (categorical and mixture beliefs, densities, sample
reweighting, mixture products and reduction, `moment_match`); item 4 and the relocation of
§6 are not. What is still missing from item 1 is an actual EP *factor* (e.g. a probit
win/draw/loss likelihood) that uses `moment_match`.

1. **`belief_logdensity` beyond Gaussians, and moment projection (EP).** Pooling of Sample
   beliefs (densities exist for Gaussians only, [[Beliefs]]), and the first
   non-Gaussian factor whose messages stay Gaussian: a win/draw/loss outcome factor (the
   planned TrueSkill-through-time project), probit and other one-dimensional likelihoods.
2. **A mixture belief.** The important one for the implicit learners: today point inference
   returns *one* branch of a multivalued relation ([[Inference Signatures]] §3). A mixture
   holds both, e.g. the circle's $y = \pm 0.8$ or the robot arm's elbow up and down. The cost is
   combinatorial growth under products, so it comes with merging and pruning, as in
   Gaussian-sum filters.
3. **Bernoulli and categorical beliefs.** Logic factors, and switch variables: "is this enzyme
   active?" ([[Metabolomics and Proteomics]]) is a Bernoulli latent that turns a reaction on or
   off, and the mixture factor's switch $s$ is a categorical.
4. **Addition and linear maps for Sample and Dirac beliefs.** Bookkeeping, but it lets the
   particle path compose with the linear factors.

Each item is a belief type or a message rule, not a change to the graph machinery: `combine`
dispatches on belief types, and a factor's messages are its inversions
([[Messages are Inversions]]), so the catalogue extends by adding methods.

## 5. The nearest existing implementation

RxInfer.jl (and ForneyLab.jl before it) implements exactly such message rules, per node type
and distribution family, including mixtures, addition and equality nodes
([[Related Julia Projects]] §5). Its rules are the reference to check against; what this
project adds is that the factors themselves can be learned relations queried in any polarity.
Using it as the engine, and what that would gain and lose, is in [[RxInfer as a Backend]].

## 6. Interop and package layout (for later)

**Where the beliefs live is the real constraint.** `TrivialBelief`, `DiracBelief` and
`SampleBelief` are in `LenticulumCore`; `GaussianBelief` is in the top-level `Lenticulum`
package, *above* the `lib/` packages. So the diffusion and equilibrium factors cannot return a
Gaussian message even where they could compute one, e.g. the Laplace approximation from the
implicit adjoint ([[Gaussian Belief]], [[Beliefs]]). `Mycelium` holds the graph, `combine` and
the schedules; nothing in the graph machinery needs heavy dependencies.

**The plan, when this is taken up:**

1. **One low layer for all beliefs and their algebra**: the belief types, `combine`,
   densities, and later mixtures and projection. Either `LenticulumCore` itself, or a small
   dedicated beliefs package if `LenticulumCore` should stay pure interfaces. Every factor
   package can then produce and consume every belief type.
2. **Connectors as package extensions** on that layer, not as a second graph package:

   | connector (Distributions.jl extension) | what it enables |
   |---|---|
   | `SampleBelief(rng, d::Distribution, n)` | any distribution as a particle belief |
   | `GaussianBelief(::MvNormal)`, and back to `MvNormalCanon` | interop with the ecosystem |
   | belief densities through `logpdf` | densities beyond Gaussians, hence pooling of particle beliefs by importance reweighting (§4, item 1) |
   | `fit(Family, ::SampleBelief)` | infer a distribution from particles: projection onto a family, the EP step |

   A later ExponentialFamily.jl / BayesBase.jl extension could lend `combine` their
   closed-form products (`prod` with `ClosedProd`), the most developed product rules in Julia.
3. **Split `Mycelium` into a core and a full package only if** the graph machinery itself
   acquires heavy dependencies. Today it does not, and extensions cover the connectors.

Other Julia belief representations worth connecting or checking against: MonteCarloMeasurements.jl
(particles with arithmetic, so addition is convolution), ParticleFilters.jl, KernelDensity.jl,
MeasureTheory.jl (densities relative to base measures), AbstractGPs.jl (beliefs over functions),
Bijectors.jl (pushforward beliefs), and IncrementalInference.jl with ApproxManifoldProducts.jl
(multimodal, kernel-density beliefs on manifolds: the existing Julia take on non-Gaussian SLAM).

Related: [[Beliefs]], [[Gaussian Belief]], [[Dirac Belief]], [[Sample Belief]], [[Trivial Belief]],
[[Messages are Inversions]], [[Everything is a Factor]], [[Inference Signatures]],
[[Related Julia Projects]]
