# Motivating Examples

> Six problem domains that have the same shape, and why that shape is what this project is
> built for.
>
> These are **motivating sketches, not case studies.** Nothing here has been implemented; the
> library is a prototype ([[Related Julia Projects]] §10 is the honest state of it). The point
> is to show what kind of problem the machinery is aimed at.

| domain | the relations are | what is learned |
|---|---|---|
| [[SLAM and Sensor Fusion]] | geometry: poses, landmarks, motion | visual front-ends, depth priors |
| [[Trading and Financial Markets]] | no-arbitrage: parity, triangular, index–constituent | volatility surfaces, illiquid instruments |
| [[Energy Markets and Power Grids]] | Kirchhoff + market clearing | renewable and demand forecasts |
| [[Metabolomics and Proteomics]] | stoichiometry: ``Sv = 0`` | enzyme kinetics, regulation |
| [[Molecular Dynamics]] | force fields, bond constraints | ML potentials |
| [[Climate and Dynamical Systems]] | conservation laws, discretised PDEs | subgrid parametrisations |

## The shape they share

### 1. A network of relations, not a pipeline

In every one of these, the model is a set of **constraints among quantities** with no natural
input and output. Kirchhoff's law does not say "current causes voltage"; put–call parity does
not say which of the four prices is the answer; ``Sv = 0`` does not designate an output flux.

A neural network is a pipeline and needs one. A factor graph does not — which is the whole
content of [[Implicit Learners]] and the reason a `Polarity` is chosen per call rather than
baked in at construction.

### 2. Observations are sparse, heterogeneous and asynchronous

Nothing in these domains is fully observed. A grid has meters on some buses; a metabolic
network has concentrations for some metabolites; a trading venue quotes some instruments and
not others, at times that do not line up.

Partial observation is the *normal* case, and it is exactly what `Observed` / `Unobserved` /
`Latent` encodes. Asynchrony is what [[Time as a Base]] is for.

### 3. Some of the model is known and some must be fitted

This is the one that rules out both alternatives. Pure physics tools (ModelingToolkit, FBA
solvers, MD engines) cannot express a fitted component. Pure ML tools cannot express a
conservation law that must hold exactly.

Every domain below is **grey-box**: a stoichiometry you trust beside kinetics you do not; an
exact power-flow equation beside a wind forecast; a geometric constraint beside a learned
visual odometry front-end. A factor graph does not care which kind a node is, which is the
point of the factor interface.

### 4. The decisions are made under uncertainty

Dispatch a generator, trade a spread, refine a structure, publish a projection. Point
estimates are not enough and everyone in these fields knows it — which is why each has grown
its own uncertainty machinery (EnKF, bootstrapped SLAM covariances, ensemble forecasts).

### 5. One model, several questions

Each domain asks its model more than one thing, and they are the same relations with different
quantities held fixed:

| domain | one question | the other question |
|---|---|---|
| SLAM | given the map, where am I? | given my pose, what is the map? |
| grids | state estimation | contingency analysis |
| metabolism | fluxes from concentrations | concentrations from fluxes |
| climate | assimilation | prediction |

That is [[Channels and Polarity]] doing the work it exists for. **SLAM has the two directions
in its own name.**

### 6. The residual means something

This is the observation that ties the six together and it is easy to miss.

In ordinary machine learning a residual is an error — something to minimise and then forget. In
every domain below, **"how badly is this relation violated" is itself a quantity of interest**:

| domain | a nonzero residual is |
|---|---|
| SLAM | a loop-closure inconsistency; a bad data association |
| trading | **a mispricing** — the trade signal itself |
| power grids | **bad data**, a failed sensor, or an undetected topology change |
| metabolism | an unmodelled reaction or a measurement error |
| MD / structure | strain; incompatibility between experiment and force field |
| climate | model misfit; where the parametrisation is failing |

So the graded energy of [[Scalar and Multivariate Energy]] is not bookkeeping. It is
**per-relation attribution of disagreement**, and in at least two of these domains it is the
output rather than a diagnostic.

## What would have to be true

Being honest about the gap between the shape and the software:

- **Scale.** Climate and MD are ``10^6``–``10^9`` variables. Metabolic networks and grids are
  ``10^3``–``10^5``. SLAM and trading graphs are the ones the current implementation could
  plausibly reach. See [[Parallelism and Compilation]].
- **Loops.** Meshed grids, metabolic cycles and SLAM loop closures are all loopy, and loopy
  message passing gets exact means with wrong variances ([[Loopy Message Passing]]). For
  domains where the *uncertainty* is the product, that is the blocking problem.
- **Non-Gaussian everything.** Financial returns are heavy-tailed; molecular configurations are
  multimodal; flux distributions are constrained to a polytope. The Gaussian fragment is where
  the exactness results live, and none of these are in it — which is
  [[messages]] §1's gap in applied clothing.
- **Nonlinearity.** AC power flow, enzyme kinetics, atmospheric dynamics. The exact results are
  linear-Gaussian; everything else is approximate.

None of that makes the shape wrong. It makes the six notes below a description of where the
work would pay off, not a claim that it already has.

Related: [[Implicit Learners]], [[Channels and Polarity]], [[Time as a Base]],
[[Scalar and Multivariate Energy]], [[Related Julia Projects]],
[[Parallelism and Compilation]], [[The Linear Gaussian Chain]]
