#design #overview

> **What is the smallest stochastic factor from which an expressive generative model can be built
> by composition alone, with no neural network inside any factor?** Ten experiments on the branch
> `experiment-qm` answer it in four parts:
>
> - a linear constraint whose right-hand side is chosen by a discrete outcome (the
>   [[Switching Constraint Factors|switching constraint factor]]) is the building block;
> - composed, these factors give exactly the finite Gaussian mixtures, and composition pays only
>   when the wiring matches the data;
> - a chain of them trained as a diffusion is held back by its objective, not by its model class;
> - learning through the graph needs exact blocks of switches *and* mean-field iteration between
>   them.
>
> The factor and the message rule are now in the package. The experiments stay on the branch.

> Sources: original to this vault (summary of the branch `experiment-qm`: the design note `vault/Design/Minimal Stochastic Factors.md` with every proof, the overview `Minimal Stochastic Factors Overview.md`, and the experiments in `examples/`, each with a `.md` of its measurements and a test). Diffusion evidence bound: Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli, *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML 2015, App. B.4; Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020, §2, and §3.2–3.4 for the reverse kernel and the simplified loss of the neural reference. Factorial models: Ghahramani & Jordan, *Factorial Hidden Markov Models*, Machine Learning 29, 1997. The bias of factorised variational EM: Turner & Sahani, *Two problems with variational expectation maximisation for time series models*, in *Bayesian Time Series Models*, CUP 2011. Expectation propagation: Minka, UAI 2001. Linear response: Welling & Teh, Neural Comput. 16(1), 2004. Hybrid belief propagation and mean field: Riegler, Kirkelund, Manchon, Badiu & Fleury, IEEE Trans. Inf. Theory 59(1), 2013. Born machines: Han et al., *Unsupervised Generative Modeling Using Matrix Product States*, Phys. Rev. X 8, 2018. Squared mixtures: Loconte et al., arXiv:2310.00724. Hierarchical mixtures of experts: Jordan & Jacobs, Neural Comput. 6(2), 1994. Sum-product networks: Poon & Domingos, UAI 2011. Mixture rates and misspecification: Chen, Ann. Statist. 23(1), 1995; White, Econometrica 50(1), 1982. code: `lib/Lenticulum.jl/src/switching.jl`, `lib/Mycelium.jl/src/passing.jl`; experiments on the branch: [`examples/`](https://github.com/MathStruct/Lenticulum.jl/tree/experiment-qm/examples)
>
> Bibliography: [[Bibliography#^sohldickstein2015deep|Sohl-Dickstein et al. 2015]] · [[Bibliography#^ho2020ddpm|Ho et al. 2020]] · [[Bibliography#^ghahramani1997factorial|Ghahramani & Jordan 1997]] · [[Bibliography#^turner2011two|Turner & Sahani 2011]] · [[Bibliography#^minka2001ep|Minka 2001]] · [[Bibliography#^welling2004linear|Welling & Teh 2004]] · [[Bibliography#^riegler2013merging|Riegler et al. 2013]] · [[Bibliography#^han2018unsupervised|Han et al. 2018]] · [[Bibliography#^loconte2023subtractive|Loconte et al. 2023]] · [[Bibliography#^jordan1994hme|Jordan & Jacobs 1994]] · [[Bibliography#^poon2011spn|Poon & Domingos 2011]] · [[Bibliography#^chen1995optimal|Chen 1995]] · [[Bibliography#^white1982misspecified|White 1982]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion)

Every number below comes from a script on the branch; its file is named next to it, under
`https://github.com/MathStruct/Lenticulum.jl/tree/experiment-qm/examples/`. Each script has a test
there, and a `.md` beside it with the full tables, seeds and caveats.

## 1. The question

The project's implicit learners put a network inside a factor (a denoiser, an equilibrium layer).
This line of work asked the opposite question: keep every factor mathematically simple, linear or
nearly so, and let the **factor graph** carry the complexity. A factor should sample, support several
polarities (any variable may be the one inferred), be a statistical game with a free energy
([[Scalar and Multivariate Energy]]), learn by gradients, and compose under message passing and the
Bethe free energy.

The starting idea was quantum-flavoured: a linear factor whose output is an eigenvector of a
learned density matrix, drawn by its eigenvalue, as a one-shot alternative to iterative denoising.

## 2. Theory

### The building block

Linear-Gaussian factors never produce a second mode, however they are wired
([[Switching Constraint Factors]], Proposition 1). The smallest addition that does, while keeping
every message, every factor belief and every local free energy in closed form, is a discrete outcome
selecting the right-hand side of a linear constraint. A closure lemma makes "smallest" exact: a factor
with a discrete channel keeps messages in the finite conditional-Gaussian family exactly when each
outcome's potential is a sum of Gaussian sites. The resulting **switching constraint factor**
$w_k \mathcal N(\sum_i A_i x_i - b_k; 0, Q)$ has:

- a measurement-like message to the outcome, the evidence of each outcome;
- a mixture message to every continuous channel, with the per-outcome constants kept;
- an exact factor entropy, $H(k) + H(x \mid k)$.

It is the conditional-Gaussian model of Lauritzen & Wermuth, made into a factor with polarities.

### The quantum reading, done properly

The spectral construction has **no joint distribution** behind it. Its outcome weights treat a
partial input as a state, so its polarities are not compatible (inverting one query does not give
another), and its modes are forced to be orthogonal. Made into a joint, a measurement is a **Born
joint** $p(k, z) \propto \mathcal N(z)\, \tilde z^\top E_k \tilde z$ with a POVM $\{E_k\}$. Three things were proved about it:

- **Its marginal does not depend on the POVM:** $p(z) \propto \mathcal N(z) \lVert \tilde z \rVert^2$. With weighted outcomes it
  becomes one quadratic tilt $\mathcal N(z)\, \tilde z^\top M \tilde z$.
- **What it adds is in queries about the outcome:** $p(z \mid k)$ can vanish exactly on a hyperplane,
  which no finite mixture can.
- **Density needs amplitudes:** squared polynomials times a Gaussian are dense in $L^1$, but that
  needs amplitudes added *before* squaring, as in a Born machine. A factor graph multiplies
  probabilities. Its products of positive semidefinite quadratics are dense in one dimension, but
  cannot vanish on a circle in two.

So interference, the genuinely quantum ingredient, needs amplitude-level composition, which a
probabilistic factor graph does not have. The switching factor is the building block; the Born joint
is a possible later belief type for outcome queries.

### Composition

A graph of switching factors with $L$ outcomes, linear-Gaussian factors and Gaussian priors has a
finite Gaussian mixture as its joint, with at most $\prod_l K_l$ components ([[Switching Constraint Factors]],
Proposition 4). **Composition buys parameter efficiency, not a larger class.** $L$ binary switches feeding
a sum node, $z = c + \sum_l k_l \delta_l + \varepsilon$, give a $2^L$-component grid from $O(L)$ parameters: a
factorial model (Ghahramani & Jordan 1997), here one time slice. A switch over such sums is a small
sum-product network (Poon & Domingos 2011).

### The diffusion reading

A diffusion is a chain of reverse kernels. Three results place the switching factor in it:

- **For mixture data the exact reverse kernels are switching kernels.** If the data are a
  $K$-component Gaussian mixture, every true reverse conditional of the variance-preserving forward
  process is a switching linear kernel with $K$ outcomes. Its gate is the measurement of the factor,
  and its parameters follow from the noise schedule, so a shared, time-conditioned parameterisation
  comes for free (proved on the branch, design note Proposition 8).
- **The bound decouples.** With a fixed forward process $q$, the evidence bound of any reverse chain
  $p$ is
  $\mathbb E \log q_0 - \mathrm{KL}(q_T \| p_T) - \sum_t \mathbb E_{q_t} \mathrm{KL}\bigl(q(\cdot \mid z_t) \,\|\, p_t(\cdot \mid z_t)\bigr)$: the
  unconditioned variant of Sohl-Dickstein et al. (2015, App. B.4) and Ho et al. (2020, §2). Maximising
  it fits each step to the *data's own* noised reverse conditional, independently. It rewards a chain
  for following the forward process's marginals, not for reaching $q_0$ some other way.
- **The class is larger than the bound lets it be.** A chain of $L$ two-outcome kernels with constant
  gates and offset steps has *exactly* the additive model's $2^L$-component marginal (design note
  Proposition 10). So chains of few-outcome kernels can represent product grids, and any limit found
  under the bound is the objective's, not the class's.

## 3. What was measured

### The building block: E1 and the spectral baseline

On two noisy lines through the origin, 60° apart (`switching_factor_e1.md`, `measurement_factor.md`):

| model | held-out $\log p(y \mid x)$ |
|---|---|
| true model | 0.702 |
| switching factor, general form (6 parameters) | 0.702, lines within 0.015° |
| free two-component Gaussian mixture (11 parameters) | 0.702 |
| one Gaussian | −0.823 |
| spectral measurement | −0.939, a line 30° off |

On orthogonal lines the spectral measurement is exact (0.688 against 0.691), which is the only case
its construction allows. Message passing through a Mycelium graph reproduces the switching factor's
direct conditional to $5 \cdot 10^{-15}$.

### Composition: E3, E3b and structure learning

Two-dimensional data, fitted by exact maximum likelihood (`composition_e3.md`, `hierarchy_e3b.md`,
`structure_switches.md`):

| data | model | gap to the truth (nats) |
|---|---|---|
| 8- and 16-mode product grids, $n = 2000$ | additive, 14–17 parameters | 0.003, 0.005 |
| the same, $n = 150$ | additive | within 0.04 |
| the same, $n = 150$ | best flat mixture | 0.36, 0.48 |
| 8 modes on a ring | additive | 0.89 (worse than a flat 4-mixture) |
| any of these | rank-one constraints on one shared variable | 0.6–0.8 |
| 3 clusters of rotated 2 × 2 grids, $n = 2000$ / 150 | switch over sums, 29 parameters | 0.008 / 0.091 |
| the same | flat 12-mixture, 71 parameters | 0.016 / 0.996 |

- **Composition is a prior on how components are arranged.** It pays when the prior is right, and
  then most at small sample sizes.
- **Wiring matters.** Constraints on one variable *intersect*: with more switches than dimensions
  some configurations cannot be satisfied. With as many switches as dimensions (a 4-mode grid,
  $L = n = 2$) they match the additive model (0.004 against 0.005).
- **Exact EM trains these models.** The M-step is closed form, a weighted least squares. 256 EM
  starts are 4–5 times faster than 16 Adam starts at the same optima.
- **Greedy structure search works on these data.** Four moves (add or remove a switch, split or
  merge a cluster), scored by BIC under exact EM, recover the generating structure on the 8-mode grid
  and a tree, and find a more compact exact structure on the ring. Every greedy model is within 0.005
  nats of the truth.

### The diffusion reading: E4(a), the small-step test, E4(b), the objective control

- **E4(a)** (`switching_reverse_chain_e4a.md`): on a three-component mixture, the exact kernels make
  every path's bound equal $\log q_0$ to $1.6 \cdot 10^{-14}$. The shared, time-conditioned chain (17
  parameters) reaches −2.123 against the truth's −2.108. Linear-Gaussian steps fail at any step size
  (−3.58).
- **Small steps** (`switching_small_steps.md`): with each step fitted on its own, three outcomes per
  step (as many as components) match the truth up to the estimation cost of fitting $T$ kernels. Two
  outcomes keep a gap of 0.50–0.68 nats however small the steps. Small steps make each reverse
  conditional unimodal, but its mean, the score field, still needs one linear piece per component.
- **E4(b)** (`switching_chain_e4b.md`): on two moons and a ring, which no finite mixture contains, a
  chain of $T \le 20$ steps with $K$ outcomes each **scores like a flat $K$-component mixture** (within
  0.006 nats at $T = 5$). The best chain is 0.31–0.34 nats from the truth with up to 1 085 parameters; a
  flat 16-mixture is within 0.02 with 95.
- **The objective control** (`chain_objective.md`): on a 16-mode grid, the additive model rewritten as
  a 4-step chain has the same marginal ($7 \cdot 10^{-15}$). Within one family (constant-gate chains, scored
  exactly), the bound's optimum is 0.93–1.16 nats from the truth, while exact likelihood finds a member
  0.001 away. Only the objective differs.
- **A neural reference** (`neural_reference.md`): the standard Gaussian reverse kernel of Ho et al.,
  with an MLP noise predictor, trained by the same bound on the same data. At $T = 1000$ it reaches the
  ring (0.009 nats, better than any mixture) but not moons (0.315, like a 4-outcome chain). At $T = 20$ it
  is worse than the switching chains.

So under the diffusion bound with a fixed forward process, depth in time does not substitute for
outcomes per step. The bound trains a chain well only when each step's kernel can follow the true
reverse kernel; $K$-outcome, linearly gated steps cannot.

### Inference in the graph: E6, E7 and the hybrid

The additive wiring in 8 dimensions, with 4 to 12 overlapping switches and five random truths. Every
E-step feeds the same closed-form M-step, started at the true parameters
(`factorial_inference.md`, `hierarchy_inference.md`, `switching_blocks_graph.md`):

| E-step | gap at $L = 8$ | gap at $L = 12$ |
|---|---|---|
| exact enumeration | 0.03–0.04 | 0.03–0.05 |
| Mycelium star graph, one `PriorFactor` per switch (one pass of assumed-density filtering) | 0.48–0.67 | 0.48–0.79 |
| mean field | 0.49–0.92 | 0.46–0.81 |
| exact marginals, factorised pair moments (a diagnostic) | 0.42–0.64 | 0.46–0.75 |
| Gaussian EP (three seeds) | 0.16–0.40 | 0.27–1.08 |
| graph with blocks of switches as switching factors, one pass | mean 0.30 | mean 0.44 |
| the same blocks, iterated: structured mean field, or the graph with a `MeanField()` sum node | mean 0.14 | mean 0.16 |

The rows say three things:

- **The bias is in the pair moments.** With exact marginals and factorised pair moments, EM is
  nearly as far off as with the approximate marginals. At the truth the M-step misses the
  explaining-away covariance, so the truth is not a fixed point (a known hazard of factorised
  variational EM; see Turner & Sahani 2011). Linear-response pair moments (Welling & Teh 2004) made it
  worse; EP's joint Gaussian helps at $L = 8$ but is unreliable at $L = 12$ and three to four orders of
  magnitude slower.
- **It takes exact blocks and iteration.** Blocks of $L/2$ switches chosen by overlap, each a
  switching factor with $2^{L/2}$ outcomes, halve the bias in one pass. Iterating with the other blocks'
  current means halves it again. Blocks that ignore which switches overlap help much less, and
  iteration without blocks (mean field) does not help at all.
- **Mycelium now does it.** With the sum node on `MeanField()` and the blocks on belief
  propagation, the graph is the hybrid of Riegler et al. (2013). It reproduces the structured mean
  field to $9 \cdot 10^{-7}$ per point and $3 \cdot 10^{-8}$ in fitted parameters
  ([[L-ADR-0009 Factor nodes carry a message rule]]).

The same holds one level up (E7, a switch over sums): mean field over the switches inside a cluster
is 0.25–0.68 nats off, overlap blocks 0.03–0.33.

## 4. What it adds up to

1. **The building block exists and is in the package.** `SwitchingConstraintFactor` is closed form in
   every polarity, passes the factor conformance checks, and reduces to `LinearConstraintFactor` with
   one outcome.
2. **Expressivity comes from wiring, not depth.** The class of every composition is the finite
   mixtures; what composition changes is how many parameters a given arrangement of components
   costs. A sum node fed by switches is the wiring that pays.
3. **A diffusion of switching steps is limited by its objective.** The fixed-forward bound projects
   every step onto $K$ linear pieces. Exact marginal likelihood trains the same chains to the truth, but
   needs exact switch posteriors.
4. **Learning through the graph is an inference problem.** Factorised beliefs bias the parameters
   through the pair moments. Exact blocks plus mean-field iteration remove most of the bias, and that
   combination is now expressible in Mycelium with a per-node message rule.
5. **The quantum route needs amplitudes.** Born measurements, read as joints, add outcome queries and
   exact zeros, but not density: interference lives at the amplitude level, which probabilistic
   factor graphs do not compose.

## 5. Open

- The general form of the factor (per-outcome $A_k$, $Q_k$), needed for non-parallel modes, and the
  directed switching kernel for chains.
- A free energy for graphs with mean-field nodes, so that `parameter_step` can train them, and
  entropies of mixture beliefs at variables ([[Extending Mycelium]]).
- How to choose blocks of switches at scale, and whether blocks still help where exact inference is
  infeasible ($L \gtrsim 15$, untested).
- A training objective for compositional chains that scales: exact likelihood works but needs exact
  posteriors; a learned forward process is untested.
- Whether positive sums of products of positive semidefinite quadratics times a Gaussian are dense in
  $L^1(\mathbb R^n)$ for $n \ge 2$.

## 6. How the work was done

The design note was written first, with each claim marked as proved, known or expected, and then
reviewed twice by four language models (ChatGPT, Gemini, Mistral, Grok). Every review claim was
checked before it was accepted. The reviews corrected:

- a wrong inversion formula;
- a parameter count ($15K - 6$ identified parameters per kernel, not $15K - 1$);
- an invalid estimation-cost subtraction;
- the name of the graph E-step (assumed-density filtering, not belief propagation).

They also prompted the controls above: five seeds, EP, the objective control and the neural
reference. Both rounds are recorded in §13 of the branch's design note. The experiments stay on the
branch: they are evidence, not interface.

Related: [[Switching Constraint Factors]], [[Extending Mycelium]], [[Hierarchical Models]], [[Bethe Free Energy]], [[The Diffusion Family]], [[Mixture Belief]]
