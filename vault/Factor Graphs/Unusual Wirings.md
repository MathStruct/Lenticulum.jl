#design #overview

> Three wirings the current interface does not expose generically: an **output into a weight**, an **energy into an
> input**, and **beliefs on weights and energies**. Each has a literature, a reason to want it,
> and a way to go wrong; in Lenticulum each is a polarity question once the weight or the energy
> is a channel. This note collects them so the interface can plan for them;
> [[L-SPEC-003 Exposed Parameters and Energy Channels]] is the plan, and
> [[RQ-007 Risk Scalarisations and Local Decomposition]] the open question they raise.

> Sources: Schmidhuber, *Learning to Control Fast-Weight Memories*, Neural Computation 4(1), 1992; Ha, Dai & Le, *HyperNetworks*, ICLR 2017; Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021; Perez et al., *FiLM*, AAAI 2018; Finn, Abbeel & Levine, *Model-Agnostic Meta-Learning*, ICML 2017; Cruttwell et al., *Categorical Foundations of Gradient-Based Learning*, ESOP 2022 (Definition 2.3); Ghani, Hedges, Winschel & Zahn, *Compositional Game Theory*, LICS 2018; Schmidhuber, *Reinforcement Learning Upside Down*, 2019; Chen et al., *Decision Transformer*, NeurIPS 2021; Blundell et al., *Weight Uncertainty in Neural Networks*, ICML 2015; Bellemare, Dabney & Munos, *A Distributional Perspective on Reinforcement Learning*, ICML 2017; Donsker & Varadhan, CPAM 28(1), 1975; Artzner, Delbaen, Eber & Heath, *Coherent Measures of Risk*, Math. Finance 9(3), 1999; Rockafellar & Uryasev, *Optimization of Conditional Value-at-Risk*, J. Risk 2(3), 2000 (Theorem 1), and *Conditional Value-at-Risk for General Loss Distributions*, J. Banking & Finance 26(7), 2002; Myers, *Double Categories of Open Dynamical Systems*, arXiv:2005.05956, 2020; Föllmer & Schied, *Stochastic Finance*, 4th ed., 2016; St Clere Smithe & Perin, *AutoBayes*, arXiv:2503.18608, 2025 (Examples 3, 5); the question was raised by Daniel (2026-10-25); code: `lib/Mycelium.jl/src/factors.jl` (`OptimiserFactor`, `PriorFactor`), `lib/LenticulumCore.jl/src/channels.jl`, `energy.jl`
>
> Bibliography: [[Bibliography#^schmidhuber1992fastweights|Schmidhuber 1992]] · [[Bibliography#^ha2017hypernetworks|Ha et al. 2017]] · [[Bibliography#^schlag2021fastweight|Schlag et al. 2021]] · [[Bibliography#^perez2018film|Perez et al. 2018]] · [[Bibliography#^finn2017maml|Finn et al. 2017]] · [[Bibliography#^cruttwell2022gradient|Cruttwell et al. 2022]] · [[Bibliography#^ghani2018compositional|Ghani et al. 2018]] · [[Bibliography#^schmidhuber2019upsidedown|Schmidhuber 2019]] · [[Bibliography#^chen2021decision|Chen et al. 2021]] · [[Bibliography#^blundell2015weight|Blundell et al. 2015]] · [[Bibliography#^bellemare2017distributional|Bellemare et al. 2017]] · [[Bibliography#^donsker1975asymptotic|Donsker & Varadhan 1975]] · [[Bibliography#^artzner1999coherent|Artzner et al. 1999]] · [[Bibliography#^rockafellar2000cvar|Rockafellar & Uryasev 2000]] · [[Bibliography#^rockafellar2002cvar|Rockafellar & Uryasev 2002]] · [[Bibliography#^myers2020double|Myers 2020]] · [[Bibliography#^follmer2016stochastic|Föllmer & Schied 2016]] · [[Bibliography#^stclere2025autobayes|St Clere Smithe & Perin 2025]]
>
> Theory (CT-ML wiki): [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## 1. The common move: make it a channel

A factor has three kinds of port in the code: **channels** (variables of the graph, with a
polarity), **parameters** `ps` (threaded through, updated by an outer loop), and its **energy**
(read by the free energy, never by another factor). The wirings below all connect ports of
different kinds. [[Everything is a Factor]] already shows how the first gap closes: a factor may
**expose a parameter as a channel**, the move AutoBayes Example 3 (VBEM) makes when $\Theta$
moves from the parameter space onto a wire (the vault's paraphrase; Example 5 is the posterior over weights); then an `OptimiserFactor` or a `PriorFactor`
attaches to it like to any variable, and "the difference between [maximum likelihood and a
posterior over weights] is one edge". [[The Inferencer and the Optimizer]] §6 proposes a categorical
home in Myers' contravariant morphisms (Myers 2020), which plug "variables of some systems into
parameters of other systems"; that these wirings are instances of his construction is an
interpretation until the systems and morphisms are written down.

Once a weight or an energy is a channel, *what it is* (input, output, weight) is no longer fixed
by the factor; it is chosen by the polarity, as for any other channel ([[Channels and Polarity]]).
That is why the wirings are possible, and why "depending on the polarity" they are reasonable.

## 2. An output into a weight

| where | what is wired |
|---|---|
| fast weights (Schmidhuber 1992) | a slow network writes the weights of a fast one at every step |
| linear transformers (Schlag et al. 2021, §3) | are fast-weight programmers; unnormalised: $W_t = W_{t-1} + v_t k_t^\top$, output $W_t q_t$ (with feature maps $\phi(k), \phi(q)$ and a normaliser in the kernelised form) |
| hypernetworks (Ha et al. 2017) | one network generates weights (or weight modulations) of another |
| FiLM (Perez et al. 2018) | an output sets per-channel scales and shifts of another network |
| MAML (Finn et al. 2017) | the weight is the output of an inner optimisation |
| Para (Cruttwell et al. 2022, Def. 2.3) | a parameter is a second input; precomposing it with $\alpha : Q \to P$ is a reparametrisation (a data-dependent feed is a more general wiring) |

**Polarities of an exposed weight.** For a factor $y = f(x; W)$ with $W$ a channel:

| observed | unobserved | reading |
|---|---|---|
| $x$, $W$ | $y$ | forward pass; a hypernetwork when $W$ comes from another factor |
| $x$, $y$ | $W$ | learning as inference: the message to $W$ is a likelihood; combined with a prior on $W$ it gives a posterior over weights |
| $W$, $y$ | $x$ | the usual inverse problem with given weights |

**What goes wrong.**

1. **Loops.** If $y$ feeds $W$ and $W$ determines $y$, the graph has a cycle. As a static
   factor graph that is fine when the potentials define a normalisable law, and loopy message
   passing may (or may not) converge on it. As a *computation* it needs a reading: a time index
   (fast weights: $W_{t+1}$ from $y_t$) or a fixed point (an equilibrium factor, [[DEQ as a Relation]]).
2. **The factor becomes bilinear.** $y = Wx + \varepsilon$ with Gaussian $\varepsilon$ is linear in $x$ for fixed
   $W$ and linear in $W$ for fixed $x$, but not jointly: with Gaussian beliefs on both, the exact
   message is generically not Gaussian. A point on one side makes the other side's message an exact
   Gaussian kernel (possibly improper: one observation of $y$ does not determine all of $W$); beliefs
   on both need a stated approximation if a Gaussian family is required (mean-field VMP, as in
   Bayesian matrix factorisation, or EP/Laplace).
3. **Parameter sharing comes for free.** Two factors whose exposed weight channels connect to the
   same variable share the weight, with its gradient and its update handled once. Today `ps` has
   no parameter-tying interface ([[The Inferencer and the Optimizer]] §7.1): two entries can alias
   one array, but nothing makes shared gradients and updates correct.

## 3. An energy into an input

| where | what is wired |
|---|---|
| return-conditioned policies (Schmidhuber 2019; Chen et al. 2021) | the desired return-to-go is an *input* of the policy |
| open games (Ghani et al. 2018) | utility travels backwards as coutility, and can be another player's input |
| constrained optimisation | $E(x) \le c$ with a dual variable: the energy's value enters the dual's update |
| statistical mechanics | conditioning on $E(x) = e$ (below) |

**Polarities of an energy channel.** Give a factor an extra channel $e$ carrying its energy
$e = E_f(x, y)$ (a vector in its `pointwise_energyspace`, [[Scalar and Multivariate Energy]]):

| $e$ is | reading |
|---|---|
| unobserved | the energy's pushforward: a belief over values of the energy (§4) |
| observed, hard | conditioning on a level set $\{E_f = e\}$; for a scalar energy and a fixed reference measure, the **microcanonical** ensemble |
| observed, soft (finite precision $\tau$) | a quadratic penalty $\tfrac\tau2 \lVert E_f - e\rVert^2$: a soft constraint (a fixed Lagrange multiplier would instead be the linear tilt $\lambda (E_f - e)$) |
| an input of another factor | the energy as a signal: return conditioning, a learned loss, a game's payoff |

Reading the canonical ensemble ($\propto e^{-\beta E}$, $\beta$ dual to a constraint on the *mean* energy) and the
microcanonical one (a hard constraint on the energy) as two polarities of the pair
(energy, inverse temperature) is an analogy, proposed here, not a construction: a hard energy
value does not in general determine a unique temperature, and the two ensembles agree only under
hypotheses (equivalence of ensembles; not checked here). For a continuous energy at a regular
value, a level set has measure zero and conditioning on it is a disintegration (by the coarea
formula, with density proportional to $r(x) / \lVert \nabla E(x)\rVert$ for a belief with density $r$), the same
issue as a hard constraint in [[constraint]]; for discrete energies, atoms or flat regions a level set
can have positive probability.

**What goes wrong.**

1. **Self-reference.** If the free energy depends on an input that is itself an energy of the same
   graph, minimising it can reward moving the target (Goodhart). A **stop-gradient** on the energy
   wire (the value passes, the cotangent does not; [[L-SPEC-002 Stop Gradient]]) removes that
   derivative path, as with frozen targets; it does not freeze the target across iterations or
   prevent collapse in general, and it stops a downstream objective from training the upstream
   energy. A proposed default for taps, to be justified case by case, not a general remedy.
2. **Double counting.** The tapped energy must be counted once in the Bethe sum
   ([[Bethe Free Energy]]): in its own factor's term, not again in the term of the factor that
   reads it, unless that is intended (then it is a reweighting and should say so).

## 4. Weights and energies as beliefs

**Weights.** A posterior over weights is Bayesian deep learning; Blundell et al. (2015) fit a
Gaussian one by variational inference. In Lenticulum it is §1's exposed channel plus a
`PriorFactor`. Covariance MAP learning is now available through
`GaussianFactor(...; learn_noise=true, noise_prior=WishartPrecisionPrior(df, scale))`
([[LP-005 Learning the noise covariance]]). This is a local parameter penalty, not an
exposed parameter channel or a posterior belief over weights. A Wishart distribution on
precision transforms into an inverse-Wishart on covariance; its density-coordinate
Jacobian means their modes need not correspond under inversion. The implemented MAP
uses density in precision coordinates; see [[noise]].

**Energies.** A belief over the value of an energy is what distributional reinforcement learning
keeps for returns (Bellemare et al. 2017). Scalarising it is then the choice of a **risk measure**
$\rho$, and the scalarisation parameters of [[Scalar and Multivariate Energy]] §8 (issue #8: "like
the p of a norm") get a second family:

| $\rho(E)$ | name | finite when |
|---|---|---|
| $\mathbb E[E]$ | risk-neutral | $\mathbb E\lvert E\rvert < \infty$ |
| $\tfrac1\beta \log \mathbb E\, e^{\beta E}$, $\beta > 0$ | entropic risk (risk-averse) | $E$ has an exponential moment of order $\beta$ |
| $\mathrm{CVaR}_\alpha(E) = \min_t \bigl(t + \tfrac{1}{1-\alpha}\mathbb E(E - t)^+\bigr)$, $\alpha \in (0, 1)$ | conditional value-at-risk | $\mathbb E[E^+] < \infty$ |

(CVaR's formula: Rockafellar & Uryasev 2000, Theorem 1, for continuous losses, and 2002 for general
ones; CVaR is coherent in the sense of Artzner et al. 1999; the entropic risk is convex but not
coherent, since it is not positively homogeneous; Föllmer & Schied 2016, ch. 4.)

**Where free energy sits.** By the Donsker–Varadhan variational formula, for a probability law $p$,
$\beta > 0$, and $q$ ranging over laws with $q \ll p$ (the extrema attained at $q \propto p\,e^{\mp\beta E}$ when the
exponential moments and the energy's moments under them are finite),

$$
-\tfrac1\beta \log \mathbb E_p\, e^{-\beta E} \;=\; \inf_q \Bigl(\mathbb E_q[E] + \tfrac1\beta\,\mathrm{KL}(q\,\|\,p)\Bigr),
\qquad
\tfrac1\beta \log \mathbb E_p\, e^{\beta E} \;=\; \sup_q \Bigl(\mathbb E_q[E] - \tfrac1\beta\,\mathrm{KL}(q\,\|\,p)\Bigr).
$$

The left identity is the variational free energy at temperature $1/\beta$: inference minimises over
$q$, the **optimistic** certainty equivalent. The right one is the risk-averse entropic risk: the
same objective with the opponent choosing $q$, a KL-penalised worst case (distributionally robust
optimisation). So free energy and entropic risk are one formula with the sign of $\beta$ flipped,
not the same thing; [[Energies and Entropies]] keeps them apart.

**Heavy tails.** If $E$ has a heavier-than-exponential right tail (Student-$t$, log-normal), then
$\mathbb E\,e^{\beta E} = \infty$ for every $\beta > 0$ and the entropic risk is $+\infty$, while CVaR stays
finite whenever the mean of $E^+$ does (Student-$t$ with $\nu > 1$, log-normal). A heavy *left* tail does
not matter for this ($\mathbb E\,e^{\beta E} \le 1$ when $E \le 0$). With heavy-tailed energies, the choice of scalarisation
decides whether there is an objective at all.

**Local decomposition.** The expectation is additive over factors, which makes the energy part
of the Bethe free energy a sum of local terms. The entropic risk is additive for independent
energies and in general not otherwise (shared variables can make energies dependent); at a fixed
belief on a tree it is still exact by sum-product on the tilted potentials $\psi_a e^{\beta E_a}$. CVaR is
subadditive. What survives on loopy graphs, for CVaR, and under optimisation is
[[RQ-007 Risk Scalarisations and Local Decomposition]].

## 5. What the interface needs

[[L-SPEC-003 Exposed Parameters and Energy Channels]] plans three additions, none built:

1. **Exposure**: a wrapper that turns named parameter blocks of a factor into channels, with
   exact messages where the factor provides them and stated approximations otherwise.
2. **Energy taps**: a wrapper that adds the factor's energy as a channel, with the polarities of §3,
   a stop-gradient by default, and the Bethe accounting fixed.
3. **Risk scalarisations**: scalarisations that act on a belief over energies, which needs factors
   to provide that belief, not only its mean.

What to keep open meanwhile: channel spaces must allow matrix-valued and energy-space-valued
variables; `scalarise` must stay dispatchable on beliefs as well as on energy values; and nothing in
Mycelium should assume that parameters and channels are disjoint kinds.

Related: [[Everything is a Factor]], [[The Inferencer and the Optimizer]], [[Channels and Polarity]], [[Scalar and Multivariate Energy]], [[Energies and Entropies]], [[Bethe Free Energy]], [[L-SPEC-002 Stop Gradient]]
