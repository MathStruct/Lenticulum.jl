#overview

> A guide to the `experiment-qm` branch for a reader new to it: the question, the answer so far,
> what was measured, what three external reviews changed, and what is open. The mathematics is in
> [[Minimal Stochastic Factors]]; this note only points into it and into the experiments.

> Sources: original to this vault. code: `examples/measurement_factor.jl`, `examples/switching_factor_e1.jl`, `examples/switching_reverse_chain_e4a.jl`, `examples/switching_small_steps.jl`, `examples/composition_e3.jl`, `examples/switching_chain_e4b.jl`, `examples/chain_objective.jl`, `examples/hierarchy_e3b.jl`, `examples/factorial_inference.jl`, `examples/hierarchy_inference.jl`, `examples/neural_reference.jl`, `examples/switching_blocks_graph.jl`, `examples/structure_switches.jl`, `src/switching.jl`, `examples/minimal_stochastic_factors_checks.jl`

## The question

What is the smallest **stochastic factor** from which an expressive generative model can be
built by composition alone, with no neural network inside any factor? Each factor should be
linear or nearly so, sample, support several polarities (any variable may be the one inferred),
be a multivariate statistical game in the sense of [[Scalar and Multivariate Energy]], learn by
gradients, and compose under message passing and the Bethe free energy. The original idea was a
quantum-style measurement: a linear factor whose output is an eigenvector drawn by its eigenvalue,
with no iterative denoising.

## The answer so far

1. **Linear plus Gaussian never becomes multimodal**, however it is wired or queried
   ([[Minimal Stochastic Factors]] Proposition 2). Something non-Gaussian has to enter.
2. **The smallest addition is a discrete outcome selecting the right-hand side** of a linear
   constraint, $Az = b_k$, or more generally $A_k z = b_k$: the *switching constraint factor*
   $\psi(k, z) = w_k\, \mathcal N(A_k z; b_k, Q_k)$. Its messages, local free energy and gradients are closed
   form, and a closure lemma makes "smallest" precise (§2–§5). Nonlinearity comes from conditioning
   on the discrete alternatives: $\mathbb E[y \mid x]$ is a softmax-gated sum of linear maps (Proposition 3.3).
   Its shared form ($A$ and $Q$ the same for every outcome) is now a package factor,
   `SwitchingConstraintFactor` (`src/switching.jl`).
3. **Composed, these factors give exactly the finite Gaussian mixtures**; composition buys
   parameter efficiency, not a larger class (Proposition 7). It pays only when the data have the
   structure the wiring imposes, and only with the right wiring. Switches feeding a sum node give
   grids of modes; several constraints on one variable intersect instead (E3). A switch over
   such sums gives clustered grids, and exact EM trains it (E3b).
4. **The quantum reading, done properly, is a Born joint** with a POVM. As a model of $p(z)$ it is
   thinner than a mixture (its marginal does not even depend on the POVM unless outcomes are
   weighted); its distinctive capacity is in queries about the outcome (§6). The spectral
   measurement of the original idea has no joint distribution behind it, which is why it is not
   polarity-compatible and finds only orthogonal modes (§7).
5. **A diffusion is a chain of directed switching kernels.** For mixture data the exact reverse
   kernels are switching kernels whose parameters follow from the noise schedule (Proposition 8),
   so a shared, time-conditioned parameterisation comes for free (§8). Small steps do **not** let
   fewer outcomes than components suffice: each step needs one outcome per linear piece of the
   score field (the small-step test). On data that are not a mixture, a chain with $K$ outcomes per
   step scores like a flat $K$-component mixture at every depth tried, $T \le 20$ (E4(b)): depth in time does not
   substitute for outcomes. The cause is the objective, not the class: a chain of 2-outcome
   kernels can be E3's 16-mode grid exactly (Proposition 10), but the diffusion bound with a fixed
   forward process makes each step follow the data's noised marginals (Proposition 9). A neural
   Gaussian kernel trained by the same bound does reach the truth on the ring at $T = 1000$, so the
   bound fails together with few-outcome steps, not on its own.

## What was measured

| experiment | result | where |
|---|---|---|
| spectral measurement on two orthogonal lines | as good as the true model (held-out $\log p(y \mid x)$ 0.688 vs 0.691) | `examples/measurement_factor.md` |
| the same on lines 60° apart | 30° wrong, worse than one Gaussian (−0.939 vs −0.823) | `examples/measurement_factor.md` |
| **E1**: switching factor on lines 60° apart | both lines recovered (errors < 0.02°), held-out $\log p(y \mid x)$ 0.702 = true model's, 6 parameters vs a free mixture's 11; message passing in a Mycelium graph matches direct conditioning to $5\cdot10^{-15}$ | `examples/switching_factor_e1.md` |
| **E4(a)**: reverse chain on a 3-component mixture | exact kernels reproduce $\log q_0(z_0)$ on every path to $10^{-14}$; the shared chain (17 parameters) reaches −2.123 vs the truth's −2.108; linear-Gaussian steps fail; two outcomes per small step do *not* yet suffice | `examples/switching_reverse_chain_e4a.md` |
| **small-step test**: conditional per-step fits, $T$ = 10 to 100 | three outcomes per step match the truth up to the estimation cost of fitting $T$ kernels; two outcomes keep a gap of 0.50–0.68 nats however small the steps, and their samples level off at 3× the energy-distance floor | `examples/switching_small_steps.md` |
| **E3**: composition vs flat mixture on product grids (8, 16 modes) and a ring | on the grids, additive composition (14–17 parameters) is within 0.005 nats of the truth and, at 150 points, within 0.04 where the best flat mixture is 0.36–0.48 away; on the ring it is 0.89 away; rank-one constraints on a shared variable fail wherever there are more switches than dimensions, and match additive on a 4-mode grid with $L = n = 2$ | `examples/composition_e3.md` |
| **E3, fitting** | exact EM for the additive wiring (closed-form M-step): monotone; fewer starts reach the optimum than with Adam, but each is far cheaper: 256 EM starts take 2.3–6.5 s against 11.0–26.9 s for 16 Adam starts at 2 000 points, and a 1 % miss rate needs 31–75 EM or 13–37 Adam starts | `examples/composition_e3.md` |
| **E4(b)**: chains on two moons and a ring | a chain of $T \le 20$ steps with $K$ outcomes each scores like a flat $K$-component mixture (within 0.006 nats at $T = 5$, at most 0.10 better at $T = 20$); flat with 16 components (95 parameters) beats every chain (up to 1 085) | `examples/switching_chain_e4b.md` |
| **E7 in the graph**: blocks of switches as `SwitchingConstraintFactor`s in Mycelium's star graph, E6's five truths | the graph agrees with its vectorised E-step to $5 \cdot 10^{-12}$; one pass with overlap blocks cuts E6's bias from 0.61 to 0.30 nats ($L = 8$) and from 0.59 to 0.44 ($L = 12$); iterating between the same blocks (structured mean field) reaches 0.14 and 0.16 | `examples/switching_blocks_graph.md` |
| **neural reference** for E4(b): Ho et al.'s Gaussian reverse kernel with an MLP noise predictor, same bound and scores | at $T = 1000$ (20 207 parameters) 0.009 nats from the truth on the ring, better than every E4(b) model; 0.315 on moons, no better than a 4-outcome chain; worse than the chains at $T = 20$ | `examples/neural_reference.md` |
| **Propositions 9, 10**: the objective, not the class | E3's additive model is a 4-step chain of 2-outcome kernels (marginal identical to $7\cdot10^{-15}$, 0.001 nats from the truth on the 16-mode grid); chains of the same kind trained by the diffusion bound score like flat mixtures with $K$ = 2 or 4 components (gaps 0.78–1.09); in one family (constant gates), scored exactly, the bound's optimum is 0.93–1.16 nats away while exact likelihood finds a member 0.001 away | `examples/chain_objective.md` |
| **E3b**: a switch over sums, on 3 clusters of rotated 2 × 2 grids | the hierarchical wiring (29 parameters, exact EM) is 0.008 nats from the truth at 2 000 points and 0.091 at 150; a flat 12-mixture (71) is 0.016 and 0.996 away (it overfits at 150); one sum node stays 0.64–0.80 away | `examples/hierarchy_e3b.md` |
| **E6, factorial inference**: exact vs Mycelium graph (one-pass ADF), mean field, linear response and EP, $L$ = 4, 8, 12 switches in 8-D, five seeds | with 8–12 overlapping switches every factorised E-step converges 0.42–1.12 nats away even from the truth, exact EM 0.01–0.05; exact marginals with factorised pair moments are nearly as biased, so the bias is in the pair moments; EP's joint Gaussian helps at $L = 8$ (0.16–0.40) but is unreliable at $L = 12$ and ~10³× slower; the graph E-step is ~100× faster than exact at $L = 12$ | `examples/factorial_inference.md` |
| **E7, structured inference in the hierarchy**: exact top switch, exact blocks of switches, 3 clusters of 6 or 8 switches in 8-D, three seeds | mean field inside a cluster is 0.25–0.68 nats off from the truth, exact 0.03–0.06; blocks of $L/2$ switches chosen by overlap cut it to 0.03–0.33, consecutive blocks only to 0.15–0.48; from random starts every method is 0.50–0.75 off | `examples/hierarchy_inference.md` |
| **structure learning**: greedy add/remove switch, split/merge cluster, exact EM + BIC | finds the generating structure on the 8-mode grid and the tree, a more compact exact one on the ring (two rotated squares), an equivalent larger one on the 16-mode grid; always within 0.005 nats of the truth, with 12–18 fits against 45 exhaustive | `examples/structure_switches.md` |
| checks behind the review corrections | the corrected inversion and Born claims, numerically | `examples/minimal_stochastic_factors_checks.jl` |

Every experiment has a test in `test/` and runs in the top-level suite.

## What the reviews changed

The first version of the design note was reviewed by ChatGPT, Gemini and Mistral (2026-10-31),
none of which could run the repository's code. All three independently found the same three
errors, which is good evidence that they were the real ones:

- the claimed closed-form inversion against an arbitrary prior was false (the kernel's own
  normaliser does not cancel; off by $L^1$ 0.164 in a check);
- the Born joint's marginal does not depend on the POVM, so a proposed experiment was vacuous;
- a local free energy was called a KL divergence although the potential is not normalisable.

Gemini added the most consequential new point: chaining switching *potentials* is not a directed
reverse chain, so the diffusion reading needs normalised kernels. Mistral ran its own numerical
checks and contributed the closure lemma's motivation, requirement 5 of a diffusion (one
time-conditioned model), and experimental controls. Every accepted point was checked first, two
claims from Mistral were corrected rather than adopted, and every reference the reviewers supplied
was checked against Crossref or arXiv: 13 are now in the bibliography, and four had wrong details.
The full record is §13 of [[Minimal Stochastic Factors]].

A second round (ChatGPT, Gemini, Grok, Mistral; 2026-10-31) reviewed the experiments. Accepted
after checking:
- the game reading of §4 used two different joints;
- chains with state-dependent gates are not finite mixtures;
- a kernel has $15K - 6$ identified parameters, not $15K - 1$;
- the estimation-cost subtraction is invalid for the misspecified $K = 2$;
- the one-pass graph E-step is assumed-density filtering, not BP;
- the rank-one failure in E3 is dimensional.

It also prompted three new controls: five seeds and EP for E6, the objective control in one family,
and a sweep of the importance-weighted bound. Refuted: that the gated chains can be scored by exact
path enumeration, an attribution of Proposition 9, a proposed "Proposition 11", and a
linear-response formula. Grok found no errors.

## What is open

- **Training compositional chains.** Chains of few-outcome kernels can represent product grids
  (Proposition 10), but the diffusion objective with a fixed forward process pins them to $K$
  components (Proposition 9; an objective control in one family confirms it). Exact marginal
  likelihood trains them, but needs exact switch posteriors; a learned forward process is untested.
  The same bound trains a neural Gaussian kernel to the truth on the ring at $T = 1000$, so what
  fails is the bound combined with few-outcome steps.
- **Structured beliefs in Mycelium.** Mycelium now runs the switch-and-sum wiring (mixture
  inputs to linear factors are collapsed to moments), but factorised posteriors are biased by
  0.4–1.1 nats once switches overlap (E6), and the bias is in the pair moments, not the marginals.
  Gaussian EP helps only partly. Exact blocks of switches remove most of the bias when the blocks
  keep overlapping switches together (E7). With the new `SwitchingConstraintFactor` a block is one
  factor in the graph, but one pass of messages removes only part of the bias; the rest needs the
  sum node to iterate with the other blocks' current means (a variational or EP schedule), which
  Mycelium does not have yet.
- **Structure learning.** Mycelium has no rules that add or remove variables, factors or edges:
  graphs are frozen once built ([[Dynamic Graphs]]), and the only structure search is a Gaussian
  edge-addition example ([[Finding a Factor Graph]]). For switch wirings the natural moves are
  adding a switch or splitting a cluster, scored exactly by EM while the posterior is small. A
  greedy search with these moves works on the toy data (`examples/structure_switches.md`). Open:
  doing the moves on a live Mycelium graph, and scoring them once the posterior is too large.
- **Composition that scales.** E3 shows that composition pays when the data have a product
  arrangement and switches feed a sum node, and exact EM makes it cheap to fit. The next questions
  are trees wider and deeper than E3b's (where the exact posterior over $J \cdot 2^L$ states must give
  way to messages in a graph) and real data with such structure.
- **The Born arm** (E2) and the density question of §6 for products of positive semidefinite
  quadratics in two or more dimensions.
- **Learning through the graph** rather than by direct maximum likelihood, and the bias of
  collapsed beliefs (E6).

## Running it

```sh
julia --project=. examples/switching_factor_e1.jl
julia -t auto --project=. examples/switching_reverse_chain_e4a.jl   # about 9 minutes on 8 threads
julia -t auto --project=. examples/switching_small_steps.jl         # about 57 minutes on 8 threads
julia -t auto --project=. examples/composition_e3.jl                # about 4 minutes on 8 threads
julia -t auto --project=. examples/switching_chain_e4b.jl           # about 95 minutes on 8 threads
julia -t auto --project=. examples/chain_objective.jl               # about 18 minutes on 8 threads
julia -t auto --project=. examples/hierarchy_e3b.jl                 # under a minute on 8 threads
julia -t auto --project=. examples/factorial_inference.jl           # about two hours on 8 threads, most of it EP
julia -t auto --project=. examples/hierarchy_inference.jl           # about 75 minutes on 8 threads
julia -t auto --project=. examples/neural_reference.jl             # about 5 minutes on 8 threads
julia -t auto --project=. examples/switching_blocks_graph.jl        # about 3 minutes on 8 threads
julia -t auto --project=. examples/structure_switches.jl            # about 14 minutes on 8 threads
julia --project=. -e 'using Pkg; Pkg.test()'                       # includes the experiments' tests
```

Related: [[Minimal Stochastic Factors]], [[Scalar and Multivariate Energy]], [[Hierarchical Models]]
