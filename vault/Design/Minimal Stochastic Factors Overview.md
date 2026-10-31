#overview

> A guide to the `experiment-qm` branch for a reader new to it: the question, the answer so far,
> what was measured, what three external reviews changed, and what is open. The mathematics is in
> [[Minimal Stochastic Factors]]; this note only points into it and into the experiments.

> Sources: original to this vault. code: `examples/measurement_factor.jl`, `examples/switching_factor_e1.jl`, `examples/switching_reverse_chain_e4a.jl`, `examples/switching_small_steps.jl`, `examples/composition_e3.jl`, `examples/switching_chain_e4b.jl`, `examples/chain_objective.jl`, `examples/hierarchy_e3b.jl`, `examples/factorial_inference.jl`, `examples/structure_switches.jl`, `examples/minimal_stochastic_factors_checks.jl`

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
   step scores like a flat $K$-component mixture whatever its depth (E4(b)): depth in time does not
   substitute for outcomes. The cause is the objective, not the class: a chain of 2-outcome
   kernels can be E3's 16-mode grid exactly (Proposition 10), but the diffusion bound with a fixed
   forward process makes each step follow the data's noised marginals (Proposition 9).

## What was measured

| experiment | result | where |
|---|---|---|
| spectral measurement on two orthogonal lines | as good as the true model (held-out $\log p(y \mid x)$ 0.688 vs 0.691) | `examples/measurement_factor.md` |
| the same on lines 60° apart | 30° wrong, worse than one Gaussian (−0.939 vs −0.823) | `examples/measurement_factor.md` |
| **E1**: switching factor on lines 60° apart | both lines recovered (errors < 0.02°), held-out $\log p(y \mid x)$ 0.702 = true model's, 6 parameters vs a free mixture's 11; message passing in a Mycelium graph matches direct conditioning to $5\cdot10^{-15}$ | `examples/switching_factor_e1.md` |
| **E4(a)**: reverse chain on a 3-component mixture | exact kernels reproduce $\log q_0(z_0)$ on every path to $10^{-14}$; the shared chain (17 parameters) reaches −2.123 vs the truth's −2.108; linear-Gaussian steps fail; two outcomes per small step do *not* yet suffice | `examples/switching_reverse_chain_e4a.md` |
| **small-step test**: conditional per-step fits, $T$ = 10 to 100 | three outcomes per step match the truth up to the estimation cost of fitting $T$ kernels; two outcomes keep a gap of about 0.44 nats however small the steps, and their samples level off at 3× the energy-distance floor | `examples/switching_small_steps.md` |
| **E3**: composition vs flat mixture on product grids (8, 16 modes) and a ring | on the grids, additive composition (14–17 parameters) is within 0.005 nats of the truth and, at 150 points, within 0.04 where the best flat mixture is 0.36–0.48 away; on the ring it is 0.89 away; rank-one constraints on a shared variable fail everywhere | `examples/composition_e3.md` |
| **E3, fitting** | exact EM for the additive wiring (closed-form M-step): monotone; fewer starts reach the optimum than with Adam, but each costs 1–2 % as much, so 256 EM starts find the same optima 4–5× faster | `examples/composition_e3.md` |
| **E4(b)**: chains on two moons and a ring | a chain of $T \le 20$ steps with $K$ outcomes each scores like a flat $K$-component mixture (within 0.006 nats at $T = 5$, at most 0.10 better at $T = 20$); flat with 16 components (95 parameters) beats every chain (up to 1 185) | `examples/switching_chain_e4b.md` |
| **Propositions 9, 10**: the objective, not the class | E3's additive model is a 4-step chain of 2-outcome kernels (marginal identical to $7\cdot10^{-15}$, 0.001 nats from the truth on the 16-mode grid); chains of the same kind trained by the diffusion bound score like flat mixtures with $K$ = 2 or 4 components (gaps 0.78–1.09) | `examples/chain_objective.md` |
| **E3b**: a switch over sums, on 3 clusters of rotated 2 × 2 grids | the hierarchical wiring (29 parameters, exact EM) is 0.008 nats from the truth at 2 000 points and 0.091 at 150; a flat 12-mixture (71) is 0.016 and 0.996 away (it overfits at 150); one sum node stays 0.64–0.80 away | `examples/hierarchy_e3b.md` |
| **E6, factorial inference**: exact vs Mycelium graph (collapsed BP) vs mean field, $L$ = 4, 8, 12 switches in 8-D | with 4 switches all are within 0.04 nats of the truth; with 8–12 overlapping switches the factorised E-steps converge 0.60–0.92 away even from the truth, exact EM 0.03–0.04; the graph E-step is ~100× faster at $L = 12$ | `examples/factorial_inference.md` |
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

## What is open

- **Training compositional chains.** Chains of few-outcome kernels can represent product grids
  (Proposition 10), but the diffusion objective pins them to $K$ components (Proposition 9). They
  need an objective that learns the inference side, a hierarchical model with exact switch
  posteriors; E3's EM does this for one level.
- **Structured beliefs in Mycelium.** Mycelium now runs the switch-and-sum wiring (mixture
  inputs to linear factors are collapsed to moments), but factorised posteriors are biased by
  0.6–0.9 nats once switches overlap (E6). Region beliefs over groups of switches, EP with
  cavities, or exact blocks are the candidates.
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
julia -t auto --project=. examples/factorial_inference.jl           # about 10 minutes on 8 threads
julia -t auto --project=. examples/structure_switches.jl            # about 14 minutes on 8 threads
julia --project=. -e 'using Pkg; Pkg.test()'                       # includes the experiments' tests
```

Related: [[Minimal Stochastic Factors]], [[Scalar and Multivariate Energy]], [[Hierarchical Models]]
