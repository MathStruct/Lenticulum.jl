#overview

> A guide to the `experiment-qm` branch for a reader new to it: the question, the answer so far,
> what was measured, what three external reviews changed, and what is open. The mathematics is in
> [[Minimal Stochastic Factors]]; this note only points into it and into the experiments.

> Sources: original to this vault. code: `examples/measurement_factor.jl`, `examples/switching_factor_e1.jl`, `examples/switching_reverse_chain_e4a.jl`, `examples/switching_small_steps.jl`, `examples/minimal_stochastic_factors_checks.jl`

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
   parameter efficiency, not a larger class (Proposition 7).
4. **The quantum reading, done properly, is a Born joint** with a POVM. As a model of $p(z)$ it is
   thinner than a mixture (its marginal does not even depend on the POVM unless outcomes are
   weighted); its distinctive capacity is in queries about the outcome (§6). The spectral
   measurement of the original idea has no joint distribution behind it, which is why it is not
   polarity-compatible and finds only orthogonal modes (§7).
5. **A diffusion is a chain of directed switching kernels.** For mixture data the exact reverse
   kernels are switching kernels whose parameters follow from the noise schedule (Proposition 8),
   so a shared, time-conditioned parameterisation comes for free (§8). Small steps do **not** let
  fewer outcomes than components suffice: each step needs one outcome per linear piece of the
  score field (the small-step test).

## What was measured

| experiment | result | where |
|---|---|---|
| spectral measurement on two orthogonal lines | as good as the true model (held-out $\log p(y \mid x)$ 0.688 vs 0.691) | `examples/measurement_factor.md` |
| the same on lines 60° apart | 30° wrong, worse than one Gaussian (−0.939 vs −0.823) | `examples/measurement_factor.md` |
| **E1**: switching factor on lines 60° apart | both lines recovered (errors < 0.02°), held-out $\log p(y \mid x)$ 0.702 = true model's, 6 parameters vs a free mixture's 11; message passing in a Mycelium graph matches direct conditioning to $5\cdot10^{-15}$ | `examples/switching_factor_e1.md` |
| **E4(a)**: reverse chain on a 3-component mixture | exact kernels reproduce $\log q_0(z_0)$ on every path to $10^{-14}$; the shared chain (17 parameters) reaches −2.123 vs the truth's −2.108; linear-Gaussian steps fail; two outcomes per small step do *not* yet suffice | `examples/switching_reverse_chain_e4a.md` |
| **small-step test**: conditional per-step fits, $T$ = 10 to 100 | three outcomes per step match the truth up to the estimation cost of fitting $T$ kernels; two outcomes keep a gap of about 0.44 nats however small the steps, and their samples level off at 3× the energy-distance floor | `examples/switching_small_steps.md` |
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

- **Where depth could still pay.** The small-step test refuted "few outcomes per small step"
  for flat mixture data. Whether composition across steps buys anything for data with *structure*
  (a hierarchy or a product of factors) is the question E3 and E4(b) ask.
- **Composition** (E3): when $L$ binary switches beat a flat mixture of $2^L$ components.
- **The Born arm** (E2) and the density question of §6 for products of positive semidefinite
  quadratics in two or more dimensions.
- **Learning through the graph** rather than by direct maximum likelihood, and the bias of
  collapsed beliefs (E6).

## Running it

```sh
julia --project=. examples/switching_factor_e1.jl
julia -t auto --project=. examples/switching_reverse_chain_e4a.jl   # about 9 minutes on 8 threads
julia -t auto --project=. examples/switching_small_steps.jl         # about 57 minutes on 8 threads
julia --project=. -e 'using Pkg; Pkg.test()'                       # includes the experiments' tests
```

Related: [[Minimal Stochastic Factors]], [[Scalar and Multivariate Energy]], [[Hierarchical Models]]
