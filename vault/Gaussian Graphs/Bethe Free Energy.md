#theorem #algorithm

> The free energy of a factor graph: local free energies of the factors plus a counting
> correction at the variables. On a tree, at exact beliefs, it is $-\log p$ of the data. On a
> loopy graph it is the Bethe approximation, whose stationary points are the fixed points of
> belief propagation.

> Sources: Yedidia, Freeman & Weiss, *Constructing Free-Energy Approximations and Generalized Belief Propagation Algorithms*, IEEE Trans. Inf. Theory 51(7), 2005 (the Bethe free energy and its stationary points); Riegler, Kirkelund, Manchon, Badiu & Fleury, IEEE Trans. Inf. Theory 59(1), 2013, eq. (17) (the merged free energy with mean-field nodes); code: `lib/Mycelium.jl/src/free_energy.jl`, tests: `test/runtests.jl` ("the chain is exact at every length"), `test/gaussian_oracle.jl`, `test/lp020.jl`
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## The formula

Each factor $f$ has a belief $b_f \propto f\,\prod_{i \in f}\mu_{i\to f}$ over its own variables and
a local free energy

$$
F_f \;=\; \mathbb E_{b_f}[-\log f] \;-\; H(b_f).
$$

Each variable $v$ has a marginal $b_v$ and the term $F_v = -H(b_v)$. With counting numbers
$\kappa_f = 1$ and $\kappa_v = 1 - d_v$, where $d_v$ counts the factors at $v$,

$$
F_{\text{Bethe}} \;=\; \sum_f F_f \;+\; \sum_v (1 - d_v)\,F_v
\;=\; \sum_f \mathbb E_{b_f}[-\log f] \;-\; \Bigl(\sum_f H(b_f) - \sum_v (d_v - 1)\,H(b_v)\Bigr).
$$

Energies add. Entropies add with a correction: a variable of degree $d_v$ appears in $d_v$
factor beliefs, so its entropy is counted $d_v$ times and $d_v - 1$ copies are removed.

The factor belief is built from the messages *into* the factor, not from the variables'
marginals. Using the marginals would multiply the factor's own outgoing message in again
(`messages_into`, `free_energy.md` §7).

## Why it is exact on a tree

On a tree with exact beliefs the joint factorises as

$$
p(x) \;=\; \frac{\prod_f b_f(x_f)}{\prod_v b_v(x_v)^{d_v - 1}},
$$

the junction-tree factorisation. Substituting it into $\mathbb E_p[-\log \prod_f f] - H(p)$ gives
the formula above, and that quantity is $-\log Z$. AutoBayes' chain rule (Theorem 23), folded
along any elimination order of the tree, telescopes to the same sum: the chain rule and the
Bethe form are one computation written in two orders. `chain_free_energy` is the first,
`bethe_free_energy` the second; the second has no order in it and is the one to report.

The total counting number is the Euler characteristic of the graph,

$$
\sum_f 1 + \sum_v (1 - d_v) \;=\; |F| + |V| - |E| ,
$$

which is 1 for a connected tree and $1 - L$ with $L$ independent loops. It counts loops; it
does not measure the error of the approximation, which depends on the potentials
([[Finding a Factor Graph]] measures it for Gaussian couplings). The suite asserts
`total_counting_number(g) == euler_characteristic(g)` for graphs of belief-propagation nodes.

## What "exact" requires

Exactness on a tree holds when **every message is in a family its receiver takes exactly**.

| situation | exact? |
|---|---|
| Gaussian, Dirac and categorical messages between the Gaussian, constraint and table factors | yes (`test/gaussian_oracle.jl`: random trees against the dense joint, means, covariances, free energy and gradient) |
| a rank-deficient (flat) Gaussian message, e.g. from a partial observation | yes since 2026-11-10; before, `GaussianFactor` threw or returned wrong marginals ([[LP-024 Findings of the review of 2026-11-09]]) |
| a mixture message from a switching factor into a Gaussian or constraint factor | **no**: the receiver collapses the mixture to its moments. The fixed point is then one of assumed-density filtering and is not a stationary point of this free energy |
| the same graph with the switch and the variables its mixture reaches in one cluster node | yes ([[cluster]]; `test/cluster.jl` against enumeration) |
| `NonlinearFactor` | no: its messages and its free energy are those of a local linear model ([[nonlinear]]) |
| `MixtureReduction` nodes | no free energy is defined; refused |

`propagate!` reports "one sweep is exact" for a tree schedule on belief-propagation nodes. Read
it with this table: exact for the factors' message rules.

## On a loopy graph

The same formula is the Bethe approximation. Fixed points of loopy belief propagation are its
stationary points (Yedidia, Freeman & Weiss), not necessarily minima, and belief propagation
need not converge ([[Loopy Message Passing]], [[Composing Optimisers]]). For Gaussian graphs
the means at a fixed point are exact and the variances are not (Weiss & Freeman 2001;
`test/runtests.jl`, "acausal and loopy").

## The gradient

At a fixed point the beliefs are stationary, so the derivative of $F_{\text{Bethe}}$ in the
parameters is the sum of the factors' partial derivatives at fixed beliefs (the envelope
theorem):

$$
\frac{dF_{\text{Bethe}}}{d\theta} \;=\; \sum_f \partial_\theta F_f(b_f, \theta).
$$

`free_energy_gradient` computes this (`factor_belief`, `belief_free_energy`;
[[LP-020 The fixed-message parameter gradient is not the free-energy gradient]]). On a tree
where the table above says "yes" it is the gradient of $-\log p$. Where the table says "no"
the beliefs are not stationary points of the reported free energy and the envelope argument
does not apply; `fit_em` then need not descend.

## Other node rules

- **Mean-field nodes** contribute an average energy $\mathbb E_{\prod_i q_i}[-\log f]$ and no
  entropy, and the variable counting numbers count belief-propagation neighbours only:
  $\kappa_v = 1 - |N_{\mathrm{BP}}(v)|$ (Riegler et al., eq. 17;
  [[L-SPEC-005 The Hybrid Free Energy]]). The total is then no longer the Euler characteristic.
- **Expectation-propagation nodes** are counted like belief-propagation nodes
  ([[L-SPEC-008 Expectation Propagation and Mixture Reduction]]).
- **`RelayFactor`** (an equality between two variables) has zero energy and the entropy of the
  pooled belief, $F = -H(q)$: without it the relayed variable's extra degree is uncounted
  ([[LP-019 Defects found in the review of the core packages]]).

## Entropy conventions

`variable_entropy` is the differential entropy relative to Lebesgue measure for a Gaussian,
the Shannon entropy for a categorical, and 0 for a `DiracBelief`
([[LP-006 Gaussian entropies are relative to Lebesgue measure]]). The value 0 for a point is
the convention for a clamped variable: the factor beliefs then range over the other variables
only, and the sum is $-\log p$ of the clamped values (checked in `test/gaussian_oracle.jl`).

Two consequences:

- Free energies are comparable across graphs only when the same variables are clamped. A
  variable clamped in one graph and given a narrow Gaussian in another differs by an amount
  that grows without bound as the Gaussian narrows. Structure searches that compare free
  energies ([[Finding a Factor Graph]], [[RQ-004 Is the Bethe Criterion Consistent for Structure Learning]])
  must keep the data clamped in every candidate.
- Factors that pass points on unclamped variables (`message = :point` of the diffusion factor)
  run at zero temperature inside a formula written for temperature one: the entropy term
  vanishes and what remains is the energy of [[Energy-Based Factor Graphs]] §3.2.

## The grading

`factor_free_energies` returns the per-factor terms keyed by factor name, and each factor's
own scalarisation is applied to its own term ([[Scalar and Multivariate Energy]]). The
Gaussian family grades each term as `(fit, complexity, negentropy)`.

Related: [[Loopy Message Passing]], [[free_energy]], [[Energy-Based Factor Graphs]],
[[Finding a Factor Graph]], [[The Linear Gaussian Chain]]
