#implementation

> `SwitchingConstraintFactor`: a linear constraint whose right-hand side is chosen by one
> discrete outcome, $0 = \sum_i A_i x_i - b_k + \varepsilon$ with $P(k) = w_k$. It is the smallest
> factor that leaves the Gaussian class, and every message and the local free energy are closed
> form. The theory is in [[Switching Constraint Factors]]; the experiments that led to it (E1–E7,
> on the branch `experiment-qm`) are summarised in [[Composing Stochastic Factors]].

> Sources: Riegler, Kirkelund, Manchon, Badiu & Fleury, IEEE Trans. Inf. Theory 59(1), 2013 (the hybrid BP/mean-field scheme of §4b). code: `switching.jl`, `constraint.jl`, `gaussian.jl`; tests: `test/switching_constraint.jl`; experiment: [`examples/switching_blocks_graph.jl`](https://github.com/MathStruct/Lenticulum.jl/blob/experiment-qm/examples/switching_blocks_graph.jl) (branch `experiment-qm`)
>
> Bibliography: [[Bibliography#^riegler2013merging|Riegler et al. 2013]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## 1. The factor

$$
f(x_1, \dots, x_n, k) = w_k\, \mathcal N\Bigl(\sum_i A_i x_i - b_k;\ 0,\ Q\Bigr), \qquad k \in \{1, \dots, K\}.
$$

The parameters are `(A, b, logw)`: the coefficients `A` (a `NamedTuple` keyed by channel), the
`m × K` matrix `b` of right-hand sides and the unnormalised log-weights, $w = \operatorname{softmax}(\texttt{logw})$.
`Q` is a fixed hyperparameter, as in `LinearConstraintFactor`. The outcome is a channel of its
own (`switch`, default `:k`) with `K` states.

`A` and `Q` are shared by the outcomes, the "location-shift" case of §11 of the design note. That
is what keeps the factor small: every outcome induces a Gaussian with the **same precision**, so
the outcomes differ only in a mean.

## 2. Messages

With $d_k = b_k - \sum_{i \ne t} A_i m_i$ and $R = Q + \sum_{i \ne t} A_i S_i A_i^\top$ for incoming $\mathcal N(m_i, S_i)$:

- **to the outcome:** $P(k) \propto w_k\, \mathcal N(d_k; 0, R)$, a `CategoricalBelief`;
- **to a continuous channel $t$**, with $\pi_k$ the incoming outcome belief (flat if latent or absent):

$$
\mu_{c \to t}(x_t) \;\propto\; \sum_k \pi_k w_k\, \mathcal N(A_t x_t; d_k, R),
$$

a `MixtureBelief` of the canonical Gaussians $\mathcal N^{-1}(A_t^\top R^{-1} d_k,\ A_t^\top R^{-1} A_t)$. An observed
outcome (a `DiracBelief` with an index) gives one `GaussianBelief`, the `LinearConstraintFactor`
message with $c = b_k$.

The per-outcome constants go into the mixture weights, and how depends on whether the components
are proper:

- if $A_t$ has full column rank, the components are normalised Gaussians and the weight of
  outcome $k$ is $\log \pi_k w_k - \tfrac12 d_k^\top R^{-1} d_k + \tfrac12 \eta_k^\top \Lambda^{-1} \eta_k$;
- otherwise the components are improper, and Mycelium's mixture product (`messages.jl`,
  `_mixture_product`) treats each as the unnormalised exponential $\exp(\eta^\top x - \tfrac12 x^\top \Lambda x)$, so the
  weight is $\log \pi_k w_k - \tfrac12 d_k^\top R^{-1} d_k$.

The test pools both kinds with a Gaussian prior and compares the result, weights and components,
with plain Gaussian conditioning outcome by outcome.

## 3. The local free energy

The factor's belief $b_c(x, k) \propto f(x, k)\, \mu_k(k) \prod_i \mu_i(x_i)$ is, for each outcome, a Gaussian over
the free channels, and all of them have the same precision $\Lambda = G^\top Q^{-1} G + \operatorname{blockdiag}(\Lambda_i)$. So
its entropy is closed form:

$$
H(b_c) = H(q) + H(x \mid k), \qquad q_k \propto \mu_k(k)\, w_k \exp\Bigl(-\tfrac12 c_k^\top Q^{-1} c_k + \tfrac12 \eta_k^\top \Lambda^{-1} \eta_k\Bigr),
$$

with $c_k = b_k$ minus the clamped channels' contributions and $\eta_k = G^\top Q^{-1} c_k + \sum_i \eta_i$. The local
free energy $F_c = \mathbb E_{b_c}[-\log f] - H(b_c)$ is graded as `(fit, complexity, negentropy)`:

| part | value |
|---|---|
| `fit` | $\sum_k q_k\, \tfrac12 \bigl(\bar r_k^\top Q^{-1} \bar r_k + \operatorname{tr} Q^{-1} \operatorname{Cov} r\bigr)$ |
| `complexity` | $\tfrac12 \log\det 2\pi Q - \sum_k q_k \log w_k$ |
| `negentropy` | $-H(q) - H(x \mid k)$ |

The tests check it three ways: against `LinearConstraintFactor` at $K = 1$ (all three parts);
against the identity $F_c = \mathbb E_{b_c}[\log \pi + \log \rho] - \log Z$ for a Gaussian message $\pi$ on a latent
channel and a categorical one $\rho$ on the outcome; and, in a Mycelium graph (outcome prior, factor,
clamped data), the Bethe free energy equals $-\log p(z)$ of the Gaussian mixture.

## 4. Polarities and the game interface

One polarity per channel, the outcome included, with the others observed; a latent outcome is
marginalised under the factor's own weights. The open model is pure and its inversion exact, as
for `LinearConstraintModel`. The pointwise energy is $-\log f(x, k)$, a scalar. The factor passes
`check_factor_conformance` in every polarity.

## 4b. Mean-field messages

On a `MeanField()` node (Mycelium `passing.md`, "Mean-field nodes") the factor sends
$\exp \mathbb E_q[\log f]$ over the other channels' current marginals. For each outcome $\log f$ is quadratic in
the continuous channels, so only means enter. With $\bar s = \sum_{i \ne t} A_i \mathbb E_q[x_i]$ and $q$ the outcome's
marginal, the message to a continuous channel is $\mathcal N^{-1}(A_t^\top Q^{-1}(\bar b - \bar s), A_t^\top Q^{-1} A_t)$ with
$\bar b = \sum_k q_k b_k$. The message to the outcome is $P(k) \propto w_k \exp(-\tfrac12 (b_k - \bar s)^\top Q^{-1}(b_k - \bar s))$. The
test checks both against the formulas, and the slope and curvature of $\mathbb E_q[\log f]$ against Monte
Carlo. `LinearConstraintFactor` has the same rule with one outcome (`constraint.jl`).

## 5. What it does not do

- **Per-outcome coefficients or noise.** With $A_k$ or $Q_k$ depending on the outcome, the outcome
  Gaussians have different precisions: the message to a continuous channel is still a mixture,
  but the components no longer share a precision, and improper components with different
  precisions cannot be weighted consistently. Experiment E1's two crossing lines need this (different
  $A_k$). Not implemented.
- **Mixture inputs** are collapsed to their moments before use, in the messages and in the free
  energy, as for `LinearConstraintFactor` (`gaussian.md`, "Mixture inputs"). So a chain of
  switching factors on belief propagation runs as assumed-density filtering, and experiment E6 measured the
  bias that causes when switches overlap. Putting the sum node on `MeanField()` removes the
  collapse to *prior* moments: it then reads the other inputs' current means
  ([[Composing Stochastic Factors]], E7).
- **Bethe free energy with mixture marginals.** Mycelium has no entropy for a Gaussian-mixture
  variable belief, so a graph's Bethe free energy is available only when every continuous
  variable whose marginal is a mixture has degree one, or is clamped. The factor's own entropy
  is exact.
- **The directed variant** of §8 of the design note (a switching kernel normalised by
  $Z_t(z_t)$, with gates that depend on the state) is a different factor and is not here.
- An uninformative continuous input gives an uninformative continuous message, as for
  `LinearConstraintFactor`. For the outcome it gives the factor's own weights when that
  channel's `A` has full row rank (exact), and `TrivialBelief()` otherwise.

## 6. Difficulties met

- **Rank-deficient precisions round to positive definite.** With one equation and a 2-D target,
  $A_t^\top R^{-1} A_t$ has rank one but passes `isposdef` after rounding often enough. Mycelium then
  treats the components as normalised Gaussians of enormous variance and weights them by
  $\tfrac12 \eta^\top \Lambda^{-1} \eta$, which is rounding noise. The test caught it: the pooled weights were off by an
  order of magnitude. The factor now zeroes the null-space eigenvalues of a rank-deficient
  precision, and pushes them to $-\epsilon\, \lambda_{\max}$ if the result still factorises, so the
  components stay improper. The change is at rounding level.
