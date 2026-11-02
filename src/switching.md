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

Any assignment with one channel unobserved, the outcome included, and the others observed or
latent: `supported_polarities` lists all $n \cdot 2^{n-1}$ of them, the set `supports_polarity`
accepts (L-SPEC-001), in any key order. A latent outcome is marginalised under the factor's own
weights; a latent continuous channel is integrated out under a flat prior (§5). Both are graph
variables summed or integrated over by the message, not a latent space of the open model. The open model is pure and its inversion exact, as
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

**Mean-field energy (2026-11-02).** The node's term in the hybrid free energy is
$\sum_k q_k(\tfrac12\mathbb E\lVert r_k\rVert^2_{Q^{-1}} - \log w_k)$ plus the normaliser, under
independent marginals on the continuous channels and the outcome ($q$ from a categorical or point
marginal), with no entropy ([[L-SPEC-005 The Hybrid Free Energy]]). It reuses `belief_free_energy`
with the product belief (one mean column per outcome) and drops the negentropy, which would
otherwise include $H(q)$. Checked against Monte Carlo in `test/mean_field_free_energy.jl`.

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
- **Flat and improper inputs are handled, exactly** (the bullet is kept for the history). An
  uninformative continuous input (no message, `TrivialBelief()`, `Latent()`) is integrated out
  under a flat prior: $R^{-1}$ becomes the projection that removes the range of those channels'
  $A$'s (`constraint.md` §4.1, the same helper). Only the part of $d_k$ they cannot reach then
  separates the outcomes. Until 2026-11-01 the factor returned `TrivialBelief()` here unless that
  channel's `A` had full row rank, and it looked only at the first such channel. Since
  2026-11-02 a Gaussian input that is improper in some directions only is split, too: its
  proper part enters $d_k$ and $R$, its null space joins the flat directions.

## 6. Difficulties met

- **Rank-deficient precisions round to positive definite.** With one equation and a 2-D target,
  $A_t^\top R^{-1} A_t$ has rank one but passes `isposdef` after rounding often enough. Mycelium then
  treats the components as normalised Gaussians of enormous variance and weights them by
  $\tfrac12 \eta^\top \Lambda^{-1} \eta$, which is rounding noise. The test caught it: the pooled weights were off by an
  order of magnitude. Until 2026-11-02 the factor zeroed the null-space eigenvalues of a
  rank-deficient precision and pushed them below zero, starting at $-\epsilon\, \lambda_{\max}$, until
  the result no longer factorised. That made the precision indefinite at rounding level, which
  a 2026-11-01 review flagged. Now the factor decides properness by the rank of the projected
  $A_t$ and declares it (`GaussianBelief(η, Λ, true)`, LenticulumCore `gaussian_belief.md`,
  "Declared improper"); $\Lambda = (TA_t)^\top (TA_t)$ is passed as computed, a Gram matrix, with no
  repair. The mean-field message and `LinearConstraintFactor`'s message with flat inputs declare
  it the same way. The review's case, a proper neighbour with precision below $\epsilon\, \lambda_{\max}$ in
  a null direction, turned out to need a condition number above $1/\epsilon$ in the pooled
  precision: no dense Float64 representation holds that, flag or no flag. The test
  ("rank-deficient components are declared improper") pools with a prior of condition number
  about $2 \cdot 10^9$ and matches Gaussian conditioning outcome by outcome.

## 7. The parameter gradient holds the belief fixed (LP-020, 2026-11-01)

`Mycelium.factor_belief` freezes the whole belief $b_c(x, k)$: the outcome posterior $q$, one mean
per outcome, and the shared covariance. `belief_free_energy` re-evaluates
$\sum_k q_k\bigl(\tfrac12 \mathbb E\|r_k\|^2_{Q^{-1}} - \log w_k\bigr) + \tfrac12\log\det 2\pi Q$ at the current $A$, $b$ and
log-weights, with the entropy $H(q) + H(x \mid k)$ constant. So the log-weights' gradient is
$-(q - w)$, the EM M-step direction, and $A$ and $b$ get theirs at fixed responsibilities. This
matches finite differences of the converged Bethe free energy in `test/lp020.jl`. Construction:
`gaussian.md`, "The factor belief, held fixed".

## 8. Review of 2026-11-01

ChatGPT, Gemini and Mistral reviewed `switching.jl` on `master` (601b08d9). Each claim was
checked against the code; the fixes are tested in `test/switching_constraint.jl` (the last
three testsets).

| claim | verdict | what changed |
|---|---|---|
| A flat input with rank-deficient `A` discards information (ChatGPT 1, Gemini C, Mistral 2) | right; ChatGPT's example gives $P(k{=}2) = 0.000335$, not uninformative. The answer does not depend on how the flat prior is regularised, contrary to Mistral's aside | flat channels are projected out together (§5); also in `LinearConstraintFactor` |
| The check looks only at the first flat channel (Gemini C, Mistral 2) | right | all flat channels are stacked |
| A flat input returns before an invalid outcome belief is checked (Gemini C) | right | the outcome belief is read first |
| The polarity list misses the latent assignments (ChatGPT 3) | right, and `LinearConstraintFactor` too | both list $n \cdot 2^{n-1}$, capped at six channels; keys compared as sets |
| `_singular_precision` makes the precision indefinite (ChatGPT 2) | right, at rounding level; but the case it hurts needs a condition number above $1/\epsilon$, beyond any dense Float64 precision | 2026-11-02: replaced by a declared-improper flag on `GaussianBelief` (§6) |
| `_singular_precision` can still leave it positive definite (Mistral 5) | right in principle | the shift grew until it did not; since 2026-11-02 gone with the flag |
| The mean-field message lacks the repair (all three) | no: it is one Gaussian, so no weight depends on properness | a comment says so |
| Latent channels and `ispure` (ChatGPT 4) | a documentation gap: they are graph variables, not the model's latent space | docstrings say so |
| `entropy` stores $+H$ under `negentropy` (Gemini B) | no: the documented convention of `LenticulumCore.entropy`, shared by every Gaussian factor | none |
| The factor has no tests (Mistral 1) | no: `test/switching_constraint.jl` is in `runtests.jl` | none |
| An all-zero outcome belief gives NaN in `switch_statistics` (Mistral 4, Gemini 2.2) | right | it throws, as the message does; the free energy checks `q` is finite |
| A non-integer outcome throws `InexactError` (Mistral) | right | `ArgumentError` with the valid range |
| A second Cholesky of $2\pi Q$; the trace term inside the sum; two SVDs; `C \ D` solved twice (Gemini) | right | one factorisation, the trace outside, one rank, one solve; the two free energies share `_switch_free_energy` |
| `all` shadows `Base.all`; `Ηs` is a Greek capital eta (Mistral) | right | renamed (`etas`) |
| `factor_belief` recomputes what `switch_statistics` already has (Gemini 3) | right, a cost only | not changed |
| `K` and `m` are adjacent positional `Int`s (Mistral 7) | a fair API point | not changed: it would break callers |
| The pointwise energy is a scalar with the normaliser, unlike the siblings (Mistral 6) | right, and already stated in §4 | none |
| Moment matching makes "exact" conditional (ChatGPT 5) | already stated in §5 | none |

