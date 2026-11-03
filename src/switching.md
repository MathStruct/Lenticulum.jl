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
  ([[Composing Stochastic Factors]], E7). Since 2026-11-03 a node on `MixtureReduction(M)` expands
  mixture inputs instead, exactly, and reduces its outgoing message to `M` components (§9).
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


## 9. Mixture inputs expanded (2026-11-03)

`Mycelium.mixture_message(f, target, ...)` is the exact message when inputs are Gaussian mixtures,
for `MixtureReduction(M)` nodes (Extending Mycelium §4, [[L-SPEC-008 Expectation Propagation and Mixture Reduction]]).
Each mixture input $\sum_j v_{ij} \mathcal N(m_{ij}, S_{ij})$ is a hidden index; a combination $c$ of
components gives $R_c = Q + \sum_{i \ne t} A_i S_{i c_i} A_i^\top$ and $d_{kc} = b_k - \sum_{i \ne t} A_i m_{i c_i}$, and carries the
evidence $\prod_i v_{i c_i}\, \mathcal N(d_{kc}; 0, R_c)$. Unlike the components of `factor_message`, these differ in
$R_c$, so the weight keeps $-\tfrac12 \log\det R_c$, and a proper component, normalised, also
$\tfrac12 \eta^\top \Lambda_c^{-1} \eta - \tfrac12 \log\det \Lambda_c + \tfrac d2 \log 2\pi$. Improper components stay unnormalised
exponentials, as in `factor_message`.

Flat directions of the non-mixture inputs are integrated out as there (`_residual_metric`), and
the weight then needs the finite part of the diverging normaliser. For a prior $\mathcal N(0, \tau \Sigma_0)$
along the flat coefficients $G$,
$\log\det(R + \tau G \Sigma_0 G^\top) = \log\det R + \rho \log \tau + 2\sum_{i \le \rho} \log s_i + \log\det(V_\rho^\top \Sigma_0 V_\rho) + o(1)$, with $s_i$
and $V_\rho$ the nonzero singular values and right singular vectors of $L^{-1} G$ ($R = LL^\top$). The span of
$V_\rho$ is the row space of $G$ whatever $L$ is, so the last term is the same for every combination,
and `_residual_metric_logdet` returns $\log\det R_c + 2 \sum_i \log s_i$. Mixture components themselves must be
proper; a flat direction inside one would change $G$ between combinations.

`LinearConstraintFactor` is the case $K = 1$, $b_1 = c$ (`constraint.jl` calls the same kernel). Tests
(`test/mixture_rules.jl`): log-density differences of the message against the pointwise sum over
components and outcomes to $10^{-12}$, to a continuous channel and to the outcome, for both factors;
with a flat input beside the mixture, against a prior of variance $10^8$ to $10^{-6}$; and equal to
`factor_message` without mixture inputs. On a switching linear chain the reduction budget then
interpolates between assumed-density filtering and the exact posterior (`passing.md`).

## 10. Plates of switching factors (2026-11-03)

Two batched methods for a plate of tied switching factors (L-ADR-0010, `blocks.md`):

- `factor_messages` on blocks, **to the outcome**, when the continuous inputs are points, trivial,
  or proper Gaussians with one shared precision: one metric $T$, then
  $\log P_i(k) = \log w_k - \tfrac12 \lVert T(b_k - s_i)\rVert^2$ normalised per column, as a `CategoricalBlock`. The squared
  norm is taken of the difference, not expanded into $\lVert Tb_k\rVert^2 - 2\,(Tb_k)^\top Ts_i + \lVert Ts_i\rVert^2$, which would
  cancel for data far from the origin. To a continuous channel the message is a mixture and goes
  replica by replica.
- `factor_beliefs`, `belief_free_energies` and `local_free_energies` through `FrozenSwitchingBeliefs`:
  with the continuous messages as point blocks or shared-precision Gaussian blocks and the
  outcome's as a categorical or trivial block, the belief precision, covariance and $H(x \mid k)$
  are computed once; per replica the means of every outcome and the posterior $q_i$, as
  `switch_statistics` computes them one at a time. `belief_free_energies` is written without
  mutation for Zygote.

`test/plates.jl` ("Plates of switching factors on message blocks") checks both against the same
messages per edge: a mixture over observed points (40 replicas), and a latent mixture on
expectation-propagation nodes (the outcome and $z$ latent, $x = z + \delta$ observed): messages to the
sequential schedule's within $10^{-10}$, the free energy to $10^{-12}$ relative, each graded term to
$10^{-12}$, the gradient to $10^{-10}$ with ForwardDiff and Zygote and $10^{-6}$ with reference differences.
Timings (`bench/mixture_plates.jl`, `blocks.md`): at $N = 1000$ the E-step takes 0.10 ms on blocks
against 9.7 ms edge by edge and 0.044 ms by hand, $F$ 1.1 ms against 5.1 ms, $\nabla F$ 2.9 ms against
29 ms.

The first version folded the point channels with `foldl(...; init)`; Zygote's pullback of that
fails on an empty list (a plate whose continuous channels are all latent), so it is a `reduce`
behind an emptiness check.

## 11. Review of 2026-11-03 (Gemini, `agy`)

Gemini 4 Argon reviewed the diff of 703850a5 (the §4 rules and §9–§10), from the diff alone. Each
claim was checked against the code:

- **Valid, fixed.** The batched beliefs held `Float64` fields, so the scalar free energy on
  blocks could not be differentiated in `ps` with ForwardDiff (`MethodError: no method matching
  Float64(::Dual)`), while the same messages per edge could. `FrozenSwitchingBeliefs` and
  Lenticulum's `FrozenGaussianBeliefs` are parametric now. The second came with L-ADR-0010: the
  factor-analysis plate failed the same way with the committed `gaussian.jl`. `free_energy_gradient`
  was not affected (it builds the beliefs outside AD). `test/plates.jl` now compares the ForwardDiff
  gradient of the free energy on blocks and per edge, for both plates.
- **Valid, fixed.** `_expanded_message` with a target the factor cannot reach and every outcome
  impossible returned `TrivialBelief()` instead of refusing, as `factor_message` does. Found
  with it: `_logsumexp` of all `-Inf` is `NaN`, so an outcome of weight zero gave `NaN` in the
  message to the outcome; it is `-Inf` now (tested).
- **Valid, fixed** (Mycelium): `_indefinite` called `eigvals` on ForwardDiff duals; the sign test
  now runs on values. `ep_project` reuses the Cholesky factor of its positivity test for the
  precision.
- **Not bugs.** The claimed dependence of the flat-direction normaliser on the combination (the
  coefficients $G$ come from non-mixture inputs only, the same in every combination, and
  $\operatorname{rank} L^{-1}G = \operatorname{rank} G$); "the batched free energy omits the incoming-message terms" (it is
  the per-replica `local_free_energy`, term for term, which the tests compare to $10^{-12}$); the
  re-centring shift in `_improper_overlap` (a reference point is used only when the other side
  is a single component, so the shift is one constant); an improper first operand with a proper
  second still throwing (documented; `_mixture_product` puts the proper one first); outcome
  labels dropped by the outcome block (the per-replica message drops them too).

Mistral (`vibe`) reviewed the same code before the commit and found the missing `switch` check in
the outcome block method, fixed in 703850a5; its other points were not bugs (`passing.md`).

## 12. Mixture messages as one block (2026-11-03)

`_switch_mixture_block`, called by the block method of `factor_messages` for a continuous target:
with the other continuous inputs as for the outcome method (points, trivial, or proper Gaussians
of one precision), the outcome's input trivial, absent or a `CategoricalBlock`, and $K > 1$, the
messages of all replicas are one `MixtureBlock`: one metric $T$ and one precision
$(TA_t)^\top(TA_t)$, $\eta_{ki} = (TA_t)^\top T(b_k - s_i)$, and the weights of `_switch_message`
($\log \pi_{ki} + \log w_k - \tfrac12\lVert T(b_k - s_i)\rVert^2$, plus $\tfrac12 \eta^\top \Lambda^{-1} \eta$ for proper components). An
outcome impossible in some replica sends the group to the per-replica path, since
`_switch_message` drops it and the replicas would differ in $K$; a target the factor cannot reach
gives a `TrivialBlock`. Tested in `test/plates.jl` ("Mixture blocks") against the sequential
schedule; timings in `blocks.md`.

## 13. Cluster forms (2026-11-03)

`Mycelium.cluster_form` and `cluster_switch` for `SwitchingConstraintFactor` (the residual
$\sum_i A_i x_i - b_k$, noise $Q$, the normalised log-weights, the switch channel), `LinearConstraintFactor`
($K = 1$, $b_1 = c$) and `GaussianPrior` without learnable noise ($x - \mu$, noise $\Sigma$) are defined
here, so these factors can be put in a cluster node (`lib/Mycelium.jl/src/cluster.md`,
[[L-SPEC-009 Cluster Nodes]]). Three switches in one cluster send the messages of the 8-outcome
block factor that E7 built by hand (`test/cluster.jl`).
