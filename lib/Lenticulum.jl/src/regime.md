#implementation

> `RegimeFactor`: a switching linear constraint whose coefficients and noise depend on the
> outcome as well as its right-hand side, $0 = \sum_i A_{i,k} x_i - b_k + \varepsilon_k$, $\varepsilon_k \sim \mathcal N(0, Q_k)$,
> $P(k) = w_k$. It is stated only through a cluster form with per-outcome $A$ and $Q$, exact inside a
> cluster node, and outside one it is a one-factor cluster.

> Sources: code: `regime.jl`, `switching.jl`; Mycelium `lib/Mycelium.jl/src/cluster.jl` (`as_cluster`, per-outcome forms); tests: `test/regime.jl`
>
> Theory (CT-ML wiki): [[Switching Constraint Factors]], [[Composing Stochastic Factors]]

## 1. The factor

$$
f(x_1, \ldots, x_n, k) = w_k\, \mathcal N\Bigl(\textstyle\sum_i A_{i,k} x_i - b_k;\ 0,\ Q_k\Bigr), \qquad k \in \{1, \dots, K\}.
$$

`SwitchingConstraintFactor` (`switching.md`) is the case $A_{i,k} = A_i$, $Q_k = Q$, the
"location-shift" case, and listed the general one under what it does not do: with $A_k$ or $Q_k$
depending on the outcome the outcomes' Gaussians have different precisions. Instances are
mixtures of regressions (experiment E1's two crossing lines, $A_{z,k} = [-a_k\ \ 1]$), Gaussian mixtures
with one covariance per component, mixtures of factor analysers and switching linear dynamics
with one transition matrix per regime.

The parameters are `A` (a `NamedTuple` of $m \times d_i \times K$ arrays), `b` ($m \times K$) and `logw`. The noise is
fixed (one covariance per regime, or one shared), or with `learn_noise = true` a parameter `L`
($m \times m \times K$) with $Q_k = L_k L_k^\top$, $L_k$ the strictly lower triangle of `L[:, :, k]` plus the exponential of its
diagonal; the strict upper triangle is unused and gets a zero gradient. `regime_covariances(f, ps)`
returns the $Q_k$.

**Output channel.** `output = :y` fixes the coefficient of `y` at `output_coeff` in every regime and
drops it from the parameters. The factor is then a normalised conditional density of `y`, a mixture
of regressions. Without it, learning every coefficient has no lower bound with `learn_noise`
(LP-022, `test/lp022.jl`).

## 2. How it is computed

The factor implements only `Mycelium.cluster_form`: residual $\sum_i A_{i,k} x_i - b_k$ with
per-outcome `A` and `Q` arrays (`cluster.md`, "Per-outcome coefficients and noise"). That gives:

- **Inside a cluster node**, exact inference: assignments are grouped into precision classes,
  one per combination of the varying factors' outcomes, and each class is one Gaussian
  elimination.
- **Outside a cluster**, its messages, belief and Bethe free energy are those of the
  one-factor cluster `Mycelium.as_cluster(f)` over its own channels (parameters wrapped as
  `(factor = ps,)`). The message to the outcome is $\propto w_k \int f$; to a continuous channel a
  mixture of $K$ Gaussians with precisions $\Lambda_k = A_{t,k}^\top \bar Q_k^{-1} A_{t,k}$ after the other inputs are
  eliminated. The free energy is the cluster's: $\mathbb E_b[-\log f] - H(b)$ with $b$ the factor times its
  incoming messages, closed form given the outcome.

## 3. What it does not do

- **Flat inputs** are integrated out in the diffuse limit since 2026-11-04 (`cluster.md`, "Flat
  directions"), provided every regime leaves equally many directions flat; otherwise the message
  is refused. A mixture input is collapsed to its moments.
- **Improper messages with several precisions** are sent (since 2026-11-04; refused before): each
  improper component is the unnormalised exponential its weight multiplies, which is how
  Mycelium's products read it, so components with different precisions are weighted
  consistently (`test/cluster_flat.jl`).
- **No mean-field, mixture-expansion or EP rules**, no plate blocks: inside a plate its cluster
  nodes take the per-replica path (`cluster.md`).
- **No game interface** (`assemble`, `invert`): only the Mycelium factor methods.

## 4. Checked by

`test/regime.jl`:

- a mixture of factor analysers ($x \sim \mathcal N(0, I)$, three regimes with different $W_k$, $\mu_k$, $Q_k$) in
  one cluster: $F = -\log p(z)$ to $10^{-12}$ relative, the regime posterior and each regime's
  posterior mean and covariance of $x$ to $10^{-12}$;
- the free-energy gradient in $A$, $b$, $\log w$ and $L$ against central differences of $-\log p(z)$, to
  $10^{-6}$, with ForwardDiff and Zygote;
- outside a cluster: a Gaussian mixture with per-component covariance (free energy, message to
  $k$, gradient in $b$ against the closed form), the message to a continuous channel (each
  component's $\Lambda_k$, $\eta_k$ and weight against the closed form), the refusal of improper components
  with different precisions, and the message to $z$ with per-regime noise;
- over a plate of four points: the per-replica path, $F = -\sum_i \log p(z_i)$, the gradient in $b$.

`examples/regime_lines.jl` (`regime_lines.md`): E1's crossing lines with one noise level per line,
300 points, fitted by the free-energy gradient on the graph, reaches EM's maximum likelihood to
$2 \cdot 10^{-8}$ in $-\log p$; one shared noise level is 53.5 worse.

## 5. Difficulties

1. **A free-energy check that could not be made.** A first test put the prior of $x$ outside the
   cluster and compared the graph's Bethe free energy with $-\log p(z)$. The cluster's message to $x$
   is a mixture, and Mycelium has no entropy for a mixture variable belief (`switching.md` §5),
   so the graph has no free energy there. The message is tested directly instead.
2. **Zygote and generators.** The covariances were first built from a generator splatted into
   `cat`, which Zygote has no adjoint for; a `map` works.
3. **An ambiguous `noise`.** A real vector is a diagonal (as for the other factors), so
   per-regime noise is a vector of matrices or an $m \times m \times K$ array, not a vector of scalars.
