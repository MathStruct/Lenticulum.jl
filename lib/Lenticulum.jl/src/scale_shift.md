#implementation

> `ScaleShiftFactor(x, z, dim; groups)` (`scale_shift.jl`): the normalising change of variables
> $z = (x - b) \oslash s$, exactly, as the density $\delta(x - s \odot z - b)$ of $x$ given $z$, with its
> log-Jacobian $\sum_i \log s_i$ in the free energy and a parameter gradient that is the free
> energy's. Standardising data channels (`scale_shift_parameters`) and Glow's ActNorm. Backlog row U3.

> Sources: Kingma & Dhariwal, *Glow: Generative Flow with Invertible 1x1 Convolutions*, NeurIPS 2018, §3.1 (ActNorm, data-dependent initialisation); Yedidia, Freeman & Weiss, *Constructing free-energy approximations and generalized belief propagation algorithms*, IEEE Trans. Inf. Theory 51 (2005) (the Bethe free energy and BP fixed points). code: `scale_shift.jl`; `hard_constraint.jl` (messages, Bethe term); `lib/MyceliumLearning.jl/src/gradient.jl` (the envelope gradient). tests: `test/scale_shift.jl`.
>
> Bibliography: [[Bibliography#^kingma2018glow|Kingma & Dhariwal 2018]] · [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]]
>
> Theory (CT-ML wiki): [Gaussian Relations](https://mathstruct.org/CategoryTheory-ML-Wiki/Gaussian-Relations)

## Theory

The factor is $f(x, z) = \delta(x - s \odot z - b)$, a conditional density of $x$ given $z$ relative to
Lebesgue measure on $x$, with $s = \exp(\text{logs})$ and $b$ one entry per group of coordinates. With
$x$ observed at $x_0$ and a density $p_z$ on $z$,

$$
p(x_0) \;=\; \int p_z(z)\, \delta(x_0 - s \odot z - b)\, dz \;=\; \frac{p_z\bigl((x_0 - b) \oslash s\bigr)}{\prod_i s_i},
$$

the change-of-variables formula. The other reading, $\delta(z - (x - b) \oslash s)$, a density of $z$
given $x$, has no $\prod_i s_i$: with it a learned scale shrinks $x$ to raise the density of $z$ and
the free energy falls without fitting (the collapse of LP-022).

It is `HardConstraintFactor`'s relation $x - \operatorname{diag}(s)\, z = b$ ($A_x = I$,
$A_z = -\operatorname{diag}(s)$, $c = b$). Its messages are the pushforwards through the two affine
maps, and its Bethe term is $\tfrac12 \log\det(\tilde G \tilde G^\top) - H_R(b_f)$
(`hard_constraint.md`, "Free energy"). With $x$ a point, $\tilde G = -\operatorname{diag}(s)$ and the
term is $\sum_i \log s_i$, the log-Jacobian. Without hard directions in the inputs, $R$ is the graph
of $z \mapsto s \odot z + b$, $\tfrac12\log\det(GG^\top) = \tfrac12 \sum_i \log(1 + s_i^2)$ and
$H_R = H(q_z) + \tfrac12 \sum_i \log(1 + s_i^2)$, so the term is $-H(q_z) = -H(q_x) + \sum_i \log s_i$.

## The parameter gradient

Mycelium's envelope gradient differentiates each factor's term at its belief held fixed
(`gradient.md`). A deterministic factor has no such derivative: its belief lives on the relation,
which moves with $(b, s)$, and with $x$ observed it drags the belief of $z$ along, a point
$z^* = (x_0 - b) \oslash s$. The fallback, the local term at fixed incoming messages, gives only the
log-Jacobian's derivative: on a three-dimensional chain (prior on $z$, $x$ observed) it returned
$(0, 0, 1, 2)$ for $\partial(b, \text{logs})$, where the evidence's gradient is
$(-0.28, -3.17, 0.89, -11.09)$ (`ScaleShiftFactor`'s first prototype, 2026-11-20).

What is right at fixed messages is the factor's term in the message form of the Bethe free
energy, $-\log Z_f(m, \theta) = -\log \int f_\theta\, m_x m_z$: that form is stationary in the messages at
a fixed point, and the variable and edge terms do not depend on $\theta$, so

$$
\frac{dF}{d\theta} \;=\; -\,\partial_\theta \log Z_f(m^*, \theta).
$$

For a factor with a density this is $\mathbb E_{b_f}[-\partial_\theta \log f]$, the envelope gradient
(LP-020's). For this factor, $Z_f = \int m_x(s \odot z + b)\, m_z(z)\, dz$ is finite wherever the free
energy is. The code uses the identity

$$
-\log Z_f \;=\; F_f \;+\; \mathbb E_{b_f}\bigl[-\log m_x(x) - \log m_z(z)\bigr] \;+\; \text{const},
$$

where only the Gaussian directions of each input count: $\tfrac12\operatorname{tr}(\Lambda C) +
\tfrac12 \delta^\top\Lambda\delta$, with $\Lambda$ the input's precision on them, $(\mu, C)$ the factor
belief's marginal and $\delta = \mu - \mu_m$ its offset from the input's mean. The first version
used the uncentred $\tfrac12\mu^\top\Lambda\mu - \eta^\top\mu$. At a mean of $10^8$ the change of
the energy was lost to cancellation, and the default gradient came out as $5\cdot10^5$ for $-1$
([[LP-070 Findings of the review of U3]], finding 3). A point, a hard direction and a flat direction of an input add a constant.
`factor_belief` returns the messages and the value of that energy at the current parameters
(`ScaleShiftBelief`). `belief_free_energy` returns $F_f$ plus the energy minus that value, inside
`fit`. Its value is $F_f$, and its derivative at the fixed messages is $-\partial_\theta \log Z_f$.

On the prototype chain this gives the evidence's gradient to $10^{-15}$ with ForwardDiff and
Zygote, and to $10^{-9}$ with the default finite differences. The tests check it on random trees
(`test/scale_shift.jl`).

## Implementation

The messages map the input's mean, covariance and flat directions coordinatewise
(`_scale_shift_message`): $\mu \mapsto (\mu - b) \oslash s$, $\Sigma \mapsto S^{-1}\Sigma S^{-1}$,
$D \mapsto S^{-1} D$ towards $z$, and the inverse towards $x$. The `ExtendedGaussian` constructor
renormalises the result. An invertible diagonal map is exact on all three, in $O(d^2)$. The first
version went through `HardConstraintFactor`'s pushforward and preimage. Their eigenvalue cuts made
a Float32 variance of $10^{-6}$ next to 1 hard in one direction and flat in the other (LP-070,
finding 2). The Bethe term and the mean-field refusal are `HardConstraintFactor`'s. The factor
holds one, `HardConstraintFactor((x = d, z = d), d)`, builds its parameters $(A, c)$ from
$(b, \text{logs})$ on each call (`_scale_shift_relation`, without mutation so that ForwardDiff and
Zygote go through), and rebuilds it with a zero noise of the parameters' float type for Float32
(the noise sets that factor's element type). `scale_shift_parameters(f, X)` sets $b$ to each group's mean and $s$ to its standard
deviation (normalised by the count) over the data `X` (`dim × N`). It works in floating point from
the start, scales the deviations by their largest before squaring, and refuses a group whose
entries are all equal (LP-070, finding 5). For i.i.d. data under a standard
normal prior on $z$ these are the maximum-likelihood estimate.

## What it does not do

- **Cost.** The messages are $O(d^2)$ plus the `ExtendedGaussian` constructor's normalisation.
  The free energy and the gradient go through the general `HardConstraintFactor` path (SVDs and
  eigendecompositions of $2d$-dimensional relations). Not measured.
- **Limits met through it, not changed** (LP-070):
  - **Converting an input.** A proper Gaussian whose precision has an eigenvalue of $10^{-16}$
    next to 1 becomes flat in that direction when converted to an extended Gaussian.
  - **Float32 extended Gaussians.** A Float32 extended Gaussian takes a variance below
    $32\,\varepsilon_{32}$ of the largest for a hard direction (LP-069). A Float32 tree with
    $s = 10^{-3}$ and unit priors on both sides gets wrong marginals and free energy, though its
    messages are right.
  - **Mixtures of points.** `moment_match` has no method for a mixture of `DiracBelief`s, which
    both this factor and `HardConstraintFactor` call.
  - **The default gradient.** `ad = nothing` takes central differences with a fixed step. It is
    off by 2–5 % on Float32 data and by $2\cdot10^{-3}$ at a mean of $10^8$, where ForwardDiff and
    Zygote are exact.
- **Two exact observations of one variable** (fixed by I26, 2026-11-20). Two oblique exact
  projections of one variable gave the right marginals and a wrong Bethe free energy, with or
  without this factor (LP-070, finding 1). The variable's term now adds the overlap of the hard
  messages (Mycelium `free_energy.md`, "Exact messages that meet at a variable"), and a message of
  this factor meeting another exact one is tested with its gradient (`test/hard_overlap.jl`).
- **Plates.** No batched `factor_messages`; a plate calls `factor_message` per replica. Shared
  parameters across replicas work through `params = :key`.
- **Implicit adjoint.** `adjoint = :implicit` refuses graphs with extended-Gaussian messages
  ("no coordinates for ExtendedGaussian"), so the factor learns only through the envelope gradient.
- **Negative scales.** $s = \exp(\text{logs}) > 0$. A reflection is not a parameter.
- **Shared statistics.** `scale_shift_parameters` is a one-time initialisation, not batch
  normalisation's running statistics; the probabilistic version of those is row U4.

## Element types (Q4)

Data-dependent initialisation allocates its result in the input matrix's array family and
floating-point type. Energy under the factor belief starts from zero in the promoted type
of the parameters and incoming beliefs. The parameter initializer without data keeps its
Float64 default. The exact relation's general spectral path has the same BigFloat limit
as the hard constraint.

## Device arrays (Q7)

Adapt moves the stored hard relation and keeps the group indices on the host. The identity and
diagonal coefficients are built alongside the shift and scale parameters.

The frozen input energy retains its working scalar type after stopping the
gradient. A ForwardDiff value therefore carries a zero partial in the frozen
energy. The single and batched frozen beliefs are checked in `test/eltypes.jl`.

## LP-094 part 2 (2026-11-27, element types from the review of Q19)

**Finding 7.** `_scale_shift_map` took its type from `float(promote_type(eltype(ps.b),
eltype(ps.logs)))`, so integer `b`/`logs` widened a Float32 call to Float64. It is now
`_scale_shift_map(f, ps, inputs...)` with `T = _compute_eltype(Float64, ps.b, ps.logs,
inputs...)`, and `_scale_shift_relation`/`_scale_shift_message` pass the messages (resp. beliefs)
through; `local_free_energy` and `mean_field_message` pass theirs. Tests: `test/lp094.jl`,
finding 7.
