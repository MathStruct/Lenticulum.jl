#implementation

> `LinearConstraintFactor`: one equation $0 = \sum_i A_i x_i - c + \varepsilon$ over any
> number of channels, none of them distinguished. ModelingToolkit's `0 ~ ...` with a noise
> term, and the first factor in this project with **no preferred direction at all**.

> Sources: code: `constraint.jl`, `beliefs.jl`, `gaussian.jl`
>
> Theory (CT-ML wiki): [Hypergraph Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Hypergraph-Category) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Graph free-energy edge directions

`LinearConstraintFactor` declares `Mycelium.uses_cavity == true`: its local free energy
uses residual statistics and entropy of the belief formed from incoming messages.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. Why this factor exists

[[README]]'s table promises *"Symmetry handling: symmetric, no distinguished input/output"*.
`GaussianFactor` does not deliver that. It has an `in_channel` and an `out_channel`, it is
written as $y = Ax + b$, and it admits exactly two polarities no matter how you build it.
It is *bidirectional*, which is a weaker property than *acausal*: it can be run backwards,
but it still knows which way is forwards.

This factor delivers it. There are $n$ channels, the equation is a sum over all of them, and
every channel can be the one solved for, given the others: $n$ polarities, one per channel, of
which none is the factor's own. The count is the content: **"acausal" is the statement that the
number of directions equals the number of channels**, rather than being fixed at two by the
factor's constructor. (`supported_polarities` lists $n \cdot 2^{n-1}$ since 2026-11-01: each
other channel may also be latent, and the list must match `supports_polarity`, L-SPEC-001.)

See [[ModelingToolkit as an Acausal Relation]] for where this comes from, and
[[Acausal Composition is a Hypergraph Category]] for why an improper `GaussianBelief` is what
makes it possible.

## 2. It subsumes `GaussianFactor`

$y = Ax + b + \varepsilon$ is $0 = (-A)x + Iy - b + \varepsilon$, so

| `GaussianFactor` | `LinearConstraintFactor` |
|---|---|
| `ps.A` | `ps.A.x = -A` |
| — | `ps.A.y = I` |
| `ps.b` | `ps.c = b` |

and the two agree on **messages in both directions, the pointwise residual, and all three
summands of the graded free energy**. That equality is asserted in the test suite rather
than argued for here, and it is the main reason to trust the n-ary code: it reduces to code
that was already checked against closed forms.

The one formula covering every direction is

$$
d = c - \sum_{i \ne t} A_i m_i,
\qquad
R = Q + \sum_{i \ne t} A_i S_i A_i^\top,
\qquad
\mu_{c\to t} = \mathcal{N}^{-1}\bigl(A_t^\top R^{-1} d,\ A_t^\top R^{-1} A_t\bigr)
$$

The target channel enters *only* as "which $A_i$ is called $A_t$". There is no forward case
and no backward case; `gaussian.jl`'s two branches were an artefact of naming two channels
`in` and `out`.

> [!note] The message is usually improper, and that is not a defect
> $A_t^\top R^{-1} A_t$ has rank at most $m$, the number of equations. A factor with fewer
> equations than unknowns in $x_t$ — the normal case for a connector equation — produces a
> singular precision. Only the canonical form of `beliefs.jl` can write this down; the moment
> form cannot represent it at all. See §5.

## 3. The `energy` signature has a causal bias

`LenticulumCore.energy(factor, x, a, y, ps, st)` takes the arguments in AutoBayes'
$X \times \llbracket c \rrbracket \times Y$ split. An acausal factor has no such split, so
the primary API here is

```julia
residual(f, vals::NamedTuple, ps)     # ALL channels, unsplit
```

and `energy` merges whatever `NamedTuple`s it is handed, throwing a specific error if given
positional arrays. This is a small wart with a real cause:

> The core interface's own `energy` signature presumes a causal factor. It was written
> against Definition 20, where $X$ and $Y$ are given as part of the open model, and an
> acausal factor only acquires an $X$/$Y$ split once a polarity is chosen — which is one step
> *later* than `energy` is called.

The honest fix is an `energy(factor, vals::NamedTuple, ps, st)` method on the interface, with
the four-argument form as a causal convenience. That is a `LenticulumCore` change and is not
made here, because the interface is used by `Mycelium` and by `VariationalDiffusion.jl` and
changing it for one factor is the wrong trade. Recorded rather than fixed.

## 4. Implementation difficulties

### 4.1 A latent channel is integrated out under a flat prior (fixed 2026-11-01)

Until 2026-11-01, if any non-target channel carried no information, `_message` returned
`TrivialBelief()`. The reasoning was that with a flat prior on $x_j$ the direction
$\operatorname{range}(A_j)$ of the residual is unconstrained, so the equation says nothing.

**That was too blunt.** The sharp answer takes the limit $S_j \to \infty$ properly. For a prior
$\mathcal N(0, \tau \Sigma_0)$ on the flat channels, with $G$ stacking their $A$'s,
$(R + \tau G \Sigma_0 G^\top)^{-1}$ tends to the $R^{-1}$-orthogonal projection that removes
$\operatorname{range}(G)$, the same limit for every $\Sigma_0 \succ 0$:

$$
\mu_{c\to t} \;=\; \mathcal{N}^{-1}\bigl(A_t^\top \Pi\, d,\ A_t^\top \Pi\, A_t\bigr),
\qquad
\Pi = R^{-1} - R^{-1}G (G^\top R^{-1} G)^{+} G^\top R^{-1}
$$

This is *not* zero unless $G$ has full row rank. Concretely: take two equations and one free
variable that appears in only one of them; the other equation survives. The normaliser that
diverges with $\tau$ does not depend on $x_t$, so the limit is a proper statement about the
message.

The code computes $\Pi$ through a factor: `_residual_metric` returns $T = U_\perp^\top L^{-1}$
with $R = LL^\top$ and $U_\perp$ an orthonormal basis of the complement of
$\operatorname{range}(L^{-1}G)$, so $T^\top T = \Pi$. When $G$ has full row rank, $T$ has no
rows and the message is `TrivialBelief()`, as before. `_projected_rank` decides whether anything
is left of $A_t$: rounding in the projection leaves singular values of order
$\epsilon \lVert L^{-1}A_t \rVert$, and those count as zero. With no flat channel the code is
unchanged. The mean-field message reads the same rule with the other channels at their means.
Several flat channels are projected out together, so their order does not matter.

The test (`test/switching_constraint.jl`, "flat inputs are integrated out") compares the flat
message with the message for a proper input of variance $10^7$, to within $10^{-5}$. The same
helper serves the switching factor (`switching.md` §5).

This is the "marginalise a latent channel" operation the vault keeps deferring
([[Open Models and Latent Channels]] §"marginalisation is the expensive one"), here specialised to
the linear-Gaussian case, where it is a pseudo-inverse rather than an integral.

**Partially improper inputs (2026-11-02).** A Gaussian input $\mathcal N^{-1}(\eta_i, \Lambda_i)$ with a singular
$\Lambda_i$ is the same limit in some directions only. `_split_moments` (`gaussian.jl`) splits it by
the eigendecomposition of $\Lambda_i$: on its range, a proper part with mean $\Lambda_i^{+}\eta_i$ and
covariance $\Lambda_i^{+}$, which enters $d$ and $R$ as before; along its null space $N_i$, a flat
part, whose coefficient $A_i N_i$ joins $G$. Eigenvalues below $16 n \epsilon\, \lambda_{\max}$ count as zero,
which also catches the rounding of a declared-improper site. Two inputs are refused, since
neither is a limit of Gaussians: a precision with a negative eigenvalue beyond that tolerance,
and an $\eta_i$ with a component along $N_i$ beyond $\sqrt\epsilon$ relative ($\exp(\eta^\top x)$ along a flat
direction). The mean-field message reads the proper part's mean and treats $N_i$ the same way.
The test ("a partially improper input is split") compares with a proper input whose variance
along $N_i$ is $10^7$, in an axis-aligned and a rotated basis.

A message whose (projected) $A_t$ is rank-deficient is declared improper
(`GaussianBelief(η, Λ, true)`), with or without flat directions; the rank is computed on values
(`stop_gradient`), so the derivative of the message is unchanged. With flat directions the
projection needs an SVD, which ForwardDiff cannot differentiate: it then fails with a
`MethodError` rather than give a derivative that ignores how the flat directions move. No
caller differentiates through such a message today (the full test suites pass); a derivative
would need the projection written through a full-column-rank basis of $G$.

### 4.2 A loopy acausal graph cannot start itself

Every message needs every *other* channel to be informative. On a cycle with no priors, sweep
one produces `TrivialBelief` everywhere, and so does every sweep after it. The graph
deadlocks — not with an error, but with a converged-looking fixed point of no information.

The test suite works around this the way practitioners do: a weak regularising prior on every
variable, which is enough to bootstrap and which the joint oracle also carries so the
comparison stays honest. But it is a workaround, and it points at the real difference from
ModelingToolkit: **MTK never propagates. It assembles the whole system and solves it.** A
linear acausal system is one linear solve; belief propagation is a fixed-point iteration
that, on this class of problem, is strictly worse. See
[[ModelingToolkit as an Acausal Relation]] §6.

### 4.3 On a cycle the variances are wrong even when the means are exact

Asserted in the test suite on the resistive divider: Gaussian loopy BP reproduces the exact
means to machine precision and gets every variance wrong by a common factor (about $5\times$
too large, with weak priors). That is Weiss & Freeman's theorem, and the direction is
model-dependent — a negative cycle gain inflates rather than deflates.

The practical consequence for an acausal library is severe and worth stating plainly:

> [!warning] Point estimates survive the cycle; uncertainty does not
> If you use this factor to build a circuit and read off currents, the numbers are right.
> If you read off error bars, they are wrong, and nothing in the API tells you so. The
> `ConvergenceReport` reports that the *messages* stopped moving, which is a different claim
> from the marginals being correct.

See [[Loopy Message Passing]].

### 4.4 Dirac channels are substituted, not represented

`residual_statistics` moves a clamped channel into the constant, $c \mapsto c - A_j v_j$, and
drops it from the joint. This is what keeps the joint precision finite (a Dirac is
$\Lambda = \infty$) and it makes the all-clamped case fall out of the same code rather than
needing its own branch — `gaussian.jl` has three hand-written branches for what is one loop
here. The cost is that the code path depends on the *runtime type* of each incoming message,
so it is not type-stable across belief types.

### 4.5 `Q` is a hyperparameter, and the hard relation is unreachable

$Q \to 0$ recovers the hard relation $\{x : \sum_i A_i x_i = c\}$, which is what
ModelingToolkit actually means by an equation. The code cannot go there: `Q` must be positive
definite (checked in the constructor), and the tests use $10^{-3}$ or $10^{-4}$ as "hard".
Learning `Q` has the same blocker as in `gaussian.jl` — it needs a positive-definite
parametrisation.

There is a real object at the limit, and it is not a distribution: it is a *linear relation*,
represented by a subspace. The Gaussian-relations literature handles both at once
([[Acausal Composition is a Hypergraph Category]] §4); this file handles only the interior.

### 4.6 `ps.A` is a `NamedTuple` of matrices, which the optimiser story has not met yet

`initialparameters` returns `(A = (x = ..., y = ...), c = ...)` — one level deeper than
`GaussianFactor`'s `(A = ..., b = ...)`. `LuxCore.parameterlength` handles it, but
`Mycelium`'s `OptimiserFactor` has only ever seen flat parameter trees. Nothing is known to
be broken; nothing has been tested either.

## 5. Why the canonical form is load-bearing here

`beliefs.jl` argues for canonical form on two grounds: pooling becomes addition, and
likelihoods with rank-deficient precision become representable. The second one is what this
file lives on.

A connector equation — Kirchhoff, a mass balance, an interface constraint — is one scalar
equation among $n$ variables. Its message to any one of them has precision of rank 1 in a
space of dimension $\dim x_t$. In moment form that is a covariance with infinite eigenvalues,
i.e. not a covariance. Every acausal message is of this kind, so an acausal factor is
*impossible* in a moment-form library and costs nothing extra in a canonical-form one.

This is not a Julia observation; it is the content of the Gaussian-relations result that
improper priors are exactly what completes Gaussian maps into a hypergraph category. See
[[Acausal Composition is a Hypergraph Category]].

## 6. What is not here

- **No derivatives.** An MTK equation may contain `D(x)`; this one may not, so there is no
  DAE, no index, and no `mtkcompile`. [[Differential Algebra and DAE Factors]] is the note on
  what that would cost.
- **No nonlinearity.** $A_i$ is a matrix. A general residual $r_\theta$ is what
  [[Implicit Learners]] is about, and it loses the exact inversion.
- **No structural analysis.** MTK's incidence graph *is* a factor graph
  ([[ModelingToolkit as an Acausal Relation]] §4), so alias elimination and tearing have
  direct meaning here, and none of it is implemented.
- **No units, no connectors, no domains.** MTK's `@connector` generates the equations this
  factor represents; there is no sugar here that generates them for you.

Related: [[ModelingToolkit as an Acausal Relation]],
[[Acausal Composition is a Hypergraph Category]], [[Loopy Message Passing]],
[[Channels and Polarity]], [[Implicit Learners]], [[src/gaussian|gaussian]], [[beliefs]]

## LP-018 covariance validation (2026-10-29)

Noise must have exactly the residual's dimensions and be symmetric before its positive
definiteness is checked. Wrapping an unchecked matrix in `Symmetric` silently discarded
one triangle; a short variance vector also silently changed the residual metric's size.
Invalid shape or symmetry now raises `ArgumentError` with the required property.
Evidence: `test/lp018.jl`, “LP-018.7 residual covariance validation”, including the
reported asymmetric matrix and one-element noise vector, a rectangular matrix and valid SPD noise.

## LP-020: the parameter gradient holds the belief fixed (2026-11-01)

`Mycelium.factor_belief` and `belief_free_energy` are implemented here with blocks $A_i$ and
offset $-c$ (`gaussian.md`, "The factor belief, held fixed"). The fixed-belief energy uses
`inv(Symmetric(Q))` rather than `inv` of a Cholesky factor, for which Zygote has no rule. On the
two-variable example of [[LP-020 The fixed-message parameter gradient is not the free-energy gradient]]
the gradient is now exact, and alternating training reaches maximum likelihood (`test/lp020.jl`).

## Mean-field energy (2026-11-02)

A `MeanField()` node's term in the graph free energy ([[L-SPEC-005 The Hybrid Free Energy]]) is
`Mycelium.mean_field_energy`: the factor free energy at the product of the marginals, with the
entropy left out. `_product_belief` builds that product as a `FrozenGaussianBelief` (points
substituted, means stacked, covariances block diagonal, entropy zero), so the method is
`belief_free_energy` again, and the gradient at fixed marginals reuses the LP-020 code. A
Gaussian-mixture marginal enters by its moments, which is exact for the quadratic energy. Improper
or trivial marginals are refused. Checked against Monte Carlo and a closed-form naive mean field
in `test/mean_field_free_energy.jl`.

## Batched messages for plates (2026-11-02)

*Update (2026-11-03).* With message blocks (`lib/Mycelium.jl/src/blocks.md`) there is a second
method, on a `NamedTuple` of blocks: inputs as point blocks, trivial blocks, or proper Gaussian
blocks with one shared precision take the shared path below and return one `GaussianBlock`
(``H = Kc\,\mathbf 1^\top - \sum_n (K A_n) M_n``, the small products first); anything else goes
column by column through the method on vectors. The free energy over a plate row uses
`factor_beliefs`/`belief_free_energies`/`local_free_energies` on blocks, through
`FrozenGaussianBeliefs` (`gaussian.md`); test: `test/plates.jl`.

`Mycelium.factor_messages(fs, target, polarity, inputs, priors, ps, st)` computes the messages of
tied replicas (`lib/Mycelium.jl/src/plates.md`). When all replicas have one noise covariance and
dimensions, and every replica's inputs have the same covariances and flat directions, the metric
$R$, its factorisation or the residual metric $T$, and the rank test depend on the parameters only,
so they are computed once; replica $i$'s message is then $K(c - \sum_n A_n m_{n,i})$ with the shared
$K$ and precision. That is the case for data (points) and a shared prior. The input means of all
replicas are stacked, so the messages come from one product $K(c\,\mathbf 1^\top - \sum_n A_n M_n)$,
and they share one precision matrix (messages are never mutated in place). Otherwise each replica
goes through `_linear_message`. `test/plates.jl` checks both paths against `factor_message`.

## Mixture inputs (2026-11-03)

- `Mycelium.mixture_message(f::LinearConstraintFactor, ...)`, for `MixtureReduction(M)` nodes: the
  exact message with mixture inputs expanded, one component per combination of input components,
  each weighted by its evidence $\prod_i v_{i c_i}\, \mathcal N(d_c; 0, R_c)$. It is the switching factor's kernel with
  $K = 1$, $b_1 = c$ (`switching.md` §9, where the weights and the flat directions are worked out).
- `residual_statistics` now collapses a Gaussian-mixture message to its moments, as
  `factor_message` does. Before, a mixture cavity reached `_canon`, which has no method for it, so
  a linear factor next to a latent mixture had messages but no local free energy. The collapsed
  belief is the one the factor's messages were computed from (assumed-density filtering); it is
  not the exact mixture belief, and the free energy of such a graph is that of the collapsed
  beliefs. Found when a plate with a latent mixture was evaluated on its per-edge path
  (`test/plates.jl`).
