# constraint.jl — implementation note

> `LinearConstraintFactor`: one equation ``0 = \sum_i A_i x_i - c + \varepsilon`` over any
> number of channels, none of them distinguished. ModelingToolkit's `0 ~ ...` with a noise
> term, and the first factor in this project with **no preferred direction at all**.

## 1. Why this factor exists

[[README]]'s table promises *"Symmetry handling: symmetric, no distinguished input/output"*.
`GaussianFactor` does not deliver that. It has an `in_channel` and an `out_channel`, it is
written as $y = Ax + b$, and it admits exactly two polarities no matter how you build it.
It is *bidirectional*, which is a weaker property than *acausal*: it can be run backwards,
but it still knows which way is forwards.

This factor delivers it. There are $n$ channels, the equation is a sum over all of them, and
`supported_polarities` returns $n$ polarities — one per channel — of which none is the
factor's own. The count is the content: **"acausal" is the statement that the number of
supported polarities equals the number of channels**, rather than being fixed at two by the
factor's constructor.

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

### 4.1 A latent channel makes the equation vacuous, and the code says so bluntly

If any non-target channel carries no information, `_message` returns `TrivialBelief()`. The
reasoning: with a flat prior on $x_j$, the direction $\operatorname{range}(A_j)$ of the
residual is unconstrained, so the equation says nothing.

**That is too blunt.** The sharp answer takes the limit $S_j \to \infty$ properly: $R^{-1}$
tends to the projection onto $\operatorname{range}(A_j)^\perp$ in the $Q^{-1}$ metric, so

$$
\mu_{c\to t} \;=\; \mathcal{N}^{-1}\bigl(A_t^\top \Pi\, d,\ A_t^\top \Pi\, A_t\bigr),
\qquad
\Pi = Q^{-1} - Q^{-1}A_j (A_j^\top Q^{-1} A_j)^{+} A_j^\top Q^{-1}
$$

which is *not* zero unless $A_j$ has full row rank. Concretely: two equations, one free
variable appearing in only one of them, and the other equation survives. The current code
throws that information away and returns nothing.

This is the same "marginalise a latent channel" operation the vault keeps deferring
([[Copiers Cups and Caps]] §"marginalisation is the expensive one"), specialised to the
linear-Gaussian case where it is a pseudo-inverse rather than an integral. It is the single
most valuable missing piece in this file.

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
[[Channels and Polarity]], [[Implicit Learners]]

> [!note] Two sibling notes are still missing
> `beliefs.jl` and `gaussian.jl` have no markdown beside them, though their docstrings
> reference `beliefs.md`, `gaussian.md` and `The Gaussian Factor.md`. That predates this file;
> `src/` is the only source directory in the repository without parallel markdown.
