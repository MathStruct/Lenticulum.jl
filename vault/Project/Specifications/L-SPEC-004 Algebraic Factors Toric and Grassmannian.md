#design

> **Status: partly implemented (2026-10-25): §0 and §1.1–§1.7; §2 and later remain planned.** Originally planned on Daniel's request (2026-10-25, "Could you plan for that?"), following Gemini's suggestion to begin with toric models and the Grassmannian. Related: [[Toric Models as Factors]], [[Grassmann Mixing and Plücker Coordinates]], [[The Parameter is a Grassmannian]], [[Algebraic Implicit Learners]], [[Open Problems in Algebraic Implicit Learning]], [[L-SPEC-001 The Factor Interface]], [[L-SPEC-003 Exposed Parameters and Energy Channels]]. Code: `lib/AlgebraicFactors.jl/src/toric.jl` and `fit.jl`, tested in `lib/AlgebraicFactors.jl/test/runtests.jl`. These toric steps needed no Mycelium changes; later manifold update rules remain planned.

## Scope

The first code for the algebraic family ([[Algebraic Implicit Learners]]) implements finite
toric factors; the general polynomial and manifold constructions remain unimplemented. Gemini's review (Proposal 7 of `05_Architectural_Design_Proposals.md` in
the review folder) sketches a whole package `lib/AlgebraicFactors.jl`; this specification orders
it so that the cases where everything is exact or cheap come first:

1. toric factors ([[Toric Models as Factors]]): exact discrete factors with known implicit
   equations;
2. Grassmannian parameters ([[The Parameter is a Grassmannian]]): projection and retraction for
   parameters defined up to a gauge;
3. the general polynomial factor, on top of both;
4. multi-branch inference by homotopy continuation, last.

Plücker coordinates ([[Grassmann Mixing and Plücker Coordinates]]) enter as a small
deterministic factor and as test data. The contract below is a proposal; each part becomes
binding when it is implemented with its tests.

## Today

| capability | state |
|---|---|
| a factor over several discrete variables | `AlgebraicFactors.ToricFactor`: exact finite sum-product messages and joint factor free energy; tree marginals and Bethe free energy checked against enumeration |
| toric fitting and model checks | `fit_mle`: gauge-reduced Newton or cyclic binary-feature IPF; separate moment convergence and existence statuses; positive `inmodel` and supplied `binomial_residuals`, checked against closed forms and independence minors |
| parameters on a manifold | none. Mycelium's update rules (`GradientDescent`, `Momentum`, `Nesterov` through `rule_get`/`rule_put`) are Euclidean |
| a polynomial (Veronese) factor, nullspace fitting | none; the theory is in the notes |
| multi-branch roots | none; `MixtureBelief` exists to carry them |

## The proposed contract

**§0 Package.** `lib/AlgebraicFactors.jl`, depending on LenticulumCore, Mycelium and
LinearAlgebra, plus LuxCore and Random for the parameter/state interface, until §4 adds a package extension. Tests and docs are wired like the other
packages under `lib/`.

**§1 Toric factors.**
- §1.1 `ToricFactor(names::Tuple, alphabets::Tuple, A::AbstractMatrix{<:Integer}; normalised = false)`:
  one channel per name, of the alphabet's size. The joint states are ordered column-major over the
  alphabets (documented), and column $x$ of `A` is $a_x$; the constructor checks
  `size(A, 2) == prod(alphabets)`. Constructors build `A` for common cases:
  `independence(names, alphabets)` and `hierarchical(names, alphabets, generators)`
  (marginal-cell indicators per generator).
- §1.2 Parameters `ps = (η = …,)` in $\mathbb R^d$. $p_\eta = p_{\eta'}$ iff $A^\top(\eta-\eta') \in \operatorname{span}\{\mathbf 1\}$ (the docstring
  says so); an optional reduction to independent logit differences is a chart, not a change of
  model. Statements that need $\mathbf 1$ in the row space of $A$ use $\bar A = \binom{\mathbf 1^\top}{A}$ internally
  ([[Toric Models as Factors]]).
- §1.3 `energy(f, x, …)` is $-\langle\eta, a_x\rangle$, plus $\Lambda(\eta)$ when `normalised`; $x$ is a tuple of labels. Scalar.
- §1.4 `factor_message(f, target, polarity, inputs, prior, ps, st) -> (belief, st)`: the exact
  sum-product likelihood message to `target`, a `CategoricalBelief`. Observed inputs may be Diracs
  on labels or categorical beliefs; latent channels (missing or uninformative inputs, which
  Mycelium resolves to `Latent()`) are summed over with unit weight, so ordinary BP initialisation
  works. The message excludes the target's `prior`. The cost, $\prod$ alphabets per message, is stated.
- §1.5 `local_free_energy(f, msgs, ps, st)`: $\mathbb E_{b_f}E - H(b_f)$ for the factor belief
  $b_f \propto \psi_\eta \prod_u \mu_{u\to f}$, exact, in the form `PriorFactor` returns it (a scalar). With
  `normalised = true`, the game-level `free_energy` and `entropy` of [[L-SPEC-001 The Factor Interface]] §4
  are exact as well.
- §1.6 `fit_mle(f, counts; method = :newton | :ipf)`, returning $\eta$, a convergence status and,
  separately, an existence status. The stopping test is moment matching, $\lVert A\hat p - Au/n\rVert$
  (Birch), which can also approach zero at an extended MLE on the boundary. Without an existence
  certificate (§Open), a diverging $\eta$ is reported as non-convergence or suspected boundary
  convergence, never as proven non-existence, and never silently. Newton removes the gauge
  directions first.
- §1.7 `inmodel(f, p; atol)`: for positive $p$, whether $\log p$ lies in the row space of $\bar A$ (least
  squares). `binomial_residuals(f, p, moves)` evaluates $p^{u^+} - p^{u^-}$ for moves the user supplies.
  No Markov-basis computation.

**§2 Grassmannian parameters.**
- §2.1 `Grassmann(k, m)`, represented by $\Theta \in \mathbb R^{k\times m}$ with $\Theta\Theta^\top = I_k$:
  `project_tangent(M, Θ, G)` $= G - G\Theta^\top\Theta$; `retract(M, Θ, Δ)` orthonormalises the rows of
  $\Theta + \Delta$ by a thin QR of the transpose, with signs chosen so that $\operatorname{diag}(R) > 0$, so that
  $R_\Theta(0) = \Theta$ (in exact arithmetic; tested with `≈`); `rand(rng, M)`. (Edelman, Arias & Smith 1998; Absil, Mahony & Sepulchre 2008.)
- §2.2 `Retracted(rule, M) <: AbstractUpdateRule` for Mycelium's optimiser lens, at first for
  `GradientDescent` and `Momentum` only, with `rule_get(...) = Θ`. `rule_put` projects the gradient,
  applies `rule`, converts the candidate point it returns into a displacement $p' - \Theta$, projects
  that onto the tangent space and retracts. A rule's state (momentum) is projected onto the new
  tangent space, which is a vector transport for an embedded manifold (Absil, Mahony & Sepulchre
  2008). Nesterov waits: its look-ahead $p + \gamma s$ leaves the manifold, and a retracted look-ahead
  with a consistent transport of the gradient has to be specified first. This is the lens $G^*(\Theta, \Delta) = \mathrm{qf}(\Theta + \Pi_\Theta\Delta)$ of
  [[The Parameter is a Grassmannian]].
- §2.3 Gauge invariance as a test obligation: a factor whose relation depends on $\Theta$ only
  through its row space must give the same scalarised energy for $\Theta$ and $B\Theta$ ($B$ invertible)
  under its default scalarisation. The Sampson energy $\tfrac12 r^\top(JJ^\top)^{-1}r$ with $J = D_z r_\Theta$ of full row
  rank has this property ([[Algebraic versus Geometric Distance]]), but today's scalarisations see
  only the residual, not $J$; a point- and parameter-dependent scalarisation has to be specified
  first, with its behaviour at singular points (damping or a pseudoinverse does not keep the
  invariance by itself). Checked once §3 exists.
- §2.4 With [[L-SPEC-003 Exposed Parameters and Energy Channels]], an exposed Grassmannian
  parameter is a channel whose space is a manifold; that spec's "keep open" list already allows it.

**§3 The polynomial factor** (after §1 and §2), as in Gemini's Proposal 7: an exponent table with
monomial or Chebyshev basis, the residual $r_\Theta(z) = \Theta v_d(z)$ as the pointwise energy, the
Sampson scalarisation (after the interface of §2.3), `fit_nullspace` and Sampson IRLS ([[Fitting is a Nullspace Problem]]),
and single-branch messages by Newton with a Laplace Gaussian
([[Inference as Root Finding]]).

**§4 Multi-branch inference** (last): a HomotopyContinuation.jl package extension finding all
isolated real roots and returning a `MixtureBelief` over branches
([[Branches and the Discriminant]]).

**§5 Plücker.** `plucker(z, w)` and its Jacobian; a deterministic `PluckerFactor(r)` with the
forward polarity only (the inverse message would be a belief on an $SL_2$-orbit, not built); and
`plucker_relations(r)`, the quadrics $p_{ij}p_{kl} - p_{ik}p_{jl} + p_{il}p_{jk}$, as test data.

## Checked by

Implemented §0 and §1.1–§1.5: `lib/AlgebraicFactors.jl/test/runtests.jl` checks hand-written
independence columns, state order, energies, messages with labelled/Dirac/categorical/latent
inputs, local and game quantities, and five-variable trees with three toric factors against
independent enumeration. The conformance checker
runs its explicit direct-message path over every supported polarity of the test factors at
arities one, two and three, using independent site and pooled-posterior fixtures. Existing
factors retain the lens checks. No assembled lens or joint latent posterior API is supplied. See [[toric]]
for label rules, game argument conventions and the Bethe entropy accounting.

Implemented §1.6–§1.7: the `Toric fitting and membership` testset checks positive
2×3 and 3×4 independence against the outer product of margins (cell tolerance `3e-10`),
and a positive 2×3×2 chain against $\hat p_{AB}\hat p_{BC}/\hat p_B$ (`5e-10`), by both
Newton and IPF. A seeded random integer design checks moment matching (`1e-10`) and
$Au/n-Ap_\eta$ against central likelihood differences. Redundant rows and an added
intercept give the same probabilities; an independent pseudoinverse checks the minimum-norm
gauge, and normalised energies reproduce the returned probabilities. An empty-row table
and its reversed table approach the extended MLE within `3e-10`, with finite parameters,
moment convergence and `existence == :suspected_boundary`. Zero cells with positive margins
can instead give `:unknown`: only all-positive counts are currently certified `:exists`.
Positive membership agrees with all independence minors on seeded random points in and
out of the model. Invalid inputs, exact augmented-kernel checks and degenerate uniform
models are tested too. See [[fit]] for all numerical choices and the heuristic thresholds.
The earlier planned requirement that an empty-row fit must not be "converged" was wrong:
§1.6 distinguishes moment convergence at an extended MLE from finite-MLE existence.

Still planned (geometry and later families):

- §2: $\Theta\Theta^\top = I$ after `retract`; `project_tangent(M, Θ, G) * Θ' ≈ 0`; `retract(M, Θ, zero(Θ)) ≈ Θ`;
  for symmetric $S$ with $\lambda_k < \lambda_{k+1}$, generic initialisation and a suitable step rule, Riemannian
  descent on $\operatorname{tr}(\Theta S\Theta^\top)$ reaching the span of the $k$ smallest eigenvectors (the nullspace fit's
  answer), compared by principal angles and by the objective against $\sum_{i\le k}\lambda_i$.
- §3 with §5: on points $z\wedge w$ for random $z, w \in \mathbb R^4$, `fit_nullspace` of degree 2 should
  recover the Klein quadric, a one-dimensional null space, with at least 27 generic points and
  a stated rank tolerance (an expectation, agreeing with a scratch check;
  [[Grassmann Mixing and Plücker Coordinates]]).

## Open

- General MLE existence certification: the implemented sufficient certificate is all-positive
  counts. Deciding relative-interior membership of $(1,Au/n)$ in the convex hull of the
  columns of $\bar A$ (Fienberg & Rinaldo 2012) can be formulated as a linear program;
  whether to depend on an LP solver. The boundary diagnostic is only a heuristic.
- Markov bases need external software (4ti2 and similar); whether to depend on it.
- Large tables: messages are exponential in arity. Whether `hierarchical` should split into one
  factor per generator automatically is a junction-tree question
  ([[LP-010 Where the graph of a problem comes from]]).
- Beliefs on the Grassmannian (rather than points on it) are not planned.
- A causal Grassmann-mixing block ([[Grassmann Mixing and Plücker Coordinates]] §"What Lenticulum
  can take") is an experiment for LenticulumLM, not part of this package.
