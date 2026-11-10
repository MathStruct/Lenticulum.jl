#implementation #derivation

> `Grassmann(k, m)` represents a real row space by an orthonormal row matrix.
> `Retracted` wraps Mycelium's gradient descent and momentum with horizontal
> projection, positive-diagonal QR retraction and projection of the velocity.

> Sources: Edelman, Arias & Smith (1998), *The Geometry of Algorithms with Orthogonality Constraints*; Absil, Mahony & Sepulchre (2008), *Optimization Algorithms on Matrix Manifolds*; implementation of [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §2.1–§2.2; code: `grassmann.jl`, `../test/runtests.jl`, `../../Mycelium.jl/src/factors.jl`.
>
> Bibliography: [[Bibliography#^edelman1998geometry|Edelman et al. 1998]] · [[Bibliography#^absil2008optimization|Absil et al. 2008]]
>
> Theory (CT-ML wiki): [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses)

## Representation and metric

A real $k$-plane in $\mathbb R^m$ is represented by $\Theta\in\mathbb R^{k\times m}$
with $\Theta\Theta^\top=I_k$. The constructor requires $1\le k\le m$.
Orthogonal changes of row basis represent the same point. This is a Stiefel
representative modulo $O(k)$, not a coordinate chart. The horizontal lift of the
Grassmann tangent space and its Frobenius orthogonal projection are

$$
\mathcal H_\Theta=\{\Delta:\Delta\Theta^\top=0\},\qquad
\Pi_\Theta(G)=G-(G\Theta^\top)\Theta.
$$

The full Stiefel tangent space also contains vertical basis rotations. These are
removed here. For row-space-invariant objectives, projecting the Euclidean
gradient gives the Riemannian gradient for the canonical Grassmann metric.
No Fisher metric for a statistical model is computed or identified with it.
Parenthesising the two products avoids forming an $m\times m$ projector:
the dense projection costs $O(k^2m)$.

Geometry operations check shapes but assume an orthonormal base point.
`ispoint(M, Θ; atol=1e-8)` checks shape, finiteness and orthonormality with zero
relative tolerance. It is a separate check rather than an extra matrix product
inside every update. The implementation targets dense real CPU arrays; no
GPU or AD contract is supplied.

## QR and its sign ambiguity

For $A=(\Theta+\Delta)^\top=QR$, retain thin $Q$ and multiply its columns by
$\operatorname{sign}(R_{ii})$. The corresponding diagonal of $R$ is positive.
Without this fix, even the zero step can flip rows of the representative and
invalidate a stored velocity's basis convention. With the fix, an already
orthonormal matrix retracts to itself to roundoff. Arbitrary displacements are
accepted as row orthonormalisation; the update rule supplies horizontal ones.
Only the thin $m\times k$ Q is materialised, not a full $m\times m$ matrix.

Non-finite inputs throw `ArgumentError`. A QR diagonal with any
`abs(R[i,i]) <= max(k,m)*eps(T)*norm(R, Inf)` also throws. This is a
floating-point diagonal rank criterion, not an exact rank certificate or a
rank-revealing pivoted factorisation. A horizontal step from an exact point is
full row rank because

$$
(\Theta+\Delta)(\Theta+\Delta)^\top=I_k+\Delta\Delta^\top.
$$

`rand(rng, M)` uses the same sign fix on a standard Gaussian $m\times k$
matrix and returns a dense Float64 representative with Haar-distributed row
space. `rand(M)` uses Random's default RNG. Tests check orthonormality,
reproducibility and the QR convention, not a statistical goodness-of-fit test
for the Haar distribution.

## Mycelium rule and state contract

`Retracted(rule, M)` accepts only `GradientDescent` and `Momentum`; unsupported
`AbstractUpdateRule`s throw at construction. Nesterov's forward look-ahead is
off the manifold and needs a separate retraction/gradient-transport contract.
`rule_get` is the identity on the representative. `rule_put` computes

$$
g=\Pi_\Theta(G),\qquad (s',p')=\operatorname{rule\_put}(\text{rule},s,\Theta,g),
\qquad \Theta^+=R_\Theta\bigl(\Pi_\Theta(p'-\Theta)\bigr).
$$

Gradient descent's state is passed through unchanged, usually `nothing`.
Momentum's state is exactly Mycelium's plain parameter-shaped velocity matrix,
not a tuple or a wrapped state. Its recurrence is $s'=\gamma s-\eta g$
(heavy-ball momentum; Mycelium had $-\gamma s$ until LP-015 corrected it). After
retraction, return $\Pi_{\Theta^+}(s')$ as its new state: projection transport
onto the new representative's horizontal space. No differentiated retraction,
parallel transport or isometry is claimed.

Mycelium has no rule-state initializer. Callers initialise momentum with
`zero(Θ)` and GD with `nothing`, exactly as its Euclidean rules are driven.
`OptimiserFactor(:θ, rule)` emits `rule_get(rule, st.state, st.p)` from state
`(p=Θ, state=s)`; `optimiser_step` calls `rule_put` separately from inference.
No Mycelium change is needed.

The integration difficulties were the QR sign convention and the state layout:
returning an equivalent but sign-flipped row basis breaks zero-step identity,
and replacing Mycelium's velocity by a conventional positive-$\gamma$
recurrence would silently change the wrapped optimiser. Tests check the
candidate and projected velocity against an independently formed projector,
and check the factor-driven and direct trajectories at every step.

## Checked by and limits

The `Grassmann geometry` and `Retracted rules and eigenplane descent` testsets
in `../test/runtests.jl` check:

- rectangular, line and full-space points, zero-step identity, positive QR
  diagonal, horizontal projection and its idempotence, rank-deficient and
  non-finite failures, shape errors, invalid dimensions and Float32 retraction;
- row-basis invariance of $F(\Theta)=\operatorname{tr}(\Theta S\Theta^\top)$
  and covariance of its projected gradient; at step $10^{-6}$, the objective
  decrease agrees with $\eta\|\operatorname{grad}F\|_F^2$ within relative
  tolerance `2e-5`;
- a seeded symmetric $6\times6$ matrix with spectrum $(-2,-1,1,2,3,5)$,
  $k=2$ and gap $\lambda_3-\lambda_2=2$. Both GD and momentum use
  `η=0.04 < 1/(2*opnorm(S))=0.1` for **400 steps**, momentum with `γ=0.6`.
  From the seeded random start, the largest principal angle to the span of
  the first two eigenvectors is below `1e-6` radians and the objective agrees
  with $\lambda_1+\lambda_2=-3$ to absolute tolerance `2e-12`;
- the same iterates through `OptimiserFactor` / `optimiser_step`, identity
  emission, unchanged GD state, momentum tangency after every step, the
  inherited recurrence and projection transport, and rejection of Nesterov
  and another unsupported rule.

The step size and iteration budget are demonstrated for this fixed test
problem; no global convergence theorem for the wrapped momentum is claimed.
Principal angles are computed in the tests by singular values of the overlap;
there is no public distance routine. §2.3's Sampson scalarisation, exposed
manifold channel spaces, polynomial factors, variable-rank models and beliefs
on the Grassmannian remain outside this implementation.
