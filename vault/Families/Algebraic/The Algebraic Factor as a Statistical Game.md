#model #derivation

> Assembling everything into [AutoBayes Definition 20](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) and
> [Definition 27](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients). Every abstract slot gets a concrete,
> proposed mathematical inhabitant. The implemented subset is the polynomial
> residual, Sampson scalarisation and complete-data fitting; messages and the
> assembled statistical game remain planned.

> Sources: original design and analysis; code: `lib/AlgebraicFactors.jl/src/polynomial.jl`, `polyfit.jl`, `veronese.jl`, `lib/AlgebraicFactors.jl/test/polynomial.jl` (residual and fitting only).
>
> Theory (CT-ML wiki): [Lax Functor](https://mathstruct.org/CategoryTheory-ML-Wiki/Lax-Functor) · [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## The quadruple

A parameterized statistical game is $(\Theta, c)$ with $c(\theta) = (c, c', l^c, H^c)$.
For an algebraic factor of degree $d$ on $N$ channels with $k$ generators:

| slot | AutoBayes | algebraic realisation |
|---|---|---|
| parameter space $\Theta$ | any space | $\mathrm{Gr}(k,m)$, $m=\binom{N+d}{d}$ — [[The Parameter is a Grassmannian]] |
| $\Theta' $ (update space) | tangent bundle (Rem. 30) | $T_\Theta\mathrm{Gr}$, via $\Pi_\Theta(G)= G-G\Theta^\top\Theta$ |
| forward kernel $c$ | $X \rightsquigarrow \llbracket c\rrbracket\times Y$ | root-solve in the chosen polarity |
| latent space $\llbracket c\rrbracket$ | scratch | **the branch index** $\{1,\ldots,D\}$ — [[Branches and the Discriminant]] |
| inversion $c'_\pi$ | $\mathcal{P}X \to \{Y\rightsquigarrow X\times\llbracket c\rrbracket\}$ | root-solve in the *opposite* polarity, weighted by $\pi$ |
| **vector** energy $\mathbf{l}^c$ | $\to K_c$ | $r_\Theta(x) = \Theta\,v_d(x) \in \mathbb{R}^k$ |
| energy space $E_c$ | ours | $\mathbb{R}^k$ — **the residual noise space**, [[Algebraic versus Geometric Distance]] |
| scalarisation $\sigma_c$ | ours | $\tfrac12 r^\top M r$; $M=I$ (algebraic) or $M=(J\Sigma J^\top)^{-1}$ (Sampson) |
| **vector** entropy $\mathbf{H}^c$ | $\mathcal{P}X\times Y\to K_c$ | $-\sum_j w_j\log w_j$ over branches, times a direction $u\in K_c$ |
| gradient coupling | Def. 29's laxness | `ExactCoupling` — the IFT gives the exact Jacobian |

The table is a design sketch, not an implemented game contract. `PolynomialFactor`
has empty supported polarities and no messages, branch beliefs, game entropy or
`ExactCoupling` declaration. Its packed pointwise convention is x=nothing, y=z,
wrt=:y, differentiating all coordinates; [[polynomial]] explains named channels.

## The vector energy is forced, not chosen

$\mathbf{l}^c = r_\Theta$ is $\mathbb{R}^k$-valued by construction. There is no scalar
residual replacing all components without losing their Jacobian: k denotes the
chosen number of equations, not necessarily the codimension. A variety need not
be a complete intersection. So the algebraic family is the
existence proof for [[Scalar and Multivariate Energy]]: it is a family in which the paper's
$[0,\infty]$-valued $l^c$ **cannot be used without first throwing away the model**, because

- the [[Inference as Root Finding|well-posedness condition]] is $\operatorname{rank}J_u = k$,
  a statement about the vector residual's Jacobian, unstatable for a scalar;
- the [[Backpropagation by the Implicit Function Theorem|adjoint]] solves $J_u^\top\lambda = -\bar x_u$,
  requiring the $k\times q$ Jacobian;
- the correct scalarisation $M = (J\Sigma J^\top)^{-1}$ is derived *from* the Jacobian.

## The energy/entropy split is genuinely two things here

[Proposition 18](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) insists energy and entropy behave differently.
In this family the difference is visible:

$$
F^c(\pi, x_o) \;=\; \underbrace{\sum_{j=1}^{D} w_j \cdot \tfrac12\bigl\|r_\Theta(x_o, x_u^{(j)})\bigr\|_M^2}_{\text{energy: pointwise, per branch, } =\,0 \text{ at an exact root}}
\;-\; \underbrace{\Bigl(-\sum_j w_j\log w_j\Bigr)}_{\text{entropy: a functional of the branch belief}}
$$

For the displayed proposed objective the residual term vanishes at exact roots,
provided the metric is defined. This alone does not establish the correct free
energy of a branch model: priors, branch Jacobians and normalisation may add terms.
An overdetermined system can still have an exact root, so k>q alone does not
imply nonzero energy. No branch free-energy claim is tested or implemented here.

## Composition: local only

By [[Composition is Elimination]] the factors must **not** be fused. The composite game is
formed by the [Definition 22](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#composition-energies-add-entropies-chain-definition-22) laws applied to the local
data:

$$
\mathbf{l}^{dc} = \bigl(r^c_\Theta,\ r^d_\Phi\bigr) \in \mathbb{R}^{k_c}\oplus\mathbb{R}^{k_d},
\qquad
\mathbf{H}^{dc}(\pi,z) = \Bigl(\mathop{\mathbb{E}}_{(y,b)\sim d'}\bigl[\mathbf{H}^c(\pi,y)\bigr],\ \mathbf{H}^d(c_*\pi,z)\Bigr)
$$

The energy direct sum here is concrete: the composite residual is the **stacked** residual
vector, graded by factor. That is exactly the $E_G = \bigoplus_f E_f$ of
[[Scalar and Multivariate Energy]], and in this family it is literally how you would write
the joint system down — which is a good sign the grading is the right abstraction rather than
an imposition.

The expectation in the entropy law is over branches, so it is a **finite sum**, not a Monte
Carlo estimate. In this family the chain rule is exact.

## Proposed alternating learning

From [[Fitting is a Nullspace Problem]] §"missing channels":

- **E-step** $=$ the inversion $c'_\pi$: root-find the unobserved channels; obtain the branch
  belief $\{(x_u^{(j)}, w_j)\}$.
- **M-step** $=$ descent on $F$: with branches fixed, the energy is quadratic in $\Theta$, so
  the update is the bottom-$k$ eigenspace of $S = \sum_{i,j} w_{ij}\, v_i^{(j)}(v_i^{(j)})^\top$ —
  a *globally optimal algebraic-distance* parameter step.

This spectral step holds for fixed branch weights/completions and the algebraic
objective only. Root solving is not generally closed form, and a hard completion
is not an expectation under the posterior. The full Sampson fit differentiates
its varying covariance and uses an iterative non-convex solver ([[polyfit]]).
This proposed alternating procedure is not implemented or validated as EM.

## Proposed exact gradient coupling away from singularities

[Definition 29](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients) is lax in general because $\theta$ moves the
pushforward prior and the sampling distribution. Here:

- the "sampling distribution" is a **finite branch set**, and its dependence on $\theta$ is
  differentiable away from the discriminant, with derivative given by the IFT;
- so the terms Definition 29 drops are **computable**, and the coupling can be `ExactCoupling`
  rather than `DiagonalCoupling`.

The exception is at the discriminant, where the branch *count* changes — a discrete jump that
no derivative captures. So the honest statement is: **exact away from $\Delta$,
discontinuous on it.** The `GradientCoupling` annotation should be
`ExactCoupling` with a conditioning guard on $\kappa(J_u)$.

## What this factor cannot do

To be explicit, since the table above is otherwise flattering:

- $N \lesssim 20$, $d \lesssim 4$ (parameter count and sample complexity);
- $q \lesssim 10$ unobserved channels per message (Bézout);
- coefficients must be fitted, not composed ([[Composition is Elimination]]);
- no guarantee the learned real locus is nonempty or of the intended dimension
  ([[Varieties Ideals and Real Nullstellensatz]]);
- gradients meaningless near the discriminant.

Related: [[Factors are Parameterized Statistical Games]], [[Scalar and Multivariate Energy]], [[Branches and the Discriminant]], [[Open Problems in Algebraic Implicit Learning]]
