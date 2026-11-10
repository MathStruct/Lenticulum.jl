#model #derivation

> **Derivation.** The full forward (jvp) and adjoint (vjp) equations for an algebraic
> factor. Cost: one $k\times k$ linear solve. The parameter cotangent turns out to be
> **rank at most one**.

> Sources: original derivation; code: `lib/AlgebraicFactors.jl/src/polynomial.jl`, `lib/AlgebraicFactors.jl/test/polynomial.jl` (exact message Jacobians and scalar least-squares mixed-Hessian check; IFT solve derivatives are not implemented).
>
> Theory (CT-ML wiki): [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## Implementation boundary

`PolynomialFactor` now uses exact r and J for single-branch Gaussian sites and
iterated frozen-metric Gauss–Newton ([[polynomial]]). The jvp/vjp equations below
remain a derivation: no IFT rule, saved root factorisation, solve differentiation
or ExactCoupling declaration is implemented. Gauss–Newton posterior curvature
must not be described as the exact Hessian at a nonzero residual.

## Setup

$r_\Theta(x_o, x_u) = \Theta\, v_d(x_o, x_u) = 0$, with $x_o\in\mathbb{R}^p$ clamped,
$x_u\in\mathbb{R}^q$ inferred, $r\in\mathbb{R}^k$, $\Theta\in\mathbb{R}^{k\times m}$.
Assume the well-posed case $k = q$ and $J_u$ invertible at the solution
([[Inference as Root Finding]]; failure is [[Branches and the Discriminant]]).

Write, using [[The Veronese Parametrisation]] §"Jacobians, exactly",

$$
J_o = \Theta\,Dv_d(x)_{[:,o]} \in\mathbb{R}^{k\times p}, \qquad
J_u = \Theta\,Dv_d(x)_{[:,u]} \in\mathbb{R}^{k\times q}, \qquad
v := v_d(x) \in \mathbb{R}^m
$$

All three are **exact**; no automatic differentiation is involved anywhere.

## Total differential

Differentiating the constraint $r_\Theta(x_o,x_u)=0$, which holds identically in a
neighbourhood:

$$
J_o\,\mathrm{d}x_o \;+\; J_u\,\mathrm{d}x_u \;+\; (\mathrm{d}\Theta)\, v \;=\; 0
$$

(the third term because $r = \Theta v$ is *linear in $\Theta$*, so its differential in
$\Theta$ is just $(\mathrm{d}\Theta)v$). Hence the **implicit function theorem**:

$$
\boxed{\;\mathrm{d}x_u \;=\; -J_u^{-1}\Bigl(J_o\,\mathrm{d}x_o \;+\; (\mathrm{d}\Theta)\,v\Bigr)\;}
$$

giving the two sensitivities

$$
\frac{\partial x_u}{\partial x_o} \;=\; -J_u^{-1}J_o \;\in\;\mathbb{R}^{q\times p},
\qquad
\frac{\partial x_u}{\partial \Theta_{j\ell}} \;=\; -J_u^{-1} e_j\, v_\ell
$$

## Forward mode (jvp)

Given tangents $(\dot x_o, \dot\Theta)$:

$$
\dot x_u \;=\; -J_u^{-1}\bigl(J_o\,\dot x_o + \dot\Theta\, v\bigr)
$$

One solve. Cost $O(k^3)$ for the factorisation, $O(k^2)$ per additional tangent — so a
factorisation of $J_u$ is reused across the whole batch of tangents.

## Reverse mode (vjp) — the adjoint

This is what [the `put` of a lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) needs. Given a cotangent
$\bar x_u \in \mathbb{R}^q$ arriving from downstream, we want $\bar x_o$ and $\bar\Theta$
satisfying $\langle \bar x_u, \dot x_u\rangle = \langle\bar x_o,\dot x_o\rangle + \langle\bar\Theta,\dot\Theta\rangle_F$
for all tangents.

Substitute:

$$
\langle \bar x_u,\dot x_u\rangle
= -\,\bar x_u^\top J_u^{-1}\bigl(J_o\dot x_o + \dot\Theta v\bigr)
$$

Define the **adjoint variable**

$$
\boxed{\;\lambda \;=\; -\,J_u^{-\top}\,\bar x_u \;\in\;\mathbb{R}^k\;}
$$

— one linear solve with the *transpose*. Then

$$
\langle \bar x_u,\dot x_u\rangle = \lambda^\top J_o\,\dot x_o \;+\; \lambda^\top \dot\Theta\, v
= \bigl\langle J_o^\top\lambda,\ \dot x_o\bigr\rangle \;+\; \bigl\langle \lambda\, v^\top,\ \dot\Theta\bigr\rangle_F
$$

using $\lambda^\top\dot\Theta v = \operatorname{tr}(v\lambda^\top\dot\Theta) = \langle\lambda v^\top,\dot\Theta\rangle_F$.
Therefore

$$
\boxed{\;\bar x_o \;=\; J_o^\top\lambda, \qquad \bar\Theta \;=\; \lambda\, v_d(x)^\top\;}
$$

## Three things to notice

### 1. The parameter cotangent has rank at most one

$\bar\Theta = \lambda\,v^\top$ is an outer product. **Every data point contributes a parameter cotangent of rank at most one**
(it can be zero), and their unprojected minibatch sum has rank at most B. The
Grassmann projection/retraction is a separate operation.

Consequences:
- The Grassmannian update of [[The Parameter is a Grassmannian]] is a rank-$B$ perturbation
  of a $k$-plane — a subspace-tracking problem, costing $O(Bkm)$ rather than $O(km^2)$.
- Storage of the accumulated gradient can be $O(B(k+m))$ rather than $O(km)$.
- This cotangent statement alone does not imply rank B for a Gauss–Newton
  matrix: that matrix uses the full residual parameter Jacobian.

This is a real structural gift and it comes directly from the linearity in $\Theta$.

### 2. The cost is one $k\times k$ solve, shared

$J_u$ is factorised once during inference (Newton needs it anyway) and reused for the
adjoint. So **the backward pass is essentially free given the forward pass** — one
triangular solve. Compare an unrolled equilibrium solver, which pays $O(\text{iterations})$
in memory and time.

### 3. Where it breaks

At a root where Jᵤ is singular the square IFT does not apply. Inspect its smallest
singular value as well as conditioning: a nonzero scalar has condition number
one however close to zero it is (checked on the circle in the polynomial tests).
See [[Branches and the Discriminant]].
Near $\Delta$ the honest options are to regularise the solve
($\lambda = -(J_u J_u^\top + \varepsilon I)^{-1}J_u\bar x_u$, i.e. a
Tikhonov damping of the adjoint equation) and report the approximation, or refuse.
This is the minimiser of ‖Jᵤᵀλ+x̄ᵤ‖²+ε‖λ‖²; the previous formula used the
wrong transpose. Its normal equation is checked with a nonsymmetric 2×2 Jᵤ in
`Polynomial note corrections`.

## The overdetermined / minimising case

When $k > q$ there is generically no root and inference minimises
$\sigma(r) = \tfrac12\|r\|^2_M$ instead. The stationarity condition for fixed M is
$J_u^\top M\, r = 0$. Differentiation gives

$$
H_{uu}=J_u^\top M J_u+\sum_i(Mr)_i\nabla^2_{uu}r_i,\qquad
H_{uo}=J_u^\top M J_o+\sum_i(Mr)_i\nabla^2_{uo}r_i,
\qquad \frac{\partial x_u}{\partial x_o}=-H_{uu}^{-1}H_{uo}.
$$

The mixed residual-Hessian term in Hᵤₒ was missing in the earlier formula.
For r=(u²+ou−1,u−o), M=I, o=0.4, independently solved stationary points
and central differences agree with this expression (atol=1e−9); omitting the
mixed term gives an error above 1e−3. This is checked in `Polynomial note corrections`.
If M varies with coordinates, differentiating its stationarity condition adds
metric-derivative terms too.

Hᵤᵤ is the full Hessian of the fixed-metric objective. Dropping the residual
second-derivative terms in both blocks gives the Gauss–Newton sensitivity
−(JᵤᵀMJᵤ)⁻¹JᵤᵀMJₒ, agreeing at r=0 when the required inverse exists.
This is a frozen-metric approximation when the Sampson metric varies. The
second-derivative terms are also exactly computable here (the
$\nabla^2 r_i$ are again $\Theta$ times a fixed monomial Hessian), so unlike in neural
settings the Gauss–Newton approximation is a **choice**, not a necessity.

## Why this is *not* "differential algebra"

The README's table lists the implicit backward pass as "implicit function theorem /
differential algebra". Everything above is the implicit function theorem — ordinary
multivariable calculus. **No differential algebra appears, and none is needed.** Differential
algebra does have genuine roles here, but they are different jobs:

| job | tool | note |
|---|---|---|
| derivative of the solve | implicit function theorem | *this note* |
| is this polarity well-posed at all? | elimination / dimension theory | [[Composition is Elimination]] |
| composing two factors into one | Gröbner / border bases | [[Composition is Elimination]] |
| factors that are DAEs, $r(x,\dot x)=0$ | differential elimination, index reduction | [[Differential Algebra and DAE Factors]] |

The last row is where differential algebra genuinely enters a *backward* pass, and even
there it is index reduction of the adjoint DAE rather than differentiation as such.

Related: [[Inference as Root Finding]], [[The Parameter is a Grassmannian]], [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens), [[Differential Algebra and DAE Factors]]
