#model #derivation

> The single structural fact that makes the algebraic family tractable: **the residual is
> linear in the parameter.**

> Sources: original to this vault (design and analysis); code: `lib/AlgebraicFactors.jl/src/veronese.jl`, `polynomial.jl`, `polyfit.jl`, `lib/AlgebraicFactors.jl/test/polynomial.jl`.
>
> Theory (CT-ML wiki): [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category)

## Setup

Collect all channels of a factor into one vector $x \in \mathbb{R}^N$,
$N = \sum_i \dim X_i$. Let $v_d : \mathbb{R}^N \to \mathbb{R}^m$ be the **Veronese
embedding** of degree $d$ — the vector of all monomials of degree $\le d$:

$$
v_d(x) \;=\; \bigl(1,\; x_1,\ \ldots,\ x_N,\; x_1^2,\ x_1x_2,\ \ldots,\ x_N^d\bigr)^\top,
\qquad m \;=\; \binom{N+d}{d}
$$

A vector of $k$ polynomials of degree $\le d$ is then exactly a matrix
$\Theta \in \mathbb{R}^{k \times m}$, and the residual of
[[Implicit Learners|the implicit learner]] is

$$\boxed{\;r_\Theta(x) \;=\; \Theta\, v_d(x)\;}$$

with the relation
$R_\Theta = V_\mathbb{R}(r_\Theta) = \{x : \Theta\, v_d(x) = 0\}$.

## Why this is the right coordinatisation

**The model factors as a fixed nonlinear embedding followed by a linear map.**

$$\mathbb{R}^N \xrightarrow{\ v_d\ (\text{fixed, exact, sparse})\ } \mathbb{R}^m \xrightarrow{\ \Theta\ (\text{learned, linear})\ } \mathbb{R}^k$$

Implemented in [[veronese]] and [[polynomial]]: graded degree, descending lexicographic
indices, affine or degree-exact tables, monomial or tensor-product Chebyshev
values and exact polynomial Jacobians. Chebyshev expects caller-scaled data in
[-1,1] without enforcing the domain. `basis_change` gives the exact affine-table
conversion; degree-exact Chebyshev products need not be homogeneous, so a square
conversion to homogeneous monomials generally does not exist.

Implemented in [[veronese]] and [[polynomial]]: graded degree, descending lexicographic
indices, affine or degree-exact tables, monomial or tensor-product Chebyshev
values and exact polynomial Jacobians. Chebyshev expects caller-scaled data in
[-1,1] without enforcing the domain. `basis_change` gives the exact affine-table
conversion; degree-exact Chebyshev products need not be homogeneous, so a square
conversion to homogeneous monomials generally does not exist.

Everything learnable is in the linear part. Consequences, each derived in its own note:

| consequence | note |
|---|---|
| complete-data algebraic-distance fitting has a **global** SVD optimum; Sampson fitting is non-convex | [[Fitting is a Nullspace Problem]] |
| the parameter cotangent is **rank one** per data point | [[Backpropagation by the Implicit Function Theorem]] |
| $\Theta$ is identified only up to $GL_k$, so $P = Gr(k,m)$ | [[The Parameter is a Grassmannian]] |
| all Jacobians are $\Theta \cdot Dv_d(x)$ — **no autodiff needed** | below |

This is a genuinely unusual profile. A neural network is nonconvex in its parameters and
trivial in its inputs. **An algebraic factor is linear in its parameters and hard in its
inputs.** The difficulty has been moved from learning to inference.

## Jacobians, exactly

$Dv_d(x) \in \mathbb{R}^{m \times N}$ has entries $\partial x^\alpha/\partial x_j = \alpha_j\, x^{\alpha - e_j}$ —
a sparse matrix of monomials, computable exactly in integer arithmetic on the exponents. So

$$J_r(x) \;=\; \frac{\partial r_\Theta}{\partial x} \;=\; \Theta\; Dv_d(x) \;\in\; \mathbb{R}^{k \times N}$$

and, splitting $x = (x_{obs}, x_{unobs})$ by a [[Channels and Polarity|polarity]],

$$
J_o = \Theta\, Dv_d(x)_{[:,\,obs]} \in \mathbb{R}^{k\times p},
\qquad J_u = \Theta\, Dv_d(x)_{[:,\,unobs]} \in \mathbb{R}^{k\times q}
$$

The implementation needs no automatic differentiation: first derivatives are polynomial
expressions obtained by exponent arithmetic or Chebyshev recurrences. Rational inputs
can be evaluated exactly; floating-point evaluation still rounds. Higher derivatives
are not implemented, and no measured claim about their cost is made.

## The number that kills it

$$m = \binom{N+d}{d}$$

| $N$ | $d$ | $m$ |
|---:|---:|---:|
| 10 | 3 | 286 |
| 20 | 3 | 1 771 |
| 20 | 4 | 10 626 |
| 100 | 3 | 176 851 |
| 100 | 4 | 4 598 126 |
| 784 (MNIST) | 2 | 308 505 |
| 784 | 3 | 8.09 × 10⁷ |

$m$ is simultaneously:

- the **parameter count** (times $k$);
- the **sample complexity** — with $M < m$ data points the matrix
  $V \in \mathbb{R}^{M\times m}$ of [[Fitting is a Nullspace Problem]] has a nontrivial null
  space *for purely dimensional reasons*, so you can fit a variety through the data while
  learning nothing. You need $M \gg m$;
- the size of the eigenproblem, $O(m^3)$ dense (or $O(m^2 k)$ iteratively);
- the number of columns Gröbner/border-basis algorithms manipulate.

**One combinatorial count, four costs.** This is why the family is filed under "low dimensions"
and why no amount of implementation cleverness rescues it — see
[[Open Problems in Algebraic Implicit Learning]] §2 for the escape routes that have been
tried (sparse/structured supports, kernelisation, random projection) and why none of them
fully works.

## Choosing $d$ is not free either

Increasing $d$ does not merely add capacity; it also adds redundant vanishing polynomials. If
$f$ vanishes on the data and $\deg f \le d-1$, then $x_j f$ also vanishes and lies in degree
$\le d$. These products lie in the degree-$d$ null space, but can be linearly dependent;
there is no general N-fold dimension formula.

To identify new ideal generators rather than a fixed-degree vanishing space, a
procedure can work **degree by degree**, quotienting out the ideal generated by
what was already found. `fit_nullspace` returns the fixed-degree space; it does
not implement that quotient. That is exactly what approximate-vanishing-ideal and
border-basis algorithms do; see [[Fitting is a Nullspace Problem]] §"Doing it properly".

Related: [[Fitting is a Nullspace Problem]], [[The Parameter is a Grassmannian]], [[Backpropagation by the Implicit Function Theorem]]
