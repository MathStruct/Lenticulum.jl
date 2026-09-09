# Fitting is a Nullspace Problem

> **Derivation.** The learning problem for an algebraic factor has a closed-form global
> optimum: the bottom-$k$ eigenspace of the Veronese second-moment matrix. No gradient
> descent, no local minima, no initialisation.

## The objective

Given data $x^{(1)},\ldots,x^{(M)} \in \mathbb{R}^N$ believed to lie on the relation, and the
parametrisation $r_\Theta = \Theta v_d(x)$ of [[The Veronese Parametrisation]], the
**algebraic-distance** objective is

$$f(\Theta) \;=\; \tfrac12 \sum_{i=1}^M \bigl\|\Theta\, v_d(x^{(i)})\bigr\|_2^2$$

This is $\tfrac12\sum_i \sigma(\mathbf{l}^c(x^{(i)}))$ with $\mathbf{l}^c = r_\Theta$ the
[[Scalar and Multivariate Energy|vector energy]] and $\sigma = \tfrac12\|\cdot\|^2$.

Without a constraint the minimum is $\Theta = 0$, for which $r \equiv 0$ and **every point
lies on the variety**. So a normalisation is mandatory. The correct one is
$\Theta\Theta^\top = I_k$ (orthonormal rows) — see [[The Parameter is a Grassmannian]] for
why this is not an arbitrary choice but the coordinatisation of the true parameter space.

## The derivation

Write $v_i := v_d(x^{(i)})$ and stack them as rows of $V \in \mathbb{R}^{M\times m}$. Then

$$f(\Theta) = \tfrac12\sum_i \|\Theta v_i\|^2 = \tfrac12\sum_i v_i^\top \Theta^\top\Theta v_i
= \tfrac12\sum_i \operatorname{tr}\bigl(\Theta v_i v_i^\top \Theta^\top\bigr)
= \tfrac12 \operatorname{tr}\bigl(\Theta\, S\, \Theta^\top\bigr)$$

where

$$\boxed{\;S \;=\; \sum_{i=1}^M v_d(x^{(i)})\, v_d(x^{(i)})^\top \;=\; V^\top V \;\in\; \mathbb{R}^{m\times m}\;}$$

is the **uncentred second-moment matrix of the Veronese-embedded data**: symmetric, positive
semidefinite.

**The data enters only through $S$.** Whatever $M$ is, the fitting problem compresses to one
$m\times m$ PSD matrix — which is also the statement that this is a streaming/one-pass
algorithm.

Now minimise $\operatorname{tr}(\Theta S\Theta^\top)$ over $\Theta\Theta^\top = I_k$. By the
**Ky Fan** variational principle (equivalently Courant–Fischer applied $k$ times), with
eigenvalues $\lambda_1 \ge \cdots \ge \lambda_m \ge 0$ of $S$:

$$\min_{\Theta\Theta^\top = I_k} \operatorname{tr}\bigl(\Theta S \Theta^\top\bigr) \;=\; \sum_{j=0}^{k-1}\lambda_{m-j}$$

attained when the rows of $\Theta$ span the eigenspace of the $k$ **smallest** eigenvalues of
$S$. Equivalently: $\Theta^\top$ = the $k$ right-singular vectors of $V$ with smallest
singular values.

$$\boxed{\;\Theta^\star \;=\; \text{bottom-}k\text{ eigenvectors of } S = V^\top V\;}$$

**A global optimum, in closed form.** This is the property no other family in
[[Implicit Learners]] has, and it is worth being loud about: the nonconvexity that dominates
all of deep learning is simply absent here. It has been traded for the nonconvexity of
[[Inference as Root Finding|inference]].

## Choosing $k$ is a rank decision

$k = m - \operatorname{rank}(S)$ in exact arithmetic. With noise, $S$ has no exact null
space, and $k$ becomes a **numerical rank** decision from the eigenvalue spectrum — a gap
you hope to see, and often do not.

Two things make this harder than a normal PCA rank choice:

1. **Monomials of different degrees have wildly different scales.** $x^3$ and $x$ differ by
   orders of magnitude on data of magnitude $\ne 1$, so the spectrum of $S$ is dominated by
   scaling artefacts rather than structure. Column-equilibrating $V$, or replacing monomials
   by an orthogonal polynomial basis (Chebyshev / Hermite w.r.t. the empirical measure), is
   not optional.
2. **The spectrum has no gap when $d$ is too large**, because of the spurious polynomials of
   [[The Veronese Parametrisation]] §"Choosing $d$": lower-degree relations multiplied by
   variables produce a whole staircase of near-zero eigenvalues carrying no new information.

## Doing it properly: degree by degree

The literature that solved this is the **approximate vanishing ideal** line —
Heldt–Kreuzer–Pokutta–Poulisse's AVI algorithm, and the closely related
**Vanishing Component Analysis** (Livni et al.) and **GPCA** (Vidal–Ma–Sastry) for the
subspace-arrangement case. The shape of the correct algorithm:

```
for degree δ = 1, 2, ..., d:
    build the candidate set: monomials of degree δ, MINUS the span of
        {x_j · f : f already found at degree δ-1}          ← quotient out the known ideal
    orthogonalise the candidates against the already-found polynomials
    evaluate on the data, take the SVD, keep singular vectors below tolerance τ
    add the new polynomials to the basis
```

Two things this buys you:

- the output is a **border basis** rather than an arbitrary generating set, which is the
  numerically stable analogue of a Gröbner basis (Kehrein–Kreuzer, Mourrain) — see
  [[Composition is Elimination]] §"Why not Gröbner" for why this distinction is essential;
- the spurious staircase is removed by construction.

The price is a **tolerance parameter $\tau$**, and the output is discontinuous in it. There
is no principled way to set $\tau$ from data; it is the algebraic analogue of choosing a
rank, with the same lack of theory.

## What happens with missing channels — and why it is EM

The closed form above assumes **every channel is observed** for every data point. That is
the training-on-complete-data case. If some channels are unobserved (which is the
interesting case — that is what makes it a relation and not a dataset), the objective
becomes

$$f(\Theta) = \sum_i \min_{x_u^{(i)}} \tfrac12\bigl\|\Theta\, v_d(x_o^{(i)}, x_u^{(i)})\bigr\|^2$$

and the natural algorithm alternates:

- **E-step**: for fixed $\Theta$, infer $x_u^{(i)}$ by [[Inference as Root Finding|root finding]];
- **M-step**: for fixed $\{x_u^{(i)}\}$, update $\Theta$ by the eigenproblem above.

This is **exactly [[Examples from the Paper|Example 2 of AutoBayes]]** — expectation
maximisation, with the E-step an inversion $c'_\pi$ and the M-step a descent on the composite
loss. The framework predicted the algorithm; the algebra supplies both halves in closed form.

The alternation is, of course, **no longer globally convergent**: each half is exactly
solvable, the alternation is not. Convexity in $\Theta$ survives; joint convexity does not.

## Cost

| step | cost |
|---|---|
| build $S = V^\top V$ | $O(M m^2)$, one pass, parallel |
| bottom-$k$ eigenspace | $O(m^3)$ dense, or $O(m^2 k)$ per iteration with Lanczos on $S$ |
| memory | $O(m^2)$ for $S$ |

For $N=20, d=3$ ($m = 1771$): $S$ is 25 MB, the eigenproblem is seconds. For $N=100, d=3$
($m = 176{,}851$): $S$ is 250 GB dense. **The wall is at $m \approx 10^4$**, i.e. roughly
$N \le 20$ at $d = 3$.

Related: [[The Veronese Parametrisation]], [[The Parameter is a Grassmannian]], [[Algebraic versus Geometric Distance]], [[Examples from the Paper]]
