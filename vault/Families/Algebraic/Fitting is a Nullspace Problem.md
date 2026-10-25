#model #derivation

> **Derivation and implementation.** Complete-data algebraic-distance fitting has a
> closed-form global optimum: the bottom-$k$ eigenspace of the Veronese second-moment
> matrix. Sampson fitting uses a different, non-convex objective and an iterative solver.

> Sources: original analysis; Chojnacki, Brooks, van den Hengel & Gawley (2000), *On the fitting of surfaces to data with covariances*, §4.1–§4.2, Fig. 1; code: `lib/AlgebraicFactors.jl/src/polyfit.jl`, `veronese.jl`, `lib/AlgebraicFactors.jl/test/polynomial.jl`.
>
> Bibliography: [[Bibliography#^chojnacki2000fitting|Chojnacki et al. 2000]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Cospan](https://mathstruct.org/CategoryTheory-ML-Wiki/Cospan)

## The objective

Given data $x^{(1)},\ldots,x^{(M)} \in \mathbb{R}^N$ believed to lie on the relation, and the
parametrisation $r_\Theta = \Theta v_d(x)$ of [[The Veronese Parametrisation]], the
**algebraic-distance** objective is

$$f(\Theta) \;=\; \tfrac12 \sum_{i=1}^M \bigl\|\Theta\, v_d(x^{(i)})\bigr\|_2^2$$

This is $\sum_i \sigma(\mathbf{l}^c(x^{(i)}))$ with $\mathbf{l}^c = r_\Theta$ the
[[Scalar and Multivariate Energy|vector energy]] and $\sigma = \tfrac12\|\cdot\|^2$.

Without a constraint the minimum is $\Theta = 0$, for which $r \equiv 0$ and **every point
lies on the variety**. So a normalisation is mandatory. The correct one is
$\Theta\Theta^\top = I_k$ (orthonormal rows) — see [[The Parameter is a Grassmannian]] for
why this is not an arbitrary choice but the coordinatisation of the true parameter space.

## The derivation

Write $v_i := v_d(x^{(i)})$ and stack them as rows of $V \in \mathbb{R}^{M\times m}$. Then

$$
f(\Theta) = \tfrac12\sum_i \|\Theta v_i\|^2 = \tfrac12\sum_i v_i^\top \Theta^\top\Theta v_i
= \tfrac12\sum_i \operatorname{tr}\bigl(\Theta v_i v_i^\top \Theta^\top\bigr)
= \tfrac12 \operatorname{tr}\bigl(\Theta\, S\, \Theta^\top\bigr)
$$

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

**A global algebraic-distance optimum, in closed form.** This statement applies
to this fixed complete-data objective, not all learning in the family. The
orthonormal-row constraint is itself non-convex; the spectral solution solves it
globally. Sampson fitting and [[Inference as Root Finding|inference]] remain non-convex.

## Choosing $k$ is a rank decision

$k = m - \operatorname{rank}(S)$ in exact arithmetic. With noise, $S$ has no exact null
space, and $k$ becomes a **numerical rank** decision from the eigenvalue spectrum — a gap
you hope to see, and often do not.

Two things make this harder than a normal PCA rank choice:

1. **Monomials of different degrees have wildly different scales.** $x^3$ and $x$ differ by
   orders of magnitude on data of magnitude $\ne 1$, so the spectrum of $S$ is dominated by
   scaling artefacts rather than structure. Column-equilibrating $V$, or replacing monomials
   by a polynomial basis such as Chebyshev after explicit data scaling, is
   often useful, but changes the coefficient metric. `fit_nullspace` performs no
   implicit equilibration or normalisation; users choose the data scale and basis.
2. **Higher degrees can obscure the spectral gap**, because of redundant polynomials of
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
is no principled way to set $\tau$ from data; it is a numerical rank choice; this implementation reports the threshold and spectrum
instead of hiding that choice.

## What happens with missing channels — and why it is EM

The closed form above assumes **every channel is observed** for every data point. That is
the training-on-complete-data case. If some channels are unobserved (which is the
interesting case — that is what makes it a relation and not a dataset), the objective
becomes

$$f(\Theta) = \sum_i \min_{x_u^{(i)}} \tfrac12\bigl\|\Theta\, v_d(x_o^{(i)}, x_u^{(i)})\bigr\|^2$$

and the natural algorithm alternates:

- **E-step**: for fixed $\Theta$, infer $x_u^{(i)}$ by [[Inference as Root Finding|root finding]];
- **M-step**: for fixed $\{x_u^{(i)}\}$, update $\Theta$ by the eigenproblem above.

This suggests a hard-completion alternating procedure related to
[[AutoBayes Examples as Factor Graphs|Example 2 of AutoBayes]]. Minimising missing
coordinates is not a posterior expectation and is not generally a closed-form
E-step. Only the fixed-completion algebraic M-step has the spectral solution.
Neither this alternation nor missing-data inference is implemented here; no
convergence claim is made.

## Cost

| step | cost |
|---|---|
| build $S = V^\top V$ | $O(M m^2)$, one pass, parallel |
| bottom-$k$ eigenspace | $O(m^3)$ dense, or $O(m^2 k)$ per iteration with Lanczos on $S$ |
| memory | $O(m^2)$ for $S$ |

The table describes the scatter-matrix approach. The implemented direct SVD
instead stores V, avoiding squared condition numbers; see [[polyfit]] for
underdetermined tables and memory use. Earlier unmeasured timing and capacity
cutoffs have been removed.

## Implemented fits and checks

`fit_nullspace` returns Θ, all m singular values (including structural zeros),
numerical_rank, nullity and threshold. No column scaling is applied.
`fit_sampson` uses the full analytic gradient, horizontal projection, QR
retraction and Armijo line search; `method=:fns` implements the one-equation
FNS correction, not naive frozen-weight reweighting. It stops on the
Riemannian gradient norm and reports convergence separately from iteration count.
See [[polyfit]] for the gradient derivation, checked citation, numerical choices
and seeded results: the real 400-point Klein fit in both bases, conic recovery
and noisy cost decrease, retracted-curve gradient differences, finite local
minimum checks, and a two-quadric intersection. No global Sampson optimum is claimed.

Related: [[The Veronese Parametrisation]], [[The Parameter is a Grassmannian]], [[Algebraic versus Geometric Distance]], [[AutoBayes Examples as Factor Graphs]]
