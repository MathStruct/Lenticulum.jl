# Algebraic Implicit Learners

> Entry point for the first of the three families in [[Implicit Learners]]: the approximator
> is an **algebraic variety**, inference is **root finding**, and the backward pass is the
> **implicit function theorem** (plus, for a narrower job than [[README]] suggests,
> differential algebra).

## The verdict, first

This family **works, and works beautifully, in low dimension**. It is the only one of the
three where

- the fitting problem has a **closed-form global optimum** — an eigendecomposition, no
  gradient descent, no local minima ([[Fitting is a Nullspace Problem]]);
- all derivatives are **exact**, with no automatic differentiation anywhere
  ([[Backpropagation by the Implicit Function Theorem]]);
- inference can be **certified**: you can prove you found all the solutions
  ([[Inference as Root Finding]]);
- the abstract objects of the AutoBayes framework become concrete and checkable — the
  latent space $\llbracket c \rrbracket$ is literally the **set of solution branches**
  ([[Branches and the Discriminant]]).

It **dies** above roughly $N \approx 20$ variables and degree $d \approx 4$, and not for
want of engineering. Every cost in the pipeline is governed by the same number,

$$m \;=\; \binom{N+d}{d} \;=\; \dim \mathbb{R}[x_1,\ldots,x_N]_{\le d}$$

the dimension of the space of candidate polynomials. That number is the parameter count,
the **sample complexity**, the size of the eigenproblem, and the base of the Gröbner and
Bézout blowups. For $N=100$, $d=3$ it is already $176{,}851$; for MNIST's $N=784$ at $d=3$
it is $8.1\times 10^7$. The prompt's own taxonomy — algebraic for low dimensions, diffusion
for high — is correct, and this note set explains *why* with numbers rather than intuition.

## The notes

**The model class**
1. [[Varieties Ideals and Real Nullstellensatz]] — what a variety is, and the three places
   real algebraic geometry differs from the complex textbook case
2. [[The Veronese Parametrisation]] — $r_\Theta = \Theta\, v_d(x)$: the residual is
   **linear in the parameter**
3. [[Universal Approximation by Nash-Tognoli]] — the expressiveness theorem, and why it is
   weaker than Weierstraß in exactly the way that matters

**Learning**
4. [[Fitting is a Nullspace Problem]] — the global optimum is the bottom-$k$ eigenspace of
   the Veronese second-moment matrix
5. [[The Parameter is a Grassmannian]] — the parameter is a subspace, not a matrix; this
   forces a Riemannian update and instantiates $(P, P')$ from [[Parametric Lens]]
6. [[Algebraic versus Geometric Distance]] — the convexity above is bought with the wrong
   distance, and the correction is exactly a [[Scalar and Multivariate Energy|scalarisation]]

**Inference**
7. [[Inference as Root Finding]] — polarity, well-posedness, Bézout/BKK counts, homotopy
   continuation
8. [[Branches and the Discriminant]] — $\llbracket c \rrbracket$ = the branches; the
   discriminant is where the IFT, the entropy and the continuity all fail together
9. [[Backpropagation by the Implicit Function Theorem]] — the forward and adjoint equations,
   derived; the parameter cotangent is **rank one**

**Structure**
10. [[Composition is Elimination]] — composing two algebraic factors is a Gröbner
    computation, and it is catastrophic; the factor graph exists to avoid it
11. [[Differential Algebra and DAE Factors]] — where differential algebra genuinely enters,
    which is *not* backpropagation
12. [[Algebraic Statistics Bridge]] — conditional independence **is** a polynomial
    constraint, so a Bayesian network **is** a variety; marginalisation **is** elimination

**Honesty**
13. [[The Algebraic Factor as a Statistical Game]] — the full AutoBayes packaging
14. [[Open Problems in Algebraic Implicit Learning]] — what does not work, and what is open

## The one-paragraph summary

Parametrise a relation by a vector of polynomials $r_\Theta(x) = \Theta\, v_d(x)$, where
$v_d$ is the Veronese (monomial) embedding and $\Theta$ a coefficient matrix. Because $r$ is
**linear in $\Theta$** and **polynomial in $x$**, the learner has exactly the opposite
computational profile to a neural network: *convex in the parameters, hard in the
variables*. Fitting is an eigenproblem with a global optimum; inference is root finding with
$O(d^q)$ solutions; differentiating through inference is one $k \times k$ linear solve. The
parameter is a point of a Grassmannian, so updates are Riemannian. Composition of two such
factors requires elimination and blows up doubly exponentially — which is precisely the
argument for keeping factors separate in a graph and passing messages, i.e. for
`Mycelium.jl`.

Related: [[Implicit Learners]], [[Channels and Polarity]], [[Scalar and Multivariate Energy]], [[README]]
