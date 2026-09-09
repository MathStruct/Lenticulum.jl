# Varieties, Ideals, and the Real Nullstellensatz

> The minimum algebra needed, and the three places where working over $\mathbb{R}$ — which
> we must — breaks the textbook complex picture.

## Ideals and varieties

For $S \subseteq k[x_1,\ldots,x_N]$, the **variety** is
$$V(S) = \{x \in k^N : f(x) = 0 \ \ \forall f \in S\}$$
For $X \subseteq k^N$, the **vanishing ideal** is
$$I(X) = \{f : f(x) = 0 \ \ \forall x \in X\}$$

$I(X)$ is always an ideal, and always **radical** ($f^n \in I \Rightarrow f \in I$). The two
operations are almost inverse:

$$V(I(X)) = \overline{X}^{\,\text{Zar}}, \qquad I(V(J)) = ?$$

Over an algebraically closed field, **Hilbert's Nullstellensatz** answers the second:
$I(V(J)) = \sqrt{J}$. That is the clean correspondence
$\{\text{radical ideals}\} \leftrightarrow \{\text{varieties}\}$ that makes algebraic
geometry work.

## Problem 1: we are over $\mathbb{R}$, and $\mathbb{R}$ is not closed

Hilbert's Nullstellensatz **fails** over $\mathbb{R}$. The standard witness:

$$J = \langle x^2 + y^2 \rangle \subset \mathbb{R}[x,y], \qquad V_\mathbb{R}(J) = \{(0,0)\}, \qquad I(V_\mathbb{R}(J)) = \langle x, y\rangle \neq \sqrt{J} = J$$

A *curve* over $\mathbb{C}$ has a *single real point*. The correct statement is the **Real
Nullstellensatz** (Dubois, Risler, Stengle):

$$I(V_\mathbb{R}(J)) \;=\; \sqrt[\mathbb{R}]{J} \;:=\; \Bigl\{f : f^{2n} + \textstyle\sum_i \sigma_i^2 \in J \text{ for some } n, \ \sigma_i \in \mathbb{R}[x]\Bigr\}$$

the **real radical**. Two consequences for us, both bad:

- **Real dimension can collapse.** The learned $V_\mathbb{C}(r_\Theta)$ may be a nice
  $(N-k)$-dimensional variety while $V_\mathbb{R}(r_\Theta)$ is a point, or empty. Nothing in
  the [[Fitting is a Nullspace Problem|fitting procedure]] prevents this: the fit only sees
  the data points, and is free to return an ideal whose real locus is barely bigger than
  the training set.
- **Real radicals are much harder to compute** than radicals. Where a complex problem needs
  a Gröbner basis, the real problem needs real quantifier elimination or a
  Positivstellensatz certificate, and the complexity gap is large.

> [!warning] This is a genuine, unsolved-in-practice gap
> There is no known cheap regulariser that forces the learned ideal's *real* locus to be
> well-behaved. Everything computable (the residual, the eigenproblem, the Jacobian rank)
> is a statement about the complex variety. See
> [[Open Problems in Algebraic Implicit Learning]] §1.

## Problem 2: a variety does not determine its equations

$V(f) = V(f^2)$ as sets, but $\langle f \rangle \neq \langle f^2\rangle$. More importantly,
if $\Theta$ and $\Theta'$ have the same **row space**, then $r_\Theta$ and $r_{\Theta'}$ cut
out the same variety and have the same ideal-in-degree-$d$. So the parameter is identified
only up to $GL_k$ acting on the left.

That is not a nuisance to be regularised away — it is the statement that the parameter lives
on a Grassmannian. See [[The Parameter is a Grassmannian]].

## Problem 3: singularities

$V(r_\Theta)$ is generically singular, and its singular locus is exactly
$$\mathrm{Sing}(V) = \{x \in V : \operatorname{rank} J_r(x) < k\}$$
At a singular point:

- the [[Backpropagation by the Implicit Function Theorem|implicit function theorem fails]];
- the tangent space is not defined, so the local dimension is not what you think;
- the [[Algebraic versus Geometric Distance|Sampson precision]] $(JJ^\top)^{-1}$ blows up.

Nash–Tognoli ([[Universal Approximation by Nash-Tognoli]]) produces *nonsingular* models,
but nothing in the fitting procedure produces nonsingular fits. Smoothness is an
unenforced assumption throughout.

## Positivstellensatz — the constructive route back

Stengle's **Positivstellensatz** is the real analogue that *does* give certificates: it
characterises polynomials positive on a semialgebraic set in terms of sums of squares. This
is the theoretical foundation of the **Lasserre moment–SOS hierarchy**, which turns
"minimise $\|r_\Theta(x)\|^2$ over $x$" into a sequence of semidefinite programs converging
to the global minimum.

That matters because it is the one route to **globally certified inference** without root
counting — see [[Inference as Root Finding]] §"Alternatives". The cost is that the SDP at
relaxation order $\rho$ has a moment matrix of size $\binom{N+\rho}{\rho}$, so it inherits
the same exponential as everything else.

## What to remember

| complex textbook | our situation |
|---|---|
| $I(V(J)) = \sqrt{J}$ | $I(V_\mathbb{R}(J)) = \sqrt[\mathbb{R}]{J}$, much harder |
| $\dim_\mathbb{C} V$ is what you fit | $\dim_\mathbb{R} V$ can be anything smaller, including $0$ |
| generic = smooth | fitted $\neq$ generic; singular fits are the norm |
| ideal $\leftrightarrow$ variety | parameter $\leftrightarrow$ variety is $GL_k$-to-one |

Related: [[The Veronese Parametrisation]], [[The Parameter is a Grassmannian]], [[Open Problems in Algebraic Implicit Learning]]
