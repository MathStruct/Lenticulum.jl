# Inference as Root Finding

> The forward pass of an algebraic factor. Given a [[Channels and Polarity|polarity]],
> inference means solving a polynomial system — and everything interesting follows from
> counting its solutions.

## The problem

Split $x = (x_o, x_u)$ with $x_o \in \mathbb{R}^p$ observed (clamped) and
$x_u \in \mathbb{R}^q$ unobserved (to be inferred), $p+q = N$. Inference is

$$\text{find } x_u \in \mathbb{R}^q \ \text{ such that } \ r_\Theta(x_o, x_u) = 0 \in \mathbb{R}^k$$

a system of $k$ polynomial equations of degree $\le d$ in $q$ unknowns, with coefficients
depending polynomially on the clamped $x_o$.

**There is no distinguished direction.** The same $\Theta$ answers any polarity; only the
split changes. This is [[Copiers Cups and Caps|compact closure]] made completely concrete —
the "forward" and "backward" passes of an algebraic factor are the *same computation* with
different variables held fixed.

## Well-posedness is a rank condition

| | | |
|---|---|---|
| $k = q$ | **square** | generically finitely many solutions |
| $k < q$ | **underdetermined** | positive-dimensional solution set: genuinely multi-valued |
| $k > q$ | **overdetermined** | generically *no* solution: must minimise $\|r\|$ instead |

More precisely, the polarity is well-posed at a solution $x^\star$ iff

$$\boxed{\;\operatorname{rank} J_u(x^\star) \;=\; \operatorname{rank}\frac{\partial r_\Theta}{\partial x_u}(x^\star) \;=\; q \;=\; k\;}$$

which is exactly the hypothesis of the implicit function theorem. So
**`supports_polarity` from [[Channels and Polarity]] is, for this family, a computable
statement about a Jacobian rank** — not a declaration by the factor author. That is worth
noting because it is the only family where it *is* computable.

The overdetermined case is not a failure mode to be avoided; it is the README's
"output only the closest point to the variety, instead of a point on the variety". It is
handled by replacing root finding with

$$x_u^\star = \arg\min_{x_u} \ \sigma\bigl(r_\Theta(x_o,x_u)\bigr)$$

which is total ($\sigma \ge 0$ always has an infimum) where root finding is partial.
**Energy minimisation is the total version of root finding** — the same observation the
vault makes in [[Implicit Learners]], here with an explicit reason.

## How many solutions

For a square system of $q$ equations of degree $d$ in $q$ unknowns, over $\mathbb{C}$:

$$\#\{\text{isolated solutions}\} \;\le\; d^q \qquad \textbf{(Bézout)}$$

and more sharply, by **Bernstein–Khovanskii–Kushnirenko (BKK)**, the number of solutions in
the torus $(\mathbb{C}^*)^q$ is bounded by the **mixed volume** of the Newton polytopes —
much smaller for sparse systems. Since the learned $\Theta$ is typically dense in the
monomial basis, expect Bézout in practice unless sparsity is imposed.

| $d$ | $q$ | Bézout bound |
|---:|---:|---:|
| 2 | 10 | 1 024 |
| 3 | 10 | 59 049 |
| 3 | 20 | $3.5\times10^{9}$ |
| 3 | 30 | $2.1\times10^{14}$ |

**The exponential is in the number of *unobserved* channels, not the total dimension.** That
is a genuinely useful distinction: a factor in $N = 50$ variables of which only $q=6$ are
ever inferred at once is fine ($3^6 = 729$), even though its parameter count
$m = \binom{53}{3} = 23{,}426$ is uncomfortable. **Inference cost and learning cost are
governed by different exponentials**, and a graph can be designed to keep $q$ small per
message while $N$ is large. This is the strongest argument for the factor-graph
architecture within this family.

## How to actually solve it

**Homotopy continuation** is the right tool: deform a start system with known solutions into
the target and track the paths. Total-degree homotopy tracks $d^q$ paths; **polyhedral
homotopy** tracks only mixed-volume-many. It is:

- **complete** — finds all isolated complex solutions, so you know you have them all;
- **embarrassingly parallel** — one path per core;
- **certifiable** — Smale's $\alpha$-theory gives a rigorous "this path converged to a true
  solution" certificate.

In Julia this is `HomotopyContinuation.jl` (Breiding–Timme); the classical alternative is
Bertini. For positive-dimensional solution sets, **witness sets** and numerical irreducible
decomposition handle the $k<q$ case.

Cheaper local alternatives, when completeness is not needed: Newton from a warm start (the
previous message's value, or the prior's mean). This is what a message-passing schedule
would actually do in the inner loop, falling back to homotopy only when Newton fails.

### Alternatives when the system is not square

- **Gauss–Newton / Levenberg–Marquardt** on $\sigma(r)$: uses $J^\top J$ and needs the
  *vector* residual — see [[Scalar and Multivariate Energy]] §6. Local, fast, no
  completeness guarantee.
- **Moment–SOS (Lasserre) relaxation**: minimise $\|r\|^2$ globally via a hierarchy of
  SDPs, with a Positivstellensatz certificate of global optimality
  ([[Varieties Ideals and Real Nullstellensatz]] §"Positivstellensatz"). Globally certified,
  but the order-$\rho$ moment matrix has size $\binom{q+\rho}{\rho}$ and $\rho$ must often
  exceed 2. Practical for $q \lesssim 10$.

## Real versus complex

Homotopy continuation finds **complex** solutions; you want **real** ones. Three
consequences:

1. You pay for all $d^q$ complex paths and discard most of them. There is no known way to
   track only the real solutions.
2. The number of real solutions **changes** with $x_o$. Over some regions there are two,
   over others none. This is the branch structure of
   [[Branches and the Discriminant]] and it makes inference **discontinuous** in the
   observed input.
3. $V_\mathbb{R}$ may be empty even when $V_\mathbb{C}$ is large
   ([[Varieties Ideals and Real Nullstellensatz]] §"Problem 1"). Inference then has no answer
   at all and must fall back to minimising $\|r\|$.

## Where this genuinely works

Not hypothetical — these are the established successes, and they share the profile
"$N$ moderate, $d$ small, structure known":

| problem | $q$ | solution count |
|---|---|---|
| 6R serial manipulator inverse kinematics | 6 | exactly **16** (Lee–Liang; Raghavan–Roth; Primrose) |
| 5-point relative pose (essential matrix) | 5 | exactly **10** (Demazure; Nistér) |
| conic through 5 points | — | 1 |
| chemical reaction network steady states | small | varies; well studied |

The inverse-kinematics case is the archetype for the whole family: a **relation** between
joint angles and end-effector pose; inference in one direction is easy (forward kinematics),
in the other is a degree-16 root find; the multiple solutions are *physically real* (elbow
up / elbow down), not numerical artefacts; and selecting among them needs a prior.

Related: [[Branches and the Discriminant]], [[Backpropagation by the Implicit Function Theorem]], [[Channels and Polarity]], [[The Algebraic Factor as a Statistical Game]]
