#open-problem

> **Status: open (2026-10-24).** Area: analysis, differential geometry of densities · Prerequisites: implicit function theorem, tubular neighbourhoods, basic diffusion models · Related: [[Universal Approximation for Implicit Diffusion Learners]] §7, [[Implicit Diffusion Learners]], [[Open Problems in Implicit Diffusion Learning]] (T1, I12), [[Channels and Polarity]].
>
> Sources: original to this vault (question formulated from the heuristic analysis of [[Universal Approximation for Implicit Diffusion Learners]] §7 and the MNIST observation of [[Open Problems in Implicit Diffusion Learning]] I12).

## Question

A single smooth potential $U : \mathbb R^n \to \mathbb R$ is used to answer queries in many
directions: clamp some coordinates, solve for the others as stationary points of $U$ on the
slice. For which potentials, and in particular for which smoothed data densities
$U = -\log p_t$, are the answers **isolated and nondegenerate for every query direction of
interest**, including directions with more free coordinates than the codimension of the data?

## Setting

Split the coordinates as $z = (z_C, z_F)$ with $F \subseteq \{1, \dots, n\}$, $|F| = m$ (a
*polarity*). Given a clamp $z_C = c$, anchors $z_{0,F}$ and anchor precisions $\rho_F \ge 0$, the
**answers** are the solutions $y \in \mathbb R^m$ of

$$
r(y) = \nabla_{z_F} U(c, y) + \rho_F^2 \odot (y - z_{0,F}) = 0
$$

with $J = \partial_y r = (\nabla^2 U)_{FF} + \operatorname{diag}(\rho_F^2)$ positive definite
(*regular, stable* answers). Let the data lie on a smooth compact submanifold
$M \subset \mathbb R^n$ of dimension $n - k$ with a smooth positive density $q$ on $M$, and let
$p_t$ be the density of $\alpha_t X + \sigma_t \varepsilon$, $X \sim q$, $\varepsilon \sim \mathcal N(0, I)$.

**Question.** For $U = -\log p_t$, $\rho_F = 0$, small $t > 0$ and a family $\mathcal F$ of polarities:

1. If $m = |F| \le k$ and the coordinate slice meets $M$ transversally, are the answers near
   $M \cap \{z_C = c\}$ regular, with $\lambda_{\min}(J)$ bounded below like $\sigma_t^{-2}$?
2. If $m > k$, the slice meets $M$ in an $(m - k)$-dimensional set. Are the answers exactly the
   nondegenerate local modes of $q$ restricted to that set, and is $\lambda_{\min}(J)$ then of order
   the curvature of $\log q$, independent of $t$?
3. What can be said when $q$ is uniform on part of $M$ (no in-manifold curvature)? Is a positive
   $\rho_F$ necessary, and how large?

## What is known

- §7 of [[Universal Approximation for Implicit Diffusion Learners]] describes the three regimes
  (transverse slicing, conditional mode finding, soft-anchored projection) as a **heuristic
  analysis and conjecture**, not a proof.
- One energy network answers both MNIST directions, image → label ($m = 10$) and
  label → image ($m = 784$) ([[Open Problems in Implicit Diffusion Learning]] I12): an observation
  that both regimes occur in practice, not a theorem.
- For the pure distance potential $U = \tfrac12\operatorname{dist}(z, M)^2$ the Hessian on $M$ is the
  projector onto the normal space, so (1) holds and (2) fails: the answers form a continuum.

## What would count as an answer

Theorems for (1) and (2) with explicit dependence on $t$, the reach of $M$ and the derivatives of
$\log q$; or counterexamples (a manifold and density where (2) fails for small $t$). A sharp
answer to (3) would give a rule for choosing $\rho_F$.

## Why it matters here

It decides whether "one model, every question" holds for diffusion learners, and how to set the
anchor precisions that the code currently chooses by hand.

## Leads

Laplace-type expansions of $p_t$ near $M$ (the normal part concentrates like $\sigma_t^{-2}$, the
tangential part carries $\log q$); the literature on the manifold hypothesis for score-based
models; Morse theory on the slice.
