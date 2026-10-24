#open-problem

> **Status: open (2026-10-24).** Area: analysis, difference equations · Prerequisites: Fourier analysis on $\mathbb R^n$, $C^k$ norms · Related: [[Universal Approximation for Implicit Diffusion Learners]] §3, [[Implicit Diffusion Learners]].
>
> Sources: original to this vault (from [[Universal Approximation for Implicit Diffusion Learners]] §3).

## Question

Given a smooth function $\hat G$ on a compact set $K \subset \mathbb R^n$, shifts
$s_1, \dots, s_q \in \mathbb R^n$ and positive weights $\gamma_1, \dots, \gamma_q$, find a smooth $H$ with

$$
\sum_{j=1}^q \gamma_j\, H(u + s_j) = \hat G(u) \qquad (u \in K).
$$

How large must $\lVert H \rVert_{C^k(K)}$ be? With two antithetic shifts ($q = 2$, $s_{1,2} = \pm s$)
the solution is an alternating series whose norm grows at most linearly in
$\operatorname{diam}(K)/|s|$. For $q \ge 3$ the slab recurrence used in the vault only gives an
exponential bound $O((q-1)^{\operatorname{diam}(K)/d})$. **Is there always a solution with
polynomial growth as the shifts shrink, and if not, for which shift configurations?**

## Setting

The equation arises when the training field of an implicit diffusion learner is evaluated at
a finite set of noise draws $\varepsilon_j$ at one noise level: the shifts are
$s_j = \sigma^\ast \varepsilon_j$ (up to scaling), and $H$ is the potential the learner must represent
to reproduce a target relation exactly ([[Universal Approximation for Implicit Diffusion Learners]] §3).
Fix $q \ge 3$, shifts of size $|s_j| \asymp \sigma^\ast$, and ask for $H \in C^k$ on a neighbourhood of $K$
with the equation holding on $K$.

## What is known

- $q = 2$: explicit alternating solution, linear growth (the note, §3).
- $q \ge 3$, equal weights: the slab recurrence has amplification factor $q - 1$ per slab of width
  $\asymp \sigma^\ast$, giving the exponential bound; whether it is attained is **open** (the note marks
  this as a conjecture).
- If only approximation is needed, the unshifted rescaling $H = (\sum_j \gamma_j)^{-1}\hat G$ has
  error $o(1)$ in $C^2(K)$ as $\sigma^\ast \to 0$ (uniform continuity of $D^2\hat G$), which suffices for the
  universal approximation theorem as stated; $O(\sigma^\ast)$ for $\hat G \in C^3$, and $O((\sigma^\ast)^2)$ only for
  centred shifts ($\sum_j \gamma_j s_j = 0$) and $\hat G \in C^4$ (the note, §3). Exact solvability is a separate
  question.

## What would count as an answer

A construction of $H$ with polynomial norm growth for generic shifts; or a lower bound showing
exponential growth is unavoidable for some configuration (e.g. via the symbol
$\sum_j \gamma_j e^{i\langle \xi, s_j\rangle}$ having zeros near the real axis).

## Why it matters here

It tells whether exact recovery of a relation by a finite-noise-draw training field is
well-conditioned, which bears on how many noise draws per level are worth using.

## Leads

The Fourier symbol $m(\xi) = \sum_j \gamma_j e^{i\langle\xi, s_j\rangle}$: division by $m$ is
well-conditioned where $|m|$ is bounded below; the question is about its zeros and the
localisation to $K$. Mean-periodic functions and convolution equations (Schwartz) are the
classical setting.
