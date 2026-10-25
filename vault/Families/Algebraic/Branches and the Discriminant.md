#model #derivation

> The most satisfying concrete grounding in this note set: for an algebraic factor, the
> abstract latent space $\llbracket c \rrbracket$ of [Definition 1](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model) is
> **the set of solution branches** in the proposed multi-branch model. Target-rank
> loss obstructs the root IFT; it need not make the full Sampson covariance singular.

> Sources: original design and analysis; code: `lib/AlgebraicFactors.jl/src/polynomial.jl`, `lib/AlgebraicFactors.jl/test/polynomial.jl` (circle tangency and scalar conditioning checks; no branch enumeration).
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Open Model](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model)

## The projection is a branched cover

Fix a polarity. The projection

$$
\pi \;:\; V_\mathbb{R}(r_\Theta) \longrightarrow \mathbb{R}^p, \qquad (x_o, x_u) \mapsto x_o
$$

can be read locally as a **covering map of degree D** when the fibres are finite,
the total locus is smooth, the projection has no critical points and roots do
not escape to infinity. These assumptions are needed; an arbitrary polynomial
relation need not be a finite proper cover. Under them the real root count is
constant on each component away from critical values.

$$
\boxed{\;\llbracket c \rrbracket \;=\; \pi^{-1}(x_o) \;=\; \{1,\ldots,D\}\;\text{, the branch index}\;}
$$

So an algebraic factor's forward kernel really does have the type of
[Definition 1](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model),

$$
c \;:\; X \rightsquigarrow \llbracket c \rrbracket \times Y
$$

and the latent variable has a **meaning**: *which solution you are on*. Elbow up or elbow
down. Which of the ten essential matrices. This is not a bookkeeping device; it is the
physically meaningful hidden state.

Everything the vault says abstractly about $\llbracket c \rrbracket$ now has a concrete
reading:

| abstract ([[Open Models and Latent Channels]], [Composition of Open Models](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Model#composition-without-integration-definition-4)) | algebraic reading |
|---|---|
| latent space $\llbracket c \rrbracket$ | the set of branches over $x_o$ |
| `reveal` — a free retyping | tell the consumer *which* root was taken |
| marginalising $\llbracket c \rrbracket$ | sum/average over all real roots |
| composition files $Y$ into the latent | the intermediate branch choice is remembered |
| a *pure* model, $\llbracket c\rrbracket\cong 1$ | $D = 1$: the relation is a **function** in this polarity |

That last row is worth pausing on. **A factor is a function exactly when its branch count is
one.** Explicit learning is the special case $D \equiv 1$, and the entire "implicit" story
is about what happens when $D > 1$. [[Implicit Learners]]'s table row "may be multi-valued"
is the statement $D > 1$; "or have no solution" is $D = 0$.

## The discriminant

$D$ is not constant. It changes across the **discriminant locus**

$$
\Delta \;=\; \bigl\{x_o \;:\; \exists\, x_u,\ r_\Theta(x_o,x_u) = 0 \ \text{and} \ \operatorname{rank} J_u(x_o,x_u) < q\bigr\}
$$

— the set of observed values over which two branches collide. For suitable finite square systems elimination/discriminant polynomials describe
critical values (often a hypersurface); this is not a universal assertion that
one resultant describes every real projection's bifurcations. Roots escaping to
infinity can change the count without belonging to this finite critical locus.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[font=\small, scale=1.7]
  \draw[thick] (0,0) circle (1);
  \node at (0,1.35) {$V_\mathbb{R} = \{x^2+y^2-1=0\}$};
  \draw[->] (-2.1,-1.8) -- (2.4,-1.8); \node at (2.6,-1.8) {$x_o$};
  % fibre over x = -0.4 : two real points
  \draw[dashed] (-0.4,0.917) -- (-0.4,-1.8);
  \draw[dashed] (-0.4,-0.917) -- (-0.4,-1.8);
  \fill (-0.4,0.917) circle (0.04);
  \fill (-0.4,-0.917) circle (0.04);
  \fill (-0.4,-1.8) circle (0.04);
  \node at (-0.4,-2.1) {$D=2$};
  % fibre over x = 1 : one point, discriminant
  \draw[dashed] (1,0) -- (1,-1.8);
  \fill (1,0) circle (0.04);
  \fill (1,-1.8) circle (0.04);
  \node at (1.15,-2.1) {$D=1$};
  % x = 1.8 : none
  \fill (1.8,-1.8) circle (0.04);
  \node at (1.95,-2.1) {$D=0$};
  \node at (0,-2.5) {the discriminant is $x_o = \pm 1$};
\end{tikzpicture}
\end{document}
```

## What target-rank loss does and does not imply

The following effects require different assumptions; they do not all coincide:

1. **The implicit function theorem fails.** $J_u$ is singular, so
   $\partial x_u/\partial x_o = -J_u^{-1}J_o$ is undefined and the adjoint solve of
   [[Backpropagation by the Implicit Function Theorem]] is ill-conditioned. For the circle, |dy/dx|=|x|/√(1−x²) grows like the inverse **square root** of
   the distance to x=±1, not inverse distance. Other degeneracies have other rates.
2. **Inference is discontinuous.** Two real roots merge and become a complex-conjugate pair.
   Smooth separate branch labellings fail at collision. A root set can nevertheless
   approach the merged root continuously from the side with real solutions;
   set-valued continuity needs a specified topology.
3. **The entropy jumps.** With a uniform belief over branches,
   $\mathbf{H}^c = \log D$, which is a step function of $x_o$. The
   [statistical game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)'s regulariser is genuinely discontinuous here.
4. **The target site can flatten while the full Sampson metric stays finite.**
   Full J and target Jᵤ are different matrices. On the unit circle, with Σ_f=δI,
   JΣ_fJᵀ=4δ on the entire circle, including (±1,0). The y site precision is
   Jᵤ²/A=y²/δ and tends to **zero**, not infinity. It is loss of full residual
   rank (or covariance support), not target rank alone, that makes the undamped
   residual covariance singular. The implemented site is finite and flat at
   tangency; singular A still throws ([[polynomial]]).

> [!warning] This is intrinsic, not a defect
> Take the circle: $y = \pm\sqrt{1-x^2}$ is genuinely undefined for $|x|>1$ and genuinely
> two-valued for $|x|<1$, and $dy/dx \to \infty$ at $x=\pm1$. **No parametrisation, solver
> or regulariser removes this**, because it is a property of the relation, not of the
> representation. Any implicit learner that permits multi-valued relations inherits it.
>
> Detect target-rank loss and inspect singular values, rather than assume
> the condition number alone detects proximity. For a nonzero scalar Jᵤ,
> κ(Jᵤ)=1 even as it tends to zero. The circle tests check that fact and the
> square-root derivative rate. No IFT gradient is returned by PolynomialFactor.

The branch-index model below remains planned (§4). The implemented
`PolynomialModel` is a pure **local branch approximation**, with no latent branch
index or branch probabilities. This does not assert that the full relation has
only one solution.

## The branch belief

The natural output of an algebraic inversion is therefore **not a point** but a belief over
a finite set:

$$
c'_\pi(x_o) \;=\; \sum_{j=1}^{D} w_j\, \delta_{x_u^{(j)}}, \qquad w_j \propto \pi\bigl(x_u^{(j)}\bigr)
$$

a `SampleBelief` supported on the real roots, weighted by the prior. Two readings:

- **Selection as a prox.** Taking $\arg\max_j w_j$ is
  $\arg\min_{x_u : r=0} -\log\pi(x_u)$ — a proximal step onto the variety. That is exactly
  the structure of the diffusion family in [[ImplicitREDDiff]], with the hard constraint
  $r = 0$ in place of a soft energy. **The algebraic and diffusion families differ in how
  they enforce the constraint, not in what they compute.**
- **Entropy from branch count.** $\mathbf{H}^c(\pi, x_o) = -\sum_j w_j\log w_j \le \log D$.
  The abstract "entropy or regularizer" of [Definition 20](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) is here a
  perfectly ordinary discrete entropy over a set you can enumerate.

## Monodromy — the branches are not independently labelled

Transporting $x_o$ around a loop in $\mathbb{R}^p \setminus \Delta$ can **permute** the
branches. The resulting **monodromy group** $\subseteq S_D$ measures how globally
inconsistent any branch labelling is: if the monodromy is transitive, there is *no*
continuous global choice of branch, and the "elbow up" branch is only locally well defined.

Two consequences:

- Any implementation that caches "which branch we took last time" is making a **local**
  choice that cannot be made global. Message passing around a cycle in the factor graph can
  return to a different branch than it started on — a genuine and under-appreciated failure
  mode for [[Implicit Learners|equilibrium-style]] iteration.
- Monodromy is also a *tool*: monodromy loops are the cheapest known way to find many
  solutions of a polynomial system from one, and are how modern solvers bootstrap.

Related: [[Inference as Root Finding]], [[Open Models and Latent Channels]], [[Factors are Parameterized Statistical Games]], [[The Algebraic Factor as a Statistical Game]]
