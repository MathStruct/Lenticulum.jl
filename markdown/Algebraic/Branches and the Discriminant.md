# Branches and the Discriminant

> The most satisfying concrete grounding in this note set: for an algebraic factor, the
> abstract latent space $\llbracket c \rrbracket$ of [[Open Model|Definition 1]] is
> **literally the set of solution branches**, and the discriminant is where four different
> things fail simultaneously.

## The projection is a branched cover

Fix a polarity. The projection

$$\pi \;:\; V_\mathbb{R}(r_\Theta) \longrightarrow \mathbb{R}^p, \qquad (x_o, x_u) \mapsto x_o$$

is, away from a proper closed subset, a **covering map of degree $D$**: every $x_o$ in a
connected component of the complement has exactly $D$ preimages, varying smoothly.

$$\boxed{\;\llbracket c \rrbracket \;=\; \pi^{-1}(x_o) \;=\; \{1,\ldots,D\}\;\text{, the branch index}\;}$$

So an algebraic factor's forward kernel really does have the type of
[[Open Model|Definition 1]],

$$c \;:\; X \rightsquigarrow \llbracket c \rrbracket \times Y$$

and the latent variable has a **meaning**: *which solution you are on*. Elbow up or elbow
down. Which of the ten essential matrices. This is not a bookkeeping device; it is the
physically meaningful hidden state.

Everything the vault says abstractly about $\llbracket c \rrbracket$ now has a concrete
reading:

| abstract ([[Open Model]], [[Composition of Open Models]]) | algebraic reading |
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

$$\Delta \;=\; \bigl\{x_o \;:\; \exists\, x_u,\ r_\Theta(x_o,x_u) = 0 \ \text{and} \ \operatorname{rank} J_u(x_o,x_u) < q\bigr\}$$

— the set of observed values over which two branches collide. For a square system,
$\Delta = \{x_o : \operatorname{Res}(\cdot) = 0\}$, the vanishing of a resultant, hence
itself an algebraic hypersurface.

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

## Four failures, one locus

At $\Delta$, all of the following happen **at the same points**, and they are the same
phenomenon seen from four sides:

1. **The implicit function theorem fails.** $J_u$ is singular, so
   $\partial x_u/\partial x_o = -J_u^{-1}J_o$ is undefined and the adjoint solve of
   [[Backpropagation by the Implicit Function Theorem]] is ill-conditioned. Gradients blow
   up like $1/\mathrm{dist}(x_o,\Delta)$.
2. **Inference is discontinuous.** Two real roots merge and become a complex-conjugate pair.
   The map $x_o \mapsto \{\text{real roots}\}$ is continuous *only* on the complement of
   $\Delta$.
3. **The entropy jumps.** With a uniform belief over branches,
   $\mathbf{H}^c = \log D$, which is a step function of $x_o$. The
   [[Statistical Game|statistical game]]'s regulariser is genuinely discontinuous here.
4. **The Sampson precision blows up.** $(JJ^\top)^{-1}$ of
   [[Algebraic versus Geometric Distance]] is singular exactly on $\Delta$ — the noise model
   says "infinite uncertainty along the collapsing direction", which is correct and useless.

> [!warning] This is intrinsic, not a defect
> Take the circle: $y = \pm\sqrt{1-x^2}$ is genuinely undefined for $|x|>1$ and genuinely
> two-valued for $|x|<1$, and $dy/dx \to \infty$ at $x=\pm1$. **No parametrisation, solver
> or regulariser removes this**, because it is a property of the relation, not of the
> representation. Any implicit learner that permits multi-valued relations inherits it.
>
> The honest engineering response is to *detect* it — $\kappa(J_u)$ is a cheap, exact
> proximity indicator to $\Delta$ — and to report it, rather than to return a confident
> gradient that is numerically meaningless.

## The branch belief

The natural output of an algebraic inversion is therefore **not a point** but a belief over
a finite set:

$$c'_\pi(x_o) \;=\; \sum_{j=1}^{D} w_j\, \delta_{x_u^{(j)}}, \qquad w_j \propto \pi\bigl(x_u^{(j)}\bigr)$$

a `SampleBelief` supported on the real roots, weighted by the prior. Two readings:

- **Selection as a prox.** Taking $\arg\max_j w_j$ is
  $\arg\min_{x_u : r=0} -\log\pi(x_u)$ — a proximal step onto the variety. That is exactly
  the structure of the diffusion family in [[ImplicitREDDiff]], with the hard constraint
  $r = 0$ in place of a soft energy. **The algebraic and diffusion families differ in how
  they enforce the constraint, not in what they compute.**
- **Entropy from branch count.** $\mathbf{H}^c(\pi, x_o) = -\sum_j w_j\log w_j \le \log D$.
  The abstract "entropy or regularizer" of [[Statistical Game|Definition 20]] is here a
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

Related: [[Inference as Root Finding]], [[Open Model]], [[Statistical Game]], [[The Algebraic Factor as a Statistical Game]]
