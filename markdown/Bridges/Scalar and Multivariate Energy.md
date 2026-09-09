# Scalar and Multivariate Energy

> **This note is Lenticulum's own contribution, not the paper's.** AutoBayes has one
> $[0,\infty]$-valued energy. We keep two, and adapt the chain rule accordingly.
> Requested design constraint; here is the construction and its proof obligations.

## 1. What the paper does, and what it costs

[[Statistical Game|Definition 20]] types the energy as

$$l^c : X \times \llbracket c \rrbracket \times Y \longrightarrow [0,\infty]$$

and [[Composition of Statistical Games|Definition 22]] composes it by **addition**:

$$l^{dc}(x,a,y,b,z) = l^c(x,a,y) + l^d(y,b,z)$$

That composition law is precisely the statement that $([0,\infty], +, 0)$ is a commutative
monoid and $l$ is monoid-valued. Fine — but **addition is a lossy projection**. The moment
you add, you can no longer say:

- which factor contributed how much (per-factor loss attribution);
- *in which coordinates* a factor is wrong (the residual direction, not just its size);
- what the Jacobian of the residual is — and therefore no Gauss–Newton step, no Fisher
  metric, no implicit function theorem.

The last one is fatal for us, because [[Parameterized Statistical Game|Definition 27]]
explicitly says the default semantics is gradient descent **with respect to the Fisher
information metric**, and because [[Implicit Learners|implicit inference]] is root-finding
on a vector residual. Both need the vector.

## 2. Energy spaces

> **Definition (energy space).** An *energy space* is a pair $(E, K)$ of a real topological
> vector space $E$ in which barycentres of the relevant measures exist, together with a
> closed convex cone $K \subseteq E$ with $K \cap -K = \{0\}$. $K$ induces the partial order
> $e \le e' :\Leftrightarrow e' - e \in K$. Energies take values in $K$.
>
> The **direct sum** is $(E_1 \oplus E_2,\; K_1 \oplus K_2)$, with unit $(0, \{0\})$.

$([0,\infty] \subseteq \mathbb{R})$ is the terminal-ish example. $\mathbb{R}^k$ with
$K = \mathbb{R}^k$ (no order) or $K = \mathbb{R}^k_{\ge 0}$ (componentwise) are the ones we
use. The cone exists so that "energy is non-negative" still means something and so that
monotonicity of scalarisations is expressible.

> **Definition (scalarisation).** A *scalarisation* of $(E,K)$ is a map
> $\sigma : E \to \mathbb{R}$ with $\sigma(0) = 0$, $\sigma(K) \subseteq [0,\infty]$, and
> $\sigma$ monotone for $\le_K$. It is **linear** if $\sigma \in K^*$ (the dual cone), and
> **convex** otherwise.

Two scalarisations do all the work:

| $\sigma$ | $E$ | reading |
|---|---|---|
| $\sigma_\lambda(e) = \langle \lambda, e\rangle$, $\lambda \in K^*$ | any | precision / temperature weighting; $\beta$-VAE; the $\rho$'s of [[ImplicitREDDiff]] |
| $\sigma_{\|\cdot\|}(e) = \tfrac12\|e\|_M^2$ | $\mathbb{R}^k$ | least squares on a residual; $M$ a precision matrix |

## 3. The multivariate statistical game

> **Definition (multivariate statistical game).** A *multivariate statistical game*
> $c : X \multimap Y$ over an energy space $(E_c, K_c)$ consists of
>
> - a [[Bayesian Lens]] $(c, c')$,
> - a **vector energy** $\mathbf{l}^c : X \times \llbracket c \rrbracket \times Y \to K_c$,
> - a **vector entropy** $\mathbf{H}^c : \mathcal{P}X \times Y \to K_c$,
> - a scalarisation $\sigma_c : E_c \to \mathbb{R}$,
>
> with **vector loss** $\mathbf{F}^c : \mathcal{P}X \times Y \to E_c$
> $$\mathbf{F}^c(\pi, y) \;=\; \mathop{\mathbb{E}}_{(x,a)\sim c'_\pi(y)}\bigl[\mathbf{l}^c(x,a,y)\bigr] \;-\; \mathbf{H}^c(\pi, y)$$
> and **scalar loss** $F^c := \sigma_c \circ \mathbf{F}^c$.

Note the entropy is promoted to $E_c$ as well. It must be: otherwise $\mathbb{E}[\mathbf{l}] - H$
is a type error. In practice $\mathbf{H}^c$ is usually $H^c \cdot u$ for a fixed
$u \in K_c$ — the entropy is scalar but has to be told *which coordinate it regularises*.
That is not busywork: it is what lets you regularise one block of a factor and not another.

Recovering the paper: take $E_c = \mathbb{R}$, $K_c = [0,\infty)$, $\sigma_c = \mathrm{id}$.

## 4. The adapted chain rule

> **Definition (composition).** For $c : X \multimap Y$ over $E_c$ and $d : Y \multimap Z$
> over $E_d$, the composite $d \diamond c$ is over $E_{dc} := E_c \oplus E_d$, with
>
> $$\mathbf{l}^{dc}(x,a,y,b,z) \;=\; \bigl(\;\mathbf{l}^c(x,a,y),\;\; \mathbf{l}^d(y,b,z)\;\bigr)$$
> $$\mathbf{H}^{dc}(\pi,z) \;=\; \Bigl(\;\mathop{\mathbb{E}}_{(y,b)\sim d'_{c_*\pi}(z)}\bigl[\mathbf{H}^c(\pi,y)\bigr],\;\; \mathbf{H}^d(c_*\pi, z)\;\Bigr)$$
> $$\sigma_{dc}(e_c, e_d) \;=\; \sigma_c(e_c) + \sigma_d(e_d)$$

**The addition of Definition 22 has become a direct sum.** That is the entire change. The
entropy law is unchanged in shape — still averaged under the downstream inversion, still
evaluated at the pushforward prior — it just lands in a summand instead of being added in.

> **Theorem (multivariate chain rule).**
> $$\mathbf{F}^{dc}(\pi, z) \;=\; \Bigl(\;\mathop{\mathbb{E}}_{(y,b)\sim d'_{c_*\pi}(z)}\bigl[\mathbf{F}^c(\pi,y)\bigr],\;\; \mathbf{F}^d(c_*\pi, z)\;\Bigr) \;\in\; E_c \oplus E_d$$

*Proof.* Expand, using linearity of $\mathbb{E}$ and that the tuple's components are
independent:
$$\mathbf{F}^{dc} = \mathop{\mathbb{E}}_{(x,a,y,b)}\bigl[\mathbf{l}^{dc}\bigr] - \mathbf{H}^{dc}
= \Bigl(\mathop{\mathbb{E}}_{(y,b)}\bigl[\mathop{\mathbb{E}}_{(x,a)}[\mathbf{l}^c] - \mathbf{H}^c\bigr],\;\; \mathop{\mathbb{E}}[\mathbf{l}^d] - \mathbf{H}^d\Bigr)$$
which is the claim. $\square$

Compare [[Composition of Statistical Games|Theorem 23]]: same recursion, with $+$ replaced
by a pair. It is not a weaker theorem — it is the *same* theorem before the projection.

### Graded by the graph

Iterating, a graph $G$ of factors has

$$E_G \;=\; \bigoplus_{f \in \mathrm{Factors}(G)} E_f$$

**The total energy of a factor graph is a vector indexed by its factors**, and within each
factor by that factor's own residual coordinates. Per-factor loss attribution is not a
feature bolted on for logging; it is what the composite loss *is*, before you collapse it.

## 5. How the two are related: scalarisation is lax

> **Proposition.** Let $\sigma_{dc} = \sigma_c \oplus \sigma_d$ as above.
>
> 1. If $\sigma_c$ is **linear**, then $\sigma_{dc} \circ \mathbf{F}^{dc} = F^{dc}$: the
>    scalar chain rule of Theorem 23 holds exactly, and scalarisation is a **strict
>    morphism of games**.
> 2. If $\sigma_c$ is **convex**, then $\sigma_{dc} \circ \mathbf{F}^{dc} \le F^{dc}$, with
>    $$F^{dc}(\pi,z) \;-\; \sigma_{dc}\bigl(\mathbf{F}^{dc}(\pi,z)\bigr) \;=\;
>    \underbrace{\mathop{\mathbb{E}}_{(y,b)}\bigl[\sigma_c(\mathbf{F}^c)\bigr] - \sigma_c\Bigl(\mathop{\mathbb{E}}_{(y,b)}\bigl[\mathbf{F}^c\bigr]\Bigr)}_{\text{Jensen gap}} \;\ge\; 0$$

*Proof.* $\sigma_{dc}(\mathbf{F}^{dc}) = \sigma_c\bigl(\mathbb{E}[\mathbf{F}^c]\bigr) + \sigma_d(\mathbf{F}^d)$
by definition of $\sigma_{dc}$ and the theorem above, while Theorem 23 gives
$F^{dc} = \mathbb{E}[\sigma_c(\mathbf{F}^c)] + \sigma_d(\mathbf{F}^d)$. Subtract; the
$\sigma_d$ terms cancel; linearity gives equality and Jensen gives the inequality. $\square$

**The only place the two chain rules can disagree is $\sigma(\mathbb{E}[\cdot])$ versus
$\mathbb{E}[\sigma(\cdot)]$** — i.e. the expectation over the *downstream* inversion. Nowhere
else. That is a tight and checkable statement.

### The gap, computed

For $\sigma_c = \tfrac12\|\cdot\|^2$ on $E_c = \mathbb{R}^k$ the Jensen gap is exactly a
variance:

$$F^{dc} \;=\; \sigma_{dc}\bigl(\mathbf{F}^{dc}\bigr) \;+\; \tfrac12 \operatorname{tr}\operatorname{Cov}_{(y,b)\sim d'_{c_*\pi}(z)}\bigl(\mathbf{F}^c(\pi,y)\bigr)$$

So the multivariate composite is the **squared mean residual** and the scalar composite is
the **mean squared residual**; the difference is the posterior variance of the upstream
factor's loss. This is a genuine bias/variance decomposition of the composite objective,
and the variance term is a first-class diagnostic: *how much does the downstream posterior
disagree with itself about what the upstream factor should be doing?*

> [!note] This mirrors Remark 26 exactly
> AutoBayes measures the laxness of the *tensor* by mutual information. We measure the
> laxness of *scalarisation* by a variance. Both are "the amount of structure destroyed by
> a projection, quantified". Report them; do not hide them.

## 6. The gradient, adapted

For a [[Parameterized Statistical Game|parameterized]] multivariate game the derivative of
$\mathbf{F}^c$ is not a gradient but a **Jacobian**

$$J^c_\theta \;:=\; D_\theta \mathbf{F}^c(\pi, y; \theta) \;\in\; \mathrm{Hom}(\Theta, E_c)$$

and the scalar gradient is obtained by pulling back along the scalarisation's differential:

$$\nabla_\theta F^c \;=\; \bigl(J^c_\theta\bigr)^{\!*}\, \mathrm{d}\sigma_c\bigl(\mathbf{F}^c\bigr)$$

For $\sigma = \tfrac12\|\cdot\|^2$ this is the familiar $J^\top r$.

**Composite.** By the multivariate chain rule, and keeping only the terms
[[Composition of Gradients|Definition 29]] keeps:

$$D_{(\theta,\varphi)}\mathbf{F}^{dc} \;=\;
\begin{pmatrix}
\mathbb{E}_{(y,b)}\bigl[J^c_\theta\bigr] & \ast_1 \\[4pt]
\ast_2 & J^d_\varphi
\end{pmatrix}
\;:\; \Theta \times \Phi \longrightarrow E_c \oplus E_d$$

- $\ast_1 = D_\varphi \mathbb{E}_{(y,b)\sim d'(\cdot\,;\varphi)}[\mathbf{F}^c]$ — $\varphi$ moves the sampling distribution;
- $\ast_2 = D_\theta \mathbf{F}^d(c(\theta)_*\pi, z; \varphi)$ — $\theta$ moves the pushforward prior.

Definition 29 is the **block-diagonal** part. The laxness of the gradient assignment is
unchanged by going multivariate; it is orthogonal to the scalarisation question. Good — the
two approximations do not interact.

### Why the Jacobian is the point

Once you have $J^c_\theta$ rather than only $\nabla_\theta F^c$, you get for free:

1. **Gauss–Newton / Levenberg–Marquardt.** $\nabla^2_\theta F \approx J^\top J$, positive
   semidefinite by construction, no Hessian needed. $J^\top r$ alone cannot produce it —
   **scalarising early destroys the metric, irrecoverably.**
2. **The Fisher metric of Definition 27.** When $\mathbf{l}^c$ is the vector of per-component
   log-density contributions, its Jacobian is the score and
   $\mathcal{I}(\theta) = \mathbb{E}[s s^\top] = \mathbb{E}[J^\top J]$. The natural gradient
   of the **Bayesian learning rule** is therefore available compositionally, from data the
   multivariate energy already carries. This is the strongest single argument for the design.
3. **The implicit function theorem.** For a residual factor $r_\theta(x) = 0$ split by
   [[Channels and Polarity|polarity]] into $x = (x_{obs}, x_{unobs})$,
   $$\frac{\partial x_{unobs}}{\partial x_{obs}} = -\Bigl(\frac{\partial r}{\partial x_{unobs}}\Bigr)^{-1}\frac{\partial r}{\partial x_{obs}}$$
   which needs the square Jacobian block of the *vector* residual. A scalar energy cannot
   even express the shape requirement $\dim E_c = \dim x_{unobs}$.
4. **Post-hoc reweighting.** $\sigma_\lambda$ can be changed without recomposing the graph:
   $\beta$-annealing, precision scheduling, curriculum weights, Pareto fronts. With the
   scalar energy each $\lambda$ is a different graph.

## 7. Interface consequences

```julia
energyspace(f)              # -> AbstractEnergySpace, the E_f of this factor
energy(f, x, ps, st)        # -> element of K_f          (the vector 𝐥)
entropy(f, π, y, ps, st)    # -> element of K_f          (the vector 𝐇)
scalarisation(f)            # -> AbstractScalarisation σ_f
scalarise(σ, e)             # -> Real
islinear(σ)                 # -> Bool: strict vs lax composition (Prop. §5)
```

`islinear` is not decoration: it is the trait that tells the composition machinery whether
`scalarise ∘ compose == compose ∘ scalarise` may be assumed, and therefore whether the
scalar loss can be accumulated eagerly (cheap) or must be deferred until the vector loss is
assembled (correct). See [[energy]] for the implementation and its difficulties.

Related: [[Statistical Game]], [[Composition of Statistical Games]], [[Composition of Gradients]], [[Implicit Learners]], [[energy]]
