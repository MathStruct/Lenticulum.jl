# The Parameter is a Grassmannian

> **Derivation.** The parameter of an algebraic factor is not a matrix; it is a $k$-plane.
> This is the cleanest concrete instance in the whole vault of
> [[Parametric Lens|Definition 2.5]]'s insistence that $P' \ne P$.

## The non-identifiability

If $A \in GL_k(\mathbb{R})$ then $r_{A\Theta} = A\,r_\Theta$, so

$$V(r_{A\Theta}) = V(r_\Theta) \qquad\text{and}\qquad \langle \text{rows of } A\Theta\rangle = \langle\text{rows of } \Theta\rangle$$

The variety, the ideal, and the relation all depend on $\Theta$ **only through its row
space**. The parameter is therefore genuinely a point of the **Grassmannian**

$$P \;=\; \mathrm{Gr}(k, m), \qquad \dim P = k(m-k)$$

not of $\mathbb{R}^{k\times m}$ (dimension $km$). The $k^2$ missing dimensions are pure gauge.

This is not pedantry. Three things go wrong if you ignore it:

1. **Gradient descent on $\Theta \in \mathbb{R}^{k\times m}$ drifts along the gauge
   directions**, changing the parameter without changing the model. Step sizes become
   meaningless; two runs that learned the same relation look maximally different.
2. **$\Theta \to 0$ is a global minimiser** of any residual objective, and it is reachable
   by descent. The normalisation $\Theta\Theta^\top = I_k$ of
   [[Fitting is a Nullspace Problem]] is not a regulariser bolted on to prevent collapse —
   it is a *chart* on $\mathrm{Gr}(k,m)$, and collapse is the symptom of using the wrong
   space.
3. **The "spurious polynomial" pathology** is the same phenomenon: a polynomial with tiny
   coefficients that "almost vanishes" everywhere is a point of $\mathbb{R}^{k\times m}$ near
   the origin, which is not a point of the Grassmannian at all.

## The Riemannian gradient, derived

Represent a point by $\Theta \in \mathbb{R}^{k\times m}$ with $\Theta\Theta^\top = I_k$ (a
Stiefel representative; the Grassmannian is the quotient by the $O(k)$ action). The
**horizontal** (tangent-to-$\mathrm{Gr}$) space at $\Theta$ is

$$T_\Theta \;=\; \{\Delta \in \mathbb{R}^{k\times m} \;:\; \Delta\Theta^\top = 0\}$$

i.e. row-wise orthogonal to the current row space. Given a Euclidean gradient
$G = \nabla_\Theta f$, the projection onto $T_\Theta$ is

$$\boxed{\;\Pi_\Theta(G) \;=\; G - G\,\Theta^\top\Theta\;}$$

*Check:* $\Pi_\Theta(G)\,\Theta^\top = G\Theta^\top - G\Theta^\top(\Theta\Theta^\top) = G\Theta^\top - G\Theta^\top = 0$. ✓

The update is then a **retraction**, not an addition:

$$\Theta^+ \;=\; \mathrm{qf}\bigl(\Theta - \eta\,\Pi_\Theta(G)\bigr)$$

where $\mathrm{qf}$ orthonormalises the rows (thin QR of the transpose, or a polar
factor). Edelman–Arias–Smith is the standard reference for the geometry; the point here is
only that **$\Theta^+ \ne \Theta + \Delta$**.

## This is exactly $(P, P')$

[[Parametric Lens|Definition 2.5]] carries a parameter **pair** $(P, P')$ with
$f^* : P\times A\times B' \to P'\times A'$, and the vault's note on it says the distinction
matters "for a parameter constrained to a manifold, where the update lives in the tangent
space, not in the manifold". Here that is not a hypothetical:

$$P = \mathrm{Gr}(k,m), \qquad P' = T_\Theta\,\mathrm{Gr}(k,m)$$

and [[Learning Components as Parametric Lenses|Definition 3.11]]'s gradient-update lens
$G^*(p, p') = p + p'$ is **type-incorrect**. The correct lens is

$$G(\Theta) = \Theta, \qquad G^*(\Theta, \Delta) = \mathrm{qf}\bigl(\Theta + \Pi_\Theta(\Delta)\bigr)$$

which is still a perfectly good lens $(P,P)\to(P,P')$ — the framework accommodates it
without modification. **The categorical setup was right and the naive implementation was
wrong**, which is a satisfying vindication of taking the $(P,P')$ distinction seriously.

## And it is the natural gradient Definition 27 asks for

[[Parameterized Statistical Game|Definition 27]] says the default semantics is gradient
descent with respect to the **Fisher information metric** — the Bayesian learning rule. On
$\mathrm{Gr}(k,m)$ the canonical (Riemannian) metric is
$\langle \Delta_1,\Delta_2\rangle = \operatorname{tr}(\Delta_1\Delta_2^\top)$ restricted to
the horizontal space, and $\Pi_\Theta$ above *is* the metric-correct gradient. So in this
family the natural gradient is not an expensive approximation of an intractable Fisher
matrix — it is a projection costing one $k\times m$ matrix product.

## Interaction with the rank-one structure

[[Backpropagation by the Implicit Function Theorem]] derives that each data point
contributes a **rank-one** cotangent $\bar\Theta_i = \lambda_i\, v_d(x_i)^\top$. So a
minibatch of size $B$ gives $G$ of rank $\le B$, and

$$\Pi_\Theta(G) = G - G\Theta^\top\Theta$$

is also rank $\le B$. **The Riemannian update is a rank-$B$ perturbation of a $k$-plane.**
That is a well-studied object (subspace tracking; incremental SVD), and it means the update
costs $O(Bkm)$ rather than $O(km^2)$ — which is the difference between feasible and not at
$m \sim 10^4$.

## Caveat: the Grassmannian is the right space only for a fixed $k$ and $d$

Model selection — choosing how many generators and what degree — moves you between
Grassmannians of different dimensions, and there is no smooth path. The **numerical rank
decision** of [[Fitting is a Nullspace Problem]] is therefore a discrete jump between
manifolds, not a continuous shrinkage. No continuous relaxation of it is known that behaves
well; see [[Open Problems in Algebraic Implicit Learning]] §4.

Related: [[Fitting is a Nullspace Problem]], [[Parametric Lens]], [[Parameterized Statistical Game]], [[Backpropagation by the Implicit Function Theorem]]
