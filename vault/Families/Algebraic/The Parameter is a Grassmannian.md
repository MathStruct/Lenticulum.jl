#model #derivation

> **Derivation.** The parameter of an algebraic factor is not a matrix; it is a $k$-plane.
> This is the cleanest concrete instance in the whole vault of
> [Definition 2.5](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens)'s distinction between parameter points and updates.
> Grassmannian projection, QR retraction and wrapped GD/momentum are implemented;
> the polynomial factor and its Sampson scalarisation remain planned.

> Sources: original to this vault (design and analysis); Edelman, Arias & Smith (1998), *The Geometry of Algorithms with Orthogonality Constraints*; Absil, Mahony & Sepulchre (2008), *Optimization Algorithms on Matrix Manifolds*; code: `lib/AlgebraicFactors.jl/src/grassmann.jl`, `lib/AlgebraicFactors.jl/test/runtests.jl` (geometry and trace-objective descent, not polynomial residual scalarisation).
>
> Bibliography: [[Bibliography#^edelman1998geometry|Edelman et al. 1998]] · [[Bibliography#^absil2008optimization|Absil et al. 2008]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Gradient Based Learning with Parametric Lenses](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses)

## The non-identifiability

If $A \in GL_k(\mathbb{R})$ then $r_{A\Theta} = A\,r_\Theta$, so

$$V(r_{A\Theta}) = V(r_\Theta) \qquad\text{and}\qquad \langle \text{rows of } A\Theta\rangle = \langle\text{rows of } \Theta\rangle$$

The variety, the ideal, and the relation all depend on $\Theta$ **only through its row
space**. The parameter is therefore genuinely a point of the **Grassmannian**

$$P \;=\; \mathrm{Gr}(k, m), \qquad \dim P = k(m-k)$$

not of $\mathbb{R}^{k\times m}$ (dimension $km$). The $k^2$ missing dimensions are pure gauge.

This is not pedantry. Three things go wrong if you ignore it:

1. **A residual loss can change along the gauge directions**: changing the row
   scales or basis changes the residual while preserving its zero set. Euclidean
   distances between matrices therefore need not measure differences between
   relations. A truly gauge-invariant objective has no first-order derivative
   along gauge directions; drift is not inevitable for every objective.
2. **$\Theta = 0$ is a global minimiser** of an unnormalised squared-residual objective, and it is reachable
   by descent. The normalisation $\Theta\Theta^\top = I_k$ of
   [[Fitting is a Nullspace Problem]] is not a regulariser bolted on to prevent collapse —
   it chooses a Stiefel representative of a point of $\mathrm{Gr}(k,m)$, still
   modulo orthogonal row-basis changes. It is not a coordinate chart; collapse
   leaves the fixed-rank parameter space.
3. **Small raw residuals can come from small coefficients** rather than a better
   geometric fit. Scaling a full-rank matrix towards zero preserves its
   Grassmann point at every nonzero scale; only the rank-deficient limit
   leaves the fixed-rank space. Orthonormal representatives remove that scale freedom.

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

[Definition 2.5](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) carries a parameter **pair** $(P, P')$ with
$f^* : P\times A\times B' \to P'\times A'$, and the vault's note on it says the distinction
matters "for a parameter constrained to a manifold, where the update lives in the tangent
space, not in the manifold". Here that is not a hypothetical:

$$P = \mathrm{Gr}(k,m), \qquad P' = T_\Theta\,\mathrm{Gr}(k,m)$$

and [Definition 3.11](https://mathstruct.org/CategoryTheory-ML-Wiki/Gradient-Based-Learning-with-Parametric-Lenses)'s gradient-update lens
$G^*(p, p') = p + p'$ is **type-incorrect**. The correct lens is

$$G(\Theta) = \Theta, \qquad G^*(\Theta, \Delta) = \mathrm{qf}\bigl(\Theta + \Pi_\Theta(\Delta)\bigr)$$

which is still a perfectly good lens $(P,P)\to(P,P')$ — the framework accommodates it
without modification. **The categorical setup was right and the naive implementation was
wrong**, which is a satisfying vindication of taking the $(P,P')$ distinction seriously.

## The metric choice is not a Fisher-metric identification

[Definition 27](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#parameterized-statistical-games-and-their-gradients) uses the Fisher information metric of a statistical model.
Here the canonical Grassmann metric on horizontal lifts is
$\langle\Delta_1,\Delta_2\rangle=\operatorname{tr}(\Delta_1\Delta_2^\top)$,
and $\Pi_\Theta(G)$ is the Riemannian gradient for **that metric**.
No statistical model or Fisher information calculation in this note identifies
the two metrics. The earlier claim that the projection automatically computes
the Fisher natural gradient was unsupported. The implementation makes only the
canonical-metric claim, checked by the first-order trace-objective descent test.
The dense projection costs $O(k^2m)$ when evaluated as $(G\Theta^\top)\Theta$.

## Interaction with the rank-one structure

[[Backpropagation by the Implicit Function Theorem]] derives that each data point
contributes a **rank-one** cotangent $\bar\Theta_i = \lambda_i\, v_d(x_i)^\top$. So a
minibatch of size $B$ gives $G$ of rank $\le B$, and

$$\Pi_\Theta(G) = G - G\Theta^\top\Theta$$

is also rank $\le B$. **The Riemannian update is a rank-$B$ perturbation of a $k$-plane.**
If $G=LV^\top$ is retained in factored form, its projection can be formed by
projecting $V$ before multiplying by $L$, at cost $O(Bkm)$. This does not make
the complete QR update cost $O(Bkm)$: the implemented dense retraction costs
$O(k^2m)$ and does not exploit a supplied low-rank gradient representation.
Forming an $m\times m$ projector is unnecessary in either case.

## Caveat: the Grassmannian is the right space only for a fixed $k$ and $d$

Model selection — choosing how many generators and what degree — moves you between
Grassmannians of different dimensions, and there is no smooth path. The **numerical rank
decision** of [[Fitting is a Nullspace Problem]] is therefore a discrete jump between
manifolds, not a continuous shrinkage within a fixed Grassmannian. This implementation
has no variable-rank relaxation; see [[Open Problems in Algebraic Implicit Learning]] §4.

Related: [[Grassmann Mixing and Plücker Coordinates]], [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]], [[Fitting is a Nullspace Problem]], [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens), [[Factors are Parameterized Statistical Games]], [[Backpropagation by the Implicit Function Theorem]]
