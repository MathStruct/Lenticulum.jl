# Composition is Elimination

> Composing two algebraic factors into one is a Gröbner basis computation. It is
> **doubly exponential**, numerically unstable, and not even closed. This is the strongest
> single argument for the factor-graph architecture — `Mycelium.jl` exists so that you never
> have to do it.

## The composite relation

Given relations $R_1 \subseteq X\times Y$ and $R_2 \subseteq Y\times Z$ cut out by ideals
$I_1 \subseteq k[x,y]$ and $I_2 \subseteq k[y,z]$, the composite relation is

$$R_2 \circ R_1 \;=\; \bigl\{(x,z) \;:\; \exists\, y,\ (x,y)\in R_1 \ \wedge\ (y,z)\in R_2\bigr\}$$

Algebraically: form $I_1 + I_2 \subseteq k[x,y,z]$ (the fibre product / pullback), then
**project away $y$**. The ideal of the projection is the **elimination ideal**

$$\boxed{\;I_{12} \;=\; (I_1 + I_2)\;\cap\;k[x,z]\;}$$

computed by a Gröbner basis with respect to a block order in which $y > x,z$: the elements
of the basis not involving $y$ generate $I_{12}$.

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}[row sep=large, column sep=large]
& V(I_1+I_2) \subseteq X\times Y\times Z \arrow[dl] \arrow[dr] \arrow[dd, "\pi_{XZ}"description, near start] & \\
R_1 \subseteq X\times Y \arrow[dr] & & R_2 \subseteq Y\times Z \arrow[dl] \\
& \overline{\pi_{XZ}(V(I_1+I_2))} = V(I_{12}) &
\end{tikzcd}
\end{document}
```

## Failure 1: the composite is not a variety

**Chevalley's theorem**: the image of a variety under a projection is a **constructible
set** — a finite boolean combination of varieties — not a variety. Over $\mathbb{R}$ it is
worse: by **Tarski–Seidenberg** the image is **semialgebraic**, i.e. defined by equations
*and inequalities*.

Concretely, project the hyperbola $xy = 1$ to the $x$-axis: the image is
$\{x \ne 0\}$, which is not closed. Its Zariski closure is all of $\mathbb{R}$. So

$$V(I_{12}) \;=\; \overline{R_2\circ R_1} \;\supsetneq\; R_2\circ R_1$$

**Elimination computes the closure, and the closure is strictly bigger.** Composing two
algebraic factors and re-expressing the result as one algebraic factor therefore *loses
information*: you gain spurious solutions along the boundary.

> [!warning] This is a laxness, in the vault's sense
> The category of affine varieties and polynomial relations is **not closed under
> composition** — you must take Zariski closures, and the closure is a lax operation. This is
> the same shape as [[Composition of Bayesian Lenses|Remark 16]]'s lossy tensor and
> [[Scalar and Multivariate Energy|§5]]'s lax scalarisation: a projection that leaves the
> category, corrected by a closure. It is the third independent instance in this vault of
> the same pattern, which is some evidence the pattern is real.

## Failure 2: degree and generator blowup

Even accepting the closure, the composite is a *worse* object:

- **Degree.** Eliminating raises degree. Generically the composite of relations of degree
  $d_1, d_2$ has degree up to $d_1 d_2$ (a Bézout-type bound), so a chain of $L$ factors of
  degree $d$ has degree $d^L$. Since the parameter count is
  $\binom{N + d^L}{d^L}$ ([[The Veronese Parametrisation]]), this is instantly hopeless.
- **Number of generators.** A Gröbner basis can have far more elements than the input, and
  their coefficients grow explosively even for small inputs.

## Failure 3: the complexity is doubly exponential

- **Mayr–Meyer**: ideal membership in $k[x_1,\ldots,x_n]$ is **EXPSPACE-complete**, and
  Gröbner bases can require degrees $2^{2^{\Omega(n)}}$.
- For **zero-dimensional** ideals (finitely many solutions) it is much better: single
  exponential, and FGLM converts between orders in $O(nD^3)$ where $D$ is the solution count.
  But $D \le d^n$ already.

So the *good* case is single-exponential and the general case is doubly exponential. Neither
is a basis for a library operation.

## Failure 4: Gröbner bases are numerically unstable

This is the one that rules them out even at small sizes. **Leading terms are discontinuous
in the coefficients**: an arbitrarily small perturbation of an input coefficient can change
which monomial is leading, which changes the entire combinatorial structure of the basis, and
hence the output. There is no useful notion of "approximate Gröbner basis".

Since a *learned* $\Theta$ is a floating-point object fitted to noisy data, its Gröbner basis
is meaningless.

### Why not Gröbner: border bases

The numerically stable replacement is the **border basis** (Kehrein–Kreuzer; Mourrain),
which is defined relative to an order ideal of monomials rather than a term order and
therefore varies continuously with the coefficients. This is why
[[Fitting is a Nullspace Problem]] §"Doing it properly" produces a border basis: the AVI
family of algorithms was designed for exactly this reason.

Border bases fix *stability*; they do not fix *complexity* or the *closure* problem.

> [!note] Failures 1 and 2 are also *expressivity* statements
> Read the other way round, this section answers "can a flat graph do what a deep one does?".
> Failure 1 says the flattened relation is a **different** relation (the closure is strictly
> bigger); Failure 2 says that even accepting it, the flat model needs degree $d^L$ where the
> deep one needed $L$ factors of degree $d$. At $d = 1$ the bound gives $1$ and depth buys
> nothing — which is exactly the linear-Gaussian fragment. See
> [[Depth in Implicit Learning]].

## The conclusion: do not compose, schedule

Every failure above is a failure of the operation "turn two factors into one factor". None
of them is a failure of "keep two factors and pass messages between them".

This is precisely what the AutoBayes framework already prescribes. Its whole point
([[Bayesian Inversion]], [[Composition of Statistical Games]]) is that you attach local
inversions to local factors and compose *those*, obtaining a globally correct structure
without ever forming the global object. Read in the algebraic case, the statement becomes
concrete and sharp:

> **Composing the varieties requires elimination and is doubly exponential. Composing the
> *inversions* requires only a sequence of root-finds, each exponential in the local
> $q$ only.**

Compare the numbers. A chain of $L$ factors, each with $q$ unobserved channels and degree
$d$:

| strategy | cost |
|---|---|
| eliminate, then solve once | degree $d^L$, then $(d^L)^q$ roots — plus Gröbner |
| solve locally, pass messages | $L \times d^q$ roots |

$L\cdot d^q$ versus $d^{Lq}$. **This is the quantitative justification for `Mycelium.jl`**,
and it is the sharpest one available anywhere in the vault, because in this family both
sides can actually be counted.

## The residual role for elimination

Elimination is still the right tool for **compile-time, small-scale, structural** questions
that are asked once rather than in a loop:

- **Is this polarity well-posed?** The dimension of the elimination ideal
  $I \cap k[x_o]$ tells you whether the projection is dominant, hence whether generic $x_o$
  has any preimage at all. This is the exact, decidable version of `supports_polarity`.
- **Is the relation actually a function in this polarity?** Compute whether the projection
  is birational — i.e. whether $D = 1$ ([[Branches and the Discriminant]]).
- **What is the discriminant?** A resultant computation, done once per factor, giving a
  polynomial whose sign/vanishing you can then evaluate cheaply at run time.

All three are *static analysis of a single factor*, not a per-message operation, and at that
scale the complexity is survivable.

Related: [[Branches and the Discriminant]], [[Composition of Statistical Games]],
[[Algebraic Statistics Bridge]], [[Open Problems in Algebraic Implicit Learning]],
[[Depth in Implicit Learning]]
