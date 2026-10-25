#model #derivation

> A **toric model** is a discrete exponential family whose sufficient statistics are integer
> vectors: $p(x) \propto \theta^{a_x}$, a monomial in the parameters. As a factor it is a log-linear
> potential over discrete channels, and every slot of the statistical game is exact: messages are
> finite sums, the free energy is closed-form, the likelihood is concave with a moment-matching
> optimum, and the model's implicit equations are known binomials. It is the easiest member of the
> algebraic family to build, and it is implemented as `AlgebraicFactors.ToricFactor` over several discrete
> variables (§1.1–§1.5 only; fitting is still planned). [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] is the plan.

> Sources: Sullivant, *Algebraic Statistics*, AMS GSM 194, 2018; Drton, Sturmfels & Sullivant, *Lectures on Algebraic Statistics*, 2009; Sturmfels, *Gröbner Bases and Convex Polytopes*, AMS ULECT 8, 1996; Diaconis & Sturmfels, *Algebraic Algorithms for Sampling from Conditional Distributions*, Ann. Statist. 26(1), 1998; Geiger, Meek & Sturmfels, *On the Toric Algebra of Graphical Models*, Ann. Statist. 34(3), 2006; Lauritzen, *Graphical Models*, 1996; Darroch & Ratcliff, *Generalized Iterative Scaling for Log-Linear Models*, Ann. Math. Statist. 43(5), 1972; Fienberg & Rinaldo, *Maximum Likelihood Estimation in Log-Linear Models*, Ann. Statist. 40(2), 2012; Craciun, Dickenstein, Shiu & Sturmfels, *Toric Dynamical Systems*, J. Symbolic Comput. 44(11), 2009; the factor-level statements are derivations original to this vault; code: `lib/AlgebraicFactors.jl/src/toric.jl`; enumeration and tree-BP checks in `lib/AlgebraicFactors.jl/test/runtests.jl`.
>
> Bibliography: [[Bibliography#^sullivant2018algebraic|Sullivant 2018]] · [[Bibliography#^drton2009lectures|Drton et al. 2009]] · [[Bibliography#^sturmfels1996grobner|Sturmfels 1996]] · [[Bibliography#^diaconis1998algebraic|Diaconis & Sturmfels 1998]] · [[Bibliography#^geiger2006toric|Geiger et al. 2006]] · [[Bibliography#^lauritzen1996graphical|Lauritzen 1996]] · [[Bibliography#^darroch1972generalized|Darroch & Ratcliff 1972]] · [[Bibliography#^fienberg2012maximum|Fienberg & Rinaldo 2012]] · [[Bibliography#^craciun2009toric|Craciun et al. 2009]]
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Conditional Independence](https://mathstruct.org/CategoryTheory-ML-Wiki/Conditional-Independence)

## The model

Let the factor's channels be discrete variables with finite alphabets, and let $x$ range over the
$m$ joint states (the product of the alphabets). A **design matrix** $A \in \mathbb Z^{d \times m}$
gives each state a column $a_x$. With natural parameter $\eta \in \mathbb R^d$,

$$
p_\eta(x) = \exp\bigl(\langle \eta, a_x\rangle - \Lambda(\eta)\bigr), \qquad
\Lambda(\eta) = \log \sum_x e^{\langle \eta, a_x\rangle}.
$$

With $\theta_i = e^{\eta_i}$ this is $p(x) \propto \theta^{a_x}$, a monomial: hence *toric*. The model
$\mathcal M_A$ is the closure of $\{p_\eta\}$ in the probability simplex.

- **Independence** of two variables with alphabets $[a], [b]$: $x = (i, j)$, and $a_x$ stacks the
  indicator of row $i$ and the indicator of column $j$.
- **Hierarchical log-linear models**: one block of rows per generator (a set of variables), each
  the indicator of $x$'s cell in that generator's marginal table. Discrete **undirected graphical
  models** are the case whose generators are the cliques; their positive part is toric (Geiger, Meek
  & Sturmfels 2006). Bayesian networks with hidden variables are not toric in general; see
  [[Algebraic Statistics Bridge]] §"The semialgebraic gap".

**A gauge, again.** $p_\eta = p_{\eta'}$ iff $A^\top(\eta - \eta') \in \operatorname{span}\{\mathbf 1\}$: the parameter
is identified only modulo that subspace, which contains the null space of $A^\top$ and, when $\mathbf 1$
is in the row space of $A$, a constant-logit shift. As for [[The Parameter is a Grassmannian]],
the identifiable parameter is a quotient; independent logit differences are a chart. For the
unnormalised potential in a graph, a constant-logit shift rescales the potential and shifts its
energy by a constant.

**The all-ones vector.** Several statements below need $\mathbf 1$ in the row space of $A$ (constant
**nonzero** column sums suffice). For arbitrary $A$, replace $A$ by $\bar A = \binom{\mathbf 1^\top}{A}$
throughout; this does not change the model.

## Its implicit equations are binomials

When $\mathbf 1$ lies in the row space of $A$, $\mathcal M_A$ is the simplex part of the variety of the
**toric ideal**

$$
I_A = \bigl\langle\, p^{u^+} - p^{u^-} \;:\; u \in \ker_{\mathbb Z} A \,\bigr\rangle
$$

(Sturmfels 1996; Sullivant 2018). For independence these are the $2\times 2$ minors
$p_{ij}p_{kl} - p_{il}p_{kj}$ of [[Algebraic Statistics Bridge]]. So the two views of an algebraic
factor, a relation $V(I)$ and a statistical model, coincide here *with the relation known in
closed form*: no [[Fitting is a Nullspace Problem|nullspace fit]] is needed to find it.

Two practical points:

- A finite set of integer moves $u \in \ker_{\mathbb Z}A$ is a **Markov basis** iff its binomials
  $p^{u^+} - p^{u^-}$ generate $I_A$; Diaconis & Sturmfels (1998) use one to
  sample tables with fixed sufficient statistics (exact conditional tests). A lattice basis of
  $\ker_{\mathbb Z} A$ is in general **not** a Markov basis, and computing one needs specialised
  software (4ti2 and similar).
- Membership needs no generators. For **positive** $p$ (and $\mathbf 1$ in the row space of
  $A$): $p \in \mathcal M_A$ iff $\log p$ lies in the row space of $A$, a linear least-squares
  test (derivation: $\log p_\eta = A^\top\eta - \Lambda(\eta)\mathbf 1$; conversely
  $\log p = A^\top \eta'$ gives $p = p_{\eta'}$, since $p$ sums to one).

## As a factor every slot is exact

The factor's potential is $\psi_\eta(x) = e^{\langle \eta, a_x\rangle}$ in a graph (the graph's partition
function normalises), and $p_\eta$ when the factor stands alone as a game. All derivations:

- **Pointwise energy**: $E(x) = -\langle \eta, a_x\rangle$, plus $\Lambda(\eta)$ when normalised; scalar.
- **Message** to a channel $v$: $\mu_{f\to v}(x_v) = \sum_{x_{\partial f\setminus v}} \psi_\eta(x) \prod_{u\ne v}\mu_{u\to f}(x_u)$, an exact
  categorical belief, with clamped channels entering as indicators. The cost is $m$ per message,
  exponential in the factor's arity as for any table factor; a hierarchical model is better split
  into one factor per generator, which is a junction-tree question
  ([[LP-010 Where the graph of a problem comes from]]).
- **Bethe term**: with the factor belief $b_f \propto \psi_\eta \prod_u \mu_{u\to f}$, the energy part is
  $-\langle \eta, A b_f\rangle$ and the entropy part $H(b_f)$, both finite sums.
- **Learning**: for counts $u$ with total $n$, $\nabla_\eta \tfrac1n\log L = A u/n - A p_\eta$. $\Lambda$ is
  convex, so the log-likelihood is concave; its Hessian is $-\operatorname{Cov}_{p_\eta}(a_x)$. **Birch's
  theorem**: for $n > 0$ there is a unique $\hat p \in \mathcal M_A$ (the closure) with $A\hat p = Au/n$, the
  **extended MLE** (Sullivant 2018; Fienberg & Rinaldo 2012, Theorem 9 in the numbering of arXiv:1104.3618). The ordinary MLE, at a
  finite $\eta$, exists iff $\hat p$ is strictly positive, equivalently iff $Au/n$ lies in the relative
  interior of the convex hull of the columns $a_x$ (Fienberg & Rinaldo 2012: Theorem 9 states it as a facial-set condition on the marginal cone,
  Theorem 13 for general exponential families as $Au/n$ in the relative interior of the convex support). So zero counts can make it fail: for independence on a full table
  it fails exactly when a row or column total is zero, and the extended MLE is still the outer
  product of the empirical margins. (Counterexample to the cone version: $A = (1\ 2)$, $u = (1, 0)$.)
- **Algorithms**: iterative proportional fitting for hierarchical models; generalised iterative
  scaling (Darroch & Ratcliff 1972), which needs non-negative features with constant positive
  column sums (a correction row is added when they are not); Newton with the Hessian above,
  after removing the gauge directions.
- **Closed form**: for a **decomposable** graphical model with strictly positive empirical clique
  margins, the MLE is $\hat p = \prod_C \hat p_C / \prod_S \hat p_S$, one separator factor per edge of a
  junction tree (Lauritzen 1996); for the extended MLE, states with a zero separator margin get
  probability zero. On the junction tree the entropy is exactly
  $\sum_C H(b_C) - \sum_{\text{edges}} H(b_S)$; the Bethe counting on the original variables need not be
  exact for the clique factorisation. A test oracle, and the setting of
  [[RQ-004 Is the Bethe Criterion Consistent for Structure Learning]].

## Why it is the easy case

Against [[Open Problems in Algebraic Implicit Learning]]:

| problem there | for a toric factor |
|---|---|
| §1 the real locus may be tiny | the positive part is the image of a parametrisation: nonempty, of dimension $\operatorname{rank}\bar A - 1$ |
| §2 the $\binom{N+d}{d}$ wall | the support is the columns of $A$, chosen by the model's structure |
| §4 model selection jumps between manifolds | it is the choice of generators, i.e. classical log-linear model selection; still discrete |
| §5 approximate vanishing ideals | not needed: the ideal is known |
| §7 the discriminant | none: $\eta \mapsto p_\eta$ is smooth, and injective modulo the gauge |

What it does not escape: tables grow exponentially with arity, and Markov bases can be large.

## Beyond tables

For a mass-action reaction network whose rate constants admit a complex-balanced equilibrium,
the strictly positive steady states are the positive points of a toric variety (Craciun,
Dickenstein, Shiu & Sturmfels 2009, Theorem 7 in the numbering of arXiv:0708.3431); boundary
steady states are not covered. A route from this note to
[[Differential Algebra and DAE Factors]]. Not planned yet.

Related: [[Algebraic Statistics Bridge]], [[Algebraic Implicit Learners]], [[The Algebraic Factor as a Statistical Game]], [[Grassmann Mixing and Plücker Coordinates]], [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]]
