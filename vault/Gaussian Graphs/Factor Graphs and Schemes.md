#comparison #derivation

> Three precise connections to algebraic geometry: singular statistical evidence has an
> RLCT correction, dual-number points encode Jacobian kernels, and linear relations admit
> a cellular sheaf of consistent states. None makes a general probabilistic factor graph a scheme.

> Sources: Watanabe, *A Widely Applicable Bayesian Information Criterion* (2013), §3 Fundamental Conditions (1)–(4), Theorems 1–2, eqs. (15)–(16), Lemma 3; The Stacks Project, Definition 26.9.1 (tag 01II), Definitions 33.16.1 and 33.16.3, Lemma 33.16.4 (tag 0B28; numbering accessed 2026-10-27); Hansen & Ghrist, *Toward a Spectral Theory of Cellular Sheaves* (2019 author PDF / arXiv:1808.01513v2), Definitions 2.4–2.5, §2.2.2, Theorem 3.1. The constructions and elementary proofs below are original to this vault; code: `lib/AlgebraicFactors.jl/src/toric.jl`, `lib/AlgebraicFactors.jl/src/plucker.jl`, `lib/AlgebraicFactors.jl/src/polynomial.jl`, `lib/Mycelium.jl/src/free_energy.jl` (connections, not a scheme or sheaf implementation).
>
> Bibliography: [[Bibliography#^watanabe2013wbic|Watanabe 2013]] · [[Bibliography#^stacksproject|The Stacks Project Authors 2026]] · [[Bibliography#^hansen2019sheaves|Hansen & Ghrist 2019]]
>
> Theory (CT-ML wiki): [Information Geometry of Probabilistic Programs](https://mathstruct.org/CategoryTheory-ML-Wiki/Information-Geometry-of-Probabilistic-Programs) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)

## Scope

[[LP-012 Factor graphs and schemes]] asked what schemes add to [[Algebraic Statistics Bridge]].
A scheme is a locally ringed space locally isomorphic to an affine scheme (Stacks,
Definition 26.9.1). Attaching vector spaces to a graph supplies neither these local rings
nor affine open charts. The three connections below concern different structures; they do
not identify the objects.

[[Toric Models as Factors]], [[Grassmann Mixing and Plücker Coordinates]] and
[[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] already use polynomial equations.
Positive probabilities, real roots, a noise metric and a probability normaliser are additional
data: the equations alone do not supply them. See [[Varieties Ideals and Real Nullstellensatz]]
and [[Branches and the Discriminant]].

## 1. Singular evidence: what replaces parameter counting

Let $X_1,\ldots,X_n$ be independent with true distribution $q$, model $p(x\mid w)$ and
prior density $\varphi(w)$. Write

$$
L_n(w)=-\frac1n\sum_{r=1}^n\log p(X_r\mid w),\qquad
\mathcal F_n=-\log\int e^{-nL_n(w)}\varphi(w)\,dw.
$$

Watanabe's Theorem 2, under §3 Fundamental Conditions (1)–(4), gives

$$
\mathcal F_n=nL_n(w_0)+\lambda\log n-(m-1)\log\log n+O_p(1).
$$

Here $w_0$ minimises population cross-entropy, $\lambda$ is the real log canonical
threshold (RLCT) and $m$ its multiplicity. This is **not** an unrestricted formula for any
neural model: the source assumes a compact semianalytic parameter domain, an analytic times
smooth positive prior, analytic log-density ratios in $L^s(q)$ for $s\geq6$, and its local
moment inequality (12). The theorem states a stronger convergence property for the remainder;
the displayed $O_p(1)$ is sufficient here.

Theorem 1 resolves the excess population loss to $K(g(u))=u^{2k}$ and the prior volume to
$b(u)|u^h|\,du$. Equations (15)–(16) define $\lambda$ by the smallest chartwise ratio
$(h_j+1)/(2k_j)$ and $m$ by the maximum number of simultaneous minimisers on the essential
charts. They depend on the loss and prior near the true distribution, not just the zero set
of the model equations. For a regular, realisable model of parameter dimension $d$, with
an interior optimum and positive prior density there, Lemma 3 gives $\lambda=d/2$, $m=1$:
the leading BIC penalty is recovered.

**Elementary singular example (proof).** Take $p(x\mid a,b)=\mathcal N(ab,1)$ and
$q=\mathcal N(0,1)$, with uniform prior on $[-1,1]^2$. The population KL loss is
$K(a,b)=a^2b^2/2$. The Fisher matrix for $(a,b)$ is

$$
I(a,b)=\begin{pmatrix}b^2&ab\\ab&a^2\end{pmatrix},
$$

because the score is $(x-ab)(b,a)$. It has rank at most one, and is zero at the origin.
Products $ab$ identify distributions, so different parameter pairs represent the same law.
For the deterministic loss integral, symmetry and $t=ab$ give

$$
\int_{[-1,1]^2}e^{-na^2b^2/2}\,da\,db
=4\int_0^1e^{-nt^2/2}(-\log t)\,dt
\asymp n^{-1/2}\log n.
$$

The last order follows by $s=\sqrt n\,t$: the coefficient of $\log n$ tends to a positive
Gaussian integral, and the integral of $e^{-s^2/2}\log s$ is finite. Hence
$\lambda=1/2$, $m=2$ (also directly from $k=(1,1)$, $h=(0,0)$). This integral illustrates
the local exponents; it is not a replacement for Theorem 2's random-data statement.

**Consequence for Lenticulum.** [[Finding a Factor Graph]] fits parameters and approximates
an integral over **states**. Watanabe's $\mathcal F_n$ integrates **parameters**. BP over
states does not perform this second integral, and a Bethe entropy correction is not an RLCT
estimator. Applying singular model selection to learned factors is an **expectation**, with
no RLCT/WBIC implementation or measurement here. A singular equation locus, redundant
parameters, and degenerate Fisher information should not be treated as equivalent tests:
the example $p(x\mid w)=\mathcal N(w^3,1)$ is injective but has Fisher information $9w^4$,
which vanishes at zero (differentiate its mean to verify).

## 2. Dual-number points and forward-mode AD

Write $D=k[\varepsilon]/(\varepsilon^2)$ (Stacks Definition 33.16.1). At a $k$-rational
point $p$ of $X$, tangent vectors are $k$-morphisms $\operatorname{Spec}D\to X$ reducing to
$p$ (Definition 33.16.3). For

$$
X=\operatorname{Spec}k[z_1,\ldots,z_d]/(f_1,\ldots,f_r),
$$

such a morphism sends $z_i$ to $p_i+\varepsilon v_i$. Substitution proves

$$
f_j(p+\varepsilon v)=f_j(p)+\varepsilon(J_f(p)v)_j,\qquad
T_pX=\ker J_f(p).
$$

This follows by multiplying monomials and discarding all terms containing $\varepsilon^2$;
therefore it is a proof for polynomial equations over any field. Stacks Lemma 33.16.4 gives
the coordinate-free description by differentials/derivations. Over $\mathbb R$, forward-mode
AD propagates the same multiplication rule, $(a,a')(b,b')=(ab,ab'+a'b)$, and the chain rule;
one seed $v$ computes $J_f(p)v$, rather than the entire Jacobian.

**What schemes retain (proof).** $\operatorname{Spec}k[z]/(z^2)$ and
$\operatorname{Spec}k[z]/(z)$ have the same underlying point. At zero, the first accepts
$z=\varepsilon v$ for every $v$, while the second requires $v=0$. Passing to the reduced
point loses that infinitesimal distinction. Tangent vectors do not promise finite motions:
over $k[t]/(t^3)$, the first example requires $(tv+t^2w)^2=t^2v^2=0$, which forces $v=0$
over a field. First-order feasibility can be obstructed at higher order.

[[polynomial]] uses Jacobians of residual equations, and [[plucker]] uses tangent directions
on a Grassmannian. These are useful first-order calculations; the code does not construct
dual-number points of schemes. In particular, replacing a residual $f$ by $f^2$ changes its
Jacobian and infinitesimal scheme even if the real zero set is unchanged (the preceding
example proves this at zero). A residual presentation and the radical ideal of its zero set
must be distinguished before interpreting an AD kernel as geometric tangent data.

## 3. A cellular sheaf for linear hard relations

Work over $\mathbb R$ and use the **bipartite incidence graph** as a one-dimensional cell complex. A variable vertex
$i$ carries $V_i$. Suppose a factor $a$ imposes a homogeneous linear relation
$R_a\subseteq\bigoplus_{i\in a}V_i$; its vertex carries $R_a$. The incidence edge $(i,a)$
carries $V_i$. The two vertex-to-edge restriction maps are

$$
V_i\xrightarrow{\mathrm{id}}V_i,\qquad
R_a\xrightarrow{\operatorname{pr}_i}V_i.
$$

This is a cellular sheaf of vector spaces in Hansen–Ghrist Definition 2.4: on a graph
there are no higher-dimensional incidences requiring additional composition checks.
A vertex assignment $s=(x_i,r_a)$ is a global section (Definition 2.5) precisely when
$x_i=\operatorname{pr}_i r_a$ on every incidence. Thus sections correspond bijectively to
variable assignments satisfying every relation: each $r_a$ is necessarily the tuple of
its neighbouring variables, and that tuple must belong to $R_a$. This proves the
correspondence, including factors with more than two variables.

Choose finite-dimensional inner products and orient each incidence. Section §2.2.2's
coboundary becomes $(\delta s)_{ia}=x_i-\operatorname{pr}_i r_a$. Define

$$
L=\delta^*\delta,\qquad
\langle s,Ls\rangle=\|\delta s\|^2
=\sum_{(i,a)}\|x_i-\operatorname{pr}_i r_a\|^2.
$$

Taking the inner product proves $\ker L=\ker\delta=\Gamma(G;\mathcal S)$; this is the
degree-zero case of Hansen–Ghrist Theorem 3.1. The quadratic form measures inconsistency
of these **state assignments in the chosen metrics**. It is not a Bethe free energy and
has no entropy or probability normaliser. Affine constraints require translated affine
spaces or homogenisation; nonlinear algebraic relations do not automatically give the
vector-space sheaf above.

This supplies a precise limited reading of [[Composition is Elimination]]: eliminating
factor-vertex states leaves exactly the simultaneous linear constraints on the variables
(proved by the section correspondence). A Laplacian solver could seek consistency in this
fragment. Identifying general Mycelium BP with a global-section algorithm, or identifying
belief marginal consistency with this state consistency, remains a **conjectural extension**.
No sheaf solver is implemented here. A sheaf of vector spaces on a cell complex and a
scheme's structure sheaf of rings share the local-to-global language, but Definition 26.9.1
requires much more than this construction provides.

Related: [[LP-012 Factor graphs and schemes]], [[Finding a Factor Graph]],
[[Algebraic Statistics Bridge]], [[Toric Models as Factors]], [[Branches and the Discriminant]],
[[L-SPEC-004 Algebraic Factors Toric and Grassmannian]].
