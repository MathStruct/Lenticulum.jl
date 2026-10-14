#theorem #derivation

> **What an implicit diffusion learner can represent.** For a fixed node set, the stable answers
> of an energy network's learner approximate the stable answers of any target energy
> whose answers are nondegenerate, uniformly over a compact set of queries; an ε-network's
> learner does the same for any target vector field. The answers move by at most $2\delta/\mu$,
> where $\delta$ is the $C^1$ distance of the residuals and $\mu$ measures how well-conditioned the
> target's answers are. The theorem is about expressiveness. Whether training reaches such
> parameters is a separate question, and §6 lists what stays open.
>
> The diffusion counterpart of [[Universal Approximation by Nash-Tognoli]].

> Sources: original to this vault (statement and proofs, not peer reviewed); uses Pinkus, *Approximation Theory of the MLP Model in Neural Networks*, Acta Numerica 1999 (Theorem 4.1, density in $C^m$); Hornik, *Approximation Capabilities of Multilayer Feedforward Networks*, Neural Networks 1991; for §6, Genovese, Perone-Pacifico, Verdinelli & Wasserman, *Nonparametric ridge estimation*, Annals of Statistics 2014
>
> Bibliography: [[Bibliography#^pinkus1999mlp|Pinkus 1999]] · [[Bibliography#^hornik1991approximation|Hornik 1991]] · [[Bibliography#^genovese2014ridge|Genovese et al. 2014]]

## 1. The objects

The learner of [[Implicit Diffusion Learners]] has fixed nodes $(t_k, \varepsilon_k, w_k)$, $w_k > 0$,
weight $\lambda > 0$, and the field

$$
g_\theta(z) = \sum_k w_k\lambda_k\bigl(\varepsilon_\theta(\alpha_k z + \sigma_k\varepsilon_k, t_k) - \varepsilon_k\bigr),
\qquad \lambda_k = \lambda\sigma_k/\alpha_k .
$$

For an energy network, $\varepsilon_\theta = \sigma_t\nabla E_\theta$, the field is $g_\theta = \nabla U_\theta$ with

$$
U_\theta(z) = \sum_k c_k\,E_\theta(\alpha_k z + \sigma_k\varepsilon_k, t_k) - \ell^\top z,
\qquad c_k = \frac{w_k\lambda\sigma_k^2}{\alpha_k^2} > 0, \quad \ell = \sum_k w_k\lambda_k\varepsilon_k
$$

([[energy_network]] §3). A **query** clamps the coordinates $C$ at $z_C$ and anchors the free
coordinates $F$ (with $m = \lvert F\rvert$) at $z_{0,F}$ with precisions $\rho_F \in [0,\infty)^m$. Write
$p = (z_C, z_{0,F})$ for its data and $y = z_F$ for the unknowns. Its residual is

$$
r_\theta(p, y) = g_{\theta,F}(z_C, y) + \rho_F^2 \odot (y - z_{0,F}),
$$

and an **answer** is a zero $y$ of $r_\theta(p, \cdot)$. It is **stable** if
$\operatorname{sym}\partial_y r_\theta(p, y) \succ 0$, the flag `stable` of `implicit_infer`. For an
energy network that is a strict local minimum of $U_\theta(z_C, \cdot) + \tfrac12\lVert\rho_F\odot(y - z_{0,F})\rVert^2$.

A **target** is any $C^1$ vector field $G : \mathbb R^n \to \mathbb R^n$, or for energy networks
$G = \nabla\Psi$ with $\Psi \in C^2$. The query turns it into a residual $r_G$ in the same way.

**Regularity.** Let $Q$ (query data) and $Y \subset \mathbb R^m$ (an answer window) be compact. A
residual $r$, $C^1$ on an open set $N \supset Q\times Y$, is **regular on $(Q, Y)$** if

- (R1) at every zero $(p, y) \in Q\times Y$, $J = \partial_y r(p, y)$ is invertible and
  $\operatorname{sym}J$ has no zero eigenvalue;
- (R2) $r(p, y) \neq 0$ for $p \in Q$ and $y \in \partial Y$.

(R1) says every answer is nondegenerate and is clearly stable or clearly unstable. For a
gradient, $J$ is the Hessian block and (R1) says that every critical point in the fibre is
nondegenerate (a Morse condition on the fibres). It excludes branch points (§5).

## 2. Answers persist

> **Lemma 1 (persistence of answers).** Let $r$ be regular on $(Q, Y)$. There are a compact
> neighbourhood $W \subset N$ of $Q\times Y$ and constants $\mu > 0$, $\delta_0 > 0$ such that for every
> $\tilde r \in C^1(N)$ with
> $$
> \sup_W \bigl(\lvert\tilde r - r\rvert + \lVert\partial_y\tilde r - \partial_y r\rVert\bigr) \le \delta \le \delta_0
> $$
> and every $p \in Q$, the zeros of $\tilde r(p, \cdot)$ in $Y$ and those of $r(p, \cdot)$ in $Y$ are in a
> bijection that moves no zero by more than $2\delta/\mu$ and preserves stability. Moreover
> $\tilde r$ is regular on $(Q, Y)$ with constant $\mu/2$.

Here $\mu = \min$ over the zeros of $r$ in $Q\times Y$ of $\min\{\sigma_{\min}(J), \lvert\lambda_{\min}(\operatorname{sym}J)\rvert\}$.

*Proof.* Let $Z = \{(p, y) \in Q\times Y : r(p, y) = 0\}$; it is compact, and $\mu > 0$ because
$\sigma_{\min}$ and $\lambda_{\min}\circ\operatorname{sym}$ are continuous and positive in absolute value on $Z$
by (R1). By (R2), $d_0 = \min_{(p,y)\in Z}\operatorname{dist}(y, \mathbb R^m\setminus Y) > 0$. Choose $r_0 > 0$
so small that $W = \{(p, y) : p \in Q,\ \operatorname{dist}(y, Y) \le r_0\} \subset N$ and, by uniform
continuity of $\partial_y r$ on $W$, $\lVert\partial_y r(p, y) - \partial_y r(p, y_0)\rVert \le \mu/4$ whenever
$(p, y_0) \in Z$ and $\lvert y - y_0\rvert \le r_0$.

*Step 1: one zero per ball.* Fix $(p, y_0) \in Z$, $J_0 = \partial_y r(p, y_0)$, $B = \bar B(y_0, r_0)$, and let
$\delta \le \mu/4$. For $y, y' \in B$, $\tilde r(p, y) - \tilde r(p, y') = M(y - y')$ with
$M = \int_0^1 \partial_y\tilde r(p, y' + s(y - y'))\,ds = J_0 + E$, $\lVert E\rVert \le \mu/4 + \delta \le \mu/2$. So
$\sigma_{\min}(M) \ge \mu/2 > 0$ and $\tilde r(p, \cdot)$ has at most one zero in $B$. (With $\tilde r = r$:
distinct zeros of $r(p, \cdot)$ are at least $r_0$ apart.)

*Step 2: existence.* $\Phi(y) = y - J_0^{-1}\tilde r(p, y)$ has $\lVert\partial\Phi\rVert = \lVert J_0^{-1}(J_0 - \partial_y\tilde r)\rVert \le \mu^{-1}\cdot\mu/2 = 1/2$
on $B$, and $\lvert\Phi(y_0) - y_0\rvert = \lvert J_0^{-1}(\tilde r - r)(p, y_0)\rvert \le \delta/\mu$. With $\rho = 2\delta/\mu \le r_0$,
$\Phi$ maps $\bar B(y_0, \rho)$ into itself ($\rho/2 + \delta/\mu = \rho$) and contracts, so it has a fixed
point $\tilde y$: $\tilde r(p, \tilde y) = 0$ and $\lvert\tilde y - y_0\rvert \le 2\delta/\mu$.

*Step 3: stability is kept.* $\partial_y\tilde r(p, \tilde y) = J_0 + E'$ with $\lVert E'\rVert \le \mu/2$. By Weyl's
inequality $\lvert\lambda_{\min}(\operatorname{sym}(J_0 + E')) - \lambda_{\min}(\operatorname{sym}J_0)\rvert \le \mu/2 < \lvert\lambda_{\min}(\operatorname{sym}J_0)\rvert$:
the sign is the same, and both $\sigma_{\min}$ and $\lvert\lambda_{\min}\rvert$ are at least $\mu/2$.

*Step 4: no other zeros.* The set $S = \{(p, y) \in Q\times Y : \lvert y - y_0\rvert \ge r_0 \text{ for all } (p, y_0) \in Z\}$
is closed: its complement in $Q\times Y$ is open, because by the implicit function theorem
($J_0$ invertible, $y_0$ interior to $Y$) each zero continues to nearby $p'$. So $S$ is compact,
$r \ne 0$ on it, and $c = \min_S\lvert r\rvert > 0$. For $\delta < c$, every zero of $\tilde r(p, \cdot)$ in $Y$ lies
within $r_0$ of a zero $y_0$ of $r(p, \cdot)$, and by Step 1 it is the $\tilde y$ of Step 2 for that $y_0$.

Take $\delta_0 < \min\{\mu/4,\ c,\ \mu r_0/4,\ \mu d_0/2\}$. Then $\rho \le r_0/2$, so different $y_0$ (at least
$r_0$ apart) get different $\tilde y$; $\rho < d_0$, so $\tilde y \in Y$; and by Step 4 every zero of
$\tilde r(p, \cdot)$ in $Y$ is reached. $\square$

The constant is **effective**: an answer moves by at most $2\delta/\mu$, where $\mu$ is the worst
conditioning of the target's answers. Near a branch point $\mu \to 0$ (§5), and so does the
guarantee.

## 3. The node sum can realise any target

The network enters $U_\theta$ only through the node sum. The question is whether every target
energy is $U$ for some $E$. It is, on a compact set, even though the nodes of one level see the
network at shifted points (antithetic pairs see it at $\alpha z \pm \sigma\varepsilon$).

> **Lemma 2 (inverting a finite sum of shifts).** Let $s_1, \dots, s_q \in \mathbb R^n$ be distinct,
> $\gamma_1, \dots, \gamma_q \ne 0$, and $\hat G \in C^k(\mathbb R^n; \mathbb R^d)$ with compact support. There is
> $H \in C^k(\mathbb R^n; \mathbb R^d)$ with
> $$
> \sum_{j=1}^q \gamma_j\,H(z + s_j) = \hat G(z) \qquad\text{for all } z .
> $$

*Proof.* Pick $v$ with the values $\langle s_j, v\rangle$ pairwise distinct (avoid finitely many
hyperplanes) and number the shifts so that $\langle s_1, v\rangle$ is the largest. Let
$d = \min_{j\ge2}\langle s_1 - s_j, v\rangle > 0$ and
$m_0 = \min_{z\in\operatorname{supp}\hat G}\langle z, v\rangle + \langle s_1, v\rangle$. Define $H = 0$ on $\{\langle u, v\rangle < m_0\}$ and

$$
H(u) = \frac1{\gamma_1}\Bigl(\hat G(u - s_1) - \sum_{j\ge2}\gamma_j H(u - s_1 + s_j)\Bigr)
$$

elsewhere. The points $u - s_1 + s_j$ lie at least $d$ lower in the direction $v$, so by induction
over the slabs $\{\langle u, v\rangle < m_0 + Nd\}$, $H$ is well defined there and equals a finite
linear combination of translates of $\hat G$. It is therefore $C^k$. On $\{\langle u, v\rangle < m_0\}$ the formula
also gives $0$, since $\hat G(u - s_1) = 0$ there and the other terms lie lower still. So the formula
holds everywhere, and at $u = z + s_1$ it is the claimed identity. $\square$

> **Proposition 3 (exact realisation).** Let $K \subset \mathbb R^n$ be compact and fix a level $t^\ast$ of
> the nodes. For every $\Psi \in C^k(\mathbb R^n)$ there is $E^\ast \in C^k$ with $U_{E^\ast} = \Psi$ on $K$, and
> $E^\ast(\cdot, t) = 0$ at every other node level. For every $G \in C^k(\mathbb R^n; \mathbb R^n)$ there is a noise
> predictor $\varepsilon^\ast$ with $g_{\varepsilon^\ast} = G$ on $K$.

*Proof (energy).* Let $I^\ast = \{k : t_k = t^\ast\}$ and $s_k = \sigma^\ast\varepsilon_k/\alpha^\ast$. Group equal shifts,
adding their $c_k$ into $\gamma_j > 0$. Take a smooth cutoff $\chi = 1$ on $K$ with compact support,
and apply Lemma 2 to $\hat G = \chi\cdot(\Psi + \ell^\top z)$. Set $E^\ast(u, t^\ast) = H(u/\alpha^\ast)$, so that
$E^\ast(\alpha^\ast z + \sigma^\ast\varepsilon_k, t^\ast) = H(z + s_k)$, and make $E^\ast(\cdot, t) = 0$ at the other levels with a smooth
factor in $t$. Then $U_{E^\ast} = \hat G - \ell^\top z = \Psi$ on $K$. The field case is the same with $d = n$,
weights $w_k\lambda_k$ and $\hat G = \chi\cdot(G + \sum_k w_k\lambda_k\varepsilon_k)$. $\square$

With `noisefree_nodes` ($\varepsilon = 0$, one node) this is just $E^\ast = (\Psi + \ell^\top z)/c$. With
antithetic pairs it is an alternating sum, $H(u) = \gamma^{-1}\sum_{i\ge0}(-1)^i\hat G(u + (2i+1)s)$.

## 4. The theorem

> **Theorem 4 (universal approximation of answers).** Fix the nodes, $\lambda$, and the predictor's
> time features $\varphi(t)$, with $\varphi(t^\ast) \ne \varphi(t_k)$ for the node levels $t_k \ne t^\ast$ of some level $t^\ast$.
> Let the network be $E_\theta(u, t) = N_\theta(u, \varphi(t))$ (resp. $\varepsilon_\theta(u, t) = N_\theta(u, \varphi(t))$) with
> $N_\theta$ a one-hidden-layer network whose activation is $C^2$ (resp. $C^1$) and not a polynomial.
> Let finitely many queries (polarities and precisions), each with compact data $Q$ and window $Y$,
> have regular target residuals $r_\Psi$ for a target energy $\Psi \in C^2$ (resp. $r_G$ for a target field
> $G \in C^1$). Then for every $\eta > 0$ there are parameters $\theta$ such that, for every one of
> these queries and every $p \in Q$, the answers of the learner in $Y$ and of the target in $Y$
> are in a bijection that moves no answer by more than $\eta$ and preserves stability. In
> particular the learner's stable answers are within Hausdorff distance $\eta$ of the target's,
> uniformly in $p$.

*Proof.* Take $K$ compact containing every $z = (z_C, y)$ with $(p, y)$ in the neighbourhoods $W$ of
Lemma 1, for all the queries, and take the smallest $\delta_0, \mu$ among them. Proposition 3 gives
$E^\ast$ with $U_{E^\ast} = \Psi$ on $K$, and it can be written as $\tilde E(u, \varphi(t))$ with $\tilde E$ $C^2$ on
$\mathbb R^{n+f}$ (a smooth factor in the feature vector that is $1$ at $\varphi(t^\ast)$ and $0$ at the other
levels' features). By Pinkus' theorem, one-hidden-layer networks with a $C^m$ activation that is
not a polynomial are dense in $C^m$ on compact sets (with all derivatives up to order $m$). So
for every $\beta > 0$ some $N_\theta$ is within $\beta$ of $\tilde E$ in $C^2$ on the compact set
$\bigcup_k\{(\alpha_k z + \sigma_k\varepsilon_k, \varphi(t_k)) : z \in K\}$. Since $\alpha_k \le 1$, the chain rule gives
$\lVert U_\theta - \Psi\rVert_{C^2(K)} \le \beta\sum_k c_k$. The residuals differ by
$\nabla_F U_\theta - \nabla_F\Psi$ (the precision term is the same), so their $C^1$ distance in $y$ is at most
$\beta\sum_k c_k$. Choose $\beta\sum_k c_k \le \min\{\delta_0, \mu\eta/2\}$ and apply Lemma 1. For fields, use $C^1$ and
$\sum_k w_k\lambda_k$. $\square$

**Remarks.**
- **Time features.** The package's sinusoidal features include the frequency $1/2$,
  $(\sin\pi t, \cos\pi t)$, which is injective on $[0, 1]$. So every level can be singled out.
- **Depth.** A deeper network with the same activation can emulate the shallow one: with
  $\varsigma'(a) \ne 0$, $(\varsigma(a + hx) - \varsigma(a))/(h\varsigma'(a)) \to x$ in $C^k$ on compact sets as $h \to 0$, so the
  extra layers approximate the identity. `swish`, `tanh` and `softplus` qualify. ReLU is not $C^1$
  and is excluded, as it is for energy networks anyway.
- **Noise blocks.** With held block levels the node inputs are $\alpha\odot z + \sigma\odot\varepsilon$ with per-coordinate
  factors, and the weights change ([[implicit]] §9). The proof only uses that each node input is
  an invertible affine map of $z$ and that the weights are positive, so it carries over.
- **ε-networks are universal for a larger class:** any field with regular answers, gradient or
  not. Energy networks reach exactly the gradient targets. Both reach every relation that has
  an energy (§5).

## 5. Which relations are regular targets

The theorem is about targets with regular answers. Which relations have one?

1. **Functions.** For $y = f(x)$, $f \in C^2$ on a compact $D$, take $\Psi = \tfrac12\lvert y - f(x)\rvert^2$ and
   the query $x \to y$: $r = y - f(x)$, $J = I$, $\mu = 1$. With $Y$ a ball containing $f(D)$ in
   its interior, the learner's single answer approximates $f$ uniformly on $D$. **Universal
   approximation of functions is the special case of one branch.** Since Theorem 4 makes
   $U_\theta$ close to $\Psi$ in $C^2$, mixed derivatives included, the implicit function theorem also
   makes the derivative of the answer, $\partial_x y$, close to $f'$.
2. **Finitely many branches with $d_y = 1$.** For disjoint branches $f_1 < \dots < f_q$ over $D$, take
   $\Psi = \tfrac12\prod_i(y - f_i(x))^2$. The fibre derivative is a polynomial of degree $2q - 1$ with
   $2q - 1$ simple real roots: the minima $f_i$ (with $\Psi_{yy} = \prod_{j\ne i}(f_i - f_j)^2 > 0$) and one
   maximum between each pair (Rolle). So (R1) holds, and (R2) holds for $Y = [-R, R]$ with $R$
   large.
3. **The circle away from its branch points.** $\Psi = \tfrac14(x^2 + y^2 - 1)^2$, query $x \to y$,
   $D = [-1+\kappa, 1-\kappa]$. The answers $y = \pm\sqrt{1 - x^2}$ are stable with $\Psi_{yy} = 2y^2 \ge 2\kappa(2-\kappa)$,
   and $y = 0$ is unstable with $\Psi_{yy} = x^2 - 1 < 0$. Then $\mu \sim \kappa$, so the guarantee
   $2\delta/\mu$ degrades as the query approaches $x = \pm1$. This is T3 made quantitative. The same
   $\Psi$ serves the query $y \to x$.
4. **Chains and products.** A [[product]] of learners, one per factor, reaches every
   target of the factorised form $\sum_k\Psi_k(x_{k-1}, x_k) - \sum_k\Psi_{S,k}(x_k)$. Apply Theorem 4 to each
   factor; the errors add. `examples/chain.jl` shows such chains of three to five relations
   ([[product]] §4).

## 6. What the theorem does not say

| | explicit (Weierstraß, networks) | algebraic ([[Universal Approximation by Nash-Tognoli\|Nash–Tognoli]]) | implicit diffusion (Theorem 4) |
|---|---|---|---|
| approximated object | continuous functions | compact smooth manifolds, up to diffeomorphism | the answers of any regular target, per query |
| in the data's own space? | yes | not necessarily (Borel–Haefliger) | yes |
| effective? | yes (rates) | no | the persistence step yes ($2\delta/\mu$); the network step as effective as $C^2$ rates for networks |
| nondegenerate output? | n/a | singular fits are generic | answers within $\delta_0$ of a regular target are regular |

Gap 2 of Nash–Tognoli does not arise here: everything stays in the data's coordinates, and
smooth functions carry no algebraic-cycle obstruction. Gap 3 (singular fits) is answered near a
regular target, because Lemma 1 makes the learner's answers regular there as well. What remains
open:

1. **Expressiveness is not learning.** Training does not aim at $\Psi$. At its population optimum
   the network is the true score, $E^\ast(\cdot, t) = -\log p_t$. The learner's relation is then the
   set of stable answers of the smoothed density's $U^\ast$, the ridge of
   [[Implicit Diffusion Learners]] §5. That is not the data's relation: the smoothing bias (T2)
   adds up along chains (`examples/chain.jl`). Theorem 4 does say something here. If $U^\ast$ is
   regular and the trained network is $C^2$-close to $-\log p_t$ on the query region, the answers
   are those of $U^\ast$ within $2\delta/\mu$. But score matching controls an $L^2(p_t)$ error, not a $C^2$
   error on the query region, and the two differ exactly where data are sparse (T5).
   **Consistency** is open: that the answers of $U^\ast$ approach the data's relation as the levels
   go to $0$ and the data grow. Related results exist for ridges of kernel density estimates
   (Genovese et al. 2014).
2. **Rates.** Lemma 1 is effective, and the network step inherits whatever $C^2$ approximation
   rates hold for $\tilde E$. Those rates depend on its smoothness and on the input dimension
   $n + f$. The construction of Lemma 2 is not frugal: with antithetic shifts $\pm s$ it sums about
   $\operatorname{diam}(K)/\lvert s\rvert$ translates. In practice the natural target $-\log p_t$ is smooth
   at the node levels and needs no deconvolution. Rates for the realistic target are open.
3. **Branch points (T3).** (R1) excludes them, and the constant degrades like $1/\kappa$ near them.
   Folds of the answer set are generic in families of functions and structurally stable, so a
   statement that the fold itself persists and moves by $O(\delta)$ should hold. It is not proved
   here.
4. **Which relations are regular targets.** §5 proves it for functions, for scalar multi-branch
   relations and for the circle. For a general compact submanifold $M \subset \mathbb R^n$ and every
   polarity at once, it is open. $\Psi = \tfrac12\lvert\phi\rvert^2$ with a defining submersion $\phi$ makes the
   answers on $M$ regular exactly where $M$ is a local graph over the clamped coordinates. But it
   needs a trivial normal bundle, and it does not control the other critical points of the
   fibres. This is the implicit counterpart of the question Nash–Tognoli answers for varieties.
5. **Finding the answers (T1, T11).** The theorem gives existence and the count of answers per
   query, not that the solver finds all of them, or any of them, from a given start.
6. **Queries outside $Q$ (T4, T5).** Nothing is claimed off the compact query set. The learner may
   have answers there that the target lacks.

Related: [[Implicit Diffusion Learners]], [[Open Problems in Implicit Diffusion Learning]],
[[Universal Approximation by Nash-Tognoli]], [[energy_network]], [[product]], [[Implicit Learners]]
