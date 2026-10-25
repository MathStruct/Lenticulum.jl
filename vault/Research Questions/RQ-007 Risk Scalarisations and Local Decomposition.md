#open-problem

> **Status: open (2026-10-25; revised the same day after Codex's review, which showed the tree case solved).** Area: variational inference, risk measures · Prerequisites: Bethe free energy, convex risk measures · Related: [[Unusual Wirings]] §4, [[L-SPEC-003 Exposed Parameters and Energy Channels]] §3, [[Bethe Free Energy]], [[Scalar and Multivariate Energy]], [[Energies and Entropies]].
>
> Sources: Donsker & Varadhan, CPAM 28(1), 1975; Artzner, Delbaen, Eber & Heath, *Coherent Measures of Risk*, Math. Finance 9(3), 1999; Rockafellar & Uryasev, *Optimization of Conditional Value-at-Risk*, J. Risk 2(3), 2000 (Theorem 1, continuous losses), and *Conditional Value-at-Risk for General Loss Distributions*, J. Banking & Finance 26(7), 2002; Föllmer & Schied, *Stochastic Finance*, 4th ed., 2016, ch. 4; Yedidia, Freeman & Weiss, IEEE Trans. Inf. Theory 2005; the question is original to this vault.
>
> Bibliography: [[Bibliography#^donsker1975asymptotic|Donsker & Varadhan 1975]] · [[Bibliography#^artzner1999coherent|Artzner et al. 1999]] · [[Bibliography#^rockafellar2000cvar|Rockafellar & Uryasev 2000]] · [[Bibliography#^rockafellar2002cvar|Rockafellar & Uryasev 2002]] · [[Bibliography#^follmer2016stochastic|Föllmer & Schied 2016]] · [[Bibliography#^yedidia2005bethe|Yedidia et al. 2005]]

## Question

A factor graph's free energy scalarises energies by their **expectation**, which is additive over
factors. If the total energy $E = \sum_a E_a$ is instead scalarised by a risk measure $\rho$ (the
entropic risk, CVaR), **how well can $\rho(E)$ be computed, bounded and optimised by local message
passing on a loopy graph, and with how large messages?** On a tree the entropic risk is exact
(below); the question is what survives on loops, for CVaR, and under optimisation.

## Setting

Variables $x = (x_v)$, factors $a$ with energies $E_a(x_{\partial a})$, and a normalised joint law
$q(x) = Z_0^{-1} \prod_a \psi_a(x_{\partial a})$ on $x$ (the model, or a belief of that form). A
Bethe approximation alone, with locally consistent pseudomarginals $q_a, q_v$, does **not** define
a joint law, so $\rho(E)$ is undefined for it unless a reconstruction is specified (on a tree, the
pseudomarginals determine $q$). Under $q$, $E$ is a random variable; its parts $E_a$ may be dependent.

- Entropic risk: $\rho_\beta(E) = \tfrac1\beta \log \mathbb E_q\, e^{\beta E} \in (-\infty, +\infty]$, $\beta > 0$.
- CVaR: $\mathrm{CVaR}_\alpha(E) = \min_t \bigl(t + \tfrac1{1-\alpha} \mathbb E_q (E - t)^+\bigr)$, $\alpha \in (0, 1)$.

"Local" means computed by messages on the graph's edges; the size of a message is part of the
answer.

## What is known

- **The expectation is additive**, $U = \mathbb E_q E = \sum_a \mathbb E_{q_a} E_a$, without any
  independence. That is the energy part of the Bethe free energy
  $F_B = U - \sum_a H(q_a) + \sum_v (d_v - 1) H(q_v)$; the entropy part is Bethe's approximation, exact
  on trees (Yedidia et al. 2005).
- **The entropic risk at a fixed $q$ is a ratio of partition functions** (derivation):
  $\mathbb E_q e^{\beta E} = Z_\beta / Z_0$ with $Z_\beta = \int \prod_a \psi_a e^{\beta E_a}$, the partition function of
  the tilted potentials. So $\rho_\beta(E) = \tfrac1\beta(\log Z_\beta - \log Z_0)$. **On a tree** sum-product
  computes both exactly (when the integrals are tractable; $\rho_\beta = +\infty$ when $Z_\beta$ is), with
  messages of the same size as for $Z_0$. (Codex checked tree elimination against enumeration on
  a finite tree in its review; not a test in the repository yet.)
- **On a loopy graph** each partition function has a Bethe approximation, so
  $\tfrac1\beta(\log Z^B_\beta - \log Z^B_0)$ is a natural local estimate. It is a difference of two
  approximations, with no sign or error guarantee known here. **Open.**
- **Additivity of the entropic risk** holds for independent parts and fails in general under
  dependence; shared variables can, but need not, make the parts dependent.
- **CVaR is coherent**, hence subadditive, $\mathrm{CVaR}_\alpha(\sum_a E_a) \le \sum_a \mathrm{CVaR}_\alpha(E_a)$
  (Artzner et al. 1999; the minimisation formula for general distributions is Rockafellar & Uryasev
  2002, the 2000 paper's Theorem 1 assuming continuous losses). The sum of local CVaRs is a local
  upper bound; its gap is open.
- **CVaR on a chain is exact with larger messages**: carry the accumulated cost $S_i = S_{i-1} + E_i$
  in the state, so each forward message is a distribution over $(x_i, S_i)$, and the terminal message
  gives the law of $E$, hence every risk measure of it. Exact, but the message size grows with the
  support of $S$ (derivation, after Codex's review).
- **Donsker–Varadhan**: for $p$ a probability law, $q' \ll p$, and $E$ with $\mathbb E_p e^{\beta E} < \infty$,
  $\rho_\beta(E) = \sup_{q'} \bigl(\mathbb E_{q'} E - \tfrac1\beta \mathrm{KL}(q' \| p)\bigr)$, attained at $q' \propto p\,e^{\beta E}$. Restricting
  $q'$ to a factorised family gives a lower bound, the variational counterpart of the tilted
  partition function above.
- **Heavy tails**: if the total $E$ has a heavier-than-exponential right tail, $\rho_\beta(E) = +\infty$ for
  every $\beta > 0$. A single heavy-tailed part suffices when the other parts are bounded below; in
  general parts can cancel ($E_1 = T$, $E_2 = -T$). CVaR is finite when $\mathbb E E^+ < \infty$.

## What would count as an answer

Additive, translation-invariant error bounds (energies may be negative, so ratios are not
meaningful) for the loopy estimate $\tfrac1\beta(\log Z^B_\beta - \log Z^B_0)$ or the local CVaR bound, in terms
of the graph's loops and the dependence of the parts; classes of loopy graphs where they are exact;
for CVaR, lower bounds on the message size any exact local method needs; and for optimisation, a
message-passing method for $\min_\theta \rho(E)$ with its stationary points characterised. A
measurement on Gaussian graphs (below), with a test, would be a first step.

## Why it matters here

It decides whether [[L-SPEC-003 Exposed Parameters and Energy Channels]] §3's risk scalarisations
can be used *inside* message passing or only as an outer objective over sampled energies, and
which of them makes sense for heavy-tailed energies.

## Leads

- **Gaussian graphs with quadratic energies.** If each $E_a$ is quadratic in its own variables, the
  tilted model has precision $\Sigma^{-1} - \beta M$ with the sparsity of the original graph, so it is
  again a Gaussian graph on the same edges (proper while $\Sigma^{-1} - \beta M \succ 0$). Its partition
  function, hence $\rho_\beta$, is a log-determinant; for a centred $x \sim \mathcal N(0, \Sigma)$ and $E = \tfrac12 x^\top M x$,
  $\mathbb E e^{\beta E} = \det(I - \beta\,\Sigma^{1/2} M \Sigma^{1/2})^{-1/2}$. On loops, Gaussian BP gives exact means when it
  converges but not exact log-determinants, which makes this the cleanest place to measure the
  loopy estimate's error.
- **Risk-sensitive control** has the entropic risk of a quadratic cost on a chain (risk-sensitive
  LQG) with a Riccati recursion; a dynamic-programming relative of the tree case. To be checked and
  cited before use.
- **CVaR by an outer scalar.** For fixed $t$, $\mathbb E (E - t)^+$ is not additive over factors, but the
  outer minimisation is one-dimensional; combined with the chain construction above, the cost is
  one augmented forward pass per $t$.
