#model

> **Can the GAN diagram be achieved with two factors?**
>
> The wiring: yes — two *parameter sets*, three *factor nodes*, and the third is the second
> one again under weight tying. The objective: **not with one free energy**. A game needs a
> scalarisation per player of a vector payoff *and* an owner for each: the parameters that
> descend it. One player with several scalarisations is multi-objective optimisation, not a game.
>
> Implemented in `lib/Adversarial.jl`; see [[Adversarial]].

> Sources: original to this vault (design and analysis). Open games and the parallel composite of decisions: Ghani, Hedges, Winschel & Zahn, *Compositional game theory*, LICS 2018. Local Nash equilibria of continuous games: Ratliff, Burden & Sastry, Allerton 2013. Gradient dynamics in games and symplectic gradient adjustment: Balduzzi et al., *The Mechanics of n-Player Differentiable Games*, ICML 2018. code: `lib/LenticulumCore.jl/src/energy.jl` (graded energies and scalarisations), `lib/Adversarial.jl/src/ratio.jl`
>
> Bibliography: [[Bibliography#^ghani2018compositional|Ghani et al. 2018]] · [[Bibliography#^ratliff2013nash|Ratliff et al. 2013]] · [[Bibliography#^balduzzi2018mechanics|Balduzzi et al. 2018]]
>
> Theory (CT-ML wiki): [Open Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Open-Game) · [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) · [Para Construction](https://mathstruct.org/CategoryTheory-ML-Wiki/Para-Construction) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens) · [Symmetric Monoidal Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Symmetric-Monoidal-Category)

## 1. The diagram

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  var/.style={circle, draw, minimum size=8mm, inner sep=0pt},
  fac/.style={rectangle, draw, fill=black!12, minimum size=6mm, inner sep=2pt},
  dat/.style={rectangle, draw, fill=black!55, text=white, minimum size=6mm, inner sep=2pt},
  every node/.style={font=\small}
]
\node[fac] (q)   at (0,0)     {$q(z)$};
\node[var] (z)   at (1.6,0)   {$z$};
\node[fac] (G)   at (3.2,0)   {$G_\theta$};
\node[var] (xf)  at (4.8,0)   {$x_{\text{fake}}$};
\node[fac] (Df)  at (6.6,0)   {$D_\varphi$};
\node[var] (xr)  at (4.8,-1.8) {$x_{\text{real}}$};
\node[dat] (dat) at (3.2,-1.8) {data};
\node[fac] (Dr)  at (6.6,-1.8) {$D_\varphi$};
\draw (q)--(z) (z)--(G) (G)--(xf) (xf)--(Df);
\draw (dat)--(xr) (xr)--(Dr);
\draw[dashed] (Df) -- node[right, xshift=2pt] {\footnotesize tied} (Dr);
\end{tikzpicture}
\end{document}
```

Circles are variables, grey squares factors, the dark square a data clamp. The dashed line is
**not an edge** — it is a shared parameter set.

## 2. Counting

| | count | what |
|---|---|---|
| learnable parameter sets | **2** | $\theta$ (generator), $\varphi$ (discriminator) |
| factor nodes | **3** | $G$, $D$ at the fake branch, $D$ at the real branch |
| structural factors | 2 | the latent prior $q(z)$, the data clamp |
| variables | 3 | $z$, $x_{\text{fake}}$, $x_{\text{real}}$ |

So the answer to "two factors?" is: **two in the sense that matters** — two things with
parameters — and three nodes, because the discriminator is *evaluated twice*:

$$
\mathbb{E}_{x\sim p_{\text{data}}}\bigl[\log D_\varphi(x)\bigr]
\;+\;
\mathbb{E}_{z\sim q}\bigl[\log\bigl(1 - D_\varphi(G_\theta(z))\bigr)\bigr]
$$

Two evaluation sites, one parameter set. A factor node in `Mycelium` has fixed channels, so
one node cannot attach to two different variables; you need two nodes that share $\varphi$.

> [!note] Weight tying is already in the vault's vocabulary
> [[Lux as a Parametric Lens]]'s dictionary lists *reparametrisation $\alpha : Q \to P$* as
> "`Optimisers.jl` rules, **weight tying**, LoRA". Two factor nodes sharing one parameter set
> is exactly a $\mathbf{Para}$ reparametrisation, and it is the mechanism the GAN diagram
> needs. `Mycelium` has no implementation of it — `ps` is a flat `NamedTuple` keyed by factor
> name, so two nodes cannot name the same entry.
>
> **That is the one genuinely missing piece of graph machinery**, and it is small.

You can collapse to literally two nodes by making the discriminator's input a *mixture*
variable carrying both real and fake particles with a label. That is what the implementation
does in effect, and it trades a node for a bookkeeping obligation.

## 3. And the whole thing is a DAG

All three factors in `lib/Adversarial.jl` are unidirectional — `isunidirectional` is `true` for
each, asserted in the test suite. So:

> **A GAN uses none of Mycelium's bidirectionality.** The diagram is a DAG; it is Lux-shaped.

Which is worth sitting with, given that [[Implicit Learners]] files diffusion, equilibrium and
algebraic models as the three implicit families. A GAN is implicit in a *different* sense —
[[Three Senses of Implicit]] — and that sense does not buy you a polarity. The factor-graph
formulation gains parameter management, channel names and free-energy accounting, and gains no
directions.

## 4. The obstruction: one scalar

Here is what the graph cannot hold.

The Bethe free energy ([[Bethe Free Energy]]) is a **single scalar**:

$$F \;=\; \sum_c F_c \;+\; \sum_v (d_v-1)H_v$$

and every learnable factor descends $\partial F/\partial\theta_c$. A GAN is a **minimax**:

$$\min_\theta \max_\varphi V(\theta,\varphi)$$

$\theta$ descends, $\varphi$ **ascends**, on the same quantity. There is no assignment of
per-factor free energies whose common descent reproduces that, because descent has one
direction and the game has two.

`RatioFactor` emits the likelihood potential $r(x)$, with `logratio(x)` as its
log-density. Pooling with a particle prior applies the ratio once; inversion and explicit
`reweight` return that posterior. See [[ratio]] and the LP-016 conformance tests.

The graph's `local_free_energy(::RatioFactor, …)` returns
$\mathbb{E}_{b_f}[-\log r]$ under $b_f \propto r\mu$, with $\mu$ the incoming cavity.
This is the factor's energy contribution, with no continuous particle entropy. It generally
differs from $\mathbb{E}_q[-\log r]=\mathrm{KL}(q\|p)$ and does not implement a GAN
training objective. The graph also has no minimax training update.

### One player with several scalarisations is not a game

Lenticulum's energies are already vectors, scalarised late ([[L-ADR-0002 Energies are vectors, scalarised late]]).
So write the payoff of an $n$-player game as one vector $u = (u_1, \dots, u_n)$ with the
scalarisations $\sigma_i(u) = u_i$, each to be minimised. Does that represent the game? It holds
every player's payoff, but not the game, because it says nothing about **who moves what**.

If one player moves all the parameters against all the $\sigma_i$, the problem is
multi-objective optimisation and its solution concept is Pareto optimality. For the GAN the two
come apart completely: with $u_2 = -u_1$, a point dominates another only if it lowers $u_1$ and
raises it at once, so **every** point is Pareto optimal and the saddle is not singled out.

What turns the vector into a game is an **owner** for each scalarisation: a block $\theta_i$ of
the parameters that moves against $\sigma_i$ alone, the other blocks held fixed. A point
$\theta^\ast$ is then a local Nash equilibrium when each $\theta_i^\ast$ is a local minimiser of
$u_i(\cdot, \theta_{-i}^\ast)$. First order: $\partial u_i/\partial\theta_i = 0$ for every $i$.
Ratliff, Burden & Sastry give a second-order version (*differential* Nash equilibria) and
compute such points. The pairs $(\sigma_i, \theta_i)$
are the players again; in a graph they are the "two factors" of this note's title.

In Lenticulum this would be a scalarisation chosen **per parameter key** over one graded energy:
the generator's parameters descend $\sigma_{\text{gen}}$, the discriminator's descend
$\sigma_{\text{disc}}$. Today every learnable node descends the one scalar $F$. A per-factor sign,
`objective_sign(factor)` $\in \{+1, -1\}$, is the special case $n = 2$, $u_2 = -u_1$; owned
scalarisations also cover games that are not zero-sum. This is a design, not implemented.

### What simultaneous gradient play computes

Each block descending its own scalarisation (simultaneous gradient play) has as fixed points
exactly the points where every player is stationary in its own parameters. They are not the
minimisers of any one function, but they are characterised: by the per-player first-order
conditions, and as local Nash equilibria by the second-order ones.

What fails is **convergence**. The bilinear game $u_1(x, y) = xy$, $u_2 = -xy$ (player 1 moves $x$,
player 2 moves $y$) has its only equilibrium at $0$. Simultaneous steps of size $\eta$,

$$
\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} 1 & -\eta \\ \eta & 1 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix},
$$

multiply the distance to it by $\sqrt{1 + \eta^2} > 1$ every step: the iterates spiral outwards for
every step size, and the continuous-time flow circles. Balduzzi et al. split the Jacobian of the
joint gradient field into a symmetric part (a potential, which descent handles) and an
antisymmetric part (the rotation above), and correct the step for the second (symplectic gradient
adjustment).

What does **not** carry over is the meaning of the number. There is no single $F$ that decreases,
so "on a tree the free energy is $-\log Z$" has no counterpart. What the graph computes would have
to be read from the game: each node's own scalarisation, stationarity per player, and the
stability of the joint Jacobian. That is a different promise from the one the rest of the vault
makes ([Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)).

> [!note] The obstruction generalises — and so does the escape
> [[The Two-Part Diagram]] places this finding in a taxonomy: GANs are the $f = -g$ row of a
> bilevel problem, and the sign obstruction is specific to *competitive* architectures. The
> $f = g$ row — EM, VAE, active inference, LQG control — is **not** obstructed, because
> coordinate descent on one objective is exactly what a single-signed free energy expresses.
>
> So the right summary is not "the framework cannot do two-part architectures". It is "the
> framework does the cooperative ones; the arguing ones need owned scalarisations, which it does
> not have yet".

## 5. What the missing structure actually is

A GAN is a two-player zero-sum game. The categorical treatment of games that composes like a
lens exists: **open games**, Ghani, Hedges, Winschel & Zahn
([arXiv:1603.04641](https://arxiv.org/abs/1603.04641)), with the Bayesian generalisation in
Bolt, Hedges & Zahn ([arXiv:1910.03656](https://arxiv.org/abs/1910.03656)).

An open game is a morphism in a symmetric monoidal category with a forward *play* map and a
backward pass carrying **coutility** and a **best-response** condition, drawn with string
diagrams, composing sequentially and in parallel — and faithful in the sense of preserving
Nash equilibria and off-equilibrium best responses.

An $n$-player game there is not one player with $n$ payoffs. It is the parallel composite of
$n$ one-player decisions, each reading its own component of the coutility, and the best-response
relation of a parallel composite is the product of the components' relations. That is the
open-game form of the ownership of §4: a decision is a scalarisation together with the strategy
that answers to it.

That completes a three-way pattern the vault is halfway through documenting:

| framework | the backward pass carries | vault note |
|---|---|---|
| parametric lens (Cruttwell et al.) | a **gradient** | [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens) |
| statistical game (AutoBayes) | a **posterior** | [[Factors are Parameterized Statistical Games]] |
| **open game** (Ghani et al.) | a **best response** | — |

All three are lens-shaped, all three have a forward and a backward pass, and they differ only
in what flows backwards. Lenticulum implements the second. A GAN needs the third.

> [!warning] A terminological collision worth flagging
> AutoBayes calls its central object a **statistical game** (Definition 20), and Lenticulum's
> factors are parameterized statistical games (Definition 27). That is *not* a game in the
> GAN or game-theoretic sense — there is one player and one loss. It is "game" as in
> "a lens with an objective attached".
>
> So "Lenticulum factors are games, and a GAN is a game, therefore…" is a pun, not an
> argument. The two notions meet only in the open-game framework, where a statistical game
> would be the one-player degenerate case.

## 6. What would follow from doing it properly

Not a to-do list — a note of what the open-game reading would buy, if anyone took it up:

- **Equilibria instead of minima.** The graph's solution concept becomes Nash rather than
  stationary, which is what a GAN actually converges to when it converges.
- **Adversarial factors compose.** Open games compose sequentially and monoidally, so a graph
  containing several adversarial pairs would have a meaning rather than a training script.
- **The discriminator's inexactness gets a home.** [[ratio]] §5 records that a `RatioFactor`'s
  *energy* is estimated and nothing accounts for it. In a game, "the other player has not best
  responded yet" is a first-class off-equilibrium condition rather than an unmodelled error.

Related: [[Implicit Generative Models]], [[Three Senses of Implicit]], [[Adversarial]],
[[The Two-Part Diagram]], [[The Inferencer and the Optimizer]],
[[Factors are Parameterized Statistical Games]], [Parametric Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Parametric-Lens), [[Bethe Free Energy]],
[[Lux as a Parametric Lens]], [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy)
