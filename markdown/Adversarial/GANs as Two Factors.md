# GANs as Two Factors

> **Can the GAN diagram be achieved with two factors?**
>
> The wiring: yes — two *parameter sets*, three *factor nodes*, and the third is the second
> one again under weight tying. The objective: **no**, and the obstruction is one sign.
>
> Implemented in `lib/Adversarial.jl`; see [[Adversarial]].

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

$$\mathbb{E}_{x\sim p_{\text{data}}}\bigl[\log D_\varphi(x)\bigr]
\;+\;
\mathbb{E}_{z\sim q}\bigl[\log\bigl(1 - D_\varphi(G_\theta(z))\bigr)\bigr]$$

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

## 4. The obstruction: one sign

Here is what the graph cannot hold.

The Bethe free energy ([[Bethe Free Energy]]) is a **single scalar**:

$$F \;=\; \sum_c F_c \;+\; \sum_v (1-d_v)H_v$$

and every learnable factor descends $\partial F/\partial\theta_c$. A GAN is a **minimax**:

$$\min_\theta \max_\varphi V(\theta,\varphi)$$

$\theta$ descends, $\varphi$ **ascends**, on the same quantity. There is no assignment of
per-factor free energies whose common descent reproduces that, because descent has one
direction and the game has two.

`lib/Adversarial.jl` therefore computes the number and cannot act on it:
`local_free_energy(::RatioFactor, …)` returns $\mathbb{E}_b[-\log r]$, an estimate of
$\mathrm{KL}(q\|p)$ — the quantity $G$ descends and $D$ ascends. **The graph holds the number;
it cannot hold the two signs.**

### The minimal patch, and why it is not enough

A per-factor sign — `objective_sign(factor) ∈ {+1,-1}` — would make simultaneous gradient
descent-ascent expressible. That is genuinely all GAN *training* does in practice.

It is not enough as *theory*, for a reason the vault should care about: simultaneous
gradient descent-ascent is not guaranteed to converge to anything, and the fixed points it
does find are not characterised by minimising any function. A sign flag would let you run the
algorithm while telling you nothing about what it computes. The framework's whole selling
point is that the free energy *means* something ([[Variational Free Energy]]).

> [!note] The obstruction generalises — and so does the escape
> [[The Two-Part Diagram]] places this finding in a taxonomy: GANs are the $f = -g$ row of a
> bilevel problem, and the sign obstruction is specific to *competitive* architectures. The
> $f = g$ row — EM, VAE, active inference, LQG control — is **not** obstructed, because
> coordinate descent on one objective is exactly what a single-signed free energy expresses.
>
> So the right summary is not "the framework cannot do two-part architectures". It is "the
> framework does the cooperative ones and not the arguing ones".

## 5. What the missing structure actually is

A GAN is a two-player zero-sum game. The categorical treatment of games that composes like a
lens exists: **open games**, Ghani, Hedges, Winschel & Zahn
([arXiv:1603.04641](https://arxiv.org/abs/1603.04641)), with the Bayesian generalisation in
Bolt, Hedges & Zahn ([arXiv:1910.03656](https://arxiv.org/abs/1910.03656)).

An open game is a morphism in a symmetric monoidal category with a forward *play* map and a
backward pass carrying **coutility** and a **best-response** condition, drawn with string
diagrams, composing sequentially and in parallel — and faithful in the sense of preserving
Nash equilibria and off-equilibrium best responses.

That completes a three-way pattern the vault is halfway through documenting:

| framework | the backward pass carries | vault note |
|---|---|---|
| parametric lens (Cruttwell et al.) | a **gradient** | [[Parametric Lens]] |
| statistical game (AutoBayes) | a **posterior** | [[Statistical Game]] |
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
[[Statistical Game]], [[Parametric Lens]], [[Bethe Free Energy]],
[[Lux as a Parametric Lens]], [[Variational Free Energy]]
