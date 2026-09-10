# The Two-Part Diagram

> GANs, actor–critic reinforcement learning, and a controller with an observer all draw the
> same picture: **the architecture splits in two and the halves close a loop.**
>
> This note says what is actually shared, and — more usefully — what *is not*. The shared part
> is the wiring, and the vault already has the structure for it. The unshared part is the
> **objective**, and that is what decides whether an architecture fits in this framework at
> all.

## 1. The picture

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  box/.style={rectangle, draw, minimum width=22mm, minimum height=9mm, inner sep=3pt},
  every node/.style={font=\small}
]
\node[box] (A) at (0,0)    {acts / produces};
\node[box] (B) at (0,-2.2) {evaluates / estimates};
\draw[->, thick] (A.east) -- ++(1.6,0) |- node[pos=0.25, right] {\footnotesize $x$} (B.east);
\draw[->, thick] (B.west) -- ++(-1.6,0) |- node[pos=0.25, left] {\footnotesize score} (A.west);
\end{tikzpicture}
\end{document}
```

One half emits a configuration; the other half judges it; the judgement returns. Filling in
the boxes:

| | acts / produces | evaluates / estimates | what returns |
|---|---|---|---|
| **GAN** | generator $G_\theta$ | discriminator $D_\varphi$ | $\log(1-D(G(z)))$ |
| **actor–critic** | policy $\pi_\theta$ | value $V_\varphi$ | the advantage |
| **control** | controller | plant + observer | the measurement / error |
| **VAE** | decoder | encoder | the ELBO |
| **EM** | M-step | E-step | the posterior |
| **active inference** | action | perception | the free energy |

The convergence is real. What it is *not* is a single theory — the last three rows behave
completely differently from the first, and §3 is about why.

## 2. The wiring is not the problem

A loop in a diagram is a **trace**, and the vault has already dealt with traces.

[[Lux as a Parametric Lens]] §"Constraint 1" records why a Lux `Chain` cannot express this:
lens composition is function composition, and function composition around a cycle does not
terminate. [[Copiers Cups and Caps]] is the fix — compact closure lets you bend an output wire
into an input wire, so a cycle becomes a straight line with a bent end, and every compact
closed category has a canonical trace. [[Acausal Composition is a Hypergraph Category]] goes
one rung further.

So:

> [!important] Lenticulum can already *draw* every diagram in §1
> A factor graph has no direction, so it has no cycles to worry about. Two factors sharing two
> variables is a "loop" only if you insist on reading the edges as arrows, and a factor graph
> does not.
>
> The obstruction is never the wiring. It is always the objective.

## 3. What actually differs: one objective or two

Every architecture in §1 is an instance of **bilevel optimisation**:

$$\min_\theta\; f\bigl(\theta,\ \varphi^\ast(\theta)\bigr)
\qquad\text{subject to}\qquad
\varphi^\ast(\theta) = \arg\min_\varphi\; g(\theta,\varphi)$$

and the whole taxonomy is the relationship between $f$ and $g$:

| case | is | examples | fits Lenticulum? |
|---|---|---|---|
| $f = g$ | **coordinate descent on one objective** | EM, VAE, active inference, LQG | **yes** |
| $f = -g$ | **minimax**, zero-sum | GAN, $H_\infty$ / robust control | no |
| $f \ne \pm g$ | general bilevel / Stackelberg | actor–critic, meta-learning | no |

The Bethe free energy ([[Bethe Free Energy]]) is a **single scalar** that every learnable
factor descends. That is exactly the top row and nothing else.

[[GANs as Two Factors]] §4 established this for GANs — *two parameter sets, three factor
nodes, and one sign the graph cannot hold*. The point of this note is that the finding
generalises, and generalises *favourably*: the sign obstruction is specific to the
competitive rows. **The cooperative row is not obstructed at all**, and it contains most of
control theory and all of variational inference.

### Why $f = g$ is not really a loop

Coordinate descent alternates, but it descends one function. There is no equilibrium to seek,
no oscillation to damp, no best-response to compute — just a sequence of partial
minimisations of a single objective, each of which decreases it.

That is the precise sense in which the cooperative architectures are *not* feedback loops even
though they are drawn as one. They are **alternating minimisation wearing a loop's clothes**,
and it matters because alternating minimisation converges under conditions you can state,
while simultaneous gradient descent–ascent does not.

## 4. Willems: control is interconnection

Control theory has its own version of "the loop is an artefact of insisting on arrows", and
the vault already imported it.

[[ModelingToolkit as an Acausal Relation]] §3 records Willems' behavioural view: a system *is*
its set of admissible trajectories, and **interconnection is variable sharing** — not plugging
an output into an input. In that framework a controller is simply another system you
interconnect with the plant, restricting the joint behaviour to what you want.

> A block diagram with a feedback arrow and a factor graph with a shared variable are the same
> object. The arrow is a choice of representation; the shared variable is the system.

Which means the controller/plant loop, in this project's terms, is *two factors sharing two
variables* — and nothing about that is hard. The vault demonstrated exactly this shape
already, and found it loopy in the graph-theoretic sense: the resistive divider in
[[ModelingToolkit as an Acausal Relation]] §6 is a three-factor cycle, and the honest finding
there was that message passing handles it badly compared to a direct solve.

So the difficulty with control in Lenticulum is not the feedback. It is that a loopy graph
gets exact means and wrong variances ([[Loopy Message Passing]]), which for a controller means
your gains are right and your confidence in them is not.

## 5. Active inference: the case that fits

If the cooperative row is the one that fits, the natural question is what lives there. The
fullest answer is **active inference**: perception and action both minimising *one* variational
free energy — perception over beliefs, action over policies.

That is $f = g$ exactly, and it is the same functional the Bethe machinery already computes.
It is also not speculative in Julia: **ForneyLab.jl** and its successor **RxInfer.jl** do
Forney-style factor graphs with (variational) message passing and build active-inference
agents that minimise free energy by message passing. That ecosystem is the closest existing
neighbour to this project, and the closest thing to a demonstration that the cooperative
architectures work as factor graphs.

The difference in ambition is worth stating plainly: RxInfer does *inference* on a specified
probabilistic model, extremely well. Lenticulum is trying to do inference on a graph whose
factors may be **learned, implicit and non-probabilistic** — an arbitrary residual, a DEQ, a
diffusion prior. That is more general and much less finished.

## 6. What this means for the three competitive rows

They are not out of reach, but they need structure that is not here:

- **Minimax ($f = -g$)** needs [[GANs as Two Factors]] §5's open games — a third lens-shaped
  object where the backward pass carries a **best response** rather than a gradient or a
  posterior. The solution concept becomes Nash instead of stationary.
- **General bilevel ($f \ne \pm g$)** — actor–critic, meta-learning — needs the *hypergradient*
  $\mathrm{d}f/\mathrm{d}\theta$ through $\varphi^\ast(\theta)$, which is the implicit function
  theorem applied to the inner optimum. The vault has that machinery:
  [[Backpropagation by the Implicit Function Theorem]], and `ImplicitLayers`'s
  `ift_sensitivity` computes exactly this shape of object for a fixed point.

  That is a genuine and unexploited connection: **bilevel optimisation and deep equilibrium
  models have the same backward pass.** Both differentiate through an argmin/fixed point via
  one linear solve. `ImplicitLayers` implements it for the equilibrium family and nobody has
  pointed it at the outer problem.

## 7. The summary

| | wiring | objective | status |
|---|---|---|---|
| the loop | a trace; compact closure | — | **solved, twice over** |
| cooperative ($f=g$) | shared variables | one free energy | **expressible today** |
| minimax ($f=-g$) | shared variables | two signs | needs open games |
| general bilevel | shared variables | hypergradient | needs the IFT at the outer level |

The one-line version: **the two-part diagram is never a wiring problem, and whether it is a
problem at all depends entirely on whether the two halves are arguing.**

And the architecture in [[The Inferencer and the Optimizer]] is deliberately one where they
are not.

## Sources

- Willems, *The Behavioral Approach to Open and Interconnected Systems*, IEEE CSM 2007 —
  control as interconnection.
- Ghani, Hedges, Winschel & Zahn, *Compositional Game Theory*,
  [arXiv:1603.04641](https://arxiv.org/abs/1603.04641) — open games, for the minimax row.
- Friston, *The free-energy principle: a unified brain theory?*, Nat. Rev. Neurosci. 2010;
  Friston et al., *Active Inference: A Process Theory*, Neural Computation 2017.
- [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl) and
  [ForneyLab.jl](https://github.com/biaslab/ForneyLab.jl) — Forney-style factor graphs,
  message passing, and active-inference agents, in Julia.
- Feldbaum, *Dual control theory* I–IV, 1960–61 — why separation fails; see
  [[The Inferencer and the Optimizer]] §5.

Related: [[The Inferencer and the Optimizer]], [[GANs as Two Factors]],
[[Copiers Cups and Caps]], [[Acausal Composition is a Hypergraph Category]],
[[ModelingToolkit as an Acausal Relation]], [[Bethe Free Energy]],
[[Loopy Message Passing]], [[Lux as a Parametric Lens]],
[[Backpropagation by the Implicit Function Theorem]]
