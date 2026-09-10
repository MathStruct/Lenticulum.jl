# ModelingToolkit.jl as an Acausal Relation

> The companion to [[Lux as a Parametric Lens]]. That note says: *Lux converged on the
> parametric lens without meaning to.* This one says: **ModelingToolkit converged on the
> relation** — the thing [[README]] promises and `GaussianFactor` does not deliver — and it
> did so twenty years after control theory got there first.
>
> Lux is the explicit column of the README's table. MTK is the implicit column, minus the
> probability.

## 1. The claim

[[Implicit Learners]] defines an implicit learner by a residual

$$r_\theta : X_1\times\cdots\times X_n \to E,
\qquad (x_1,\ldots,x_n) \in R_\theta \iff r_\theta(x_1,\ldots,x_n) \approx 0$$

A ModelingToolkit model is a list of equations `0 ~ expr`. That *is* $r_\theta = 0$, with
$\theta$ the parameters and $\approx$ replaced by $=$. MTK is not *like* an implicit learner;
it is the deterministic, hard-constraint, hand-written special case of one:

| | approximator | $\approx$ | learned? |
|---|---|---|---|
| MTK | symbolic equations you write | **exact** ($=$) | no — parameters are calibrated, not learned |
| Lenticulum | any residual, incl. neural | soft, $\|r\|^2_{Q^{-1}}$ | yes |

So the interesting question is not "is MTK a lens" — it is not — but *what structure does
acausal composition have*, and the answer is not a lens either. It is a hypergraph category;
see [[Acausal Composition is a Hypergraph Category]].

## 2. Three different lens-shaped things in MTK, only one of which is a lens

This is worth separating carefully, because "MTK is a lens" is both true and false depending
on which of these you mean.

### 2.1 `SymbolicIndexingInterface` — a genuine, classical lens

`getu(sys, x)` reads a variable out of a solution; `setu!(prob, x)` writes one in. That is

$$\mathrm{get} : S \to A, \qquad \mathrm{put} : S \times A \to S$$

which is the **classical** (non-bimorphic, non-parametric) lens of [[Lens]] — the functional
programming one, $A' = A$ and $S' = S$. It is the least interesting of the three and the only
one that is literally a lens.

### 2.2 The adjoint / sensitivity pass — a parametric lens, exactly as in Lux

`SciMLSensitivity` gives $\partial(\text{solution})/\partial(\text{parameters})$ via the
adjoint ODE/DAE. Compose that with the solve and you have

$$f : P \times A \to B \quad (\text{solve}),
\qquad f^* : P\times A\times B' \to P'\times A' \quad (\text{adjoint})$$

which is [[Parametric Lens]] on the nose. **In this reading MTK is just Lux with a stiff
solver in the middle**, and everything in [[Lux as a Parametric Lens]] applies verbatim,
including the DAG restriction — because a `solve` has a direction.

For DAEs of index $\ge 2$ there is a genuine subtlety here, and the vault already has it:
[[Differential Algebra and DAE Factors]] §"The adjoint of a DAE".

### 2.3 Acausal composition — **not** a lens, and this is the interesting one

`connect(a.port, b.port)` has no direction. It is not a `get` and there is no `put`. Two
components joined at a port do not compose as functions; they compose as *relations*, by
sharing variables. Function composition cannot express it, which is exactly why
[[Lux as a Parametric Lens]] §"Constraint 1" says the wiring must be a DAG.

> [!important] The three structures answer three different questions
> - §2.1: how do I *read* a variable? — a lens.
> - §2.2: how do I *differentiate* a solve? — a parametric lens.
> - §2.3: how do I *build a model out of components*? — a hypergraph category.
>
> Lenticulum needs the third. It is the only one with no counterpart in Lux, and it is the
> reason `Mycelium.jl` exists.

## 3. Willems got there first

MTK's acausal semantics are, essentially exactly, Jan Willems' **behavioral approach** to
systems theory. The correspondence is close enough that it is worth learning the older
vocabulary, because Willems is much more explicit about what the operations mean.

A system, for Willems, is $(\mathbb{T}, \mathbb{W}, \mathcal{B})$ with
$\mathcal{B} \subseteq \mathbb{W}^{\mathbb{T}}$ the **behavior** — the set of admissible
trajectories. A system *is* its behavior; it is not an input–output map, and asking which
variables are inputs is a question about a *representation*, not about the system.

| Willems | ModelingToolkit | Lenticulum |
|---|---|---|
| behavior $\mathcal{B}$ | the equations of a `System` | the relation $R_\theta$, softened to a density |
| manifest variables | `unknowns` / `observed` you asked for | channels with an edge |
| **latent** variables | internal variables introduced by components | `Latent()` polarity, $\llbracket c\rrbracket$ |
| **interconnection = variable sharing** | `connect` | **a shared variable node** |
| elimination of latent variables | `mtkcompile` (alias elimination, tearing) | marginalisation |
| manifest behavior = projection | the simplified system | the marginal belief |
| "tearing, zooming and linking" | hierarchical components | subgraphs of a factor graph |

Two of those rows deserve emphasis.

**Interconnection is variable sharing.** Willems' point is that joining two systems is not
plugging an output into an input; it is *asserting that two variables are equal* (or equal and
opposite, for through-variables — which is exactly MTK's `flow` and Kirchhoff's law). In a
factor graph a variable node shared by two factors is precisely that assertion. **`connect`
and "a variable in a factor graph" are the same operation.**

**Elimination is projection is marginalisation.** Willems' elimination theorem says the latent
variables of a linear differential system can always be eliminated, at the cost of raising the
differential order. MTK's `mtkcompile` does exactly this computationally. And in the
probabilistic setting, projecting a *set* becomes marginalising a *measure* — the same arrow,
one level up. That is the single cleanest statement of what Lenticulum adds to MTK:

> MTK eliminates latent variables by projection. Lenticulum eliminates them by
> marginalisation. Set-theoretic projection is the $Q \to 0$ limit of Gaussian
> marginalisation.

## 4. MTK's incidence graph *is* a factor graph

This is the structural match, and it is exact rather than analogical.

MTK's structural analysis — `mtkcompile`, alias elimination, tearing, the Pantelides
algorithm — all operate on the **bipartite incidence graph of equations against variables**:
an edge whenever variable $v$ appears in equation $e$. That is a factor graph, with equations
as factors:

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[
  var/.style={circle, draw, minimum size=7mm, inner sep=0pt},
  fac/.style={rectangle, draw, fill=black!12, minimum size=5mm, inner sep=1.5pt},
  every node/.style={font=\small}
]
\node[fac] (e1) at (0,1.2)  {$e_1$};
\node[fac] (e2) at (0,0)    {$e_2$};
\node[fac] (e3) at (0,-1.2) {$e_3$};
\node[var] (v)  at (2.6,1.2)  {$v$};
\node[var] (i1) at (2.6,0)    {$i_1$};
\node[var] (i2) at (2.6,-1.2) {$i_2$};
\draw (e1)--(v) (e1)--(i1) (e2)--(v) (e2)--(i2) (e3)--(i1) (e3)--(i2);
\node at (1.3,-2.1) {\footnotesize equations $\times$ variables $=$ a factor graph};
\end{tikzpicture}
\end{document}
```

| MTK structural analysis | Mycelium |
|---|---|
| incidence graph of equations $\times$ variables | [[Factor Graphs\|`FactorGraph`]] |
| one equation | a factor |
| one variable | a variable node |
| a variable appearing in an equation | an edge |
| **matching** equations to variables to solve for | choosing a [[Polarity Resolution\|polarity]] per factor |
| Dulmage–Mendelsohn / maximal matching | — (Mycelium has no matching step) |
| **tearing** — pick a small set of variables to iterate on | — (Mycelium has [[Schedules\|schedules]] instead) |
| alias elimination ($x = -y$) | — |
| Pantelides index reduction | — ([[Differential Algebra and DAE Factors]]) |

The left column's first four rows are implemented; the last four are not. And the gap is
informative: **MTK's structural analysis is about finding a good *elimination order*, whereas
Mycelium's schedules are about finding a good *message order*.** On a tree these coincide
(see [[The Linear Gaussian Chain]] §6, which flags the same absence for GTSAM). Off a tree they
do not, and MTK's answer is better — see §6.

> [!note] A polarity is a matching
> `resolve_polarity` marks exactly one channel `Unobserved()`: "this factor is being solved
> for this variable". Doing that for every factor at once, consistently, is precisely a
> **perfect matching** in the incidence graph, which is what MTK computes with
> Dulmage–Mendelsohn. Mycelium picks the matching one message at a time and never checks that
> the choices are globally consistent, because on a tree they always are.

## 5. The dictionary

| ModelingToolkit | Lenticulum / Mycelium |
|---|---|
| `System` (was `ODESystem`, `NonlinearSystem`, …) | a `FactorGraph`, or a container factor |
| one equation `0 ~ expr` | one factor; `LinearConstraintFactor` in the linear case |
| `@variables` | variable nodes |
| `@parameters` | `ps` |
| `@connector`, `connect` | a shared variable node |
| `flow` variable (through) — sums to zero | a balance constraint, $\sum_i x_i = 0$ |
| non-`flow` variable (across) — equal | a shared variable node, i.e. the Frobenius spider |
| `defaults` | a prior factor |
| `guess` (initialisation hint) | the initial message in the store |
| `mtkcompile` | *nothing* — see §4 |
| `observed` equations | derived quantities; a `RelayFactor` |
| `solve` | `infer!` |
| `getu` / `setu` | reading a marginal out of the returned `NamedTuple` |
| residual tolerance | $Q$, the equation's noise covariance |
| — | **beliefs**: MTK has no distributions |
| `SciMLSensitivity` adjoint | the gradient half of the statistical game |

The two rows with a dash in them are the whole story: MTK has structural machinery Lenticulum
lacks, and Lenticulum has probability MTK lacks.

## 6. Why MTK does not propagate, and what that says about `Mycelium`

A linear acausal system is **one linear solve**. MTK assembles the whole sparse Jacobian and
factorises it. Belief propagation instead iterates local messages to a fixed point, and on
this class of problem that is strictly worse:

- on a tree, BP is exact and equivalent to elimination — fine, but so is a direct solve;
- **off a tree, BP gets the means right and the variances wrong** (Weiss–Freeman), whereas a
  direct solve gets both right;
- an acausal cycle with no priors **cannot even start**: every message needs every other
  channel to be informative, so the first sweep produces nothing and so does every sweep
  after it. `constraint.md` §4.2 records this; the test suite works around it with weak
  regularising priors, which is what practitioners do and is still a workaround.

Circuits are loopy. Mechanisms are loopy. Chemical networks are loopy. **The acausal models
people actually build are exactly the ones where belief propagation is the wrong algorithm**,
and MTK's choice to solve rather than propagate is not a limitation but the correct call.

> [!warning] The honest conclusion for this repository
> Lenticulum's value over MTK is *not* a better way to solve acausal equations. It is that
> the equations may be **soft, learned, and noisy** — a factor with $Q > 0$ and parameters
> fitted from data, which MTK cannot express at all. Where the equations are hard and known,
> MTK is better, and a Lenticulum graph should be calling a solver rather than passing
> messages.
>
> That suggests the right long-term relationship is not "reimplement MTK" but **an
> `MTKFactor` whose inversion is `solve`** — a subgraph of hard equations collapsed into one
> factor, exactly as [[Composition is Elimination]] describes. See §8.

## 7. What was implemented

`LinearConstraintFactor` (`src/constraint.jl`, note in `constraint.md`): the acausal linear
equation

$$0 \;=\; \sum_i A_i x_i - c + \varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,Q)$$

over any number of channels, none distinguished. It is MTK's `0 ~ ...` softened by a noise
term, and it is the n-ary generalisation of `GaussianFactor` — the two agree exactly on
messages, residual and free energy, which the test suite asserts.

What it buys, concretely:

- **`supported_polarities` returns $n$, not 2.** That number *is* acausality: [[README]]'s
  "no distinguished input/output" row, finally true of a factor in this repository.
- **Connector equations become expressible.** Kirchhoff's law at a $d$-way node is one
  `LinearConstraintFactor` with $A_i = I$ and $c = 0$. The test suite reconciles three
  inconsistent flow measurements against one conservation law — a real industrial problem,
  and the smallest graph in which one factor is used in three directions in a single sweep.
- **The loopy story is testable.** The resistive divider is asserted to give exact means and
  wrong variances, which is the acausal difficulty in its smallest form.

What was deliberately *not* done:

> [!important] ModelingToolkit is not a dependency, and should not become one
> This repository depends on **LuxCore**, not Lux — the tiny interface package. The
> corresponding choice on the MTK side is `SymbolicIndexingInterface` at most, and nothing at
> all for now. Taking on MTK's dependency tree to gain one factor type would be a very bad
> trade at the current size of this project, and the categorical content — which is what
> [[Prompt3]] asked for — needs no dependency whatsoever.

## 8. Open problems

> [!important] Two of these are now written up
> [[The Structural Gap to ModelingToolkit]] answers "what does MTK have that Lenticulum does
> not", in five ranked items — and item 3 of that note **corrects**
> [[Acausal Composition is a Hypergraph Category]] on through-variables.
> [[Time as a Base]] designs the temporal extension that items 1 and 2 of it demand, and
> argues that problem 1 below is the only arrangement that survives.


1. **An `MTKFactor` whose inversion is a solver.** Wrap a hard subsystem, expose its manifest
   variables as channels, implement `invert` by calling `solve`, and charge entropy for solver
   error. This is the [[Statistical Game]] reading of a simulator, and it is the version of
   this work that would actually be worth an MTK dependency. Blocked on nothing but scope.
2. **A matching step.** §4 notes that a globally consistent assignment of polarities is a
   perfect matching. Mycelium never computes one. Doing so would catch structurally singular
   graphs — MTK's most useful error message — before any message is passed.
3. **Latent-channel marginalisation.** `constraint.md` §4.1: the projection
   $\Pi = Q^{-1} - Q^{-1}A_j(A_j^\top Q^{-1}A_j)^+A_j^\top Q^{-1}$ is the sharp answer the
   code currently throws away, and it is exactly Willems' elimination in the linear-Gaussian
   case.
4. **Derivatives.** No `D(x)`, hence no DAE and no index. See
   [[Differential Algebra and DAE Factors]] for what that costs and why index $\ge 2$ makes
   the *adjoint* wrong rather than merely slow.
5. **Units and domains.** MTK's `@connector` carries physical types. Lenticulum channels carry
   a dimension and nothing else.

## Sources

- Willems, *The Behavioral Approach to Open and Interconnected Systems*, IEEE Control Systems
  Magazine 27(6), 2007 — behaviors, interconnection as variable sharing, the elimination
  theorem.
- [ModelingToolkit.jl documentation](https://docs.sciml.ai/ModelingToolkit/stable/) — v10
  unified the per-domain `System` types and renamed `structural_simplify` to `mtkcompile`.
- Pantelides, *The consistent initialization of differential-algebraic systems*, 1988 — index
  reduction on the incidence graph.

Related: [[Lux as a Parametric Lens]], [[Acausal Composition is a Hypergraph Category]],
[[The Structural Gap to ModelingToolkit]], [[Time as a Base]],
[[Implicit Learners]], [[Factor Graphs]], [[Polarity Resolution]],
[[Differential Algebra and DAE Factors]], [[Composition is Elimination]],
[[Loopy Message Passing]], [[The Linear Gaussian Chain]]
