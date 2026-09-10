# The Structural Gap to ModelingToolkit

> [[ModelingToolkit as an Acausal Relation]] asks what MTK and Lenticulum have in common. This
> note asks the harder question in the other direction: **what does MTK contain, structurally,
> that Lenticulum does not?**
>
> Five things, ranked by how deep they go. Two of them are additions; three are not.
> The design that follows from them is [[Time as a Base]].

## 0. The ranking

| | gap | addable? |
|---|---|---|
| 1 | models are **terms**, not closures | **no** — a different design |
| 2 | a distinguished **independent variable**, and a derivation on it | **no** — structure on the variable category, not a factor |
| 3 | connectors carry **two** interacting structures, not one | yes, and it corrects [[Acausal Composition is a Hypergraph Category]] |
| 4 | **subsystems** are first-class and instantiable | yes; the formalisation is already in the vault |
| 5 | **events** — a system that rewrites its own equations | no |

Items 1 and 2 are the ones that matter. 3 is a correction. 4 is unfinished work. 5 is a
different subject.

## 1. The models are terms, not closures

An MTK equation is a **symbolic expression** in a term algebra. A Lenticulum factor is a Julia
object with methods — opaque to the framework.

Mycelium knows *which* channels a factor touches; that is the incidence graph, and
[[ModelingToolkit as an Acausal Relation]] §4 shows it is the same bipartite graph MTK's
structural analysis runs on. What Mycelium cannot know is *how*.

So MTK can ask questions Lenticulum structurally cannot:

- is this equation linear in $x$?
- are these two equations aliases of one another?
- can I solve this one for $x$ symbolically and substitute it into the others?

Which is exactly what `mtkcompile` is: **inspection followed by rewriting**. Alias elimination,
tearing and Pantelides index reduction all require looking inside a node.

The gap surfaces precisely in [[Composition is Elimination]]. That note argues elimination *is*
the composition operation of the framework, and the only version Lenticulum can implement is
the numerical one — marginalisation, or a solve. MTK does it symbolically, exactly, at compile
time.

> [!important] This one is not addable
> Making factors symbolic means making them **terms in a free algebra** rather than objects
> carrying methods. That is a different architecture, not a feature — and it would cost the
> thing the current design buys, namely that a factor's internals can be an arbitrary neural
> network ([[Lux as a Parametric Lens]]). You cannot do equational reasoning on a U-Net.
>
> The realistic conclusion is a division of labour, not a merge. See [[Time as a Base]] §8.

## 2. Time is external and shared; Lenticulum's is internal and sealed

Lenticulum *does* have time. `NeuralODEFactor` has a `tspan`; `VPSDE` has $t$. The difference
is where it lives:

> MTK's time is **external and shared** — every equation in the system lives over one base.
> Lenticulum's time is **internal and private** — sealed inside a single factor and integrated
> out before that factor talks to the graph.

`NeuralODEFactor` exposes $(z_0, z_1)$ and nothing else. **You cannot attach a factor to
$z(0.5)$.** And that is a little damning, because it is the same move
[[DEQ as a Relation]] criticises `DeepEquilibriumNetwork` for: sealing the solve inside and
handing back the endpoints. The factor un-seals the *direction* and keeps the *time* sealed.

### The algebraic content

MTK's variables live in a **differential ring**: `D = Differential(t)` is a derivation, and
equations relate variables *and their derivatives*. Lenticulum's variables live in a plain
space — `Channel{name,S}(space)` carries a space and nothing else, and there is no derivation
anywhere in `LenticulumCore` or `Mycelium`.

And `D` **cannot be added as a factor.** It relates a variable to itself at infinitesimally
nearby times, which a bipartite graph cannot express. You would introduce $\dot x$ as a second
variable with $\dot x = D(x)$ — but that relation is not an arbitrary factor. It is a fixed,
universal, non-learnable relation that the *solver* has to know about. It is structure on the
category of variables, not content in the graph.

> A factor graph's honest answer to time is **discretisation into a chain**, which is
> [[The Linear Gaussian Chain]] — i.e. Kalman filtering. That is a legitimate answer and a
> different one, not a worse one. [[Time as a Base]] is about what the continuous answer would
> look like.

## 3. A connector carries two interacting structures, not one

**This corrects [[Acausal Composition is a Hypergraph Category]] §2.**

That note argues a Mycelium variable of degree $d$ is a Frobenius spider — all legs carry equal
values, and the spider theorem says a $d$-way junction has no internal structure. That is right
for MTK's **across** variables: voltage, temperature, position, pressure. Equal at the junction.

But `@connector` also declares **through** variables — current, force, heat flow, mass flow —
and those do not copy. They **sum to zero**.

So a connector is not one spider. It is a *copying* structure and an *adding* structure on the
same object, which in the vocabulary that note already cites (Bonchi–Sobociński–Zanasi,
*Interacting Hopf Algebras*) is exactly an **interacting Hopf algebra** — strictly richer than
the single special commutative Frobenius algebra.

| | across / effort | through / flow |
|---|---|---|
| examples | voltage, temperature, position | current, heat flow, force |
| junction law | all equal | sum to zero |
| algebra | the copy comonoid — the spider | the add monoid, with the antipode for sign |
| in Mycelium | **a shared variable node** | **not available** |

And this is precisely why Kirchhoff's law had to be written as an explicit
`LinearConstraintFactor` in the divider test of
[[ModelingToolkit as an Acausal Relation]] §7: the sum-to-zero structure is not available *in
the variable node*, so it has to be added as a separate factor.

The deeper consequence: in MTK the interconnection semantics are attached to the **port type**,
not chosen at the call site. That is what makes acausal composition work uniformly across
electrical, mechanical, thermal and hydraulic domains — you connect two pins and the right
equations appear, because the pins know what they are. `LenticulumCore.Channel` carries a
dimension. No domain, no units, no through/across flag.

> [!note] This one is addable, and cheaply
> A `through`/`across` flag on `Channel`, plus generating the balance equation at
> `connect!`-time rather than making the user write it. The algebra is richer than what
> [[Acausal Composition is a Hypergraph Category]] describes, but it is well understood and
> the graphical-linear-algebra literature has a complete axiomatisation of it.

## 4. Subsystems are first-class

MTK has `@component`, hierarchical namespacing (`circuit.resistor1.p.v`), and instantiation —
one resistor model, twenty instances, flattened by `mtkcompile`.

Lenticulum has `AbstractLenticulumContainerFactor{factors}`, which nests **parameters**, and
flat graphs with one symbol namespace. There is no "collapse this subgraph into a factor"
operation.

The formalisation is already in the vault and is not the problem:
[[Acausal Composition is a Hypergraph Category]] §6 gives it as decorated cospans — an open
system is a subgraph with a boundary, and a boundary is a set of channels, so *an open subgraph
**is** a factor*. [[Composition is Elimination]] says what collapsing one means. §8 of
[[ModelingToolkit as an Acausal Relation]] sketches it as `MTKFactor`.

Nothing implements it. This is unfinished work rather than a structural obstacle — and
[[Time as a Base]] §4 notes that it is the *same* missing capability as the temporal extension,
because a belief over a trajectory is itself a chain-structured subgraph.

## 5. Events

MTK has continuous and discrete callbacks: a system that **changes its own equations** at event
times. A ball bounces; a switch closes; a controller saturates.

Lenticulum has no notion of a graph that rewires itself. That is genuinely deep — it is a
*dependent* structure, in which the graph is a function of the state, and none of the
machinery in this project (fixed `FactorGraph`, fixed schedules, fixed counting numbers, a
Bethe free energy whose Euler characteristic is computed once) survives a graph that changes
shape mid-inference.

### A smaller one, with a connection

MTK's **initialization system** solves a nonlinear system for consistent initial conditions —
necessary for any DAE of index $\ge 1$, because not every assignment of values satisfies the
hidden constraints.

That is precisely the problem whose absence appeared as the bootstrap deadlock in
`constraint.md` §4.2: a loopy acausal graph where every message needs every other channel to be
informative, so the first sweep produces nothing and so does every sweep after it. MTK has an
answer; Lenticulum works around it with weak regularising priors.

## 6. The other direction, for honesty

MTK has no probability, no beliefs, no learned factors, and no notion of an *inexact* inversion
whose cost is measured. Its equations are hard and known, and its parameters are calibrated
rather than learned.

That is the whole of Lenticulum's claim over it, and it is a real one. But it is narrower than
"Lenticulum is MTK plus noise" — items 1–5 above are the price, and items 1 and 2 are not
payable.

Related: [[ModelingToolkit as an Acausal Relation]], [[Time as a Base]],
[[Acausal Composition is a Hypergraph Category]], [[Composition is Elimination]],
[[Differential Algebra and DAE Factors]], [[The Linear Gaussian Chain]],
[[Lux as a Parametric Lens]]
