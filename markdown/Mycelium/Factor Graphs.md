# Factor Graphs

> The data structure `Mycelium.jl` is built on, and the two acyclicity questions that get
> confused with each other.

## Bipartite: wires and nodes

A factor graph has two kinds of node:

- **variables** — wires. They carry a belief and have no behaviour of their own.
- **factors** — [[Statistical Game|statistical games]]. They have channels, energies,
  entropies, inversions and (sometimes) parameters.

An **edge** attaches one *named channel* of a factor to one variable. Channels are named, not
positional, because a factor has no distinguished input: which channel is input is decided per
message, by [[Polarity Resolution]].

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[font=\small,
  fac/.style={draw, fill=black!8, minimum size=7mm},
  var/.style={draw, circle, minimum size=7mm}]
  \node[fac] (dx) at (0,0)   {$\mathrm{data}_x$};
  \node[var] (x)  at (2.2,0) {$x$};
  \node[fac] (m)  at (4.4,0) {model};
  \node[var] (y)  at (6.6,0) {$y$};
  \node[fac] (l)  at (8.8,0) {loss};
  \node[var] (t)  at (8.8,-2){$t$};
  \node[fac] (dt) at (8.8,-4){$\mathrm{data}_t$};
  \draw[->]     (dx) -- node[above,font=\tiny]{$x$} (x);
  \draw[<->]    (x)  -- node[above,font=\tiny]{in} (m);
  \draw[<->]    (m)  -- node[above,font=\tiny]{out} (y);
  \draw[->]     (y)  -- node[above,font=\tiny]{pred} (l);
  \draw[->]     (t)  -- node[right,font=\tiny]{target} (l);
  \draw[->]     (dt) -- node[right,font=\tiny]{$t$} (t);
  \node at (4.4,-2.6) {\footnotesize $\rightarrow$ Emitting/Absorbing \quad $\leftrightarrow$ Bidirectional};
\end{tikzpicture}
\end{document}
```

Four factors, three variables, six edges. Note what is **not** a variable: the data. See
[[Everything is a Factor]].

## Edges are directed; bidirectional is a case, not the default

Each edge carries one of three directions:

| direction | messages | the attached channel is |
|---|---|---|
| `Emitting()` | factor → variable only | never `Observed()` |
| `Absorbing()` | variable → factor only | never `Unobserved()` |
| `Bidirectional()` | both | either, per message |

**A bidirectional edge is exactly what "implicit" means.** A factor that can be run in either
direction along a wire is a relation; one that cannot is a function.

This is also where non-learnable and one-way factors enter naturally: a data source is
emitting-only, a loss is absorbing-only, and neither has parameters. They are not special
cases bolted onto the framework — they are ordinary factors whose edge directions happen to be
constrained.

## The two acyclicity questions

These get confused constantly, and they are independent.

| predicate | on what | decides |
|---|---|---|
| `istree(g)` | the **undirected** bipartite graph | whether message passing is **exact** |
| `isdag(g)` | the **directed** multigraph | whether the graph is **Lux-compatible** |

- `istree` is connectivity plus loop-freeness. On a tree, two sweeps give the true marginals
  and the true free energy ([[Schedules]]). Off a tree you are running loopy BP and nothing is
  guaranteed ([[Loopy Message Passing]]).
- `isdag` reads the directions: an emitting edge contributes the arc factor → variable, an
  absorbing edge variable → factor, and a **bidirectional edge contributes both**, which is
  already a 2-cycle.

So **any graph containing a bidirectional edge fails `isdag` by construction**, and that is
the correct behaviour rather than an artefact: `isdag(g)` is precisely the predicate *"this
graph could have been written in Lux"*.

The supervised example above is a **tree that is not a DAG** — exact inference, not
Lux-expressible. That combination is the whole point of the library.

## The Euler characteristic

$$\chi(g) \;=\; |F| + |V| - |E|$$

`1` for a connected tree, `1 - L` for a graph with `L` independent loops. It reappears in
[[Bethe Free Energy]] as the sum of the counting numbers, which makes it a single integer
measuring how badly the free-energy bookkeeping over-counts. The test suite asserts the two
computations agree, since that is the cheapest possible check that the graph and the accounting
have not drifted apart.

## Copying is a variable of degree > 2

The **copier** of [[Copiers Cups and Caps]] does not need a node type. A variable connected to
three factors *is* a copier: all three see the same value. Fan-out is therefore free, and the
degree $d_v$ is exactly the number of consumers — which is also, not coincidentally, what the
Bethe counting number corrects for.

Related: [[Everything is a Factor]], [[Messages are Inversions]], [[Schedules]], [[graph]]
