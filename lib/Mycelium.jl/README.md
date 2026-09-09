# Mycelium.jl

Factor graphs and message passing for [Lenticulum.jl](../../README.md).

`LenticulumCore` says what a **factor** is (a parameterized statistical game, AutoBayes
Definition 27). `Mycelium` says how factors are **wired** and in what **order** they talk.
This is the layer with no counterpart in Lux, because in Lux the wiring is a DAG and the
message order is implied by it.

```
FactorGraph(3 variables, 4 factors, 6 edges; χ=1, tree, cyclic)
istree = true   isdag = false   χ = 1
tree schedule: 8 messages, 4 pruned by direction
ConvergenceReport(converged after 1 sweep(s), residual 0.0; tree schedule: one sweep is exact)
marginals: x=2.0 y=2.0 t=5.0
graded energy E_G = ⊕_f E_f: (data_x = 0.0, model = 0.0, data_t = 0.0, loss = 4.5)
counting numbers: (x = -1, y = -1, t = -1)  total = 1
scalar Bethe free energy = 4.5
```

## Five ideas

1. **Bipartite.** Variables are wires; factors are nodes. *Everything with content is a
   factor* — data, priors, losses and optimisers included, which is forced by AutoBayes
   Remark 24. → `factors.jl`
2. **Edges are directed**, one of `Emitting` / `Absorbing` / `Bidirectional`. A bidirectional
   edge is exactly what "implicit" means. → `graph.jl`
3. **A factor → variable message *is* an inversion** $c'_\pi$: the target channel becomes
   `Unobserved()`, channels with messages become `Observed()`, the rest `Latent()`.
   → `polarity_resolution.jl`, `passing.jl`
4. **Scheduling is the whole problem.** On a tree, two sweeps are exact; off a tree you are
   running loopy BP. → `schedules.jl`
5. **The free energy of a graph is the Bethe form**: energies add, entropies get a counting
   correction $(1-d_v)$ — the same asymmetry AutoBayes identifies, one level up.
   → `free_energy.jl`

## Two acyclicity notions, not one

| predicate | on | decides |
|---|---|---|
| `istree(g)` | the **undirected** bipartite graph | whether message passing is **exact** |
| `isdag(g)` | the **directed** multigraph | whether the graph is **Lux-compatible** |

They are independent. The example above is a **tree that is not a DAG**, which is the
combination this library exists for.

## Documentation

Concept notes: `markdown/Mycelium/` — [[Factor Graphs]], [[Everything is a Factor]],
[[Messages are Inversions]], [[Polarity Resolution]], [[Schedules]], [[Bethe Free Energy]],
[[Loopy Message Passing]].

Implementation notes sit next to each source file, per [[Start here]].

## Tests

```
julia --project=lib/Mycelium.jl/test -e 'include("lib/Mycelium.jl/test/runtests.jl")'
```
