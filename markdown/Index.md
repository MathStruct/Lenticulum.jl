# Index — Lenticulum concept vault

This is the map of content for the theory behind Lenticulum.jl. Two papers are the
backbone, and one bridge layer connects them to Julia:

- **[Categorical Foundations of Gradient-Based Learning](https://arxiv.org/html/2103.01931v2)**
  (Cruttwell, Gavranović, Ghani, Wilson, Zanasi) — how *ordinary* deep learning is a
  **parametric lens**. This is the theory Lux.jl already implements without saying so.
- **[AutoBayes](https://arxiv.org/html/2503.18608v2)** (St Clere Smithe & Perin) — how
  *Bayesian* inference is a **statistical game**, which is a parametric lens whose
  backward pass is a posterior rather than a gradient. This is what Lenticulum.jl
  implements.

Read in this order.

## 1. The classical story (what Lux.jl is)

1. [[Para]] — parameters as a categorical construction
2. [[Lens]] — forward/backward pairs, `get` and `put`
3. [[Parametric Lens]] — Definition 2.5, the object Lux layers really are
4. [[Cartesian Reverse Differential Category]] — where the gradient comes from
5. [[Learning Components as Parametric Lenses]] — model, loss, learning rate, optimiser
6. [[Lux as a Parametric Lens]] — **the concrete Lux.jl ↔ lens dictionary**

## 2. The Bayesian story (what Lenticulum.jl is)

7. [[Open Model]] — Definition 1: a kernel with an explicit latent space
8. [[Composition of Open Models]] — Definitions 4–7
9. [[Copiers Cups and Caps]] — how the DAG restriction is lifted
10. [[Bayesian Inversion]] — Bayes' law as a chain rule
11. [[Bayesian Lens]] — Definitions 9–11
12. [[Composition of Bayesian Lenses]] — Definition 12, Theorem 13
13. [[Variational Free Energy]] — Definition 17, Proposition 18: the energy/entropy split
14. [[Statistical Game]] — Definition 20, **the definition Lenticulum factors satisfy**
15. [[Composition of Statistical Games]] — Definition 22, Theorem 23 (the chain rule)
16. [[Parameterized Statistical Game]] — Definition 27, what actually gets optimised
17. [[Composition of Gradients]] — Definitions 28–29, Remark 30: laxness
18. [[Examples from the Paper]] — Appendix A, worked through

## 3. The bridge to Lenticulum.jl

19. [[Channels and Polarity]] — reconciling AutoBayes' X/⟦c⟧/Y with the README's
    $P_{in} + P_{out} + P_{latent} = \mathrm{Id}$
20. [[Scalar and Multivariate Energy]] — **the design decision that is ours, not the
    paper's**: two energies, and the chain rule adapted to the multivariate one
21. [[Implicit Learners]] — the three model families, and why the multivariate energy
    is what makes them work
22. [[AutoBayes to Lenticulum]] — the full naming dictionary paper → Julia

## 3b. The algebraic family, worked out

The first of the three [[Implicit Learners]] families in full detail — the case where every
abstract slot of the framework becomes computable, so the framework can be checked.
Entry point: [[Algebraic Implicit Learners]].

- *model class*: [[Varieties Ideals and Real Nullstellensatz]], [[The Veronese Parametrisation]],
  [[Universal Approximation by Nash-Tognoli]]
- *learning*: [[Fitting is a Nullspace Problem]], [[The Parameter is a Grassmannian]],
  [[Algebraic versus Geometric Distance]]
- *inference*: [[Inference as Root Finding]], [[Branches and the Discriminant]],
  [[Backpropagation by the Implicit Function Theorem]]
- *structure*: [[Composition is Elimination]], [[Differential Algebra and DAE Factors]],
  [[Algebraic Statistics Bridge]]
- *verdict*: [[The Algebraic Factor as a Statistical Game]],
  [[Open Problems in Algebraic Implicit Learning]]

## 3c. Factor graphs and message passing

`Mycelium.jl` — how factors are wired and in what order they talk. The layer with no
counterpart in Lux.

- [[Factor Graphs]] — bipartite structure; the **two** acyclicity notions (`istree` vs `isdag`)
- [[Everything is a Factor]] — data, priors, losses and optimisers as graph nodes, forced by Remark 24
- [[Messages are Inversions]] — a factor → variable message **is** ``c'_\pi``; both exclusion principles
- [[Polarity Resolution]] — target + available messages → a `Polarity`; the two legality checks
- [[Schedules]] — the schedule zoo; tree exactness; pruning by edge direction
- [[Bethe Free Energy]] — Theorem 23 generalised to graphs; energies add, entropies get a counting correction
- [[Loopy Message Passing]] — what breaks off a tree, and the monodromy problem

## 4. Implementation

Implementation notes live next to the code, per [[Start here]]:

**LenticulumCore.jl** — [[LenticulumCore]], [[abstract_types]], [[channels]], [[energy]],
[[open_model]], [[lens]], [[statistical_game]]

**Mycelium.jl** — [[Mycelium]], [[graph]], [[polarity_resolution]], [[messages]],
[[schedules]], [[passing]], [[free_energy]], [[factors]]

## Existing notes

- [[README]] — the pitch
- [[ImplicitREDDiff]] — the diffusion-based implicit learner
