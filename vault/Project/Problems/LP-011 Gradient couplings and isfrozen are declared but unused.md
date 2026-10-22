#open-problem

> **Status: open (2026-10-22).** Related: [[L-SPEC-002 Stop Gradient]], [[Factors are Parameterized Statistical Games]], [[The Type Discipline of a Factor Graph]] §4. Code: `abstract_types.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/graph.jl`.

**Problem.** The design says the choice of gradient coupling is made per edge (`connect!(…; coupling = DiagonalCoupling())`, `ComposedFactor(…; coupling)`), and that a frozen factor (`isfrozen`) differs from a non-learnable one in the backward pass. Both are stored or declared, and no code reads them: a search for uses of the `coupling` field or `isfrozen` finds only their definitions. So a user who sets `coupling = ScoreFunctionCoupling()` gets no error and no effect.

**Options.**
1. Implement them as part of the graph-level backward pass ([[Backward Signatures]]) and the stop-gradient specification.
2. Until then, warn (or throw) when a non-default coupling is set, so the setting is not silently ignored.

**Leaning.** Option 2 now, option 1 with [[L-SPEC-002 Stop Gradient]].
