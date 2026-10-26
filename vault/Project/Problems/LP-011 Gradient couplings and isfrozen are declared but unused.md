#open-problem

> **Status: partly addressed (2026-10-26): option 2; option 1 remains with L-SPEC-002.** Related: [[L-SPEC-002 Stop Gradient]], [[Factors are Parameterized Statistical Games]], [[The Type Discipline of a Factor Graph]] §4. Code: `abstract_types.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/graph.jl`.

> Sources: repository audit of `src/`, `lib/` (source, extensions, tests, examples), `test/`, `docs/literate/` and `bench/` on 2026-10-26; code: `lib/LenticulumCore.jl/src/abstract_types.jl`, `statistical_game.jl`, `lib/Mycelium.jl/src/graph.jl`, `polarity_resolution.jl`; tests: both packages' `test/runtests.jl`, the `LP-011` testsets.
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game)

**Original problem (verified before the guards).** The design says the choice of gradient coupling is made per edge (`connect!(…; coupling = DiagonalCoupling())`, `ComposedFactor(…; coupling)`), and that a frozen factor (`isfrozen`) differs from a non-learnable one in the backward pass. Neither coupling field had a runtime reader; `isfrozen` had only its default definition and a top-level conformance assertion checking `Bool`. No production override returned true and no test/example/tutorial/benchmark selected a non-default coupling. A user setting `coupling = ScoreFunctionCoupling()` therefore got no error and no gradient effect.

**Options.**
1. Implement them as part of the graph-level backward pass ([[Backward Signatures]]) and the stop-gradient specification.
2. Until then, warn (or throw) when a non-default coupling is set, so the setting is not silently ignored.

## Option 2 implemented (2026-10-26)

- `connect!`, direct/parameterized `Edge`, and keyword/positional/parameterized
  `ComposedFactor` now throw `ArgumentError` for non-default couplings, explaining that the
  setting would be silently ignored and naming LP-011 / L-SPEC-002. Fields and types remain.
- `build` does not call `validate`; both now reject every graph factor declaring
  `isfrozen == true`, naming the factor. This includes `islearnable == false`: the existing
  optimiser-list filter does not implement parameter freezing with preserved input
  cotangents, and no graph backward path supplies that guarantee.
- Checks occur at construction/validation only; message passing is unchanged. Accepting
  `DiagonalCoupling()` preserves existing behavior rather than inserting AD stops.
- The two packages' `LP-011` testsets cover all three non-default built-in coupling families
  (`PathwiseCoupling`, `ScoreFunctionCoupling` with and without baseline, `ExactCoupling`),
  a custom subtype, constructor bypasses, independent build/validate rejection for both
  learnability values, errors naming the frozen factor, and default paths.
- No existing non-default tests or examples needed conversion: the audit found none.

**Still open.** Option 1, the actual gradient and freezing semantics, remains with
[[L-SPEC-002 Stop Gradient]].
