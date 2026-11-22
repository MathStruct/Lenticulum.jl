#problem

> **Status: closed (2026-11-22); finding 1 recorded as the caller's own confusion, finding 4 a limit.** A review of I1's AD step (the graph drivers' ForwardDiff calls take one
> dual type whatever the structure; a Zygote rule for `_unflatten`; `offset_tree` as a loop). Codex made six
> findings. Two were defects of the new rule and are fixed with regression tests (the cotangent's element
> type; a search by name per field). Two are real but older than I1: they go through the default-tag path,
> as all code did before; finding 2 is fixed since (below), finding 1 is recorded. One is a limit of the design, recorded. One asked for tests,
> partly added. The suites found one more defect, the same as finding 3 (forward over reverse).
> Related: [[Feature Backlog]] (I1, I30), [[LP-075 Findings of the review of I27 and I28]].

> Sources: review by Codex (GPT, `codex exec -s read-only`, effort high, reading only: the machine was running
> the suites), with the requirement and the diff, not the reasoning. Brief, diff, answer and the session's
> probes: `../Lenticulum.jl-Review-2026-11-22/i1/`.
> code: `lib/LenticulumCore.jl/ext/LenticulumCoreForwardDiffDIExt.jl` (`_graph_ad`, `_has_dual`),
> `ext/LenticulumCoreChainRulesCoreExt.jl` (the rule of `_unflatten`), `src/energy.jl`
> (`_graph_parameter_gradient`), `src/parameter_tree.jl`; notes `energy.md` ("One dual type for the graph
> drivers"), `parameter_tree.md` ("Trees with a field per factor"), `lib/Mycelium.jl/src/passing.md`
> ("The AD paths (I1)").
> tests: `lib/LenticulumCore.jl/test/i1.jl`, `lib/Lenticulum.jl/test/structure_compile.jl`.

## Codex

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | a caller who passes its own tag, `AutoForwardDiff(tag = ForwardDiff.Tag(f, Float64))`, and differentiates `f` with ForwardDiff around the gradient gets 0 instead of 1: both perturbations carry one tag | right, and older than I1 (`f12.jl`: 0 instead of 1 through the plain `_parameter_gradient`); a caller's tag is passed through unchanged, as before, and reusing one's own tag inside is the caller's perturbation confusion | none; recorded |
| 2 | under a caller's ForwardDiff, a loss that does not depend on the parameters leaves the gradient's coordinates after the first unwritten (ForwardDiff writes as many as the caller's dual has partials) | right, and older than I1 (`f12.jl`: `[0.0, 6.9e-310]` through both `_parameter_gradient` and `_graph_parameter_gradient`; the right value is zero). The cause is ForwardDiff's `extract_gradient!`, which loops over the partials of the dual it is given, the caller's | fixed after the fact (below, "Finding 2") |
| 3 | the rule's buffer had θ's element type, so a cotangent of a wider type was rounded (`big(1)//big(10)^400` onto `[1//2]` gave 0) | right; the same defect broke four of VariationalDiffusion's testsets in the first suite run (ForwardDiff over a Zygote gradient: dual cotangents onto `Float64` coordinates, `no method matching Float64(::Dual)`) | the buffer's type is θ's promoted with the cotangents' (`_cotangent_type`); tests "review of the rule": Float32 coordinates with Float64 cotangents, a `Rational{BigInt}`, and forward over reverse against differences |
| 4 | the dual type still varies with the structure below 12 coordinates (the chunk is the length), and `BigFloat` coordinates keep the default tag; the test compares backends from a direct call, not the dual type that reaches a factor | right as a description; a limit of the design (a chunk longer than the input is not used), stated in `energy.md`. The graph test does call `_graph_ad` with the graph's real contexts (`ps`, `g`, `st`, the schedule), not those of the driver | none |
| 5 | `_scatter!` read each field's cotangent by `hasproperty`, a search of the names: O(keys²) for a wide tree | right (runtime, not compilation) | a cotangent with the tree's fields in order is read by position, one by name otherwise; tests for both orders and a `Tangent` with some fields (correctness only; the cost is not measured) |
| 6 | not tested: a dual in the initial state through a replay; the rule's pullback on device arrays with a sparse `Tangent` and a thunk under `allowscalar(false)` | right | the guard's contexts test has duals in nested `NamedTuple`s, `Ref`s, closures and cycles, which is how a state reaches it; the device form of the rule stays untested (`parameter_tree.md` says so) |

## Found by the suites

The first run of the suites had the rule before finding 3's fix: VariationalDiffusion's mixed derivatives
(`_mixed_at`, ForwardDiff over Zygote) failed in four testsets. That run is not the gate; the suites were run
again on the final code.

The second run found a defect of the change: ImplicitLayers' "LP-002 DEQ implicit solve: four AD backends"
threw `EnzymeRuntimeActivityError` in `offset_tree`, which its solves call inside code Enzyme differentiates.
The loop that replaced the recursion builds the tree from a `Vector{Any}`, whose type is not inferred.
`offset_tree` now keeps the recursion for trees of up to 32 fields and takes the loop for wider ones (only
graph-wide trees, whose drivers take ForwardDiff and Zygote); `_eltype`, also made a loop, is a splat again.
The suite passed after that (`~/.cache/lenticulum/i1/IL_revert.log`); its failing testset is the regression
test. A first call of `offset_tree` on a tree of 960 fields takes 0.02 s.

## Finding 2

`LenticulumCoreForwardDiffDIExt` now has its own `_gradient` for `AutoForwardDiff`: the loss is wrapped
(`_LiftToInner`) so that a value carrying a dual of another tag than the coordinates' gets this call's
zero added, which gives it this call's partials, all zero; a value of this call's tag, or a plain number,
is returned unchanged. Device coordinates keep their own, more specific method. Test
`lib/LenticulumCore.jl/test/i1.jl`, "LP-076 finding 2" (39): without the method six of its checks fail
(two sizes of θ give memory that is not zero); a loss free of the parameters, one through `stop_gradient`, a
caller's Jacobian with more partials than coordinates, a mixed derivative against $2a$.

A review of the fix by Codex (`codex exec -s read-only`, effort high, reading only; brief, diff and answer in
`../Lenticulum.jl-Review-2026-11-22/lp076/`):

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | the fixed graph tag created before a caller's tag, with the caller's dual captured by `fun` rather than passed in the contexts, makes ForwardDiff refuse the call ("Cannot determine ordering of Dual tags"); the guard looked at `args` only | right (a probe: also for a loss that depends on both, `u -> t * sum(abs2, u)`, which threw); older than this fix, from I1 | the guard walks `fun` too (`_graph_gradient`, `_graph_jacobian` in `energy.jl`); tests in the same testset |
| 2 | Float16 caller's duals over Float32 coordinates: the added zero promotes the result to Float32 where chunk mode gave Float16 | right as described; the values are equal, the type is wider | none |
| 3 | device coordinates under a caller's ForwardDiff still fail (the pushforward method stores a dual into a `Float32` buffer) | plausible, older, not on the path changed; not run (needs the device environment) | none; open, no row |
| 4 | the tests can pass with an inner gradient that is wrong in its own partials, since they compare the caller's derivative only | right | a direct check of `_LiftToInner` with a caller's tag created first: the result's inner partials are zero and its value is the caller's dual |

**Open.** Codex's third point on the fix (device coordinates under a caller's ForwardDiff), not run. Not
run: the device tests, the docs builds.
