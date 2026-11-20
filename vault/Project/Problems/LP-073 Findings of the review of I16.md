#open-problem

> **Status: resolved (2026-11-20).** A review of I16, which makes Zygote's gradient of the graph free
> energy in `ps` linear in the factors and keys: the free energy reads `ps`, the state and the message
> store once for all nodes, and rules add the nodes' cotangents per key or edge. Its requirement is that no
> value or gradient changes. Codex made five findings, each with a probe. Four were defects and are fixed
> with regression tests: a pin set inside AD lost its gradient, the sums were regrouped (a Float32 value
> and a ForwardDiff derivative changed), a state with one field per factor stayed quadratic, and the rule
> of `_subtrees` threw on some cotangent forms. One is a limit, recorded: a node's own reads at a
> high-degree variable still cost O(degree²).
> Related: [[Feature Backlog]] (I16), [[Backlog Done 2026-11]], [[LP-072 Findings of the review of I4]].

> Sources: review by Codex (GPT, `codex exec -s workspace-write`, twelve probes). Brief, diff, answer
> and probes: `../Lenticulum.jl-Review-2026-11-20/i16/`.
> code: `lib/Mycelium.jl/src/free_energy.jl` (`_local_stores`, `_local_messages`, `_local_pins`,
> `_LocalStore`, `_EdgeSlice`, `_sum_onto`, `_factor_terms`, `scalar_free_energy`),
> `lib/Mycelium.jl/src/passing.jl` (`_subtrees`, `_states`), `lib/Mycelium.jl/ext/MyceliumChainRulesCoreExt.jl`;
> `passing.md` ("The backward pass by key"). tests: `lib/Lenticulum.jl/test/zygote_keys.jl` (testsets
> "I16 review: …").

**Problem.** The reviewer had the requirement and the diff, not the reasoning, and compared the reviewed
diff with the code before it on graphs the tests did not build. Each regression test below fails on the
reviewed diff (six checks in four testsets, `codex-probes/fixed_on_old.log`) and passes after the fixes.

## Codex

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | a pin set inside the differentiated function (`pin!` on a store built there, a mean-field node reading the pin) lost its gradient: 14.0 before, `nothing` after | right (probe 5; regression test fails on the reviewed diff) | Zygote keeps a `Dict`'s cotangent in its own context, which an `rrule` cannot reach, and the rule read the pins. The pins are now read outside the rule by plain code (`_local_pins`); only the messages go through it (`_local_messages`) |
| 2 | `sum` is pairwise beyond 1 024 terms and the variable terms were grouped before being added: a Float32 free energy changed from 1 to 0 (cancellation of $2^{25}$), and on 1 026 tied factors the value and the ForwardDiff derivative changed from 1 to 0 | right (probes 3 and 7; regression test, three of its four checks fail on the reviewed diff) | `_sum_onto(acc, terms)` adds from the left, as the loops did; its rule gives each term the sum's cotangent |
| 3 | with one state field per factor, the parameter pullback stayed quadratic: 8.2 → 51.2 MB from 256 to 512 factors, against 0.7 → 1.4 MB without state | right (probe 6; the regression test measured 10.0× at 4× the factors on the reviewed diff) | the state is read once for all nodes (`_states`, the same rule as `_subtrees`), now 3.9×. A first measurement on the stateless chain had shown no gain, so it had been left out |
| 4 | at one high-degree variable the cost is still superlinear: a belief-propagation hub of degree 512 → 1 024 allocated 11.2 → 40.7 MB backward; mean-field factors on a hub read the whole hub each, 64 → 128 factors 24.6 → 136.5 MB. The gradients were right | right as a measurement; a limit, not a regression: a node's reads of its own `d` messages still give cotangents of length `d` each, O(d²) where the code before had O(d × edges); mean-field factors at a hub compute the hub's marginal each, O(d²) already in the forward pass | recorded in `passing.md` ("What it does not do"); no row |
| 5 | called directly, the rules threw on a `nothing` or thunked `nothing` cotangent, and the per-key sum threw on `NamedTuple`s naming different fields and on a `Tangent` with a `NamedTuple` | right (probe 10); Zygote normalised these forms in every graph probe, so no gradient through the library was wrong | `_iszero` catches `nothing` everywhere; `_acc` unthunks, reads a structural `Tangent` as its `NamedTuple` and adds over the union of fields; regression test on the rule of `_subtrees` |

Passed probes: tied keys with arrays of structural cotangents and with disjoint fields through Zygote
(probes 1, 2), gradients through message vectors built inside AD with shared mean-field reads, detached
edges and a fixed pin (probe 4), the hard-overlap term through message vectors in Float64 and Float32
(probe 9), no method ambiguities (probe 10), the benchmark's smoke run and the supplied tests (probe 12).
Not checked by the reviewer: the package suites, device arrays, the store with plate block rows beyond the
supplied test.
