#open-problem

> **Status: resolved (2026-11-20).** Two reviews of I26, the overlap term of the Bethe free energy where
> exact messages meet at a variable. Codex made four findings, all right. Two were defects of the change
> and are fixed with regression tests: near-parallel independent directions were refused as one common
> direction, and Zygote failed in the coupled replay. The other two predate it and are rows I27 and I28.
> Mistral found no defect in six probes; it independently derived the formula and checked it against
> dense oracles.
> Related: [[Feature Backlog]] (I26, I27, I28), [[Backlog Done 2026-11]], [[LP-070 Findings of the review of U3]].

> Sources: reviews by Codex (GPT, `codex exec -s workspace-write`, reasoning effort high, seven probes)
> and Mistral Large 4 (`vibe --agent review`, 100 turns, \$0, six probes, five recorded in its answer
> file). Specification, briefs, diff, the tests Codex wrote from the specification, the reviews and the
> probes: `../Lenticulum.jl-Review-2026-11-20/i26/`.
> code: `lib/Mycelium.jl/src/free_energy.jl` (`_hard_overlap`, `_gram_logdet`), `free_energy.md`
> ("Exact messages that meet at a variable"). tests: `lib/Lenticulum.jl/test/hard_overlap.jl`
> (testset "I26: review"), `lib/Mycelium.jl/test/hard_overlap_term.jl`.

**Problem.** The reviewers had the requirement and the diff. Before the review, Codex had turned the
specification (`spec.md`) into `test/hard_overlap.jl` without seeing the code (2204 tests, all passing on the
first implementation). Its two `@test_broken` record a limit of the core: a redundant clamp's predictive
variance of exactly zero can round to $-3\cdot10^{-17}$, which the extended Gaussian's positive-semidefinite
check refuses during inference (two observations of one direction, an infinite free energy either way).

## Codex

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | two independent directions at a small angle $t$ ($[1\,0\,0]$, $[1\,t\,0]$) were refused as one common direction at $t = 10^{-7}$ (Float64) and $10^{-3}$ (Float32); the reference is $\log 2\pi t$. Already at $t = 10^{-6}$ the total was off by $10^{-5}$ | right (probe; the new regression test fails on the first implementation) | the Gram determinant from the singular values of the stacked bases, not the eigenvalues of $B^\top B$, which square them; the derivative is $\operatorname{tr}(B^+\Delta)$ at constant rank, as for `HardConstraintFactor`'s $\tfrac12\log\det\tilde G\tilde G^\top$. Regression tests at $t = 10^{-6}$, $10^{-7}$ (Float64) and $10^{-3}$ (Float32) |
| 2 | `free_energy_gradient` returns finite gradients for a model whose free energy is infinite (two exact observations of one direction) | right, but not caused by the change: the fixed-belief gradient evaluates neither the variable terms nor the factors' rank checks, and did not before | row I27 |
| 3 | Zygote through the coupled replay (`ExactCoupling`, `schedule`) failed in `_hard_overlap`: `MethodError` on a `Core.Box` tangent, since the tuple of bases was rebound in a loop and then captured by a closure | right (probe; the regression test fails on the first implementation) | the sum over the bases moved into a function of its own (`_overlap_of_bases`); regression test with `ad = nothing`, ForwardDiff and Zygote against the dense gradient |
| 4 | a `MeanField()` likelihood next to the variable where the messages meet throws (`_product_belief` refuses an `ExtendedGaussian` marginal), while the same tree under belief propagation gives the dense value | right, and older than the change (`lib/Lenticulum.jl/src/constraint.jl`) | row I28 |

Its other probes passed: mixed ranks on both sides of a `ScaleShiftFactor` with a singular noise (to
$5\cdot10^{-15}$, gradients to $10^{-9}$), a plate of points with `SplitFactor` and a detached edge, and a
loopy graph (no oracle, no throw).

## Mistral

It derived the formula independently ("the uncorrected Bethe sum exceeds the dense $-\log Z$ by exactly
$+\tfrac12[(p-1)\log\det G - \sum_i \log\det G_{-i}]$") and recorded five probes, all passing: stars with up to
four messages of mixed rank, chains through variables without a prior, `SplitFactor` and `ScaleShiftFactor`
messages meeting a `HardConstraintFactor`'s, a singular nonzero noise, angles down to $10^{-6}$, refusals at
$10^{-9}$, and a plate whose row path falls back correctly. It hit its turn limit before recording probe 6,
which the session ran: its gradient graph passes under all three backends. Its second graph stops at an
existing, documented refusal (a `GaussianFactor` with a learned noise takes no hard messages), which is not a
defect. It recorded no defect.

## Found by the suite

`test/split.jl`, LP-069 F8, failed with the overlap term: a prior on the line $z = (T, T)$ (the test fixture
`LinePrior`, exact along $(1,-1)/\sqrt2$) and the clamped piece $z_1 = 0$ meet at $z$ at 45°, and the total
came out $-\log Z + \tfrac12\log 2$. The expected value is right ($Z = \sqrt2\,\mathcal N(0; 0, 2) = (2\pi)^{-1/2}$, the
pushforward of $T$, by hand). The fixture's term left out the Jacobian of its own exact direction on its
cavity's support, which `HardConstraintFactor`'s term has and the overlap term assumes (the two errors had
cancelled before). The fixture now adds it; the contract is written in Mycelium `free_energy.md` ("The contract
for a factor that sends exact messages"). The library has no other source of hard messages.
