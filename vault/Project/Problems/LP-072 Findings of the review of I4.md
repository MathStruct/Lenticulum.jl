#open-problem

> **Status: resolved (2026-11-20).** A review of I4, the priority queue of `ResidualSchedule`, whose
> requirement is the full scan's commits in the same order, bit for bit. Codex made five findings, each
> with a probe. Three were defects and are fixed with regression tests: a committed message's distance to
> itself was taken as 0 (it is `Inf` for a belief without a distance, NaN for one with a NaN), residuals of
> another type were rounded to `Float64` (two `BigFloat` residuals tied, or rounded to 0), and the queue
> probed after the budget's last commit (a probe that throws threw where the scan returned). One is an
> accepted difference on graphs that throw in both; one was extra probes, two of which were removed.
> Related: [[Feature Backlog]] (I4), [[Backlog Done 2026-11]], [[LP-071 Findings of the review of I26]].

> Sources: review by Codex (GPT, `codex exec -s workspace-write`, reasoning effort high, nine probes).
> Brief, diff, answer and probes: `../Lenticulum.jl-Review-2026-11-20/i4/`.
> code: `lib/Mycelium.jl/src/passing.jl` (`_TaskHeap`, `_dependents!`, `sweep!` for `ResidualSchedule`,
> `_residual_sweep_scan!`), `passing.md` ("The residual schedule's queue"). tests:
> `lib/Lenticulum.jl/test/residual_queue.jl` (testset "I4: the review's findings (LP-072)").

**Problem.** The reviewer had the requirement and the diff, not the reasoning. Its probes compare the queue
with the scan it replaced on graphs the test did not build. Each regression test below fails on the first
implementation (five of its eight comparisons) and passes after the fixes.

## Codex

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | after a commit not in its own dependents, the queue set the task's residual to 0; the scan probes it again and gets the distance of the message to itself, `Inf` for a potential belief without a distance (the scan then commits that edge again, the queue another) and NaN for a `DataFactor` of `[NaN]` (the queue committed an extra time) | right (probes 1 and 3; regression tests fail on the first implementation) | the key is now `_residual(b, b)` of the committed belief `b`: what the probe would compute, without calling the factor |
| 2 | residuals were converted to `Float64` for the heap: `BigFloat` residuals $1$ and $1 + 2^{-80}$ tied (another edge chosen), $2^{-1200}$ rounded to 0 (the sweep stopped) | right (probe 2; regression tests) | the heap keeps `Float64` keys while every residual is one, and widens to `Real` keys when another type appears (`_TaskHeap{R}`, `_update!`) |
| 3 | after the last commit of the budget the queue still probed the dependents; an expectation-propagation probe there threw (contradictory hard clamps) where the scan returned | right (probe 4; regression test) | no probe after the last commit, which also saves its cost |
| 4 | with contradictory hard clamps and belief propagation both throw, but at different commits: an exact message builds its cavity (`_emit` calls `excluded_marginal`) though it does not read it, and the queue does not probe it again | right as a description; not a defect: both throw the same `ArgumentError` on a graph with no valid answer, and only the partial store left behind differs | recorded, not changed; the regression testset checks that both throw, and that the same graph with the variable pinned agrees |
| 5 | extra probes: dependents of a commit at a pinned variable (its cavity and marginal are the pin), mean-field messages after a commit to a mean-field node (they read marginals), a damped message when damping does not apply to the pair | right (probe 7), cost only: the commits agreed | the first two removed in `_dependents!`; the third left (it needs the previous message's type at the commit, one probe) |

Passed probes: two edges of one factor to one variable under belief propagation, mean field and mixture
reduction, with pins, detach and damping; `propagate!` with `rtol` and ForwardDiff duals of both signs
(probe 5, 184 checks); tied stateful nodes in plates with `HardConstraintFactor` and `SplitFactor` (probe 6,
56 checks); an expectation-propagation site that becomes eligible after a prior's commit (probe 9).
Not checked by the reviewer: the package suites, the benchmark, device arrays, nested duals, I24's
partial-sweep stop under `:marginals` with `rtol`.
