#open-problem

> **Status: resolved (2026-11-12).** Two defects were found and fixed with tests, and one gap in the
> tests was filled. One suspicion was not reproduced, and one finding predates the change. Related:
> [[Feature Backlog]] (I18, P19), [[LP-025 Findings of the review of I5]].

> Sources: a review of the uncommitted I18 change by Mistral (`vibe --agent review`), run with
> Julia in the helper worktree `Lenticulum.jl-mistral` (2026-11-12). Its answer is in
> `../Lenticulum.jl-Review-2026-11-12/i18-mistral.md`, and its scratch tests are
> `scratch/i18_core_review.jl`, `scratch/i18_memo_review.jl` and `scratch/i18_device_review.jl` in that
> worktree. code: `lib/LenticulumCore.jl/src/gaussian_belief.jl`,
> `lib/LenticulumCore.jl/ext/LenticulumCoreChainRulesCoreExt.jl`, `lib/Lenticulum.jl/src/gaussian.jl`;
> notes: `lib/LenticulumCore.jl/src/gaussian_belief.md`, "Per-message costs";
> `lib/Lenticulum.jl/src/gaussian.md`, "What the checks cost".

**Problem.** The change was required to leave every result unchanged. It covered a loop
`_nearly_symmetric`, a fused `_symmetric_part`, `_mean_cov` (the mean and the covariance from one
factorisation), and a memo of `_rank_deficient` keyed by the array's identity. Mistral found these
clean, with run output for each:
- `_symmetric_part`, bit for bit (42 000 cases, ForwardDiff duals among them);
- `_mean_cov` against the two functions, for values, Zygote and ForwardDiff;
- `_mean_cov` on JLArrays with `allowscalar(false)`;
- the memo under in-place changes, after it overflows, and with 8 threads (40 000 calls);
- the suites of LenticulumCore, Mycelium and Lenticulum.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| F1 | `_nearly_symmetric` of a non-square dense matrix returns `false`; the generic method throws `DimensionMismatch` (so `Gaussian(zeros(2), zeros(2,3))` threw `ArgumentError` instead) | right, reproduced | a non-square matrix goes to the generic method | LenticulumCore `test/i18.jl`, "the loop agrees with the generic test": both throw `DimensionMismatch` for 2×3 and 3×2 |
| F2 | `_mean_cov`'s error names `belief_cov`, while the old `_moments` failed in `belief_mean` | right, reproduced (same type, different message) | the check names `belief_mean` | `test/i18.jl`, "_mean_cov from one factorisation": the message equals `belief_mean`'s for a declared-improper, a singular and an asymmetric belief |
| F3 | no test that the memo leaves arrays to the collector, or that a reused object id is caught | right (a gap; Mistral's scratch test found both properties hold) | tests added | Lenticulum `test/i18.jl`: after `GC.gc`, some of 200 dropped arrays have dead weak references. Two planted entries under an array's id, one with another array's reference and one with a dead reference, both carrying the wrong answer: the memo recomputes. A random search for a reused id found none in 20 000 tries, so the reuse is planted |
| F4 | the SVD runs before `copy(A)`, so a change to `A` from another thread between the two stores a new copy with the old answer | not reproduced; it needs a data race on `A` itself, which is already undefined in Julia | nothing changed | — |
| F5 | a wide `A` with a NaN returns `true` where `svdvals` throws | not a change: the old `_rank_deficient` also returned `true` for `m < n` before looking at the values | nothing changed | — |
| F6 | `gaussian.md` and the backlog not updated | right; they were not written yet when the review ran | updated with the change | — |

**Leaning / resolution.** Fixed with a test where reproduced. F4 is recorded only.

Not checked: Enzyme through `_mean_cov` and the memo (no test drives Enzyme through a Gaussian
belief).
