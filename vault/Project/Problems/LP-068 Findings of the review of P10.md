#open-problem

> **Status: resolved (2026-11-19).** The review of P10 (noise-free nodes against nodes with noise on the
> calibration benchmark) found two missing argument checks and a skipped verdict recorded as passed in the
> code, a test that did not state its scope, and eight places where the notes said more than the log
> showed. All ten were right; the code findings are fixed with tests, the notes rewritten.
> Related: [[Feature Backlog]] (P10), [[Backlog Done 2026-11]], [[Implicit Diffusion Learners]].

> Sources: review by Codex (GPT, `codex exec -s read-only`, one run, probes of its own in Julia). Brief,
> diff and answer: `../Lenticulum.jl-Review-2026-11-19/p10/`. Benchmark log: `~/.cache/lenticulum/p10/nodes-full.log`.
> code: `lib/VariationalDiffusion.jl/src/implicit.jl` (`noisefree_nodes`),
> `lib/VariationalDiffusion.jl/examples/calibration_setup.jl`, `lib/VariationalDiffusion.jl/examples/nodes_calibration.md`.

**Problem.** The requirement given to the reviewer: the new method `noisefree_nodes(n; levels)`, the
benchmark script that reuses `calibration.jl`'s setup without changing that script's behaviour, the test,
and every number and claim of the notes against the benchmark's log.

| # | finding | right? | what was done |
|---|---|---|---|
| 1 | `noisefree_nodes` checks only that `levels` is not empty: `n = 0`, levels ≤ 0, > 1, `NaN` or `Inf` construct and fail later (DomainError, a field of NaNs, a zero field) | right (its probes; the new tests threw no `ArgumentError` before the fix) | `n ≥ 1` and every level in (0, 1] checked; tests for `[0]`, `[-0.001]`, `[NaN]`, `[Inf]`, `[0.01, 1.5]`, `n = 0` |
| 2 | the noisy nodes of the benchmark (`samples = 8`, 128 nodes) are called the package default, which is `samples = 4`; "2·samples evaluations" holds only unbatched and antithetic | right | script comment, note and docstrings say so |
| 3 | "`field_nodes` smooths the density a second time" conflates averaging $-\log p_t$ over shifts with smoothing the density; and the halving of the bias holds only where $\sigma_t^2/\alpha_t^2 \gg s^2$ (the ratio of §5 tends to 1 below) | right | docstring, note and §5 of [[Implicit Diffusion Learners]] reworded and qualified |
| 4 | the recall and bias summaries overstate the log: the lemniscate at mid recalls less noise-free (0.190 against 0.214); on the circle the bias ratio is 89 %, 43–46 %, 30 %, not "half"; on the two circles recall is 0.290 against 0.282, not "equal" | right | "eight of nine settings"; the ratios given per range |
| 5 | the table gives one time for rows with two draws (98 / 84 s, 266 / 301 s); the largest speed-up from the printed seconds is 24.5, not 24 | right | both times in the table; "4 to 25 times" |
| 6 | the bias test holds with 200 pairs per level, not the benchmark's 8 (the reviewer measured 0.379 with 8 at $t\le0.1$); it lets two outward biases pass; the gradient oracle calls the production `_weight` | right | the test states 200 pairs, checks both radii inward, and writes the weight as $(\sigma_t/\alpha_t)^2$; `implicit.md` gives the scope and the reviewer's 0.38 |
| 7 | with `ref = nothing`, `query_grid` recorded the round trip and coverage as passed | right (by inspection; no present caller filtered on them) | a keyword `remedies = false`, under which both are `missing`, so a filter on them fails loudly |
| 8 | the minority circle's 0.012 is one matched branch entry, not "one branch point", and contradicts "the minority branch is unchanged" | right | reworded; the ridge filter removes that entry |
| 9 | "the best filter of F3" for the ridge, and "repeat `calibration.md` exactly" for both draws, overstate | right | the ridge is the filter that needs no held-out data; "exactly" restricted to the first draw |
| 10 | the log cannot show compilation or iteration counts, nor the thread count | right for the first two (claims removed or marked as possible; the noise-free grid ran first); the thread count is known to the session (8 cores, `-t auto`) | — |

Not done: timing with compilation separated, and the solver's iteration counts (recorded under "Not checked"
in `nodes_calibration.md`).
