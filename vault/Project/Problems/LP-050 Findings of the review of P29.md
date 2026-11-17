#open-problem

> **Status: resolved (2026-11-17).** The review of P29 (a `NonlinearFactor`'s difference Jacobian cut
> to the rank its accuracy supports) found three defects, all reproduced by tests and fixed: the
> derivative of $J$ was lost in the cut directions where the rank changes at the point; one global
> error bound cut singular values that their own entries resolve; a non-finite $f$ threw in the SVD.
> Related: [[Feature Backlog]] (P29), [[lib/Lenticulum.jl/src/nonlinear|nonlinear]] "The rank of a difference Jacobian".

> Sources: a review of the uncommitted P29 change by a Claude Sonnet subagent with a fresh context
> (the brief only, not the maintainer's reasoning), in the helper worktree `Lenticulum.jl-mistral` at
> `27770961` with the diff applied. Mistral (`vibe --agent review`) was tried first, three times: each
> run stopped at its first shell command with an internal error of vibe
> (`validation error for tagged-union[RustAcceptedApplyResult,RustRejectedApplyResult] … arguments
> Input should be an object`), also with `vibe -c` and in a fresh session; no probe ran, $0.
> The brief, the answer and the probes are in `../Lenticulum.jl-Review-2026-11-17/p29/`.
> code: `lib/Lenticulum.jl/src/nonlinear.jl`; test: `lib/Lenticulum.jl/test/p29_rank_tolerance.jl`.

**Problem.** The requirement given to the reviewer: when $\operatorname{rank}(\partial f/\partial x) = r$
is below $\min(n_{\text{out}}, n_{\text{in}})$ the difference Jacobian has numerical rank $r$ for the
$\varepsilon$-relative rank tests, so a backward message is declared improper and the no-input forward
message equals the exact one; the rank is never above the exact one, and below it only where an exact
singular value is not resolvable; elsewhere agreement to about $10^{-5}$; $2n_{\text{in}}$ evaluations of
$f$ for a well-conditioned Jacobian; correct derivatives with dual numbers; any shape.

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | With dual numbers, $J V V^\top$ ($V$ a constant) drops the derivative of $J$ in the cut directions: for $f(x) = x^2$ along $x_0 + tv$, $v = (1, 2)$, $\mathrm dJ/\mathrm dt$ was $\operatorname{diag}(0, 4)$ at $x_0 = (0, 1)$ and $0$ at $(0, 0)$ instead of $\operatorname{diag}(2, 4)$ (probe 3) | yes, reproduced (2 failures) | $(J - \operatorname{value}(J))(I - P)$ added, value exactly zero | "P29 review: a rank that changes at the expansion point keeps its derivative" |
| 2 | The error bound was global ($\max\lvert f\rvert$ over all outputs, the smallest step over all inputs): $[10^{-3}x_1,\ 10^7 + x_2]$ at $(0.5, 0.5)$ and $[10^{-3}x_1,\ x_2]$ at $(0, 10^6)$ were cut to rank 1 although the singular value $10^{-3}$ is resolved by its own entries to about $10^{-12}$ (probe 6) | yes, reproduced (4 failures) | per-entry bounds, the rank decided in the error-scaled coordinates $D_r^{-1}JD_c^{-1}$, cut by an oblique projection | "P29 review: the error bound is per entry, not global" |
| 3 | A non-finite $f$ (`1 ./ y` at 0, a `NaN` output) threw `ArgumentError` from LAPACK in the SVD, where the old code returned the non-finite matrix (probe 8) | yes, by the reviewer's probe; the test was not run against the old code (the run stopped at the failing testset before it) | finite check, $J$ returned as it is | "P29 review: a non-finite f gives its non-finite Jacobian, no error" |
| 4 | $f = [x_1 + 10^{10},\ x_2]$: $J$ is cut to zero against exact rank 2 (suspicion) | not a defect: the rounding of $f$ at $10^{10}$ is about $10^{-6}$ per value, $10^{-6}/10^{-6}$ per entry, so $J$ is not resolvable; the documented limit | — | — |

Passed (reviewer's probes): 720 random cases up to $5\times5$, rank never above the exact one; a dual
expansion point with a fixed null space, six shapes (relative error $\le 2\cdot10^{-8}$); scalar and
constant $f$, $1\to1$, $x = 0$ and $10^6$, both rectangular shapes, the evaluation counts; shapes up to
$30\times30$; a user `jacobian` unchanged; `Int` and `Float32` inputs.

After the fixes `bench/p29_difference_rank.jl` gives: rank never above the exact one, below it in 337
of 20 000 draws with cut exact singular values of at most $1.7\cdot10^{-8}$ (the first version cut up to
$4\cdot10^{-7}$). The fixes were not reviewed again.
