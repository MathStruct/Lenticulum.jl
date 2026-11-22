#problem

> **Status: closed (2026-11-22).** Three pieces of Codex's work of 2026-11-22 and what was found in them:
> the Mycelium coverage tests found two defects (both fixed); Mistral's review of the Anderson and block
> AD fixes found one behaviour that predates the change and one overstated claim in the notes (recorded,
> the note corrected); GLM's review of the new package `ImplicitSolvers` (P14 W1) found three defects
> (fixed with tests). Related: [[Feature Backlog]] (P14), [[LP-053 Findings of the review of I23]] (finding 2),
> [[LP-044 Findings of the review of P6]] (finding 1), [[Shared Solver Layer]].

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; briefs and answers in
> `../Lenticulum.jl-Review-2026-11-22/coverage/`, `small-fixes/`, `p14-w1/`); Mistral Large
> (`vibe --agent review`, \$0; probes in `scratch/small/` of `../Lenticulum.jl-mistral`); GLM-5.3
> (`vibe --agent review --model glm-5-3`, \$2.30; probes in `scratch/p14w1/` of `../Lenticulum.jl-glm`).
> Each reviewer had the requirement and the diff, not the reasoning. code: `lib/Mycelium.jl/src/anderson.jl`,
> `lib/Mycelium.jl/src/plates.jl` (`_ep_sites`), `lib/Mycelium.jl/src/blocks.jl`; `lib/ImplicitSolvers.jl/src/tolerance.jl`, `lib/ImplicitSolvers.jl/src/linsolve.jl`.
> tests: `lib/Mycelium.jl/test/anderson.jl`, `lib/Mycelium.jl/test/plates.jl`, `lib/Mycelium.jl/test/block_ad.jl`;
> `lib/Lenticulum.jl/test/p9_relative_tol.jl` ("I23"); `lib/ImplicitSolvers.jl/test/`.

## The coverage tests (Codex)

Tests for `anderson.jl`, `blocks.jl` and `plates.jl` written from their notes, against closed forms
(coverage 29/55/61 % → 100/97/99 %). Two of them failed on the code and were kept as `@test_broken`:

| # | finding | right? | done | test |
|---|---|---|---|---|
| C1 | `AndersonSchedule(; ridge = 0)` throws `SingularException` at an exact fixed point: the residual history is zero and the normal equations singular | yes: the test threw | singular normal equations fall back to the least-squares solution by pivoted QR (γ = 0 when ΔF = 0); the regularised solve is unchanged, bit for bit | `lib/Mycelium.jl/test/anderson.jl`, "Anderson on an empty store and at an exact fixed point" |
| C2 | The block EP path (`_ep_sites`) stores a Float64 site's information vector in a Float32 cavity's array, rounding it (4.371972 against 4.371972130535554) | yes: the test failed | the site's array takes the promoted element type | `lib/Mycelium.jl/test/plates.jl`, "EP block with a Float32 cavity and Float64 mixture" |

## Mistral: the Anderson and block AD fixes

The fixes (Codex): Anderson fits its coefficients to the primal residuals and mixes the dual messages
(LP-053 finding 2); `GaussianBlock`, `ComponentMixtureBlock` and `belief_block` build under Zygote
(LP-044 finding 1). Mistral checked, with scripts, that float runs are bitwise those of the old code in
48 settings, that unmixable messages leave the sweep to the inner schedule, that the blocks copy their
inputs and give Zygote's gradient equal to ForwardDiff's, and that duals of two tags nest.

| # | finding | right? | done | test |
|---|---|---|---|---|
| M1 | A float run on a store that holds the dual messages of an earlier run writes duals into every mixed edge, and takes 42 sweeps where the old code took 90 | the observation is right (reproduced with `probe4c.jl`); not a defect of the change. The inner sweep reads the stale dual messages, so the float run's factors carry their tangents on under any schedule: the old code left 15 of 18 messages dual too, it only did not mix them. A rule that took the run's element type from the rewritten messages did not change that (tried, a bitwise oracle failed for the same reason) and was dropped | recorded in `anderson.md` ("One store, dual and float runs"): use a fresh store for a float run after a dual one | — |
| M2 | Under duals the sweep count can differ from the float run's by far more than one: around `flooding_schedule`, 1193 against 949, and 849 against 888 at source $10^4$ | yes: the notes claimed agreement within one sweep in general. The derivatives match the dense solve in every case (within $4.4\cdot10^{-13}$) | `anderson.md` and `passing.md` corrected: the agreement holds around the spanning-tree schedule | — |

## GLM: `ImplicitSolvers` (P14 W1)

GLM confirmed by scripts: `nfe` equals the operator calls counted by a wrapper for every solver; no
false `:converged` in 600 random solves; the Broyden update is the good inverse update and the limited
form equals the dense one while the memory lasts; duals leave the iteration counts and the primal report
as the float run's, nested duals too; host and JLArrays runs agree.

| # | finding | right? | done | test |
|---|---|---|---|---|
| G1 | `Tolerance` takes a negative or NaN `atol`/`rtol`; then GMRES runs on past an exact solve, divides 0/0 and reports `:nonfinite` with the exact solution in hand (`probe3b.jl`) | yes | `Tolerance` refuses them; a GMRES cycle from a zero residual returns at once | `lib/ImplicitSolvers.jl/test/tolerance_report.jl` (`@test_throws`) |
| G2 | GMRES on a complex system diverges (residual $2\cdot10^{14}$) with real Givens rotations; `DenseSolve` solves it | yes (`probe8.jl`); the report was honest (`:maxiters`) | GMRES refuses complex systems | `lib/ImplicitSolvers.jl/test/linsolve.jl`, "GMRES refuses complex systems" |
| G3 | No device test of `adjoint_solve` and `fixed_point_adjoint` | yes (they work, `probe7c.jl`) | added | `lib/ImplicitSolvers.jl/test/device.jl` |

Not defects, recorded by GLM: GMRES at a `Float32` rounding floor below the tolerance runs its budget
and reports `:maxiters` honestly; `LimitedBroyden(memory = 2)` stagnates on a 2 × 2 linear system
(reproduced by an independent drop-oldest implementation: the method's); `DenseSolve` on a singular
consistent system reports `:stalled`.

Not run: device tests on a GPU; the docs and vault builds.
