#problem

> **Status: closed (2026-11-23).** P14 W5 (Krylov.jl out, Codex) and what its two reviews found. The
> shared GMRES of ImplicitSolvers gains left and right preconditioners, judged on the true residual, and
> complex systems; ImplicitLayers' `KrylovIFT` runs every path on it (the Krylov extension is gone, the
> code moved into `ImplicitLayersDifferentiationInterfaceExt`); MyceliumLearning's `:krylov` is an alias of
> `:gmres`; Krylov.jl remains only as the oracle of ImplicitSolvers' tests. Gemini found one defect (GMRES
> on complex dual numbers threw), stale docstrings and missing tests; Mistral's probes found no defect.
> Related: [[Shared Solver Layer]] (W5), [[LP-081 Findings of the review of P14 W4|LP-081]],
> [[LP-079 Findings of the reviews of P14 W2 and W3|LP-079]], [[Feature Backlog]] (P14, P37).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort, 38 min; brief and answer in
> `../Lenticulum.jl-Review-2026-11-23/p14-w5/` (`w5-brief.md`, `IMPL-p14-w5.md`), its scripts in
> `scratch/w5/` of `../Lenticulum.jl-codex`). Gemini 4 Hydrogen (`agy --mode plan`, read-only; it answered
> first with a plan and waited for approval, and was continued in the same conversation:
> `REVIEW-from-gemini-2.md` in the same folder). Mistral Large (`vibe --agent review`, stopped at its turn
> limit; answer file with probes 1–5, `scratch/w5r/` of `../Lenticulum.jl-mistral`; probes 6 and 7 written
> but not recorded, run by the session). code: `lib/ImplicitSolvers.jl/src/linsolve.jl`, `tolerance.jl`,
> `adjoint.jl`, `lib/ImplicitLayers.jl/ext/ImplicitLayersDifferentiationInterfaceExt.jl`,
> `lib/MyceliumLearning.jl/src/graph_adjoint.jl`. tests: `lib/ImplicitSolvers.jl/test/linsolve.jl`,
> `krylov_oracle.jl`, `device.jl`, `lib/ImplicitLayers.jl/test/solver_numerics.jl`.

## What W5 measured (Codex)

- **Oracle** (`lib/ImplicitSolvers.jl/test/krylov_oracle.jl`, Krylov.jl 0.10.12): 648 generated systems (sizes 1, 2, 10, 50, 200;
  nonsymmetric, symmetric positive definite, condition number about $10^6$; `Float32`, `Float64`,
  `ComplexF64`; matrices and functions; no preconditioner, left and right Jacobi, a left `ldiv` solve;
  restart below $n$ and not). In every case both true residuals meet the D2 bound and
  $\lVert x_{\text{shared}} - x_{\text{Krylov}}\rVert \le \lVert A^{-1}\rVert_2(\lVert r_{\text{shared}}\rVert + \lVert r_{\text{Krylov}}\rVert)(1+10^{-6})$.
  The 2×2 symmetric positive definite and ill-conditioned fixtures use equal diagonals: with a nonuniform
  left Jacobi there, Krylov.jl stopped on its preconditioned estimate above the D2 bound (Codex's
  departure; the larger fixtures use nonuniform Jacobi).
- **Unchanged results** (recorded on `f5cdaff0`, `scratch/w5/record.jl`, `compare.jl`): the real
  unpreconditioned `KrylovIFT` results (100) and the 52 graph `:gmres` gradients bitwise equal; 192 solution
  vectors and reports of the shared GMRES bitwise equal; the new `:krylov` bitwise equal to `:gmres`.
  Preconditioned `KrylovIFT` against the old Krylov.jl path: at most $2.8\cdot10^{-14}$ absolute on the CPU,
  $1.8\cdot10^{-7}$ on the `Float32` device fixtures.
- **Load times** (median of three fresh processes): ImplicitLayers 36.5 → 40.1 ms (+9.8 %),
  MyceliumLearning 33.7 → 36.3 ms (+7.6 %), inside the 10 % limit.

## Gemini's review

| # | finding | right? | done |
|---|---|---|---|
| G1 | the docstrings of `DEQFactor` (`deq.jl` 24–25) and of the module (`ImplicitLayers.jl` 60–61) still say to load Krylov | right | fixed: they say DifferentiationInterface and ImplicitSolvers' GMRES |
| G2 | `primal` does not unwrap `Complex{Dual}`, so tangents could steer GMRES's checks | right, worse than stated: with a complex dual right-hand side `linsolve(…, GMRES())` threw `MethodError: no method matching Float64(::Dual)` in `_bound` (the scale stayed a `Dual`). New with W5, which let complex systems into GMRES | fixed: `primal(z::Complex) = complex(primal(real(z)), primal(imag(z)))` (`tolerance.jl`). Test "GMRES on complex dual numbers (LP-082, Gemini's finding 2)" in `lib/ImplicitSolvers.jl/test/linsolve.jl`: 1 failure and 1 error before, 16 of 16 after; it checks the iteration count equal to the primal solve's for tangents of 0, 1 and $10^8$, and the tangent of $x$ equal to $A^{-1}\dot b$ |
| G3a | no test of `KrylovIFT` on a complex system | right | not added at the DEQ level (complex DEQ parameters are untested end to end, `solve.md`); the linear solve is covered by the oracle and `lib/ImplicitSolvers.jl/test/linsolve.jl`, and Codex's comparison ran a complex `KrylovIFT` linear system against the old code ($5.6\cdot10^{-16}$) |
| G3b | no test reaches `_refine_broyden` (a stalled first solve) | right | not added as a test; Mistral's probe 7 (repaired, below) ran a zeroed Broyden history: the solve stalls and throws "DEQ Krylov IFT did not converge: stalled" within its `itmax`. A case where the correction rescues a stalled solve is not known; the refinement stays untested |
| G3c | the factory forms `ctx -> M` and `ctx -> (; left, right)` untested | right | added to "LP-002 Krylov preconditioning" (`solver_numerics.jl`): both forms against the dense gradient under four AD backends |
| G3d | no unit test of degenerate preconditioners (annihilating, nonfinite) | right | added: "GMRES with degenerate preconditioners (LP-082, Gemini's finding 3)" (`lib/ImplicitSolvers.jl/test/linsolve.jl`, 19 checks): annihilating left or right → `:stalled`, overflowing or NaN left → `:nonfinite`, never `converged`, the report's residual the true one; a left preconditioner scaled by $10^{-20}$ does not decide convergence |

Found sound: the true-residual stopping with left preconditioning, the update after each restart with
right and with both, the complex rotations (unitary; $Ax=b$ solved, `adjoint_solve` with $J^{H}$), the real
path bitwise unchanged, no scalar indexing, the `KrylovIFT` mapping and messages, the alias, the oracle's
bound.

## Mistral's review

No defect. Probes 1–5 (recorded): preconditioned solves against the dense solve with restarts; 300 random
badly scaled or rank-deficient preconditioners with budgets 0–25 (no false convergence, `maxiters`
respected, the report's residual the true one); complex systems solve $Ax=b$, not $\bar A x=b$ or
$A^{H}x=b$; the old `_linsolve` reconstructed from the diff gives bitwise the same results on 220 real
systems, also under ForwardDiff; JLArrays with `allowscalar(false)` for every element type and
preconditioner; the package's own tests. Probe 6 ran ImplicitLayers' `KrylovIFT` testsets (88 of 88 preconditioning, 60 of 60 IFT pass; one error in its own preamble). Probe 7
(`KrylovIFT` internals) called `_ift_solve` in the extension module, where it is not defined; repaired by
the session: `on_solve`'s keys and true residual, the Broyden orientations, the fallback message, the
stalled Broyden budget pass. Its two failures had a wrong oracle: the system meant to be unsolvable with
zero tolerance is $-I$, solved exactly in one step. Its last part stopped at a fixed point that diverges in
its own setup.

## Also changed

`lib/ImplicitSolvers.jl/test/Project.toml` allowed only JLArrays 0.4, so its device tests could not run
under 0.5 (LP-081's case); `[compat]` is now `"0.4, 0.5"`. The local test environment stays at 0.4.2
(0.5 needs other packages moved).

## Gates (2026-11-23)

Logs in `~/.cache/lenticulum/w5/`, `JULIA_NUM_THREADS=auto`.

- On the merged code with the fixes above (`fix/`): ImplicitSolvers 4300 pass, 3 broken (the known ones),
  0 fail; ImplicitLayers with `LENTICULUM_DEVICE_TESTS=true` passed (device DEQ solves and IFT 584 of 584,
  device Euler and RK4 360 of 360); MyceliumLearning passed.
- On Codex's commit `99d4f747`, before the fixes (`gates/`; the fixes touch only `primal` for complex
  numbers, two docstrings and tests): Lenticulum, both tiers (`LENTICULUM_SLOW_TESTS=true`), passed;
  VariationalDiffusion passed (1040 of 1040).

Not run: the docs builds, a real GPU (JLArrays only), AlgebraicFactors and AdversarialFactors (they do
not use the changed solvers), Lenticulum and VariationalDiffusion on the fixes themselves.
