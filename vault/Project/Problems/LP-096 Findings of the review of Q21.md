#problem

> **Status: closed (2026-11-26).** Q21 (the package test environments on JLArrays 0.5, and a test hook
> that makes JLArrays refuse a host array in a kernel as CUDA does). Mistral's review found no defect.
> The session's own checks found two limits of JLArrays 0.5: it still accepts a host array in a broadcast,
> which the hook closes, and DifferentiationInterfaceTest and LinearSolve hold some environments back.
> Related: [[LP-095 Findings of the CUDA run and the review of Q20|LP-095]],
> [[LP-089 Findings of the review of Q5|LP-089]], [[Backlog Q Quality and Engineering]] (Q21).

> Sources: Mistral Large (`vibe --agent review`, \$0, two runs: 60 turns to the limit with no answer,
> then a continuation that wrote `scratch/q21r/ANSWER.md` in `../Lenticulum.jl-mistral`; its fresh-resolve
> probes are in the same folder); the session's runs (`~/.cache/lenticulum/q21/`).
> code: `lib/LenticulumCore.jl/test/support/strict_jlarrays.jl`, `gpu/jlarrays_vs_cuda.jl`,
> `lib/LenticulumCore.jl/src/device_arrays.md` ("JLArrays against CUDA").

## Found by the session

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | DifferentiationInterfaceTest 0.11 (the latest) caps JLArrays at 0.3 through its weak compat, so LenticulumCore's and Lenticulum's test environments ran JLArrays 0.3.3, not 0.4, despite their compat entries | right (`Pkg.update` with compat "0.5": unsatisfiable, naming DIT) | DIT removed from both; its two scenario rows are now plain checks against the finite-difference reference, as LP-089 did for Mycelium | both resolve to 0.5.0 (suite logs) |
| 2 | ImplicitLayers' test environment cannot take JLArrays 0.5: 0.5 needs KernelAbstractions 0.10, and LinearSolve (through SciMLSensitivity and OrdinaryDiffEq) caps it at 0.9 | right (unsatisfiable) | compat "0.4, 0.5" with the reason in a comment; it moves when LinearSolve does | resolves to 0.4.2 |
| 3 | JLArrays 0.5 still runs a broadcast that mixes a host and a device array, and a host vector in a matrix-vector product, and `dot` of a device and a host vector; CUDA refuses all three | right (`gpu/jlarrays_vs_cuda.jl`, JLArrays 0.5.0 against CUDA 6.5.1) | broadcasts: the hook `strict_jlarrays.jl` in every `runtests.jl`; the other two recorded as caught only by `gpu/runtests.jl` | with LP-095 fix 3 undone, `device_inference.jl` passes 2,520 on plain 0.5 and errors 6 times with the hook (CUDA's 6); "a host array in a device broadcast is refused" in LenticulumCore's `device_arrays.jl` |

## Mistral's review

No defect. Verified by its probes: fresh resolution without Manifests gives JLArrays 0.5.0 in the test
environments of LenticulumCore, ImplicitSolvers, Mycelium, MyceliumLearning, VariationalDiffusion and
AlgebraicFactors, and 0.4.2 in ImplicitLayers'. No `DifferentiationInterfaceTest` or `scenario=` is left
in code. All eight includes are `@__DIR__`-relative. Its one open point (Lenticulum's test target not
fresh-resolved) is answered by the suite log: JLArrays 0.5.0. It did not run its planned probes of the
hook. The session ran them: scalars, ranges, `Ref`, tuples, device views, `adapt(JLArray, ·)`,
constructors and `copyto!` pass through; a 0-dimensional host array is refused (CUDA would refuse it
too).

## Checks

- Device files of seven packages on JLArrays 0.5 with the hook (`gpu/runtests.jl`'s list): 25,265 passed,
  2 broken.
- Suites with `LENTICULUM_DEVICE_TESTS=true`: LenticulumCore, Mycelium, MyceliumLearning, ImplicitLayers
  (with the hook), AlgebraicFactors, ImplicitSolvers, VariationalDiffusion, Lenticulum both tiers.
- LenticulumCore's device files on CUDA: 786 passed (the new test included).

Not run: CI itself (the change is not pushed); the CUDA run of the other packages (no library code
changed).
