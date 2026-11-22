#problem

> **Status: open (2026-11-23).** P14's work packages W2 (ImplicitLayers onto `ImplicitSolvers`) and W3
> (VariationalDiffusion onto it), both implemented by Codex, and what their reviews found. W3: Mistral
> found no wrong result; two newly refused inputs are recorded as accepted behaviour changes. W2: under
> review. Related: [[Shared Solver Layer]], [[Feature Backlog]] (P14),
> [[LP-078 Findings of the reviews of the Mycelium coverage tests and the Anderson and block AD fixes|LP-078]] (W1).

> Sources: Codex (GPT-6.2, `codex exec -s workspace-write`, high effort; briefs in
> `~/.cache/lenticulum/codex-queue/w2-brief.md`, `w3-brief.md`; answers in
> `../Lenticulum.jl-Review-2026-11-22/p14-w2/`, `p14-w3/`); Mistral Large (`vibe --agent review`, \$0;
> probes in `scratch/w3r/` of `../Lenticulum.jl-glm`, answer `p14-w3/ANSWER-mistral.md`). The reviewer had
> the requirement and the diff, not the reasoning. code: `lib/VariationalDiffusion.jl/src/proxdm.jl`,
> `lib/VariationalDiffusion.jl/src/krylov.jl`, `lib/VariationalDiffusion.jl/src/implicit.jl`,
> `lib/ImplicitLayers.jl/src/solve.jl`. tests: `lib/VariationalDiffusion.jl/test/runtests.jl`,
> `lib/ImplicitLayers.jl/test/`.

## W3: VariationalDiffusion (Mistral)

| # | finding | right? | done | test |
|---|---|---|---|---|
| F1 | `prox_infer` throws `ArgumentError` for a negative or NaN `tol` and a negative `maxiters`, and `KrylovSolve(rtol < 0)` at its first solve; the old code ran to `maxiters` and returned `converged = false` (with `iters = -3` for `maxiters = -3`) | yes (the reviewer's probe `scratch/w3r/probe1.jl`) | accepted: the shared layer validates its tolerance and budget (D2); an invalid argument is refused, not run | not added |
| F2 | The dense adjoint throws `ArgumentError("nonfinite dense adjoint system")` when the matrix or the right-hand side has a NaN or Inf; the old `A \ b` returned NaNs silently. A singular system still throws `SingularException`, and the near-singular guard is unchanged | yes (`probe2.jl`) | accepted: a gradient built from NaNs was silent before | not added |
| F3 | The prox's Picard loop tests $\lVert w_k - T(w_k)\rVert \le \mathrm{tol}\,(1 + \lVert w_k\rVert)$ and returns $w_k$; the old loop tested against $\lVert T(w_k)\rVert$ and returned $T(w_k)$. `maxiters = 0` can accept the initial iterate | yes, and stated in the brief's answer | the departure Codex reported (`proxdm.md`); on the ring case 19 steps instead of 20 and the answers 1.5e-9 apart, within the old bound 2.0e-9 (Codex's measurement) | `test/runtests.jl`, the initial-fixed-point check |

Checked by the reviewer without a finding: the adjoints solve the transposed system the old code
solved (dense equal to `J' \ b`; Krylov within 5e-16 on a nonsymmetric `J`); the GMRES mapping
(`restart`, `maxiters`, `rtol`, `atol = 0`; a 30×30 system over several restarts takes the same 56 Arnoldi
steps as the old `_gmres`); `ImplicitSolution.residual` primal under ForwardDiff, also in `many.jl`;
Zygote through the rules equal to ForwardDiff (6e-12 for the prox, 9e-16 for the implicit solve); the moved
paths on JLArrays with `allowscalar(false)`.

Gate (2026-11-23, on the branch): `julia -t auto --project=lib/VariationalDiffusion.jl -e 'using Pkg;
Pkg.test()'` ended with `Testing VariationalDiffusion tests passed`. Not run by me: the smoke examples
(Codex's run: eight of eight, the same printed numbers before and after), the Reactant suite, CUDA.

## W2: ImplicitLayers

Under review. Known so far:

- Three old assertions changed to the shared API's behaviour (accepted): a residual nonfinite at the
  start stops with `:nonfinite` after zero iterations (`test/solver_numerics.jl`); `fd_jacobian`'s keyword
  is `step` (`test/lp019.jl`); on a device array the host-only `fd_jacobian` fails with a scalar-indexing
  error instead of an `ArgumentError` (`test/device_arrays.jl`).
- With Enzyme 0.13.205 the suite aborts in Enzyme's compiler ("Illegal address space cast from decayed
  ptr", LLVM's GC invariant verifier) at `test/parameter_ad.jl:22`; the same Manifest passes on
  `dev-danielb`, and the branch passes with Enzyme 0.13.221. The cause in W2's code is not found.
