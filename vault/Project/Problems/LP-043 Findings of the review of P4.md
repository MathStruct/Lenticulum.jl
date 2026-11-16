#open-problem

> **Status: resolved (2026-11-16).** The review of P4's AD half (ForwardDiff and Zygote through
> extended Gaussians whose subspaces move, and the coupled replay through a `HardConstraintFactor`)
> found one defect: under Zygote, `hard_directions` of a belief with no Gaussian direction
> (two or more hard directions and none with variance) threw a `DimensionMismatch`. Fixed, with a
> regression test. The other failure the reviewer chased came from its own probe.
> Related: [[Feature Backlog]] (P4), [[lib/LenticulumCore.jl/src/extended_gaussian|extended_gaussian]] "Derivatives (P4, 2026-11-16)".

> Sources: a review of the uncommitted P4 change by Mistral (`vibe --agent review`, Mistral Large 4,
> `--max-price 3`, $0 against the budget), in the helper worktree `Lenticulum.jl-mistral` at
> `60903bc4` with the diff applied. The brief, the probes, the logs of the maintainer's runs and the
> reproducer are in `../Lenticulum.jl-Review-2026-11-16/p4ad/`.
> code: `lib/LenticulumCore.jl/src/extended_gaussian.jl` (`_psd_split`); tests:
> `lib/LenticulumCore.jl/test/extended_gaussian_ad.jl`.

**Problem.** The requirement given to the reviewer: `ExtendedGaussian{T<:Real}`; product, preimage,
pushforward, conversion from an improper Gaussian, `support_entropy`, `hard_directions` and the
canonical part differentiable by ForwardDiff and Zygote at constant rank, equal to central
differences; a tangent that changes a rank at first order refused under ForwardDiff; Float64 values
unchanged; the coupled replay through a hard relation equal to the derivative of the dense −log Z.

The run stopped at its 100-turn limit with no answer file (stdout: "Discipline. Final probe3,
written carefully and completely:"). The maintainer's session ran its probes in the LenticulumCore
test environment.

| # | probe | right? | what was done | test |
|---|---|---|---|---|
| 1 | `probe1.jl`: how to seed a `Dual` | not a check of the change | — | — |
| 2 | `probe2.jl`: the change's own test files | passed (same files as the suites) | — | `extended_gaussian_ad.jl`, `hard_gradient.jl` |
| 3 | `diag.jl`, `diag2.jl`, `diag3.jl`: derivatives of pushforward, the hard projector and a product over eight shapes; an "indefinite Σ" | seven shapes passed; the eighth failed because the probe built `b0` from `randn(n,n) * randn(n,n)' + 4I`, two different matrices, which need not be positive semidefinite. The constructor refused it correctly | — | — |
| 4 | `probe3.jl`: ForwardDiff against central differences, Zygote against ForwardDiff, over 20 random beliefs including ones with no Gaussian direction | **right** (once its missing imports were added and Zygote's `nothing` for a constant function read as 0, `probe3_fixed.jl`): Zygote threw `DimensionMismatch: variable with size(x) == (0,) cannot have a gradient with size(dx) == (0, 2)` in `_psd_split`. With no nonzero eigenvalue the broadcasts by the empty λ₀ have a pullback Zygote cannot project. The repository's loop draws at least one Gaussian direction, so it missed the case | `_psd_split` returns before the broadcasts when λ₀ is empty (S stays 0 at constant rank; N₀ spans everything and does not move) | "P4: no Gaussian direction (all hard or flat)" in `extended_gaussian_ad.jl`; `repro_empty.jl` failed for (2,0,2), (3,1,2) and (3,0,3) before the fix (`repro_empty_before_fix.log`) and passes for all six shapes after it |

Not probed: Zygote through `preimage` with a moving `M` and a rank change (Zygote has no rank
check, by design; [[lib/LenticulumCore.jl/src/extended_gaussian|extended_gaussian]]); the plate methods
with hard messages under AD; mixtures of extended Gaussians (P27).

**Reviewer's tooling.** It again spent its turns on probe variants and did not write an answer file,
though asked to append after each probe; its last probe was the one that found the defect.

**Leaning / resolution.** Fixed (finding 4).
