#open-problem

> **Status: resolved (2026-11-17).** The review of P5 (`HardConstraintFactor`'s free energy with a
> singular nonzero noise) found no defect in the new code. One of its probes found a flat factor
> belief that the positive definite path accepted, in code P5 shares with `LinearConstraintFactor`;
> fixed with a regression test. A loss of accuracy with a nearly singular noise is recorded as a
> conditioning limit of the positive definite path, not fixed.
> Related: [[Feature Backlog]] (P5), [[hard_constraint]] "Free energy with a singular noise".

> Sources: a review of the uncommitted P5 change by Mistral (`vibe --agent review`, Mistral Large 4,
> `--max-price 3`, $0 against the budget), in the helper worktree `Lenticulum.jl-mistral` at
> `d494fd73` with the diff applied. The brief, the answer, the probes, their outputs and the
> maintainer's scripts (`maintainer_f1_repro.jl`, `probe3_fixed.jl`) are in
> `../Lenticulum.jl-Review-2026-11-17/p5/`.
> code: `lib/Lenticulum.jl/src/hard_constraint.jl`, `lib/Lenticulum.jl/src/constraint.jl`; tests:
> `lib/Lenticulum.jl/test/hard_free_energy.jl`, `lib/Lenticulum.jl/test/hard_gradient.jl`.

**Problem.** The requirement given to the reviewer: with a singular nonzero noise the term is the
limit of the noisy relation's with noise $Q + \varepsilon P$ ($P$ the projector onto $Q$'s null space); on a
tree the sum of the terms is $-\log Z$ and its gradient that of $-\log Z$ (ForwardDiff, Zygote); the
term throws when the exact equations are redundant or the factor belief is flat along the solution
set, and is finite otherwise; zero and positive definite noise unchanged.

The run wrote three probes beyond the first and its answer file records only the first (no summary;
the pattern of earlier runs). The maintainer ran probes 2, 2c and 3; probe 3 read a field `Q` of the
parameters that does not exist and was repaired (`probe3_fixed.jl`, the noise passed in).

| # | finding | right? | what was done | test |
|---|---|---|---|---|
| 1 | Probe 1: the term against an independent 512-bit limit of the noisy relation's, noise built from a known range and null space; 112 accepted draws | no defect: largest relative difference $3.2\cdot10^{-10}$ | — | the P5 limit testset covers the same ($3.8\cdot10^{-15}$ on 70 draws) |
| 2 | Probe 2: every refusal and acceptance of 370 draws against an independent properness check; 79 refusals right, 1 of 291 acceptances wrong (trial 32) | — | — | — |
| 3 | F1, trial 32 (probe 2c): a **positive definite** noise (1 × 1, 0.777) with inputs flat in two directions; the factor belief's precision has rank 3 of 4, its fourth eigenvalue came out as $1.0\cdot10^{-16}$, `GaussianBelief` took it for proper and the term was finite (negentropy $-24.6$, made of rounding) | **right**, in `_reduced_residual_statistics` (`constraint.jl`), which the positive definite path shares with `LinearConstraintFactor`; P5's own path decides flatness by `_eig_tol` and was not affected. Reproduced: two uninformative inputs, 25 of 400 random draws accepted (`maintainer_f1_repro.jl`) | `_reduced_residual_statistics` refuses a precision whose smallest eigenvalue is below `_eig_tol`, as the hard path does, with the reason | "Review of P5, F1: …" in `hard_free_energy.jl`, both factors, 800 checks; 38 fail without the fix |
| 4 | Probe 3 (repaired): 60 random trees, $F$ and marginals against a 512-bit dense $-\log Z$; random-rank noise (largest error $2.5\cdot10^{-14}$), eigenvalues from $10^{-6}$ to $10^{3}$ ($3.8\cdot10^{-10}$), and a noise of rank $r$ plus one eigenvalue $10^{-11}$, where 6 of 20 trees are off by up to $1.6\cdot10^{-6}$ in $F$ and $4.2\cdot10^{-5}$ in the marginals | not a defect of P5: a relative $10^{-11}$ is above `_eig_tol` ($\approx 16 m\epsilon$), so the noise is positive definite and goes through the old path, messages included, which inverts a noise of condition number $\approx 10^{11}$; the error is the loss of digits that implies | recorded as open in [[hard_constraint]] "What it does not do"; no change | — |
