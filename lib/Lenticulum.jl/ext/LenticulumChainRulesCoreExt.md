#implementation

> First-derivative rules for the full covariance eigenvalue floor.

> Sources: original implementation; code: `LenticulumChainRulesCoreExt.jl`, `src/noise.jl`, `test/learnable_noise.jl`.
>
> Theory (CT-ML wiki): [Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Free-Energy).

See `src/noise.md` for the spectral divided-difference derivative, density coordinates
and limitations. ForwardDiff propagates each dual direction through that derivative;
ChainRulesCore supplies the adjoint consumed by Zygote. Repeated eigenvalues on a
smooth branch use the scalar derivative. At the clipping kink the saturated-side
zero derivative is selected. The cap is configuration and receives no derivative.
The tests compare both paths against finite differences on rotated eigenspaces,
including repeated eigenvalues. Higher derivatives and device execution are untested.

## The stored covariance in the working type (Q4, LP-091)

`_covariance_at(Q, inputs...)` (`gaussian.jl`) returns a factor's stored covariance converted to the
element type of the call; the further arguments only choose that type. Its rule gives them no tangent
and passes the cotangent of the result back to `Q`, projected to `Q`'s type. Without the rule Zygote
traced the type selection on every message and free energy: MyceliumLearning's benchmark
`kalman/zygote_gradient` was 16 % slower than before Q4 (2,389 more allocations per gradient); with it,
1.6 % (69 more), within the gate's noise (`benchmark/compare.jl --interleave`, 2026-11-25). A learned
covariance reaches `_covariance_at` already computed from the parameters, so its derivative flows through
`Q`; the rule loses nothing. Lenticulum's suites (both tiers, with the AD matrix) pass with it.
