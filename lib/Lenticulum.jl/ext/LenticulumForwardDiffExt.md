#implementation

> First-derivative rules for the full covariance eigenvalue floor, and (P25) for the relation
> factors' flat path when its flat directions move.

> Sources: original implementation; code: `LenticulumForwardDiffExt.jl`, `src/noise.jl`, `test/learnable_noise.jl`, `src/gaussian.jl`, `src/constraint.jl`, `test/forms_crossing.jl`.
>
> Theory (CT-ML wiki): [Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Free-Energy).

See `src/noise.md` for the spectral divided-difference derivative, density coordinates
and limitations. ForwardDiff propagates each dual direction through that derivative;
ChainRulesCore supplies the adjoint consumed by Zygote. Repeated eigenvalues on a
smooth branch use the scalar derivative. At the clipping kink the saturated-side
zero derivative is selected. The cap is configuration and receives no derivative.
The tests compare both paths against finite differences on rotated eigenspaces,
including repeated eigenvalues. Higher derivatives and device execution are untested.

The flat path (P25): `_moving_flat_split` (the split of a flagged input whose precision is dual)
and `_range_complement` (the complement of range(B) in the residual metric, and the log
pseudo-determinant) take the derivative at constant rank from the decomposition at the value; a
tangent that changes the rank at first order is refused. Theory, limits and tests: `src/gaussian.md`,
"Flat directions that move".
