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
