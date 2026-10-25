#design

> **Status: accepted (2026-10-25, Daniel; proposed by Claude).** Related: [[L-SPEC-004 Algebraic Factors Toric and Grassmannian]] §2.3, [[Algebraic versus Geometric Distance]], [[The Parameter is a Grassmannian]], [[L-ADR-0002 Energies are vectors, scalarised late]], [[L-ADR-0003 Pointwise energy and free energy are separate levels]], [[energy]].

## Context

`scalarise(σ, e)` sees only the energy vector. The Sampson energy $\tfrac12 r^\top(J\Sigma J^\top)^{-1}r$ of [[Algebraic versus Geometric Distance]] needs the Jacobian $J = D_u r$ at the point where $r$ is evaluated, so its metric depends on the point and the parameters: it is the pushed-forward noise precision, a field of scalarisations rather than one. L-SPEC-004 §2.3 asks for it, because it is invariant under the gauge $\Theta \mapsto B\Theta$ of a residual $r_\Theta = \Theta v(z)$: $r \mapsto Br$, $J \mapsto BJ$, so $(BJ\Sigma J^\top B^\top)^{-1}$ cancels $B$. At points where $J\Sigma J^\top$ is singular, a pseudoinverse or damping by $\lambda I$ breaks that invariance; damping by $\lambda R$ keeps it exactly when $R \mapsto BRB^\top$, for example $R = \Theta\Theta^\top$, since then $J\Sigma J^\top + \lambda R = \Theta(D\Sigma D^\top + \lambda I)\Theta^\top$ with $D = D_z v$.

## Options

1. Put $J$ into the energy: a pointwise energy object carrying $(r, J)$. No interface change, but the energy is no longer an element of its `EuclideanEnergySpace`, and `energy_jacobian`, sums and `jensen_gap` stop making sense for it.
2. Change the signature to `scalarise(σ, e, context)` everywhere.
3. Keep `scalarise(σ, e)` and add one step before it: `localise(σ, f, x, a, y, ps, st)` returns an ordinary scalarisation valid at that point (default: `σ` itself). Point-dependent scalarisations are a subtype that cannot be applied without localising.

## Decision

Option 3, accepted by Daniel on 2026-10-25. In `LenticulumCore`:

- `AbstractPointScalarisation <: AbstractScalarisation`; `scalarise` on one throws, asking for `localise`; `islinear` is false; `jensen_gap` on one throws (samples at different points have different metrics).
- `localise(σ, f, x, a, y, ps, st)`, identity by default; `scalar_energy` and the pointwise path localise before scalarising. The free-energy path is unchanged.
- `SampsonNorm(; wrt = :y, Σ = nothing, λ = 0)`, localising to `SquaredNorm(inv(J Σ Jᵀ + λ R))` with `J = energy_jacobian(f, …; wrt)` (a factor may add an exact `energy_jacobian` method) and `R = sampson_damping(f, x, a, y, ps, st)`, by default the identity, which a factor with a gauge overrides. A numerically singular $J\Sigma J^\top + \lambda R$ throws; there is no silent pseudoinverse.

## Consequences

- No existing factor or call changes: every current scalarisation localises to itself.
- The Sampson energy becomes available to any residual factor (`NonlinearFactor` with errors in the inputs, the §3 polynomial factor), not only to algebraic ones.
- Gauge invariance is a property of the factor's `sampson_damping`, documented per factor; the default `λ R = λ I` is not invariant, and the tests show it.
- The scalar energy is no longer a fixed function of the residual, so `jensen_gap` and eager accumulation do not apply to it; [[energy]] has to say so.
