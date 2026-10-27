#implementation #derivation

> Exact collapsed linear-Gaussian composition, with the internal channel integrated out.
> Tensor games use explicit independent products of component channel beliefs.

> Sources: original Gaussian marginalisation, checked in `test/game_composition.jl`; code: `composition.jl`, `lib/LenticulumCore.jl/src/statistical_game.jl`.
>
> Theory (CT-ML wiki): [Composition of Statistical Games](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game#composition-energies-add-entropies-chain-definition-22)

## Collapsing a pair (2026-10-27, LP-009)

For $u=Ax+a+\epsilon_1$, $y=Bu+b+\epsilon_2$, independent Gaussian noises give

$$
y\mid x\sim N(BAx+Ba+b,\;BQ_1B^\top+Q_2).
$$

`ComposedFactor(::GaussianFactor,::GaussianFactor)` uses this exact conditional for
its energy space, residual, scalarisation, messages, inversion, entropy, game and graph
free energy. The endpoint channels are exposed; the shared internal channel disappears.
Dimensions and intermediate channel names must match; endpoint names must differ.
Parameter/state trees keep `(first,second)`. Current learned covariances are read at every
call; component noise hyperpriors are rejected because a collapsed likelihood does not
account for them. Residual scalarisation is localised to the parameters' current covariance.
`pushforward` on each GaussianFactor and on the pair supports proper Gaussian/Dirac inputs.

The resulting free energy is exactly the GaussianFactor's existing incoming-message
quantity: expected normalised conditional energy minus posterior entropy. A prior's own
negative log density remains a separate prior-factor contribution. This is a **collapsed**
game, not `compose_free_energy`'s pathwise pair retaining internal-channel entropy. Summing
component free energies at their means would produce neither of these laws.

The common checker runs in both endpoint polarities. Tests independently eliminate the
internal Gaussian, derive the posterior, fit, normaliser and entropy, and compare sequential
pushforward with the collapsed mean and covariance. Point and soft Gaussian observations
are checked, as are incompatible dimensions and unsupported generic components.

## Independent products

Core TensorFactor accepts `π` and `y` as NamedTuples keyed exactly like `parts`, in order.
Each entry supplies its component's complete prior/observation channel game. This is an
explicit product belief convention; a correlated joint Gaussian is rejected rather than
projected to marginals. Results retain the branch grading, states propagate per branch,
and scalarisation adds their scalar free energies and entropies.

Only components declaring an exact product game are accepted: GaussianFactor,
GaussianPrior, the supported Gaussian pair, and nested products of these. Other components
throw. Tests compare the graded components, entropy and scalar sum against closed forms.
No tensor graph wiring or product-belief inversion is added: this API combines independent
channel games, so the common wired-factor checker is inapplicable to TensorFactor itself.
General composed factors (including nested chains) remain unsupported with an explicit
reason. No gradient coupling behavior is added.

## LP-011 implementation (2026-10-27)

The collapsed Gaussian endpoint game accepts all four coupling
annotations. Its historical diagonal derivative already integrates the internal wire,
so exact and analytic pathwise endpoint gradients agree with it. Score gradients sample
the exact internal posterior for Dirac endpoints and return GradientEstimate. The
replicate differentiates joint NLL + log posterior and adds the posterior score term
weighted by reference endpoint free energy minus the fixed baseline. The baseline equal
to endpoint NLL cancels that score noise. Other score endpoint domains throw.
The generic core fallback rejects non-diagonal gradient evaluation for unsupported families.

Evidence: `lib/Mycelium.jl/test/gradient_couplings.jl` and
`test/gradient_couplings.jl`. API, mixed Jacobians, costs and unsupported combinations
are specified in [[gradient_couplings]] and [[L-SPEC-002 Stop Gradient]].
