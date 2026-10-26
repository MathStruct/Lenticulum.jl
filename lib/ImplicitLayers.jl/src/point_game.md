#implementation

> Incoming-point factor games and solved-point lens games for DEQ and NeuralODE factors;
> the residual Jacobian supplies the anchor metric linking them.

> Sources: original derivation; code: `point_game.jl`, `deq.jl`, `neuralode.jl`, `../test/conformance.jl`
>
> Theory (CT-ML wiki): [Statistical Game](https://mathstruct.org/CategoryTheory-ML-Wiki/Statistical-Game) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens)

## Two evaluations

The polarity-free factor game maps `π` to the first channel and `y` to the second, following
`GaussianFactor`. Both must be Dirac beliefs of the channel dimensions. It delegates to
`local_free_energy`, so it scores incoming points without an inversion. Entropy is scalar zero.

An assembled lens restricts the relation to one polarity. Its game accepts the same named
observed-channel inputs as `invert`, or a single Dirac observation. It runs `invert`, then
scores the residual at the returned point and adds a cost for the prior on the unknown.
This realizes Definition 20's expectation under that point inversion, with the specified
residual-plus-anchor energy and zero point-belief entropy. It does not claim to condition
the inversion on the prior. DEQ warm-starts from the prior; NeuralODE ignores it in inversion.

The prior cost is zero for `TrivialBelief`, zero only at exactly the same point for a Dirac
and infinity otherwise, and the quadratic penalty for a proper Gaussian. The latter is
relative to the Gaussian's mean: `-log p(u*) + log p(μ)`, without normalization constants.
Channel precision does not soften a Dirac. Soft observations, singular Gaussian anchors,
nonpoint observations and other prior families throw `ArgumentError`.

Both game values and entropies inhabit `ScalarEnergySpace()` with identity scalarisation.
Core's `scalar_free_energy` accepts a lens when its family supplies the game and
scalarisation; no generic game is assumed for other lenses. The assembled lens already
fixes its polarity, so call its scalar wrapper without a `polarity` override.
Arbitrary-point scoring uses `energy`/`scalar_energy`, as in
[[L-ADR-0003 Pointwise energy and free energy are separate levels]].

## The identity and its limit

For fixed observed `x`, suppose `r(x,u*)=0` and the residual is affine in `u`. Then

$$
r(x,u_0)=J_u(u_0-u^*),\qquad
\tfrac12\|r(x,u_0)\|^2=\tfrac12(u^*-u_0)^\top J_u^\top J_u(u^*-u_0).
$$

The right side is the lens game with a Gaussian anchor centered at `u₀`, precision
`J_u'J_u`, since the solved residual is zero. Equivalently it is the unanchored lens game
plus that anchor. The caller computes `J_u` via `energy_jacobian` and supplies the Gaussian;
the game does not derive a precision from the prior. This representation needs positive
definite `J_u'J_u`; a rank-deficient metric is not supported as a Gaussian anchor here.

For a smooth nonlinear residual and a Jacobian frozen at the root, Taylor expansion gives
`r(x,u₀)=J_u δ + O(‖δ‖²)`. Squared energies differ by `O(‖δ‖³)` in general, not an exact
identity. `test/conformance.jl` checks the closed-form DEQ/zero-flow identities and derives
the cubic gap for a quadratic DEQ; [[deq]] and [[neuralode]] give the arithmetic.

## Implementation limits

The game threads the state returned by inversion into residual evaluation, and returns
the resulting state. Entropy only validates its domain and returns zero without a solve.
The point-belief convention is not differential entropy relative to Lebesgue measure.
No distributional inversion, Gaussian covariance transport or normalized posterior game
is implemented. A DEQ solve may stop before reaching a root, and a reversed discrete ODE
flow may have round-trip error. The game retains the resulting residual energy; the exact
identity above requires its zero-residual premise. Stateful layers have the same running
statistics limitations as the existing solvers/integrators; see [[deq]] and [[neuralode]].

NeuralODE lens dispatch accepts both `SolverInversion` and `ExactInversion`. Explicit and
adaptive endpoint integration retains the actual reverse round-trip residual in the game;
only symmetric fixed-step configurations on matching grids can declare discrete exactness.
See [[neuralode]] §2 and §4.3 and `test/sciml.jl` for conformance in both cases.

`_incoming_point_messages` is inlined so Enzyme can eliminate the temporary
NamedTuple of fixed beliefs before activity analysis. An out-of-line return of this
constant aggregate raised an activity error in the parameter-only free-energy
gradient. `test/parameter_ad.jl` compares DEQ endpoint free-energy gradients with
the closed-form residual gradient for all four AD backends.
