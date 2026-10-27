#implementation

> `NeuralODEFactor` exposes both endpoints of a numerical flow. Reverse integration
> approximates the continuous inverse; discrete exactness depends on the integrator.

> Sources: Hairer, Lubich, Wanner, *Geometric Numerical Integration*, 2nd ed., II.3 (adjoint and symmetric methods); code: `neuralode.jl`, `flow.jl`, `../examples/flow_roundtrip.jl`, `../test/roundtrip.jl`, `../test/sciml.jl`, `../test/parameter_ad.jl`, `../ext/ImplicitLayersEnzymeCoreActivityExt.jl`
>
> Bibliography: [[Bibliography#^hairer2006geometric|Hairer et al. 2006]]
>
> Theory (CT-ML wiki): [Bayesian Inversion](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Inversion) · [Bayesian Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Bayesian-Lens) · [Variational Free Energy](https://mathstruct.org/CategoryTheory-ML-Wiki/Variational-Free-Energy) · [Reverse Derivative Category](https://mathstruct.org/CategoryTheory-ML-Wiki/Reverse-Derivative-Category) · [Lens](https://mathstruct.org/CategoryTheory-ML-Wiki/Lens)

## Graph free-energy edge directions

`NeuralODEFactor` declares `Mycelium.uses_cavity == true`: its local free energy evaluates
the endpoint residual from incoming points, returning zero if an endpoint is absent.
Graph free-energy evaluation rejects an attached `Emitting()` edge with `ArgumentError`;
use `Bidirectional()` if that channel must also send messages. Missing messages on permitted
edges retain their existing local behaviour. See [[free_energy]] and
[[LP-014 Emitting edges drop the cavity message from the free energy]].

## 1. Two directions, both integrations

$$
z_0 \text{ observed} \;\Rightarrow\; z_1 = \Phi_{t_0\to t_1}(z_0)
\qquad\qquad
z_1 \text{ observed} \;\Rightarrow\; z_0 = \Phi_{t_1\to t_0}(z_1)
$$

Both are the same integrator with the endpoints swapped ([[flow]] §1). Compare the other
bidirectional factors in the project:

| factor | direction A | direction B | cost of B |
|---|---|---|---|
| `GaussianFactor` | pushforward | likelihood | different arithmetic, both closed-form |
| `LinearConstraintFactor` | solve for $x_t$ | solve for $x_s$ | identical; acausal by construction |
| `DEQFactor` | root-find for $z$ | root-find for $x$ | may not converge; needs a square residual |
| **`NeuralODEFactor`** | integrate | integrate | same scheme; work and accuracy depend on direction |

`supported_polarities` always returns two elements, with no dimension condition. This is
an interface capability, not an accuracy or convergence guarantee.

## 2. The inversion label describes the discrete map

The residual uses the numerical forward map, $r = z_1 - \Phi_h(z_0)$.
The exact continuous flow is invertible wherever the ODE has a unique solution over the
interval in both directions; this does not make a discretisation reversible.

`assemble` uses `inversion_kind(integrator, t₀, t₁)`. Euler, RK4 and adaptive SciML solves
receive `SolverInversion(integrator)`, the existing nonexact solver label. No shared core
types change. Both inversion labels use the same endpoint integration and point lens game.

Fixed-step `ImplicitMidpoint` and `Trapezoid` are symmetric: reversing a step inverts it
(Hairer–Lubich–Wanner, II.3). `ExactInversion` is used only with `adaptive=false`, a positive
`dt` dividing the interval, no custom step limiter, and only the whitelisted solve options
`dt`, `adaptive`, `abstol`, `reltol`, `maxiters`. The matching grid matters for time-dependent
fields: a shortened final step would occur at the other end on reversal. Exactness here is
up to nonlinear-solve tolerance and roundoff, **not** continuous-flow accuracy, and presumes
converged stages and deterministic dynamics. Nonempty Lux state conservatively forces
`SolverInversion`, since state updates may change the field. See [[flow]] for the extension.

The lens game scores the actual endpoint residual. Forward-produced endpoints satisfy their
own residual even with a poor integrator; reverse-produced endpoints need not.

## 3. A flow cannot change dimension

`NeuralODEFactor` takes **one** `dim`, not two, because $z(t)$ lives in one space for all $t$.
That is why `channels` gives both channels the same space, and it is a real modelling
restriction: a NeuralODE cannot be an encoder that compresses. `DEQFactor`'s two channels are
independent; this one's are the same space at two times.

It is also why the density story of [[flow]] §3 works at all — a change of variables needs a
bijection between spaces of equal dimension.

## 4. Implementation difficulties

### 4.1 State threading depends on the integration path

Built-in Euler/RK4 endpoint flows use a functional fold carrying `(z, st)` from step
to step. RK4 threads state through its four stages in order. There is no `Ref` mutation
on this path, so Zygote can differentiate the actual discrete map. The public
`vectorfield`, adaptive SciML and divergence paths retain their existing `Ref` interface.

For a stateless dynamics layer that is correct. For one with running statistics it is
arbitrary — and worse here than in [[deq]] §4.1, because RK4 evaluates the field four times
per step at *intermediate* points that are not on the trajectory. A `BatchNorm` inside a
NeuralODE would accumulate statistics from points the solution never visits. Adaptive solves
also thread state through rejected trials; implicit solves and divergence probes add more
evaluations. This preserves existing threading, but state-dependent dynamics need not define
a deterministic ODE or a reversible map.

### 4.2 `flow_logdet` exists and nothing can use it

It computes the divergence correction for transporting a density, but inversion still
returns a `DiracBelief`. `GaussianBelief` is available in `LenticulumCore`; a general
non-Gaussian pushforward belief is not implemented, so the correction does not yet transport
beliefs through this factor.

### 4.3 Truncation error and inverse conditioning are separate

Reversing Euler or RK4 does not invert their forward step even in exact arithmetic. For
$z'=az$, $h=T/N$, Euler's forward multiplier is $1+ah$, its reverse multiplier is $1-ah$,
whereas the discrete inverse is $(1+ah)^{-1}$. Thus the relative endpoint round-trip residual is

$$
\left|1-(1-a^2h^2)^N\right| = a^2Th + O(h^2).
$$

Euler has an $O(h)$ round trip. RK4 has $O(h^4)$ forward global error and an $O(h^4)$
round-trip bound for smooth dynamics on a fixed interval with bounded error amplification.
For this **scalar linear round trip the sharp order is fifth**, not fourth. Its polynomial
$R(x)=1+x+x^2/2+x^3/6+x^4/24$ satisfies

$$
R(x)R(-x)=1+\frac{x^6}{72}+\frac{x^8}{576},
\qquad
\left|[R(ah)R(-ah)]^N-1\right| = \frac{a^6T}{72}h^5 + O(h^7).
$$

Consequently doubling $N$ gives ratios approaching 2 (Euler round trip), 32 (linear RK4
round trip), and 16 (RK4 forward error against $e^{aT}$). `test/roundtrip.jl` asserts all
three: Euler at $N=400,800,1600$ within 4%, RK4 at $N=100,200,400$ within 2% for the round
trip and 4% for the forward error. These grids keep $|a|h$ small; the margins allow
finite-step corrections. BigFloat at 256 bits avoids the Float64 cancellation floor.

Reproduce this table with:

```sh
julia --project=lib/ImplicitLayers.jl lib/ImplicitLayers.jl/examples/flow_roundtrip.jl
```

The script starts from endpoint 3 on $[0,1]$, reverses, then measures
$|3-\Phi_h(z_0)|/3$ and $|z_0-3e^{-a}|/|3e^{-a}|$. Values below are its 256-bit results,
rounded to six digits; tests also check their exact discrete polynomial formulas.

| scheme | a | N | relative residual | relative start error |
|---|---:|---:|---:|---:|
| Euler | 0.5 | 1 | 2.500000e-01 | 1.756394e-01 |
| Euler | 0.5 | 25 | 9.952147e-03 | 5.054864e-03 |
| Euler | 0.5 | 50 | 4.987770e-03 | 2.513628e-03 |
| Euler | 0.5 | 100 | 2.496909e-03 | 1.253396e-03 |
| Euler | 0.5 | 200 | 1.249223e-03 | 6.258477e-04 |
| Euler | -5.0 | 1 | 2.500000e+01 | 9.595723e-01 |
| Euler | -5.0 | 25 | 6.396033e-01 | 3.572253e-01 |
| Euler | -5.0 | 50 | 3.949939e-01 | 2.090267e-01 |
| Euler | -5.0 | 100 | 2.214430e-01 | 1.139515e-01 |
| Euler | -5.0 | 200 | 1.175376e-01 | 5.962588e-02 |
| RK4 | 0.5 | 1 | 2.237956e-04 | 3.959794e-04 |
| RK4 | 0.5 | 25 | 2.222333e-11 | 6.778736e-10 |
| RK4 | 0.5 | 50 | 6.944531e-13 | 4.201538e-11 |
| RK4 | 0.5 | 100 | 2.170146e-14 | 2.615041e-12 |
| RK4 | 0.5 | 200 | 6.781689e-16 | 1.630999e-13 |
| RK4 | -5.0 | 1 | 8.951823e+02 | 5.595067e-01 |
| RK4 | -5.0 | 25 | 2.233357e-05 | 5.645323e-05 |
| RK4 | -5.0 | 50 | 6.953127e-07 | 3.833894e-06 |
| RK4 | -5.0 | 100 | 2.170817e-08 | 2.497951e-07 |
| RK4 | -5.0 | 200 | 6.782214e-10 | 1.594056e-08 |

At the default $N=50$, Euler with $a=-5$ has residual 0.395 and start error 20.9%.
RK4 gives $6.953127\times10^{-7}$ there, but residual 895.18 at $N=1$. Neither label nor
fixed steps make these discrepancies harmless.

Independently of discretisation, inverting a contracting continuous flow amplifies absolute
endpoint perturbations by $e^{|a|T}$ (the norm of the inverse derivative for scalar $a<0$).
For $a=-5,T=1$ this is about 148.4. This is absolute error amplification, not a scalar
relative condition number of 148.4. A symmetric integrator can invert its own discrete map
without accurately recovering the continuous start point or removing this conditioning.
`test/sciml.jl` verifies symmetric round trips at `abstol=reltol=1e-12`, and measures
Tsit5 residuals as its tolerance is tightened; neither promises a global residual bound.

### 4.4 `local_free_energy` is identically zero on the relation

$r = z_1 - \Phi(z_0)$, and if $z_1$ came *from* $\Phi(z_0)$ then $r = 0$ by construction. So
the free energy is a good invariant (the tests use it as one) and a useless diagnostic: it
cannot distinguish a well-integrated flow from a badly-integrated one, because both satisfy
their own residual.

Reverse-solved endpoints can have nonzero forward residual from round-trip error. The
lens game retains that residual. Comparing with a finer integration would estimate
discretisation error; the game does not perform that comparison.

### 4.5 The `prior` argument is genuinely unused

Unlike [[deq]], where it warm-starts the solver, a flow has nothing to warm-start. The
inversion accepts and ignores it. The lens game separately scores its anchor cost at the
returned point; it does not change the flow to condition on the prior.

Related: [[flow]], [[deq]], [[luxfactor]], [[NeuralODE as an Invertible Factor]],
[[The Equilibrium Family]], [[Inversions and Bayesian Lenses]]

## Incoming factor game and solved-point lens game (LP-009)

`free_energy(f, π, y, ps, st)` follows `GaussianFactor`: `π` is the incoming Dirac on the
first (start) channel, `y` the incoming Dirac on the second (end) channel. This fixed order
corresponds to start unobserved, end observed; it does not depend on an inversion polarity.
It delegates to `local_free_energy`, with scalar zero entropy. Non-Dirac arguments throw.

`lens = first(assemble(f, p, ps, st))` restricts the flow relation to polarity `p`.
`free_energy(lens, π, inputs, ps, st)` evaluates the endpoint residual at `invert`'s output
plus an anchor cost on the unobserved endpoint. `inputs` is the named Dirac observation or
that Dirac directly. This is Definition 20's expectation under the point inversion. Entropy
is scalar zero. Trivial priors cost zero; a Dirac prior costs infinity when the solved point
differs, zero when equal; a proper Gaussian costs `½(u*-μ)'Λ(u*-μ)`, relative to its mean,
without normalization constants. Channel precision does not soften Diracs. Soft observations,
singular Gaussian anchors and other belief families throw. The inversion itself still
ignores the prior, so this is scoring its output, not a prior-conditioned posterior flow.

For `dz/dt=0`, endpoints `1,3` give incoming energy `½(3-1)²=2`. Solving either endpoint
gives residual zero. The unknown-endpoint Jacobian is `+1` or `-1`, so a Gaussian anchor
centered on the incoming unknown, with precision `J'J=1`, costs `½(3-1)²=2` at the solution.
`test/conformance.jl` checks this identity in both polarities, full factor conformance
without `game_gap`, lens energies and incompatible hard priors. The same identity is exact
for any residual affine in the unknown, with precision `J'J`; a nonlinear residual gives
the first-order Gauss–Newton approximation described and tested in [[deq]]. It presumes a
zero residual at the solved point; finite integration error need not satisfy that premise.

Scoring arbitrary endpoints uses `energy`/`scalar_energy` (the pointwise level of
[[L-ADR-0003 Pointwise energy and free energy are separate levels]]), not a keyword on
`free_energy`. Shared implementation and limitations: [[point_game]].

## Parameter derivatives (LP-002, 2026-10-26)

The optional core DI extension differentiates built-in Euler/RK4 endpoint energies
with user-selected backends. This is discretise-then-differentiate, including functional
Lux state updates, not a continuous adjoint. `test/parameter_ad.jl` checks ForwardDiff,
Zygote, Mooncake, runtime-activity Enzyme (Reverse and Forward) and FD against scalar
linear dynamics: with $h=T/N$ and multiplier $P(ah)$,

$$
z_1=P(ah)^N z_0,\qquad D_a z_1=N h P(ah)^{N-1}P'(ah)z_0.
$$

Euler has $P(q)=1+q$; RK4 has $P(q)=1+q+q^2/2+q^3/6+q^4/24$.
Tests also compare AD and FD for state-dependent stage dynamics, and fit a known
Euler parameter through the graph's summed local free energy using each AD backend.

Loading SciMLSensitivity activates an endpoint extension for `SciMLIntegrator`;
see [[flow]] for its sensitivity algorithm and state policy. The parameter vector
is an explicit `ODEProblem.p`, rebuilt using the core's tree helpers, so adjoints
see every Lux parameter. Empty nested Lux state is supported for sensitivities.
Nonempty incoming state falls back to the legacy evaluation-state path for plain
forward/reverse flows and inference, including when `sensealg` is explicit.
Loading SciMLSensitivity preserves these primal calls. The factor derivative and
Mycelium graph gradient/`parameter_step` entry points reject AD with nonempty state
through `_check_factor_ad(ad, f, st)` before tracing. Reference differences
(`ad=nothing`) remain available. An arbitrary outer DI call bypasses this check;
its caller must enforce the empty-state restriction, as with Enzyme runtime activity.
`test/sciml_state.jl` checks stateful primals and inference both before and after
loading SciMLSensitivity; `test/lp002_followups.jl` checks the AD errors.
Built-in state threading is unchanged. `flow_logdet`
still uses FD divergence and has no tested AD path. GPU and higher-order
sensitivities remain untested.

Enzyme needs **runtime activity** for these endpoint flows:

```julia
using Enzyme, ADTypes, DifferentiationInterface
ad = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
# Alternatively: AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))
J = energy_jacobian(f, z0, nothing, y, ps, st; wrt = :ps, ad)
g = energy_gradient(f, z0, nothing, y, ps, st; ad)
```

The functional stage fold carries constant model/stage memory through an active
computation. Static activity analysis cannot always distinguish that memory from
active storage; without runtime activity the flow can raise
`EnzymeRuntimeActivityError`. This is an activity configuration requirement, not an
unsupported flow derivative. Enzyme's [runtime-activity documentation](https://enzyme.mit.edu/julia/stable/faq/#faq-runtime-activity)
explains the memory checks enabled by this mode.

The optional EnzymeCore activity extension checks `ad.mode` with
`EnzymeCore.runtime_activity` (verified with EnzymeCore 0.8.26). Core's factor derivative
entry points and Mycelium's graph gradient call `_check_factor_ad` before entering AD.
For built-in Euler/RK4, a default or explicitly non-runtime mode raises an
`ArgumentError` containing both configurations above. The check does not need
ForwardDiff; the separate DEQ IFT extension still does. There are no custom `_flow`
Enzyme rules: normal differentiation handles the stages. An arbitrary outer DI
function bypasses these factor entry checks and must select runtime activity itself.

`test/parameter_ad.jl` checks both Enzyme modes against the discrete closed forms for
Euler/RK4 Jacobians, scalar energy and free-energy gradients, against finite differences
for `ADCountingDynamics` with stage-dependent state, and in graph `parameter_step`
learning with Nesterov look-ahead. It also verifies the configuration error. DEQ IFT
and scored LuxFactor paths retain their passing plain `AutoEnzyme()` checks; they
do not require this runtime-activity setting.
