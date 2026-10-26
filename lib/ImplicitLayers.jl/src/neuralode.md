#implementation

> `NeuralODEFactor`: the cleanest bidirectional factor in the project, and the reason is a
> theorem about ODEs rather than anything about neural networks.

> Sources: code: `neuralode.jl`
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
| **`NeuralODEFactor`** | integrate | **integrate** | **identical, and unconditional** |

No dimension condition, no solver, no convergence question. `supported_polarities` returns two
elements always.

## 2. `ExactInversion`, and the defence of that label

`assemble` produces a lens with `LenticulumCore.ExactInversion` — not `SolverInversion`.

The claim is that $\Phi_{t_1\to t_0}$ *is* the inverse of $\Phi_{t_0\to t_1}$, exactly, as a
mathematical fact about flows. What is inexact is the **discretisation**, and that is a
property of the integrator, not of the inversion: refine `steps` and the error goes to zero
like $h^4$, with no bias term left over.

Contrast `DEQFactor`, whose `SolverInversion` may converge to a *different root*, or to none.
That is a qualitatively different kind of inexactness, and collapsing the two under one label
would lose the distinction the vault cares about
([[Inversions and Bayesian Lenses]]: *"nothing constrains an inversion to be exact… the quality of the choice
is what the free energy measures"*).

The lens game evaluates the actual endpoint residual. Forward integration followed by
the same forward residual gives zero; reverse integration followed by a forward residual
can expose round-trip error. This does not estimate error against the true continuous flow.

## 3. A flow cannot change dimension

`NeuralODEFactor` takes **one** `dim`, not two, because $z(t)$ lives in one space for all $t$.
That is why `channels` gives both channels the same space, and it is a real modelling
restriction: a NeuralODE cannot be an encoder that compresses. `DEQFactor`'s two channels are
independent; this one's are the same space at two times.

It is also why the density story of [[flow]] §3 works at all — a change of variables needs a
bijection between spaces of equal dimension.

## 4. Implementation difficulties

### 4.1 The state `Ref` is a wart with a cause

`integrate` wants a pure `(u,t) -> du`. `LuxCore` wants `st` threaded in and out of every
call. `vectorfield` bridges them with a `Ref` that the closure writes on each evaluation, so
**the state after an integration is whatever the last stage of the last step wrote**.

For a stateless dynamics layer that is correct. For one with running statistics it is
arbitrary — and worse here than in [[deq]] §4.1, because RK4 evaluates the field four times
per step at *intermediate* points that are not on the trajectory. A `BatchNorm` inside a
NeuralODE would accumulate statistics from points the solution never visits.

### 4.2 `flow_logdet` exists and nothing can use it

It computes the divergence correction for transporting a density, but inversion still
returns a `DiracBelief`. `GaussianBelief` is available in `LenticulumCore`; a general
non-Gaussian pushforward belief is not implemented, so the correction does not yet transport
beliefs through this factor.

### 4.3 The round trip is not bit-exact, and dissipative dynamics make it worse

Asserted in the tests: `flow_reverse(flow_forward(z₀)) ≈ z₀` to `rtol = 1e-9` and
`!= z₀` exactly. The gap is $O(h^4)$ and harmless here.

It is not harmless in general. What contracts forwards expands backwards, so for strongly
dissipative dynamics the reverse integration is unstable — the well-known reverse-mode
NeuralODE problem, and the reason real implementations checkpoint rather than re-integrate.
The test uses $A$ with mildly negative eigenvalues, which is the easy case, and nothing warns
about the hard one.

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
