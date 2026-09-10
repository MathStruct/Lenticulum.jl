# neuralode.jl — implementation note

> `NeuralODEFactor`: the cleanest bidirectional factor in the project, and the reason is a
> theorem about ODEs rather than anything about neural networks.

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
([[Bayesian Lens]]: *"nothing constrains an inversion to be exact… the quality of the choice
is what the free energy measures"*).

> If you disagree, the change is one line and the consequence is that the free energy would
> start charging for discretisation error — which nothing currently computes, since the
> residual on the relation is identically zero. See §4.4.

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

It computes exactly the correction needed to transport a density
($\log p(z_1) = \log p(z_0) - \Delta$), and the inversion still returns a `DiracBelief` —
because the belief type that could hold the result is not reachable from a `lib/` package
([[deq]] §4.1, and [[The Equilibrium Family]] §5).

So the most valuable thing in this file is dead code with a passing test. That is the
sharpest single illustration of why `GaussianBelief` is in the wrong package.

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

The quantity that *would* be informative is the discretisation error, i.e. the difference
between this integrator's answer and a finer one. Nothing computes it.

### 4.5 The `prior` argument is genuinely unused

Unlike [[deq]], where it warm-starts the solver, a flow has nothing to warm-start. The
argument is accepted and ignored, which is correct and worth stating so nobody looks for the
place it should have been used.

Related: [[flow]], [[deq]], [[luxfactor]], [[NeuralODE as an Invertible Factor]],
[[The Equilibrium Family]], [[Bayesian Lens]]
